#!/usr/bin/env python
"""Interactive peak review & stub re-analysis dashboard.

For every experimental group: loads the raw cohort of auto-detected candidate peaks (or an
existing manually-reviewed pickle to continue a prior session), lets the user inspect and pick
among the auto-detected candidates per replicate -- side by side, one column per replicate,
including ones that came up empty (stubs) -- and interactively redraw RT windows with live
re-scoring. The result is saved back out as a single canonical `{group}_manually_reviewed.pkl.gz`
per group.
"""
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import io
import base64

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import dash
    from dash import Dash, html, dcc, Input, Output, State, MATCH
except ImportError:
    print("This tool requires the 'dash' package. Install it with:\n    pip install dash")
    sys.exit(1)

from analysis_containers import ReplicateGroupResults
from pipeline_functions import pipeline_find_and_score_peak

DEFAULT_RT_RANGE = (3.0, 15.0)
DEFAULT_MZ_RANGE = (400, 2000)
CONSENSUS_MARGIN = 0.15
SLIDER_PADDING = 1.0
IMG_STYLE = {"width": "100%", "height": "auto", "display": "block"}

# Process-lifetime caches, keyed by group index so navigation between groups is instant on repeat visits.
_GROUP_CACHE = {}
_DYNAMIC_STATE = {}  # (group_idx, block_idx, rep_id) -> {"peak": PeptidePeak|None, "is_stub": bool, "rt_start": float, "rt_end": float}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive Peak Review & Stub Re-analysis Dashboard"
    )
    parser.add_argument(
        "-i",
        "--input_excel",
        required=True,
        type=Path,
        help="Path to the master input Excel file describing the sample matrix layout.",
    )
    parser.add_argument(
        "-r",
        "--reports_dir",
        required=True,
        type=Path,
        help="Path to the root output reports directory.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        help="Port to serve the interactive dashboard on.",
    )
    parser.add_argument(
        "--max_ppm",
        type=float,
        default=15.0,
        help="Maximum mass error tolerance (absolute ppm) for surfacing a candidate peak.",
    )
    return parser.parse_args()


def load_and_group_samples(excel_path: Path) -> pd.core.groupby.DataFrameGroupBy:
    """Loads the sample sheet from an Excel file and groups it identically to the pipeline."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Input Excel file not found at: {excel_path}")
    df_samples = pd.read_excel(excel_path)
    required_grouping_cols = [
        "experiment_id",
        "precursor_plasmid",
        "maturase_name",
        "maturase_plasmid",
        "expression_temperature",
        "digestion",
    ]
    missing_cols = [c for c in required_grouping_cols if c not in df_samples.columns]
    if missing_cols:
        raise KeyError(f"Input Excel missing required grouping columns: {missing_cols}")

    return df_samples.groupby(required_grouping_cols)


def find_group_pickle(reports_dir: Path, group_info: tuple) -> Path | None:
    """Searches for the raw cohort pickle (auto-detected candidates) across candidate directory layouts."""
    ex_id = str(group_info[0]).replace(" ", "_")
    group_folder_name = "_".join([str(val).replace(" ", "_") for val in group_info])

    candidates = [
        reports_dir / group_folder_name / f"{group_folder_name}_master_pipeline_output.pkl",
        reports_dir / group_folder_name / f"{group_folder_name}_cohort.pkl.gz",
        reports_dir / group_folder_name / f"{group_folder_name}_cohort.pkl",
        reports_dir / ex_id / group_folder_name / f"{group_folder_name}_master_pipeline_output.pkl",
        reports_dir / ex_id / group_folder_name / f"{group_folder_name}_cohort.pkl.gz",
    ]

    for cand in candidates:
        if cand.is_file():
            return cand

    excluded_suffixes = ("_CURATED_APPROVED.pkl", "_manually_reviewed.pkl.gz", "_stub_reanalyzed.pkl.gz")
    matches = list(reports_dir.glob(f"**/{group_folder_name}*.pkl*"))
    for m in matches:
        if not m.name.endswith(excluded_suffixes):
            return m

    return None


def load_replicate_group_results(filepath: Path) -> ReplicateGroupResults:
    """Loads a ReplicateGroupResults object from a gzipped pickle file."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    return ReplicateGroupResults.load(filepath)


def discover_groups(input_excel: Path, reports_dir: Path) -> list[dict]:
    """Builds the ordered list of groups, resolving each one's starting pickle: a prior
    manually-reviewed session if it exists, otherwise the raw auto-detected cohort."""
    grouped_samples = load_and_group_samples(input_excel)
    groups_meta = []
    for group_info, _ in grouped_samples:
        group_folder_name = "_".join([str(val).replace(" ", "_") for val in group_info])
        output_pkl_path = reports_dir / group_folder_name / f"{group_folder_name}_manually_reviewed.pkl.gz"

        is_continuation = output_pkl_path.is_file()
        input_pkl_path = output_pkl_path if is_continuation else find_group_pickle(reports_dir, group_info)

        if input_pkl_path is None:
            print(f"[!] Skipping '{group_folder_name}': no cohort or manually reviewed pickle found.")
            continue

        groups_meta.append({
            "group_folder_name": group_folder_name,
            "input_pkl_path": input_pkl_path,
            "output_pkl_path": output_pkl_path,
            "is_continuation": is_continuation,
        })
    return groups_meta


# ==============================================================================
# --- DATA HELPERS ---
# ==============================================================================

def build_group_index(group_results: ReplicateGroupResults):
    """Single pass over every replicate's found_peaks/selected_peaks that builds:
      - rep_index[rep_id] = {"found": {(cut,mod,charge): [non-stub peaks]},
                              "stub":  {(cut,mod,charge): stub peak},
                              "selected": {(cut,mod,charge): curated peak}}
      - combos: every (fragment_cut, charge, modification, target) seen anywhere in the cohort

    Replaces what used to be a fresh linear scan of found_peaks/selected_peaks on every single
    lookup (get_candidates / find_selected_peak / find_stub_peak), repeated for every block and
    every render -- that scan-per-lookup pattern is what made opening a group with many
    target/charge combinations so slow. Now every lookup below is a plain dict access.
    """
    rep_index = {}
    scope = {}  # (fragment_cut, modification) -> {"target": target, "charges": set()}

    for rep_id, rep_results in group_results.replicates.items():
        found_by_key = {}
        stub_by_key = {}
        selected_by_key = {}

        for target, peaks in (getattr(rep_results, "found_peaks", {}) or {}).items():
            cut_mod = (target.fragment_cut, str(target.modifications))
            entry = scope.setdefault(cut_mod, {"target": target, "charges": set()})
            for p in peaks:
                entry["charges"].add(p.charge)
                key = (cut_mod[0], cut_mod[1], p.charge)
                if getattr(p, "is_stub", False):
                    stub_by_key.setdefault(key, p)
                else:
                    found_by_key.setdefault(key, []).append(p)

        for target, peaks in (getattr(rep_results, "selected_peaks", {}) or {}).items():
            cut_mod = (target.fragment_cut, str(target.modifications))
            entry = scope.setdefault(cut_mod, {"target": target, "charges": set()})
            for p in peaks:
                entry["charges"].add(p.charge)
                key = (cut_mod[0], cut_mod[1], p.charge)
                selected_by_key.setdefault(key, p)

        rep_index[rep_id] = {"found": found_by_key, "stub": stub_by_key, "selected": selected_by_key}

    combos = [
        (cut, int(charge), mod, entry["target"])
        for (cut, mod), entry in scope.items()
        for charge in entry["charges"]
    ]
    return rep_index, combos


def get_candidates(rep_index_entry: dict, fragment_cut, modification, charge, max_ppm) -> list:
    """Non-stub candidates for one replicate, filtered by mass error and ranked by area. Takes one
    replicate's pre-built index entry from build_group_index (an O(1) lookup) rather than
    re-scanning that replicate's found_peaks from scratch."""
    candidates = []
    for p in rep_index_entry["found"].get((fragment_cut, modification, charge), []):
        ppm = (getattr(p, "quality_metrics", {}) or {}).get("median_error_ppm", np.nan)
        ppm_ok = ppm is None or (isinstance(ppm, float) and np.isnan(ppm)) or abs(ppm) <= max_ppm
        if ppm_ok:
            candidates.append(p)
    candidates.sort(key=lambda p: -(getattr(p, "trapz_area", 0.0) or 0.0))
    return candidates


def find_selected_peak(rep_index_entry: dict, fragment_cut, modification, charge):
    """Finds a previously curated real peak for this cut/modification/charge (session continuation)."""
    return rep_index_entry["selected"].get((fragment_cut, modification, charge))


def find_stub_peak(rep_index_entry: dict, fragment_cut, modification, charge):
    """Finds the original placeholder stub PeptidePeak for this cut/modification/charge."""
    return rep_index_entry["stub"].get((fragment_cut, modification, charge))


def pick_default_candidate(candidates: list, consensus_rt: float):
    """Picks the candidate whose RT is closest to the cross-replicate consensus, favoring the peak
    everyone agrees on over a higher-area outlier."""
    if not candidates:
        return None
    return min(candidates, key=lambda p: (abs(p.rt - consensus_rt), -(getattr(p, "trapz_area", 0.0) or 0.0)))


def _replicate_sort_key(rep_id):
    try:
        return int(str(rep_id).split("_Rep_")[-1])
    except (ValueError, IndexError):
        return str(rep_id)


def _fmt(value, fmt="{:.3f}", na="N/A"):
    try:
        if value is None:
            return na
        if isinstance(value, float) and np.isnan(value):
            return na
        return fmt.format(value)
    except Exception:
        return na


def _in_notebook() -> bool:
    """Detects whether this is running inside a Jupyter kernel (vs. a plain terminal)."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def _triggered_component_id():
    """Returns the dict/str id of whichever component triggered the current callback."""
    ctx = dash.callback_context
    tid = getattr(ctx, "triggered_id", None)
    if tid is not None:
        return tid
    prop_id = ctx.triggered[0]["prop_id"].rsplit(".", 1)[0]
    try:
        return json.loads(prop_id)
    except (ValueError, TypeError):
        return prop_id


# ==============================================================================
# --- GROUP LOADING / CACHING ---
# ==============================================================================

def load_group(group_idx: int, groups_meta: list, max_ppm: float) -> dict:
    """Lazily loads and caches everything needed to render one group's dashboard."""
    if group_idx in _GROUP_CACHE:
        return _GROUP_CACHE[group_idx]

    meta = groups_meta[group_idx]
    print(f"[Loading] Group {group_idx + 1}/{len(groups_meta)}: {meta['group_folder_name']}")

    group_results = load_replicate_group_results(meta["input_pkl_path"])

    spectra = {}
    for rep_id, rep_results in group_results.replicates.items():
        try:
            spectra[rep_id] = rep_results.experiment.load_spectrum()
        except Exception as e:
            print(f"  [!] Could not load spectrum for {rep_id}: {e}")
            spectra[rep_id] = None

    rep_index, combos = build_group_index(group_results)

    blocks = []
    for fragment_cut, charge, modification, target in combos:
        # Candidates are computed exactly once per (block, replicate) here and reused both for
        # the consensus RT and for the row itself -- previously this same lookup ran twice
        # (once in compute_consensus_rt, once again to build the row).
        rows = []
        top_rts = []
        for rep_id in group_results.replicates.keys():
            candidates = get_candidates(rep_index[rep_id], fragment_cut, modification, charge, max_ppm)
            if candidates:
                top_rts.append(candidates[0].rt)
            rows.append({"rep_id": rep_id, "candidates": candidates})
        consensus_rt = float(np.median(top_rts)) if top_rts else 8.0
        rt_lo, rt_hi = consensus_rt - CONSENSUS_MARGIN, consensus_rt + CONSENSUS_MARGIN
        rows.sort(key=lambda r: _replicate_sort_key(r["rep_id"]))
        blocks.append({
            "fragment_cut": fragment_cut, "charge": charge, "modification": modification,
            "target": target, "rows": rows, "consensus_rt": consensus_rt,
            "consensus_window": (rt_lo, rt_hi),
        })
    blocks.sort(key=lambda b: (str(b["fragment_cut"]), b["charge"], b["modification"]))

    cache = {
        "group_folder_name": meta["group_folder_name"],
        "output_pkl_path": meta["output_pkl_path"],
        "meta": meta,  # same dict object as groups_meta[group_idx] -- mutating it here is visible there too
        "group_results": group_results,
        "rep_index": rep_index,
        "spectra": spectra,
        "blocks": blocks,
    }
    _GROUP_CACHE[group_idx] = cache

    _seed_dynamic_state(group_idx, cache)
    return cache


def _seed_dynamic_state(group_idx: int, cache: dict):
    """Initializes the dynamic peak for every column: reuse a previously curated selection if one
    exists (session continuation), else the auto-detected default candidate, else build a fresh
    peak at the consensus RT window for a true stub replicate."""
    rep_index = cache["rep_index"]

    for block_idx, block in enumerate(cache["blocks"]):
        target = block["target"]
        charge = block["charge"]
        fragment_cut = block["fragment_cut"]
        modification = block["modification"]
        consensus_rt = block["consensus_rt"]
        rt_lo, rt_hi = block["consensus_window"]

        for row in block["rows"]:
            rep_id = row["rep_id"]
            key = (group_idx, block_idx, rep_id)
            if key in _DYNAMIC_STATE:
                continue

            prior_peak = find_selected_peak(rep_index[rep_id], fragment_cut, modification, charge)
            if prior_peak is not None:
                r_start = prior_peak.rt_start if prior_peak.rt_start is not None else rt_lo
                r_end = prior_peak.rt_end if prior_peak.rt_end is not None else rt_hi
                _DYNAMIC_STATE[key] = {"peak": prior_peak, "is_stub": False, "rt_start": r_start, "rt_end": r_end}
                continue

            default_candidate = pick_default_candidate(row["candidates"], consensus_rt)
            if default_candidate is not None:
                r_start = default_candidate.rt_start if default_candidate.rt_start is not None else rt_lo
                r_end = default_candidate.rt_end if default_candidate.rt_end is not None else rt_hi
                _DYNAMIC_STATE[key] = {
                    "peak": default_candidate, "is_stub": False, "rt_start": r_start, "rt_end": r_end
                }
                continue

            # No curated selection and no auto-detected candidate -> a true stub. Rather than
            # eagerly building and fully scoring a peak here for every stub column (expensive --
            # process_and_find_peaks' find_peaks_cwt call alone can take many seconds, and this
            # used to run for every stub column in every block before the page could even render),
            # just seed the consensus RT window and leave it unbuilt. The "Stay a stub" checkbox
            # defaults to checked to match; unchecking it and clicking "Refresh preview" builds it
            # on demand, scoped to that one column.
            _DYNAMIC_STATE[key] = {"peak": None, "is_stub": True, "rt_start": rt_lo, "rt_end": rt_hi}


def save_group(group_idx: int) -> str:
    """Writes the current dynamic state back into the ReplicateGroupResults and saves it."""
    cache = _GROUP_CACHE.get(group_idx)
    if cache is None:
        return "Nothing loaded for this group yet."

    group_results = cache["group_results"]
    for block_idx, block in enumerate(cache["blocks"]):
        target = block["target"]
        charge = block["charge"]
        for row in block["rows"]:
            rep_id = row["rep_id"]
            state = _DYNAMIC_STATE.get((group_idx, block_idx, rep_id))
            rep_results = group_results.replicates[rep_id]

            existing = [p for p in rep_results.selected_peaks.get(target, []) if p.charge != charge]
            if state and not state.get("is_stub") and state.get("peak") is not None:
                existing.append(state["peak"])

            if existing:
                rep_results.selected_peaks[target] = existing
            elif target in rep_results.selected_peaks:
                del rep_results.selected_peaks[target]

    out_path = cache["output_pkl_path"]
    group_results.save(out_path)
    cache["meta"]["is_continuation"] = True  # future navigations should show it as "previously reviewed"
    return f"Saved '{cache['group_folder_name']}' -> {out_path.name}"


# ==============================================================================
# --- RENDERING ---
# ==============================================================================

def fig_to_img(plot_callable, figsize=(3.6, 2.6), **plot_kwargs):
    """Renders a PeptidePeak matplotlib plotting method to an inline base64 <img>."""
    fig, ax = plt.subplots(figsize=figsize)
    try:
        plot_callable(ax=ax, **plot_kwargs)
    except Exception as e:
        ax.clear()
        ax.axis("off")
        ax.text(0.5, 0.5, f"Plot error:\n{e}", ha="center", va="center", fontsize=7, wrap=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=92)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return html.Img(src=f"data:image/png;base64,{encoded}", style=IMG_STYLE)


def render_placeholder(message: str):
    return html.Div(message, style={
        "backgroundColor": "#fff5f5", "color": "#c92a2a", "fontWeight": "bold",
        "textAlign": "center", "padding": "40px 8px", "border": "1px dashed #c92a2a",
        "borderRadius": "4px", "fontSize": "12px", "whiteSpace": "pre-line",
    })


def render_peak_rows(peak):
    """EIC / Zoomed / Raw MS1 trio for a PeptidePeak."""
    rt_center = peak.rt if peak.rt is not None and not (isinstance(peak.rt, float) and np.isnan(peak.rt)) else 8.0
    return html.Div([
        fig_to_img(peak.plot_ms1_eic, time_window_ecc=(rt_center - 1.5, rt_center + 1.5)),
        fig_to_img(peak.plot_ms1_zoomed),
        fig_to_img(peak.plot_ms1_raw),
    ])


def render_summary_rows(peak):
    """Summary table + key stats caption."""
    metrics = getattr(peak, "quality_metrics", {}) or {}
    stats_text = (
        f"RT: {_fmt(peak.rt)} min  [{_fmt(peak.rt_start)} - {_fmt(peak.rt_end)}]  |  "
        f"Area: {_fmt(getattr(peak, 'trapz_area', None), '{:,.0f}')}  |  "
        f"Score: {_fmt(getattr(peak, 'quality_score', None), '{:.2f}')} ({getattr(peak, 'tier', 'N/A')})  |  "
        f"Median Err: {_fmt(metrics.get('median_error_ppm'), '{:.2f}')} ppm"
    )
    return html.Div([
        fig_to_img(peak.plot_summary_table, figsize=(3.6, 2.3)),
        html.Div(stats_text, style={"fontSize": "11px", "padding": "4px", "textAlign": "center"}),
    ])


def render_dynamic_panel(state):
    """Rows driven by the current dynamic state of one column."""
    if state is None or state.get("is_stub") or state.get("peak") is None:
        return render_placeholder("Marked as STUB\n(no peak reconstructed)")
    peak = state["peak"]
    return html.Div([render_peak_rows(peak), render_summary_rows(peak)])


def build_column(group_idx: int, block_idx: int, row: dict, block: dict, cache: dict):
    rep_id = row["rep_id"]
    candidates = row["candidates"]
    fragment_cut = block["fragment_cut"]
    modification = block["modification"]
    charge = block["charge"]
    consensus_rt = block["consensus_rt"]
    rep_idx_entry = cache["rep_index"][rep_id]

    prior_peak = find_selected_peak(rep_idx_entry, fragment_cut, modification, charge)
    default_candidate = pick_default_candidate(candidates, consensus_rt)

    if prior_peak is not None:
        reference_peak = prior_peak
        origin_label = "Previously Curated"
        badge_color = "#d0ebff"
    elif default_candidate is not None:
        reference_peak = default_candidate
        origin_label = "Auto-detected"
        badge_color = "#e9ecef"
    else:
        reference_peak = find_stub_peak(rep_idx_entry, fragment_cut, modification, charge)
        origin_label = "STUB"
        badge_color = "#ffe3e3"

    static_block = render_peak_rows(reference_peak) if reference_peak is not None else render_placeholder(
        "No original data available"
    )

    state = _DYNAMIC_STATE.get((group_idx, block_idx, rep_id))
    window_lo, window_hi = block["consensus_window"]
    if state:
        window_lo = min(window_lo, state["rt_start"])
        window_hi = max(window_hi, state["rt_end"])
    slider_lo = max(0.0, window_lo - SLIDER_PADDING)
    slider_hi = window_hi + SLIDER_PADDING
    slider_value = [state["rt_start"], state["rt_end"]] if state else [window_lo, window_hi]

    slider_id = {"type": "rt-slider", "group": group_idx, "block": block_idx, "rep": rep_id}
    start_input_id = {"type": "rt-start-input", "group": group_idx, "block": block_idx, "rep": rep_id}
    end_input_id = {"type": "rt-end-input", "group": group_idx, "block": block_idx, "rep": rep_id}
    stub_check_id = {"type": "stub-check", "group": group_idx, "block": block_idx, "rep": rep_id}
    refresh_btn_id = {"type": "refresh-btn", "group": group_idx, "block": block_idx, "rep": rep_id}
    panel_id = {"type": "dynamic-panel", "group": group_idx, "block": block_idx, "rep": rep_id}
    candidate_dropdown_id = {"type": "candidate-dropdown", "group": group_idx, "block": block_idx, "rep": rep_id}
    candidate_store_id = {"type": "candidate-store", "group": group_idx, "block": block_idx, "rep": rep_id}

    header_style = {
        "fontWeight": "bold", "textAlign": "center", "padding": "4px", "borderRadius": "4px",
        "backgroundColor": badge_color,
    }
    header_label = f"{rep_id}  [{origin_label}]"
    number_input_style = {"width": "58px", "fontSize": "11px"}

    # The dropdown+store are always rendered (even with zero candidates) so this column's
    # pattern-matching callback wiring stays identical to every other column -- otherwise the
    # RT-slider/number-input sync would silently stop working for stub-only columns.
    if candidates:
        candidate_options = [
            {
                "label": (
                    f"Candidate {i + 1}: RT {c.rt:.2f} | Area {_fmt(getattr(c, 'trapz_area', 0.0), '{:,.0f}')} "
                    f"| Score {_fmt(getattr(c, 'quality_score', None), '{:.2f}')}"
                ),
                "value": i,
            }
            for i, c in enumerate(candidates)
        ]
        candidate_windows = [
            [
                c.rt_start if c.rt_start is not None else c.rt - CONSENSUS_MARGIN,
                c.rt_end if c.rt_end is not None else c.rt + CONSENSUS_MARGIN,
            ]
            for c in candidates
        ]
        dropdown_placeholder = f"Jump to one of {len(candidates)} detected candidate(s)..."
    else:
        candidate_options = []
        candidate_windows = []
        dropdown_placeholder = "No alternative candidates detected"

    return html.Div([
        html.Div(header_label, style=header_style),
        html.Div(origin_label, style={"fontSize": "10px", "color": "#868e96", "textAlign": "center", "marginTop": "4px"}),
        static_block,
        html.Div("Adjustable window", style={
            "fontSize": "10px", "color": "#868e96", "textAlign": "center", "marginTop": "8px"
        }),
        dcc.Store(id=candidate_store_id, data=candidate_windows),
        dcc.Dropdown(
            id=candidate_dropdown_id, options=candidate_options, value=None,
            placeholder=dropdown_placeholder, disabled=not candidates, clearable=True,
            style={"fontSize": "11px", "marginBottom": "4px"},
        ),
        html.Div([
            dcc.Input(id=start_input_id, type="number", value=round(slider_value[0], 3), step=0.01,
                      style=number_input_style),
            html.Div(dcc.RangeSlider(
                id=slider_id, min=round(slider_lo, 2), max=round(slider_hi, 2), step=0.01,
                value=slider_value, allowCross=False, updatemode="mouseup",
                tooltip={"placement": "bottom", "always_visible": False},
            ), style={"flex": "1"}),
            dcc.Input(id=end_input_id, type="number", value=round(slider_value[1], 3), step=0.01,
                      style=number_input_style),
        ], style={"display": "flex", "alignItems": "center", "gap": "6px"}),
        dcc.Checklist(
            id=stub_check_id,
            options=[{"label": " Stay a stub", "value": "stub"}],
            value=["stub"] if (state and state.get("is_stub")) else [],
            style={"fontSize": "11px", "textAlign": "center", "margin": "4px 0"},
        ),
        html.Button("Refresh preview", id=refresh_btn_id, n_clicks=0, style={
            "width": "100%", "fontSize": "11px", "padding": "4px", "marginBottom": "6px",
        }),
        html.Div(render_dynamic_panel(state), id=panel_id),
    ], style={
        "flex": "0 0 340px", "width": "340px", "border": "1px solid #ced4da", "borderRadius": "6px",
        "padding": "8px", "margin": "6px", "backgroundColor": "#fff",
    })


def build_block_section(group_idx: int, block_idx: int, block: dict, cache: dict):
    title = f"Cut {block['fragment_cut']}  |  Charge +{block['charge']}  |  Mod: {block['modification']}"
    columns = [build_column(group_idx, block_idx, row, block, cache) for row in block["rows"]]
    return html.Div([
        html.H3(title, style={"borderBottom": "2px solid #dee2e6", "paddingBottom": "4px"}),
        # flexWrap keeps replicates side by side, matched row-for-row, wrapping only if the
        # window is too narrow to fit them all -- this is what makes cross-replicate QC easy.
        html.Div(columns, style={"display": "flex", "flexWrap": "wrap"}),
    ], style={"marginBottom": "24px", "padding": "10px", "backgroundColor": "#f8f9fa", "borderRadius": "6px"})


def build_group_selector_options(groups_meta: list) -> list:
    """Dropdown options for jumping straight to a group, tagged with its current review status."""
    return [
        {
            "label": f"{i + 1}. {m['group_folder_name']}  [{'reviewed' if m['is_continuation'] else 'auto'}]",
            "value": i,
        }
        for i, m in enumerate(groups_meta)
    ]


def build_group_page(group_idx: int, groups_meta: list, max_ppm: float):
    cache = load_group(group_idx, groups_meta, max_ppm)
    if not cache["blocks"]:
        return html.Div(
            "No target/charge combinations were found for this group -- nothing to review.",
            style={"padding": "20px", "color": "#868e96"},
        )
    return html.Div([
        build_block_section(group_idx, block_idx, block, cache)
        for block_idx, block in enumerate(cache["blocks"])
    ])


# ==============================================================================
# --- APP ---
# ==============================================================================

def review_single_group(pkl_path: Path | str, port: int = 8051, max_ppm: float = 15.0):
    """Launches the same interactive review dashboard scoped to just one already-saved
    ReplicateGroupResults pickle -- for re-reviewing a single group in more detail (e.g. from a
    notebook) without needing the full input Excel / reports_dir group discovery. Saves back to
    the same file, exactly like continuing a session normally would.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"No such file: {pkl_path}")

    suffix = "_manually_reviewed.pkl.gz"
    group_folder_name = pkl_path.name[: -len(suffix)] if pkl_path.name.endswith(suffix) else pkl_path.stem

    groups_meta = [{
        "group_folder_name": group_folder_name,
        "input_pkl_path": pkl_path,
        "output_pkl_path": pkl_path,
        "is_continuation": True,
    }]
    run_review_app(groups_meta, port=port, max_ppm=max_ppm)


def peak_reviewing_pipeline(input_excel: Path, reports_dir: Path, port: int = 8051, max_ppm: float = 15.0):
    """Launches the interactive peak review & stub re-analysis dashboard for every group in
    `input_excel`. Blocks until the server is stopped (Ctrl+C)."""
    groups_meta = discover_groups(input_excel, reports_dir)

    if not groups_meta:
        print("No groups with a cohort or manually reviewed pickle were found. Nothing to do.")
        return

    run_review_app(groups_meta, port=port, max_ppm=max_ppm)


def run_review_app(groups_meta: list, port: int = 8051, max_ppm: float = 15.0):
    """Builds and runs the Dash review app for an already-resolved list of groups (each a dict
    with group_folder_name/input_pkl_path/output_pkl_path/is_continuation). Blocks until the
    server is stopped (Ctrl+C), or -- inside a Jupyter notebook -- runs in the background and
    prints a URL to open."""
    total_groups = len(groups_meta)
    print(f"Found {total_groups} group(s) ready for review.")

    # Every group index needs a fresh cache slot: run_review_app may be called more than once in
    # the same process (e.g. re-running a notebook cell), and stale entries from a previous call
    # would otherwise leak into this one since _GROUP_CACHE/_DYNAMIC_STATE are process-lifetime.
    _GROUP_CACHE.clear()
    _DYNAMIC_STATE.clear()

    app = Dash(__name__)
    app.config.suppress_callback_exceptions = True
    app.title = "Peak Review Dashboard"

    app.layout = html.Div([
        dcc.Store(id="group-index", data=0),
        html.Div(id="header-bar", style={
            "position": "sticky", "top": 0, "backgroundColor": "#343a40", "color": "white",
            "padding": "10px 16px", "zIndex": 100,
        }),
        html.Div([
            html.Button("< Previous Group", id="prev-btn", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("Skip (no save) >", id="skip-btn", n_clicks=0, style={"marginRight": "8px"}),
            html.Button("Save & Next >", id="save-next-btn", n_clicks=0, style={"fontWeight": "bold"}),
            html.Span("  Jump to group:", style={"marginLeft": "16px", "fontSize": "13px"}),
            dcc.Dropdown(
                id="group-selector", options=build_group_selector_options(groups_meta), value=0, clearable=False,
                style={"display": "inline-block", "width": "420px", "verticalAlign": "middle", "fontSize": "13px"},
            ),
            html.Span(id="save-status", style={"marginLeft": "16px", "color": "#2f9e44"}),
        ], style={"padding": "10px 16px", "display": "flex", "alignItems": "center"}),
        dcc.Loading(html.Div(id="group-content", style={"padding": "0 16px 40px 16px"})),
    ])

    @app.callback(Output("header-bar", "children"), Input("group-index", "data"))
    def update_header(group_idx):
        meta = groups_meta[group_idx]
        source_note = (
            f"Source: {meta['input_pkl_path'].name}  (previously reviewed -- continuing)"
            if meta["is_continuation"] else
            f"Source: {meta['input_pkl_path'].name}  (auto-generated cohort -- first review)"
        )
        return html.Div([
            html.Div(f"Peak Review -- Group {group_idx + 1}/{total_groups}: {meta['group_folder_name']}",
                     style={"fontWeight": "bold"}),
            html.Div(source_note, style={"fontSize": "12px", "color": "#ced4da"}),
        ])

    @app.callback(Output("group-content", "children"), Input("group-index", "data"))
    def update_content(group_idx):
        return build_group_page(group_idx, groups_meta, max_ppm)

    # group-selector.value and group-index.data are kept in sync in both directions (picking a
    # group from the dropdown jumps straight there; stepping via Prev/Skip/Save&Next updates the
    # dropdown's displayed selection to match), via two independent writers to group-index.data
    # (allow_duplicate=True). They're kept as separate callbacks -- rather than one callback
    # reacting to both the buttons and the dropdown -- specifically so that the dropdown-driven
    # echo triggered by a button click can never clobber the "Saved..." status message.
    @app.callback(
        Output("group-index", "data", allow_duplicate=True),
        Output("save-status", "children"),
        Output("group-selector", "options"),
        Input("prev-btn", "n_clicks"),
        Input("skip-btn", "n_clicks"),
        Input("save-next-btn", "n_clicks"),
        State("group-index", "data"),
        prevent_initial_call=True,
    )
    def navigate_via_buttons(prev_clicks, skip_clicks, save_clicks, group_idx):
        triggered = _triggered_component_id()
        status = ""
        new_idx = group_idx
        if triggered == "prev-btn":
            new_idx = max(0, group_idx - 1)
        elif triggered == "skip-btn":
            new_idx = min(total_groups - 1, group_idx + 1)
        elif triggered == "save-next-btn":
            status = save_group(group_idx)
            new_idx = min(total_groups - 1, group_idx + 1)
        return new_idx, status, build_group_selector_options(groups_meta)

    @app.callback(
        Output("group-index", "data", allow_duplicate=True),
        Input("group-selector", "value"),
        prevent_initial_call=True,
    )
    def navigate_via_selector(selected_group_idx):
        if selected_group_idx is None:
            return dash.no_update
        return int(selected_group_idx)

    @app.callback(Output("group-selector", "value"), Input("group-index", "data"), prevent_initial_call=True)
    def sync_selector_from_index(group_idx):
        return group_idx

    # Keeps the RT slider, its two flanking number inputs, and the candidate dropdown all in
    # sync. Typing a value beyond the slider's current min/max (or jumping to a candidate outside
    # the visible window) grows the slider's bounds to match. None of this triggers a recompute --
    # only the "Refresh preview" button below does, so dragging/typing stays snappy.
    @app.callback(
        Output({"type": "rt-start-input", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        Output({"type": "rt-end-input", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        Input({"type": "rt-slider", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def sync_inputs_from_slider(slider_value):
        if not slider_value:
            return dash.no_update, dash.no_update
        return round(slider_value[0], 3), round(slider_value[1], 3)

    @app.callback(
        Output({"type": "rt-slider", "group": MATCH, "block": MATCH, "rep": MATCH}, "min"),
        Output({"type": "rt-slider", "group": MATCH, "block": MATCH, "rep": MATCH}, "max"),
        Output({"type": "rt-slider", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        Input({"type": "rt-start-input", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        Input({"type": "rt-end-input", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        Input({"type": "candidate-dropdown", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        State({"type": "candidate-store", "group": MATCH, "block": MATCH, "rep": MATCH}, "data"),
        State({"type": "rt-slider", "group": MATCH, "block": MATCH, "rep": MATCH}, "min"),
        State({"type": "rt-slider", "group": MATCH, "block": MATCH, "rep": MATCH}, "max"),
        prevent_initial_call=True,
    )
    def reposition_slider(start_value, end_value, candidate_idx, candidate_windows, current_min, current_max):
        triggered = _triggered_component_id()

        if isinstance(triggered, dict) and triggered.get("type") == "candidate-dropdown":
            if candidate_idx is None or not candidate_windows or int(candidate_idx) >= len(candidate_windows):
                return dash.no_update, dash.no_update, dash.no_update
            rt_start, rt_end = candidate_windows[int(candidate_idx)]
        else:
            if start_value is None or end_value is None:
                return dash.no_update, dash.no_update, dash.no_update
            rt_start, rt_end = float(start_value), float(end_value)
            if rt_start > rt_end:
                rt_start, rt_end = rt_end, rt_start

        new_min = max(0.0, min(current_min, rt_start - 0.1))
        new_max = max(current_max, rt_end + 0.1)
        return round(new_min, 2), round(new_max, 2), [rt_start, rt_end]

    # The only thing that recomputes and re-renders the plots -- fires solely on a button click,
    # never from dragging the slider or typing, so exploring RT windows stays smooth.
    @app.callback(
        Output({"type": "dynamic-panel", "group": MATCH, "block": MATCH, "rep": MATCH}, "children"),
        Input({"type": "refresh-btn", "group": MATCH, "block": MATCH, "rep": MATCH}, "n_clicks"),
        State({"type": "rt-slider", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        State({"type": "stub-check", "group": MATCH, "block": MATCH, "rep": MATCH}, "value"),
        prevent_initial_call=True,
    )
    def update_dynamic_panel(n_clicks, rt_value, stub_value):
        # dash collapses outputs_list to a plain dict (instead of a list-of-one) when the
        # callback has a single Output, so both shapes have to be handled here.
        outputs_list = dash.callback_context.outputs_list
        out_id = (outputs_list[0] if isinstance(outputs_list, list) else outputs_list)["id"]
        group_idx, block_idx, rep_id = out_id["group"], out_id["block"], out_id["rep"]

        cache = _GROUP_CACHE.get(group_idx)
        if cache is None or rt_value is None:
            return dash.no_update

        block = cache["blocks"][block_idx]
        stay_stub = "stub" in (stub_value or [])
        rt_start, rt_end = rt_value

        if stay_stub:
            new_state = {"peak": None, "is_stub": True, "rt_start": rt_start, "rt_end": rt_end}
        else:
            rep_results = cache["group_results"].replicates[rep_id]
            spectrum = cache["spectra"].get(rep_id)
            if spectrum is None:
                new_state = {"peak": None, "is_stub": True, "rt_start": rt_start, "rt_end": rt_end}
            else:
                try:
                    peak = pipeline_find_and_score_peak(
                        experiment=rep_results.experiment, spectrum=spectrum, target=block["target"],
                        charge=block["charge"], rt_start=rt_start, rt_end=rt_end,
                    )
                    new_state = {"peak": peak, "is_stub": False, "rt_start": rt_start, "rt_end": rt_end}
                except Exception as e:
                    print(f"  [!] Recompute failed for {rep_id}: {e}")
                    new_state = {"peak": None, "is_stub": True, "rt_start": rt_start, "rt_end": rt_end}

        _DYNAMIC_STATE[(group_idx, block_idx, rep_id)] = new_state
        return render_dynamic_panel(new_state)

    print(f"\nOpen the dashboard in your browser at: http://127.0.0.1:{port}\n")
    if _in_notebook():
        # Force "external" rather than Dash's notebook default of an inline iframe -- this
        # dashboard is wide and dense (many replicate columns per block), which an inline iframe
        # is too cramped for. Falls back to a plain blocking run() if this Dash version doesn't
        # support jupyter_mode (older versions predating Jupyter auto-detection).
        try:
            app.run(debug=False, port=port, jupyter_mode="external")
            return
        except TypeError:
            pass
    app.run(debug=False, port=port)


def main():
    args = parse_args()
    peak_reviewing_pipeline(args.input_excel, args.reports_dir, port=args.port, max_ppm=args.max_ppm)


if __name__ == "__main__":
    main()
