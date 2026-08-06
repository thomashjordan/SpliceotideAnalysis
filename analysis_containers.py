import numpy as np
import pandas as pd
import scipy as sp
import pyteomics as pyteomics
import warnings
import pathlib as Path
import gzip
import pickle as pkl
import ast
import textwrap
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import DBSCAN

import matplotlib as mpl
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
import seaborn as sns

from modification_dictionary import aa_comp_custom as aa_custom

from peptide_functions import *
from peptide_classes import *
from classes_spectrometry import *
# ==============================================================================
# --- BASE MIXIN FOR SERIALIZATION ---
# ==============================================================================

class SerializableResult:
    """Provides automated, compressed save/load functionality using gzip + pickle."""

    def save(self, filepath: str | Path) -> None:
        """Saves the entire object to a compressed gzip pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(filepath, 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
        print(f"Successfully saved {self.__class__.__name__} to: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path):
        """Loads a compressed result object from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No saved file found at: {filepath}")
        with gzip.open(filepath, 'rb') as f:
            obj = pkl.load(f)
        print(f"Successfully loaded {cls.__name__} from: {filepath}")
        return obj


class AnalysisResults(SerializableResult):
    """Manages targets, peaks, and manual override tracking for a single replicate run."""

    def __init__(self, experiment):
        self.experiment = experiment
        self.found_peaks = {}  # {TargetPeptide: [list of PeptidePeak]}
        self.selected_peaks = {}
        self.summary_df = None

    def expand_and_initialize_targets(self, mod_dict, verbose=False):

        """Separately expands the base (unmodified) sequence and any modified variants."""
        mod_target_objects = []
        comp_dict = globals().get('aa_comp_custom', mod_dict)

        if not hasattr(self.experiment, 'target_peptides') or self.experiment.target_peptides is None:
            self.experiment.target_peptides = []

        def _parse_mods(mods):
            if not mods:
                return {"Unmodified": True}
            if isinstance(mods, dict):
                return mods
            if isinstance(mods, str):
                return {mods: 1} if mods else {"Unmodified": True}
            if isinstance(mods, (list, tuple)):
                if len(mods) == 0:
                    return {"Unmodified": True}
                if len(mods) == 2 and isinstance(mods[0], str) and isinstance(mods[1], (int, float, bool)):
                    return {mods[0]: mods[1]}
                parsed = {}
                for item in mods:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        parsed[item[0]] = item[1]
                    elif isinstance(item, str):
                        parsed[item] = 1
                    elif isinstance(item, dict):
                        parsed.update(item)
                return parsed if parsed else {"Unmodified": True}
            return {"Unmodified": True}

        # 1. Process Base Sequence
        base_seq = getattr(self.experiment, 'peptide_sequence', None) or getattr(self.experiment, 'base_sequence', None)

        if base_seq:
            base_seq_clean = str(base_seq).replace('*', '').strip()
            if base_seq_clean:
                base_peptides = generate_targets(
                    peptide_sequence=base_seq_clean,
                    digestion=getattr(self.experiment, 'digestion', None),
                    mod_dict=mod_dict,
                    core_start=getattr(self.experiment, 'core_start', 0),
                    core_end=getattr(self.experiment, 'core_end', 0)
                )
                for tgt_coords in base_peptides:
                    tgt_obj = TargetPeptide(
                        sequence=tgt_coords[1], mod_dict=mod_dict,
                        fragment_cut=(tgt_coords[0], tgt_coords[2]),
                        modifications={"Unmodified": True}
                    )
                    tgt_obj.get_isotopologues(mod_dict=comp_dict)
                    # FIX: Pass mod_dict to generate_fragments
                    tgt_obj.generate_fragments(mod_dict=mod_dict, types=['b', 'y'], max_charge=1)
                    mod_target_objects.append(tgt_obj)

        # 2. Process Variant (Modified) Sequences
        mod_seqs = getattr(self.experiment, 'modified_sequences', None)
        mod_list = getattr(self.experiment, 'modifications', None)

        if mod_seqs and mod_list:
            for mod_seq, mods in zip(mod_seqs, mod_list):
                mod_seq_clean = str(mod_seq).replace('*', '').strip()
                if not mod_seq_clean:
                    continue
                mod_peptides = generate_targets(
                    peptide_sequence=mod_seq_clean,
                    digestion=getattr(self.experiment, 'digestion', None),
                    mod_dict=mod_dict,
                    core_start=getattr(self.experiment, 'core_start', 0),
                    core_end=getattr(self.experiment, 'core_end', 0)
                )
                mod_desc = _parse_mods(mods)

                for tgt_coords in mod_peptides:
                    tgt_obj = TargetPeptide(
                        sequence=tgt_coords[1], mod_dict=mod_dict,
                        fragment_cut=(tgt_coords[0], tgt_coords[2]),
                        modifications=mod_desc
                    )
                    tgt_obj.get_isotopologues(mod_dict=comp_dict)
                    # FIX: Pass mod_dict to generate_fragments
                    tgt_obj.generate_fragments(mod_dict=mod_dict, types=['b', 'y'], max_charge=1)
                    mod_target_objects.append(tgt_obj)

        self.experiment.target_peptides = mod_target_objects
        print(f"[{self.experiment.spectrum_id}] Initialized {len(mod_target_objects)} target peptides.")

    def promote_stub_to_peak(self, target, charge: int, real_extracted_df: pd.DataFrame,
                             real_apex_df: pd.DataFrame, rt: float, rt_start: float, rt_end: float):
        """Mutates an existing placeholder stub into an active validated true peak."" """
        if target not in self.found_peaks:
            print(f"Target matching {target} not initialized in results.")
            return

        for peak in self.found_peaks[target]:
            if peak.charge == charge and getattr(peak, 'is_stub', False):
                print(f"Promoting stub for target {target.fragment_cut} [z={charge}] to True Peak...")

                # Overwrite internal state properties
                peak.is_stub = False
                peak.extracted = real_extracted_df
                peak.raw_apex = real_apex_df
                peak.rt = rt
                peak.rt_start = rt_start
                peak.rt_end = rt_end

                # Re-run quantification pipelines with new data arrays
                if hasattr(peak, 'integrate_peak'):
                    peak.integrate_peak()
                if hasattr(peak, 'validate_ms1_pattern'):
                    peak.validate_ms1_pattern()
                if hasattr(peak, 'process_ms1_quality'):
                    peak.process_ms1_quality()
                return True
        return False

    def reload_peak_diagnostics(self, peak, spectrum_container, rt_window_expansion=2.0):
        """Fetches raw spectral components and expanded EICs directly from source files

        for targeted manual re-inspection.
        """
        target = peak.target
        charge = peak.charge
        expected_rt = peak.rt

        # Determine scanning coordinates
        rt_range = [expected_rt - rt_window_expansion, expected_rt + rt_window_expansion]

        # 1. Force a raw extraction across the target mass range
        expanded_eic = spectrum_container.get_extracted_compound_chromatogram(
            target.monoisotopic_mass,
            filtered=False,
            tolerance_ppm=25,
            mz_range=[400, 2000],
            rt_range=rt_range
        )

        # 2. Extract localized raw profile matrices
        raw_scans = []
        if hasattr(spectrum_container, 'get_raw_spectra_in_rt_range'):
            raw_scans = spectrum_container.get_raw_spectra_in_rt_range(rt_range)

        return {
            "expanded_eic": expanded_eic if expanded_eic is not None else pd.DataFrame(),
            "raw_scans": raw_scans,
            "target_mzs": getattr(peak, 'isotopologues', {})
        }

    def remove_duplicate_peaks(self):
        self.all_found_peaks = self.found_peaks.copy()
        for target, peaks in self.found_peaks.items():
            if len(peaks) < 2: continue
            sorted_peaks = sorted(peaks, key=lambda x: getattr(x, 'trapz_area', 0.0) or 0.0, reverse=True)
            kept = []
            for p in sorted_peaks:
                p_start, p_end = getattr(p, 'rt_start', None), getattr(p, 'rt_end', None)
                if p_start is None or p_end is None:
                    kept.append(p)
                    continue
                if not any((p_start >= (getattr(k, 'rt_start', 0.0) or 0.0) and p_end <= (
                        getattr(k, 'rt_end', 0.0) or 0.0)) for k in kept):
                    kept.append(p)
            self.found_peaks[target] = kept

    def organize_by_structure(self):
        self.remove_duplicate_peaks()
        organized = {}
        for target, peaks in self.found_peaks.items():
            cut = target.fragment_cut
            mod_key = str(target.modifications)
            for p in peaks:
                charge = getattr(p, 'charge', 0)
                organized.setdefault(cut, {}).setdefault(charge, {}).setdefault(mod_key, []).append(p)
        return organized

    def summarize(self) -> pd.DataFrame:
        org = self.organize_by_structure()
        records = []
        for cut, charges in org.items():
            for charge, mods in charges.items():
                for mod_key, peaks in mods.items():
                    areas = [getattr(p, 'trapz_area', 0.0) for p in peaks if not getattr(p, 'is_stub', False)]
                    valid_areas = [a for a in areas if a is not None]
                    max_area = max(valid_areas) if valid_areas else 0.0
                    records.append({
                        'fragment_cut': cut, 'charge': charge, 'modification': mod_key,
                        'peak_count': len(peaks), 'max_area': max_area, 'peaks': peaks
                    })
        self.summary_df = pd.DataFrame(records)
        return self.summary_df

    def add_peaks(self, target, peaks: list):
        """Appends observed peaks or stubs to a target peptide."""
        if target not in self.found_peaks:
            self.found_peaks[target] = []
        self.found_peaks[target].extend(peaks)


# ==============================================================================
# --- COHORT CONTAINER WITH RATIO PROFILING ---
# ==============================================================================

class ReplicateGroupResults(SerializableResult):
    """Aggregates replicate cohorts, processing modifications ratios, average intensities,

    and technical validation properties.
    """

    def __init__(self, group_name: str):
        self.group_name = group_name
        self.replicates = {}  # {replicate_id: AnalysisResults}

    def add_replicate(self, replicate_id: str, results: AnalysisResults):
        self.replicates[replicate_id] = results

    def generate_modification_state_matrix(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculates a matrix of modification states using only manually curated 'selected_peaks'.

        This function identifies all (fragment_cut, charge) combinations that have at least
        one selected peak across any replicate. For these combinations, it builds a complete
        matrix, ensuring every replicate and every possible modification state is represented,
        creating placeholder "stubs" for missing data points. This allows for robust
        calculation of relative abundances and cohort-wide metrics.

        Combinations of (fragment_cut, charge) with no selected peaks are ignored.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                - A long-form DataFrame with details for each replicate.
                - A summary DataFrame with cohort-averaged metrics.
        """
        # 1. Determine the scope of analysis from selected peaks
        scope = defaultdict(set)  # {fragment_cut: {charge1, charge2, ...}}
        all_targets_by_cut = defaultdict(dict)  # {fragment_cut: {mod_key: TargetPeptide}}

        for rep_id, rep_results in self.replicates.items():
            if not hasattr(rep_results, 'selected_peaks') or not rep_results.selected_peaks:
                continue
            for target, peaks in rep_results.selected_peaks.items():
                cut = target.fragment_cut
                mod_key = str(target.modifications)

                # Store the target object to get all possible mods for this cut later
                if mod_key not in all_targets_by_cut[cut]:
                    all_targets_by_cut[cut][mod_key] = target

                for peak in peaks:
                    scope[cut].add(peak.charge)

        if not scope:
            print("Warning: No selected peaks found in any replicate. Cannot generate modification matrix.")
            return pd.DataFrame(), pd.DataFrame()

        # 2. Build the full matrix with real data and stubs
        replicate_records = []
        num_reps = len(self.replicates)

        # Create a lookup map of selected peaks for faster access: {(rep_id, cut, charge, mod_key): peak}
        selected_peak_map = {}
        for rep_id, rep_results in self.replicates.items():
            if not hasattr(rep_results, 'selected_peaks'): continue
            for target, peaks in rep_results.selected_peaks.items():
                cut = target.fragment_cut
                mod_key = str(target.modifications)
                for peak in peaks:
                    key = (rep_id, cut, peak.charge, mod_key)
                    # If multiple peaks were approved for the same state, take the one with the largest area.
                    if key not in selected_peak_map or (getattr(peak, 'trapz_area', 0) or 0) > (getattr(selected_peak_map[key], 'trapz_area', 0) or 0):
                        selected_peak_map[key] = peak

        # Iterate through the defined scope and build the records for the matrix
        for cut, charges in scope.items():
            possible_mods = all_targets_by_cut[cut].keys()
            for charge in charges:
                for rep_id in self.replicates.keys():
                    for mod_key in possible_mods:
                        key = (rep_id, cut, charge, mod_key)
                        peak = selected_peak_map.get(key)

                        if peak:
                            # A selected peak was found for this combination
                            is_stub = False
                            area = getattr(peak, 'trapz_area', 0.0) or 0.0
                            rt = getattr(peak, 'rt', np.nan)
                            metrics = getattr(peak, 'quality_metrics', {})
                            qc_score = metrics.get('median_error_ppm', np.nan)
                        else:
                            # No selected peak found, create a stub record
                            is_stub = True
                            area = 0.0
                            rt = np.nan
                            qc_score = np.nan

                        replicate_records.append({
                            'replicate_id': rep_id, 'fragment_cut': cut, 'charge': charge,
                            'modification': mod_key, 'absolute_area': area, 'rt': rt,
                            'qc_score': qc_score, 'is_stub': is_stub
                        })

        df_replicates = pd.DataFrame(replicate_records)
        if df_replicates.empty:
            return pd.DataFrame(), pd.DataFrame()

        # 3. Calculate relative areas within each (replicate, cut, charge) group
        df_replicates['overall_signal_intensity'] = df_replicates.groupby(
            ['replicate_id', 'fragment_cut', 'charge']
        )['absolute_area'].transform('sum')

        df_replicates['relative_area'] = np.nan
        mask = df_replicates['overall_signal_intensity'] > 0
        df_replicates.loc[mask, 'relative_area'] = (
                df_replicates.loc[mask, 'absolute_area'] / df_replicates.loc[mask, 'overall_signal_intensity']
        )

        # 4. Collapse into Cohort Average Summary
        group_cols = ['fragment_cut', 'charge', 'modification']
        summary_records = []

        for keys, group in df_replicates.groupby(group_cols):
            cut, charge, mod = keys

            # Retention time calculations omit stubs entirely
            valid_rts = group[~group['is_stub']]['rt'].dropna()
            mean_rt = valid_rts.mean() if not valid_rts.empty else np.nan

            mean_abs_area = group['absolute_area'].mean()
            mean_rel_area = group['relative_area'].mean()
            mean_overall_signal = group['overall_signal_intensity'].mean()
            mean_qc_score = group['qc_score'].dropna().mean() if not group['qc_score'].dropna().empty else np.nan

            active_counts = np.sum(~group['is_stub'])

            summary_records.append({
                'fragment_cut': cut, 'charge': charge, 'modification': mod,
                'avg_retention_time': mean_rt, 'avg_absolute_area': mean_abs_area,
                'avg_relative_area': mean_rel_area, 'avg_overall_signal_intensity': mean_overall_signal,
                'avg_qc_score': mean_qc_score, 'cohort_detection_frequency': active_counts / num_reps
            })

        df_summary = pd.DataFrame(summary_records)
        return df_replicates, df_summary


def _apply_compact_sci_axis(ax):
    """Applies power-of-10 (scientific) y-axis notation to one axis, as compactly as possible --
    small tick labels and a small "x10^N" offset annotation.

    Always builds a FRESH ScalarFormatter here rather than accepting/reusing a shared one: this
    report draws a dozen-plus separate y-axes, and ScalarFormatter carries mutable per-axis state
    (orderOfMagnitude etc.) internally -- sharing one instance across many axes is a known
    matplotlib gotcha where some axes silently end up rendering full un-abbreviated numbers
    instead of the intended notation, since the state gets clobbered by whichever axis last wrote
    to the shared object.
    """
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_scientific(True)
    fmt.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis='y', labelsize=5)
    ax.yaxis.get_offset_text().set_fontsize(5)


def _wrap_text_to_margin(text: str, fontsize: float, fig_width_in: float, margin_cm: float = 3.0) -> str:
    """Wraps text so every line fits within a horizontal band inset by margin_cm on each side of a
    fig_width_in-wide page.

    matplotlib's own wrap=True can't do this: its wrap width is computed as the distance from the
    text's anchor point to the FIGURE edge (doubled for a centered anchor), which for text
    centered on the page always comes out to the full page width, margin or no margin -- that's
    why the caption used to run edge-to-edge. textwrap.fill() sidesteps that, but only if it's
    given an accurate max character count for the available width; a flat guess (e.g. "width=100
    chars") drifts as soon as the font, size, or margin changes.

    Instead, this measures the caption's own natural (unwrapped) rendered width via a throwaway
    TextPath at the real fontsize -- TextPath's coordinates come out directly in points when `size`
    is given in points, so no canvas/renderer draw is needed just to measure text. That gives an
    exact chars-per-point ratio for this specific string/font/size, which is then used to pick the
    wrap width that fills (without exceeding) the margin-inset band.
    """
    margin_in = margin_cm / 2.54
    available_pt = max((fig_width_in - 2 * margin_in) * 72.0, 10.0)

    full_width_pt = TextPath((0, 0), text, size=fontsize, prop=FontProperties()).get_extents().width
    if full_width_pt <= 0 or not text:
        return text

    max_chars = max(20, int(len(text) * available_pt / full_width_pt))
    return textwrap.fill(text, width=max_chars)


DEFAULT_SI_CAPTION_TEMPLATE = (
    "SI Appendix B Figure {i}: Results of coexpression of plasmid {precursor_plasmid} with "
    "{maturase_plasmid} at {expression_temperature}°C for {expression_time} h in "
    "{expression_scale} [mL] of culture volume, the precursor is {precursor_name} and was "
    "digested using {digestion}. The fragment shown ranges from {cut_start} to {cut_end} and "
    "charge state z={charge} is plotted"
)


def format_modification_label(modifications: dict) -> str:
    """Human-readable label for a target's modifications dict, e.g. {'Unmodified': True} ->
    'Unmodified', {'unmodified': None} -> 'Unmodified', {'spliced': 2} -> 'Spliced 2'. Falls back
    to the raw dict for any other modification type (unchanged from before)."""
    if isinstance(modifications, dict) and len(modifications) == 1:
        key, value = next(iter(modifications.items()))
        key_lower = str(key).lower()
        if key_lower == "unmodified":
            return "Unmodified"
        if key_lower == "spliced":
            return f"Spliced {value}"
    return str(modifications)


class GroupedExperiments:
    def __init__(self, *analysis_results):
        if len(analysis_results) < 1:
            raise ValueError("GroupedExperiments requires at least one AnalysisResults objects.")
        self.analysis_results = analysis_results
        # Only set when built via from_replicate_group_results(); enables the matrix-driven
        # select_best_cut_charge()/generate_si_report() methods below. Plain construction (the
        # original *analysis_results calling convention, still used by generate_all_reports()
        # callers) leaves this as None.
        self.group_results = None

    @classmethod
    def from_replicate_group_results(cls, group_results: "ReplicateGroupResults") -> "GroupedExperiments":
        """Builds a GroupedExperiments directly from a curated ReplicateGroupResults -- e.g. one
        loaded from a '*_manually_reviewed.pkl.gz' produced by peak_reviewing_pipeline.py. This is
        the entry point for generate_si_report()."""
        instance = cls(*group_results.replicates.values())
        instance.group_results = group_results
        return instance

    def _get_theoretical_targets_dict(self, cut_id):
        theoretical_targets = {}
        targets = self.analysis_results[0].experiment.target_peptides
        for t in targets:
            if t.fragment_cut == cut_id:
                state_key = f"{t.sequence} {t.modifications}"
                theoretical_targets[state_key] = t
        return theoretical_targets

    def _find_selected_peak(self, replicate_id, fragment_cut, modification, charge, sequence: str = None):
        """Finds the curated PeptidePeak for one (replicate, fragment_cut, modification, charge)
        combination -- i.e. what a human already approved via the review dashboard. Returns None
        if that replicate has no such selection (a stub).

        If `sequence` is given, also requires an exact sequence match. This matters because the
        matching is otherwise only on (fragment_cut, modification, charge) -- fine when every
        replicate in the group targets the same underlying peptide, but if a group ever mixes
        replicates with different base sequences for the "same" cut/modification label (e.g. two
        different point-mutant sequences that both happen to be "Unmodified" at this cut), the
        unqualified match could return the wrong replicate's peak for a given sequence context.
        generate_si_report() always passes this."""
        rep_results = self.group_results.replicates[replicate_id]
        for target, peaks in getattr(rep_results, "selected_peaks", {}).items():
            if target.fragment_cut != fragment_cut or str(target.modifications) != modification:
                continue
            if sequence is not None and target.sequence != sequence:
                continue
            for p in peaks:
                if p.charge == charge and not getattr(p, "is_stub", False):
                    return p
        return None

    def select_best_cut_charge(self):
        """Picks the single (fragment_cut, charge) for this group with the fewest stub replicates
        first, then the highest average area, then the highest average quality_score --
        mirroring plotting_analysis_functions.extract_best_cut_charge()'s selection priority
        (minimal stubs takes precedence). Requires from_replicate_group_results() construction.

        Returns (fragment_cut, charge), or (None, None) if there's no curated data at all.
        """
        if self.group_results is None:
            raise ValueError(
                "select_best_cut_charge() requires a ReplicateGroupResults -- build this "
                "GroupedExperiments via GroupedExperiments.from_replicate_group_results(...)."
            )
        df_replicates, _ = self.group_results.generate_modification_state_matrix()
        if df_replicates.empty:
            return None, None

        # Pull the real 0-1 quality_score off each curated peak -- the matrix's own 'qc_score'
        # column is just the median mass error in ppm, not a real quality metric.
        quality_scores = []
        for _, row in df_replicates.iterrows():
            peak = self._find_selected_peak(row["replicate_id"], row["fragment_cut"], row["modification"], row["charge"])
            quality_scores.append(getattr(peak, "quality_score", np.nan) if peak is not None else np.nan)
        df_replicates = df_replicates.assign(_quality_score=quality_scores)

        detected = df_replicates[~df_replicates["is_stub"]]
        if detected.empty:
            return None, None

        stats = (
            detected.groupby(["fragment_cut", "charge"])
            .agg(non_stub=("is_stub", "size"), avg_area=("absolute_area", "mean"),
                 avg_quality_score=("_quality_score", "mean"))
            .reset_index()
            .sort_values(by=["non_stub", "avg_area", "avg_quality_score"], ascending=[False, False, False])
        )
        best = stats.iloc[0]
        return best["fragment_cut"], int(best["charge"])

    def generate_si_report(self, output_dir: Path, plot_mz: bool = True, plot_ms2: bool = False,
                           dpi: int = 400, rt_range: tuple = None, page_number: int = None,
                           source_path: Path = None, caption_template: str = None) -> Path | None:
        """Generates a single-page, high-resolution A4 SI PDF (vector) for this group's best
        (fragment_cut, charge) -- selected via select_best_cut_charge() (fewest stub replicates
        first, then area, then quality). Only manually curated peaks (selected_peaks) are
        plotted; a replicate with no curated selection for a given modification state is listed
        in the summary table as a stub but is not plotted. Requires from_replicate_group_results()
        construction.

        The whole report is drawn directly onto one fixed A4-sized (210x297mm) matplotlib figure
        -- content rows shrink proportionally to fit as more modification states are added,
        rather than the page growing -- so it stays natively vector (matplotlib's PDF backend)
        and exactly one page regardless of content amount. Groups with many modification states
        will render each row smaller to compensate; there's a practical legibility floor to how
        many states can usefully fit on one A4 page.

        rt_range: optional (min, max) in minutes to cap the overall TIC/EIC chromatogram panel to
        (does not affect the per-state zoomed EIC panels, which are already windowed to the peak).
        page_number: if given, printed at the bottom of the page (and fills in `{i}` in the
        caption below) -- pass a running counter across a batch so pages read consistently once
        combined via plotting_analysis_functions.combine_pdfs().
        source_path: the pickle this group was loaded from, shown at the bottom for provenance.
        caption_template: a str.format() template rendered as the figure caption at the bottom of
        the page, like a real SI figure. Defaults to DEFAULT_SI_CAPTION_TEMPLATE (module-level,
        in this file) -- pass your own string with the same {placeholder} style to customize the
        wording/appendix label/etc.; available placeholders are: i (the page number),
        precursor_plasmid, precursor_name, maturase_plasmid, maturase_name, digestion,
        expression_temperature, expression_time, expression_scale, cut (the raw (start, end)
        tuple), cut_start, cut_end, charge.
        """
        if self.group_results is None:
            raise ValueError(
                "generate_si_report() requires a ReplicateGroupResults -- build this "
                "GroupedExperiments via GroupedExperiments.from_replicate_group_results(...)."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cut_id, charge = self.select_best_cut_charge()
        if cut_id is None:
            print("No curated peaks found for this group -- skipping SI report.")
            return None

        target_dict = self._get_theoretical_targets_dict(cut_id)
        all_possible_states = sorted(list(target_dict.keys()))
        state_labels = {sk: format_modification_label(target_dict[sk].modifications) for sk in all_possible_states}

        all_collected_peaks = []
        stub_entries = []      # (exp_id, state_key) -- no curated selection at all: table only
        excluded_entries = []  # (exp_id, state_key, peak) -- curated but Reject-tier: table only
        final_peaks_map = defaultdict(dict)

        for state_key in all_possible_states:
            target_obj = target_dict[state_key]
            modification = str(target_obj.modifications)

            for rep_id, rep_results in self.group_results.replicates.items():
                exp_id = rep_results.experiment.spectrum_id
                # Requiring an exact sequence match (not just cut+modification+charge) here is
                # what guarantees only true matrix-approved peaks for THIS state ever make it
                # into all_collected_peaks -- see _find_selected_peak's docstring.
                peak = self._find_selected_peak(rep_id, cut_id, modification, charge,
                                                sequence=target_obj.sequence)

                if peak is None:
                    stub_entries.append((exp_id, state_key))
                    continue

                # A curated selection exists here, but if its own quality tier is "Reject" --
                # calculate_quality_score()'s designation for a near-zero score, usually driven by
                # a failed/NaN mass-error match -- it isn't a real peak regardless of who/what
                # marked it "selected" (this is commonly a review-dashboard auto-picked default
                # candidate that was never actually vetted, since candidates with an unmeasurable
                # ppm error bypass the max_ppm quality filter there by design). Excluded from every
                # plot, but still listed in the table with its real (bad) metrics so the
                # discrepancy between "selected" and "good" stays visible instead of silently
                # rendered as if it were legitimate data.
                if getattr(peak, "tier", None) == "Reject":
                    excluded_entries.append((exp_id, state_key, peak))
                    continue

                peak.quality_metrics["is_selected"] = True
                peak.quality_metrics["group_pass"] = True
                if not hasattr(peak, "target"):
                    peak.target = peak
                all_collected_peaks.append(peak)
                final_peaks_map[exp_id][state_key] = (peak, True, True)

        df_ratios = self._get_ratios_dataframe(final_peaks_map, all_possible_states)

        safe_cut = str(cut_id).replace(" ", "_").replace("/", "-")
        page_prefix = f"{page_number:03d}_" if page_number is not None else ""
        filename = f"{page_prefix}{self.group_results.group_name}_SI_Report_Cut_{safe_cut}_z{charge}.pdf"
        output_path = output_dir / filename

        # --- Header text (scientific/experimental metadata) ---
        rep_exp = self.analysis_results[0].experiment
        header_text = (
            f"Maturase Plasmid: {getattr(rep_exp, 'maturase_plasmid', 'N/A')}   |   "
            f"Maturase: {getattr(rep_exp, 'maturase_name', 'N/A')}   |   "
            f"Precursor Plasmid: {getattr(rep_exp, 'precursor_plasmid', 'N/A')}   |   "
            f"Precursor: {getattr(rep_exp, 'precursor_name', 'N/A')}\n"
            f"Digestion: {getattr(rep_exp, 'digestion', 'N/A')}   |   "
            f"Temp: {getattr(rep_exp, 'expression_temperature', 'N/A')}°C   |   "
            f"Scale: {getattr(rep_exp, 'expression_scale', 'N/A')} mL   |   "
            f"Time: {getattr(rep_exp, 'expression_time', 'N/A')} h\n"
            f"Cut: {cut_id}   |   Charge: +{charge}"
        )

        # --- Figure caption (the "like a real SI figure" bit) ---
        cut_start, cut_end = cut_id if isinstance(cut_id, (tuple, list)) and len(cut_id) == 2 else (cut_id, cut_id)
        caption_context = {
            "i": page_number if page_number is not None else "?",
            "precursor_plasmid": getattr(rep_exp, "precursor_plasmid", "N/A"),
            "precursor_name": getattr(rep_exp, "precursor_name", "N/A"),
            "maturase_plasmid": getattr(rep_exp, "maturase_plasmid", "N/A"),
            "maturase_name": getattr(rep_exp, "maturase_name", "N/A"),
            "digestion": getattr(rep_exp, "digestion", "N/A"),
            "expression_temperature": getattr(rep_exp, "expression_temperature", "N/A"),
            "expression_time": getattr(rep_exp, "expression_time", "N/A"),
            "expression_scale": getattr(rep_exp, "expression_scale", "N/A"),
            "cut": cut_id,
            "cut_start": cut_start,
            "cut_end": cut_end,
            "charge": charge,
        }
        caption_text = (caption_template or DEFAULT_SI_CAPTION_TEMPLATE).format(**caption_context)

        # --- Provenance line (kept small/secondary -- the caption above is the real SI-figure
        # text; this is just an internal breadcrumb for tracing a page back to its source file).
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        if source_path is not None:
            source_name = Path(source_path).name
            _suffix = "_manually_reviewed.pkl.gz"
            if source_name.endswith(_suffix):
                source_name = source_name[: -len(_suffix)]
        else:
            source_name = self.group_results.group_name
        footer_left = f"{source_name}   |   Generated: {generated_at}"
        footer_right = f"Page {page_number}" if page_number is not None else ""

        rep_spectrum = MassSpectrum(mzml_path=rep_exp.mzml_path, output_path=rep_exp.output_path,
                                    spectrum_id=rep_exp.spectrum_id)
        rep_spectrum.load()

        palette = mpl.pyplot.cm.tab10.colors
        while len(palette) < len(all_possible_states):
            palette = palette + palette
        color_map = {state: palette[i] for i, state in enumerate(all_possible_states)}

        num_states = len(all_possible_states)
        base_rows = 5 if plot_mz else 3
        total_rows = base_rows + (num_states * 2)
        hr = [1.5, 2.0, 0.2] + [1] * (total_rows - 3)

        # A4 portrait, 210x297mm, fixed regardless of content. Deliberately NOT using
        # constrained_layout here: combined with this figure's nested GridSpecFromSubplotSpec
        # calls it produced unpredictable results in practice (large unexplained blank margins,
        # and titles/labels from adjacent panels overlapping instead of being kept apart). Manual
        # margins below are fully deterministic instead.
        fig = mpl.pyplot.figure(figsize=(8.2677, 11.6929))

        # Header and footer are thin fixed margins (not extra GridSpec rows), so they can't
        # balloon in size the way the previous nested-outer-GridSpec approach did. Content fills
        # everything in between. The footer margin is taller than the header's to leave room for
        # the wrapped, multi-line figure caption above the small provenance/page-number line.
        header_top_y = 0.988
        # content_bottom is raised (0.10 -> 0.17) to leave a real gap between the last plot row's
        # x-axis labels and the caption below it, and caption_y is dropped further down into that
        # freed space (closer to the footer than to the plots) per feedback that the caption
        # should sit "much lower" -- it still has ~0.06 of figure height (>2.5cm) below it before
        # the footer line, comfortably enough for several wrapped lines at fontsize 7.5.
        content_top, content_bottom = 0.935, 0.17
        caption_y = 0.11
        footer_y = 0.012

        fig.text(0.5, header_top_y, header_text, ha='center', va='top', fontsize=7, linespacing=1.5)

        gs = mpl.gridspec.GridSpec(total_rows, 8, figure=fig, height_ratios=hr,
                                   left=0.07, right=0.97, top=content_top, bottom=content_bottom,
                                   hspace=0.7, wspace=0.5)

        sns.set_style("white")
        mpl.pyplot.rcParams.update({'font.size': 6})

        # --- 1. Table ---
        ax_table = fig.add_subplot(gs[0, :6])
        self._plot_si_summary_table(ax_table, all_collected_peaks, stub_entries, state_labels, excluded_entries)

        # --- 2. Bar Plot ---
        ax_bar = fig.add_subplot(gs[0, 6:])
        self._plot_si_ratio_bar(ax_bar, df_ratios, color_map)

        # --- 3. Chromatograms ---
        gs_chrom = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], hspace=0.0, height_ratios=[1, 1])
        ax_tic = fig.add_subplot(gs_chrom[0, :])
        ax_eic = fig.add_subplot(gs_chrom[1, :], sharex=ax_tic)
        self._plot_si_chromatograms(ax_tic, ax_eic, rep_spectrum, all_collected_peaks, color_map, rt_range=rt_range)

        # --- 4. MZ Overview + 5. Per-state detail rows ---
        current_row = 3
        if plot_mz:
            ax_mz = fig.add_subplot(gs[current_row:current_row + 2, :])
            current_row += 2
            self._plot_si_mz_overview(ax_mz, all_collected_peaks, color_map)

            for i, state in enumerate(all_possible_states):
                row_idx = current_row + (i * 2)
                state_label = state_labels[state]
                isoform_peaks = [p for p in all_collected_peaks
                                if f"{p.target.sequence} {p.target.modifications}" == state]
                isoform_peaks.sort(key=lambda x: getattr(x, 'trapz_area', 0) or 0, reverse=True)
                color = color_map[state]

                ax_peak_ecc = fig.add_subplot(gs[row_idx:row_idx + 1, 4:])
                ax_raw_apex = fig.add_subplot(gs[row_idx:row_idx + 2, :4])
                ax_ms2 = None
                if plot_ms2:
                    ax_raw_apex = fig.add_subplot(gs[row_idx:row_idx + 2, :2])
                    ax_ms2 = fig.add_subplot(gs[row_idx:row_idx + 2, 2:4])
                ax_seq = fig.add_subplot(gs[row_idx + 1:row_idx + 2, 4:])

                self._plot_si_zoomed_ecc(ax_peak_ecc, isoform_peaks, color, state_label)
                self._plot_si_isotope_pattern(ax_raw_apex, isoform_peaks, color, state_label)

                passed_isoform_peaks = [p for p in isoform_peaks if p.quality_metrics.get('overall_pass', False)]
                rep_peak = (passed_isoform_peaks[0] if passed_isoform_peaks
                           else isoform_peaks[0] if isoform_peaks else None)
                b_ions, y_ions = self._compute_ms2_matches(isoform_peaks)

                if plot_ms2:
                    self._plot_si_ms2_spectrum(ax_ms2, b_ions, y_ions)
                self._plot_si_sequence_annotations(ax_seq, rep_peak, b_ions, y_ions, state_label)

        # Figure caption -- wrap=True lets matplotlib break it across lines to fit the page width,
        # Wrapped manually (rather than fig.text(wrap=True), which wraps to the full page width
        # with no margin control -- that ran the caption edge-to-edge) so there's a real left/right
        # margin, and anchored va='top' so it grows DOWNWARD from caption_y regardless of how many
        # lines that takes, rather than growing upward into the plotted content above it.
        # 3cm left/right margin: matplotlib's wrap=True can't honor an arbitrary margin (see
        # _wrap_text_to_margin's docstring), so the wrap width is measured directly from the page
        # width in inches minus the requested margin, at this exact fontsize.
        wrapped_caption = _wrap_text_to_margin(caption_text, fontsize=7.5,
                                               fig_width_in=fig.get_size_inches()[0], margin_cm=3.0)
        fig.text(0.5, caption_y, wrapped_caption, ha='center', va='top', fontsize=7.5, linespacing=1.4)

        fig.text(0.03, footer_y, footer_left, ha='left', va='bottom', fontsize=6)
        fig.text(0.97, footer_y, footer_right, ha='right', va='bottom', fontsize=8, weight='bold')

        # No bbox_inches='tight' here -- that would crop the page down to its content bounding
        # box, which would defeat the point of an exact A4-sized page. The PDF backend is vector
        # natively, so dpi only matters for anything that ends up rasterized.
        mpl.pyplot.savefig(output_path, dpi=dpi)
        mpl.pyplot.close(fig)
        return output_path

    def _plot_si_summary_table(self, ax, collected_peaks: list, stub_entries: list, state_labels: dict,
                               excluded_entries: list = None):
        """Row 1: the per-peak QC/selection table, plus one row per undetected replicate (STUB)
        and one row per curated-but-Reject-tier replicate (EXCLUDED, real metrics shown so the
        "selected yet obviously bad" discrepancy stays visible instead of just disappearing)."""
        ax.axis('off')
        table_data = []

        for p in collected_peaks:
            qc_val = p.quality_metrics.get('overall_pass', False)
            qc_str = "PASS" if qc_val else "FAIL"
            qc_group = "PASS" if p.quality_metrics.get('group_pass', False) else "FAIL"
            is_selected = "Yes" if p.quality_metrics.get('is_selected', False) else "No"
            score = getattr(p, 'quality_score', 'N/A')
            tier = getattr(p, 'tier', 'N/A')
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)

            table_data.append([
                p.experiment.spectrum_id,
                p.target.sequence,
                format_modification_label(p.target.modifications),
                f"{p.rt:.2f}",
                f"{p.quality_metrics.get('final_trapz_area') or 0:.2e}",
                f"{p.quality_metrics.get('median_error_ppm') or 0:.2f}",
                qc_str,
                qc_group,
                is_selected,
                score_str,
                tier
            ])

        for exp_id, state_key, p in (excluded_entries or []):
            score = getattr(p, 'quality_score', 'N/A')
            score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
            rt_str = f"{p.rt:.2f}" if p.rt is not None else "-"
            table_data.append([
                exp_id,
                p.target.sequence,
                format_modification_label(p.target.modifications),
                rt_str,
                f"{p.quality_metrics.get('final_trapz_area') or 0:.2e}",
                f"{p.quality_metrics.get('median_error_ppm') or 0:.2f}",
                "FAIL", "EXCL", "Yes*", score_str, getattr(p, 'tier', 'Reject'),
            ])

        for exp_id, state_key in (stub_entries or []):
            seq, _, _ = state_key.partition(" ")
            mod_label = state_labels.get(state_key, "-")
            table_data.append([exp_id, seq, mod_label, "-", "-", "-", "STUB", "STUB", "No", "-", "Stub"])

        if not table_data:
            table_data = [["No Selected Peaks", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]]

        col_labels = ["Exp", "Seq", "Mod", "RT", "Area", "Err", "QC", "GroupOK", "Selected", "Score", "Tier"]
        # Narrower for the columns that only ever hold a short status word/number (PASS/FAIL/EXCL,
        # a 4-digit RT, etc.). Exp holds full spectrum_id strings (often the longest content in
        # the table), so it gets extra width taken from Seq -- Seq is shrunk to compensate but
        # stays legible since its own font is reduced below (see small_font_cols).
        col_widths = [0.16, 0.17, 0.14, 0.05, 0.08, 0.05, 0.05, 0.055, 0.055, 0.05, 0.055]
        # bbox=[0,0,1,1] fills the whole axes with the table (no outer margin), making it as big
        # as the allotted grid cell allows instead of matplotlib's default auto-sized/padded table.
        table = ax.table(cellText=table_data, colLabels=col_labels, colWidths=col_widths,
                         loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(5)
        # Tighten row height (matplotlib's default table row padding is generous) now that the
        # narrow status columns are smaller too -- keeps the whole table more compact overall.
        table.scale(1, 0.8)
        # Exp (spectrum_id strings) and Seq (full peptide sequences) are consistently the longest
        # cell contents in the table -- a smaller font just for these two columns lets their
        # widened/narrowed cells above actually fit the text instead of it being cut off or
        # forcing the whole table to auto-shrink the shared font size.
        small_font_cols = {0, 1}
        for (row, col), cell in table.get_celld().items():
            if col in small_font_cols:
                cell.set_fontsize(4)

    def _plot_si_ratio_bar(self, ax, df_ratios: pd.DataFrame, color_map: dict):
        """Row 1 (right): isoform ratio bar+strip plot across replicates."""
        if not df_ratios.empty and df_ratios['Ratio'].sum() > 0:
            bar_palette = {}
            for _, row in df_ratios.iterrows():
                bar_palette[row['ShortLabel']] = color_map.get(row['State'], 'gray')

            passed_ratios = df_ratios[df_ratios['GroupOK'] == True]
            failed_ratios = df_ratios[df_ratios['GroupOK'] == False]

            if not passed_ratios.empty:
                sns.barplot(data=passed_ratios, x='ShortLabel', y='Ratio', hue='ShortLabel',
                            palette=bar_palette, ax=ax, errorbar='sd', capsize=0.1, alpha=0.8, dodge=False,
                            legend=False)
                sns.stripplot(data=passed_ratios, x='ShortLabel', y='Ratio', color='black', ax=ax, jitter=True,
                              alpha=0.6, size=3)

            if not failed_ratios.empty:
                sns.barplot(data=failed_ratios, x='ShortLabel', y='Ratio', hue='ShortLabel',
                            palette=bar_palette, ax=ax, errorbar='sd', capsize=0.1, alpha=0.3, dodge=False,
                            legend=False, hatch='//')
                sns.stripplot(data=failed_ratios, x='ShortLabel', y='Ratio', color='grey', ax=ax, jitter=True,
                              alpha=0.4, size=3)

        ax.set_ylabel("Rel. Abundance")
        ax.set_xlabel("")
        mpl.pyplot.setp(ax.get_xticklabels(), visible=False)
        ax.set_title("Isoform Ratios (Selected Peaks)")
        ax.set_ylim(0, 1.1)
        # This panel sits immediately right of the summary table in the grid; a normal left-side
        # y-axis would put its ticks/label right on top of the table's right edge. Mirroring the
        # axis to the right puts them over blank page margin instead, where nothing else is drawn.
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')
        ax.tick_params(axis='y', labelsize=5)

    def _plot_si_chromatograms(self, ax_tic, ax_eic, rep_spectrum, collected_peaks: list, color_map: dict,
                               rt_range: tuple = None):
        """Row 2: overall TIC + EIC-per-isoform, optionally capped to rt_range."""
        tic = rep_spectrum.get_tic(filtered=False)
        if rt_range is not None and not tic.empty:
            tic = tic[(tic.index >= rt_range[0]) & (tic.index <= rt_range[1])]
        if not tic.empty:
            ax_tic.plot(tic.index, tic.values, color='black')

        for peak in collected_peaks:
            state_key = f"{peak.target.sequence} {peak.target.modifications}"
            group_pass = peak.quality_metrics.get('group_pass', False)

            color = color_map.get(state_key, 'grey') if group_pass else 'grey'
            ls = '-' if group_pass else '--'
            lw = 1.0 if group_pass else 0.7
            alpha = 0.8 if group_pass else 0.4

            if hasattr(peak, 'extracted') and peak.extracted is not None:
                extracted = peak.extracted
                if rt_range is not None:
                    extracted = extracted[(extracted.index >= rt_range[0]) & (extracted.index <= rt_range[1])]
                ax_eic.plot(extracted.index, extracted.sum(axis=1), color=color, linestyle=ls, linewidth=lw,
                           alpha=alpha)
                ax_tic.axvspan(peak.raw_apex['time'].min(), peak.raw_apex['time'].max(), alpha=0.1, color=color)

        legend_handles = []
        for state, color in color_map.items():
            short_label = self._get_short_label_from_state(state)
            legend_handles.append(mpl.lines.Line2D([], [], color=color, marker='s', linestyle='None', label=short_label))
        ax_eic.legend(handles=legend_handles, title="Isoforms", fontsize='small', loc='upper right')

        ax_tic.set_title("Chromatograms (Solid=GroupPass, Dashed=Reject), representative TIC in black, EIC in color")
        ax_tic.set_ylabel("Intensity [a.u.]")
        _apply_compact_sci_axis(ax_tic)
        mpl.pyplot.setp(ax_tic.get_xticklabels(), visible=False)

        ax_eic.set_xlabel("Retention Time [min]")
        ax_eic.set_ylabel("Intensity [a.u.]")
        _apply_compact_sci_axis(ax_eic)

        if rt_range is not None:
            ax_tic.set_xlim(rt_range)
            ax_eic.set_xlim(rt_range)

    def _plot_si_mz_overview(self, ax, collected_peaks: list, color_map: dict):
        """Row 3-4: stacked apex isotope pattern across all replicates/samples."""
        unique_samples = sorted(list(set(p.experiment.spectrum_id for p in collected_peaks)))
        offset_y = 0
        for sample in unique_samples:
            sample_peaks = [p for p in collected_peaks if p.experiment.spectrum_id == sample]
            if not sample_peaks:
                continue
            max_int_local = 0
            for peak in sample_peaks:
                state_key = f"{peak.target.sequence} {peak.target.modifications}"
                if peak.quality_metrics.get('group_pass', False):
                    ax.plot(peak.raw_apex['mz_array'], peak.raw_apex['intensity_array'] + offset_y,
                           color=color_map[state_key], alpha=0.7)
                else:
                    ax.plot(peak.raw_apex['mz_array'], peak.raw_apex['intensity_array'] + offset_y,
                           color="grey", alpha=0.4, linestyle='--')
                max_int_local = max(max_int_local, peak.raw_apex['intensity_array'].max())
            offset_y += max_int_local * 1.2
        ax.set_title("Apex Isotope Patterns")
        ax.set_xlabel("m/z [Da]")
        ax.set_ylabel("Intensity [a.u.]")
        _apply_compact_sci_axis(ax)

    def _plot_si_zoomed_ecc(self, ax, isoform_peaks: list, color, state_label: str):
        """Per-state zoomed EIC overlay, one line per replicate."""
        ax.set_title(state_label, fontsize=7, weight='bold')
        ax.set_xlabel("RT [min]")
        ax.set_ylabel("Intensity")
        _apply_compact_sci_axis(ax)
        # This panel sits immediately right of the raw-apex (or MS2) panel in the same row; a
        # left-side y-axis would land its ticks/label on top of that neighboring plot. Right-side
        # axis puts them against the page's right margin instead.
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position('right')

        if not isoform_peaks:
            ax.text(0.5, 0.5, "Not Detected", ha='center', va='center', transform=ax.transAxes)
            return

        for p in isoform_peaks:
            is_sel = p.quality_metrics.get('group_pass', False)
            c = color if is_sel else "grey"
            ls = '-' if is_sel else '--'
            alpha = 0.8 if is_sel else 0.3
            if hasattr(p, 'zoomed_extracted'):
                ax.plot(p.zoomed_extracted.index, p.zoomed_extracted.sum(axis=1), color=c, alpha=alpha, linestyle=ls)

    def _plot_si_isotope_pattern(self, ax, isoform_peaks: list, color, state_label: str):
        """Per-state raw apex isotope pattern, replicates stacked vertically with a legend for the
        theoretical monoisotopic/13C marker lines and a grey baseline hline per replicate. The x
        range is fixed to the theoretical monoisotopic-to-last-isotopologue span (the same span
        used to define the mz_boundaries window when the peak objects were built), split into
        equal thirds -- 1/3 before the monoisotopic peak, 1/3 spanning the isotopologues
        themselves, 1/3 after the last one -- so every state's panel uses the same proportions
        and is directly comparable regardless of what happened to be in the raw data extracted."""
        ax.set_title(state_label, fontsize=7, weight='bold')
        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        _apply_compact_sci_axis(ax)

        if not isoform_peaks:
            return

        theo = getattr(isoform_peaks[0], 'isotopologues', None) or {}
        if theo:
            mono_mz = theo.get("monoisotopic", min(theo.values()))
            last_iso_mz = max(theo.values())
            span = max(last_iso_mz - mono_mz, 1e-6)  # guard a degenerate zero-width span
            ax.set_xlim(mono_mz - span, last_iso_mz + span)

        diff_x = 0
        diff_y = 0
        if len(isoform_peaks) > 1:
            p0 = isoform_peaks[0]
            diff_x = (p0.raw_apex['mz_array'].max() - p0.raw_apex['mz_array'].min()) * 0.05
            diff_y = (p0.raw_apex['intensity_array'].max() - p0.raw_apex['intensity_array'].min()) * 0.1

        for idx, p in enumerate(isoform_peaks):
            is_sel = p.quality_metrics.get('group_pass', False)
            c = color if is_sel else "grey"
            alpha = 0.8 if is_sel else 0.3
            baseline = diff_y * idx
            ax.vlines(p.raw_apex['mz_array'] + (diff_x * idx), baseline, p.raw_apex['intensity_array'] + baseline,
                     color=c, alpha=alpha)
            # Grey baseline hline at the low end of this replicate's stacked vlines, to better
            # demarcate where its own baseline sits within the vertically-offset stack.
            ax.axhline(baseline, color='grey', alpha=0.4, linewidth=0.5, zorder=0)

        if theo:
            legend_handles = []
            seen_labels = set()
            for name, mz in theo.items():
                is_mono = name == "monoisotopic"
                ls = '--' if is_mono else ':'
                c_line = 'green' if is_mono else 'red'
                ax.axvline(mz, color=c_line, linestyle=ls, alpha=0.5)

                legend_label = "Monoisotopic" if is_mono else "Isotopologue (+13C)"
                if legend_label not in seen_labels:
                    legend_handles.append(mpl.lines.Line2D([], [], color=c_line, linestyle=ls, label=legend_label))
                    seen_labels.add(legend_label)
            if legend_handles:
                ax.legend(handles=legend_handles, fontsize=5, loc='upper right')

    def _compute_ms2_matches(self, isoform_peaks: list):
        """Combines MS2 fragment matches across all of a state's replicates into best-match b/y
        ion tables (isotopologue-independent -- computed once per state, not per isotopologue)."""
        all_ms2_dfs = []
        for p in isoform_peaks:
            if hasattr(p, 'ms2_matches') and p.ms2_matches is not None:
                df_ms2 = p.ms2_matches.copy()
                if not df_ms2.empty:
                    if 'error_ppm' not in df_ms2.columns:
                        df_ms2['error_ppm'] = 0.0
                    all_ms2_dfs.append(df_ms2)

        if not all_ms2_dfs:
            return None, None

        combined_df = pd.concat(all_ms2_dfs, ignore_index=True)
        combined_df['abs_error'] = combined_df['error_ppm'].abs()
        counts = combined_df.groupby('fragment').size().reset_index(name='n_replicates')
        best_matches = combined_df.sort_values(['abs_error', 'intensity'],
                                               ascending=[True, False]).drop_duplicates(subset=['fragment'])
        combined_ms2 = pd.merge(best_matches, counts, on='fragment')
        b_ions = combined_ms2[combined_ms2['type'] == 'b'].dropna(subset=['observed_mz'])
        y_ions = combined_ms2[combined_ms2['type'] == 'y'].dropna(subset=['observed_mz'])
        return b_ions, y_ions

    def _plot_si_ms2_spectrum(self, ax, b_ions, y_ions):
        """Per-state raw MS2 b/y ion match plot."""
        if b_ions is None or y_ions is None or (b_ions.empty and y_ions.empty):
            ax.axis('off')
            return

        ax.vlines(b_ions['observed_mz'], ymin=0, ymax=b_ions['intensity'], color='blue', label='b ions')
        ax.vlines(y_ions['observed_mz'], ymin=0, ymax=y_ions['intensity'], color='red', label='y ions')
        for _, row in b_ions.iterrows():
            ax.text(row['observed_mz'], row['intensity'], f"b{row['fragment'][1:]}", color='blue',
                   va='bottom', ha='center', fontsize=6)
        for _, row in y_ions.iterrows():
            ax.text(row['observed_mz'], row['intensity'], f"y{row['fragment'][1:]}", color='red',
                   va='bottom', ha='center', fontsize=6)
        ax.legend()
        ax.set_title("MS2")
        ax.set_xlabel("m/z for fragments")
        ax.set_ylabel("Intensity")
        _apply_compact_sci_axis(ax)

    def _plot_si_sequence_annotations(self, ax, rep_peak, b_ions, y_ions, state_label: str):
        """Per-state peptide sequence plot with b/y ion match annotations spread across two text
        lines (fragment+replicate count on the first line, ppm error on the second)."""
        ax.axis('off')
        #ax.set_title(state_label, fontsize=7, weight='bold')

        if not (rep_peak and hasattr(rep_peak.target, 'sequence')):
            return

        parsed_modx_seq = pyteomics.parser.parse(rep_peak.target.sequence)
        seq_length = len(parsed_modx_seq)
        dynamic_fs = 12 if seq_length < 20 else 10 if seq_length < 40 else 8
        x_positions_peptide = np.linspace(0.1, 0.9, seq_length)
        x_pos_diff = x_positions_peptide[1] - x_positions_peptide[0] if seq_length > 1 else 0

        y_position_peptide = 0.5
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        text_y_offset = 0.19
        line_y_start_offset = 0.12
        line_y_end_offset = 0.17

        for j, aa in enumerate(parsed_modx_seq):
            ax.text(x_positions_peptide[j], y_position_peptide, aa, fontsize=dynamic_fs, ha='center', va='center')

        if b_ions is not None and not b_ions.empty:
            for _, row in b_ions.iterrows():
                # bN covers the first N residues, so its cleavage site (and label anchor) sits
                # AT residue N -- the last residue that's actually part of the fragment.
                frag_pos = int(row['fragment'][1:])
                if 0 < frag_pos <= len(x_positions_peptide):
                    x_pos_current = x_positions_peptide[frag_pos - 1]
                    # clip_on=False: these are small rotated 2-line labels near the panel edges,
                    # and clipping them at the axes boundary was cutting them off mid-label.
                    ax.text(float(x_pos_current), y_position_peptide + text_y_offset,
                           f"{row['fragment']} (n={row['n_replicates']})\nppm={round(row.get('error_ppm', 0), 2)}",
                           color='blue', fontsize=5, ha='left', va='bottom', rotation=45, clip_on=False)
                    if frag_pos < len(x_positions_peptide):
                        lines_x = [x_pos_current + x_pos_diff / 2, x_pos_current + x_pos_diff / 2, x_pos_current]
                        lines_y = [y_position_peptide, y_position_peptide + line_y_start_offset,
                                  y_position_peptide + line_y_end_offset]
                        ax.plot(lines_x, lines_y, color='blue', linewidth=1)

        if y_ions is not None and not y_ions.empty:
            for _, row in y_ions.iterrows():
                # yM covers the LAST M residues, i.e. residues (L-M+1)..L. Its cleavage site (and
                # label anchor) sits AT residue (L-M+1) -- the FIRST residue that's actually part
                # of the fragment (the +1 was previously missing, which anchored the label one
                # residue too far left -- e.g. a real y3 rendered anchored where a y4 belongs,
                # reading as "should be y3" one position over from where "y2" was drawn).
                frag_pos = seq_length - int(row['fragment'][1:]) + 1
                if 0 < frag_pos <= len(x_positions_peptide):
                    x_pos_current = x_positions_peptide[frag_pos - 1]
                    ax.text(float(x_pos_current), y_position_peptide - text_y_offset,
                           f"{row['fragment']} (n={row['n_replicates']})\nppm={round(row.get('error_ppm', 0), 2)}",
                           color='red', fontsize=5, ha='right', va='top', rotation=45, clip_on=False)
                    if frag_pos > 1:
                        lines_x = [x_pos_current - x_pos_diff / 2, x_pos_current - x_pos_diff / 2, x_pos_current]
                        lines_y = [y_position_peptide, y_position_peptide - line_y_start_offset,
                                  y_position_peptide - line_y_end_offset]
                        ax.plot(lines_x, lines_y, color='red', linewidth=1)

    def generate_all_reports(self, output_dir: Path, plot_mz: bool = True, plot_ms2: bool = False):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Identify Scope
        report_scope = set()
        for res in self.analysis_results:
            for t, peaks in res.found_peaks.items():
                for p in peaks:
                    report_scope.add((t.fragment_cut, p.charge))

        print(f"Generating reports for {len(report_scope)} Cut/Charge groups...")

        all_plotting_data = []
        all_rt_grouped = []

        for cut_id, charge in report_scope:
            safe_cut = str(cut_id).replace(" ", "_").replace("/", "-")
            filename = f"Report_Cut_{safe_cut}_z{charge}.png"

            target_dict = self._get_theoretical_targets_dict(cut_id)
            all_possible_states = sorted(list(target_dict.keys()))

            all_collected_peaks = []  # This will hold peaks that pass the group filter
            final_peaks_map = defaultdict(dict)  # Mapping for ratio calculation
            all_rt_groups_metadata = []  # For the pickle

            # --- A. GROUPING AND VALIDATION ---
            for state_key in all_possible_states:
                target_obj = target_dict[state_key]

                exp_selected_map = {}
                exp_found_map = {}
                rts_for_clustering = []

                for res in self.analysis_results:
                    exp_id = res.experiment.spectrum_id

                    # FIX: Iterate over the PEAK objects (the values), not the TARGET objects (the keys).
                    # The values are lists of peaks, so we need to flatten them.
                    sel = [peak for peak_list in res.selected_peaks.values() for peak in peak_list
                           if f"{peak.target.sequence} {peak.target.modifications}" == state_key
                           and peak.charge == charge]
                    fnd = [peak for peak_list in res.found_peaks.values() for peak in peak_list
                           if f"{peak.target.sequence} {peak.target.modifications}" == state_key
                           and peak.charge == charge]

                    exp_selected_map[exp_id] = sel
                    exp_found_map[exp_id] = fnd

                    # Use selected peaks for clustering; if none, use found peaks as fallback
                    peaks_to_cluster = sel if sel else fnd
                    # FIX: Access the .rt attribute directly from the PeptidePeak object.
                    rts_for_clustering.extend([p.rt for p in peaks_to_cluster])

                # 1. Fuzzy Grouping (DBSCAN)
                rt_tolerance = 0.10
                clusters = []
                if rts_for_clustering:
                    rts_array = np.array(rts_for_clustering).reshape(-1, 1)
                    # min_samples=1 ensures that every point belongs to a cluster (no noise points)
                    db = DBSCAN(eps=rt_tolerance, min_samples=1).fit(rts_array)
                    labels = db.labels_

                    # Group RTs by their cluster label
                    num_clusters = len(set(labels))
                    clusters = [[] for _ in range(num_clusters)]
                    for rt, label in zip(rts_for_clustering, labels):
                        clusters[label].append(rt)

                # 2. Build Potential Groups
                state_rt_groups = []
                for cluster in clusters:
                    consensus_rt = np.mean(cluster)
                    group_entries = []
                    represented_exps = set()

                    # Add high-confidence selected peaks
                    for exp_id, sel_list in exp_selected_map.items():
                        for p in sel_list:
                            if abs(p.rt - consensus_rt) <= rt_tolerance:
                                group_entries.append({'peak': p, 'is_selected': True})
                                represented_exps.add(exp_id)

                    # Backfill from found_peaks for missing replicates
                    for exp_id, fnd_list in exp_found_map.items():
                        if exp_id not in represented_exps:
                            candidates = [p for p in fnd_list if abs(p.rt - consensus_rt) <= rt_tolerance]
                            if candidates:
                                # Fallback quality metric: Maximize Area
                                best_match = max(candidates,
                                                 key=lambda x: x.quality_metrics.get('final_trapz_area', 0) or 0)
                                group_entries.append({'peak': best_match, 'is_selected': False})

                    if not group_entries:
                        continue

                    sel_count = sum(1 for e in group_entries if e['is_selected'])
                    area_sel = sum(e['peak'].quality_metrics.get('final_trapz_area', 0) or 0 for e in group_entries if
                                   e['is_selected'])
                    area_all = sum(e['peak'].quality_metrics.get('final_trapz_area', 0) or 0 for e in group_entries)

                    qc_passing_peaks = [e['peak'] for e in group_entries if
                                        e['peak'].quality_metrics.get('overall_pass', False)]
                    qc_pass_count = len(qc_passing_peaks)
                    # FIX: Safely calculate mean, defaulting to 0.0 if no scored peaks are found to prevent NaN.
                    avg_score = np.mean([p.quality_score for p in qc_passing_peaks if
                                         hasattr(p, 'quality_score')]) if qc_passing_peaks else 0.0

                    state_rt_groups.append({
                        'state_key': state_key,
                        'consensus_rt': consensus_rt,
                        'peak_entries': group_entries,
                        'selected_peak_count': sel_count,
                        'qc_pass_count': qc_pass_count,
                        'avg_score': avg_score,
                        'total_area_selected': area_sel,
                        'total_area_all': area_all
                    })

                # 3. Sorting and the 30% Area Rule
                if state_rt_groups:
                    # Sort groups by quality: prioritize average_score then qc_pass_count, then total_area_all
                    state_rt_groups.sort(key=lambda x: (x['avg_score'], x['qc_pass_count'], x['total_area_all']),
                                         reverse=True)
                    # --- New Composite Score Calculation ---
                    for group in state_rt_groups:
                        qc_pass_count = group['qc_pass_count']
                        avg_score = group['avg_score']
                        # Use a small epsilon to prevent log10(0) or log10(1) issues
                        total_area = group['total_area_all'] + 1.0

                        # The area component is log-transformed to moderate its influence
                        # while still heavily rewarding larger peaks.
                        area_component = np.log10(total_area)

                        # The composite score prioritizes groups with more high-quality replicates (qc_pass_count)
                        # and large, confident signals (area_component), refined by the average peak quality (avg_score).
                        # We add 1 to qc_pass_count to avoid multiplying by zero for groups with no QC-passing peaks.
                        group['composite_score'] = (qc_pass_count + 1) * avg_score * area_component

                    # Sort groups by the new composite score, which provides a more balanced assessment of group quality.
                    state_rt_groups.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
                    winning_groups = []
                    primary_group = state_rt_groups[0]
                    primary_area = primary_group['total_area_all']

                    # The primary group is a "winner" and its pass status is determined by a quality gate.
                    # A slightly more lenient gate for the primary group to ensure high-area peaks are not lost.
                    primary_group['group_pass'] = (primary_group['qc_pass_count'] >= 1) and (
                                primary_group['avg_score'] >= 0.5)
                    winning_groups.append(primary_group)

                    # Evaluate secondary groups.
                    for group in state_rt_groups[1:]:
                        # A secondary group must have good quality AND significant area to be considered.
                        is_good_quality = (group['qc_pass_count'] >= 1) and (group['avg_score'] >= 0.6)
                        is_significant_area = group['total_area_all'] >= 0.3 * primary_area
                        if is_good_quality and is_significant_area:
                            group['group_pass'] = True  # If it's a winning group, it's a passing group.
                            winning_groups.append(group)

                    # Now that all winning groups are selected, add their peaks to the final lists
                    for group in winning_groups:
                        for entry in group['peak_entries']:
                            p = entry['peak']
                            is_sel = entry['is_selected']

                            # Store all relevant info for plotting and the table
                            p.quality_metrics['is_selected'] = is_sel
                            p.quality_metrics['group_pass'] = group.get('group_pass', False)

                            # To keep your existing plotting code compatible, we ensure 'p.target' exists
                            if not hasattr(p, 'target'):
                                p.target = p

                            all_collected_peaks.append(p)
                            exp_id = p.experiment.spectrum_id
                            final_peaks_map[exp_id][state_key] = (p, is_sel, True)  # set true here from is_selected

                    all_rt_grouped.extend(winning_groups)

            # --- B. REBUILD MAPS & RATIOS ---
            df_ratios = self._get_ratios_dataframe(final_peaks_map, all_possible_states)

            # --- C. PLOT ---
            plotting_data = {
                "collected_peaks": all_collected_peaks,
                "peaks_map": final_peaks_map,
                "target_dict": target_dict,
                "all_states": all_possible_states,
                "df_ratios": df_ratios,
                "output_path": output_dir / filename,
                "plot_mz": plot_mz,
                "plot_ms2": plot_ms2,
                "cut_id": cut_id,
                "charge": charge,
                "rt_groups_metadata": all_rt_groups_metadata  # Added for manual reassignment later
            }
            print(f"All plotting data, {plotting_data}")
            all_plotting_data.append(plotting_data)
            self._create_single_figure(**plotting_data)

        self.all_plotting_data = all_plotting_data
        self.all_rt_grouped = all_rt_grouped

        with open(output_dir / "plotting_data.pkl", "wb") as fp:
            pkl.dump(all_plotting_data, fp)

    def _get_short_label_from_state(self, state_string: str) -> str:
        """Extracts the modification part from a state string for a cleaner label, e.g.
        "SEQUENCE {'Unmodified': True}" -> "Unmodified", "SEQUENCE {'spliced': 2}" -> "Spliced 2"."""
        try:
            mod_part = state_string.split(' ', 1)[1]
        except IndexError:
            # If the split fails for any reason, fall back to the original string.
            return state_string
        try:
            mod_dict = ast.literal_eval(mod_part)
            if isinstance(mod_dict, dict):
                return format_modification_label(mod_dict)
        except (ValueError, SyntaxError):
            pass
        return mod_part

    def _get_ratios_dataframe(self, peaks_map: dict, all_possible_states: list[str]):
        data = []
        all_exp_ids = [res.experiment.spectrum_id for res in self.analysis_results]

        for exp_id in all_exp_ids:
            total_area = 0
            state_areas = {state: 0.0 for state in all_possible_states}

            if exp_id in peaks_map:
                # First, calculate total area of ALL identified peaks for this experiment
                # to ensure ratios sum to 1.
                for state, (peak, is_sel, group_pass) in peaks_map[exp_id].items():
                    if peak and group_pass:
                        area = peak.quality_metrics.get('final_trapz_area', 0)
                        if isinstance(area, (int, float)) and not np.isnan(area):
                            total_area += area

                # Now, calculate ratio for each state, regardless of pass/fail status
                for state, (peak, is_sel, group_pass) in peaks_map[exp_id].items():
                    if peak and group_pass:
                        area = peak.quality_metrics.get('final_trapz_area', 0) or 0
                        ratio = (area / total_area) if total_area > 0 else 0.0
                        short_label = self._get_short_label_from_state(state)
                        data.append({
                            'Experiment': exp_id,
                            'State': state,
                            'ShortLabel': short_label,
                            'Ratio': ratio,

                            'GroupOK': group_pass
                        })
        return pd.DataFrame(data)

    def _create_single_figure(self, collected_peaks: list, peaks_map: dict, target_dict: dict,
                              all_states: list[str],
                              df_ratios: pd.DataFrame,
                              output_path: Path, plot_mz: bool, plot_ms2: bool, cut_id, charge,
                              rt_groups_metadata: dict = None, stub_entries: list = None, dpi: int = 100):

        # --- FIX for AttributeError and Enhanced Title Info ---
        rep_exp = self.analysis_results[0].experiment

        # Safely get attributes for the title from the experiment object
        enzyme_name = getattr(rep_exp, 'maturase_name', 'N/A')
        pep_plasmid = getattr(rep_exp, 'precursor_plasmid', 'N/A')
        exp_scale = getattr(rep_exp, 'expression_scale', 'N/A')
        exp_temp = getattr(rep_exp, 'expression_temp', 'N/A')
        exp_time = getattr(rep_exp, 'expression_time', 'N/A')

        super_title = (
            f"Precursor: {pep_plasmid} | Enzyme: {enzyme_name} | Cut: {cut_id} (z={charge})\n"
            f"Expression: Scale={exp_scale}, Temp={exp_temp}°C, Time={exp_time}h"
        )

        # Load spectrum object to get the TIC, which was causing an AttributeError
        rep_spectrum = MassSpectrum(mzml_path=rep_exp.mzml_path, output_path=rep_exp.output_path,
                                    spectrum_id=rep_exp.spectrum_id)
        rep_spectrum.load()  # This will use the cache if available
        # --- END FIX ---

        palette = mpl.pyplot.cm.tab10.colors
        while len(palette) < len(all_states):
            palette = palette + palette
        color_map = {state: palette[i] for i, state in enumerate(all_states)}

        num_states = len(all_states)
        base_rows = 5 if plot_mz else 3
        total_rows = base_rows + (num_states * 2)

        # Use constrained_layout for more robust automatic spacing
        fig = mpl.pyplot.figure(figsize=(9, 15 * total_rows / 8), constrained_layout=True)
        hr = [1.5, 2.0, 0.2] + [1] * (total_rows - 3)
        gs = mpl.gridspec.GridSpec(total_rows, 8, figure=fig, height_ratios=hr)

        fig.suptitle(super_title, fontsize=12, y=1.0)
        sns.set_style("white")
        mpl.pyplot.rcParams.update({'font.size': 6})

        # --- 1. Table ---
        ax_table = fig.add_subplot(gs[0, :6])
        ax_table.axis('off')

        table_peaks = [p for p in collected_peaks]

        if table_peaks:
            table_data = []
            for p in table_peaks:
                # Standard QC metrics
                qc_val = p.quality_metrics.get('overall_pass', False)
                qc_str = "PASS" if qc_val else "FAIL"
                qc_group = "PASS" if p.quality_metrics.get('group_pass',
                                                           False) else "FAIL"  # This is the GroupOK column

                # New metrics for the expanded table
                is_selected = "Yes" if p.quality_metrics.get('is_selected', False) else "No"
                score = getattr(p, 'quality_score', 'N/A')
                tier = getattr(p, 'tier', 'N/A')
                score_str = f"{score:.2f}" if isinstance(score, float) else str(score)

                table_data.append([
                    p.experiment.spectrum_id,
                    p.target.sequence,
                    str(p.target.modifications),
                    f"{p.rt:.2f}",
                    f"{p.quality_metrics.get('final_trapz_area') or 0:.2e}",
                    f"{p.quality_metrics.get('median_error_ppm') or 0:.2f}",
                    qc_str,
                    qc_group,
                    is_selected,
                    score_str,
                    tier
                ])
        elif not stub_entries:
            table_data = [["No Selected Peaks", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]]
        else:
            table_data = []

        # Stub replicates (no curated selection for this state) are noted in the table for
        # transparency, but never contribute a plotted line -- they're not in collected_peaks.
        for exp_id, state_key in (stub_entries or []):
            seq, _, mod_str = state_key.partition(" ")
            table_data.append([exp_id, seq, mod_str, "-", "-", "-", "STUB", "STUB", "No", "-", "Stub"])

        col_labels = ["Exp", "Seq", "Mod", "RT", "Area", "Err", "QC", "GroupOK", "Selected", "Score", "Tier"]
        col_widths = [0.15, 0.18, 0.13, 0.06, 0.08, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06]
        table = ax_table.table(cellText=table_data, colLabels=col_labels, colWidths=col_widths, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(5)
        table.scale(1, 1.2)

        # --- 2. Bar Plot ---
        ax_bar = fig.add_subplot(gs[0, 6:])
        if not df_ratios.empty and df_ratios['Ratio'].sum() > 0:

            bar_palette = {}
            for _, row in df_ratios.iterrows():
                label = row['ShortLabel']
                state = row['State']
                # Use .get() to provide a fallback color (gray) if the state isn't in your map
                bar_palette[label] = color_map.get(state, 'gray')

            # Separate passed and failed data for distinct plotting
            passed_ratios = df_ratios[df_ratios['GroupOK'] == True]
            failed_ratios = df_ratios[df_ratios['GroupOK'] == False]

            # Plot passed bars (solid)
            if not passed_ratios.empty:
                sns.barplot(data=passed_ratios, x='ShortLabel', y='Ratio', hue='ShortLabel',
                            palette=bar_palette, ax=ax_bar, errorbar='sd', capsize=0.1, alpha=0.8, dodge=False,
                            legend=False)
                sns.stripplot(data=passed_ratios, x='ShortLabel', y='Ratio', color='black', ax=ax_bar, jitter=True,
                              alpha=0.6, size=3)

            # Plot failed bars (hatched and transparent)
            if not failed_ratios.empty:
                sns.barplot(data=failed_ratios, x='ShortLabel', y='Ratio', hue='ShortLabel',
                            palette=bar_palette, ax=ax_bar, errorbar='sd', capsize=0.1, alpha=0.3, dodge=False,
                            legend=False, hatch='//')
                sns.stripplot(data=failed_ratios, x='ShortLabel', y='Ratio', color='grey', ax=ax_bar, jitter=True,
                              alpha=0.4, size=3)

        ax_bar.set_ylabel("Rel. Abundance")
        # Hide x-axis labels as requested
        ax_bar.set_xlabel("")
        mpl.pyplot.setp(ax_bar.get_xticklabels(), visible=False)
        ax_bar.set_title("Isoform Ratios (Selected Peaks)")
        ax_bar.set_ylim(0, 1.1)

        # --- 3. Chromatograms ---
        gs_chrom = mpl.gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], hspace=0.0, height_ratios=[1, 1])
        ax_tic = fig.add_subplot(gs_chrom[0, :])

        # Use the loaded spectrum to get the TIC
        tic = rep_spectrum.get_tic(filtered=False)
        if not tic.empty:
            ax_tic.plot(tic.index, tic.values, color='black')

        ax_eic = fig.add_subplot(gs_chrom[1, :], sharex=ax_tic)

        # Create formatter for scientific notation on intensity axes
        sci_formatter = mpl.ticker.ScalarFormatter(useMathText=True)
        sci_formatter.set_scientific(True)
        sci_formatter.set_powerlimits((-2, 3))

        # Plot EICs for all peaks in the table, colored by their group status
        for peak in collected_peaks:
            state_key = f"{peak.target.sequence} {peak.target.modifications}"
            group_pass = peak.quality_metrics.get('group_pass', False)

            color = color_map.get(state_key, 'grey') if group_pass else 'grey'
            ls = '-' if group_pass else '--'
            lw = 1.0 if group_pass else 0.7
            alpha = 0.8 if group_pass else 0.4

            if hasattr(peak, 'extracted') and peak.extracted is not None:
                ax_eic.plot(peak.extracted.index, peak.extracted.sum(axis=1),
                            color=color, linestyle=ls, linewidth=lw, alpha=alpha)
                ax_tic.axvspan(peak.raw_apex['time'].min(), peak.raw_apex['time'].max(), alpha=0.1, color=color)

        # Add legend to EIC plot as requested
        legend_handles = []
        for state, color in color_map.items():
            short_label = self._get_short_label_from_state(state)
            legend_handles.append(mpl.lines.Line2D([], [], color=color, marker='s', linestyle='None', label=short_label))
        ax_eic.legend(handles=legend_handles, title="Isoforms", fontsize='small', loc='upper right')

        ax_tic.set_title("Chromatograms (Solid=GroupPass, Dashed=Reject)")
        ax_tic.set_ylabel("Intensity [a.u.]")
        ax_tic.yaxis.set_major_formatter(sci_formatter)
        mpl.pyplot.setp(ax_tic.get_xticklabels(), visible=False)

        ax_eic.set_xlabel("Retention Time [min]")
        ax_eic.set_ylabel("Intensity [a.u.]")
        ax_eic.yaxis.set_major_formatter(sci_formatter)

        # --- 4. MZ Overview ---
        current_row = 3
        if plot_mz:
            ax_mz = fig.add_subplot(gs[current_row:current_row + 2, :])
            current_row += 2
            offset_y = 0

            # FIXED: We want to plot all peaks, not just passed ones, so fails render in grey
            valid_peaks = collected_peaks

            unique_samples = sorted(list(set(p.experiment.spectrum_id for p in valid_peaks)))
            for sample in unique_samples:
                sample_peaks = [p for p in valid_peaks if p.experiment.spectrum_id == sample]
                if not sample_peaks: continue
                max_int_local = 0
                for peak in sample_peaks:
                    state_key = f"{peak.target.sequence} {peak.target.modifications}"
                    if peak.quality_metrics.get('group_pass', False):
                        ax_mz.plot(peak.raw_apex['mz_array'], peak.raw_apex['intensity_array'] + offset_y,
                                   color=color_map[state_key], alpha=0.7)
                    else:
                        ax_mz.plot(peak.raw_apex['mz_array'], peak.raw_apex['intensity_array'] + offset_y,
                                   color="grey", alpha=0.4, linestyle='--')
                    max_int_local = max(max_int_local, peak.raw_apex['intensity_array'].max())
                offset_y += max_int_local * 1.2
            ax_mz.set_title("Apex Isotope Patterns")
            ax_mz.set_xlabel("m/z [Da]")
            ax_mz.set_ylabel("Intensity [a.u.]")
            ax_mz.yaxis.set_major_formatter(sci_formatter)

            # --- 5. Detailed Loop ---
            for i, state in enumerate(all_states):
                row_idx = current_row + (i * 2)
                isoform_peaks = [p for p in collected_peaks if f"{p.target.sequence} {p.target.modifications}" == state]
                isoform_peaks.sort(key=lambda x: getattr(x, 'trapz_area', 0) or 0, reverse=True)

                ax_peak_ecc = fig.add_subplot(gs[row_idx:row_idx + 1, 4:])
                ax_raw_apex = fig.add_subplot(gs[row_idx:row_idx + 2, :4])
                if plot_ms2:
                    ax_raw_apex = fig.add_subplot(gs[row_idx:row_idx + 2, :2])
                    ax_ms2 = fig.add_subplot(gs[row_idx:row_idx + 2, 2:4])

                # Set axis labels for detailed plots
                ax_peak_ecc.set_xlabel("RT [min]")
                ax_peak_ecc.set_ylabel("Intensity")
                ax_peak_ecc.yaxis.set_major_formatter(sci_formatter)
                ax_raw_apex.set_xlabel("m/z")
                ax_raw_apex.set_ylabel("Intensity")
                ax_raw_apex.yaxis.set_major_formatter(sci_formatter)

                ax_seq = fig.add_subplot(gs[row_idx + 1:row_idx + 2, 4:])
                ax_seq.axis('off')
                color = color_map[state]

                if not isoform_peaks:
                    ax_peak_ecc.text(0.5, 0.5, "Not Detected", ha='center', va='center',
                                     transform=ax_peak_ecc.transAxes)
                    continue

                # STRICT FIX: Safely define rep_peak for the MS2 / Sequence plots to use.
                # We prioritize the highest area peak that actually passed QC.
                passed_isoform_peaks = [p for p in isoform_peaks if p.quality_metrics.get('overall_pass', False)]
                rep_peak = passed_isoform_peaks[0] if passed_isoform_peaks else isoform_peaks[0]

                for idx, p in enumerate(isoform_peaks):
                    is_sel = p.quality_metrics.get('group_pass', False)
                    c = color if is_sel else "grey"
                    ls = '-' if is_sel else '--'
                    alpha = 0.8 if is_sel else 0.3

                    if hasattr(p, 'zoomed_extracted'):
                        ax_peak_ecc.plot(p.zoomed_extracted.index, p.zoomed_extracted.sum(axis=1),
                                         color=c, alpha=alpha, linestyle=ls)

                diff_x = 0;
                diff_y = 0
                if len(isoform_peaks) > 1:
                    p0 = isoform_peaks[0]
                    diff_x = (p0.raw_apex['mz_array'].max() - p0.raw_apex['mz_array'].min()) * 0.05
                    diff_y = (p0.raw_apex['intensity_array'].max() - p0.raw_apex['intensity_array'].min()) * 0.1

                for idx, p in enumerate(isoform_peaks):
                    is_sel = p.quality_metrics.get('group_pass', False)
                    c = color if is_sel else "grey"
                    alpha = 0.8 if is_sel else 0.3
                    ax_raw_apex.vlines(p.raw_apex['mz_array'] + (diff_x * idx), diff_y * idx,
                                       p.raw_apex['intensity_array'] + (diff_y * idx),
                                       color=c, alpha=alpha)

                if hasattr(isoform_peaks[0], 'isotopologues'):
                    theo = isoform_peaks[0].isotopologues
                    for name, mz in theo.items():
                        ls = '--' if name == "monoisotopic" else ':'
                        c_line = 'green' if name == "monoisotopic" else 'red'
                        ax_raw_apex.axvline(mz, color=c_line, linestyle=ls, alpha=0.5)

                        # --- MS2 ---
                        # Combine MS2 matches across all replicates for this isoform
                        all_ms2_dfs = []
                        for p in isoform_peaks:
                            if hasattr(p, 'ms2_matches') and p.ms2_matches is not None:
                                df_ms2 = p.ms2_matches.copy()
                                if not df_ms2.empty:
                                    if 'error_ppm' not in df_ms2.columns:
                                        df_ms2['error_ppm'] = 0.0
                                    all_ms2_dfs.append(df_ms2)

                        b_ions, y_ions = None, None
                        if all_ms2_dfs:
                            combined_df = pd.concat(all_ms2_dfs, ignore_index=True)
                            combined_df['abs_error'] = combined_df['error_ppm'].abs()

                            # Count replicates per fragment
                            counts = combined_df.groupby('fragment').size().reset_index(name='n_replicates')

                            # Get best match per fragment (lowest absolute error)
                            best_matches = combined_df.sort_values(['abs_error', 'intensity'],
                                                                   ascending=[True, False]).drop_duplicates(
                                subset=['fragment'])

                            combined_ms2 = pd.merge(best_matches, counts, on='fragment')
                            b_ions = combined_ms2[combined_ms2['type'] == 'b'].dropna(subset=['observed_mz'])
                            y_ions = combined_ms2[combined_ms2['type'] == 'y'].dropna(subset=['observed_mz'])

                        if state and plot_ms2 and b_ions is not None and y_ions is not None:
                            ax_ms2.vlines(b_ions['observed_mz'], ymin=0, ymax=b_ions['intensity'], color='blue',
                                          label='b ions')
                            ax_ms2.vlines(y_ions['observed_mz'], ymin=0, ymax=y_ions['intensity'], color='red',
                                          label='y ions')
                            for _, row in b_ions.iterrows():
                                ax_ms2.text(row['observed_mz'], row['intensity'], f"b{row['fragment'][1:]}",
                                            color='blue',
                                            va='bottom', ha='center', fontsize=6)
                            for _, row in y_ions.iterrows():
                                ax_ms2.text(row['observed_mz'], row['intensity'], f"y{row['fragment'][1:]}",
                                            color='red', va='bottom', ha='center', fontsize=6)
                            ax_ms2.legend()
                            ax_ms2.set_title(f"MS2")
                            ax_ms2.set_xlabel("m/z for fragments")
                            ax_ms2.set_ylabel("Intensity")
                            ax_ms2.yaxis.set_major_formatter(sci_formatter)

                        # Sequence Plot
                        ax_seq = fig.add_subplot(gs[row_idx + 1:row_idx + 2, 4:])
                        ax_seq.axis('off')
                        if rep_peak and hasattr(rep_peak.target, 'sequence'):
                            parsed_modx_seq = pyteomics.parser.parse(rep_peak.target.sequence)
                            seq_length = len(parsed_modx_seq)
                            # 2. Dynamically scale font size for long sequences
                            dynamic_fs = 12 if seq_length < 20 else 10 if seq_length < 40 else 8
                            label_fs = 6 if seq_length < 30 else 4
                            x_positions_peptide = np.linspace(0.1, 0.9, seq_length)
                            x_pos_diff = x_positions_peptide[1] - x_positions_peptide[0]

                            # --- UNIFIED SCALING ---
                            # Use fixed vertical coordinates to ensure all sequence plots are uniform.
                            # This prevents short sequences from creating overly large plots that spill their boundaries.
                            y_position_peptide = 0.5
                            ax_seq.set_xlim(0, 1)
                            ax_seq.set_ylim(0, 1)  # Set a fixed, uniform y-axis range.

                            # Define fixed offsets for text and lines relative to the new y-axis range.
                            text_y_offset = 0.19
                            line_y_start_offset = 0.12
                            line_y_end_offset = 0.17
                            # --- END UNIFIED SCALING ---

                            for j, aa in enumerate(parsed_modx_seq):
                                ax_seq.text(x_positions_peptide[j], y_position_peptide, aa, fontsize=dynamic_fs,
                                            ha='center',
                                            va='center')

                            # Plot b-ion matches
                            if b_ions is not None and not b_ions.empty:
                                for _, row in b_ions.iterrows():
                                    frag_pos = int(row['fragment'][1:])
                                    if 0 < frag_pos <= len(x_positions_peptide):
                                        x_pos_current = x_positions_peptide[frag_pos - 1]
                                        # Position text at a fixed offset from the center.
                                        ax_seq.text(float(x_pos_current), y_position_peptide + text_y_offset,
                                                    f"{row['fragment']} (n={row['n_replicates']}) ppm:{round(row.get('error_ppm', 0), 2)}",
                                                    color='blue', fontsize=5,
                                                    ha='left', va='bottom', rotation=45, clip_on=True)
                                        if frag_pos < len(x_positions_peptide):
                                            # Draw lines using fixed offsets.
                                            lines_x = [x_pos_current + x_pos_diff / 2, x_pos_current + x_pos_diff / 2,
                                                       x_pos_current]
                                            lines_y = [y_position_peptide, y_position_peptide + line_y_start_offset,
                                                       y_position_peptide + line_y_end_offset]
                                            ax_seq.plot(lines_x, lines_y, color='blue', linewidth=1)

                            # Plot y-ion matches
                            if y_ions is not None and not y_ions.empty:
                                for _, row in y_ions.iterrows():
                                    frag_pos = seq_length - int(row['fragment'][1:])
                                    if 0 < frag_pos <= len(x_positions_peptide):
                                        x_pos_current = x_positions_peptide[frag_pos - 1]
                                        # Position text at a fixed offset from the center.
                                        ax_seq.text(float(x_pos_current), y_position_peptide - text_y_offset,
                                                    f"{row['fragment']} (n={row['n_replicates']}) ppm:{round(row.get('error_ppm', 0), 2)}",
                                                    color='red', fontsize=5,
                                                    ha='right', va='top', rotation=45, clip_on=True)
                                        if frag_pos < len(x_positions_peptide):
                                            # Draw lines using fixed offsets.
                                            lines_x = [x_pos_current - x_pos_diff / 2, x_pos_current - x_pos_diff / 2,
                                                       x_pos_current]
                                            lines_y = [y_position_peptide, y_position_peptide - line_y_start_offset,
                                                       y_position_peptide - line_y_end_offset]
                                            ax_seq.plot(lines_x, lines_y, color='red', linewidth=1)
        # plt.tight_layout() is replaced by constrained_layout=True in figure creation for better stability
        mpl.pyplot.tight_layout()
        mpl.pyplot.savefig(output_path, dpi=dpi, bbox_inches='tight')
        mpl.pyplot.close(fig)
