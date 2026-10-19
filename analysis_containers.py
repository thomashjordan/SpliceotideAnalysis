"""Result containers, aggregation and SI-figure rendering.

* AnalysisResults       -- all candidate (found_peaks) and curated (selected_peaks) peaks for one
                           replicate, plus target expansion from the sample metadata.
* ReplicateGroupResults -- the replicates of one experimental group; builds the modification-
                           state matrix (relative areas / modification ratios across replicates)
                           and is what peak_reviewing_pipeline.py saves as
                           ``{group}_manually_reviewed.pkl.gz``.
* GroupedExperiments    -- plotting/report layer on top of a group: overview, MS1/MS2 multipanel
                           figures and the paginated supplementary-information (SI) PDF pages.

Both result classes save/load as gzip-compressed pickles (``SerializableResult``).
"""
import numpy as np
import pandas as pd
import scipy as sp
import pyteomics as pyteomics
import warnings
from pathlib import Path
import os
import gzip
import pickle as pkl
import ast
import re
import string
import textwrap
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import DBSCAN

import matplotlib as mpl
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

from modification_dictionary import aa_comp_custom as aa_custom

from peptide_functions import *
from peptide_classes import *
from classes_spectrometry import *


def _load_fresh_experiment_pickle(old_exp):
    """Finds and loads a freshly regenerated MSExperiment pickle for the same sample `old_exp`
    describes -- used by ReplicateGroupResults.refresh_experiment_metadata() to pull corrected
    metadata (e.g. precursor_plasmid/precursor_name after fixing and re-running the input Excel)
    without touching any of the actual curated peak data.

    Searches old_exp.output_folder_name for the same candidate filenames
    pipeline_functions.run_new_analysis_for_group() itself looks for when first loading an
    experiment (duplicated here rather than imported -- pipeline_functions.py already imports
    FROM this module, so importing it back here would be circular). Returns None (not a raised
    error) if nothing is found, so a batch caller can skip/report just that one replicate instead
    of aborting.
    """
    output_folder = getattr(old_exp, "output_folder_name", None)
    if not output_folder:
        return None
    sample_path = Path(output_folder)
    if not os.path.isdir(to_long_path(sample_path)):
        return None

    spec_id = getattr(old_exp, "spectrum_id", "")
    exp_id = getattr(old_exp, "experiment_id", "")
    candidates = [
        sample_path / f"{spec_id}_experiment.pkl",
        sample_path / f"{spec_id}.pkl",
        sample_path / f"{exp_id}_experiment.pkl",
        sample_path / f"{exp_id}.pkl",
    ]
    found = next((c for c in candidates if os.path.isfile(to_long_path(c))), None)

    if found is None:
        # os.listdir on the long-path-safe dir (Path.glob would miss files in a >260-char folder).
        try:
            names = os.listdir(to_long_path(sample_path))
        except OSError:
            names = []
        pkl_matches = [
            sample_path / n for n in names
            if ".pkl" in n and not n.endswith("_analysis_results.pkl.gz") and not n.endswith("_cohort.pkl.gz")
        ]
        found = pkl_matches[0] if pkl_matches else None

    if found is None:
        return None

    with open(to_long_path(found), "rb") as f:
        return pkl.load(f)


# ==============================================================================
# --- BASE MIXIN FOR SERIALIZATION ---
# ==============================================================================

class SerializableResult:
    """Provides automated, compressed save/load functionality using gzip + pickle."""

    def save(self, filepath: str | Path) -> None:
        """Saves the entire object to a compressed gzip pickle file."""
        filepath = Path(filepath)
        os.makedirs(to_long_path(filepath.parent), exist_ok=True)
        with gzip.open(to_long_path(filepath), 'wb') as f:
            pkl.dump(self, f, protocol=pkl.HIGHEST_PROTOCOL)
        print(f"Successfully saved {self.__class__.__name__} to: {filepath}")

    @classmethod
    def load(cls, filepath: str | Path):
        """Loads a compressed result object from a file."""
        filepath = Path(filepath)
        if not os.path.exists(to_long_path(filepath)):
            raise FileNotFoundError(f"No saved file found at: {filepath}")
        with gzip.open(to_long_path(filepath), 'rb') as f:
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
                    # Pass mod_dict to generate_fragments
                    tgt_obj.generate_fragments(mod_dict=mod_dict, types=['b', 'y'], max_charge=1)
                    mod_target_objects.append(tgt_obj)

        # 2. Process Variant (Modified) Sequences
        mod_seqs = getattr(self.experiment, 'modified_sequences', None)
        mod_list = getattr(self.experiment, 'modifications', None)

        # Normalise both to parallel lists BEFORE zipping. Depending on how the experiment was
        # built, these can already be lists (main.py pre-parsed them) or still be the raw
        # comma-joined strings straight off the Excel row (the fallback path in
        # run_new_analysis_for_group that does MSExperiment(**row_dict) when no pickle is found).
        # Zipping two raw strings iterates them CHARACTER by character -- silently producing garbage
        # targets or none -- so parse here and make this function self-sufficient regardless of the
        # upstream path. This is the single authoritative modifications parser.
        if isinstance(mod_seqs, str):
            mod_seqs = [s.strip().replace('*', '') for s in mod_seqs.split(',') if s.strip()]
        if isinstance(mod_list, str):
            try:
                mod_list = ast.literal_eval(f"[{mod_list}]")
            except (ValueError, SyntaxError):
                print(f"[{self.experiment.spectrum_id}] Could not parse modifications string "
                      f"{mod_list!r}; skipping modified-sequence targets.")
                mod_list = None

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
                    # Pass mod_dict to generate_fragments
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
            if len(peaks) < 2:
                continue

            # De-duplication is an RT-containment test, so it must be scoped to a single charge
            # state: a +2 peak whose RT window happens to sit inside a +3 peak's window is a
            # different ion, not a duplicate. Grouping by charge first stops the containment check
            # from silently deleting real peaks across charges (which collapsed whole charge states
            # out of the results before this).
            by_charge = defaultdict(list)
            for p in peaks:
                by_charge[getattr(p, 'charge', None)].append(p)

            deduped = []
            for _charge, charge_peaks in by_charge.items():
                sorted_peaks = sorted(charge_peaks, key=lambda x: getattr(x, 'trapz_area', 0.0) or 0.0,
                                      reverse=True)
                kept = []
                for p in sorted_peaks:
                    p_start, p_end = getattr(p, 'rt_start', None), getattr(p, 'rt_end', None)
                    if p_start is None or p_end is None:
                        kept.append(p)
                        continue
                    if not any((p_start >= (getattr(k, 'rt_start', 0.0) or 0.0) and p_end <= (
                            getattr(k, 'rt_end', 0.0) or 0.0)) for k in kept):
                        kept.append(p)
                deduped.extend(kept)

            self.found_peaks[target] = deduped

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

    def repoint_storage_root(self, old_root: str, new_root: str, verbose: bool = True) -> "ReplicateGroupResults":
        """Rewrites the storage-root prefix on every replicate's experiment so cached spectra and
        per-sample pickles are found at a NEW location -- e.g. after the data was copied off an
        offline SSD (`D:\\Spliceotide_Data_Analysis`) into the project (`...\\PAPER_WRITING`) while
        the original drive isn't connected. Any stored path attribute (`input_mzml_folder`,
        `output_folder_name`, and, on older experiment objects, stored `output_path`/`mzml_path`)
        that starts with `old_root` has that prefix swapped for `new_root`; derived properties
        (MSExperiment.output_path/mzml_path) follow automatically. Case-sensitive prefix match.

        Rendering the SI / plotting a peak then serves each replicate from its cached _ms1/_ms2
        pickles under the new root, so the raw mzML files do NOT need to be present (they're only
        read on a cache miss). This mutates in place and returns self; call `.save(path)` afterward
        only if you want the new paths baked into the file (undo it when the SSD is back).
        """
        old_l, new_l = old_root.rstrip("\\/"), new_root.rstrip("\\/")

        def _swap(v):
            if v is None:
                return v, False
            s = v if isinstance(v, str) else os.fspath(v)
            if s.startswith(old_l):
                s2 = new_l + s[len(old_l):]
                return (s2 if isinstance(v, str) else type(v)(s2)), True
            return v, False

        changed = 0
        for ar in self.replicates.values():
            exp = getattr(ar, "experiment", None)
            if exp is None:
                continue
            for attr in ("input_mzml_folder", "output_folder_name", "output_path", "mzml_path"):
                if attr in vars(exp):        # only real stored attrs -- never a read-only property
                    new_v, did = _swap(getattr(exp, attr))
                    if did:
                        setattr(exp, attr, new_v)
                        changed += 1
        if verbose:
            print(f"repoint_storage_root: rewrote {changed} path attr(s) across "
                  f"{len(self.replicates)} replicate(s) ({old_l} -> {new_l})")
        return self

    def refresh_experiment_metadata(self, keys: tuple = ("precursor_plasmid", "precursor_name"),
                                    verbose: bool = True) -> "ReplicateGroupResults":
        """Patches specific metadata attributes (precursor_plasmid/precursor_name by default) on
        every replicate's EXISTING experiment object, pulling the corrected values from a freshly
        regenerated MSExperiment pickle -- NOT a re-pick. Nothing about the curated peaks
        themselves (selected_peaks, found_peaks, target_peptides, raw spectral data, etc.) is
        touched; only the named attributes on `replicate.experiment` are overwritten in place.

        The fresh MSExperiment is looked up at the SAME location each replicate's own (stale)
        experiment object already points to (its output_folder_name + spectrum_id/experiment_id --
        the same candidate filenames run_new_analysis_for_group() itself searches for), so this
        assumes you've already re-run whatever stage regenerates those per-sample MSExperiment
        pickles (e.g. main.py) from your corrected input Excel, and that each sample's
        output_folder_name hasn't itself changed. A replicate whose fresh pickle can't be found is
        left untouched and reported as a warning, not a hard failure -- so one missing/renamed
        sample doesn't abort the whole group.

        This deliberately does NOT touch self.group_name (or any file/folder path derived from
        it) -- group_name was fixed at the time this group was first picked and is left exactly
        as-is, even if precursor_plasmid changed (which is one of group_name's original
        components) -- so the folder/filename this ReplicateGroupResults lives under will keep
        reading the OLD code even after this call fixes the actual metadata inside it. Rename the
        folder/file yourself afterward if you want the two back in sync.

        Returns self (for chaining), so a batch driver can do
        `group_results.refresh_experiment_metadata().save(path)` in one line.
        """
        for rep_id, rep_results in self.replicates.items():
            old_exp = rep_results.experiment
            fresh_exp = _load_fresh_experiment_pickle(old_exp)
            if fresh_exp is None:
                if verbose:
                    print(f"  Warning: no fresh experiment pickle found for replicate {rep_id} "
                         f"(spectrum_id={getattr(old_exp, 'spectrum_id', '?')}) -- left unchanged.")
                continue
            changed = []
            for key in keys:
                if hasattr(fresh_exp, key):
                    old_val = getattr(old_exp, key, None)
                    new_val = getattr(fresh_exp, key)
                    if old_val != new_val:
                        setattr(old_exp, key, new_val)
                        changed.append(f"{key}: {old_val!r} -> {new_val!r}")
            if verbose and changed:
                print(f"  {rep_id}: " + "; ".join(changed))
        return self

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


def _measure_text_width_pt(text: str, fontsize: float) -> float:
    """Rendered width (points) of `text` at `fontsize`, mathtext-aware -- TextPath understands
    matplotlib mathtext ($...$) natively, so a bold/superscripted span is measured at its real
    glyph width, not its raw character count."""
    if not text:
        return 0.0
    return TextPath((0, 0), text, size=fontsize, prop=FontProperties()).get_extents().width


def _measure_text_height_pt(text: str, fontsize: float) -> float:
    """Rendered height (points) of `text` at `fontsize`, as typeset (i.e. NOT accounting for any
    rotation applied later via ax.text(..., rotation=...) -- rotating a TextPath's own geometry
    isn't a thing; a 90-degree-rotated label's on-page height is its UNROTATED WIDTH instead, see
    _measure_text_width_pt()). Use this one for text that will stay horizontal (rotation=0)."""
    if not text:
        return 0.0
    return TextPath((0, 0), text, size=fontsize, prop=FontProperties()).get_extents().height


# Vector/promoter-cassette tokens stripped from the FRONT of a precursor construct name to leave
# just its biologically-meaningful fusion/core name for the SI caption (see _strip_vector_prefix).
# Case-insensitive; extend this set if a new backbone element shows up as a leading token.
VECTOR_PREFIX_TOKENS = {"pacyc", "k", "t7", "laco", "elvj", "100k"}


def _strip_vector_prefix(name: str) -> str:
    """Trims a leading run of vector/promoter-cassette tokens (e.g. pACYC-T7-lacO-ElvJ-100k-...)
    off a precursor construct name, leaving the descriptive fusion/core name -- e.g.
    'pACYC-T7-lacO-ElvJ-100k-Fh8-chaos-HypA1-HypA1' -> 'Fh8-chaos-HypA1-HypA1', and
    'pACYC-K-T7-lacO-Fh8-HypAc-TEV-Artificial-1' -> 'Fh8-HypAc-TEV-Artificial-1'.

    Tokens (split on '-') are dropped only while they're in VECTOR_PREFIX_TOKENS (case-insensitive);
    the first token NOT in that set is where the real name begins, so everything from there on is
    kept verbatim. Returns the name unchanged if it has no recognized vector prefix, or if stripping
    would leave nothing -- an unexpected naming scheme is never mangled."""
    if not name or not isinstance(name, str):
        return name
    parts = name.split("-")
    i = 0
    while i < len(parts) and parts[i].lower() in VECTOR_PREFIX_TOKENS:
        i += 1
    stripped = "-".join(parts[i:])
    return stripped if stripped else name


# --- Sequence-ladder line-wrapping geometry (plot_si_report_simple's per-state MS2 sequence panel,
# panels E/G). A long peptide is split across several stacked lines so its letters and b/y-ion label
# stacks stay legible instead of collapsing into an unreadable block (as a 60-mer previously did on
# one line). Both _plot_si_sequence_annotations() (the drawing) and plot_si_report_simple() (which
# must know the line count up front to size that state's grid row) compute the wrap the SAME way via
# _seq_wrap_line_count(), so the reserved row height always matches what actually gets drawn.
SEQ_WRAP_TARGET_FONTSIZE = 10.0        # per-residue letter size used once a sequence has to wrap
# Min residue pitch as a multiple of mean letter width. Tuned so a sequence that already fits
# legibly on one line (historically up to the mid-20s of residues at font size 10) is NOT wrapped
# -- only genuinely over-long peptides (whose letters would otherwise touch/overlap) spill onto a
# second line. Lowering it wraps later (denser lines); raising it wraps sooner.
SEQ_WRAP_MIN_SPACING_FACTOR = 1.25
# Hard upper bound on residues per line, applied on top of the width-based fit above. The width fit
# alone allows ~26 in the half-width panel; capping it here keeps lines a bit shorter/roomier (per
# request). Lowering this wraps sooner (shorter lines); raising it lets lines run longer.
SEQ_WRAP_MAX_RESIDUES_PER_LINE = 20
_SEQ_RESIDUE_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _seq_residues_per_line(panel_width_in: float, fontsize: float = SEQ_WRAP_TARGET_FONTSIZE) -> int:
    """How many residues fit on one line of a sequence panel `panel_width_in` inches wide, keeping a
    pitch of at least SEQ_WRAP_MIN_SPACING_FACTOR x the mean amino-acid letter width so letters (and
    the fragmentation ticks between them) don't touch, and never exceeding
    SEQ_WRAP_MAX_RESIDUES_PER_LINE. Residues span x in [0.1, 0.9] of the panel (0.8 of its width),
    matching _plot_si_sequence_annotations()'s own layout."""
    mean_letter_in = max((_measure_text_width_pt(_SEQ_RESIDUE_ALPHABET, fontsize) /
                          len(_SEQ_RESIDUE_ALPHABET)) / 72.0, 1e-4)
    usable_in = max(0.8 * panel_width_in, mean_letter_in)
    width_fit = int(usable_in / (mean_letter_in * SEQ_WRAP_MIN_SPACING_FACTOR))
    return max(min(width_fit, SEQ_WRAP_MAX_RESIDUES_PER_LINE), 1)


def _seq_wrap_line_count(n_residues: int, panel_width_in: float,
                         fontsize: float = SEQ_WRAP_TARGET_FONTSIZE) -> int:
    """Number of stacked lines a sequence of `n_residues` needs at this panel width (>=1)."""
    if n_residues <= 0:
        return 1
    return max(1, int(np.ceil(n_residues / _seq_residues_per_line(panel_width_in, fontsize))))


def _tokenize_preserving_math(text: str) -> list:
    """Splits `text` on whitespace into words, except any $...$ mathtext span is kept intact as one
    atomic word -- so line-wrapping/justification (_wrap_and_justify_to_margin) never breaks, or
    inserts padding spaces inside, a math expression like $\\mathbf{A}$ or $[M+2H]^{2+}$."""
    return re.findall(r'\$[^$]*\$|\S+', text)


def _wrap_and_justify_to_margin(text: str, fontsize: float, fig_width_in: float,
                                margin_cm: float = 1.5) -> str:
    """Real, both-edges-flush paragraph justification with no external LaTeX dependency.

    matplotlib has no built-in justified-text renderer (usetex=True + a LaTeX \\justifying
    minipage can do it, but that hard-depends on a working local TeX install -- risky to require
    for report generation, since a missing/broken install would fail the whole PDF instead of just
    looking slightly uglier). This gets the same *visual* result without that dependency: it
    greedily word-wraps `text` into lines that fit within a margin_cm-inset band (mathtext-aware
    word measurement via TextPath, see _tokenize_preserving_math()/_measure_text_width_pt()), then
    stretches every line except the last flush to the target width by inserting extra literal
    space characters between its words -- the classic "space-padding" justification technique,
    proportioned across a line's own gaps rather than dumped in one spot.
    """
    margin_in = margin_cm / 2.54
    target_width_pt = max((fig_width_in - 2 * margin_in) * 72.0, 10.0)

    words = _tokenize_preserving_math(text)
    if not words:
        return text

    # A blank space glyph measures as ~zero width on its own (TextPath has nothing to draw), so
    # its width is inferred from the difference between a short string with and without one.
    space_width_pt = max(_measure_text_width_pt("I I", fontsize) - _measure_text_width_pt("II", fontsize), 1.0)

    lines, current, current_width = [], [], 0.0
    for word in words:
        w_width = _measure_text_width_pt(word, fontsize)
        extra = w_width if not current else space_width_pt + w_width
        if current and current_width + extra > target_width_pt:
            lines.append(current)
            current, current_width = [word], w_width
        else:
            current.append(word)
            current_width += extra
    if current:
        lines.append(current)

    justified_lines = []
    for i, line_words in enumerate(lines):
        is_last = (i == len(lines) - 1)
        if is_last or len(line_words) == 1:
            justified_lines.append(" ".join(line_words))
            continue
        natural_width = (sum(_measure_text_width_pt(w, fontsize) for w in line_words)
                        + space_width_pt * (len(line_words) - 1))
        slack_pt = max(target_width_pt - natural_width, 0.0)
        n_gaps = len(line_words) - 1
        extra_spaces_per_gap = int(round(slack_pt / space_width_pt / n_gaps)) if n_gaps else 0
        gap = " " * (1 + extra_spaces_per_gap)
        justified_lines.append(gap.join(line_words))

    return "\n".join(justified_lines)


def _words(text: str, color: str = None) -> list:
    """Splits `text` into individual (word, color) tokens, all tagged with the same `color`
    (None = default/inherit) -- the building block plot_si_report_simple()'s caption is assembled
    from (see _build_si_simple_caption()), so a caption is just one flat list of these instead of
    a plain string. That's what lets _render_justified_colored_caption() give a specific word (a
    modification-state name, say) its own color while every other word around it stays default,
    without any fragile after-the-fact substring search through a finished string.

    Uses _tokenize_preserving_math() rather than a plain text.split() -- a $...$ mathtext span
    that itself contains spaces (e.g. "$\\mathbf{SI\\ Appendix\\ A}$") would otherwise get torn
    into multiple broken, unbalanced fragments (each missing its own matching "$"), which either
    renders as garbled literal text or errors outright. This keeps any such span intact as ONE
    token no matter how many words it visually contains.
    """
    return [(w, color) for w in _tokenize_preserving_math(text)]


def _caption_line_height_frac(fontsize: float, fig_height_in: float, linespacing: float = 1.4) -> float:
    """Figure-fraction height of one caption text line at `fontsize`/`linespacing` on a
    fig_height_in-tall page -- shared by _render_caption_lines() (actually drawing lines) and
    plot_si_report_simple()'s own pagination estimate (how many lines fit in a given space,
    BEFORE anything is drawn), so the two always agree on how tall a line actually is.
    """
    return (fontsize * linespacing / 72.0) / fig_height_in


# Sentinel (word, color) token that forces a hard line break in caption text -- see _line_break().
# An otherwise-unreachable string, so it can never collide with real caption text.
_CAPTION_LINE_BREAK_WORD = "\x00LINEBREAK\x00"


def _line_break() -> list:
    """A single sentinel token that forces _break_caption_into_lines() to end the current line
    right there and start a fresh one, regardless of how much width is left on it -- for a hard,
    paragraph-style break inside a caption's token stream (e.g. plot_si_report_simple()'s intro/
    representative-replicate block vs. its panel-by-panel breakdown), rather than relying on the
    normal automatic width-based wrapping to happen to land there. The line ending in a forced
    break is never stretched to the margin (see _break_caption_into_lines()'s `no_justify` return
    value) -- it wasn't cut short because it ran out of room, so justifying it would just look like
    an odd, unnecessarily wide gap between its words.
    """
    return [(_CAPTION_LINE_BREAK_WORD, None)]


def _break_caption_into_lines(tokens: list, fontsize: float, fig_width_in: float,
                              margin_cm: float = 1.5) -> tuple:
    """Greedily wraps `tokens` (a flat list of (word, color) pairs, see _words()/_line_break())
    into lines that fit within a margin_cm-inset band of a fig_width_in-wide page -- PURE line-
    breaking, no rendering. Splitting this out from actually drawing the caption
    (_render_caption_lines()) is what lets plot_si_report_simple() measure how many lines a
    caption needs -- and therefore decide whether it fits in the space available, or needs to
    spill onto a second page -- before committing to draw anything.

    Returns (lines, no_justify_indices): `lines` is a list of lines, each itself a list of (word,
    color, width_pt) triples. `no_justify_indices` is a set of indices into `lines` that should
    never be stretched to the margin when rendered (see _render_caption_lines()) -- currently just
    the lines ending in a _line_break() sentinel, which end early by deliberate choice rather than
    running out of width.
    """
    margin_in = margin_cm / 2.54
    target_width_pt = max((fig_width_in - 2 * margin_in) * 72.0, 10.0)
    if not tokens:
        return [], set()

    space_width_pt = max(_measure_text_width_pt("I I", fontsize) - _measure_text_width_pt("II", fontsize), 1.0)

    lines, current, current_width = [], [], 0.0
    no_justify = set()
    for word, color in tokens:
        if word == _CAPTION_LINE_BREAK_WORD:
            if current:
                lines.append(current)
                no_justify.add(len(lines) - 1)
                current, current_width = [], 0.0
            continue
        w_width = _measure_text_width_pt(word, fontsize)
        extra = w_width if not current else space_width_pt + w_width
        if current and current_width + extra > target_width_pt:
            lines.append(current)
            current, current_width = [(word, color, w_width)], w_width
        else:
            current.append((word, color, w_width))
            current_width += extra
    if current:
        lines.append(current)
    return lines, no_justify


def _render_caption_lines(fig, lines: list, x_frac: float, y_frac: float, fontsize: float,
                          fig_width_in: float, fig_height_in: float, margin_cm: float = 1.5,
                          linespacing: float = 1.4, default_color: str = "black",
                          justify_last_line: bool = False, no_justify_indices: frozenset = frozenset()) -> float:
    """Renders an ALREADY-BROKEN set of caption lines (see _break_caption_into_lines()) as a real,
    both-edges-flush justified paragraph, one word at a time -- so each word can carry its own
    color -- inset margin_cm on each side of a fig_width_in-wide page, anchored top-left at
    (x_frac, y_frac) in figure-fraction coordinates.

    Unlike _wrap_and_justify_to_margin() (whole-line space-padding, i.e. only ever adding WHOLE
    extra space characters -- inherently a bit imprecise, and can't vary color mid-line), this
    places EVERY WORD as its own individually-positioned Text artist: each line's exact leftover
    slack is distributed as a precise (fractional-point) extra gap between every word on that
    line, so the right edge lands exactly on the margin instead of merely close to it.

    justify_last_line: by default the LAST line in `lines` is left plain (not stretched to the
    margin) -- the normal look for a paragraph's true final line. Pass True when `lines` is only a
    FRAGMENT of a longer caption that continues elsewhere (e.g. onto a second page) -- there, this
    chunk's last line isn't really the paragraph's last line, so it should still be justified like
    every other line in it.
    no_justify_indices: line indices (into THIS `lines` list -- remap if you sliced a bigger list
    before calling) that should never be stretched to the margin, e.g. lines ending in a
    _line_break() sentinel (see _break_caption_into_lines()'s own return value).

    Returns the y-fraction just below the last line rendered, in case the caller wants to know how
    much vertical space was actually used.
    """
    margin_in = margin_cm / 2.54
    target_width_pt = max((fig_width_in - 2 * margin_in) * 72.0, 10.0)
    fig_width_pt = fig_width_in * 72.0
    if not lines:
        return y_frac

    space_width_pt = max(_measure_text_width_pt("I I", fontsize) - _measure_text_width_pt("II", fontsize), 1.0)
    line_height_frac = _caption_line_height_frac(fontsize, fig_height_in, linespacing)
    y = y_frac

    for i, line in enumerate(lines):
        is_last = ((i == len(lines) - 1) and not justify_last_line) or (i in no_justify_indices)
        n_words = len(line)
        natural_width = sum(w for _, _, w in line) + space_width_pt * max(n_words - 1, 0)
        if is_last or n_words <= 1:
            gap_pt = space_width_pt
        else:
            slack_pt = max(target_width_pt - natural_width, 0.0)
            gap_pt = space_width_pt + slack_pt / (n_words - 1)

        cursor_pt = 0.0
        for word, color, w_width in line:
            x_this = x_frac + (cursor_pt / fig_width_pt)
            fig.text(x_this, y, word, transform=fig.transFigure, ha='left', va='top',
                    fontsize=fontsize, color=(color or default_color))
            cursor_pt += w_width + gap_pt

        y -= line_height_frac

    return y


def _render_justified_colored_caption(fig, tokens: list, x_frac: float, y_frac: float,
                                      fontsize: float, fig_width_in: float, fig_height_in: float,
                                      margin_cm: float = 1.5, linespacing: float = 1.4,
                                      default_color: str = "black") -> float:
    """Convenience one-shot wrapper: breaks `tokens` into lines (_break_caption_into_lines()) and
    renders all of them starting at (x_frac, y_frac) (_render_caption_lines()) -- for the common
    case where the caption fits on one page/region and there's no need to inspect the line count
    first. plot_si_report_simple() itself calls the two pieces separately so it can measure the
    caption's needed length before deciding where to place it.
    """
    lines, no_justify = _break_caption_into_lines(tokens, fontsize, fig_width_in, margin_cm)
    return _render_caption_lines(fig, lines, x_frac, y_frac, fontsize, fig_width_in, fig_height_in,
                                 margin_cm, linespacing, default_color, no_justify_indices=no_justify)


DEFAULT_SI_CAPTION_TEMPLATE = (
    "SI Appendix B Figure {i}: Results of coexpression of plasmid {precursor_plasmid} with "
    "{maturase_plasmid} at {expression_temperature}°C for {expression_time} h in "
    "{expression_scale} [mL] of culture volume, the precursor is {precursor_name} and was "
    "digested using {digestion}. The fragment shown ranges from {cut_start} to {cut_end} and "
    "charge state z={charge} is plotted"
)


# Nice display name for each raw modification-dict key seen in practice in the input Excel sheets
# (e.g. a "modifications" cell like [("spliced", 1), ("dehydrated", 1)] becomes the TargetPeptide
# dict {'spliced': 1, 'dehydrated': 1}). format_modification_label() falls back to a title-cased
# version of any key not listed here rather than crashing -- extend this dict (or pass your own via
# format_modification_label's key_nice_names=) once a new chemistry tag shows up.
MODIFICATION_KEY_NICE_NAMES = {
    "unmodified": "Unmodified",
    "spliced": "Spliced",
    "dehydrated": "Dehydrated",
    "deformylated": "Retro-Aldol",
}


def format_modification_label(modifications: dict, key_nice_names: dict = None) -> str:
    """Human-readable label for a target's modifications dict. Handles the real shape these dicts
    come in from the input Excel sheets -- a single unmodified flag, or one-or-more chemistry tags
    -- e.g.:
        {'Unmodified': True}                -> 'Unmodified'
        {'unmodified': None}                -> 'Unmodified'
        {'spliced': 1}                      -> 'Spliced'
        {'spliced': 2}                      -> 'Spliced 2'   (a distinct splice-site index, not a
                                                                count -- kept visible when != 1)
        {'spliced': 1, 'dehydrated': 1}     -> 'Spliced + Dehydrated'
        {'spliced': 1, 'deformylated': 1}   -> 'Spliced + Retro-Aldol'

    key_nice_names: raw modification key (case-insensitive) -> display name, defaults to the
    module-level MODIFICATION_KEY_NICE_NAMES. A key not found there falls back to str(key).title()
    instead of raising, so an unrecognized chemistry tag still renders (just less prettily) rather
    than breaking report generation.
    """
    if not isinstance(modifications, dict) or not modifications:
        return str(modifications)

    names = key_nice_names or MODIFICATION_KEY_NICE_NAMES

    # A single {'unmodified'/'Unmodified': ...} dict is always just "Unmodified", regardless of
    # what its value is (True, None, whatever) -- it's an on/off flag, not a count or index.
    if len(modifications) == 1:
        only_key, _ = next(iter(modifications.items()))
        if str(only_key).lower() == "unmodified":
            return "Unmodified"

    # "spliced" is always the structural/primary tag when present -- float it to the front
    # regardless of the dict's own insertion order, then keep the rest in their existing order.
    ordered_items = sorted(modifications.items(), key=lambda kv: 0 if str(kv[0]).lower() == "spliced" else 1)

    parts = []
    for key, value in ordered_items:
        nice = names.get(str(key).lower(), str(key).title())
        # A plain "on" flag (True/None/1) needs no number; anything else (e.g. a distinct
        # "spliced" site index, or an explicit multiplicity) is worth keeping visible.
        if value not in (True, None, 1):
            nice = f"{nice} {value}"
        parts.append(nice)

    return " + ".join(parts)


# Canonical color order for the "simple" SI report style (plot_si_report_simple() and the
# _plot_*_representative() helpers it calls) -- a mod state's color is its index into
# sns.color_palette() by its position in this list, so the same state always renders the same
# color across every page/precursor in a run (e.g. Unmodified is always palette color 0/blue,
# Spliced is always color 1/orange), not just consistently within one page. Override per-call via
# state_color_order= if a given study's canonical ordering differs from this default.
DEFAULT_SI_STATE_COLOR_ORDER = ["Unmodified", "Spliced", "Spliced + Dehydrated", "Spliced + Retro-Aldol"]

# Conventional English names for the default seaborn ("deep") 10-color palette, in order -- used
# to describe a state's color in prose in the plot_si_report_simple() caption (e.g. "the Blue
# trace corresponds to the Unmodified peptide"), matching how the palette's colors actually read.
_SEABORN_DEEP_COLOR_NAMES = ["Blue", "Orange", "Green", "Red", "Purple", "Brown", "Pink", "Gray",
                            "Olive", "Cyan"]

# Overflow registry for state labels NOT in whatever color_order was used -- populated the first
# time such a label is seen, so it stays consistent for the rest of this Python session (this
# session's whole batch of SI pages) even though it wasn't pre-declared in the canonical list.
# Stores the assigned PALETTE INDEX (not the color itself), so both the color and its prose name
# (_SEABORN_DEEP_COLOR_NAMES) can be derived from the one shared source of truth.
_EXTRA_STATE_COLOR_INDEX: dict = {}


def _state_palette_index(nice_label: str, color_order: list = None) -> int:
    """The palette index a mod-state nice label resolves to -- its position in `color_order`
    (DEFAULT_SI_STATE_COLOR_ORDER by default) if it's in there, else its (memoized,
    first-seen-this-session) overflow slot from _EXTRA_STATE_COLOR_INDEX. Shared source of truth
    for both get_state_color_map() (the actual color) and _state_color_name() (its prose name)."""
    order = list(color_order) if color_order is not None else list(DEFAULT_SI_STATE_COLOR_ORDER)
    if nice_label in order:
        return order.index(nice_label)
    if nice_label not in _EXTRA_STATE_COLOR_INDEX:
        _EXTRA_STATE_COLOR_INDEX[nice_label] = len(order) + len(_EXTRA_STATE_COLOR_INDEX)
    return _EXTRA_STATE_COLOR_INDEX[nice_label]


def get_state_color_map(nice_labels: list, color_order: list = None) -> dict:
    """Builds {nice_label: RGB color} for a set of mod-state nice labels, from a FIXED canonical
    order (DEFAULT_SI_STATE_COLOR_ORDER by default) using the default seaborn color palette, so the
    same state name always maps to the same color everywhere it's plotted -- across every panel of
    one page AND across every page/precursor in a run, e.g. a real SI figure legend needs to stay
    consistent.

    Any label not present in `color_order` still gets a color rather than crashing: see
    _state_palette_index() -- extend `color_order` instead once you know what that state should
    permanently be called/colored.
    """
    indices = [_state_palette_index(label, color_order) for label in nice_labels]
    palette = sns.color_palette(n_colors=max((max(indices) + 1) if indices else 0, 10))
    return {label: palette[idx] for label, idx in zip(nice_labels, indices)}


def _state_color_name(nice_label: str, color_order: list = None) -> str:
    """Prose color name (e.g. 'Blue') for a mod-state nice label -- see get_state_color_map()."""
    idx = _state_palette_index(nice_label, color_order)
    if idx < len(_SEABORN_DEEP_COLOR_NAMES):
        return _SEABORN_DEEP_COLOR_NAMES[idx]
    return f"palette color {idx}"


# Proton mass (Da) used to convert a TargetPeptide's neutral isotopologue masses into charged m/z
# -- the exact same constant/formula PeptidePeak.__init__ uses, duplicated here so theoretical mz
# windows can be computed straight from a TargetPeptide with no PeptidePeak/observed data needed
# (e.g. for a stub state that has no PeptidePeak at all).
_PROTON_MASS = 1.007276466812


def _theoretical_isotopologue_mz(target, charge: int) -> dict:
    """{isotopologue_name: m/z} for a TargetPeptide at a given charge state, computed the same way
    PeptidePeak.__init__ does -- usable even when there's no PeptidePeak object (e.g. a stub)."""
    return {
        name: (mass + charge * _PROTON_MASS) / charge
        for name, mass in zip(target.isotopologues["name"], target.isotopologues["mass"])
    }


_ELEMENT_ORDER = ["C", "H", "N", "O", "S", "P"]


def _format_composition_diff(from_target, to_target) -> str:
    """Elemental formula for what's lost/gained going from one TargetPeptide's composition to
    another's (e.g. Unmodified -> a spliced/modified state), computed directly from
    TargetPeptide.composition arithmetic (a pyteomics Composition, so plain dict subtraction just
    works) -- Hill-like element order (C, H, then everything else alphabetically), sign-prefixed.
    This is what drives the mz-change-plot arrow labels: it needs no hand-maintained per-chemistry
    mass table, so it's correct for any modification combination in MODIFICATION_KEY_NICE_NAMES or
    not.

    Returns a mathtext-ready FRAGMENT (element counts as real subscripts via "_{n}", e.g.
    "C_{8}H_{9}NO" -- standard chemistry convention of leaving a count of 1 implicit, so N/O above
    have no subscript at all) -- the caller is responsible for wrapping it in $...$ so those
    subscripts actually render instead of showing as literal underscores/braces.
    """
    diff = from_target.composition - to_target.composition  # positive count = atoms LOST from->to
    nonzero = {el: n for el, n in diff.items() if n != 0}
    if not nonzero:
        return "no\\ change"

    elements = [e for e in _ELEMENT_ORDER if e in nonzero]
    elements += sorted(e for e in nonzero if e not in _ELEMENT_ORDER)

    def _fmt(el, n):
        mag = abs(n)
        return el if mag == 1 else f"{el}_{{{mag}}}"

    lost = "".join(_fmt(el, nonzero[el]) for el in elements if nonzero[el] > 0)
    gained = "".join(_fmt(el, nonzero[el]) for el in elements if nonzero[el] < 0)

    if lost and gained:
        return f"-{lost}\\ +{gained}"
    if lost:
        return f"-{lost}"
    return f"+{gained}"


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
        os.makedirs(to_long_path(output_dir), exist_ok=True)

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
        mpl.pyplot.savefig(to_long_path(output_path), dpi=dpi)
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
                _apex_win = getattr(peak, 'raw_apex_scans', None)
                if _apex_win is None or _apex_win.empty:
                    _apex_win = peak.raw_apex
                if _apex_win is not None and not _apex_win.empty and 'time' in _apex_win.columns:
                    ax_tic.axvspan(_apex_win['time'].min(), _apex_win['time'].max(), alpha=0.1, color=color)

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

    def _plot_si_sequence_annotations(self, ax, rep_peak, b_ions, y_ions, state_label: str,
                                      residues_per_line: int = None):
        """Per-state peptide sequence plot with b/y ion match annotations. Each matched ion gets
        THREE separate labels stacked outward from the sequence (rather than one combined block):
        its ion name (e.g. "b3") in plain HORIZONTAL orientation closest to the sequence, then its
        observed fragment m/z as its own VERTICAL (rotated 90-degree) line, then -- furthest out --
        its ppm error and replicate count together as a second VERTICAL line reading
        "ppm=<err> n=<count>". y-ions mirror this exact same near-to-far ordering below the
        sequence.

        Long peptides WRAP: rather than crushing the whole sequence onto one line (where a 60-mer's
        letters and label stacks collapse into an unreadable block), the sequence is split into
        `residues_per_line` chunks drawn as stacked "ghost" lines, top to bottom, each with only the
        b/y ions whose fragmentation site falls within that chunk. `residues_per_line` should be the
        value plot_si_report_simple() already used to size this panel's grid row (so the drawn line
        count matches the reserved height); if None, it's derived from the axes' own width. Every
        stacked line reproduces the exact single-line layout, just scaled to its share of the panel
        height -- so a wrapped panel reads like several normal sequence panels stacked, and a
        peptide short enough for one line is drawn identically to before.

        The gap between the m/z tier and the ppm/n tier is computed DYNAMICALLY from the actual
        rendered length of the longest m/z string among all of this panel's ions (converted from
        points into an axes-fraction offset via the axes' own real physical height) rather than a
        fixed guessed constant -- a fixed gap either wastes space (short numbers) or, worse,
        isn't wide enough and the two tiers visually collide into unreadable overlapping text
        (long numbers, or a compact axes -- exactly what a fixed constant did here before). Using
        one SHARED (not per-ion) offset for the whole panel, sized to its longest string, keeps
        every ion's labels aligned at the same height while still guaranteeing none of them
        overlap.

        If there are no matched ions at all (b_ions and y_ions both empty/None -- this state has
        no usable MS2 data), a "No MS2 detected" label is drawn above the (still-drawn) bare
        sequence -- clear of the sequence letters, not overlapping them -- rather than silently
        leaving the panel with an unannotated sequence and no explanation. Shared by both
        plot_si_report_simple() and generate_si_report(), so this applies to either report style.
        """
        ax.axis('off')
        #ax.set_title(state_label, fontsize=7, weight='bold')

        if not (rep_peak and hasattr(rep_peak.target, 'sequence')):
            return

        parsed_modx_seq = pyteomics.parser.parse(rep_peak.target.sequence)
        seq_length = len(parsed_modx_seq)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # How many residues per line, and therefore how many stacked lines. Prefer the value the
        # caller reserved row height for; otherwise derive it from this axes' own physical width.
        if residues_per_line is not None and residues_per_line > 0:
            per_line = int(residues_per_line)
        else:
            panel_w_in = max(ax.get_position().width * ax.figure.get_size_inches()[0], 0.1)
            per_line = _seq_residues_per_line(panel_w_in)
        n_lines = max(1, int(np.ceil(seq_length / per_line))) if seq_length else 1
        # Balance residues evenly across the chosen number of lines rather than filling each line to
        # `per_line` and dumping the remainder on the last one -- otherwise a sequence just over a
        # multiple of per_line leaves a near-empty final line (e.g. 21 -> 20 + 1). n_lines (and thus
        # the reserved row height) is unchanged; only the per-line split tightens.
        if seq_length:
            per_line = int(np.ceil(seq_length / n_lines))

        # Single-line peptides keep the original size-by-length letter sizing (unchanged output);
        # wrapped ones use the fixed readable wrap size on each line.
        letter_fs = (SEQ_WRAP_TARGET_FONTSIZE if n_lines > 1
                     else (12 if seq_length < 20 else 10 if seq_length < 40 else 8))

        # `scale` = one line's share of the panel height (axes fraction). Every fixed axes-fraction
        # constant below is multiplied by it so each line is laid out like a full single-line panel
        # within its own band; the text-derived offsets scale automatically because axes_height_in
        # grows with the (row-height-grown) panel. n_lines == 1 -> scale == 1 -> original behavior.
        scale = 1.0 / n_lines

        # Per-line residue slices and their x-positions (residues span x in [0.1, 0.9] of the panel).
        lines = [parsed_modx_seq[k * per_line:(k + 1) * per_line] for k in range(n_lines)]
        line_x = [np.linspace(0.1, 0.9, len(res)) if len(res) > 1 else np.array([0.5]) for res in lines]

        def _y_center(line_idx):
            return 1.0 - (line_idx + 0.5) * scale

        def _locate(frag_pos):
            """(line index, 0-based local position) of a 1-based global residue anchor."""
            idx0 = frag_pos - 1
            return idx0 // per_line, idx0 - (idx0 // per_line) * per_line

        # Draw the sequence letters, line by line.
        for li, res in enumerate(lines):
            yc = _y_center(li)
            xs = line_x[li]
            for j, aa in enumerate(res):
                ax.text(float(xs[j]), yc, aa, fontsize=letter_fs, ha='center', va='center')

        has_matches = (b_ions is not None and not b_ions.empty) or (y_ions is not None and not y_ions.empty)
        if not has_matches:
            # Placed clear of the first line's sequence letters rather than directly on top of them.
            ax.text(0.5, _y_center(0) + 0.28 * scale, "No MS2 detected", ha='center', va='center',
                   fontsize=9, color='dimgray', style='italic', weight='bold', zorder=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='dimgray', alpha=0.9))
            return

        # Pre-compute every b-/y-ion's label strings first (rather than inline in the drawing
        # loops below) so the shared per-tier offsets can be sized to the single longest string
        # across ALL of them before anything is actually drawn.
        #
        # Layout, inner -> outer from the sequence: the ion NAME (ion colour, horizontal, closest),
        # then the observed m/z in BLACK (rotated 90 deg, the prominent value), then a small ppm tag
        # OUTERMOST drawn as plain HORIZONTAL text nudged to the right. Keeping the ppm horizontal
        # (short in the vertical direction) is what leaves the m/z room to be drawn larger. The
        # fragmentation-site tick starts a clear gap ABOVE/BELOW the amino-acid letters so it never
        # runs into the sequence text. A per-panel scale-to-fit + clip_on backstop keeps the whole
        # stack inside its own line band.
        name_fontsize, mz_fontsize, ppm_fontsize = 9, 7, 4.5

        def _collect(ions_df, frag_pos_fn):
            rows = []
            if ions_df is None or ions_df.empty:
                return rows
            for _, row in ions_df.iterrows():
                frag_pos = frag_pos_fn(row)
                if 0 < frag_pos <= seq_length:
                    mz_val = row.get('observed_mz')
                    mz_str = f"{mz_val:.3f}" if pd.notna(mz_val) else "N/A"
                    ppm_val = row.get('error_ppm', 0)
                    ppm_str = f"{ppm_val:+.1f}ppm" if pd.notna(ppm_val) else "n/a"
                    rows.append((frag_pos, row['fragment'], mz_str, ppm_str))
            return rows

        # bN covers the first N residues, so its cleavage site (and label anchor) sits AT residue
        # N -- the last residue that's actually part of the fragment.
        b_rows = _collect(b_ions, lambda row: int(row['fragment'][1:]))
        # yM covers the LAST M residues, i.e. residues (L-M+1)..L. Its cleavage site (and label
        # anchor) sits AT residue (L-M+1) -- the FIRST residue that's actually part of the
        # fragment (the +1 is required -- without it, the label anchors one residue too far left,
        # e.g. a real y3 rendered where a y4 belongs).
        y_rows = _collect(y_ions, lambda row: seq_length - int(row['fragment'][1:]) + 1)

        axes_height_in = max(ax.get_position().height * ax.figure.get_size_inches()[1], 0.1)

        def _rotated_text_frac(text, fontsize):
            """Axes-fraction height a ROTATED (90-degree) label of this string/fontsize occupies --
            its own (unrotated) rendered WIDTH becomes the vertical extent once rotated."""
            return _measure_text_width_pt(text, fontsize) / 72.0 / axes_height_in

        def _horizontal_text_frac(text, fontsize):
            """Axes-fraction height of a plain HORIZONTAL (rotation=0) label -- its rendered
            height, converted the same way."""
            return _measure_text_height_pt(text, fontsize) / 72.0 / axes_height_in

        # Clearance so the fragmentation-site tick starts off the amino-acid letters instead of
        # touching them: half a letter's height plus a small margin. The bare fixed fractions are
        # scaled to one line's share of the panel (see `scale`); the letter-height term already is.
        aa_clearance = 0.5 * _horizontal_text_frac("G", letter_fs) + 0.02 * scale
        gap_frac = 0.012 * scale        # breathing room between tiers
        tick_rise = 0.05 * scale        # height of the little site-marker tick, measured from aa_clearance

        # Tiers stacked outward from the sequence: name (closest, on top of the tick), then the m/z,
        # then the ppm furthest out. Each tier starts only after the previous tier's own measured
        # footprint (plus a gap). total_extent is the far edge of the outermost (ppm) tier; if it
        # would run past the room available above/below the sequence within ITS line band, shrink the
        # three fonts by the factor needed to make it fit -- targeting the font-dependent part only
        # (the tick clearance and gaps are fixed and don't scale). Sized once from the longest labels
        # across all lines so every line's tiers align at the same offsets.
        all_names = [r[1] for r in b_rows + y_rows]
        all_rows = b_rows + y_rows

        def _layout(name_fs, mz_fs, ppm_fs):
            name_h = max((_horizontal_text_frac(n, name_fs) for n in all_names), default=0.0)
            mz_h = max((_rotated_text_frac(r[2], mz_fs) for r in all_rows), default=0.0)
            ppm_h = max((_horizontal_text_frac(r[3], ppm_fs) for r in all_rows), default=0.0)
            name_off = aa_clearance + tick_rise + gap_frac
            mz_off = name_off + name_h + gap_frac
            ppm_off = mz_off + mz_h + gap_frac
            total = ppm_off + ppm_h
            return name_off, mz_off, ppm_off, total

        name_y_offset, mz_y_offset, ppm_y_offset, total_extent = _layout(
            name_fontsize, mz_fontsize, ppm_fontsize)

        available = 0.48 * scale        # room above (and below) the sequence within one line band
        fixed_extent = aa_clearance + tick_rise + 3 * gap_frac
        font_extent = total_extent - fixed_extent
        if total_extent > available and font_extent > 0:
            shrink = max((available - fixed_extent) / font_extent, 0.01)
            name_fontsize = max(name_fontsize * shrink, 5.0)
            mz_fontsize = max(mz_fontsize * shrink, 4.0)
            ppm_fontsize = max(ppm_fontsize * shrink, 3.0)
            name_y_offset, mz_y_offset, ppm_y_offset, total_extent = _layout(
                name_fontsize, mz_fontsize, ppm_fontsize)

        # clip_on=True is the hard guarantee that no label can render outside this panel and collide
        # with a neighbour. b-ions stack upward from each line's sequence; the site tick starts a
        # clear gap above the letters. Each ion is drawn on the line its fragmentation site lands on.
        for frag_pos, frag_name, mz_str, ppm_str in b_rows:
            li, local = _locate(frag_pos)
            xs = line_x[li]
            yc = _y_center(li)
            x_pos_diff = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
            ppm_x_shift = 0.1 * x_pos_diff
            x_pos_current = float(xs[local])
            ax.text(x_pos_current, yc + name_y_offset, frag_name,
                   color='blue', fontsize=name_fontsize, ha='center', va='bottom', rotation=0, clip_on=True)
            ax.text(x_pos_current, yc + mz_y_offset*1.1, mz_str, color='black',
                   fontsize=mz_fontsize, ha='right', va='bottom', rotation=90, clip_on=True)
            ax.text(x_pos_current + ppm_x_shift, yc + mz_y_offset*1.1, ppm_str,
                   color='blue', fontsize=ppm_fontsize, ha='left', va='bottom', rotation=90, clip_on=True)
            if local < len(xs) - 1:
                y_base = yc + aa_clearance
                lines_x = [x_pos_current + x_pos_diff / 2, x_pos_current + x_pos_diff / 2, x_pos_current]
                lines_y = [y_base, y_base + tick_rise * 0.6, y_base + tick_rise]
                ax.plot(lines_x, lines_y, color='blue', linewidth=1)

        for frag_pos, frag_name, mz_str, ppm_str in y_rows:
            li, local = _locate(frag_pos)
            xs = line_x[li]
            yc = _y_center(li)
            x_pos_diff = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
            ppm_x_shift = 0.1 * x_pos_diff
            x_pos_current = float(xs[local])
            ax.text(x_pos_current, yc - name_y_offset, frag_name,
                   color='red', fontsize=name_fontsize, ha='center', va='top', rotation=0, clip_on=True)
            ax.text(x_pos_current, yc - mz_y_offset, mz_str, color='black',
                   fontsize=mz_fontsize, ha='right', va='top', rotation=90, clip_on=True)
            ax.text(x_pos_current + ppm_x_shift, yc - mz_y_offset, ppm_str,
                   color='red', fontsize=ppm_fontsize, ha='left', va='top', rotation=90, clip_on=True)
            if local > 0:
                y_base = yc - aa_clearance
                lines_x = [x_pos_current - x_pos_diff / 2, x_pos_current - x_pos_diff / 2, x_pos_current]
                lines_y = [y_base, y_base - tick_rise * 0.6, y_base - tick_rise]
                ax.plot(lines_x, lines_y, color='red', linewidth=1)

    # ==============================================================================================
    # --- "SIMPLE" SI FIGURE STYLE (plot_si_report_simple() and the helpers it calls) ---
    # A second, differently-laid-out single-figure SI style (requested separately from
    # generate_si_report() above): one "representative" replicate drives the TIC/EIC-overview/
    # zoomed-EIC/mz-change panels, while the bar plot and MS2 panels pool every curated replicate
    # group-wide. See plot_si_report_simple()'s own docstring for the full panel layout.
    # ==============================================================================================

    def _combine_raw_ms2_data(self, isoform_peaks: list) -> pd.DataFrame:
        """Concatenates every replicate's raw (unfiltered) MS2 spectrum for one state into a single
        DataFrame -- the faint background trace in _plot_ms2_mirror pools every fragment ion ever
        observed for this state across all its curated replicates, not just one peak's own data."""
        raw_dfs = [p.ms2_data for p in isoform_peaks
                  if getattr(p, 'ms2_data', None) is not None and not p.ms2_data.empty]
        if not raw_dfs:
            return pd.DataFrame(columns=['fragment_mz', 'fragment_int'])
        return pd.concat(raw_dfs, ignore_index=True)

    def _plot_ms2_mirror(self, ax, isoform_peaks: list, show_title: bool = False):
        """Head-to-tail mirrored MS2 plot for one modification state -- really does copy
        PeptidePeak.plot_ms2_mirrored()'s own approach (including its per-ion fragment-name text
        labels, rotated 90 degrees under each matched vline), adapted for this multi-replicate
        GroupedExperiments context: instead of one peak's own raw/matched MS2 data, both halves are
        pooled across every curated replicate that has a selection for this state (isoform_peaks),
        via _combine_raw_ms2_data()/_compute_ms2_matches(). Matched b-ions plot downward in blue,
        y-ions downward in red (each labeled with its fragment name, e.g. "b3"/"y4"); the full raw
        fragmentation pattern plots upward in black, mirrored against them.

        One deliberate fix vs. the original: PeptidePeak.plot_ms2_mirrored() labels y-ions with
        f"y{row['fragment']}", but row['fragment'] is already "y4" etc., so that renders as "yy4" --
        not reproduced here, the label is just row['fragment'] for both ion types.

        show_title: if True, sets an axes title ("Mirror plot of ms2 fragments matching the
        peptide") -- off by default; plot_si_report_simple()'s panels are identified by their
        corner letter (see subpanel_labels there) rather than individual titles.

        If this state has no MS2 data at all (no raw spectrum AND no matched ions -- a curated MS1
        peak was selected for it, but no fragmentation data ever matched it), the axes is left
        blank apart from a centered "No MS2 detected" label instead of silently rendering an
        empty/uninformative plot.
        """
        raw_ms2 = self._combine_raw_ms2_data(isoform_peaks)
        b_ions, y_ions = self._compute_ms2_matches(isoform_peaks)
        has_any_ms2 = (not raw_ms2.empty) or (b_ions is not None and not b_ions.empty) \
            or (y_ions is not None and not y_ions.empty)

        if not has_any_ms2:
            ax.text(0.5, 0.5, "No MS2 detected", ha='center', va='center', transform=ax.transAxes,
                   fontsize=7, color='dimgray', style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("m/z")
            ax.set_ylabel("Intensity [a.u.]")
            if show_title:
                ax.set_title("Mirror plot of ms2 fragments matching the peptide", fontsize=7, weight='bold')
            return

        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.7)

        max_raw_ms2 = 1
        if not raw_ms2.empty:
            ax.vlines(raw_ms2['fragment_mz'], ymin=0, ymax=raw_ms2['fragment_int'], color='black',
                      linewidth=0.6, alpha=0.4, label='Raw MS2')
            max_raw_ms2 = raw_ms2['fragment_int'].max()

        matched_frames = [df for df in (b_ions, y_ions) if df is not None and not df.empty]
        max_match_int = pd.concat(matched_frames)['intensity'].max() if matched_frames else 1
        # Distance between each matched vline's tip and its fragment-name label -- bumped up from
        # a plain 3% of max intensity to 8% to reduce label/line clutter in a dense spectrum.
        label_gap = max_match_int * 0.08

        if b_ions is not None and not b_ions.empty:
            ax.vlines(b_ions['observed_mz'], ymin=0, ymax=-b_ions['intensity'], color='blue',
                      linewidth=1.0, label='Matched b-ions')
            for _, row in b_ions.iterrows():
                ax.text(row['observed_mz'], -row['intensity'] - label_gap, row['fragment'],
                       color='blue', va='top', ha='center', fontsize=5, rotation=90, clip_on=False)
        if y_ions is not None and not y_ions.empty:
            ax.vlines(y_ions['observed_mz'], ymin=0, ymax=-y_ions['intensity'], color='red',
                      linewidth=1.0, label='Matched y-ions')
            for _, row in y_ions.iterrows():
                ax.text(row['observed_mz'], -row['intensity'] - label_gap, row['fragment'],
                       color='red', va='top', ha='center', fontsize=5, rotation=90, clip_on=False)

        # Left/right x-margin so the lowest- and highest-m/z ions (and their rotated name labels)
        # clear the axis spines instead of sitting on top of them -- this is what stops the leftmost
        # fragment labels from colliding with the intensity y-axis and its tick numbers. The left
        # pad is a touch wider than the right since the labels hang from the ions and a low-m/z one
        # otherwise lands right against the y-axis.
        all_mz = []
        if not raw_ms2.empty:
            all_mz += list(raw_ms2['fragment_mz'])
        for _df in (b_ions, y_ions):
            if _df is not None and not _df.empty:
                all_mz += list(_df['observed_mz'])
        if all_mz:
            x_lo, x_hi = min(all_mz), max(all_mz)
            x_range = (x_hi - x_lo) or 1.0
            ax.set_xlim(x_lo - 0.06 * x_range, x_hi + 0.03 * x_range)

        # Headroom above zero is just enough for the raw trace; headroom BELOW zero is sized to the
        # actual rendered length of the longest matched-ion name label (fontsize 5, rotated 90deg so
        # its on-page length is its unrotated WIDTH) so those labels -- which hang from each downward
        # vline tip -- never spill past the bottom axis. The label's data-unit length depends on the
        # very y-span we're solving for, so two fixed-point passes settle it. Asymmetric limits keep
        # the top compact while still guaranteeing the bottom labels fit inside the panel.
        pos_top = max(max_raw_ms2, max_match_int) * 1.15
        axes_h_in = max(ax.get_position().height * ax.figure.get_size_inches()[1], 0.1)
        label_len_in = max(
            (_measure_text_width_pt(str(r['fragment']), 5) / 72.0
             for _df in (b_ions, y_ions) if _df is not None and not _df.empty
             for _, r in _df.iterrows()),
            default=0.0,
        )
        neg_bottom = max_match_int * 1.25
        for _ in range(2):
            data_per_in = (pos_top + neg_bottom) / axes_h_in
            needed = max_match_int + label_gap + label_len_in * data_per_in
            neg_bottom = max(needed * 1.03, max_match_int * 1.25)
        ax.set_ylim(-neg_bottom, pos_top)
        # Mirrored intensities are stored as negative y-values purely so the b/y ions plot below
        # the raw trace -- the axis should still read as plain (always-positive), compact
        # scientific-notation intensity. _apply_compact_sci_axis() builds the formatter; its own
        # per-tick strings are then reused with any leading minus sign stripped (rather than a
        # manual "int(abs(y))" label, which read as a plain integer and skipped scientific
        # notation -- exactly the "too much space" this report tries to avoid everywhere else).
        _apply_compact_sci_axis(ax)
        _fmt = ax.yaxis.get_major_formatter()
        _ticks = ax.get_yticks()
        _fmt.set_locs(_ticks)
        # set_ticks() first (pinning them via a FixedLocator) before set_yticklabels() -- without
        # it matplotlib warns that the tick positions aren't fixed yet and could still change on
        # redraw, since get_yticks() alone doesn't freeze the locator that produced them.
        ax.set_yticks(_ticks)
        ax.set_yticklabels([_fmt(y).replace("−", "").lstrip("-") for y in _ticks])

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity [a.u.]")
        if show_title:
            ax.set_title("Mirror plot of ms2 fragments matching the peptide", fontsize=7, weight='bold')
        ax.legend(fontsize=5, loc='upper right')

    def _find_representative_replicate(self, cut_id, charge) -> tuple:
        """Picks the single "representative" replicate for one (fragment_cut, charge): the one
        whose curated (non-stub, non-Reject-tier) selections span the most DISTINCT modification
        states at that cut/charge; ties are broken by the highest total peak area (summed
        trapz_area across that replicate's curated peaks at this cut/charge). Requires
        from_replicate_group_results() construction.

        Returns (replicate_id, {state_key: PeptidePeak}): the state->peak map holds every state
        this replicate has a curated selection for at this cut/charge -- a state present in the
        group's target_dict but missing from this map is a stub for THIS replicate specifically
        (it may still be detected in other replicates -- see the group-wide isoform_peaks_by_state
        built in plot_si_report_simple() for that). Returns (None, {}) if nothing is curated at
        this cut/charge in any replicate.
        """
        if self.group_results is None:
            raise ValueError(
                "_find_representative_replicate() requires a ReplicateGroupResults -- build this "
                "GroupedExperiments via GroupedExperiments.from_replicate_group_results(...)."
            )

        target_dict = self._get_theoretical_targets_dict(cut_id)
        best_rep_id, best_peaks_map, best_key = None, {}, (-1, -1.0)

        for rep_id in self.group_results.replicates.keys():
            peaks_map = {}
            for state_key, target_obj in target_dict.items():
                modification = str(target_obj.modifications)
                peak = self._find_selected_peak(rep_id, cut_id, modification, charge,
                                                sequence=target_obj.sequence)
                if peak is not None and getattr(peak, "tier", None) != "Reject":
                    peaks_map[state_key] = peak

            n_states = len(peaks_map)
            total_area = sum((getattr(p, 'trapz_area', 0.0) or 0.0) for p in peaks_map.values())
            key = (n_states, total_area)
            if key > best_key:
                best_key = key
                best_rep_id, best_peaks_map = rep_id, peaks_map

        return best_rep_id, best_peaks_map

    def _order_states_canonically(self, all_possible_states: list, state_labels: dict,
                                  color_order: list = None) -> list:
        """Orders raw state keys by their nice label's position in `color_order`
        (DEFAULT_SI_STATE_COLOR_ORDER by default), Unmodified first -- so every panel in
        plot_si_report_simple() (mz-change, EIC overlays, MS2 rows) lays states out in the same
        fixed order every time, matching how their colors are pinned (get_state_color_map()). A
        state whose nice label isn't in `color_order` keeps its original relative order, sorted
        after every canonical one (sorted() is stable, so ties preserve all_possible_states' own
        ordering).
        """
        order = color_order or DEFAULT_SI_STATE_COLOR_ORDER

        def _rank(state_key):
            label = state_labels.get(state_key, state_key)
            return order.index(label) if label in order else len(order)

        return sorted(all_possible_states, key=_rank)

    def _draw_state_apex_panel(self, ax, mono_mz: float, last_iso_mz: float, peak, color,
                               is_stub: bool):
        """Draws one state's slice of the mz-change panel onto `ax`: a translucent colored box
        spanning its theoretical monoisotopic-to-last-isotopologue mz window, the representative
        peak's real raw apex as black vlines on top of it -- or, if this state is a stub for the
        representative replicate, vertical "Not Detected" text centered in that window instead."""
        ax.axvspan(mono_mz, last_iso_mz, color=color, alpha=0.2, zorder=0)
        if not is_stub and peak is not None and getattr(peak, 'raw_apex', None) is not None \
                and not peak.raw_apex.empty:
            ax.vlines(peak.raw_apex['mz_array'], ymin=0, ymax=peak.raw_apex['intensity_array'],
                      color='black', linewidth=0.8, zorder=2)
        else:
            mid_mz = (mono_mz + last_iso_mz) / 2
            ax.text(mid_mz, 0.5, "Not Detected", rotation=90, ha='center', va='center', fontsize=6,
                   color='dimgray', transform=ax.get_xaxis_transform(), zorder=2)

    def _draw_mz_change_arrows(self, fig, axes_for_state: dict, ordered_states: list,
                               state_labels: dict, windows: dict, target_dict: dict,
                               color_map: dict, unmod_key):
        """Draws one arrow per non-Unmodified state, from the Unmodified box to that state's own
        box, labeled with the state's nice name and its elemental formula loss/gain relative to
        Unmodified (_format_composition_diff()) -- colored to match the target state's box. The
        label sits centered ON the arrow's actual visual midpoint (with a small translucent white
        backing box so it stays legible crossing the arrow line), not just above it.

        Uses a matplotlib ConnectionPatch (added directly to the Figure), which can span two
        different Axes -- so this works identically whether every state shares one Axes
        (continuous mode, axes_for_state maps every key to the SAME ax) or each has its own
        (broken-axis mode).
        """
        if unmod_key is None or unmod_key not in axes_for_state:
            return

        ax_unmod = axes_for_state[unmod_key]
        unmod_mono, unmod_last = windows[unmod_key]
        unmod_mid = (unmod_mono + unmod_last) / 2
        unmod_target = target_dict[unmod_key]

        targets = [s for s in ordered_states if s != unmod_key and s in axes_for_state]
        for i, state_key in enumerate(targets):
            ax_state = axes_for_state[state_key]
            mono_mz, last_iso_mz = windows[state_key]
            mid_mz = (mono_mz + last_iso_mz) / 2
            label = state_labels[state_key]
            color = color_map[label]
            formula = _format_composition_diff(unmod_target, target_dict[state_key])
            y_frac = 0.78 + 0.07 * (i % 3)

            con = mpl.patches.ConnectionPatch(
                xyA=(unmod_mid, y_frac), coordsA=ax_unmod.get_xaxis_transform(),
                xyB=(mid_mz, y_frac), coordsB=ax_state.get_xaxis_transform(),
                arrowstyle='->', color=color, linewidth=1.2, mutation_scale=10, zorder=5,
            )
            fig.add_artist(con)

            # The arrow's true on-page midpoint, computed in FIGURE (pixel) space rather than by
            # averaging the two endpoints' raw data-x values: in broken-axis mode the two ends live
            # in two entirely separate mini-axes with unrelated data-x ranges, so only their actual
            # rendered positions are meaningfully "between" each other. In continuous mode (both
            # endpoints on the same ax) this just degrades to the ordinary same-axes midpoint.
            p_a_disp = ax_unmod.get_xaxis_transform().transform((unmod_mid, y_frac))
            p_b_disp = ax_state.get_xaxis_transform().transform((mid_mz, y_frac))
            mid_fig = fig.transFigure.inverted().transform((p_a_disp + p_b_disp) / 2)

            fig.text(mid_fig[0], mid_fig[1], f"{label} ${formula}$", transform=fig.transFigure,
                    ha='center', va='center', fontsize=5, color=color, zorder=6,
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.8))

    def _plot_mz_change_plot_representative(self, fig, subplot_spec, ordered_states: list,
                                            state_labels: dict, representative_peaks: dict,
                                            target_dict: dict, charge: int, color_map: dict,
                                            break_axis: bool = False):
        """Panel E: representative-replicate apex m/z panel across every state, with the shift
        between them made visually explicit.

        For each state (canonical order, Unmodified first): a translucent box colored per
        color_map spans its theoretical mz window (monoisotopic to the last searched
        isotopologue), with the representative replicate's real apex drawn as black vlines on top
        (_draw_state_apex_panel()) -- or "Not Detected" text if that state is a stub for the
        representative replicate. An arrow runs from the Unmodified box to every other state's box
        (_draw_mz_change_arrows()).

        break_axis: if True, gives each state its own mini mz-window (a real broken x-axis, with
        break marks between adjacent states) instead of one continuous shared mz axis -- off by
        default, since states are usually close enough in mass for one shared axis to read fine,
        and it makes the cross-state arrows trivial to draw. Returns the single axis that ended up
        holding the panel's labels/legend (in broken-axis mode, its leftmost mini-axis).
        """
        windows = {}
        for state_key in ordered_states:
            peak = representative_peaks.get(state_key)
            if peak is not None and getattr(peak, 'isotopologues', None):
                iso_mz = peak.isotopologues
            else:
                iso_mz = _theoretical_isotopologue_mz(target_dict[state_key], charge)
            mono_mz = iso_mz.get("monoisotopic", min(iso_mz.values()))
            last_iso_mz = max(iso_mz.values())
            windows[state_key] = (mono_mz, last_iso_mz)

        unmod_key = next((s for s in ordered_states if state_labels[s] == "Unmodified"), None)

        if not break_axis:
            ax = fig.add_subplot(subplot_spec)
            all_mono = [w[0] for w in windows.values()]
            all_last = [w[1] for w in windows.values()]
            span = (max(all_last) - min(all_mono)) if windows else 1.0
            margin = max(span * 0.15, 1.0)
            ax.set_xlim(min(all_mono) - margin, max(all_last) + margin)

            for state_key in ordered_states:
                peak = representative_peaks.get(state_key)
                mono_mz, last_iso_mz = windows[state_key]
                color = color_map[state_labels[state_key]]
                self._draw_state_apex_panel(ax, mono_mz, last_iso_mz, peak, color, is_stub=(peak is None))

            # Extra headroom above the tallest real apex, reserved for the arrows/labels drawn at
            # a fixed axes-fraction height above -- drawn AFTER the real data so the final ylim
            # (and therefore the arrows' actual on-page position) is settled first.
            ax.set_ylim(0, max(ax.get_ylim()[1] * 1.4, 1.0))

            axes_for_state = {s: ax for s in ordered_states}
            self._draw_mz_change_arrows(fig, axes_for_state, ordered_states, state_labels, windows,
                                        target_dict, color_map, unmod_key)
            result_ax = ax
        else:
            # Mini-axes are laid out left-to-right in ASCENDING theoretical monoisotopic m/z --
            # NOT `ordered_states`' canonical (Unmodified-first, color-driven) order. A broken axis
            # is still meant to read as one continuous mz axis with a few gaps cut out of it, so it
            # has to be monotonically increasing left-to-right regardless of which state happens to
            # be "Unmodified" or first in the color-legend order (a lower-mass modified state, e.g.
            # after splicing, can easily sit to the LEFT of Unmodified in real mz terms).
            spatial_order = sorted(ordered_states, key=lambda s: windows[s][0])
            n = len(spatial_order)
            gs_inner = mpl.gridspec.GridSpecFromSubplotSpec(1, n, subplot_spec=subplot_spec, wspace=0.06)
            axes = [fig.add_subplot(gs_inner[0, i]) for i in range(n)]
            axes_for_state = dict(zip(spatial_order, axes))

            for state_key, ax_i in axes_for_state.items():
                peak = representative_peaks.get(state_key)
                mono_mz, last_iso_mz = windows[state_key]
                # Same 1/3-1/3-1/3 framing used for the isolated per-state isotope-pattern panels
                # elsewhere -- appropriate here since each state now gets its OWN mini axis.
                iso_span = max(last_iso_mz - mono_mz, 1e-6)
                ax_i.set_xlim(mono_mz - iso_span, last_iso_mz + iso_span)
                color = color_map[state_labels[state_key]]
                self._draw_state_apex_panel(ax_i, mono_mz, last_iso_mz, peak, color, is_stub=(peak is None))
                if ax_i is not axes[0]:
                    ax_i.spines['left'].set_visible(False)
                    ax_i.tick_params(axis='y', which='both', left=False, labelleft=False)
                if ax_i is not axes[-1]:
                    ax_i.spines['right'].set_visible(False)

            # A SHARED y-scale across every mini-axis, not each one independently autoscaling to
            # its own peak: since each state has its own separate Axes here, matplotlib's default
            # per-axes autoscaling would otherwise render a weak peak and a strong peak at the same
            # visual height in their own boxes -- exactly the kind of distortion a broken axis is
            # supposed to avoid. Read each axis's own natural ylim (already autoscaled by the vlines
            # just drawn on it) first, then reapply the largest one to all of them.
            global_ymax = max((ax_i.get_ylim()[1] for ax_i in axes), default=1.0)
            for ax_i in axes:
                ax_i.set_ylim(0, max(global_ymax * 1.4, 1.0))

            # Break marks: small diagonal ticks at the touching edge of each adjacent axis pair --
            # the standard matplotlib "broken axis" idiom.
            for ax_left, ax_right in zip(axes[:-1], axes[1:]):
                d = 0.4
                kwargs = dict(marker=[(-1, -d), (1, d)], markersize=8, linestyle='none', color='k',
                              mec='k', mew=1, clip_on=False)
                ax_left.plot([1], [0], transform=ax_left.transAxes, **kwargs)
                ax_left.plot([1], [1], transform=ax_left.transAxes, **kwargs)
                ax_right.plot([0], [0], transform=ax_right.transAxes, **kwargs)
                ax_right.plot([0], [1], transform=ax_right.transAxes, **kwargs)

            self._draw_mz_change_arrows(fig, axes_for_state, ordered_states, state_labels, windows,
                                        target_dict, color_map, unmod_key)
            result_ax = axes[0]

        # Cosmetics common to both modes, applied once to the leftmost/only axis so labels aren't
        # duplicated across every mini-axis in broken-axis mode.
        result_ax.set_xlabel("m/z")
        result_ax.set_ylabel("Intensity [a.u.]")
        _apply_compact_sci_axis(result_ax)

        legend_handles = [mpl.patches.Patch(facecolor=color_map[state_labels[s]], alpha=0.2,
                                            label=state_labels[s]) for s in ordered_states]
        result_ax.legend(handles=legend_handles, fontsize=5, loc='upper right', framealpha=0.85)

        return result_ax

    def _plot_eic_representative(self, ax, ordered_states: list, state_labels: dict,
                                 representative_peaks: dict, color_map: dict, charge: int,
                                 rt_range: tuple = None):
        """Representative-replicate zoomed EIC overlay, one trace per state (each state's own
        zoomed_extracted -- the same RT window that peak was actually integrated over), with the
        area under each curve filled in that state's own color (alpha 0.2, dashed fill edge). A
        state missing from representative_peaks (a stub for this replicate) gets a legend entry
        reading "<label>: peak not detected" instead of a plotted line/fill.

        The legend is titled "<charge-state notation> peptide EIC trace" (e.g. "$[M+2H]^{2+}$
        peptide EIC trace", mathtext) and each entry's label spells out that state's theoretical
        monoisotopic m/z (not every isotopologue -- too much for a legend entry to stay readable)
        and its integrated peak area, not just its name.

        Uses zoomed_extracted's own precomputed 'total_intensity' column rather than summing the
        DataFrame's columns itself -- zoomed_extracted already carries that column alongside the
        per-isotopologue ones it was summed from, so re-summing the whole frame would double-count.
        """
        legend_handles = []
        for state_key in ordered_states:
            label = state_labels[state_key]
            color = color_map[label]
            peak = representative_peaks.get(state_key)
            zoomed = getattr(peak, 'zoomed_extracted', None) if peak is not None else None

            if zoomed is not None and not zoomed.empty and 'total_intensity' in zoomed.columns:
                ax.plot(zoomed.index, zoomed['total_intensity'], color=color, alpha=0.9, linewidth=1.2)
                ax.fill_between(zoomed.index, 0, zoomed['total_intensity'], color=color, alpha=0.2,
                               linestyle='--', edgecolor=color, linewidth=0.6)

                iso_mz = getattr(peak, 'isotopologues', None) or {}
                mono_mz = iso_mz.get("monoisotopic")
                mono_bit = f"monoisotopic: {mono_mz:.4f}" if mono_mz is not None else "monoisotopic: N/A"
                area = peak.quality_metrics.get('final_trapz_area') if peak.quality_metrics.get(
                    'final_trapz_area') is not None else (getattr(peak, 'trapz_area', 0) or 0)
                legend_label = f"{label}\n{mono_bit}\narea: {area:.2e}"
                legend_handles.append(mpl.lines.Line2D([], [], color=color, label=legend_label))
            else:
                legend_handles.append(mpl.lines.Line2D([], [], color=color, linestyle='None',
                                                        marker='x', label=f"{label}: peak not detected"))

        if rt_range is not None:
            ax.set_xlim(rt_range)
        ax.set_xlabel("RT [min]")
        ax.set_ylabel("Intensity [a.u.]")
        _apply_compact_sci_axis(ax)
        legend_title = f"$[M+{charge}H]^{{{charge}+}}$ peptide EIC trace"
        ax.legend(handles=legend_handles, fontsize=4.5, loc='upper right', title=legend_title,
                 title_fontsize=5.5, framealpha=0.85)

    def _plot_tic_representative(self, ax, rep_spectrum, rt_range: tuple = None):
        """Panel A: TIC for the representative replicate only, plotted in black."""
        tic = rep_spectrum.get_tic(filtered=False)
        if rt_range is not None and not tic.empty:
            tic = tic[(tic.index >= rt_range[0]) & (tic.index <= rt_range[1])]
        if not tic.empty:
            ax.plot(tic.index, tic.values, color='black', linewidth=0.8)
        if rt_range is not None:
            ax.set_xlim(rt_range)
        ax.set_ylabel("Intensity [a.u.]")
        _apply_compact_sci_axis(ax)

    def _plot_eic_overview_representative(self, ax, ordered_states: list, state_labels: dict,
                                          representative_peaks: dict, color_map: dict,
                                          rt_range: tuple = None):
        """Panel B: full-range (not zoomed-to-peak) EIC overlay for the representative replicate,
        one trace per detected state, over the same rt_range as the TIC above it."""
        for state_key in ordered_states:
            peak = representative_peaks.get(state_key)
            if peak is None or getattr(peak, 'extracted', None) is None or peak.extracted.empty:
                continue
            extracted = peak.extracted
            if rt_range is not None:
                extracted = extracted[(extracted.index >= rt_range[0]) & (extracted.index <= rt_range[1])]
            if extracted.empty:
                continue
            color = color_map[state_labels[state_key]]
            ax.plot(extracted.index, extracted.sum(axis=1), color=color, alpha=0.85, linewidth=1.0)

        if rt_range is not None:
            ax.set_xlim(rt_range)
        ax.set_xlabel("Retention Time [min]")
        ax.set_ylabel("Intensity [a.u.]")
        _apply_compact_sci_axis(ax)

    def _plot_bar_all_replicates(self, ax, df_ratios: pd.DataFrame, ordered_states: list,
                                 state_labels: dict, color_map: dict):
        """Panel C: relative-abundance bar+strip plot pooling ALL curated replicates group-wide
        (not just the representative one) -- x-axis uses nice state labels in canonical order,
        bars/dots colored via color_map so this legend matches every other panel's colors. Bars
        show the mean relative area across replicates (error bars = sd); each individual
        replicate's own value is overlaid as a black dot.
        """
        order = [state_labels[s] for s in ordered_states]

        if df_ratios.empty:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            return

        plot_df = df_ratios.copy()
        plot_df['NiceLabel'] = plot_df['State'].map(state_labels).fillna(plot_df['State'])
        present_order = [lbl for lbl in order if lbl in set(plot_df['NiceLabel'])]
        palette = {lbl: color_map.get(lbl, 'grey') for lbl in present_order}

        sns.barplot(data=plot_df, x='NiceLabel', y='Ratio', hue='NiceLabel', palette=palette,
                   ax=ax, errorbar='sd', capsize=0.1, alpha=0.8, dodge=False, legend=False,
                   order=present_order)
        sns.stripplot(data=plot_df, x='NiceLabel', y='Ratio', color='black', ax=ax, jitter=True,
                     alpha=0.7, size=3, order=present_order)

        ax.set_ylabel("Normalized Area")
        ax.set_xlabel("")
        ax.set_ylim(0, 1.1)
        mpl.pyplot.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=5)

    def _build_si_simple_caption(self, appendix_name: str, figure_name: str, rep_exp, charge: int,
                                 ordered_states: list, state_labels: dict, ms2_states: list,
                                 detected_group_wide: set, stub_group_wide: set, color_map: dict,
                                 color_order: list, panel_letters: dict, footer=False) -> list:
        """Builds the dynamic, panel-by-panel SI figure caption for plot_si_report_simple() as a
        flat list of (word, color) tokens (see _words()/_render_justified_colored_caption()) --
        rather than a plain string -- so every mention of a modification-state name (e.g.
        "Unmodified") can be rendered in that state's own color, matching its color everywhere
        else in the figure, while the rest of the caption stays plain/default-colored.

        Panel sentences are built into `sentence_by_key` in whatever order is convenient, then
        emitted in the ACTUAL DRAWN LETTER ORDER (sorted by panel_letters' own values) rather than
        in build order -- this is what guarantees the caption always starts with panel A's own
        sentence and reads A, B, C, ... regardless of which layout ("alpha"/"beta") produced
        panel_letters, since panels are drawn (and therefore lettered) in different orders between
        them, and a fixed build order would silently drift out of sync with that (e.g. "bar" used
        to always be appended before "eic_rep"/"mz_change" even on layouts where it was lettered
        AFTER them).

        panel_letters: {panel_key: letter} exactly as assigned while the figure was drawn (keys:
        'tic', 'eic_overview', 'bar', 'eic_rep', 'mz_change', and 'mirror_<state_key>'/
        'seq_<state_key>' per state in ms2_states) -- a panel_key only gets a sentence if it's
        actually present, so the caption adapts to whichever layout/state-count produced it. Every
        panel reference is written "<bold letter>)" (e.g. "A)"), and the charge-state notation
        ([M+zH]^{z+}) is rendered as real mathtext with a proper superscript.
        """
        def _ref(key: str) -> str:
            return f"$\\mathbf{{{panel_letters[key]}}}$)"

        def _punct(tokens: list, mark: str) -> None:
            """Appends punctuation (',', ';', '.') directly onto the LAST existing token instead
            of as its own separate token -- a mark like ';' is not itself a word, so giving it a
            token of its own would render with an errant space before it ("peptide ;" instead of
            "peptide;"). No-op on an empty token list."""
            if tokens:
                word, color = tokens[-1]
                tokens[-1] = (word + mark, color)

        def _colored_state_clauses(states: list) -> list:
            """Tokens for '<state> peptide' clauses in sequence, joined by '; ', 'the <color>
            trace corresponds to the <State> peptide' -- each state's own label word(s) colored to
            match color_map."""
            tokens = []
            for i, s in enumerate(states):
                if i > 0:
                    _punct(tokens, ";")
                label = state_labels[s]
                tokens += _words(f"the {_state_color_name(label, color_order)} trace corresponds to the")
                tokens += _words(label, color=color_map[label])
                tokens += _words("peptide")
            return tokens

        def _colored_state_list(states: list) -> list:
            """Tokens for a plain comma-separated list of state names, each in its own color."""
            tokens = []
            for i, s in enumerate(states):
                if i > 0:
                    _punct(tokens, ",")
                label = state_labels[s]
                tokens += _words(label, color=color_map[label])
            return tokens

        charge_notation = f"$[M+{charge}H]^{{{charge}+}}$"
        detected_states_ordered = [s for s in ordered_states if s in detected_group_wide]
        stub_states_ordered = [s for s in ordered_states if s in stub_group_wide]

        # Bold "<appendix> <figure>" heading and italic "E. coli" species name -- both need their
        # own internal spaces escaped as "\ " (mathtext, like real TeX, otherwise collapses plain
        # spaces inside math mode), and BOTH "$...$" spans need their own closing "$" (previously
        # missing on both -- "$\mathbf{...}" and "$\mathtext{...}" ran straight into the following
        # plain text with no closing delimiter, and "\mathtext" isn't a real mathtext command to
        # begin with, e.g. \mathrm/\mathit/\mathbf/\mathtt are). _words() now also tokenizes via
        # _tokenize_preserving_math() specifically so a multi-word span like this survives as ONE
        # token instead of being torn into several unbalanced, unmatched-"$" fragments.
        appendix_math = str(appendix_name).replace(" ", "\\ ")
        figure_math = str(figure_name).replace(" ", "\\ ")

        # Plasmid codes are followed by the construct's own name in parentheses -- the maturase name
        # verbatim (e.g. "TJ-C-560 (HypX-HypY)"), the precursor name with its vector/promoter prefix
        # stripped so only the descriptive fusion/core part shows (e.g. "TJ-C-593
        # (Fh8-chaos-HypA1-HypA1)"), see _strip_vector_prefix(). A missing/empty name just omits the
        # parenthetical rather than printing "(None)".
        maturase_plasmid = getattr(rep_exp, 'maturase_plasmid', 'N/A')
        precursor_plasmid = getattr(rep_exp, 'precursor_plasmid', 'N/A')
        maturase_name = getattr(rep_exp, 'maturase_name', None)
        precursor_name = _strip_vector_prefix(getattr(rep_exp, 'precursor_name', None))
        maturase_bit = f"{maturase_plasmid} ({maturase_name})" if maturase_name else f"{maturase_plasmid}"
        precursor_bit = f"{precursor_plasmid} ({precursor_name})" if precursor_name else f"{precursor_plasmid}"

        runs = _words(
            f"$\\mathbf{{{appendix_math}\\ {figure_math}}}$: HPLC-MS analysis of IMAC purified "
            f"precursor peptides from $\\mathit{{E.\\ coli}}$ BL21(DE3)Tuner containing splicease "
            f"encoding plasmid "
            f"{maturase_bit} and precursor peptide encoding plasmid "
            f"{precursor_bit} after expression at "
            f"{getattr(rep_exp, 'expression_temperature', 'N/A')}°C for "
            f"{getattr(rep_exp, 'expression_time', 'N/A')} h in "
            f"{getattr(rep_exp, 'expression_scale', 'N/A')} mL cultures after digestion with "
            f"{getattr(rep_exp, 'digestion', 'N/A')}."
        )

        # Two fixed follow-on sentences, right after the digestion clause and before anything
        # panel-specific: the overall replicate count for this group (not just however many
        # happened to have a curated selection for this particular cut/charge/state), and a
        # one-line explainer for the "x" sequence-notation prefix used to mark spliced residues.
        n_replicates = len(self.group_results.replicates)
        runs += _words(f"Data from n={n_replicates} independent biological replicates.")
        runs += _words("Spliced amino acids have an x as a prefix.")

        # A dedicated, separate sentence (not folded into the intro paragraph's own wording) --
        # always right after the intro, always before any panel-by-panel description below. A
        # hard line break follows it (_line_break()) so the panel-by-panel breakdown always starts
        # on its own fresh line, visually demarcating it from the intro block above rather than
        # running straight on from wherever the automatic width-based wrapping happened to land.
        rep_panel_keys = [k for k in ('tic', 'eic_overview', 'eic_rep', 'mz_change') if k in panel_letters]
        if rep_panel_keys:
            rep_letters = ", ".join(f"$\\mathbf{{{panel_letters[k]}}}$" for k in rep_panel_keys)
            runs += _words(f"A representative experimental replicate is shown in panels {rep_letters}.")
            runs += _line_break()

        sentence_by_key = {}

        if 'tic' in panel_letters:
            sentence_by_key['tic'] = (
                _words(f"{_ref('tic')} Total ion Chromatogram (TIC) for representative replicate "
                      f"plotted in black lines.")
            )
        if 'eic_overview' in panel_letters:
            tokens = _words(f"{_ref('eic_overview')} Extracted ion chromatograms for the")
            tokens += _words(f"{charge_notation} charge state of the peptide fragment;")
            tokens += _colored_state_clauses(detected_states_ordered)
            _punct(tokens, ".")
            sentence_by_key['eic_overview'] = tokens
        if 'bar' in panel_letters:
            sentence_by_key['bar'] = _words(
                f"{_ref('bar')} Bar plot showing the relative abundance of the various peptide "
                f"states at charge z={charge}; the bars represent the average relative abundances "
                f"normalized to total peak areas, with each individual experimental replicate "
                f"value plotted as a black dot."
            )
        if 'eic_rep' in panel_letters:
            tokens = _words(f"{_ref('eic_rep')} Close-up plot of the representative extracted ion "
                            f"chromatogram (EIC) of the various peptide states;")
            tokens += _colored_state_clauses(detected_states_ordered)
            _punct(tokens, ".")
            sentence_by_key['eic_rep'] = tokens
        if 'mz_change' in panel_letters:
            tokens = _words(
                f"{_ref('mz_change')} Close-up plot of the m/z values of the peptides detected via "
                f"mass spec, with the shaded colored areas corresponding to where the expected m/z "
                f"values for the peptide m/z traces should occur, with the observed spectrum "
                f"plotted in black."
            )
            # Each clause (detected states / stub states) is only added at all if that list is
            # actually non-empty -- previously an empty stub list still produced a trailing "and
            # the no peptide(s) were not detected" clause, which is nonsensical (there's nothing
            # to report there) rather than just being omitted.
            clauses = []
            if detected_states_ordered:
                clause = _words("The" if not clauses else "the")
                clause += _colored_state_list(detected_states_ordered)
                clause += _words("peptide(s) were found in the mass spectrum")
                clauses.append(clause)
            if stub_states_ordered:
                clause = _words("The" if not clauses else "the")
                clause += _colored_state_list(stub_states_ordered)
                clause += _words("peptide(s) were not detected")
                clauses.append(clause)
            for i, clause in enumerate(clauses):
                if i > 0:
                    tokens += _words("and")
                tokens += clause
            if clauses:
                _punct(tokens, ".")
            sentence_by_key['mz_change'] = tokens

        for state_key in ms2_states:
            label = state_labels[state_key]
            color = color_map[label]
            mirror_key, seq_key = f'mirror_{state_key}', f'seq_{state_key}'
            if mirror_key in panel_letters:
                tokens = _words(f"{_ref(mirror_key)} Mirrored MS2 plot for the")
                tokens += _words(label, color=color)
                tokens += _words(
                    "peptide, with the matched fragment b ions shown in blue and y ions plotted "
                    "in red, the full fragmentation pattern is plotted in black and mirrored to "
                    "the matched fragments."
                )
                sentence_by_key[mirror_key] = tokens
            if seq_key in panel_letters:
                tokens = _words(f"{_ref(seq_key)} MS2 spectrum of the fragments corresponding to the")
                tokens += _words(label, color=color)
                tokens += _words(
                    "peptide sequence, with matched b (blue, above) and y ions (red, below) plotted."
                )
                sentence_by_key[seq_key] = tokens

        for key in sorted(sentence_by_key.keys(), key=lambda k: panel_letters[k]):
            runs += sentence_by_key[key]

        return runs

    def plot_si_report_simple(self, output_dir: Path, appendix_name: str, figure_name: str,
                              style: str = "white", layout: str = "alpha", cut_id=None,
                              charge: int = None, rt_range: tuple = None, dpi: int = 400,
                              page_number: int = None, source_path: Path = None,
                              state_color_order: list = None, break_axis: bool = False,
                              subpanel_labels: bool = False, caption_margin_cm: float = 1.5,
                              footer=False) -> tuple:
        """Generates a high-resolution A4 SI PDF in the "simple" layout: one representative
        replicate (see _find_representative_replicate()) drives the TIC/EIC-overview/zoomed-EIC/
        mz-change panels, while the relative-abundance bar plot and every MS2 panel pool every
        curated replicate group-wide.

        Normally ONE page -- but the plot grid gets a fixed target height per row rather than
        always stretching to fill the page (see content_bottom_p1 below), so a group with a lot of
        group-wide-detected modification states (more MS2 rows), or a caption that's grown very
        long (many states -> many panel sentences), can outgrow that budget. When that happens,
        this automatically spills the OVERFLOW -- trailing MS2 state rows and/or trailing caption
        text -- onto a second page within the SAME output PDF (via PdfPages), rather than either
        cramming everything unreadably small onto one page or silently overflowing the page
        bounds. Both pages get their own correct footer "Page N" (N and N+1), and whichever page
        doesn't have the full figure/caption gets a short cross-reference note saying so and
        naming the other page.

        Returns (output_path, pages_used) -- pages_used is 1 normally, 2 if either the plot grid
        or the caption had to continue onto a second page. `page_number` is treated as the
        STARTING page number for this figure; batch callers generating many groups' reports in
        sequence should advance their own running page counter by the returned `pages_used`, not
        always by 1, to keep a combined multi-group document's page numbers correct.

        Two selectable panel arrangements (6-column grid either way), via `layout`:

        layout="alpha" (default) -- 3 fixed header rows, then one row per group-wide-detected
        modification state:
            Row 1, cols 1-3:   TIC for the representative replicate                    (panel A)
            Row 2, cols 1-3:   full-range EIC overview for the representative replicate (panel B)
            Rows 1-2, cols 4-5: zoomed EIC overlay for the representative replicate     (panel C)
            Rows 1-2, col 6:   relative-abundance bar plot, all replicates              (panel D)
            Row 3, cols 1-6:   mz-change panel for the representative replicate, full width (E)
            Row 4+, cols 1-3:  MS2 mirror plot, all replicates, one row per state       (F, H, ...)
            Row 4+, cols 4-6:  MS2 sequence plot, all replicates, one row per state     (G, I, ...)

        layout="beta" -- a more compact 2 fixed header rows (no TIC/EIC-overview panels at all),
        then one row per group-wide-detected modification state:
            Rows 1-2, cols 1-2: zoomed EIC overlay for the representative replicate     (panel A)
            Rows 1-2, cols 3-5: mz-change panel for the representative replicate        (panel B)
            Rows 1-2, col 6:    relative-abundance bar plot, all replicates             (panel C)
            Row 3+, cols 1-3:   MS2 mirror plot, all replicates, one row per state       (D, F, ...)
            Row 3+, cols 4-6:   MS2 sequence plot, all replicates, one row per state     (E, G, ...)
        The fixed header panels never move to page 2 -- only trailing MS2 state rows can.

        Every panel is lettered A, B, C, ... just OUTSIDE its own top-left corner as it's drawn
        (never overlapping the plotted data, and the SAME letter sequence continues across both
        pages if a second one is needed), and the figure caption (_build_si_simple_caption()) is
        assembled to match that exact lettering AFTER both pages' panels are drawn -- it adapts
        automatically to whichever layout produced it and to however many states got MS2 rows,
        since it only ever describes panels that actually got drawn.

        appendix_name/figure_name: caption prefix, e.g. plot_si_report_simple(..., "SI Appendix A",
        "Figure 3") -> captions starting "SI Appendix A Figure 3: ...".
        style: passed straight to seaborn's sns.set_style() (e.g. "white", "ticks", "darkgrid").
        cut_id/charge: which (fragment_cut, charge) to report on -- defaults to
        select_best_cut_charge()'s pick (fewest stub replicates first, then area, then quality) if
        either is left as None.
        rt_range: optional (min, max) minutes to cap the TIC/EIC-overview panels to (layout="alpha"
        only -- beta has no such panels) -- does not affect the zoomed EIC panel, which is already
        windowed to each peak.
        state_color_order: overrides DEFAULT_SI_STATE_COLOR_ORDER for this call -- the canonical
        nice-label order that fixes each state's color (and its left-to-right/top-to-bottom
        position in every panel) consistently across every page in a run.
        break_axis: passed through to _plot_mz_change_plot_representative() (see its docstring) --
        off by default (one continuous shared mz axis for the mz-change panel).
        subpanel_labels: if True, individual panels also get their own descriptive title (e.g. the
        MS2 mirror panel's "Mirror plot of ms2 fragments matching the peptide") -- off by default,
        since every panel is already identified by its corner letter and the caption below.
        caption_margin_cm: left/right margin (cm) the caption text is wrapped and JUSTIFIED to
        (both edges flush) -- see _wrap_and_justify_to_margin()'s docstring for why this is real
        space-padding justification rather than requiring a LaTeX install.
        page_number/source_path: as in generate_si_report() -- printed in the footer and folded
        into the filename/page numbering when batch-generating many groups.
        footer: off by default.
        """
        if layout not in ("alpha", "beta"):
            raise ValueError(f"Unknown layout {layout!r} -- expected 'alpha' or 'beta'.")

        if self.group_results is None:
            raise ValueError(
                "plot_si_report_simple() requires a ReplicateGroupResults -- build this "
                "GroupedExperiments via GroupedExperiments.from_replicate_group_results(...)."
            )

        if cut_id is None or charge is None:
            auto_cut, auto_charge = self.select_best_cut_charge()
            cut_id = cut_id if cut_id is not None else auto_cut
            charge = charge if charge is not None else auto_charge
        if cut_id is None:
            print("No curated peaks found for this group -- skipping simple SI report.")
            return None, 0

        target_dict = self._get_theoretical_targets_dict(cut_id)
        all_possible_states = sorted(target_dict.keys())
        state_labels = {sk: format_modification_label(target_dict[sk].modifications) for sk in all_possible_states}
        ordered_states = self._order_states_canonically(all_possible_states, state_labels, state_color_order)
        color_map = get_state_color_map([state_labels[s] for s in ordered_states], state_color_order)

        # Group-wide (every replicate, not just the representative one) curated peaks per state --
        # drives the bar plot and decides which states get MS2 rows.
        isoform_peaks_by_state = defaultdict(list)
        final_peaks_map = defaultdict(dict)
        for state_key in ordered_states:
            target_obj = target_dict[state_key]
            modification = str(target_obj.modifications)
            for rep_id, rep_results in self.group_results.replicates.items():
                exp_id = rep_results.experiment.spectrum_id
                peak = self._find_selected_peak(rep_id, cut_id, modification, charge,
                                                sequence=target_obj.sequence)
                if peak is None or getattr(peak, "tier", None) == "Reject":
                    continue
                peak.quality_metrics["is_selected"] = True
                peak.quality_metrics["group_pass"] = True
                if not hasattr(peak, "target"):
                    peak.target = peak
                isoform_peaks_by_state[state_key].append(peak)
                final_peaks_map[exp_id][state_key] = (peak, True, True)

        detected_group_wide = {s for s in ordered_states if isoform_peaks_by_state.get(s)}
        stub_group_wide = set(ordered_states) - detected_group_wide
        ms2_states = [s for s in ordered_states if s in detected_group_wide]
        df_ratios = self._get_ratios_dataframe(final_peaks_map, all_possible_states)

        rep_id, representative_peaks = self._find_representative_replicate(cut_id, charge)
        if rep_id is None:
            print("No curated peaks found for this group -- skipping simple SI report.")
            return None, 0
        rep_exp = self.group_results.replicates[rep_id].experiment
        rep_spectrum = MassSpectrum(mzml_path=rep_exp.mzml_path, output_path=rep_exp.output_path,
                                    spectrum_id=rep_exp.spectrum_id)
        rep_spectrum.load()

        output_dir = Path(output_dir)
        os.makedirs(to_long_path(output_dir), exist_ok=True)
        safe_cut = str(cut_id).replace(" ", "_").replace("/", "-")
        safe_appendix = str(appendix_name).replace(" ", "_")
        safe_figure = str(figure_name).replace(" ", "_")
        page_prefix = f"{page_number:03d}_" if page_number is not None else ""
        # One filename/one file regardless of how many pages it ends up needing -- tagged with the
        # STARTING page number, since the whole point is this is still "one figure".
        filename = (f"{page_prefix}{self.group_results.group_name}_{safe_appendix}_{safe_figure}"
                   f"_{layout}_Cut_{safe_cut}_z{charge}.pdf")
        output_path = output_dir / filename

        sns.set_style(style)
        mpl.pyplot.rcParams.update({'font.size': 6})

        # A4 portrait, fixed regardless of content -- see generate_si_report()'s docstring for why
        # constrained_layout is deliberately avoided in favor of these manual margins.
        PAGE_W_IN, PAGE_H_IN = 8.2677, 11.6929
        # A per-row height BUDGET (inches) rather than always stretching the grid to fill whatever
        # space is nominally available -- this is what makes the figure "a bit smaller" and hands
        # the freed-up space to the caption automatically (fewer rows -> a smaller natural grid ->
        # more room left over below it), instead of a plot grid that's always exactly as tall as
        # the page regardless of how little content it actually holds.
        UNIT_ROW_IN = 1.3
        # MS2 rows get MORE than one plain unit each -- they carry 3-tier rotated ion labels
        # (name / m/z / ppm+n, stacked outward from the sequence) that need real vertical room to
        # stay legible and non-overlapping, unlike the compact header panels. Giving them their
        # own taller ratio (rather than reusing UNIT_ROW_IN's plain 1.0 for every row) keeps the
        # rest of the page compact while still leaving these specific panels enough height.
        MS2_ROW_HEIGHT_RATIO = 1.4
        content_top = 0.97
        # The FLOOR the grid is allowed to shrink content_bottom down to before a state count is
        # considered "too many for one page" and starts spilling MS2 rows onto a second page.
        min_content_bottom = 0.26
        # Vertical gap (figure fraction) between the bottom of the plot grid and the top of the
        # caption -- widened so the caption reads as clearly separated from the figure rather than
        # crowding right up under the last panel row.
        caption_gap = 0.055
        footer_y = 0.012
        margin_frac_left = (caption_margin_cm / 2.54) / PAGE_W_IN

        if layout == "alpha":
            header_ratio_sum, header_rows, header_height_ratios = 1.3 + 1.3 + 1.6, 3, [1.3, 1.3, 1.6]
        else:  # "beta"
            header_ratio_sum, header_rows, header_height_ratios = 1.6 + 1.6, 2, [1.6, 1.6]

        # Per-state MS2 row height: a state whose representative peptide is long enough to wrap its
        # sequence ladder (panels E/G) onto several stacked lines needs proportionally more vertical
        # room, so its grid row is grown by the wrap-line count. Computed up front, via the SAME
        # residues-per-line the drawing uses (passed down to _plot_si_sequence_annotations so the
        # reserved height and the drawn line count can't disagree). One-line states are unchanged.
        seq_panel_w_in = (0.98 - 0.05) * 0.5 * PAGE_W_IN   # sequence panel = cols 3:6 (~half width)
        ms2_per_line = _seq_residues_per_line(seq_panel_w_in)

        def _state_seq_len(state_key):
            peaks = isoform_peaks_by_state.get(state_key, [])
            if not peaks:
                return 0
            passed = [p for p in peaks if p.quality_metrics.get('overall_pass', False)]
            rep_peak = passed[0] if passed else peaks[0]
            seq = getattr(getattr(rep_peak, 'target', rep_peak), 'sequence', '') or ''
            if not seq:
                return 0
            try:
                return len(pyteomics.parser.parse(seq))
            except Exception:
                return len(seq)

        ms2_wrap_lines = {s: max(1, int(np.ceil(_state_seq_len(s) / ms2_per_line))) for s in ms2_states}
        ms2_row_ratio = {s: MS2_ROW_HEIGHT_RATIO * ms2_wrap_lines[s] for s in ms2_states}

        # Paginate the MS2 rows across as MANY pages as needed (not a hard 2-page cap). The first
        # page also carries the fixed header panels, so it has less room for rows; continuation
        # pages are rows-only and fit more. Each page is filled with whole state rows (a row is never
        # split across a page break) down to min_content_bottom. This is what stops many/tall
        # (wrapped) rows from overflowing a single "page 2" and leaving the caption no room.
        content_top_cont = 0.92   # top of a rows-only continuation page (band above = "continued" note)
        rows_budget_p1 = max((content_top - min_content_bottom) * PAGE_H_IN / UNIT_ROW_IN - header_ratio_sum, 0.0)
        rows_budget_cont = max((content_top_cont - min_content_bottom) * PAGE_H_IN / UNIT_ROW_IN, 0.0)

        figure_page_states, _cur, _used, _first = [], [], 0.0, True
        for s in ms2_states:
            budget = rows_budget_p1 if _first else rows_budget_cont
            r = ms2_row_ratio[s]
            if _cur and _used + r > budget:            # start a new page (keep this row whole)
                figure_page_states.append(_cur)
                _cur, _used, _first = [], 0.0, False
            _cur.append(s)
            _used += r
        figure_page_states.append(_cur)   # always >= 1 page (the header page, even with no MS2 rows)

        panel_letters = {}
        letter_iter = iter(string.ascii_uppercase)

        def _next_letter(target_ax, key):
            letter = next(letter_iter)
            panel_letters[key] = letter
            # Placed OUTSIDE the axes' own bounds (negative x, y > 1 in axes-fraction,
            # clip_on=False) -- a real SI-figure panel label sits just above/left of its panel's
            # box, never floating inside the plotted data.
            target_ax.text(-0.04, 1.05, letter, transform=target_ax.transAxes, fontsize=9,
                          weight='bold', va='top', ha='left', clip_on=False, zorder=10)
            return letter

        def _draw_footer(target_fig, this_page_number):
            if not footer:
                return
            generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
            if source_path is not None:
                source_name = Path(source_path).name
                _suffix = "_manually_reviewed.pkl.gz"
                if source_name.endswith(_suffix):
                    source_name = source_name[: -len(_suffix)]
            else:
                source_name = self.group_results.group_name
            footer_left = (f"{source_name}   |   Representative replicate: {rep_exp.spectrum_id}   |   "
                          f"Generated: {generated_at}")
            footer_right = f"Page {this_page_number}" if this_page_number is not None else ""
            target_fig.text(0.03, footer_y, footer_left, ha='left', va='bottom', fontsize=6)
            target_fig.text(0.97, footer_y, footer_right, ha='right', va='bottom', fontsize=8, weight='bold')

        # --- Panel-drawing helpers, shared across every figure page. ---
        def _draw_ms2_row(target_fig, gs_obj, row_idx, state_key):
            isoform_peaks = isoform_peaks_by_state[state_key]
            label = state_labels[state_key]
            ax_mirror = target_fig.add_subplot(gs_obj[row_idx, 0:3])
            self._plot_ms2_mirror(ax_mirror, isoform_peaks, show_title=subpanel_labels)
            _next_letter(ax_mirror, f'mirror_{state_key}')
            ax_seq = target_fig.add_subplot(gs_obj[row_idx, 3:6])
            b_ions, y_ions = self._compute_ms2_matches(isoform_peaks)
            passed_isoform_peaks = [p for p in isoform_peaks if p.quality_metrics.get('overall_pass', False)]
            rep_peak_for_seq = passed_isoform_peaks[0] if passed_isoform_peaks else isoform_peaks[0]
            # residues_per_line is the SAME value the row-height reservation above used, so a long
            # sequence wraps onto exactly the number of lines this row was sized for.
            self._plot_si_sequence_annotations(ax_seq, rep_peak_for_seq, b_ions, y_ions, label,
                                               residues_per_line=ms2_per_line)
            _next_letter(ax_seq, f'seq_{state_key}')
            return ax_mirror, ax_seq

        def _draw_header(target_fig, gs_obj):
            """Draws the fixed header panels (layout-dependent) on the first page; returns the
            left-column intensity axes for y-label alignment."""
            if layout == "alpha":
                ax_tic = target_fig.add_subplot(gs_obj[0, 0:3])
                self._plot_tic_representative(ax_tic, rep_spectrum, rt_range=rt_range)
                mpl.pyplot.setp(ax_tic.get_xticklabels(), visible=False)
                _next_letter(ax_tic, 'tic')
                ax_eic_overview = target_fig.add_subplot(gs_obj[1, 0:3], sharex=ax_tic)
                self._plot_eic_overview_representative(ax_eic_overview, ordered_states, state_labels,
                                                       representative_peaks, color_map, rt_range=rt_range)
                _next_letter(ax_eic_overview, 'eic_overview')
                ax_eic_rep = target_fig.add_subplot(gs_obj[0:2, 3:5])
                self._plot_eic_representative(ax_eic_rep, ordered_states, state_labels,
                                              representative_peaks, color_map, charge)
                _next_letter(ax_eic_rep, 'eic_rep')
                ax_bar = target_fig.add_subplot(gs_obj[0:2, 5:6])
                self._plot_bar_all_replicates(ax_bar, df_ratios, ordered_states, state_labels, color_map)
                _next_letter(ax_bar, 'bar')
                ax_mz_change = self._plot_mz_change_plot_representative(
                    target_fig, gs_obj[2, 0:6], ordered_states, state_labels, representative_peaks,
                    target_dict, charge, color_map, break_axis=break_axis)
                _next_letter(ax_mz_change, 'mz_change')
                return [ax_tic, ax_eic_overview]
            # layout == "beta"
            ax_eic_rep = target_fig.add_subplot(gs_obj[0:2, 0:2])
            self._plot_eic_representative(ax_eic_rep, ordered_states, state_labels,
                                          representative_peaks, color_map, charge)
            _next_letter(ax_eic_rep, 'eic_rep')
            ax_mz_change = self._plot_mz_change_plot_representative(
                target_fig, gs_obj[0:2, 2:5], ordered_states, state_labels, representative_peaks,
                target_dict, charge, color_map, break_axis=break_axis)
            _next_letter(ax_mz_change, 'mz_change')
            ax_bar = target_fig.add_subplot(gs_obj[0:2, 5:6])
            self._plot_bar_all_replicates(ax_bar, df_ratios, ordered_states, state_labels, color_map)
            _next_letter(ax_bar, 'bar')
            return [ax_eic_rep]

        # --- Draw every figure page: the header on page 1, then the MS2 rows spread across pages per
        # figure_page_states. figure_pages holds (fig, content_bottom) in order; the last page's
        # content_bottom is where the caption begins. ---
        figure_pages = []
        for pidx, states in enumerate(figure_page_states):
            is_first = (pidx == 0)
            fig = mpl.pyplot.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
            if is_first:
                top = content_top
                h_ratios = header_height_ratios + [ms2_row_ratio[s] for s in states]
                used_units = header_ratio_sum + sum(ms2_row_ratio[s] for s in states)
                n_rows = header_rows + len(states)
            else:
                top = content_top_cont
                h_ratios = [ms2_row_ratio[s] for s in states]
                used_units = sum(ms2_row_ratio[s] for s in states)
                n_rows = len(states)
            bottom = top - used_units * UNIT_ROW_IN / PAGE_H_IN
            gs = mpl.gridspec.GridSpec(max(n_rows, 1), 6, figure=fig, height_ratios=(h_ratios or [1.0]),
                                       left=0.05, right=0.98, top=top, bottom=bottom,
                                       hspace=0.85, wspace=0.5)
            left_axes = []
            if is_first:
                left_axes += _draw_header(fig, gs)
                row0 = header_rows
            else:
                fig.text(margin_frac_left, 0.965,
                         f"{appendix_name} {figure_name} (figure continued from the previous page).",
                         ha='left', va='top', fontsize=7, style='italic', color='dimgray')
                row0 = 0
            for i, state_key in enumerate(states):
                ax_mirror, _ax_seq = _draw_ms2_row(fig, gs, row0 + i, state_key)
                left_axes.append(ax_mirror)
            if len(left_axes) > 1:
                fig.align_ylabels(left_axes)
            figure_pages.append((fig, bottom))

        # --- Caption: built AFTER every panel is drawn/lettered (so panel_letters is complete), with
        # real per-word justification and each modification-state name colored to match the figure.
        # It starts in the room left under the LAST figure page and spills onto additional
        # caption-only pages if it doesn't fit -- so the caption ALWAYS has room, no matter how many
        # pages the figure itself needed (previously it could be squeezed off a full page 2). ---
        caption_tokens = self._build_si_simple_caption(
            appendix_name, figure_name, rep_exp, charge, ordered_states, state_labels, ms2_states,
            detected_group_wide, stub_group_wide, color_map, state_color_order, panel_letters,
        )
        caption_lines, caption_no_justify = _break_caption_into_lines(
            caption_tokens, fontsize=7.5, fig_width_in=PAGE_W_IN, margin_cm=caption_margin_cm,
        )
        line_height_frac = _caption_line_height_frac(7.5, PAGE_H_IN, linespacing=1.4)

        def _caption_capacity(y_start):
            return max(int((y_start - footer_y - 0.02) / line_height_frac), 1)

        def _new_caption_page(continued):
            fig = mpl.pyplot.figure(figsize=(PAGE_W_IN, PAGE_H_IN))
            note = (f"{appendix_name} {figure_name}, caption continued from the previous page."
                    if continued else f"{appendix_name} {figure_name}, caption.")
            fig.text(margin_frac_left, 0.965, note, ha='left', va='top', fontsize=7,
                     style='italic', color='dimgray')
            return fig

        caption_pages = []          # extra pages created solely for caption overflow
        last_fig, last_bottom = figure_pages[-1]
        cur_fig = last_fig
        y_start = last_bottom - caption_gap
        # If the last figure page is essentially full (< 2 caption lines of room left), start the
        # caption on its own fresh page rather than cramming a stray line at the very bottom.
        if caption_lines and _caption_capacity(y_start) < 2:
            cur_fig.text(margin_frac_left, footer_y + 0.025, "(Caption on the following page.)",
                         ha='left', va='bottom', fontsize=7, style='italic', color='dimgray')
            cur_fig = _new_caption_page(continued=False)
            caption_pages.append(cur_fig)
            y_start = 0.93

        idx, n = 0, len(caption_lines)
        while idx < n:
            cap = _caption_capacity(y_start)
            remaining = n - idx
            take = remaining if remaining <= cap else max(cap - 1, 1)   # leave a line for the note
            chunk = caption_lines[idx:idx + take]
            nj = {i - idx for i in caption_no_justify if idx <= i < idx + take}
            _render_caption_lines(cur_fig, chunk, x_frac=margin_frac_left, y_frac=y_start,
                                  fontsize=7.5, fig_width_in=PAGE_W_IN, fig_height_in=PAGE_H_IN,
                                  margin_cm=caption_margin_cm, linespacing=1.4,
                                  justify_last_line=(idx + take < n), no_justify_indices=nj)
            idx += take
            if idx < n:
                cur_fig.text(margin_frac_left, footer_y + 0.025, "(Caption continued on the next page.)",
                             ha='left', va='bottom', fontsize=7, style='italic', color='dimgray')
                cur_fig = _new_caption_page(continued=True)
                caption_pages.append(cur_fig)
                y_start = 0.93

        # --- Save: every figure page, then every caption-overflow page, in order, into one PDF. ---
        all_figs = [f for f, _ in figure_pages] + caption_pages
        pages_used = len(all_figs)
        for k, fig in enumerate(all_figs):
            _draw_footer(fig, (page_number + k) if page_number is not None else None)
        with PdfPages(to_long_path(output_path)) as pdf:
            for fig in all_figs:
                pdf.savefig(fig, dpi=dpi)
        for fig in all_figs:
            mpl.pyplot.close(fig)

        return output_path, pages_used

    def generate_all_reports(self, output_dir: Path, plot_mz: bool = True, plot_ms2: bool = False):
        output_dir = Path(output_dir)
        os.makedirs(to_long_path(output_dir), exist_ok=True)

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

                    # Iterate over the PEAK objects (the values), not the TARGET objects (the keys).
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
                    # Access the .rt attribute directly from the PeptidePeak object.
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
                    # Safely calculate mean, defaulting to 0.0 if no scored peaks are found to prevent NaN.
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

                            # Downstream plotting code expects every entry to expose `.target`
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

        with open(to_long_path(output_dir / "plotting_data.pkl"), "wb") as fp:
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
                _apex_win = getattr(peak, 'raw_apex_scans', None)
                if _apex_win is None or _apex_win.empty:
                    _apex_win = peak.raw_apex
                if _apex_win is not None and not _apex_win.empty and 'time' in _apex_win.columns:
                    ax_tic.axvspan(_apex_win['time'].min(), _apex_win['time'].max(), alpha=0.1, color=color)

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

                # Safely define rep_peak for the MS2 / Sequence plots to use.
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
        mpl.pyplot.savefig(to_long_path(output_path), dpi=dpi, bbox_inches='tight')
        mpl.pyplot.close(fig)
