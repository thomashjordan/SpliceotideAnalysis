"""Core automated peak-picking pipeline (driven by peak_picking_pipeline.py).

For each experimental group: extract per-replicate EICs for every target/charge, find candidate
peaks, cluster them across replicates with DBSCAN, build and score PeptidePeak objects, add empty
"stub" peaks where a replicate shows nothing, and return a ReplicateGroupResults cohort.
``pipeline_find_and_score_peak`` re-runs the single-peak scoring for a user-drawn RT window and is
used by the interactive review dashboard.
"""
import sys
import os
import pickle
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage
from sklearn.cluster import DBSCAN
from pyteomics import parser as peptide_parser

from modification_dictionary import aa_comp_custom
from analysis_containers import AnalysisResults, ReplicateGroupResults
from classes_spectrometry import MSExperiment, MassSpectrum
from peptide_classes import TargetPeptide, PeptideFragment, PeptidePeak


class NoRawApexError(ValueError):
    """Raised when a requested RT window holds no raw MS1 signal at the requested charge, so no
    meaningful PeptidePeak can be built from it."""


def extract_eic_matrices(spectra: list, target_peptide, mz_range: list, rt_range: list, filtered: bool) -> list:
    """Extracts and isolates Extracted Ion Chromatograms (EICs) across all samples

    without dropping channels. Optimized for speed and safety.
    """
    replicate_chromatograms = []

    # 1. Safely parse isotopologue names and masses from target_peptide
    isotopologues = getattr(target_peptide, 'isotopologues', None)
    if isinstance(isotopologues, pd.DataFrame):
        iso_names = isotopologues["name"].tolist()
        iso_masses = isotopologues["mass"].tolist()
    elif isinstance(isotopologues, dict):
        iso_names = isotopologues.get("name", [])
        iso_masses = isotopologues.get("mass", [])
    else:
        iso_names = ["Monoisotopic"]
        iso_masses = [getattr(target_peptide, 'monoisotopic_mass', getattr(target_peptide, 'mass', 0.0))]

    # Pre-compute fallback m/z window once outside the loop
    fallback_mz_range = [mz_range[0] - 5, mz_range[1] + 5] if mz_range else None

    for idx, spectrum in enumerate(spectra):
        all_traces = []

        for name, mass in zip(iso_names, iso_masses):
            # Primary EIC extraction call
            chromatogram = spectrum.get_extracted_compound_chromatogram(
                mass, filtered=filtered, tolerance_ppm=10, mz_range=mz_range, rt_range=rt_range
            )

            # Fallback extraction with expanded m/z range if primary came back empty
            if (chromatogram is None or chromatogram.empty) and fallback_mz_range:
                chromatogram = spectrum.get_extracted_compound_chromatogram(
                    mass, filtered=filtered, tolerance_ppm=10,
                    mz_range=fallback_mz_range, rt_range=rt_range
                )

            # Process and annotate trace if valid data was found
            if chromatogram is not None and not chromatogram.empty:
                # Use shallow copy to eliminate deep-copy memory allocation overhead
                df_trace = chromatogram.copy(deep=False)
                df_trace['isotopologue'] = name
                # The 'charge' column is already present in the 'chromatogram' DataFrame
                # returned by get_extracted_compound_chromatogram. No need to set it manually.
                df_trace['rep_idx'] = idx  # Preserve underlying sample replicate index
                all_traces.append(df_trace)

        # 2. Combine traces for the current replicate
        if not all_traces:
            replicate_chromatograms.append(pd.DataFrame())
        elif len(all_traces) == 1:
            # Fast path: bypass pd.concat overhead when only 1 isotopologue trace exists
            df_single = all_traces[0]
            if 'time' in df_single.columns and not df_single['time'].is_monotonic_increasing:
                df_single = df_single.sort_values(by=['time'])
            replicate_chromatograms.append(df_single)
        else:
            # Fast path: concat with ignore_index=True and sort in-place
            combined_df = pd.concat(all_traces, ignore_index=True)
            if 'time' in combined_df.columns:
                combined_df.sort_values(by=['time'], inplace=True)
            replicate_chromatograms.append(combined_df)

    return replicate_chromatograms


def find_peaks_single_replicate_overlap(interp_df: pd.DataFrame, iso_names: list,
                                        pt_widths: list, pt_distance: int,
                                        min_channels: int = 3) -> list:
    """Finds peaks within a single replicate by treating the summed chromatogram

    as an anchor and confirming positional interval intersection with isotopologues.
    """
    # 1. Extract the master anchor peaks from the Summed Chromatogram
    summed_signal = interp_df.sum(axis=1).to_numpy()
    if summed_signal.max() <= 0:
        return []

    prominence_floor = max(summed_signal.max() * 0.05, 1000)
    summed_indices, summed_props = signal.find_peaks(
        summed_signal, rel_height=0.90, height=3000,
        width=pt_widths, distance=pt_distance, prominence=prominence_floor
    )

    # 2. Extract candidate peaks from individual isotopologues
    iso_peaks = {iso: [] for iso in iso_names}
    for iso in iso_names:
        if iso not in interp_df.columns or interp_df[iso].sum() <= 0:
            continue
        current_signal = interp_df[iso].to_numpy()
        indices, props = signal.find_peaks(
            current_signal, rel_height=0.90, height=3000,
            width=pt_widths, distance=pt_distance, prominence=prominence_floor
        )
        for p_idx, pt_index in enumerate(indices):
            iso_peaks[iso].append({
                "rt": float(interp_df.index[pt_index]),
                "start": float(props["left_ips"][p_idx]) if "left_ips" in props else pt_index,
                "end": float(props["right_ips"][p_idx]) if "right_ips" in props else pt_index,
            })

    validated_peaks = []

    # 3. Intersection Cross-Verification Loop
    for s_idx, s_pt in enumerate(summed_indices):
        s_rt = float(interp_df.index[s_pt])
        s_start = float(summed_props["left_ips"][s_idx]) if "left_ips" in summed_props else s_pt
        s_end = float(summed_props["right_ips"][s_idx]) if "right_ips" in summed_props else s_pt

        matching_channels = ['summed']

        for iso, peaks in iso_peaks.items():
            for p in peaks:
                # Traditional interval overlap formula: Max of starts <= Min of ends
                overlap = max(s_start, p['start']) <= min(s_end, p['end'])
                if overlap:
                    matching_channels.append(iso)
                    break  # Move to the next isotopologue track

        # GATE: If enough channels overlap with this master window, it's real!
        if len(matching_channels) >= min_channels:
            validated_peaks.append({
                "rt": s_rt,
                "left_ips": s_start,
                "right_ips": s_end,
                "coinciding_tracks": matching_channels
            })

    return validated_peaks


def process_and_group_all_replicates(replicate_chromatograms: list, target_peptide,
                                     interpolation_distance: float, min_peak_width: float,
                                     max_peak_width: float, distance_peaks: float,
                                     min_replicates: int) -> dict:
    """Mother Function: Generates clean intra-replicate profiles and links them
    across sample files using cross-replicate DBSCAN clustering alignment.
    """
    iso_names = target_peptide.isotopologues["name"]

    pt_widths = [round(min_peak_width / interpolation_distance), round(max_peak_width / interpolation_distance)]
    pt_distance = round(distance_peaks / interpolation_distance)

    all_validated_features = []

    for rep_list_idx, total_chromatogram in enumerate(replicate_chromatograms):
        if total_chromatogram.empty:
            continue

        for charge in sorted(total_chromatogram['charge'].unique()):
            charge_chromatogram = total_chromatogram[total_chromatogram['charge'] == charge]
            if len(charge_chromatogram) < 10:
                continue

            # Extract true physical replicate index before dropping metadata column via pivot
            real_rep_idx = int(charge_chromatogram['rep_idx'].iloc[0])

            pivot_df = charge_chromatogram.pivot_table(
                index='time', columns='isotopologue', values='intensity_array', aggfunc='sum', fill_value=0
            ).sort_index()

            time_min, time_max = pivot_df.index.min(), pivot_df.index.max()
            if time_max <= time_min:
                continue

            new_time_points = np.arange(time_min, time_max, interpolation_distance)
            interp_df = pd.DataFrame(index=new_time_points)

            for iso in iso_names:
                if iso in pivot_df.columns and pivot_df[iso].sum() > 0:
                    old_timepoints = pivot_df.index.to_numpy()
                    old_intensities = pivot_df[iso].to_numpy()
                    new_intensities = np.interp(new_time_points, old_timepoints, old_intensities, left=0, right=0)
                    interp_df[iso] = ndimage.gaussian_filter1d(new_intensities, sigma=1)
                else:
                    interp_df[iso] = 0.0

            single_rep_peaks = find_peaks_single_replicate_overlap(
                interp_df=interp_df, iso_names=iso_names,
                pt_widths=pt_widths, pt_distance=pt_distance, min_channels=3
            )

            max_idx = len(interp_df.index) - 1

            for peak in single_rep_peaks:
                # Convert integer position indices (ips) to actual retention time values
                # before passing them out of this function's scope.
                rt_start_index = np.clip(int(round(peak["left_ips"])), 0, max_idx)
                rt_end_index = np.clip(int(round(peak["right_ips"])), 0, max_idx)

                # --- NEW LOGIC TO CONSTRAIN PEAK WIDTH ---
                # Get the original boundaries from scipy.find_peaks
                rt_start_original = interp_df.index[rt_start_index]
                rt_end_original = interp_df.index[rt_end_index]

                # Define a maximum allowed half-width (e.g., 0.1 min for a total of 0.2 min)
                max_half_width = 0.1
                peak_center_rt = peak["rt"]

                # Constrain the boundaries: they cannot exceed the max width around the center,
                # but can be narrower if find_peaks detected a narrower peak.
                final_rt_start = max(rt_start_original, peak_center_rt - max_half_width)
                final_rt_end = min(rt_end_original, peak_center_rt + max_half_width)

                all_validated_features.append({
                    "rep_idx": real_rep_idx,  # Pass true array coordinate onward
                    "charge": int(charge),
                    "rt": peak["rt"],
                    "rt_start": final_rt_start,
                    "rt_end": final_rt_end,
                    "eic_intp": interp_df,
                    "iso": "summed"
                })

    grouped_clusters = {}
    if not all_validated_features:
        return grouped_clusters

    df_feats = pd.DataFrame(all_validated_features)

    for charge in df_feats['charge'].unique():
        charge_feats = df_feats[df_feats['charge'] == charge].copy()

        if charge_feats['rep_idx'].nunique() <= 1:
            for idx, (_, row) in enumerate(charge_feats.iterrows()):
                grouped_clusters[f"charge_{charge}_single_{idx}"] = [row.to_dict()]
            continue

        coordinates = charge_feats['rt'].to_numpy().reshape(-1, 1)
        db = DBSCAN(eps=distance_peaks, min_samples=min_replicates).fit(coordinates)
        labels = db.labels_

        for label in set(labels):
            cluster_rows = charge_feats[labels == label]

            if label == -1:
                for n_idx, (_, row) in enumerate(cluster_rows.iterrows()):
                    grouped_clusters[f"charge_{charge}_noise_{n_idx}"] = [row.to_dict()]
                continue

            if cluster_rows['rep_idx'].nunique() >= min_replicates:
                cluster_id = f"charge_{charge}_cluster_{label}"
                grouped_clusters[cluster_id] = cluster_rows.to_dict(orient='records')
            else:
                # Added enumeration identifier to prevent name collisions during mapping
                for fb_idx, (_, row) in enumerate(cluster_rows.iterrows()):
                    grouped_clusters[f"charge_{charge}_noise_fallback_{row['rep_idx']}_{fb_idx}"] = [row.to_dict()]

    return grouped_clusters


def construct_peak_objects(grouped_clusters: dict, spectra: list, experiments: list,
                           target_peptide, mz_boundaries: float, rt_window: float = 0.2) -> list:
    """Builds and refines PeptidePeak objects from clustered features."""
    processed_peaks = []

    for cluster_id, items in grouped_clusters.items():
        df_items = pd.DataFrame(items)
        charge = int(df_items['charge'].iloc[0])

        # Calculate consensus boundaries across all replicates in the cluster
        global_rts = df_items['rt'].to_numpy()
        global_center = float(np.median(global_rts))

        # Determine the unique replicates contributing to this cluster
        for rep_idx in df_items['rep_idx'].unique():
            rep_subset = df_items[df_items['rep_idx'] == rep_idx]
            spectrum = spectra[rep_idx]
            experiment = experiments[rep_idx]

            interp_df = rep_subset['eic_intp']
            max_idx = len(interp_df.index) - 1

            # Convert point coordinates back into retention times
            left_bounds = [interp_df.index[np.clip(int(round(x)), 0, max_idx)] for x in rep_subset['left_ips']]
            right_bounds = [interp_df.index[np.clip(int(round(x)), 0, max_idx)] for x in rep_subset['right_ips']]

            # Optimization: RT windows dynamically adjust bounds relative to global variance if needed
            dynamic_window = max(rt_window, np.std(global_rts) * 3 if len(global_rts) > 1 else rt_window)
            rt_start = max(float(np.median(left_bounds)), global_center - dynamic_window)
            rt_end = min(float(np.median(right_bounds)), global_center + dynamic_window)

            # Extract underlying spectral coordinates
            mass_mono = target_peptide.isotopologues["mass"]
            mz_values = [(m + charge * 1.007276466812) / charge for m in mass_mono]
            mz_min, mz_max = min(mz_values) - mz_boundaries / charge, max(mz_values) + mz_boundaries / charge

            raw_apex_data = spectrum.get_raw_data(mz_range=[mz_min, mz_max], rt_range=[rt_start, rt_end],
                                                  filtered=False)

            # Match fragment spectra
            ms2_list = []
            for m_iso in mass_mono:
                potential_ms2 = spectrum.get_ms2_precursor_matches(
                    mass=m_iso, charge=charge, rt_range=[rt_start, rt_end], tolerance=10, tolerance_unit="ppm"
                )
                if potential_ms2 is not None and not potential_ms2.empty:
                    ms2_list.append(potential_ms2)

            ms2_df = pd.concat(ms2_list) if ms2_list else pd.DataFrame()

            # Initialize and analyze peak metrics
            peak_obj = PeptidePeak(
                experiment=experiment, target=target_peptide, charge=charge, rt=global_center,
                raw_apex=raw_apex_data, extracted=interp_df, ms2_data=ms2_df, rt_start=rt_start, rt_end=rt_end
            )

            peak_obj.cluster_id = cluster_id
            peak_obj.replicate_id = int(rep_idx)
            peak_obj.is_stub = False

            # Run data processing steps
            peak_obj.integrate_peak()
            peak_obj.process_and_find_peaks(snr_level=2, prominence=0.5, height=1000, bin_size=0.001)
            peak_obj.validate_ms1_pattern(tolerance_ppm=15)

            if hasattr(peak_obj, 'match_ms2_fragmentation'):
                try:
                    peak_obj.match_ms2_fragmentation(tolerance=20, tolerance_unit="ppm")
                except Exception:
                    pass

            peak_obj.process_ms1_quality()
            if hasattr(peak_obj, 'calculate_quality_score'):
                peak_obj.calculate_quality_score()

            processed_peaks.append(peak_obj)

    return processed_peaks


def pipeline_find_and_score_peak(experiment, spectrum, target, charge: int, rt_start: float, rt_end: float,
                                 rt_range: tuple = (3.0, 15.0), mz_range: tuple = (400, 2000),
                                 mz_boundaries: float = 2.0, apex_rt_window: float = 0.02):
    """Builds and fully scores a real PeptidePeak for a target/charge within an explicit RT window.

    Mirrors the per-feature construction steps used in run_peak_picking_pipeline's Step C so that
    manually redrawn RT windows (e.g. during stub re-analysis) produce peaks that are scored
    identically to the ones generated by the main pipeline.
    """
    proton_mass = 1.007276466812
    mass_mono = target.isotopologues["mass"]
    mz_values = [(m + charge * proton_mass) / charge for m in mass_mono]
    mz_min, mz_max = min(mz_values) - mz_boundaries / charge, max(mz_values) + mz_boundaries / charge

    # The EIC extraction window must always cover the requested [rt_start, rt_end], not just the
    # default rt_range. Without this, dragging the RT sliders (e.g. in the peak-review dashboard)
    # to a window outside the default 3-15 min range finds no data to extract at all, so the apex
    # search below silently falls back to a naive (rt_start+rt_end)/2 guess -- and the raw MS1
    # apex is then queried in only a +-0.02 min window around that guess, almost always missing
    # the real peak, which looks like the raw apex plot simply isn't updating.
    extraction_rt_range = (min(rt_range[0], rt_start - 1.0), max(rt_range[1], rt_end + 1.0))

    full_eic_list = extract_eic_matrices(
        spectra=[spectrum], target_peptide=target, mz_range=list(mz_range), rt_range=list(extraction_rt_range),
        filtered=True
    )
    full_eic = full_eic_list[0] if full_eic_list else pd.DataFrame()
    if full_eic is None or full_eic.empty:
        # Fall back to unfiltered extraction if the filtered pass came back empty
        full_eic_list = extract_eic_matrices(
            spectra=[spectrum], target_peptide=target, mz_range=list(mz_range), rt_range=list(extraction_rt_range),
            filtered=False
        )
        full_eic = full_eic_list[0] if full_eic_list else pd.DataFrame()

    # The apex must be located in *this charge's* trace only. full_eic spans every charge state
    # the compound chromatogram swept, so an unrestricted idxmax happily pins the apex to a
    # neighbouring charge state's maximum -- a retention time where this charge may have nothing.
    charge_eic = full_eic[full_eic['charge'] == charge] if (
        not full_eic.empty and 'charge' in full_eic.columns) else full_eic

    window_slice = charge_eic[charge_eic['time'].between(rt_start, rt_end)] if not charge_eic.empty else pd.DataFrame()
    if not window_slice.empty and window_slice['intensity_array'].max() > 0:
        apex_time = float(window_slice.loc[window_slice['intensity_array'].idxmax(), 'time'])
    else:
        apex_time = (rt_start + rt_end) / 2.0

    raw_apex_data = spectrum.get_raw_data(
        mz_range=[mz_min, mz_max],
        rt_range=[apex_time - apex_rt_window, apex_time + apex_rt_window],
        filtered=False
    )

    # Mirrors the guard run_peak_picking_pipeline's Step C applies before building a peak. Without
    # it a peak with no MS1 signal whatsoever at this charge is still constructed, scored (against
    # nothing), rendered with a blank raw m/z panel and -- in the review dashboard -- saved into
    # selected_peaks as if it were a real detection.
    if raw_apex_data is None or raw_apex_data.empty:
        raise NoRawApexError(
            f"No raw MS1 signal for charge +{charge} in m/z {mz_min:.3f}-{mz_max:.3f} at RT "
            f"{apex_time:.3f} min (window {rt_start:.3f}-{rt_end:.3f}). The EIC in this window "
            f"belongs to a different charge state."
        )

    ms2_list = []
    for m_iso in mass_mono:
        potential_ms2 = spectrum.get_ms2_precursor_matches(
            mass=m_iso, charge=charge, rt_range=[rt_start, rt_end], tolerance=10, tolerance_unit="ppm"
        )
        if potential_ms2 is not None and not potential_ms2.empty:
            ms2_list.append(potential_ms2)
    ms2_df = pd.concat(ms2_list) if ms2_list else pd.DataFrame()

    peak = PeptidePeak(
        experiment=experiment, target=target, charge=charge, extracted=full_eic, raw_apex=raw_apex_data,
        rt=apex_time, ms2_data=ms2_df, rt_start=rt_start, rt_end=rt_end, is_stub=False
    )

    peak.integrate_peak()
    peak.process_and_find_peaks()
    peak.validate_ms1_pattern()
    if hasattr(peak, 'match_ms2_fragmentation'):
        try:
            peak.match_ms2_fragmentation(tolerance=20, tolerance_unit="ppm")
        except Exception:
            pass
    peak.process_ms1_quality()
    if hasattr(peak, 'calculate_quality_score'):
        peak.calculate_quality_score()

    return peak


def generate_placeholder_stubs(real_peaks: list, experiments: list, spectra: list, all_targets: list) -> list:
    """Generates placeholder PeptidePeak objects for missing data points to ensure a complete data matrix."""
    stubs_to_add = []

    # 1. Determine the universal set of charges found for each peptide backbone
    universal_charges = defaultdict(set)
    for peak in real_peaks:
        universal_charges[peak.target.fragment_cut].add(peak.charge)

    # 2. Create a lookup map of existing peaks
    presence_map = set()
    for peak in real_peaks:
        presence_map.add((peak.target, peak.experiment.spectrum_id, peak.charge))

    # 3. Iterate through all combinations and create stubs for missing ones
    for target in all_targets:
        relevant_charges = universal_charges.get(target.fragment_cut, set())

        for i, experiment in enumerate(experiments):
            spectrum = spectra[i]  # Fetch the matching spectral data container
            for charge in relevant_charges:
                if (target, experiment.spectrum_id, charge) not in presence_map:
                    # Pass the spectrum explicitly to bypass the experiment attribute error
                    stub = PeptidePeak.create_stub(
                        experiment,
                        target,
                        charge,
                        expected_rt=8.0,
                        spectrum=spectrum
                    )
                    stubs_to_add.append(stub)

    return stubs_to_add


import pickle
from pathlib import Path
import pandas as pd

from analysis_containers import ReplicateGroupResults


def run_new_analysis_for_group(group_info: tuple,
                               group_df: pd.DataFrame,
                               report_output_dir: Path | str,
                               mod_dict: dict = None,
                               force_reload: bool = False,
                               manual_review: bool = False,
                               force_reanalyze: bool = True) -> ReplicateGroupResults:
    """Loads existing pre-parsed MSExperiment pickle files directly from each

    sample's individual experiment directory, loads the corresponding MassSpectrum,
    and runs the peak picking pipeline.
    """
    from classes_spectrometry import MassSpectrum, MSExperiment

    if mod_dict is None:
        mod_dict = globals().get('aa_comp_custom', {})

    # 1. Setup group summary output directory
    group_name = "_".join([str(val).replace(" ", "_") for val in group_info])
    report_output_dir = Path(report_output_dir)
    group_output_dir = report_output_dir / group_name
    group_output_dir.mkdir(parents=True, exist_ok=True)

    cohort_pkl_path = group_output_dir / f"{group_name}_cohort.pkl.gz"

    # 2. Return cached cohort if available and re-analysis isn't forced
    if not force_reanalyze and not force_reload and cohort_pkl_path.exists():
        print(f"--> Found existing cohort cache at: {cohort_pkl_path}. Loading...")
        return ReplicateGroupResults.load(cohort_pkl_path)

    experiments = []
    spectra = []

    print(
        f"Loading existing MSExperiment pickles and spectra for {len(group_df)} replicates in group '{group_name}'...")

    for idx, (_, row) in enumerate(group_df.iterrows()):
        row_dict = row.to_dict()
        exp_pkl_path = None

        # Sample-sheet columns that may point at the per-sample output folder / experiment pickle.
        # 'output_folder_name' is the one used by the sample sheets and MSExperiment.
        path_keys = ['output_folder_name', 'output_path', 'output_dir',
                     'experiment_pkl_path', 'output_file', 'experiment_path']

        # ------------------------------------------------------------------
        # 1. RESOLVE EXPERIMENT PICKLE LOCATION FROM ROW METADATA
        # ------------------------------------------------------------------
        for key in path_keys:
            val = row_dict.get(key)
            if pd.notna(val) and str(val).strip():
                sample_path = Path(str(val).strip())

                # Scenario A: The path directly points to a .pkl file
                if sample_path.is_file() and sample_path.suffix in ['.pkl', '.gz']:
                    exp_pkl_path = sample_path
                    break

                # Scenario B: The path is the individual experiment directory
                if sample_path.is_dir():
                    spec_id = row_dict.get('spectrum_id', '')
                    exp_id = row_dict.get('experiment_id', '')

                    # Exact name candidates
                    candidates = [
                        sample_path / f"{spec_id}_experiment.pkl",
                        sample_path / f"{spec_id}.pkl",
                        sample_path / f"{exp_id}_experiment.pkl",
                        sample_path / f"{exp_id}.pkl"
                    ]

                    found = next((c for c in candidates if c.is_file()), None)
                    if found:
                        exp_pkl_path = found
                        break

                    # Search inside directory for any *experiment*.pkl file
                    pkl_matches = [
                        f for f in sample_path.glob("*.pkl*")
                        if not f.name.endswith("_analysis_results.pkl.gz")
                           and not f.name.endswith("_cohort.pkl.gz")
                    ]
                    if pkl_matches:
                        exp_pkl_path = pkl_matches[0]
                        break

        # ------------------------------------------------------------------
        # 2. LOAD MSEXPERIMENT OBJECT
        # ------------------------------------------------------------------
        if exp_pkl_path and exp_pkl_path.is_file():
            print(f"  [+] Loading experiment metadata: {exp_pkl_path}")
            with open(exp_pkl_path, 'rb') as f:
                exp = pickle.load(f)
        else:
            print(f"  [!] Warning: Pickle not found in sample output path. Initializing from row dict...")
            exp = MSExperiment(**row_dict)

        # ------------------------------------------------------------------
        # 3. INSTANTIATE & LOAD MASSSPECTRUM FROM MSEXPERIMENT
        # ------------------------------------------------------------------
        spec = MassSpectrum(
            mzml_path=exp.mzml_path,
            output_path=exp.output_path,
            spectrum_id=exp.spectrum_id
        )
        spec.load(override=force_reload)

        experiments.append(exp)
        spectra.append(spec)

    # 4. Delegate to core peak picking pipeline using loaded objects
    cohort_results = run_peak_picking_pipeline(
        experiments=experiments,
        spectra=spectra,
        mod_dict=mod_dict,
        group_name=group_name
    )

    # 5. Save cohort summary archive
    cohort_results.save(cohort_pkl_path)
    print(f"Successfully saved group cohort archive to: {cohort_pkl_path}")

    return cohort_results

def run_peak_picking_pipeline(experiments: list, spectra: list, mod_dict: dict,
                              group_name: str, interpolation_distance: float = 0.05,
                              min_peak_width: float = 0.1, max_peak_width: float = 0.8,
                              distance_peaks: float = 0.1, min_replicates: int = 2,
                              rt_range=[3, 15], mz_range=[400, 2000]) -> ReplicateGroupResults:
    """Runs full EIC extraction, DBSCAN grouping, Peak construction, and

    stub-generation workflow, returning a complete ReplicateGroupResults container.
    """
    # 1. Initialize empty results containers for each replicate
    cohort_results = ReplicateGroupResults(group_name=group_name)
    replicate_map = {}  # {rep_idx: (replicate_id, AnalysisResults, spectrum)}

    for idx, (exp, spec) in enumerate(zip(experiments, spectra)):
        rep_id = f"{group_name}_Rep_{idx + 1}"
        ans_results = AnalysisResults(exp)

        # Expand and isolate unmodified and modified target peptides
        ans_results.expand_and_initialize_targets(mod_dict=aa_comp_custom)

        replicate_map[idx] = (rep_id, ans_results, spec)
        cohort_results.add_replicate(rep_id, ans_results)

    # 2. Get the master list of targets (unmodified + modified) from the first experiment
    master_targets = experiments[0].target_peptides

    # 3. Process every unique target peptide
    for t_idx, target in enumerate(master_targets):
        t_target_start = time.time()
        # Step A: Extract EIC profiles across all replicates
        t0 = time.time()
        print(f"  --> Target [{t_idx + 1}/{len(master_targets)}]: {target.sequence} (Cut: {target.fragment_cut})",
              end="", flush=True)
        replicate_chromatograms = extract_eic_matrices(
            spectra=spectra,
            target_peptide=target,
            mz_range=mz_range,
            rt_range=rt_range,
            filtered=True
        )
        print(f"  ├── [Step A] Extract EICs: {time.time() - t0:.3f}s")

        # Step B: Perform DBSCAN cross-replicate grouping
        t0 = time.time()
        grouped_clusters = process_and_group_all_replicates(
            replicate_chromatograms=replicate_chromatograms,
            target_peptide=target,
            interpolation_distance=interpolation_distance,
            min_peak_width=min_peak_width,
            max_peak_width=max_peak_width,
            distance_peaks=distance_peaks,
            min_replicates=min_replicates
        )
        total_features = sum(len(feats) for feats in grouped_clusters.values())
        print(
            f"  ├── [Step B] DBSCAN Grouping: {time.time() - t0:.3f}s (Clusters: {len(grouped_clusters)}, Features: {total_features})")
        detected_combinations = defaultdict(set)  # {rep_idx: { (charge) }}

        # Step C: Convert grouped clusters into real PeptidePeak objects
        t0 = time.time()
        for cluster_id, features in grouped_clusters.items():
            for f_idx,feat in enumerate(features):
                t_feat = time.time()
                rep_idx = feat["rep_idx"]
                charge = feat["charge"]
                rt = feat["rt"]
                rt_start = feat["rt_start"]
                rt_end = feat["rt_end"]

                rep_id, ans_results, spec = replicate_map[rep_idx]

                full_eic = replicate_chromatograms[rep_idx]
                if full_eic is None or full_eic.empty:
                    continue

                # 1. Slice EIC for the peak region
                extracted_slice = full_eic[(full_eic['time'] >= rt_start) & (full_eic['time'] <= rt_end)]

                # 2. SKIP if region has no points or zero intensity
                if extracted_slice.empty or extracted_slice['intensity_array'].max() <= 0:
                    continue

                # 3. Find raw apex time and get the corresponding MS1 scan
                apex_time_rt = extracted_slice.loc[extracted_slice['intensity_array'].idxmax()]['time']

                # Define mz range for fetching the MS1 scan
                mz_boundaries = 2.0
                mass_mono = target.isotopologues["mass"]
                mz_values = [(m + charge * 1.007276466812) / charge for m in mass_mono]
                mz_min, mz_max = min(mz_values) - mz_boundaries / charge, max(mz_values) + mz_boundaries / charge

                # Fetch the actual MS1 scan data at the apex using a small time window
                apex_rt_window = 0.02
                raw_apex_data = spec.get_raw_data(
                    mz_range=[mz_min, mz_max],
                    rt_range=[apex_time_rt - apex_rt_window, apex_time_rt + apex_rt_window],
                    filtered=False
                )

                if raw_apex_data is None or raw_apex_data.empty:
                    print(f"  │    ├── Warning: No raw MS1 data found at apex RT {apex_time_rt:.2f} for target {target.sequence}")
                    continue

                # Match fragment spectra
                ms2_list = []
                for m_iso in mass_mono:
                    potential_ms2 = spec.get_ms2_precursor_matches(
                        mass=m_iso, charge=charge, rt_range=[rt_start, rt_end], tolerance=10, tolerance_unit="ppm"
                    )
                    if potential_ms2 is not None and not potential_ms2.empty:
                        ms2_list.append(potential_ms2)

                ms2_df = pd.concat(ms2_list) if ms2_list else pd.DataFrame()

                # 4. Construct real peak only when valid data exists
                real_peak = PeptidePeak(
                    experiment=ans_results.experiment,
                    target=target,
                    charge=charge,
                    extracted=full_eic,
                    raw_apex=raw_apex_data,
                    rt=rt,
                    ms2_data=ms2_df,
                    rt_start=rt_start,
                    rt_end=rt_end,
                    is_stub=False
                )

                # Sub-timers inside PeptidePeak processing
                t_sub = time.time()
                real_peak.integrate_peak()
                t_integrate = time.time() - t_sub

                t_sub = time.time()
                real_peak.process_and_find_peaks()
                t_find = time.time() - t_sub

                t_sub = time.time()
                real_peak.validate_ms1_pattern()
                t_val = time.time() - t_sub

                t_sub = time.time()
                t_ms2 = 0.0
                if hasattr(real_peak, 'match_ms2_fragmentation'):
                    try:
                        real_peak.match_ms2_fragmentation(tolerance=20, tolerance_unit="ppm")
                    except Exception:
                        pass
                t_ms2 = time.time() - t_sub

                t_sub = time.time()
                real_peak.process_ms1_quality()
                t_qual = time.time() - t_sub

                if hasattr(real_peak, 'calculate_quality_score'):
                    real_peak.calculate_quality_score()

                print(f"  │    ├── Cluster {cluster_id} Feat {f_idx + 1}: "
                      f"Integrate={t_integrate:.3f}s | Find={t_find:.3f}s | "
                      f"ValidateMS1={t_val:.3f}s | MatchMS2={t_ms2:.3f}s | Quality={t_qual:.3f}s | "
                      f"Total={time.time() - t_feat:.3f}s")

                ans_results.add_peaks(target, [real_peak])
                detected_combinations[rep_idx].add(charge)
        print(f"  └── [Step C Completed] Total Features Time: {time.time() - t0:.3f}s")
        print(f"  [Target Completed in {time.time() - t_target_start:.2f}s]")

        # 4. Generate stubs for target-charge combinations that were not detected
        all_charges_observed = set()
        for rep_idx, charges in detected_combinations.items():
            all_charges_observed.update(charges)

        if not all_charges_observed:
            all_charges_observed = {2}

        for idx, (rep_id, ans_results, spec) in replicate_map.items():
            detected_charges = detected_combinations[idx]
            missing_charges = all_charges_observed - detected_charges

            for charge in missing_charges:
                stub = PeptidePeak.create_stub(
                    experiment=ans_results.experiment,
                    target=target,
                    charge=charge,
                    expected_rt=8.0,
                    spectrum=spec
                )
                ans_results.add_peaks(target, [stub])

    # Clean up duplicate overlapping windows and save individual AnalysisResults
    for _, ans_results, _ in replicate_map.values():
        ans_results.remove_duplicate_peaks()

        # Save individual replicate results file safely
        save_file = ans_results.experiment.output_path / f"{ans_results.experiment.spectrum_id}_analysis_results.pkl.gz"
        ans_results.save(save_file)

    return cohort_results