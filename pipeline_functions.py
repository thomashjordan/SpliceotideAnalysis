import sys
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from sklearn.cluster import DBSCAN
from pyteomics import parser as peptide_parser

from modification_dictionary import aa_comp_custom
# Use explicit imports for clarity and to avoid circular dependencies
from classes_spectrometry import (
    MSExperiment, MassSpectrum, TargetPeptide, PeptidePeak, AnalysisResults,
    GroupedExperiments, interpolate_uneven_data
)


def generate_targets(peptide_sequence, digestion, mod_dict=aa_comp_custom, missed_cleavages=1, min_length=6, max_length=67, core_only=True, core_start=0, core_end=None):
    # we generate the target masses for the given peptide sequence and digestion
    if digestion == "TEV":
        # manual cleavage 'ENLYFQ[A-Z]' as regex
        peptides = list(peptide_parser.icleave(peptide_sequence, rule='ENLYFQ[A-Z]', missed_cleavages=missed_cleavages))
        # we add a third element to the tuple, which is the end position of the peptide in the original sequence
        peptides = [(pep[0], pep[1], pep[0] + peptide_parser.length(pep[1])) for pep in peptides]
        peptides = [pep for pep in peptides if min_length <= peptide_parser.length(pep[1]) <= max_length]
        if core_only:
            # we only keep the peptides covering at least part of the core
            if not core_end and core_start > 0:
                peptides = [fragment for fragment in peptides if (fragment[2]) >= core_start]
            elif core_end and core_start:
                peptides = [fragment for fragment in peptides if fragment[0] <= core_end]
                peptides = [fragment for fragment in peptides if fragment[2] >= core_start]
        # we check whether the peptides have been spliced (contain a x or xdhA modification), if so we add an extra 1 to
        # the third tuple element indicating the end position
        peptides = [(pep[0], pep[1], pep[2]+1) if ('x' in pep[1] or 'xdhA' in pep[1]) else pep for pep in peptides]
    else:

        # we use the pyteomics parser to digest the peptide sequence
        peptides = list(peptide_parser.icleave(peptide_sequence, rule=digestion, missed_cleavages=missed_cleavages))
        # we add a third element to the tuple, which is the end position of the peptide in the original sequence
        peptides = [(pep[0], pep[1], pep[0] + peptide_parser.length(pep[1])) for pep in peptides]
        peptides = [pep for pep in peptides if min_length <= peptide_parser.length(pep[1]) <= max_length]
        if core_only:
            # we only keep the peptides covering at least part of the core
            if not core_end and core_start > 0:
                peptides = [fragment for fragment in peptides if (fragment[2]) >= core_start]
            elif core_end and core_start:
                peptides = [fragment for fragment in peptides if fragment[0] <= core_end]
                peptides = [fragment for fragment in peptides if fragment[2] >= core_start]
        # we check whether the peptides have been spliced (contain a x or xdhA modification), if so we add an extra 1 to
        # the third tuple element indicating the end position
        peptides = [(pep[0], pep[1], pep[2]+1) if ('x' in pep[1] or 'xdhA' in pep[1]) else pep for pep in peptides]

    return peptides

def find_ms_peaks_targeted_ms2(spectrum, experiment, target_peptide, **kwargs):
    # we first look at the isotopologues and try to find matches in the MS2
    all_matches = []
    for isotopologue in target_peptide.isotopologues:
        # we call the spectrum's get_ms2_precursor_matches function
        matches = spectrum.get_ms2_precursor_matches(isotopologue.mass, filtered=True, tolerance=20, mz_range=[400,2000], rt_range=None)
        if matches:
            all_matches = all_matches + matches
    if not all_matches:
        return []
    # we transform the list of matches into a dataframe


def find_ms_peaks_targeted_ms1_old(spectrum, experiment, target_peptide, mz_boundaries: float=2,
                               time_boundaries:float=0.2, interpolation_distance=0.01,
                               distance_peaks=0.1,
                               min_peak_width=0.05, max_peak_width=0.4,
                               filtered:bool=False, **kwargs):
    # we first look at the isotopologues and try to find matches in the MS1
    all_chromatograms = []
    for i in range(len(target_peptide.isotopologues["name"])):
        # we call the spectrum's get_extracted_compound_chromatogram function
        chromatogram = spectrum.get_extracted_compound_chromatogram(target_peptide.isotopologues["mass"][i], filtered=filtered, tolerance_ppm=20, mz_range=[400,2000], rt_range=None)
        # we rename the index to be the retention time in seconds
        if chromatogram is not None:
            chromatogram.set_index('time')
            chromatogram['isotopologue'] = target_peptide.isotopologues["name"][i]
            all_chromatograms.append(chromatogram)
        elif chromatogram is None:
            chromatogram = spectrum.get_extracted_compound_chromatogram(target_peptide.isotopologues["mass"][i], filtered=filtered, tolerance_ppm=20, mz_range=[395,2005], rt_range=None)
            chromatogram.set_index('time')
            chromatogram['isotopologue'] = target_peptide.isotopologues["name"][i]
            all_chromatograms.append(chromatogram)
    if not all_chromatograms:
        return []
    # we concatenate all the chromatograms into a single dataframe
    total_chromatogram = pd.concat(all_chromatograms)
    # we display the total chromatogram
    total_chromatogram = total_chromatogram.sort_values(by=['time'])

    # we look what the unique charges are
    unique_charges = total_chromatogram['charge'].unique()
    # we sort the charges
    unique_charges = np.sort(unique_charges)
    # we generate PeptidePeak objects for each cluster center
    peptide_peaks = []
    for charge in unique_charges:
        # we check the number of points for this charge
        charge_chromatogram = total_chromatogram[total_chromatogram['charge'] == charge]
        if len(charge_chromatogram) < 10:
            continue
        # we check that all the required isotopologues are present, that is all the target_peptide.isotopologues["name"] values are also in the charge_chromatogram['isotopologue'] column values
        required_isotopologues = target_peptide.isotopologues["name"]
        required_isotopologues_mz =  [(mass_topo + charge * 1.007276466812) / charge for mass_topo in target_peptide.isotopologues["mass"]]
        bounds_mz_min = min(required_isotopologues_mz) - mz_boundaries / charge
        bounds_mz_max = max(required_isotopologues_mz) + mz_boundaries / charge

        # we transform the dataframe so that the columns are the isotopologues and the index is the retention time, the values are the intensities
        charge_chromatogram = charge_chromatogram.pivot_table(index='time', columns='isotopologue',
                                                            values='intensity_array',
                                                            aggfunc='sum', fill_value=0)
        charge_chromatogram['total_intensity'] = charge_chromatogram.sum(axis=1)
        # we sort the index
        charge_chromatogram = charge_chromatogram.sort_index()

        # we find the smallest time difference between two consecutive points
        time_diffs = charge_chromatogram.index.to_series().diff().min()

        # we make a new array of time points from the minimum to the maximum with a step of time_diffs
        new_time_points = np.arange(charge_chromatogram.index.min(), charge_chromatogram.index.max(), time_diffs)
        # we use the interpolate_uneven_data function to interpolate the data
        interpolated_charge_chromatogram = pd.DataFrame(index=new_time_points)

        for iso in required_isotopologues:
            try:
                # we get the index as a numpy array
                index_np = charge_chromatogram.index.to_numpy()
                # we get the signal as a numpy array
                signal_np = charge_chromatogram[iso].to_numpy()
                new_signal_array = interpolate_uneven_data(index_np, signal_np,
                                                           new_time_points,
                                                           distance=time_diffs,) # was 0.05
                interpolated_charge_chromatogram[iso] = new_signal_array
            except KeyError:
                # if the isotopologue is not present, we fill the column with zeros
                interpolated_charge_chromatogram[iso] = 0
        charge_chromatogram = charge_chromatogram.sort_index()
        interpolated_charge_chromatogram = interpolated_charge_chromatogram.sort_index()

        # we check how much distance there is between two consecutive points in the interpolated data
        # and scale 0.1 minutes to number of points
        time_delta = interpolated_charge_chromatogram.index.to_series().diff().min()
        peak_widths = [round(min_peak_width/time_delta), round(max_peak_width/time_delta)]
        distance_in_points = round(distance_peaks/time_delta)

        absolute_noise_threshold = 1000

        peak_dict = {
            "charge":[],
            "iso":[],
            "rt":[],
            "peak_info":[]
        }

        for iso in required_isotopologues:
            # we call find_peaks on the isotopologue intensity
            peaks_iso_indices, properties_iso = sp.signal.find_peaks(interpolated_charge_chromatogram[iso],
                                                                    rel_height=0.9,
                                                                    height=absolute_noise_threshold,
                                                                    width=peak_widths,
                                                                    distance=distance_in_points
                                                                    )

            peak_rts = interpolated_charge_chromatogram.index[peaks_iso_indices]
            peak_dict["charge"].append(int(charge))
            peak_dict["iso"].append(str(iso))
            peak_dict["rt"].append(peak_rts.tolist())
            peak_dict["peak_info"].append(properties_iso)

        # we flatten the list of lists and sort it
        all_peak_rts = sorted(rt for sublist in peak_dict["rt"] for rt in sublist)
        flattened_all_rts = np.array(all_peak_rts).reshape(-1, 1)

        db = DBSCAN(eps=distance_peaks, min_samples=len(required_isotopologues)).fit(flattened_all_rts)
        labels = db.labels_

        clusters = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points

            # Grab the RTs that belong to this cluster
            cluster_points = flattened_all_rts[labels == label].flatten()

            clusters.append({
                "center": np.mean(cluster_points),
                "start": np.min(cluster_points),
                "end": np.max(cluster_points),
                "points": sorted(cluster_points.tolist()),
                "count": len(cluster_points)
            })
        # we check if clusters is empty, otherwise we retry with one isotopologue missing
        if len(clusters) == 0:
            db = DBSCAN(eps=distance_peaks, min_samples=len(required_isotopologues)-1).fit(flattened_all_rts)
            labels = db.labels_
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points

                # Grab the RTs that belong to this cluster
                cluster_points = flattened_all_rts[labels == label].flatten()

                clusters.append({
                    "center": np.mean(cluster_points),
                    "start": np.min(cluster_points),
                    "end": np.max(cluster_points),
                    "points": sorted(cluster_points.tolist()),
                    "count": len(cluster_points)
                })
        # if the clusters is still empty then we just move on nothing to see here
        if len(clusters) == 0:
            continue

        # Sort clusters by RT for a clean output
        clusters = sorted(clusters, key=lambda x: x["center"])

        # we only keep the peaks that have a rt that is near the clusters bounds
        for cluster in clusters:
            # we keep the peak_info
            cluster_rts = []
            cluster_peak_starts, cluster_peak_ends = [], []

            for z, iso, rt_list, peak_info in zip(peak_dict["charge"], peak_dict["iso"], peak_dict["rt"], peak_dict["peak_info"]):
                # Convert to numpy array to ensure vector math works
                rt_arr = np.array(rt_list)

                # Calculate distances from the cluster center
                distances = np.abs(cluster["center"] - rt_arr)

                if any(distances < distance_peaks):
                    rt_peak_idx = np.argmin(distances)

                    # Store the RT of the matching peak
                    cluster_rts.append(rt_arr[rt_peak_idx])
                    cluster_peak_starts.append(interpolated_charge_chromatogram
                                               .index[peak_info["left_ips"][rt_peak_idx]])
                    cluster_peak_ends.append(interpolated_charge_chromatogram
                                             .index[peak_info["right_ips"][rt_peak_idx]])
            # we get the median values for the peak start and end
            median_peak_start, median_peak_end = float(np.median(cluster_peak_starts)), float(np.median(cluster_peak_ends))
            center = float(np.median(cluster_rts))
            raw_apex_data =  spectrum.get_raw_data(mz_range = [bounds_mz_min, bounds_mz_max],
                                              rt_range=[median_peak_start, median_peak_end], filtered=False)

            ms2_matches = []
            for iso_tgt_mass in target_peptide.isotopologues["mass"]:
                potential_ms2_precursors = spectrum.get_ms2_precursor_matches(mass=iso_tgt_mass,
                                                    charge=charge, rt_range=[median_peak_start, median_peak_end],
                                                    tolerance=20, tolerance_unit="ppm")
                if potential_ms2_precursors is not None and not potential_ms2_precursors.empty:
                    ms2_matches.append(potential_ms2_precursors)
            if ms2_matches:
                ms2_matches_df = pd.concat(ms2_matches)
            else:
                ms2_matches_df = pd.DataFrame()

            peptide_peak = PeptidePeak(experiment=experiment, target=target_peptide,
                                       charge=charge, rt=center, raw_apex=raw_apex_data,
                                       extracted=interpolated_charge_chromatogram, ms2_data=ms2_matches_df,
                                       rt_start=median_peak_start, rt_end=median_peak_end)
            peptide_peak.integrate_peak()
            peaks_df = peptide_peak.process_and_find_peaks(snr_level=2, prominence=0.5, height=1000,
                                                           settings_1=("bin", "find_peaks_cwt"),
                                                           settings_2=("autopique", "find_peaks"), bin_size=0.001)
            matches_df = peptide_peak.validate_ms1_pattern(tolerance_ppm=10)
            matches_ms2 = peptide_peak.match_ms2_fragmentation(tolerance=20, tolerance_unit="ppm")
            quality_pass = peptide_peak.process_ms1_quality()
            quality_score = peptide_peak.calculate_quality_score()

            if peptide_peak.best_matches is not None and not peptide_peak.best_matches.empty:
                peptide_peaks.append(peptide_peak)
    if not peptide_peaks:
        return []
    return peptide_peaks


def find_ms_peaks_targeted_ms1(spectrum, experiment, target_peptide, mz_boundaries: float=2,
                               time_boundaries=0.05,
                               interpolation_distance=0.01,
                               distance_peaks=0.1,
                               min_peak_width=0.05, max_peak_width=0.4,
                               filtered:bool=False, **kwargs):

    ########### GET ECCS FOR ISOTOPOLOGUES

    # we first look at the isotopologues and try to find matches in the MS1
    all_chromatograms = []
    for i in range(len(target_peptide.isotopologues["name"])):
        # we call the spectrum's get_extracted_compound_chromatogram function
        chromatogram = spectrum.get_extracted_compound_chromatogram(target_peptide.isotopologues["mass"][i], filtered=filtered, tolerance_ppm=20, mz_range=[400,2000], rt_range=None)
        # we rename the index to be the retention time in seconds
        if chromatogram is not None:
            chromatogram.set_index('time')
            chromatogram['isotopologue'] = target_peptide.isotopologues["name"][i]
            all_chromatograms.append(chromatogram)
        elif chromatogram is None:
            chromatogram = spectrum.get_extracted_compound_chromatogram(target_peptide.isotopologues["mass"][i], filtered=filtered, tolerance_ppm=20, mz_range=[395,2005], rt_range=None)
            chromatogram.set_index('time')
            chromatogram['isotopologue'] = target_peptide.isotopologues["name"][i]
            all_chromatograms.append(chromatogram)
    if not all_chromatograms:
        return []
    # we concatenate all the chromatograms into a single dataframe
    total_chromatogram = pd.concat(all_chromatograms)
    # we display the total chromatogram
    total_chromatogram = total_chromatogram.sort_values(by=['time'])

    # we look what the unique charges are
    unique_charges = total_chromatogram['charge'].unique()
    # we sort the charges
    unique_charges = np.sort(unique_charges)
    # we generate PeptidePeak objects for each cluster center
    peptide_peaks = []
    for charge in unique_charges:
        # we check the number of points for this charge
        charge_chromatogram = total_chromatogram[total_chromatogram['charge'] == charge]
        if len(charge_chromatogram) < 10:
            continue
        # we check that all the required isotopologues are present, that is all the target_peptide.isotopologues["name"] values are also in the charge_chromatogram['isotopologue'] column values
        required_isotopologues = target_peptide.isotopologues["name"]
        required_isotopologues_mz =  [(mass_topo + charge * 1.007276466812) / charge for mass_topo in target_peptide.isotopologues["mass"]]
        bounds_mz_min = min(required_isotopologues_mz) - mz_boundaries / charge
        bounds_mz_max = max(required_isotopologues_mz) + mz_boundaries / charge

        ################ ECC DATA HANDLING AND INTERPOLATION
        # we transform the dataframe so that the columns are the isotopologues and the index is the retention time, the values are the intensities
        charge_chromatogram = charge_chromatogram.pivot_table(index='time', columns='isotopologue',
                                                            values='intensity_array',
                                                            aggfunc='sum', fill_value=0)
        charge_chromatogram['total_intensity'] = charge_chromatogram.sum(axis=1)
        # we sort the index
        charge_chromatogram = charge_chromatogram.sort_index()

        # we make a new array of time points from the minimum to the maximum with a step of time_diffs
        new_time_points = np.arange(charge_chromatogram.index.min(), charge_chromatogram.index.max(), interpolation_distance)
        # we use the interpolate_uneven_data function to interpolate the data
        interpolated_charge_chromatogram = pd.DataFrame(index=new_time_points)

        for iso in required_isotopologues:
            # we check that the column with the iso exists and is not zero
            if iso in charge_chromatogram.columns:
                if charge_chromatogram[iso].sum() != 0:

                    # we get the index as a numpy array
                    index_np = charge_chromatogram.index.to_numpy()
                    # we get the signal as a numpy array
                    signal_np = charge_chromatogram[iso].to_numpy()
                    new_signal_array = np.interp(new_time_points, index_np, signal_np)
                    interpolated_charge_chromatogram[iso] = new_signal_array
            else:
                # if the isotopologue is not present, we fill the column with zeros
                interpolated_charge_chromatogram[iso] = 0
        charge_chromatogram = charge_chromatogram.sort_index()
        interpolated_charge_chromatogram = interpolated_charge_chromatogram.sort_index()

        #################### PEAK PICKING

        # we scale the peak parameters for peak picking to the interpolated data samples
        peak_widths = [round(min_peak_width/interpolation_distance), round(max_peak_width/interpolation_distance)]
        distance_in_points = round(distance_peaks/interpolation_distance)

        absolute_noise_threshold = 1000

        peak_dict = {
            "charge":[],
            "iso":[],
            "rt":[],
            "peak_info":[]
        }

        for i, iso in enumerate(required_isotopologues):
            # we call find_peaks on the isotopologue intensity
            maximum = interpolated_charge_chromatogram[iso].max()
            peaks_iso_indices, properties_iso = sp.signal.find_peaks(interpolated_charge_chromatogram[iso],
                                                                    rel_height=0.9,
                                                                    height=absolute_noise_threshold,
                                                                    width=peak_widths,
                                                                    distance=distance_in_points,
                                                                    prominence = 0.4
                                                                    )

            peak_rts = interpolated_charge_chromatogram.index[peaks_iso_indices]

            peak_dict["charge"].append(int(charge))
            peak_dict["iso"].append(str(iso))
            peak_dict["rt"].append(peak_rts.tolist())
            peak_dict["peak_info"].append(properties_iso)

        # we check if the peak_dict has any peaks, if not we skip the rest of the code and move on to the next charge state
        if all(len(rt_list) == 0 for rt_list in peak_dict["rt"]):
            continue

        ################# PEAK CLUSTERING

        # we flatten the list of lists and sort it
        all_peak_rts = sorted(rt for sublist in peak_dict["rt"] for rt in sublist)
        flattened_all_rts = np.array(all_peak_rts).reshape(-1, 1)

        db = DBSCAN(eps=time_boundaries, min_samples=len(required_isotopologues)).fit(flattened_all_rts)
        labels = db.labels_

        clusters = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise points

            # Grab the RTs that belong to this cluster
            cluster_points = flattened_all_rts[labels == label].flatten()

            clusters.append({
                "center": np.mean(cluster_points),
                "start": np.min(cluster_points),
                "end": np.max(cluster_points),
                "points": sorted(cluster_points.tolist()),
                "count": len(cluster_points)
            })
        # we check if clusters is empty, otherwise we retry with one isotopologue missing
        if len(clusters) == 0:
            db = DBSCAN(eps=time_boundaries, min_samples=len(required_isotopologues)-1).fit(flattened_all_rts)
            labels = db.labels_
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise points

                # Grab the RTs that belong to this cluster
                cluster_points = flattened_all_rts[labels == label].flatten()

                clusters.append({
                    "center": np.mean(cluster_points),
                    "start": np.min(cluster_points),
                    "end": np.max(cluster_points),
                    "points": sorted(cluster_points.tolist()),
                    "count": len(cluster_points)
                })
        # if the clusters is still empty then we just move on nothing to see here
        if len(clusters) == 0:
            continue

        # Sort clusters by RT for a clean output
        clusters = sorted(clusters, key=lambda x: x["center"])

        # we only keep the peaks that have a rt that is near the clusters bounds
        for cluster in clusters:
            # we keep the peak_info
            cluster_rts = []
            cluster_peak_starts, cluster_peak_ends = [], []

            for z, iso, rt_list, peak_info in zip(peak_dict["charge"], peak_dict["iso"], peak_dict["rt"], peak_dict["peak_info"]):
                # Convert to numpy array to ensure vector math works
                rt_arr = np.array(rt_list)

                # Calculate distances from the cluster center
                distances = np.abs(cluster["center"] - rt_arr)

                if any(distances < distance_peaks):
                    rt_peak_idx = np.argmin(distances)

                    # Store the RT of the matching peak
                    cluster_rts.append(rt_arr[rt_peak_idx])
                    cluster_peak_starts.append(interpolated_charge_chromatogram
                                               .index[int(round(peak_info["left_ips"][rt_peak_idx]))])
                    cluster_peak_ends.append(interpolated_charge_chromatogram
                                             .index[int(round(peak_info["right_ips"][rt_peak_idx]))])
            # we get the median values for the peak start and end
            median_peak_start, median_peak_end = float(np.median(cluster_peak_starts)), float(np.median(cluster_peak_ends))
            center = float(np.median(cluster_rts))
            raw_apex_data =  spectrum.get_raw_data(mz_range = [bounds_mz_min, bounds_mz_max],
                                              rt_range=[median_peak_start, median_peak_end], filtered=False)


            ms2_matches = []
            for iso_tgt_mass in target_peptide.isotopologues["mass"]:
                potential_ms2_precursors = spectrum.get_ms2_precursor_matches(mass=iso_tgt_mass,
                                                    charge=charge, rt_range=[median_peak_start, median_peak_end],
                                                    tolerance=20, tolerance_unit="ppm")
                if potential_ms2_precursors is not None and not potential_ms2_precursors.empty:
                    ms2_matches.append(potential_ms2_precursors)
            if ms2_matches:
                ms2_matches_df = pd.concat(ms2_matches)
            else:
                ms2_matches_df = pd.DataFrame()

            peptide_peak = PeptidePeak(experiment=experiment, target=target_peptide,
                                       charge=charge, rt=center, raw_apex=raw_apex_data,
                                       extracted=interpolated_charge_chromatogram, ms2_data=ms2_matches_df,
                                       rt_start=median_peak_start, rt_end=median_peak_end)
            peptide_peak.integrate_peak()
            peaks_df = peptide_peak.process_and_find_peaks(snr_level=2,
                                                           prominence=0.5,
                                                           height=1000,
                                                           settings_1=("bin", "find_peaks_cwt"),
                                                           settings_2=("autopique", "find_peaks"),
                                                           bin_size=0.001)
            matches_df = peptide_peak.validate_ms1_pattern(tolerance_ppm=10)
            matches_ms2 = peptide_peak.match_ms2_fragmentation(tolerance=20, tolerance_unit="ppm")
            quality_pass = peptide_peak.process_ms1_quality()
            quality_score = peptide_peak.calculate_quality_score()

            if peptide_peak.best_matches is not None and not peptide_peak.best_matches.empty:
                peptide_peaks.append(peptide_peak)
    if not peptide_peaks:
        return []
    return peptide_peaks



def analyze_single_replicate(experiment: MSExperiment, force_reload: bool = False, manual_review: bool = False) -> AnalysisResults:
    """
    Analyzes a single MSExperiment (one replicate), loading its spectrum data,
    finding all target peaks, and saving the results.
    This function encapsulates the logic from the `group_cuts_analysis` cell in the notebook.

    Args:
        experiment (MSExperiment): The experiment object to analyze.
        force_reload (bool): If True, forces re-parsing of the .mzML file.
        manual_review (bool): If True, activates interactive manual review mode.
    Returns:
        AnalysisResults: An object containing the found and selected peaks for the experiment.
    """
    # Load the associated spectrum data (from cache or from .mzML)
    spectrum = MassSpectrum(
        mzml_path=experiment.mzml_path,
        output_path=experiment.output_path,
        spectrum_id=experiment.spectrum_id
    )
    spectrum.load(minimum_intensity=1000)

    results = AnalysisResults(experiment)

    for target in experiment.target_peptides:

        # This is the core peak-picking function from this file.
        peak_list = find_ms_peaks_targeted_ms1(
            spectrum, experiment, target,
            mz_boundaries=2,
            time_boundaries=0.02,
            filtered=False,
            interpolation_distance = 0.01,
            distance_peaks = 0.02,
            min_peak_width = 0.02,
            max_peak_width = 0.4,
        )
        results.add_peaks(target, peak_list)

    # The run_full_pipeline method will handle scoring, manual review (if enabled), and saving.
    results.run_full_pipeline(manual=manual_review)

    # Save the processed results for this single experiment to a pickle file
    results.save(override=True)

    return results


def run_analysis_for_group(group_info: tuple, group_df: pd.DataFrame, main_output_dir: Path, force_reload: bool, manual_review: bool, force_reanalyze: bool):
    """
    Processes a single group of experiments (replicates), generates individual
    analysis results, and then creates a final grouped report.
    """
    experiment_id, precursor_plasmid, maturase_name = group_info
    group_name = f"{experiment_id}_{precursor_plasmid}_{maturase_name}"
    print(f"Processing group: {group_name}")

    report_path = main_output_dir / experiment_id / group_name
    report_path.mkdir(parents=True, exist_ok=True)

    results_list = []
    for _, row in group_df.iterrows():
        try:
            exp_output_folder = Path(row['output_folder_name'])
            pickle_name = f"{row['sample_id']}_{row['experiment_id']}_experiment.pkl"
            exp_pickle_path = exp_output_folder / pickle_name

            if not exp_pickle_path.exists():
                print(f"  Skipping replicate {row['sample_id']}. Experiment pickle not found at: {exp_pickle_path}")
                continue

            with open(exp_pickle_path, 'rb') as f:
                experiment = pickle.load(f)

            # Check for cached analysis results
            results_path = experiment.output_path / f"{experiment.spectrum_id}_results.pkl"

            # If results exist and we are NOT forcing a re-analysis, load them.
            if results_path.exists() and not force_reanalyze:
                print(f"  Loading cached analysis results for {experiment.spectrum_id}")
                single_result = AnalysisResults.load(experiment)
                if manual_review: # If manual review is on, trigger it even for cached results
                    single_result.run_full_pipeline(manual=True)
                results_list.append(single_result)
            else:
                # Otherwise, run the full analysis from scratch.
                single_result = analyze_single_replicate(experiment, force_reload, manual_review)
                results_list.append(single_result)

        except Exception as e:
            print(f"  Failed to process replicate {row['sample_id']} in group {group_name}. Error: {e}")
            continue

    if len(results_list) < 1:
        print(f"Skipping report for group {group_name}: Not enough successful replicates processed (found {len(results_list)}, need at least 2).")
        return

    try:
        print(f"Generating grouped report for {group_name}...")
        grouped_experiment = GroupedExperiments(*results_list)
        grouped_experiment.generate_all_reports(output_dir=report_path)
        print(f"Successfully generated report for {group_name} at: {report_path}")

    except Exception as e:
        print(f"Failed to generate final report for group {group_name}. Error: {e}")
