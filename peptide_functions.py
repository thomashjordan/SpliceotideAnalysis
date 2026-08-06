import numpy as np
import pandas as pd
import scipy as sp
import pyteomics as pyteomics
from collections import defaultdict
from modification_dictionary import aa_comp_custom as aa_custom



def generate_targets(peptide_sequence, digestion, mod_dict=aa_custom, missed_cleavages=1,
                     min_length=6, max_length=67, core_only=True, core_start=0, core_end=None):
    """Digests source peptide amino acid chains into target identification matrices."""
    rule_string = 'ENLYFQ' if digestion == "TEV" else digestion

    raw_peptides = list(pyteomics.parser.icleave(peptide_sequence, rule=rule_string, missed_cleavages=missed_cleavages))
    processed = []

    for pep in raw_peptides:
        start_pos = pep[0]
        seq_string = pep[1]
        end_pos = start_pos + pyteomics.parser.length(seq_string)

        # Optimization: Simplified redundant check to avoid dead code
        if 'x' in seq_string:
            end_pos += 1

        if min_length <= (end_pos - start_pos) <= max_length:
            processed.append((start_pos, seq_string, end_pos))

    if core_only:
        if core_start > 0 and not core_end:
            processed = [p for p in processed if p[2] >= core_start]
        elif core_start and core_end:
            processed = [p for p in processed if p[0] <= core_end and p[2] >= core_start]

    return processed


def interpolate_uneven_data(x, y, new_array, distance=0.01):
    '''
    We interpolate the uneven data to the new array using nearest interpolation
    :param x: a numpy array (1D)
    :param y: the data signal
    :param new_array: the new time array, regularly interspaced
    :param distance: the new spacing
    :return: np.ndarray of the interpolated signal
    '''
    # check that x and y are numpy arrays with the same length
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise ValueError("x and y must be numpy arrays")
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be 1D numpy arrays")
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")

    # we get the diff between consecutive elements in x
    x_diff = np.diff(x)
    # we get the indices where the diff is greater than the distance
    large_diff_indices = np.where(x_diff > distance)[0]
    # those diffs are nbetween element i and i+1, so we get the corresponding x values of i and i+1 and add distance to i and subtract distance to i+1
    x_fill_values = []
    y_fill_values = []
    for idx in large_diff_indices:
        x_fill_values.append(x[idx] + distance)
        x_fill_values.append(x[idx + 1] - distance)
        y_fill_values.append(0)
        y_fill_values.append(0)
    # we concatenate the x and y fill values to the original x and y
    if x_fill_values:
        x = np.concatenate((x, np.array(x_fill_values)))
        y = np.concatenate((y, np.array(y_fill_values)))
        # we sort x and y based on the new x values
        sorted_indices = np.argsort(x)
        x = x[sorted_indices]
        y = y[sorted_indices]

    # we use sp.interpolate.interp1d with kind='nearest' to interpolate the data
    f = sp.interpolate.interp1d(x, y, kind='nearest', bounds_error=False, fill_value=0)
    y_new = f(new_array)
    return y_new

def asym_peak(t, a0, a1, a2, a3, a4):
    """function takes a retention time, and five peak shape parameters:'
    #a0 = pars[0]  # peak area
    #a1 = pars[1]  # elution time
    #a2 = pars[2]  # width of gaussian
    #a3 = pars[3]  # exponential damping term
    #a4 - baseline adjustment
    returns the y-axis intensity at that retention time for those parameters.
    from Anal. Chem. 1994, 66, 1294-1301
    """
    term1 = (a0 / (2 * a3))
    term2 = np.exp((a2 ** 2 / (2 * a3 ** 2)) + (a1 - t) / a3)
    term3 = sp.special.erf(((t - a1) / (np.sqrt(2.0) * a2)) - (a2 / (np.sqrt(2.0) * a3))) + 1.0
    f = (term1 * term2 * term3) + a4
    return f


def find_best_rt_window(peaks_map, window_size=0.15):
    """
    Finds the 0.15 min window containing the maximum number of UNIQUE experiment IDs.
    Input: peaks_map[ExpID][StateKey] = (PeakObj, IsSelected)
    """
    # 1. Flatten
    flat_peaks = []
    for exp_id, target_map in peaks_map.items():
        for state_key, (peak_obj, is_selected) in target_map.items():
            flat_peaks.append({
                'rt': peak_obj.rt,
                'exp_id': exp_id,
                'state_key': state_key,
                'peak_obj': peak_obj,
                'is_selected': is_selected
            })

    if not flat_peaks:
        return 0.0, []

    # 2. Sort
    flat_peaks.sort(key=lambda x: x['rt'])

    # 3. Sliding Window
    left = 0
    current_counts = defaultdict(int)
    current_unique = 0

    best_unique = -1
    best_window = (0, 0)

    for right in range(len(flat_peaks)):
        # Add Right
        p_r = flat_peaks[right]
        if current_counts[p_r['exp_id']] == 0:
            current_unique += 1
        current_counts[p_r['exp_id']] += 1

        # Shrink Left
        while (flat_peaks[right]['rt'] - flat_peaks[left]['rt']) > window_size:
            p_l = flat_peaks[left]
            current_counts[p_l['exp_id']] -= 1
            if current_counts[p_l['exp_id']] == 0:
                current_unique -= 1
            left += 1

        # Evaluate
        # Tie-breaker: Maximize Total Peaks (density)
        current_density = right - left + 1

        # Logic: Strictly better unique count OR (Same unique count AND higher density)
        is_better = False
        if current_unique > best_unique:
            is_better = True
        elif current_unique == best_unique:
            # Calculate density of current best
            best_density = best_window[1] - best_window[0] + 1
            if current_density > best_density:
                is_better = True

        if is_better:
            best_unique = current_unique
            best_window = (left, right)

    # 4. Return Best Subset
    start, end = best_window
    best_subset_dicts = flat_peaks[start: end + 1]

    # Return just the Peak Objects for the caller
    best_peaks_list = [d['peak_obj'] for d in best_subset_dicts]

    if best_peaks_list:
        # Calculate center RT
        center_rt = (flat_peaks[start]['rt'] + flat_peaks[end]['rt']) / 2
        return center_rt, best_peaks_list

    return 0.0, []