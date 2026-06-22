import os
import pickle as pkl
import pandas as pd
import numpy as np
from pathlib import Path  # For better path management
import matplotlib.pyplot as plt  # Placeholder for potential plotting needs
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter
from sklearn.cluster import DBSCAN

from pyteomics import mass, mzml, auxiliary  # Assuming you are using pymzml
from pyteomics import parser as peptide_parser
import scipy as sp  # For MAD in _apply_scan_filter


# for circular import circumvention
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


# Placeholder for plotting library if needed
# import matplotlib.pyplot as plt

class MSExperiment:
    """
    Manages the metadata of a mass spectrometry experiment.

    This class serves as a pure data container and provides convenient
    access to experiment-specific information and paths.
    """

    def __init__(self, **kwargs):
        # Define required fields
        necessary_keys = [
            'sample_id', 'experiment_id', 'precursor_name', 'input_mzml_folder',
            'input_mzml', 'output_folder_name', 'peptide_sequence', 'digestion'
        ]
        for key in necessary_keys:
            if key not in kwargs:
                raise ValueError(f"Missing required key in metadata: {key}")

        # Set all metadata entries as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Set default values for optional but frequently used fields
        #self.digestion = metadata.get('digestion', None)
        self.core_start = kwargs.get('core_start', 0)
        self.core_end = kwargs.get('core_end', 0)

    @property
    def mzml_path(self) -> Path:
        """Returns the full, validated path to the .mzML file."""
        path = Path(self.input_mzml_folder) / self.input_mzml
        if not path.is_file():
            raise FileNotFoundError(f"The .mzML file was not found: {path}")
        return path

    @property
    def output_path(self) -> Path:
        """Returns the path to the output folder, creating it if it doesn't exist."""
        path = Path(self.output_folder_name)
        path.mkdir(parents=True, exist_ok=True)  # Creates the folder if needed
        return path

    @property
    def spectrum_id(self) -> str:
        """Creates a unique ID for the spectrum from the metadata."""
        return f"{self.sample_id}_{self.experiment_id}"

    def __repr__(self) -> str:
        return f"<MSExperiment: {self.spectrum_id}>"

    def save(self, renew: bool = False):
        """Saves the experiment object as a pickle file in the output folder."""
        save_path = self.output_path / f"{self.spectrum_id}_experiment.pkl"
        if save_path.exists() and renew:
            print(f"Experiment file already exists: {save_path}. Overwriting due to renew=True.")
            # we remove the existing file
            os.remove(save_path)
            # Save the object
            with open(save_path, 'wb') as f:
                pkl.dump(self, f)

            print(f"Experiment saved to: {save_path}")
        elif not save_path.exists():
            # Save the object
            with open(save_path, 'wb') as f:
                pkl.dump(self, f)
            print(f"Experiment saved to: {save_path}")


class MassSpectrum:
    """
    Represents and processes the data from a single .mzML file.

    Initialization is fast. The actual data processing is triggered
    by calling the .load() method.
    """

    def __init__(self, mzml_path: Path, output_path: Path, spectrum_id: str):
        self.mzml_path = Path(mzml_path)
        self.output_path = Path(output_path)
        self.spectrum_id = spectrum_id

        # Data attributes are initially set to None.
        # They will be populated by .load().
        self.ms1: pd.DataFrame | None = None
        self.ms2: pd.DataFrame | None = None
        self.ms1_filtered: pd.DataFrame | None = None

        # Define paths to the cache files
        self._ms1_cache_path = self.output_path / f"{self.spectrum_id}_ms1.pkl"
        self._ms2_cache_path = self.output_path / f"{self.spectrum_id}_ms2.pkl"
        self._ms1_filtered_cache_path = self.output_path / f"{self.spectrum_id}_ms1_filtered.pkl"

    def load(self, override: bool = False, minimum_intensity: float = 1000,
             rt_range: tuple | None = None, time_window: float = 0.1,
             mz_range: list = [500, 3300], mz_window: float = 300, mad_factor: float = 1):
        """
        Loads the spectral data. First, it attempts to load from cache.
        If cache files are not present or 'override' is True, it parses,
        processes, and caches the raw data from the .mzML file.
        """
        if not override and self._load_from_cache():
            print(f"Spectral data for '{self.spectrum_id}' successfully loaded from cache.")
            return

        print(f"Cache not found or 'override=True'. Parsing .mzML file: {self.mzml_path}")

        # 1. Parse raw data
        self._parse_mzml(minimum_intensity=minimum_intensity, rt_range=rt_range)

        # 2. Apply filter
        print("Filtering MS1 data...")
        self.ms1_filtered = self._apply_scan_filter(
            time_window=time_window, mz_range=mz_range,
            mz_window=mz_window, mad_factor=mad_factor
        )

        # 3. Save results to cache
        self._save_to_cache()
        print(f"Spectral data for '{self.spectrum_id}' processed and saved to cache.")

    def _cache_exists(self) -> bool:
        """Checks if all necessary cache files exist."""
        return all([
            self._ms1_cache_path.exists(),
            self._ms2_cache_path.exists(),
            self._ms1_filtered_cache_path.exists()
        ])

    def _load_from_cache(self) -> bool:
        """Loads MS1, MS2, and filtered MS1 data from pickle files."""
        if not self._cache_exists():
            return False
        try:
            self.ms1 = pd.read_pickle(self._ms1_cache_path)
            self.ms2 = pd.read_pickle(self._ms2_cache_path)
            self.ms1_filtered = pd.read_pickle(self._ms1_filtered_cache_path)
            return True
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return False

    def _save_to_cache(self):
        """Saves the processed dataframes as pickle files."""
        if self.ms1 is not None:
            self.ms1.to_pickle(self._ms1_cache_path)
        if self.ms2 is not None:
            self.ms2.to_pickle(self._ms2_cache_path)
        if self.ms1_filtered is not None:
            self.ms1_filtered.to_pickle(self._ms1_filtered_cache_path)

    def _parse_mzml(self, minimum_intensity: float, rt_range: tuple | None):
        """Reads the .mzML file and extracts MS1 and MS2 data."""
        if not self.mzml_path.is_file():
            raise FileNotFoundError(f"The file {self.mzml_path} does not exist.")

        # Your logic for parsing the .mzML file remains here.
        # It's already quite good. Minor adjustments:
        # - Using with-statement for the file
        # - More specific error handling

        ms1_scans = []
        ms2_scans = []

        reader = mzml.read(str(self.mzml_path))  # pymzml requires a string
        for spectrum in reader:
            try:
                if spectrum['ms level'] == 1:
                    time = spectrum['scanList']['scan'][0]['scan start time']

                    # Filtering by intensity
                    intensities = np.array(spectrum['intensity array'], dtype=float)
                    mzs = np.array(spectrum['m/z array'], dtype=float)

                    mask = intensities >= minimum_intensity
                    if not np.any(mask):
                        continue

                    ms1_scans.append({
                        'time': time,
                        'mz_array': mzs[mask],
                        'intensity_array': intensities[mask]
                    })

                elif spectrum['ms level'] == 2:
                    precursor = spectrum['precursorList']['precursor'][0]['selectedIonList']['selectedIon'][0]
                    ms2_scans.append({
                        'time': spectrum['scanList']['scan'][0]['scan start time'],
                        'precursor_mz': precursor['selected ion m/z'],
                        'precursor_charge': precursor.get('charge state'),  # .get() is safer
                        'precursor_intensity': precursor.get('peak intensity'),
                        'fragment_mz': np.array(spectrum['m/z array'], dtype=float),
                        'fragment_int': np.array(spectrum['intensity array'], dtype=float)
                    })
            except (KeyError, IndexError) as e:
                # Skips spectra that do not match the expected format
                # print(f"Warning: Skipped a spectrum due to a format issue: {e}")
                pass

        # Convert to DataFrames and "explode"
        if not ms1_scans:
            raise ValueError("No MS1 scans found in the .mzML file.")
        self.ms1 = pd.DataFrame(ms1_scans).explode(['mz_array', 'intensity_array']).dropna().astype(float)

        if rt_range:
            self.ms1 = self.ms1[self.ms1['time'].between(rt_range[0], rt_range[1])]

        if not ms2_scans:
            print("Warning: No MS2 scans found in the .mzML file.")
            self.ms2 = pd.DataFrame()  # Create an empty DataFrame as a placeholder
        else:
            self.ms2 = pd.DataFrame(ms2_scans).explode(['fragment_mz', 'fragment_int']).dropna()

    def _apply_scan_filter(self, time_window, mz_range, mz_window, mad_factor):
        """Your filter function. The code has been adopted here without changes."""
        if self.ms1 is None or self.ms1.empty:
            print("Warning: MS1 data is empty. Cannot apply filter.")
            return pd.DataFrame()  # Return empty DataFrame

        # Precompute repeated values
        min_time = min(self.ms1['time'])
        max_time = max(self.ms1['time'])
        min_mz = mz_range[0]
        max_mz = mz_range[1]

        # Pre-filter ms1 data
        ms1_filtered = self.ms1[
            (self.ms1['time'] >= min_time) & (self.ms1['time'] <= max_time) &
            (self.ms1['mz_array'] >= min_mz) & (self.ms1['mz_array'] <= max_mz)
            ].copy()

        # ... (the rest of your filter logic remains the same here) ...
        # Just as an example of how it would continue:
        time_edges = np.arange(min_time, max_time + time_window, time_window)
        mz_edges = np.arange(min_mz, max_mz + mz_window, mz_window)

        # Extend edges to include half window size on each side
        time_edges_extended = np.concatenate(
            ([time_edges[0] - time_window / 2], time_edges, [time_edges[-1] + time_window / 2]))
        mz_edges_extended = np.concatenate(([mz_edges[0] - mz_window / 2], mz_edges, [mz_edges[-1] + mz_window / 2]))

        # Calculate indices of time and mz windows for ms1_filtered
        time_indices_filtered = np.searchsorted(time_edges_extended, ms1_filtered['time'], side='right') - 1
        mz_indices_filtered = np.searchsorted(mz_edges_extended, ms1_filtered['mz_array'], side='right') - 1

        # Calculate cutoff values for each window
        median_intensity = np.array([np.median(
            ms1_filtered['intensity_array'][(time_indices_filtered == i) & (mz_indices_filtered == j)]) if len(
            ms1_filtered['intensity_array'][(time_indices_filtered == i) & (mz_indices_filtered == j)]) > 0 else 0
                                     for i in range(len(time_edges))
                                     for j in range(len(mz_edges))])
        mad_intensity = np.array([sp.stats.median_abs_deviation(
            ms1_filtered['intensity_array'][(time_indices_filtered == i) & (mz_indices_filtered == j)]) if len(
            ms1_filtered['intensity_array'][(time_indices_filtered == i) & (mz_indices_filtered == j)]) > 0 else 0
                                  for i in range(len(time_edges))
                                  for j in range(len(mz_edges))])

        # Compute cutoff
        cutoff = median_intensity + (mad_factor * mad_intensity)
        # Reshape cutoff into a 2-dimensional array
        cutoff = cutoff.reshape(len(time_edges), len(mz_edges))

        # Calculate boolean mask for points below cutoff
        mask = ms1_filtered['intensity_array'] < cutoff[time_indices_filtered, mz_indices_filtered]

        ms1_filtered = ms1_filtered[~mask]
        return ms1_filtered

    def get_raw_data(self, mz_range: list=[400, 2000], rt_range: tuple=None, filtered: bool=True):
        """Returns the raw or filtered MS1 data within specified m/z and retention time ranges."""
        if self.ms1 is None:
            raise RuntimeError("Data must be loaded first using .load().")
        elif self.ms1.empty:
            print("Warning: MS1 data is empty.")
            return pd.DataFrame()
        elif filtered and (self.ms1_filtered is None or self.ms1_filtered.empty):
            print("Warning: Filtered MS1 data is empty.")
            return pd.DataFrame()

        source_df = self.ms1_filtered if filtered else self.ms1
        if source_df is None or source_df.empty:
            print("Warning: Source dataframe is empty.")
            return pd.DataFrame()

        if rt_range:
            source_df = source_df[source_df['time'].between(rt_range[0], rt_range[1])]

        if mz_range:
            source_df = source_df[source_df['mz_array'].between(mz_range[0], mz_range[1])]

        return source_df

    def get_extracted_ion_chromatogram(self, mz, filtered=True, tolerance_ppm=20, use_ppm=True, tolerance_da=0.05, rt_range=None):
        if self.ms1 is None:
            raise RuntimeError("Data must be loaded first using .load().")

        source_df = self.ms1_filtered if filtered else self.ms1
        if source_df is None or source_df.empty:
            print("Warning: Source dataframe for XIC is empty.")
            return pd.DataFrame()

        df = source_df

        if rt_range:
            df = df[df['time'].between(rt_range[0], rt_range[1])]

        if use_ppm:
            tolerance = mz * (tolerance_ppm / 1e6)
        else:
            tolerance = tolerance_da

        # NumPy-based filtering is efficient
        mask = (df['mz_array'] >= mz - tolerance) & (df['mz_array'] <= mz + tolerance)
        xic = df[mask].copy()

        return xic

    def get_extracted_compound_chromatogram(self, mass:float, filtered=True, tolerance_ppm=20, use_ppm=True, tolerance_da=0.05, mz_range=[400,2000], rt_range=None):
        if self.ms1 is None:
            raise RuntimeError("Data must be loaded first using .load().")

        source_df = self.ms1_filtered if filtered else self.ms1
        if source_df is None or source_df.empty:
            print("Warning: Source dataframe for XIC is empty.")
            return pd.DataFrame()

        df = source_df

        if rt_range:
            df = df[df['time'].between(rt_range[0], rt_range[1])]
        highest_charge = int(mass / mz_range[0]) + 1
        lowest_charge = max(1, int(mass / mz_range[1]) - 1)
        charges = range(lowest_charge, highest_charge + 1)
        all_xics = []
        # we loop over the charges and get the xic for each charge
        for charge in charges:
            mz = (mass + (charge * 1.007276)) / charge
            
            if use_ppm:
                xic = self.get_extracted_ion_chromatogram(mz, filtered=filtered, tolerance_ppm=tolerance_ppm, use_ppm=True, rt_range=rt_range)
            else:
                corrected_tolerance_da = tolerance_da / charge
                xic = self.get_extracted_ion_chromatogram(mz, filtered=filtered, tolerance_da=corrected_tolerance_da, use_ppm=False, rt_range=rt_range)

            if not xic.empty:
                xic = xic.copy()
                xic['charge'] = charge
                all_xics.append(xic)
        if all_xics:
            return pd.concat(all_xics, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_ms2_precursor_matches(self, mass: float, charge: int, tolerance=20, tolerance_unit="ppm", rt_range=None):
        """
        Finds MS2 spectra matching a theoretical precursor m/z.

        Args:
            mass (float): Neutral mass of the peptide.
            charge (int): Charge state.
            tolerance (float): Mass tolerance (default 20).
            tolerance_unit (str): 'ppm' or 'Da'.
            rt_range (list/tuple): Optional [start, end] retention time window.
        """
        if self.ms2 is None or self.ms2.empty:
            print("Warning: MS2 data is empty.")
            return pd.DataFrame()

        # 1. Calculate Theoretical Precursor m/z
        # (M + zH) / z
        proton_mass = 1.007276
        precursor_mz = (mass + (charge * proton_mass)) / charge

        # 2. Calculate Search Window (Delta)
        if tolerance_unit.lower() == "ppm":
            # PPM is relative to the m/z value
            delta_mz = precursor_mz * (tolerance / 1e6)
        else:
            # Dalton is absolute.
            # CAUTION: Assuming user provides tolerance on the m/z scale.
            # If tolerance is on neutral mass, divide by charge.
            # Standard practice is m/z tolerance, so we usually don't divide.
            delta_mz = tolerance

        lower_bound = precursor_mz - delta_mz
        upper_bound = precursor_mz + delta_mz

        # 3. Filter Data
        # We work on a slice to avoid MemoryWarnings
        df = self.ms2

        # RT Filtering (Apply first to reduce rows for the m/z check)
        if rt_range:
            df = df[df['time'].between(rt_range[0], rt_range[1])]

        # m/z Filtering
        # Using 'between' is cleaner and often faster than double boolean mask
        matches = df[df['precursor_mz'].between(lower_bound, upper_bound)].copy()

        return matches

    def get_tic(self, filtered=True):
        """
        Calculates the Total Ion Chromatogram (TIC).
        """
        source_df = self.ms1_filtered if filtered else self.ms1
        if source_df is None or source_df.empty:
            return pd.Series(dtype=float)
        return source_df.groupby('time')['intensity_array'].sum()

class PeptideFragment:
    """Represents a theoretical fragment of a parent peptide after orbitrap fragmentation."""

    def __init__(self, sequence: str, ion_type: str, charge: int, position: str, mod_dict: dict):
        self.sequence = sequence
        self.ion_type = ion_type  # e.g., 'b', 'y'
        self.charge = charge
        self.position = position  # e.g., b3, y5
        self.monoisotopic_mass = mass.calculate_mass(sequence=sequence, ion_type=ion_type, charge=charge, aa_comp=mod_dict)

    @property
    def mz(self) -> float:
        return self.monoisotopic_mass

    def __repr__(self):
        return f"<Fragment {self.position}+{self.charge} m/z={self.mz:.4f}>"


class TargetPeptide:
    """Represents a theoretical target peptide, potentially with modifications."""

    def __init__(self, sequence: str, mod_dict: dict, fragment_cut: tuple, modifications: dict = None):
        """
        Initializes a target peptide.
        Args:
            sequence (str): The amino acid sequence.
            mod_dict (dict): Dictionary of amino acid masses, including modifications for parsing etc
            fragment_cut (tuple): the tuple describing the cut position for fragments (start, end)
            modifications (dict): e.g., {0: {'mass': 42.01}, 'N-term': 'Acetyl'} for unlocalized modifications, or
                                  pyteomics format is flexible.
        """
        self.sequence = sequence
        self.fragment_cut = fragment_cut
        self.modifications = {"unmodified": None} if modifications is None or modifications == {} else modifications

        # Use pyteomics.parser.parse to handle modifications correctly.
        # This creates a list of amino acids and modifications.
        parsed_sequence = peptide_parser.parse(
            sequence=self.sequence,
            aa_comp=mod_dict,
            show_unmodified_termini=True
        )

        # Create a Composition object from the parsed sequence
        self.composition = mass.Composition(parsed_sequence, aa_comp=mod_dict)

        self.monoisotopic_mass = self.composition.mass()
        self.formula = self.composition
        self.fragments = []
        self.isotopologues = {'name': [], 'mass': [], 'formula': [], 'abundance': []}


    def get_isotopologues(self, mod_dict: dict, isotope_threshold: float = 0.05, include_monoisotopic: bool = True):
        elements_with_isotopes = ['C']
        iterator = mass.isotopologues(composition=self.composition, report_abundance=True,
                                        overall_threshold=isotope_threshold,
                                        elements_with_isotopes=elements_with_isotopes)
        isotopologues = [(f"isotopologue {i}", mass.calculate_mass(composition=comp), comp, abundance) for
                         i, (comp, abundance) in enumerate(iterator)]
        if isotopologues and (isotopologues[0][2] == self.composition or isotopologues[0][1] == self.monoisotopic_mass):
            # we reassign the name of the first isotopologue to monoisotopic
            isotopologues[0] = ('monoisotopic', isotopologues[0][1], isotopologues[0][2], isotopologues[0][3])
        elif include_monoisotopic:
            isotopologues.insert(0, ('monoisotopic', self.monoisotopic_mass, self.composition, 0.0))
        self.isotopologues = {
            'name': [iso[0] for iso in isotopologues],
            'mass': [iso[1] for iso in isotopologues],
            'formula': [iso[2] for iso in isotopologues],
            'abundance': [iso[3] for iso in isotopologues]
        }

    def generate_fragments(self, mod_dict: dict, types=('b', 'y'), max_charge=1):
        """Generates theoretical fragment ions using pyteomics for analyzing MS2 data later"""
        self.fragments = []
        # Pass the full modification dict to the parser
        parsed_seq = peptide_parser.parse(self.sequence, aa_comp=mod_dict, modifications=self.modifications)
        for i in range(1, len(parsed_seq)):
            for ion_type in types:
                # we use the parser from pyteomics to get the fragment sequence
                if ion_type == 'b':
                    frag_seq = ''.join(parsed_seq[:i])
                    ion_name = ion_type + str(i)
                elif ion_type == 'y':
                    frag_seq = ''.join(parsed_seq[i:])
                    ion_name = ion_type + str(len(parsed_seq) - i)
                else:
                    continue
                # we make the fragment object
                frag = PeptideFragment(sequence=frag_seq, ion_type=ion_type, charge=max_charge, position=ion_name, mod_dict=mod_dict)
                self.fragments.append(frag)

    def __repr__(self):
        return f"<TargetPeptide seq='{self.sequence}' mass={self.monoisotopic_mass:.4f}>"

class PeptidePeak:
    """Represents an observed peak in the mass spectrum that matches a theoretical peptide or fragment."""

    def __init__(
            self,
            experiment,
            target,
            charge: int,
            extracted: pd.DataFrame,
            raw_apex: pd.DataFrame,
            rt: float,
            ms2_data: pd.DataFrame,
            rt_start: float = None,
            rt_end: float = None,
            is_stub: bool = False,
    ):
        self.experiment = experiment
        self.target = target
        self.charge = charge
        self.is_stub = is_stub

        # Handle raw_apex indexing safely
        if raw_apex is not None and not raw_apex.empty:
            self.raw_apex = raw_apex.sort_values(by="mz_array", ascending=True)
        else:
            self.raw_apex = pd.DataFrame(
                columns=["time", "mz_array", "intensity_array"]
            )

        self.extracted = (
            extracted if extracted is not None else pd.DataFrame()
        )
        self.rt = rt
        self.rt_start = rt_start
        self.rt_end = rt_end
        self.ms2_data = (
            ms2_data
            if ms2_data is not None
            else pd.DataFrame(columns=["fragment_mz", "fragment_int"])
        )

        # Generate theoretical isotopologue dictionary mapping
        required_isotopologues = self.target.isotopologues["name"]
        required_isotopologues_mz = [
            (mass + charge * 1.007276466812) / charge
            for mass in self.target.isotopologues["mass"]
        ]
        self.isotopologues = {
            name: mz
            for name, mz in zip(required_isotopologues, required_isotopologues_mz)
        }

        # Initialize analysis properties
        self.zoomed_extracted = pd.DataFrame()
        self.trapz_area = 0.0 if is_stub else None
        self.gaussian_params, self.gaussian_errors = None, None

        self.raw_apex_interp = pd.DataFrame()
        self.raw_apex_bin = pd.DataFrame()
        self.raw_apex_autopique = pd.DataFrame()
        self.mass_spacing = None
        self.autopique_noise_mean = 0.0
        self.autopique_threshold = 0.0
        self.autopique_lags = None
        self.autopique_autocorr = None

        self.scipy_peaks_1, self.scipy_peaks_2 = pd.DataFrame(), pd.DataFrame()
        self.mzs_1, self.intensities_1 = np.array([]), np.array([])
        self.mzs_2, self.intensities_2 = np.array([]), np.array([])

        self.best_matches = pd.DataFrame()
        self.ms2_matches = pd.DataFrame()
        self.quality_metrics = {}

        # Pre-populate stub profile data to guarantee structural reliability
        if self.is_stub:
            self._fill_stub_defaults()

    @classmethod
    def create_stub(cls, experiment, target, charge: int, expected_rt: float):
        """Factory method to generate a placeholder object when a peak is missing

        from a sample, keeping data structures uniform.
        """
        return cls(
            experiment=experiment,
            target=target,
            charge=charge,
            extracted=pd.DataFrame(),
            raw_apex=pd.DataFrame(),
            rt=expected_rt,
            ms2_data=pd.DataFrame(),
            is_stub=True,
        )

    def _fill_stub_defaults(self):
        """Pre-fills match structures with non-detection placeholders to protect

        downstream functions.
        """
        self.best_matches = self._match_peaks_to_target(method_number=1)
        self.process_ms1_quality()

    def _estimate_peak_area(self):
        if self.is_stub or self.extracted.empty or self.raw_apex.empty:
            self.trapz_area = 0.0
            return

        time_min, time_max = (
            self.raw_apex["time"].min(),
            self.raw_apex["time"].max(),
        )
        # Use .copy() to eliminate SettingWithCopyWarning errors
        zoomed = self.extracted[
            (self.extracted.index >= time_min) & (self.extracted.index <= time_max)
            ].copy()
        zoomed = zoomed.sort_index()

        zoomed["total_intensity"] = zoomed.sum(axis=1)
        self.zoomed_extracted = zoomed

        if not zoomed.empty:
            self.trapz_area = sp.integrate.trapezoid(
                zoomed["total_intensity"], zoomed.index
            )
        else:
            self.trapz_area = 0.0

    def _fit_asym_gaussian(self):
        if (
                self.is_stub
                or self.zoomed_extracted.empty
                or "total_intensity" not in self.zoomed_extracted.columns
        ):
            return

        parguess = (
            max([self.trapz_area, 101]),
            self.rt,
            0.03,
            0.1,
            max(self.zoomed_extracted["total_intensity"].min(), 101),
        )
        try:
            gaussian_params, pcov = sp.optimize.curve_fit(
                asym_peak,
                self.zoomed_extracted.index,
                self.zoomed_extracted["total_intensity"],
                p0=parguess,
                maxfev=int(1e6),
                bounds=(
                    (100, 0.1, 0.001, 0.01, 100),
                    (10e11, self.zoomed_extracted.index.max(), 2, 100, 10e9),
                ),
                method="trf",
            )
            self.gaussian_params = gaussian_params
            self.gaussian_errors = np.sqrt(np.diag(pcov))
        except Exception as e:
            print(f"Warning: Could not fit asymmetric Gaussian: {e}")
            self.gaussian_params = None
            self.gaussian_errors = None

    def _estimate_mass_spacing(self):
        if self.raw_apex.empty:
            return None
        mzs = np.sort(self.raw_apex["mz_array"].values)
        if len(mzs) < 2:
            return None
        self.mass_spacing = max(np.diff(mzs).min(), 0.005)

    def _interpolate_peak_mzs(self, interp_dist: float = 0.0005):
        if self.raw_apex.empty or self.mass_spacing is None:
            return
        mzs = self.raw_apex["mz_array"].values
        intensities = self.raw_apex["intensity_array"].values
        if len(mzs) < 2:
            return

        new_mzs = np.arange(np.min(mzs), np.max(mzs), self.mass_spacing)
        new_intensities = interpolate_uneven_data(
            mzs, intensities, new_mzs, distance=interp_dist
        )
        self.raw_apex_interp = pd.DataFrame(
            {"mz_array": new_mzs, "intensity_array": new_intensities}
        )

    def _bin_peak_mzs(self, bin_size: float = 0.0005):
        if self.raw_apex.empty:
            return
        mzs = self.raw_apex["mz_array"].values
        intensities = self.raw_apex["intensity_array"].values
        if len(mzs) < 2:
            return

        min_mz, max_mz = np.min(mzs), np.max(mzs)
        bins = np.arange(min_mz, max_mz + bin_size, bin_size)
        bin_indices = np.clip(np.digitize(mzs, bins) - 1, 0, len(bins) - 1)

        binned_intensities = np.zeros(len(bins))
        for i in range(len(mzs)):
            binned_intensities[bin_indices[i]] += intensities[i]

        self.raw_apex_bin = pd.DataFrame(
            {"mz_array": bins, "intensity_array": binned_intensities}
        )

    def _autopique_threshold_estimation(
            self, snr_level: int = 2, initial_threshold: int = None
    ):
        if self.raw_apex.empty:
            self.raw_apex_autopique = pd.DataFrame()
            return self.raw_apex_autopique

        mzs = self.raw_apex["mz_array"].values
        intensities = self.raw_apex["intensity_array"].values
        max_intensity = np.max(intensities)

        if max_intensity <= 100:
            self.raw_apex_autopique = self.raw_apex_interp.copy()
            return self.raw_apex_autopique

        resample_delta = max(self.mass_spacing, 1e-6)
        new_mzs = np.arange(min(mzs), max(mzs) + resample_delta, resample_delta)
        new_intensities = np.interp(new_mzs, mzs, intensities)

        step_size = max(max_intensity / 100.0, 0.5)
        best_threshold = 0.0
        lags_final, autocorr_final = np.array([]), np.array([])

        for i in range(100):
            threshold = step_size * i
            if threshold >= max_intensity:
                break

            filtered = np.where(new_intensities < threshold, 0, new_intensities)
            autocorr = sp.signal.correlate(filtered, filtered, method="auto")

            max_ac = np.max(autocorr)
            if max_ac == 0:
                break

            autocorr /= max_ac
            lags = (
                    np.arange(-len(autocorr) // 2 + 1, len(autocorr) // 2 + 1)
                    * self.mass_spacing
            )

            mask = (lags >= -0.5) & (lags <= 2.55)
            autocorr_window = autocorr[mask]
            lags_window = lags[mask]

            if len(autocorr_window) == 0:
                break

            prop_below = (
                    len(autocorr_window[autocorr_window <= 0]) / len(autocorr_window)
            )
            peak_index = np.argmax(autocorr_window)

            peak_widths = sp.signal.peak_widths(
                autocorr_window, [peak_index], rel_height=0.95
            )

            if len(peak_widths[2]) > 0 and len(peak_widths[3]) > 0:
                peak_left = int(peak_widths[2][0])
                peak_right = int(peak_widths[3][0])
                prop_width = (peak_right - peak_left) / len(autocorr_window)

                if prop_below >= prop_width:
                    best_threshold = threshold
                    lags_final = lags_window
                    autocorr_final = autocorr_window
                    break

        if best_threshold == 0.0:
            best_threshold = step_size * 99

        below_thresh = new_intensities[new_intensities < best_threshold]
        noise_mean = np.mean(below_thresh) if len(below_thresh) > 0 else 0.0

        snr_threshold = snr_level * (best_threshold - noise_mean) + noise_mean
        snr_threshold = min(snr_threshold, max_intensity * 0.90)
        final_filtered = np.where(new_intensities < snr_threshold, 0, new_intensities)

        self.raw_apex_autopique = pd.DataFrame(
            {"mz_array": new_mzs, "intensity_array": final_filtered}
        )
        self.autopique_threshold = snr_threshold
        self.autopique_lags = lags_final
        self.autopique_autocorr = autocorr_final

        return self.raw_apex_autopique

    def _find_ms1_peaks_scipy(
            self,
            distance: float = 0.01,
            prominence: float = 0.5,
            rel_height: float = 0.2,
            height: float = 1000,
            data_source: str = "bin",
            method: str = "find_peaks",
            bin_size: float = 0.0005,
    ):
        if self.is_stub:
            return pd.DataFrame(), np.array([]), np.array([])

        mzs, intensities = np.array([]), np.array([])
        distance_points = 1

        if data_source == "raw" and not self.raw_apex.empty:
            mzs = self.raw_apex["mz_array"].values
            intensities = self.raw_apex["intensity_array"].values
            distance_points = max(50, int(distance / self.mass_spacing))
        elif data_source == "interp" and not self.raw_apex_interp.empty:
            mzs = self.raw_apex_interp["mz_array"].values
            intensities = self.raw_apex_interp["intensity_array"].values
            distance_points = max(1, int(distance / bin_size))
        elif data_source == "autopique" and not self.raw_apex_autopique.empty:
            mzs = self.raw_apex_autopique["mz_array"].values
            intensities = self.raw_apex_autopique["intensity_array"].values
            distance_points = max(1, int(distance / np.diff(mzs)[0]))
        elif data_source == "bin" and not self.raw_apex_bin.empty:
            mzs = self.raw_apex_bin["mz_array"].values
            intensities = self.raw_apex_bin["intensity_array"].values
            distance_points = max(1, int(distance / bin_size))

        if len(mzs) == 0:
            return pd.DataFrame(), mzs, intensities

        cwt_widths = None
        if data_source != "raw":
            R = 17500
            min_width_pts = (np.min(mzs) / R) / bin_size
            max_width_pts = (np.max(mzs) / R) / bin_size
            cwt_widths = np.arange(
                max(1, int(min_width_pts * 0.8)), int(max_width_pts * 1.2)
            )

        peaks_indices = []
        properties = {}

        if data_source == "autopique" and self.autopique_threshold is not None:
            height = self.autopique_threshold * 1.5

        if method == "find_peaks":
            peaks_indices, properties = sp.signal.find_peaks(
                intensities,
                distance=distance_points,
                prominence=prominence,
                rel_height=rel_height,
                height=height,
            )
        elif method == "find_peaks_cwt" and data_source != "raw":
            approx_indices = sp.signal.find_peaks_cwt(
                intensities, cwt_widths, min_snr=2, noise_perc=50
            )
            snapped_indices = []
            window_cwt_peak = 11
            for idx in approx_indices:
                start, end = max(0, idx - window_cwt_peak), min(
                    len(intensities), idx + window_cwt_peak + 1
                )
                local_max_idx = np.argmax(intensities[start:end]) + start
                snapped_indices.append(local_max_idx)
            peaks_indices = np.unique(snapped_indices)

        if len(peaks_indices) == 0:
            return pd.DataFrame(), mzs, intensities

        peaks_df = pd.DataFrame(
            {
                "mz_array_peak": mzs[peaks_indices],
                "intensity_array_peak": intensities[peaks_indices],
                "data_source": data_source,
                "method": method,
                "properties": [properties] * len(peaks_indices)
                if method == "find_peaks"
                else None,
            }
        )
        return peaks_df, mzs, intensities

    def _match_peaks_to_target(self, method_number: int = 1, tolerance: float = 10):
        # Establish default structure columns
        columns_layout = [
            "isotopologue",
            "theoretical_mz",
            "observed_mz",
            "error_ppm",
            "theoretical_abundance",
            "observed_abundance",
            "other_peak_in_window",
            "observed_abundance_norm",
        ]

        target_names = ["decoy-1"] + self.target.isotopologues["name"]
        target_masses = [
                            self.target.monoisotopic_mass - 1.0
                        ] + self.target.isotopologues["mass"]
        target_mzs = np.array(
            [
                (mass + self.charge * 1.007276466812) / self.charge
                for mass in target_masses
            ]
        )

        # Handle structural stub assignment early to eliminate downstream issues
        if self.is_stub:
            matches = {col: [] for col in columns_layout}
            for tgt_mz, tgt_name in zip(target_mzs, target_names):
                try:
                    idx = self.target.isotopologues["name"].index(tgt_name)
                    theo_abund = float(self.target.isotopologues["abundance"][idx])
                except ValueError:
                    theo_abund = 0.0

                matches["isotopologue"].append(tgt_name)
                matches["theoretical_mz"].append(float(tgt_mz))
                matches["observed_mz"].append(np.nan)
                matches["error_ppm"].append(np.nan)
                matches["theoretical_abundance"].append(theo_abund)
                matches["observed_abundance"].append(0.0)
                matches["other_peak_in_window"].append(False)
                matches["observed_abundance_norm"].append(0.0)
            return pd.DataFrame(matches)

        if method_number == 1:
            peaks_df, mzs, intensities = (
                self.scipy_peaks_1,
                self.mzs_1,
                self.intensities_1,
            )
        else:
            peaks_df, mzs, intensities = (
                self.scipy_peaks_2,
                self.mzs_2,
                self.intensities_2,
            )

        if peaks_df.empty:
            matches = {col: [] for col in columns_layout}
            for tgt_mz, tgt_name in zip(target_mzs, target_names):
                try:
                    idx = self.target.isotopologues["name"].index(tgt_name)
                    theo_abund = float(self.target.isotopologues["abundance"][idx])
                except ValueError:
                    theo_abund = 0.0
                matches["isotopologue"].append(tgt_name)
                matches["theoretical_mz"].append(float(tgt_mz))
                matches["observed_mz"].append(np.nan)
                matches["error_ppm"].append(np.nan)
                matches["theoretical_abundance"].append(theo_abund)
                matches["observed_abundance"].append(0.0)
                matches["other_peak_in_window"].append(False)
                matches["observed_abundance_norm"].append(0.0)
            return pd.DataFrame(matches)

        peak_mzs = peaks_df["mz_array_peak"].values
        peak_intensities = peaks_df["intensity_array_peak"].values

        matches = {col: [] for col in columns_layout}
        mz_tolerance = (target_mzs[1] * tolerance / 1e6) / self.charge

        for tgt_mz, tgt_name in zip(target_mzs, target_names):
            mz_delta = 3 / (4 * self.charge)
            window_indices = np.where(
                (peak_mzs >= tgt_mz - mz_delta) & (peak_mzs <= tgt_mz + mz_delta)
            )[0]

            obs_mz, obs_int, err_ppm = np.nan, 0.0, np.nan
            other_peaks = False

            if len(window_indices) > 0:
                window_peak_mzs = peak_mzs[window_indices]
                window_peak_ints = peak_intensities[window_indices]

                close_indices = np.where(
                    np.abs(window_peak_mzs - tgt_mz) <= mz_tolerance
                )[0]

                if len(close_indices) > 0:
                    best_relative_idx = np.argmax(window_peak_ints[close_indices])
                    best_idx = close_indices[best_relative_idx]

                    obs_mz = float(window_peak_mzs[best_idx])
                    obs_int = float(window_peak_ints[best_idx])
                    err_ppm = float(((obs_mz - tgt_mz) / tgt_mz) * 1e6)

                    if len(window_indices) > 1:
                        other_ints = np.delete(window_peak_ints, close_indices)
                        if len(other_ints) > 0 and np.any(
                                other_ints > 0.3 * obs_int
                        ):
                            other_peaks = True
                else:
                    other_peaks = True

            try:
                idx = self.target.isotopologues["name"].index(tgt_name)
                theo_abund = float(self.target.isotopologues["abundance"][idx])
            except ValueError:
                theo_abund = 0.0

            matches["isotopologue"].append(tgt_name)
            matches["theoretical_mz"].append(float(tgt_mz))
            matches["observed_mz"].append(obs_mz)
            matches["error_ppm"].append(err_ppm)
            matches["theoretical_abundance"].append(theo_abund)
            matches["observed_abundance"].append(obs_int)
            matches["other_peak_in_window"].append(other_peaks)
            matches["observed_abundance_norm"].append(0.0)

        matches_df = pd.DataFrame(matches)
        total_observed = matches_df["observed_abundance"].sum()
        if total_observed > 0:
            matches_df["observed_abundance_norm"] = (
                    matches_df["observed_abundance"] / total_observed
            )

        return matches_df

    def integrate_peak(self):
        if self.is_stub:
            self.trapz_area = 0.0
            return
        self._estimate_peak_area()
        self._fit_asym_gaussian()

    def process_and_find_peaks(
            self,
            snr_level: int = 2,
            autopique_init: int = 500,
            prominence: float = 0.5,
            height: float = 1000,
            rel_height: float = 0.2,
            ms1_distance: float = 0.05,
            settings_1: tuple = ("bin", "find_peaks_cwt"),
            settings_2: tuple = ("autopique", "find_peaks"),
            bin_size: float = 0.0005,
    ):
        if self.is_stub or self.raw_apex.empty:
            return pd.DataFrame(), pd.DataFrame()

        self._estimate_mass_spacing()
        self._interpolate_peak_mzs(interp_dist=bin_size)
        self._bin_peak_mzs(bin_size=bin_size)
        self._autopique_threshold_estimation(
            snr_level=snr_level, initial_threshold=autopique_init
        )

        self.scipy_peaks_1, self.mzs_1, self.intensities_1 = (
            self._find_ms1_peaks_scipy(
                distance=ms1_distance,
                prominence=prominence,
                height=height,
                rel_height=rel_height,
                data_source=settings_1[0],
                method=settings_1[1],
                bin_size=bin_size,
            )
        )

        self.scipy_peaks_2, self.mzs_2, self.intensities_2 = (
            self._find_ms1_peaks_scipy(
                distance=ms1_distance,
                prominence=prominence,
                height=height,
                rel_height=rel_height,
                data_source=settings_2[0],
                method=settings_2[1],
                bin_size=bin_size,
            )
        )

        return self.scipy_peaks_1, self.scipy_peaks_2

    def validate_ms1_pattern(self, tolerance_ppm: float = 20):
        if self.is_stub:
            self.peak_finding_both_methods = False
            self.peak_finding_good = False
            self.peak_finding_ok = False
            self.peak_finding_warning = True
            return self.best_matches

        def _score_matches(df):
            if df is None or df.empty:
                return 0, float("inf")
            valid = df[
                (df["isotopologue"] != "decoy-1") & (df["observed_mz"].notna())
                ]
            num_matches = len(valid)
            median_error = (
                np.median(np.abs(valid["error_ppm"]))
                if num_matches > 0
                else float("inf")
            )
            return num_matches, median_error

        self.matches_1 = self._match_peaks_to_target(1, tolerance=tolerance_ppm)
        self.matches_2 = self._match_peaks_to_target(2, tolerance=tolerance_ppm)

        score1_count, score1_err = _score_matches(self.matches_1)
        score2_count, score2_err = _score_matches(self.matches_2)

        self.peak_finding_both_methods = score1_count > 0 and score2_count > 0
        self.peak_finding_good = (
                score1_count > 0
                and score1_err < tolerance_ppm
                and score2_count > 0
                and score2_err < tolerance_ppm
        )
        self.peak_finding_ok = score1_count > 0 or score2_count > 0
        self.peak_finding_warning = score1_count == 0 and score2_count == 0

        if score1_count > score2_count:
            self.best_matches = self.matches_1.copy()
        elif score2_count > score1_count:
            self.best_matches = self.matches_2.copy()
        else:
            self.best_matches = (
                self.matches_1.copy()
                if score1_err <= score2_err
                else self.matches_2.copy()
            )

        return self.best_matches

    def process_ms1_quality(self, ppm_error: float = 10):
        # Stub profile initialization handling
        if self.is_stub or self.best_matches.empty:
            self.quality_metrics = {
                "peak_finding_ok": False,
                "peak_finding_warning": True,
                "peak_matches_similar": False,
                "decoy_unmatched": True,
                "other_peaks": False,
                "median_error_ppm": np.nan,
                "quality_pass": False,
                "overall_pass": False,
                "final_trapz_area": 0.0,
                "final_gaussian_params": None,
                "final_gaussian_errors": None,
            }
            return False

        matches = self.best_matches

        # 1. Decoy (M-1) Check
        decoy = matches[matches["isotopologue"] == "decoy-1"]
        decoy_mz_is_nan = pd.isna(decoy["observed_mz"]).any()
        decoy_abundance_is_zero = (decoy["observed_abundance"] == 0.0).any()
        decoy_unmatched = bool(decoy_mz_is_nan and decoy_abundance_is_zero)

        # 2. Evaluate actual targets
        others = matches[matches["isotopologue"] != "decoy-1"]
        other_peaks_flag = bool(others["other_peak_in_window"].any())

        # 3. Mass Accuracy Check
        valid_errors = others["error_ppm"].dropna()
        if not valid_errors.empty:
            median_error = float(np.median(np.abs(valid_errors)))
            quality_pass = bool(median_error < ppm_error)
        else:
            median_error = np.nan
            quality_pass = False

        # 4. Overall pass score validation logic
        overall_pass = bool(decoy_unmatched and quality_pass)

        self.quality_metrics = {
            "peak_finding_ok": getattr(self, "peak_finding_ok", False),
            "peak_finding_warning": getattr(self, "peak_finding_warning", True),
            "peak_matches_similar": getattr(
                self, "peak_matches_similar", False
            ),
            "decoy_unmatched": decoy_unmatched,
            "other_peaks": other_peaks_flag,
            "median_error_ppm": median_error,
            "quality_pass": quality_pass,
            "overall_pass": overall_pass,
            "final_trapz_area": self.trapz_area,
            "final_gaussian_params": self.gaussian_params,
            "final_gaussian_errors": self.gaussian_errors,
        }

        return overall_pass

    def process_and_find_peaks(self, snr_level: int = 2, autopique_init:int=500,
                               prominence: float = 0.5, height: float = 1000, rel_height: float=0.2,
                               ms1_distance: float = 0.05, settings_1: tuple=('bin', 'find_peaks_cwt'),
                               settings_2: tuple=('autopique', 'find_peaks'), bin_size: float = 0.0005):
        """
        Runs the full interpolation, thresholding, and peak finding workflow.
        Returns the final DataFrame of found peaks.
        """
        if self.raw_apex is None or self.raw_apex.empty:
            print("Processing failed: Initial raw_apex data is missing.")
            scipy_peaks = pd.DataFrame()
            return scipy_peaks

        # 1. First step: interpolate and bin the raw apex data to get a cleaner spectrum for peak finding
        self._estimate_mass_spacing()
        self._interpolate_peak_mzs(interp_dist=bin_size)
        if self.raw_apex_interp is None or self.raw_apex_interp.empty:
            print("Processing failed at interpolation step.")
            scipy_peaks_1 = pd.DataFrame()
            scipy_peaks_2 = pd.DataFrame()
            return scipy_peaks_1, scipy_peaks_2
        self._bin_peak_mzs(bin_size=bin_size)
        if self.raw_apex_bin is None or self.raw_apex_bin.empty:
            print("Processing failed at binning step.")
            scipy_peaks_1 = pd.DataFrame()
            scipy_peaks_2 = pd.DataFrame()
            return scipy_peaks_1, scipy_peaks_2

        # 2. Second step
        self._autopique_threshold_estimation(snr_level=snr_level, initial_threshold=autopique_init)
        if self.raw_apex_autopique is None or self.raw_apex_autopique.empty:
            print("Processing failed at autopique thresholding step.")
            scipy_peaks_1 = pd.DataFrame()
            scipy_peaks_2 = pd.DataFrame()
            return scipy_peaks_1, scipy_peaks_2

        # 3. Do the first peak picking with binned data and cwt
        scipy_peaks_1, mzs_1, intensities_1 = self._find_ms1_peaks_scipy(distance=ms1_distance, prominence=prominence, height=height,
                                   rel_height=rel_height, data_source=settings_1[0], method=settings_1[1], bin_size=bin_size)

        # 4. Do the second peak picking with the autopique filtered data and find_peaks to get the properties
        scipy_peaks_2, mzs_2, intensities_2 = self._find_ms1_peaks_scipy(distance=ms1_distance, prominence=prominence, height=height,
                                   rel_height=rel_height, data_source=settings_2[0], method=settings_2[1], bin_size=bin_size)

        self.scipy_peaks_1, self.scipy_peaks_2 = scipy_peaks_1, scipy_peaks_2
        self.mzs_1, self.intensities_1 = mzs_1, intensities_1
        self.mzs_2, self.intensities_2 = mzs_2, intensities_2
        return self.scipy_peaks_1, self.scipy_peaks_2


    def calculate_quality_score(self):
        """
        Calculates a multi-factor quality score from 0.0 to 1.0.
        """
        scores = {}

        # 1. PPM SCORE (Gaussian decay, center at 0ppm, sigma=5ppm)
        if hasattr(self, 'best_matches') and self.best_matches is not None and not self.best_matches.empty:
            median_ppm = self.best_matches['error_ppm'].abs().median()
            # Handle case where median_ppm is NaN (e.g., no valid errors found)
            if pd.isna(median_ppm):
                scores['ppm'] = 0.0
            else:
                scores['ppm'] = np.exp(-0.5 * (median_ppm / 5) ** 2)
        else:
            scores['ppm'] = 0.0

        # 2. ISOTOPE SCORE (Cosine similarity of abundances)
        if hasattr(self, 'best_matches') and self.best_matches is not None and not self.best_matches.empty:
            obs = self.best_matches['observed_abundance_norm'].fillna(0).values
            theo = self.best_matches['theoretical_abundance'].values
            # Check for zero-length vectors or zero norm to avoid NaN/errors
            if np.linalg.norm(obs) > 0 and np.linalg.norm(theo) > 0:
                scores['iso'] = np.dot(obs, theo) / (np.linalg.norm(obs) * np.linalg.norm(theo))
            else:
                scores['iso'] = 0.0
        else:
            scores['iso'] = 0.0

        # 3. SHAPE SCORE (Gaussian Fit R-squared or presence)
        if self.gaussian_params is not None:
            scores['shape'] = 0.9  # High score for successful fit
        elif self.trapz_area and self.trapz_area > 0:
            scores['shape'] = 0.5  # Neutral score
        else:
            scores['shape'] = 0.0

        # 4. MS2 BONUS
        if hasattr(self, 'ms2_matches') and self.ms2_matches is not None and not self.ms2_matches.empty:
            # Check if there are actual matches, not just an empty dataframe of theoretical fragments
            if not self.ms2_matches['observed_mz'].isna().all():
                scores['ms2'] = 1.0
            else:
                scores['ms2'] = 0.5
        else:
            scores['ms2'] = 0.5

        # WEIGHTED SUM
        final_score = (scores.get('ppm', 0.0) * 0.35) + (scores.get('iso', 0.0) * 0.30) + \
                      (scores.get('shape', 0.0) * 0.25) + (scores.get('ms2', 0.5) * 0.10)

        # Ensure final_score is not NaN
        if pd.isna(final_score):
            final_score = 0.0

        self.quality_score = final_score

        # Assign Tiers
        if final_score > 0.85:
            self.tier = "Gold"
        elif final_score > 0.60:
            self.tier = "Silver"
        elif final_score > 0.30:
            self.tier = "Bronze"
        else:
            self.tier = "Reject"

        return final_score

    def match_ms2_fragmentation(self, tolerance=20, tolerance_unit='ppm'):
        """
        Matches theoretical fragments to MS2 data.

        Args:
            tolerance (float): Error tolerance (default 20).
            tolerance_unit (str): 'ppm' or 'Da'.
        """
        # 1. Early Exits
        if self.ms2_data is None or self.ms2_data.empty:
            print("Warning: ms2_data is empty.")
            return pd.DataFrame()
        if not self.target.fragments:
            print("Warning: target peptide has no fragments.")
            return pd.DataFrame()

        # 2. Preparation: Extract arrays for speed
        # Sorting is critical for binary search
        spectrum = self.ms2_data.sort_values('fragment_mz')
        obs_mz_vals = spectrum['fragment_mz'].values
        obs_int_vals = spectrum['fragment_int'].values

        ms2_matches_list = []

        # 3. Fast Matching Loop
        for fragment in self.target.fragments:
            theo_mz = fragment.mz

            # Determine window boundaries based on unit
            if tolerance_unit == 'ppm':
                delta = theo_mz * (tolerance / 1e6)
            else:  # Dalton
                delta = tolerance

            mz_lower = theo_mz - delta
            mz_upper = theo_mz + delta

            # Binary search for indices (Fast!)
            start_idx = np.searchsorted(obs_mz_vals, mz_lower, side='left')
            end_idx = np.searchsorted(obs_mz_vals, mz_upper, side='right')

            match_data = {
                'fragment': fragment.position,
                'type': fragment.ion_type,
                'theoretical_mz': theo_mz,
                'observed_mz': np.nan,
                'error_ppm': np.nan,
                'intensity': 0.0
            }

            # Check if we found peaks in the window
            if start_idx < end_idx:
                # Slice the specific candidates
                candidates_mz = obs_mz_vals[start_idx:end_idx]
                candidates_int = obs_int_vals[start_idx:end_idx]

                # Logic: Select Highest Intensity
                best_idx = np.argmax(candidates_int)

                observed_mz = candidates_mz[best_idx]
                intensity = candidates_int[best_idx]
                error_ppm = ((observed_mz - theo_mz) / theo_mz) * 1e6

                match_data.update({
                    'observed_mz': observed_mz,
                    'error_ppm': error_ppm,
                    'intensity': intensity
                })

            ms2_matches_list.append(match_data)

        # 4. Final DataFrame construction
        self.ms2_matches = pd.DataFrame(ms2_matches_list)
        return self.ms2_matches

    def plot_ms1_eic(self, ax=None, time_window_ecc=(3, 15)):
        """Plots the Extracted Ion Chromatogram for all isotopologues."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Extracted Ion Chromatogram", fontsize=12)

        for name, mz in self.isotopologues.items():
            if name in self.extracted.columns:
                ax.plot(self.extracted.index, self.extracted[name], label=f"{name} m/z {mz:.4f}")

        if self.raw_apex is not None and not self.raw_apex.empty:
            ax.axvline(self.raw_apex["time"].min(), color="k", linestyle="--", alpha=0.7)
            ax.axvline(self.raw_apex["time"].max(), color="k", linestyle="--", alpha=0.7)

        ax.set_xlim(time_window_ecc)
        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Intensity")
        ax.legend(prop={"size": 8}, loc="upper right")
        return ax

    def plot_ms1_zoomed(self, ax=None):
        """Plots the zoomed EIC area along with the asymmetric Gaussian fit trace."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Zoomed Extracted Ion Chromatogram", fontsize=12)

        if self.zoomed_extracted is not None and not self.zoomed_extracted.empty:
            ax.plot(self.zoomed_extracted.index, self.zoomed_extracted["total_intensity"], color="k", label="Total EIC")

        if self.gaussian_params is not None:
            x_fit = np.linspace(self.zoomed_extracted.index.min(), self.zoomed_extracted.index.max(), 100)
            # Assumes 'asym_peak' is imported/accessible in your namespace
            y_fit = asym_peak(x_fit, *self.gaussian_params)
            ax.plot(x_fit, y_fit, color="red", linestyle="--", label="Fitted Asym Gaussian")

        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Intensity")
        ax.legend(prop={"size": 8})
        return ax

    def plot_ms1_raw(self, ax=None):
        """Plots the raw profile data."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Raw Spectrum", fontsize=12)
        if self.isotopologues:
            mz_values = list(self.isotopologues.values())
            ax.set_xlim(min(mz_values) - 1.5, max(mz_values) + 1.5)

        if self.raw_apex is not None and not self.raw_apex.empty:
            ax.plot(self.raw_apex["mz_array"], self.raw_apex["intensity_array"], label="Raw Data", color="black",
                    linestyle="--", alpha=0.5)
        if self.scipy_peaks_1 is not None and not self.scipy_peaks_1.empty:
            ax.scatter(self.scipy_peaks_1["mz_array_peak"], self.scipy_peaks_1["intensity_array_peak"], color="purple",
                       marker="x", s=80, label="Scipy Peaks CWT")

        for name, mz in self.isotopologues.items():
            ax.axvline(mz, color="red", linestyle=":", alpha=0.7)

        ax.set_ylabel("Intensity")
        ax.legend(prop={"size": 7}, loc="upper right")
        return ax


    def plot_ms1_raw_binned(self, ax=None):
        """Plots raw profile data against binned histograms and CWT peaks."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Raw and Binned Spectrum", fontsize=12)

        if self.isotopologues:
            mz_values = list(self.isotopologues.values())
            ax.set_xlim(min(mz_values) - 1.5, max(mz_values) + 1.5)

        if self.raw_apex is not None and not self.raw_apex.empty:
            ax.plot(self.raw_apex["mz_array"], self.raw_apex["intensity_array"], label="Raw Data", color="black",
                    linestyle="--", alpha=0.5)

        ax.plot(self.mzs_1, self.intensities_1, label="Binned Data", color="blue")

        if self.scipy_peaks_1 is not None and not self.scipy_peaks_1.empty:
            ax.scatter(self.scipy_peaks_1["mz_array_peak"], self.scipy_peaks_1["intensity_array_peak"], color="purple",
                       marker="x", s=80, label="Scipy Peaks CWT")

        for name, mz in self.isotopologues.items():
            ax.axvline(mz, color="red", linestyle=":", alpha=0.7)

        ax.set_ylabel("Intensity")
        ax.legend(prop={"size": 7}, loc="upper right")
        return ax

    def plot_ms1_autopique(self, ax=None):
        """Plots the matched isotopic profiles passing Autopique deconvolution."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_title("Matched Isotopologues Check", fontsize=12)

        if self.isotopologues:
            mz_values = list(self.isotopologues.values())
            ax.set_xlim(min(mz_values) - 1.5, max(mz_values) + 1.5)

        if self.raw_apex is not None and not self.raw_apex.empty:
            ax.plot(self.raw_apex["mz_array"], self.raw_apex["intensity_array"], label="Raw Data", color="black",
                    linestyle="--", alpha=0.5)

        ax.plot(self.mzs_2, self.intensities_2, label="Autopique Filtered", color="green")

        if self.scipy_peaks_2 is not None and not self.scipy_peaks_2.empty:
            ax.scatter(self.scipy_peaks_2["mz_array_peak"], self.scipy_peaks_2["intensity_array_peak"], color="red",
                       marker="x", s=80, label="Matched Isotopologues")

        if getattr(self, 'autopique_threshold', None) is not None:
            ax.axhline(self.autopique_threshold, color="orange", linestyle="--", label="Threshold", alpha=0.7)

        for name, mz in self.isotopologues.items():
            ax.axvline(mz, color="red", linestyle=":", alpha=0.7)

        ax.set_xlabel("m/z")
        ax.set_ylabel("Intensity")
        ax.legend(prop={"size": 7}, loc="upper right")
        return ax

    def plot_ms2_mirrored(self, ax=None):
        """Generates a head-to-tail mirrored fragmentation layout (Raw vs b/y matches)."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Mirrored MS2 Fragmentation (Raw vs Matched)", fontsize=12)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.7)

        max_raw_ms2 = 1
        if self.ms2_data is not None and not self.ms2_data.empty:
            ax.vlines(self.ms2_data["fragment_mz"], ymin=0, ymax=self.ms2_data["fragment_int"], color="black",
                      linewidth=0.8, alpha=0.5, label="Raw MS2 Data")
            max_raw_ms2 = self.ms2_data["fragment_int"].max()

        max_match_int = 1
        b_ions = pd.DataFrame()
        y_ions = pd.DataFrame()

        if self.ms2_matches is not None and not self.ms2_matches.empty:
            b_ions = self.ms2_matches[self.ms2_matches["type"].str.contains("b")].dropna(subset=["observed_mz"])
            y_ions = self.ms2_matches[self.ms2_matches["type"].str.contains("y")].dropna(subset=["observed_mz"])

            if not b_ions.empty:
                ax.vlines(b_ions["observed_mz"], ymin=0, ymax=-b_ions["intensity"], color="blue", linewidth=1.2,
                          label="Matched b-ions")
            if not y_ions.empty:
                ax.vlines(y_ions["observed_mz"], ymin=0, ymax=-y_ions["intensity"], color="red", linewidth=1.2,
                          label="Matched y-ions")

            all_matches = pd.concat([b_ions, y_ions])
            max_match_int = all_matches["intensity"].max() if not all_matches.empty else 1

            for _, row in b_ions.iterrows():
                ax.text(row["observed_mz"], -row["intensity"] - (max_match_int * 0.03), f"{row['fragment']}",
                        color="blue", va="top", ha="center", fontsize=7, rotation=90)
            for _, row in y_ions.iterrows():
                ax.text(row["observed_mz"], -row["intensity"] - (max_match_int * 0.03), f"y{row['fragment']}",
                        color="red", va="top", ha="center", fontsize=7, rotation=90)

        y_limit_abs = max(max_raw_ms2, max_match_int) * 1.15
        ax.set_ylim(-y_limit_abs, y_limit_abs)
        ax.set_xlabel("m/z for fragments")
        ax.set_ylabel("Abundance (Mirrored Intensity)")

        # Safe tick absolute reformatting
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([str(int(abs(y))) for y in ax.get_yticks()])
        ax.legend(loc="upper right", prop={"size": 8})
        return ax

    def plot_summary_table(self, ax=None):
        """Renders a clean, boxless background table displaying isotope metrics."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis("off")

        if self.best_matches is None or self.best_matches.empty:
            return ax

        table_data = self.best_matches.copy()
        float_cols = ["theoretical_mz", "observed_mz", "error_ppm", "theoretical_abundance", "observed_abundance"]
        for col in float_cols:
            if col in table_data.columns:
                table_data[col] = table_data[col].apply(
                    lambda x: round(x, 4) if isinstance(x, (float, np.float64)) else x)

        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)
        return ax

    def plot_peptide_sequence(self, ax=None):
        """Draws sequence strings showing structural b and y ion fragmentation assignments."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.set_title("Peptide Sequence with Modifications", fontsize=10)

        # Assumes peptide_parser is accessible in global runtime namespace
        parsed_modx_seq = peptide_parser.parse(self.target.sequence)
        seq_length = len(parsed_modx_seq)

        dynamic_fs = 12 if seq_length < 20 else 10 if seq_length < 40 else 8
        x_positions_peptide = np.linspace(0.1, 0.9, seq_length)
        x_pos_diff = x_positions_peptide[1] - x_positions_peptide[0] if seq_length > 1 else 0

        y_position_peptide = 0.5
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for j, aa in enumerate(parsed_modx_seq):
            ax.text(x_positions_peptide[j], y_position_peptide, aa, fontsize=dynamic_fs, ha="center", va="center",
                    weight="bold")

        b_ions = pd.DataFrame()
        y_ions = pd.DataFrame()
        if self.ms2_matches is not None and not self.ms2_matches.empty:
            b_ions = self.ms2_matches[self.ms2_matches["type"].str.contains("b")].dropna(subset=["observed_mz"])
            y_ions = self.ms2_matches[self.ms2_matches["type"].str.contains("y")].dropna(subset=["observed_mz"])

        if not b_ions.empty:
            for _, row in b_ions.iterrows():
                frag_pos = int(row["fragment"][1:])
                if 0 < frag_pos <= len(x_positions_peptide):
                    x_pos_current = x_positions_peptide[frag_pos - 1]
                    label_text = f"{row['fragment']} ppm:{round(row.get('error_ppm', 0), 2)}"
                    ax.text(float(x_pos_current), y_position_peptide + 0.19, label_text, color="blue", fontsize=5,
                            ha="left", va="bottom", rotation=45)
                    if frag_pos < len(x_positions_peptide):
                        ax.plot([x_pos_current + x_pos_diff / 2, x_pos_current + x_pos_diff / 2, x_pos_current],
                                [y_position_peptide, y_position_peptide + 0.12, y_position_peptide + 0.17],
                                color="blue", linewidth=0.8)

        if not y_ions.empty:
            for _, row in y_ions.iterrows():
                frag_pos = seq_length - int(row["fragment"][1:])
                if 0 < frag_pos <= len(x_positions_peptide):
                    x_pos_current = x_positions_peptide[frag_pos - 1]
                    label_text = f"{row['fragment']} ppm:{round(row.get('error_ppm', 0), 2)}"
                    ax.text(float(x_pos_current), y_position_peptide - 0.19, label_text, color="red", fontsize=5,
                            ha="right", va="top", rotation=45)
                    if frag_pos < len(x_positions_peptide):
                        ax.plot([x_pos_current - x_pos_diff / 2, x_pos_current - x_pos_diff / 2, x_pos_current],
                                [y_position_peptide, y_position_peptide - 0.12, y_position_peptide - 0.17], color="red",
                                linewidth=0.8)
        return ax

    def plot_peak(self, time_window_ecc=(3, 15)):
        """The Master compilation layout function. Re-assembles individual panels

        into the complete multi-row dashboard layout.
        """
        fig = plt.figure(constrained_layout=True, figsize=(16, 14))
        gs_main = gridspec.GridSpec(4, 2, figure=fig)
        fig.suptitle(f"MS1 Peak Analysis for {self.target.sequence}, z={self.charge} at RT={self.rt:.2f} min",
                     fontsize=16)

        self.plot_ms1_eic(ax=fig.add_subplot(gs_main[0, 0]), time_window_ecc=time_window_ecc)
        self.plot_ms1_zoomed(ax=fig.add_subplot(gs_main[0, 1]))

        ax_raw = fig.add_subplot(gs_main[1, 0])
        self.plot_ms1_raw_binned(ax=ax_raw)
        self.plot_ms1_autopique(ax=fig.add_subplot(gs_main[1, 1], sharex=ax_raw))

        self.plot_ms2_mirrored(ax=fig.add_subplot(gs_main[2:, 0]))
        self.plot_summary_table(ax=fig.add_subplot(gs_main[2, 1]))
        self.plot_peptide_sequence(ax=fig.add_subplot(gs_main[3, 1]))

        plt.show()

    def __repr__(self):
        return f"<PeptidePeak target='{self.target.sequence}' charge={self.charge} rt={self.rt:.2f}min>"

# Assume the other classes (MSExperiment, TargetPeptide, LCMSPeak, etc.) are defined as before.
'''
class AnalysisResults:
    """
    A container for storing, saving, and loading all PeptidePeak results.
    """
    def __init__(self, experiment):
        self.experiment = experiment
        self.found_peaks: dict = {}
        self.selected_peaks: dict = {}

    @property
    def _save_path(self) -> Path:
        return self.experiment.output_path / f"{self.experiment.spectrum_id}_results.pkl"

    def add_peaks(self, target, peaks: list):
        if target in self.found_peaks:
            self.found_peaks[target].extend(peaks)
        else:
            self.found_peaks[target] = peaks



    def get_all_peaks(self) -> list:
        all_peaks = []
        for peak_list in self.found_peaks.values():
            all_peaks.extend(peak_list)
        return all_peaks

    def select_best_peaks(self, score_method="bin", min_area=1000, max_ppm_error=10.0, reject_interference=False, require_gaussian=False):
        """
        Filters found_peaks based on a combination of algorithmic quality metrics
        and manual review status.
        """
        new_selection = {}
        for target, candidates in self.found_peaks.items():
            valid_candidates = []

            for p in candidates:
                if score_method == "binary":
                    # 1. MS2 check, because ms2 matches are a strong indicator
                    if p.ms2_matches is not None and len(p.ms2_matches) > 0:
                        # we change the overall_pass to true
                        p.quality_metrics['overall_pass'] = True
                        valid_candidates.append(p)
                        continue

                    # 2. INTENSITY / AREA FILTER
                    # Uses data from: p._estimate_peak_area()
                    area = getattr(p, 'trapz_area', 0)
                    if area is None or area < min_area:
                        continue

                    # 3. MASS ACCURACY & INTERFERENCE FILTER
                    # Uses data from: p._match_peaks_to_target() (assuming saved to p.best_matches)
                    if hasattr(p, 'best_matches') and p.best_matches is not None and not p.best_matches.empty:


                        # Drop NaN values (unfound isotopologues) and get the absolute ppm error
                        valid_ppms = p.best_matches['error_ppm'].dropna().abs()

                        # we want the best_matches to have matched at least two isotopologues with good error
                        if len(valid_ppms) < 2:
                            continue

                        if not valid_ppms.empty:
                            median_ppm = valid_ppms.median()
                            if median_ppm > max_ppm_error:
                                continue  # PPM error too high

                        # Optional: Reject if there is a massive interfering peak in the MS1 window
                        if reject_interference and p.best_matches['other_peak_in_window'].any():
                            continue
                    else:
                        # If there's no match data at all, it's not a valid peak
                        continue

                    # 4. PEAK SHAPE FILTER
                    # Uses data from: p._fit_asym_gaussian()
                    if require_gaussian and p.gaussian_params is None:
                        continue  # EIC was too ugly to fit a curve to

                    # If it survived all checks, it's a valid peak!
                    valid_candidates.append(p)

                elif score_method == "score":
                    if p.tier == "Gold" or p.tier == "Silver":
                        valid_candidates.append(p)


                # Only add to selected_peaks if we actually have survivors
                if valid_candidates:
                    valid_candidates.sort(key=lambda x: x.rt)
                    new_selection[target] = valid_candidates


        self.selected_peaks = new_selection

    def get_selected_peaks(self) -> list:
        if not self.selected_peaks:
            self.select_best_peaks()
        selected_peaks = []
        for p_list in self.selected_peaks.values():
            selected_peaks.extend(p_list)
        return selected_peaks

    def save(self, override: bool = True, custom_path: Path = None):
        target_path = custom_path if custom_path else self._save_path
        if target_path.exists() and not override:
            print(f"Results file already exists: {target_path}. Skipping save.")
            return
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, 'wb') as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, experiment) -> 'AnalysisResults':
        path = experiment.output_path / f"{experiment.spectrum_id}_results.pkl"
        if not path.exists():
            raise FileNotFoundError(f"Could not find results file at: {path}")
        with open(path, 'rb') as f:
            results_instance = pkl.load(f)
        results_instance.experiment = experiment
        if not hasattr(results_instance, 'selected_peaks'):
            results_instance.selected_peaks = {}
        print(f"Analysis results loaded from: {path}")
        return results_instance

    def summarize(self) -> pd.DataFrame:
        """Creates a summary DataFrame of all SELECTED peaks."""

        # Ensure we have selected the peaks first
        if not self.selected_peaks:
            self.select_best_peaks()

        targets = self.experiment.target_peptides
        cuts = set(t.fragment_cut for t in targets)
        cut_grouped_list = []

        for cut in cuts:
            targets_for_cut = []
            peaks_for_cut = []

            # FIXED: Iterate over selected_peaks, not good_peaks
            for target, peaks in self.found_peaks.items():
                if target.fragment_cut == cut:

                    peaks_for_cut.extend(peaks)
                    targets_for_cut.append(target)
                else:
                    continue

            if len(peaks_for_cut) == 0:
                continue

            # Double check QC status (though select_best_peaks should have handled this)
            all_overall_passes = [
                peak.quality_metrics.get("overall_pass", False)
                for peak in peaks_for_cut
            ]

            if not any(all_overall_passes):
                continue

            charges = sorted(list(set(p.charge for p in peaks_for_cut)))
            charge_grouped_list = []

            for charge in charges:
                charge_grouped_results = {
                    'fragment_cut': cut,
                    'charge': charge,
                    'modification': [],
                    'target': [],
                    'peaks': []
                }

                peaks_for_charge = [p for p in peaks_for_cut if p.charge == charge]
                if len(peaks_for_charge) == 0:
                    continue
                else:
                    targets_for_charge = list(set(p.target for p in peaks_for_charge))

                    for target in targets_for_charge:
                        peaks_for_target = [p for p in peaks_for_charge if p.target == target]

                        charge_grouped_results['modification'].append(target.modifications)
                        charge_grouped_results['target'].append(target.sequence)
                        charge_grouped_results['peaks'].append(peaks_for_target)
                        charge_grouped_list.append(charge_grouped_results)

            cut_grouped_list.append(charge_grouped_list)

        summary = cut_grouped_list.copy()
        self.summary = summary
        return summary

    def __repr__(self):
        total_found = sum(len(p) for p in self.found_peaks.values())
        total_selected = sum(len(p) for p in self.selected_peaks.values())
        return f"<AnalysisResults: {self.experiment.spectrum_id} (Found: {total_found}, Selected: {total_selected})>"

    def run_full_pipeline(self, score_method="score", manual=False):
        """
        Orchestrates the semi-manual workflow.
        1. Runs automatic selection.
        2. Triggers manual reviewer.
        3. Updates selection based on manual feedback.
        """
        # Step 1: Initial automatic selection
        print(f"\n>> Running Initial Auto-Selection ({score_method})")
        self.select_best_peaks(score_method=score_method)

        # Step 2: Manual Review (if requested)
        if manual:
            self.manual_peak_reviewer_v2()

            # Step 3: RE-RUN Selection
            # This is critical! We run this again so that peaks you manually
            # flipped to 'PASS' or 'FAIL' are moved into or out of self.selected_peaks.
            print("\n>> Updating final selection based on manual review...")
            self.select_best_peaks(score_method=score_method)

            # Final Save
            self.save(override=True)
            print(f">> Pipeline Complete for {self.experiment.spectrum_id}")

    def manual_peak_reviewer_v2(self):
        """
        Corrected instance method for manual review. Used from not jupyter notebooks but anaconda prompt
        """
        plt.ion()

        # Create a single flat list of all found peaks and sort by score
        all_peaks_to_review = [p for peak_list in self.found_peaks.values() for p in peak_list]
        all_peaks_to_review.sort(key=lambda p: getattr(p, 'quality_score', 0), reverse=True)

        total_peaks = len(all_peaks_to_review)

        # Smart-skip counters
        consecutive_y = 0
        consecutive_n = 0

        for i, peak in enumerate(all_peaks_to_review):
            target = peak.target

            # --- Plotting Logic ---
            fig, (ax_total_ecc, ax_eic, ax_ms1) = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={'width_ratios': [2, 1, 2]})
            fig.suptitle(f"Peak {i + 1}/{total_peaks} | Target: {target.sequence} | Charge: {peak.charge}", fontsize=12)

            # --- 1. Total EIC Plot (New) ---
            ax_total_ecc.set_title(f"All Peaks for Target")
            ax_total_ecc.set_xlabel("Retention Time [min]")
            ax_total_ecc.set_ylabel("Intensity")
            if peak.extracted is not None:
                total_eic_series = peak.extracted.sum(axis=1)
                ax_total_ecc.plot(total_eic_series.index, total_eic_series.values, color='grey', alpha=0.7)

                # Find and number all peaks for this specific target
                target_peaks = [p for p in all_peaks_to_review if p.target == target]
                target_peaks.sort(key=lambda p: p.rt) # Sort by RT for numbering

                for j, p_target in enumerate(target_peaks):
                    is_current_peak = (p_target == peak)
                    color = 'red' if is_current_peak else 'blue'
                    size = 10 if is_current_peak else 6
                    marker = 'X' if is_current_peak else 'o'
                    
                    # Get intensity at peak RT for annotation
                    intensity_at_rt = np.interp(p_target.rt, total_eic_series.index, total_eic_series.values)
                    ax_total_ecc.plot(p_target.rt, intensity_at_rt, marker=marker, color=color, markersize=size)
                    ax_total_ecc.text(p_target.rt, intensity_at_rt, f" {j+1}", color=color, fontsize=8, va='bottom')

                # Add axvspan for the current peak's boundaries
                if peak.rt_start is not None and peak.rt_end is not None:
                    ax_total_ecc.axvspan(peak.rt_start, peak.rt_end, color='red', alpha=0.2, label='Integration Window')
                    ax_total_ecc.legend(fontsize='small')

            # --- 2. Zoomed EIC Plot ---
            ax_eic.set_title("Zoomed EIC")
            ax_eic.set_xlabel("RT [min]")
            if peak.zoomed_extracted is not None:
                ax_eic.plot(peak.zoomed_extracted.index, peak.zoomed_extracted.sum(axis=1))
                ax_eic.axvline(peak.rt, color='red', linestyle='--')

            # --- 3. MS1 Apex Plot ---
            ax_ms1.set_title("MS1 Apex Spectrum")
            ax_ms1.set_xlabel("m/z")
            if peak.raw_apex is not None:
                ax_ms1.plot(peak.raw_apex['mz_array'], peak.raw_apex['intensity_array'], color='black')
                # Add theoretical isotopologue lines
                for name, mz in peak.isotopologues.items():
                    ax_ms1.axvline(mz, color='green', linestyle=':', alpha=0.8, label=f'_{name}') # Underscore hides from legend

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
            plt.show(block=False)
            plt.pause(0.1)

            # --- Console Interaction ---
            current_algo_status = "PASS" if getattr(peak, 'tier', 'Reject') in ["Gold", "Silver"] else "FAIL"
            print(f"\n[{i + 1}/{total_peaks}] Target: {target.sequence} | RT: {peak.rt:.2f}")
            print(f"  Algo Tier: {getattr(peak, 'tier', 'N/A')} ({current_algo_status}) | Score: {getattr(peak, 'quality_score', 0):.3f} | Area: {getattr(peak, 'trapz_area', 0):.2e}")

            while True:
                choice = input("  Accept peak? [y=Pass, n=Fail, s=Skip/Keep Algo]: ").lower()
                if choice == 'y':
                    peak.quality_metrics['overall_pass'] = True
                    peak.quality_metrics['manual_review'] = True
                    consecutive_y += 1
                    consecutive_n = 0
                    print("  --> Marked as PASS")
                    break
                elif choice == 'n':
                    peak.quality_metrics['overall_pass'] = False
                    peak.quality_metrics['manual_review'] = True
                    consecutive_n += 1
                    print("  --> Marked as FAIL")
                    break
                elif choice in ['s', 'skip', '']:
                    consecutive_y = 0 # Reset counters on skip
                    consecutive_n = 0
                    print("  --> Skipped (algorithm's choice is kept)")
                    break
                else:
                    print("  Invalid input. Please try again.")

            plt.close(fig)

            # Smart-skip logic
            if (consecutive_y == 1 and consecutive_n >= 2) or \
               (consecutive_y == 2 and consecutive_n >= 2):
                break # Exit the loop for this replicate

        plt.ioff() # Turn off interactive mode
'''


class AnalysisResults:
    def __init__(self, experiment):
        self.experiment = experiment
        self.found_peaks = {}  # {target: [list of PeptidePeak]}
        self.selected_peaks = {}
        self.summary_df = None

    def expand_and_initialize_targets(self, mod_dict, verbose=False):
        """Expands experiment targets with modified variants and creates stubs for missing features."""
        # 1. Expand modification targets
        mod_target_objects = []
        for mod_seq, mods in zip(self.experiment.modified_sequences, self.experiment.modifications):
            mod_seq_clean = mod_seq.replace('*', '')
            # Generate targets
            mod_peptides = generate_targets(
                peptide_sequence=mod_seq_clean, digestion=self.experiment.digestion,
                mod_dict=mod_dict, core_start=self.experiment.core_start,
                core_end=self.experiment.core_end
            )

            mod_desc = {mod[0]: mod[1] for mod in mods} if isinstance(mods, list) else {mods[0]: mods[1]}

            for tgt_coords in mod_peptides:
                tgt_obj = TargetPeptide(
                    sequence=tgt_coords[1], mod_dict=mod_dict,
                    fragment_cut=(tgt_coords[0], tgt_coords[2]),
                    modifications=mod_desc
                )
                tgt_obj.get_isotopologues(mod_dict=aa_comp_custom)
                tgt_obj.generate_fragments(types=['b', 'y'], max_charge=1)
                mod_target_objects.append(tgt_obj)

        self.experiment.target_peptides += mod_target_objects

    def add_peaks(self, target, peaks: list):
        """Adds peaks and automatically creates stubs for targets without detections."""
        if target not in self.found_peaks:
            self.found_peaks[target] = []
        self.found_peaks[target].extend(peaks)

    def remove_duplicate_peaks(self):
        """De-duplicates overlapping peaks, keeping the largest/parent peak."""
        for target, peaks in self.found_peaks.items():
            if len(peaks) < 2: continue

            # Sort by area descending (largest first)
            sorted_peaks = sorted(peaks, key=lambda x: getattr(x, 'trapz_area', 0), reverse=True)
            kept = []

            for p in sorted_peaks:
                # Keep if not entirely contained within an already kept peak
                if not any((p.rt_start >= k.rt_start and p.rt_end <= k.rt_end) for k in kept):
                    kept.append(p)
            self.found_peaks[target] = kept

    def organize_by_structure(self):
        """Groups peaks into hierarchical structures by Cut -> Charge -> Mod."""
        self.remove_duplicate_peaks()

        # Structure: {cut: {charge: {mod_key: [peaks]}}}
        organized = {}

        for target, peaks in self.found_peaks.items():
            cut = target.fragment_cut
            charge = peaks[0].charge if peaks else 0
            mod_key = str(target.modifications)

            organized.setdefault(cut, {}).setdefault(charge, {}).setdefault(mod_key, [])

            for p in peaks:
                # Add stats: count of peaks found in this grouping
                p.group_peak_count = len(peaks)
                organized[cut][charge][mod_key].append(p)

        return organized

    def select_best_peaks(self):
        """Refined filtering logic using the logic requested."""
        self.remove_duplicate_peaks()
        # ... [Keep your existing filtering logic here] ...
        pass

    def summarize(self) -> pd.DataFrame:
        """Flattens the organized structure into a robust DataFrame."""
        org = self.organize_by_structure()
        records = []

        for cut, charges in org.items():
            for charge, mods in charges.items():
                for mod_key, peaks in mods.items():
                    records.append({
                        'fragment_cut': cut,
                        'charge': charge,
                        'modification': mod_key,
                        'peak_count': len(peaks),
                        'max_area': max([p.trapz_area for p in peaks]),
                        'peaks': peaks
                    })
        self.summary_df = pd.DataFrame(records)
        return self.summary_df

class GroupedExperiments:
    def __init__(self, *analysis_results):
        if len(analysis_results) < 1:
            raise ValueError("GroupedExperiments requires at least one AnalysisResults objects.")
        self.analysis_results = analysis_results

    def _get_theoretical_targets_dict(self, cut_id):
        theoretical_targets = {}
        targets = self.analysis_results[0].experiment.target_peptides
        for t in targets:
            if t.fragment_cut == cut_id:
                state_key = f"{t.sequence} {t.modifications}"
                theoretical_targets[state_key] = t
        return theoretical_targets

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

                    qc_passing_peaks = [e['peak'] for e in group_entries if e['peak'].quality_metrics.get('overall_pass', False)]
                    qc_pass_count = len(qc_passing_peaks)
                    # FIX: Safely calculate mean, defaulting to 0.0 if no scored peaks are found to prevent NaN.
                    avg_score = np.mean([p.quality_score for p in qc_passing_peaks if hasattr(p, 'quality_score')]) if qc_passing_peaks else 0.0

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
                    state_rt_groups.sort(key=lambda x: (x['avg_score'], x['qc_pass_count'], x['total_area_all']), reverse=True)
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
                    primary_group['group_pass'] = (primary_group['qc_pass_count'] >= 1) and (primary_group['avg_score'] >= 0.5)
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
                            final_peaks_map[exp_id][state_key] = (p, is_sel, True) # set true here from is_selected

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
        """Extracts the modification part from a state string for a cleaner label."""
        try:
            # The state is formatted as "SEQUENCE {'mod': count}". We want the dict part.
            mod_part = state_string.split(' ', 1)[1]
            # A common case is "{'unmodified': None}", let's simplify it.
            if "'unmodified': None" in mod_part:
                return "Unmodified"
            return mod_part
        except IndexError:
            # If the split fails for any reason, fall back to the original string.
            return state_string

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

    def _create_single_figure(self, collected_peaks: list, peaks_map: dict, rt_groups_metadata: dict, target_dict: dict, all_states: list[str],
                              df_ratios: pd.DataFrame,
                              output_path: Path, plot_mz: bool, plot_ms2: bool, cut_id, charge):
        
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

        palette = plt.cm.tab10.colors
        while len(palette) < len(all_states):
            palette = palette + palette
        color_map = {state: palette[i] for i, state in enumerate(all_states)}

        num_states = len(all_states)
        base_rows = 5 if plot_mz else 3
        total_rows = base_rows + (num_states * 2)

        # Use constrained_layout for more robust automatic spacing
        fig = plt.figure(figsize=(9, 15 * total_rows / 8), constrained_layout=True)
        hr = [1.5, 2.0, 0.2] + [1] * (total_rows - 3)
        gs = gridspec.GridSpec(total_rows, 8, figure=fig, height_ratios=hr)
        
        fig.suptitle(super_title, fontsize=12, y=1.0)
        sns.set_style("white")
        plt.rcParams.update({'font.size': 6})

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
                qc_group = "PASS" if p.quality_metrics.get('group_pass', False) else "FAIL" # This is the GroupOK column

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
        else:
            table_data = [["No Selected Peaks", "-", "-", "-", "-", "-", "-", "-", "-", "-", "-"]]

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
                            palette=bar_palette, ax=ax_bar, errorbar='sd', capsize=0.1, alpha=0.8, dodge=False, legend=False)
                sns.stripplot(data=passed_ratios, x='ShortLabel', y='Ratio', color='black', ax=ax_bar, jitter=True, alpha=0.6, size=3)

            # Plot failed bars (hatched and transparent)
            if not failed_ratios.empty:
                sns.barplot(data=failed_ratios, x='ShortLabel', y='Ratio', hue='ShortLabel',
                            palette=bar_palette, ax=ax_bar, errorbar='sd', capsize=0.1, alpha=0.3, dodge=False,
                            legend=False, hatch='//')
                sns.stripplot(data=failed_ratios, x='ShortLabel', y='Ratio', color='grey', ax=ax_bar, jitter=True, alpha=0.4, size=3)

        ax_bar.set_ylabel("Rel. Abundance")
        # Hide x-axis labels as requested
        ax_bar.set_xlabel("")
        plt.setp(ax_bar.get_xticklabels(), visible=False)
        ax_bar.set_title("Isoform Ratios (Selected Peaks)")
        ax_bar.set_ylim(0, 1.1)

        # --- 3. Chromatograms ---
        gs_chrom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, :], hspace=0.0, height_ratios=[1, 1])
        ax_tic = fig.add_subplot(gs_chrom[0, :])
        
        # Use the loaded spectrum to get the TIC
        tic = rep_spectrum.get_tic(filtered=False)
        if not tic.empty:
            ax_tic.plot(tic.index, tic.values, color='black')
            
        ax_eic = fig.add_subplot(gs_chrom[1, :], sharex=ax_tic)

        # Create formatter for scientific notation on intensity axes
        sci_formatter = ScalarFormatter(useMathText=True)
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
            legend_handles.append(mlines.Line2D([], [], color=color, marker='s', linestyle='None', label=short_label))
        ax_eic.legend(handles=legend_handles, title="Isoforms", fontsize='small', loc='upper right')

        ax_tic.set_title("Chromatograms (Solid=GroupPass, Dashed=Reject)")
        ax_tic.set_ylabel("Intensity [a.u.]")
        ax_tic.yaxis.set_major_formatter(sci_formatter)
        plt.setp(ax_tic.get_xticklabels(), visible=False)

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
                    ax_raw_apex.vlines(p.raw_apex['mz_array'] + (diff_x * idx), diff_y*idx,
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
                            parsed_modx_seq = peptide_parser.parse(rep_peak.target.sequence)
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
                                ax_seq.text(x_positions_peptide[j], y_position_peptide, aa, fontsize=dynamic_fs, ha='center',
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
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)