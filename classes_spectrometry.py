import os
import pickle as pkl
import gzip
import pandas as pd
import numpy as np
import pyteomics as pyteomics
import warnings
from pathlib import Path  # For better path management

import scipy as sp  # For MAD in _apply_scan_filter
# Import all peptide classes to allow unpickling of older MSExperiment objects
# that were saved when these classes were defined in this module.
from peptide_classes import TargetPeptide, PeptideFragment, PeptidePeak


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

        reader = pyteomics.mzml.read(str(self.mzml_path))  # pymzml requires a string
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

    def get_raw_data(self, mz_range: list = [400, 2000], rt_range: tuple = None, filtered: bool = True):
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

    def get_extracted_ion_chromatogram(self, mz, filtered=True, tolerance_ppm=20, use_ppm=True, tolerance_da=0.05,
                                       rt_range=None):
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

    def get_extracted_compound_chromatogram(self, mass: float, filtered=True, tolerance_ppm=20, use_ppm=True,
                                            tolerance_da=0.05, mz_range=[400, 2000], rt_range=None):
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
                xic = self.get_extracted_ion_chromatogram(mz, filtered=filtered, tolerance_ppm=tolerance_ppm,
                                                          use_ppm=True, rt_range=rt_range)
            else:
                corrected_tolerance_da = tolerance_da / charge
                xic = self.get_extracted_ion_chromatogram(mz, filtered=filtered, tolerance_da=corrected_tolerance_da,
                                                          use_ppm=False, rt_range=rt_range)

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

        # FIX: Initialize target_peptides as an empty list to prevent AttributeError.
        self.target_peptides = []
        print(f"--- MSExperiment INITIALIZED for {self.spectrum_id} (NEW VERSION) ---")

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

    @classmethod
    def load(cls, filepath: Path | str) -> 'MSExperiment':
        """Static factory helper to cleanly reload an MSExperiment metadata file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"No saved experiment metadata found at: {filepath}")

        with open(filepath, 'rb') as f:
            # Note: Ensure you imported pickle as `pkl` or change to `pickle`
            obj = pkl.load(f)
        return obj

    def load_spectrum(self, **load_kwargs) -> MassSpectrum:
        """Directly instantiates and loads the corresponding MassSpectrum data container."""
        spectrum = MassSpectrum(
            mzml_path=self.mzml_path,
            output_path=self.output_path,
            spectrum_id=self.spectrum_id
        )
        spectrum.load(**load_kwargs)
        return spectrum

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
