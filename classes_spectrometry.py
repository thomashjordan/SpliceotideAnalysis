"""Raw LC-MS data containers.

* MassSpectrum   -- one .mzML run. ``load()`` parses MS1/MS2 scans with pyteomics, applies a local
                    median + MAD noise filter to MS1, and caches the frames as pickles next to the
                    sample's outputs so later runs skip the (slow) mzML parse.
* MSExperiment   -- metadata for one sample: a row of the sample sheet (see README.md) plus
                    derived paths (``mzml_path``, ``output_path``) and its ``target_peptides``.
* to_long_path() -- makes deep Windows paths (> MAX_PATH) usable with open()/exists().

NOTE: saved result pickles reference these classes by module path, so do not rename/move this
module without providing an unpickling shim.
"""
import os
import pickle as pkl
import gzip
import pandas as pd
import numpy as np
import pyteomics as pyteomics
import pyteomics.mzml  # `import pyteomics` does NOT auto-import the mzml submodule; _parse_mzml needs it
import warnings
from pathlib import Path  # For better path management

import scipy as sp  # For MAD in _apply_scan_filter
# Import all peptide classes to allow unpickling of older MSExperiment objects
# that were saved when these classes were defined in this module.
from peptide_classes import TargetPeptide, PeptideFragment, PeptidePeak


def to_long_path(path) -> str:
    """Return a filesystem path safe to hand to open()/exists()/pandas/pyteomics even when it is
    very long.

    Windows' default Win32 API caps paths at 260 characters (MAX_PATH); beyond that, os.path.exists
    quietly returns False and open() raises FileNotFoundError even though the file is really there
    (e.g. result files buried under a deep project tree). Prefixing an ABSOLUTE, backslash-separated
    path with the extended-length marker ``\\\\?\\`` lifts that limit.

    This is applied only when it is actually needed and safe:
      * no-op off Windows,
      * no-op for a path already carrying the ``\\\\?\\`` prefix,
      * no-op for comfortably short paths (left byte-for-byte unchanged, so libraries that don't
        understand the prefix -- and short paths never need it anyway -- are unaffected).
    UNC paths (``\\\\server\\share``) get the ``\\\\?\\UNC\\`` form. Always returns a str.
    """
    s = os.fspath(path)
    if os.name != "nt" or s.startswith("\\\\?\\"):
        return s
    ab = os.path.abspath(s)
    if len(ab) < 250:                       # short enough that MAX_PATH is not a concern
        return s
    if ab.startswith("\\\\"):               # UNC: \\server\share -> \\?\UNC\server\share
        return "\\\\?\\UNC\\" + ab[2:]
    return "\\\\?\\" + ab


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

        # Lazily built m/z-sorted views of ms1/ms1_filtered, keyed by "raw"/"filtered". Every EIC
        # and raw-apex query is a narrow m/z slice, and scanning the full ~14M-row long-form frame
        # per query was where essentially all of the pipeline's runtime went. Sorting once (~0.2 s)
        # turns each subsequent query into two np.searchsorted calls.
        self._mz_index_cache: dict = {}

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
            os.path.exists(to_long_path(self._ms1_cache_path)),
            os.path.exists(to_long_path(self._ms2_cache_path)),
            os.path.exists(to_long_path(self._ms1_filtered_cache_path))
        ])

    def _load_from_cache(self) -> bool:
        """Loads MS1, MS2, and filtered MS1 data from pickle files."""
        if not self._cache_exists():
            return False
        try:
            self.ms1 = pd.read_pickle(to_long_path(self._ms1_cache_path))
            self.ms2 = pd.read_pickle(to_long_path(self._ms2_cache_path))
            self.ms1_filtered = pd.read_pickle(to_long_path(self._ms1_filtered_cache_path))
            return True
        except Exception as e:
            print(f"Error loading from cache: {e}")
            return False

    def _save_to_cache(self):
        """Saves the processed dataframes as pickle files."""
        if self.ms1 is not None:
            self.ms1.to_pickle(to_long_path(self._ms1_cache_path))
        if self.ms2 is not None:
            self.ms2.to_pickle(to_long_path(self._ms2_cache_path))
        if self.ms1_filtered is not None:
            self.ms1_filtered.to_pickle(to_long_path(self._ms1_filtered_cache_path))

    def _parse_mzml(self, minimum_intensity: float, rt_range: tuple | None):
        """Reads the .mzML file and extracts MS1 and MS2 data."""
        if not os.path.isfile(to_long_path(self.mzml_path)):
            raise FileNotFoundError(f"The file {self.mzml_path} does not exist.")

        # Stream the file once, splitting scans by MS level: MS1 scans become long-form
        # (time, mz, intensity) rows; MS2 scans keep their precursor info + fragment arrays.

        ms1_scans = []
        ms2_scans = []

        reader = pyteomics.mzml.read(to_long_path(self.mzml_path))  # pymzml requires a string (long-path-safe)
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
        """Local-noise filter for MS1: bins the data into (time_window x mz_window) cells and keeps
        only points whose intensity exceeds that cell's median + mad_factor * MAD."""
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

        # Grid of (time, m/z) cells over which the local noise level is estimated.
        time_edges = np.arange(min_time, max_time + time_window, time_window)
        mz_edges = np.arange(min_mz, max_mz + mz_window, mz_window)

        # Extend edges to include half window size on each side
        time_edges_extended = np.concatenate(
            ([time_edges[0] - time_window / 2], time_edges, [time_edges[-1] + time_window / 2]))
        mz_edges_extended = np.concatenate(([mz_edges[0] - mz_window / 2], mz_edges, [mz_edges[-1] + mz_window / 2]))

        # Calculate indices of time and mz windows for ms1_filtered
        time_indices_filtered = np.searchsorted(time_edges_extended, ms1_filtered['time'], side='right') - 1
        mz_indices_filtered = np.searchsorted(mz_edges_extended, ms1_filtered['mz_array'], side='right') - 1

        # Calculate cutoff values for each window.
        #
        # This used to be a pair of nested list comprehensions that, for every one of the
        # len(time_edges) x len(mz_edges) cells, boolean-masked the *entire* frame twice (once for
        # the median, once for the MAD). On a real file that is ~2000 cells x 2 passes x ~14M rows,
        # i.e. tens of billions of element comparisons per mzML. The grouped form below computes
        # exactly the same numbers -- per-cell median, and MAD as median(|x - median|), which is
        # what scipy's median_abs_deviation returns at its default scale=1.0 -- in two vectorised
        # passes instead.
        n_time, n_mz = len(time_edges), len(mz_edges)
        intensities = ms1_filtered['intensity_array'].to_numpy()

        cell_ids = time_indices_filtered * n_mz + mz_indices_filtered
        in_grid = (time_indices_filtered >= 0) & (time_indices_filtered < n_time) & \
                  (mz_indices_filtered >= 0) & (mz_indices_filtered < n_mz)

        median_intensity = np.zeros(n_time * n_mz)
        mad_intensity = np.zeros(n_time * n_mz)

        if np.any(in_grid):
            cells = pd.Series(intensities[in_grid], index=pd.Index(cell_ids[in_grid], name='cell'))
            per_cell_median = cells.groupby(level='cell').median()
            # Broadcast each cell's median back onto its rows so the MAD is a second grouped median
            # over absolute deviations, rather than a Python-level call per cell.
            abs_dev = (cells - per_cell_median.reindex(cells.index).to_numpy()).abs()
            per_cell_mad = abs_dev.groupby(level='cell').median()

            median_intensity[per_cell_median.index.to_numpy()] = per_cell_median.to_numpy()
            mad_intensity[per_cell_mad.index.to_numpy()] = per_cell_mad.to_numpy()

        # Compute cutoff
        cutoff = median_intensity + (mad_factor * mad_intensity)
        # Reshape cutoff into a 2-dimensional array
        cutoff = cutoff.reshape(n_time, n_mz)

        # Calculate boolean mask for points below cutoff
        mask = ms1_filtered['intensity_array'] < cutoff[time_indices_filtered, mz_indices_filtered]

        ms1_filtered = ms1_filtered[~mask]
        return ms1_filtered

    def _get_mz_index(self, filtered: bool):
        """Returns (mz_sorted, time_sorted, intensity_sorted) for the requested source, building
        and caching the m/z-sorted view on first use.

        Rebuilt automatically if the underlying frame is replaced (checked by identity + length),
        so re-running .load() or swapping in a different frame can never serve a stale index.
        """
        key = "filtered" if filtered else "raw"
        source_df = self.ms1_filtered if filtered else self.ms1
        if source_df is None or source_df.empty:
            return None

        stamp = (id(source_df), len(source_df))
        cached = self._mz_index_cache.get(key)
        if cached is not None and cached[0] == stamp:
            return cached[1]

        order = np.argsort(source_df['mz_array'].values, kind='stable')
        arrays = (
            source_df['mz_array'].values[order],
            source_df['time'].values[order],
            source_df['intensity_array'].values[order],
        )
        self._mz_index_cache[key] = (stamp, arrays)
        return arrays

    def _slice_mz(self, mz_low: float, mz_high: float, rt_range=None, filtered: bool = True):
        """Long-form [time, mz_array, intensity_array] rows inside an m/z (and optional RT) window.

        Equivalent to boolean-masking the whole frame, but the m/z bound is resolved with
        np.searchsorted against the cached sorted index, so cost scales with the number of rows
        returned rather than with the size of the file.
        """
        idx = self._get_mz_index(filtered)
        if idx is None:
            return pd.DataFrame(columns=['time', 'mz_array', 'intensity_array'])

        mz_sorted, time_sorted, int_sorted = idx
        lo = int(np.searchsorted(mz_sorted, mz_low, side='left'))
        hi = int(np.searchsorted(mz_sorted, mz_high, side='right'))
        if hi <= lo:
            return pd.DataFrame(columns=['time', 'mz_array', 'intensity_array'])

        times, mzs, ints = time_sorted[lo:hi], mz_sorted[lo:hi], int_sorted[lo:hi]
        if rt_range:
            keep = (times >= rt_range[0]) & (times <= rt_range[1])
            times, mzs, ints = times[keep], mzs[keep], ints[keep]

        # A fresh RangeIndex, unlike the duplicate-heavy index the exploded ms1 frame carries --
        # which is what makes .loc[idxmax] lookups downstream unambiguous.
        return pd.DataFrame({'time': times, 'mz_array': mzs, 'intensity_array': ints})

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

        if mz_range:
            return self._slice_mz(mz_range[0], mz_range[1], rt_range=rt_range, filtered=filtered)

        if rt_range:
            source_df = source_df[source_df['time'].between(rt_range[0], rt_range[1])]

        return source_df

    def get_extracted_ion_chromatogram(self, mz, filtered=True, tolerance_ppm=20, use_ppm=True, tolerance_da=0.05,
                                       rt_range=None):
        if self.ms1 is None:
            raise RuntimeError("Data must be loaded first using .load().")

        source_df = self.ms1_filtered if filtered else self.ms1
        if source_df is None or source_df.empty:
            print("Warning: Source dataframe for XIC is empty.")
            return pd.DataFrame()

        if use_ppm:
            tolerance = mz * (tolerance_ppm / 1e6)
        else:
            tolerance = tolerance_da

        return self._slice_mz(mz - tolerance, mz + tolerance, rt_range=rt_range, filtered=filtered)

    def get_extracted_compound_chromatogram(self, mass: float, filtered=True, tolerance_ppm=20, use_ppm=True,
                                            tolerance_da=0.05, mz_range=[400, 2000], rt_range=None):
        if self.ms1 is None:
            raise RuntimeError("Data must be loaded first using .load().")

        source_df = self.ms1_filtered if filtered else self.ms1
        if source_df is None or source_df.empty:
            print("Warning: Source dataframe for XIC is empty.")
            return pd.DataFrame()

        # NOTE: no full-frame RT mask here. get_extracted_ion_chromatogram already applies rt_range
        # per m/z window via the sorted-index fast path, so masking the whole ~3.5M-row frame first
        # (once per compound) was pure redundant work -- it dominated the remaining runtime.
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

        # Populated later by AnalysisResults.expand_and_initialize_targets().
        self.target_peptides = []
        print(f"--- MSExperiment initialized for {self.spectrum_id} ---")

    @property
    def mzml_path(self) -> Path:
        """Returns the path to the .mzML file. Deliberately does NOT require the file to exist: the
        raw mzML is only actually read on a spectrum-cache MISS (MassSpectrum._parse_mzml validates
        it there). Validating here instead made every consumer that merely CONSTRUCTS a MassSpectrum
        -- e.g. SI-report rendering, which normally serves the representative replicate straight from
        its cached _ms1/_ms2 pickles -- crash whenever only the raw mzML (not the cache) was missing,
        such as when the data has been moved and the original mzMLs aren't on the current disk."""
        return Path(self.input_mzml_folder) / self.input_mzml

    @property
    def output_path(self) -> Path:
        """Returns the path to the output folder, creating it if it doesn't exist."""
        path = Path(self.output_folder_name)
        os.makedirs(to_long_path(path), exist_ok=True)  # Creates the folder if needed (long-path-safe)
        return path

    @property
    def spectrum_id(self) -> str:
        """Creates a unique ID for the spectrum from the metadata."""
        return f"{self.sample_id}_{self.experiment_id}"

    @classmethod
    def load(cls, filepath: Path | str) -> 'MSExperiment':
        """Static factory helper to cleanly reload an MSExperiment metadata file."""
        filepath = Path(filepath)
        if not os.path.exists(to_long_path(filepath)):
            raise FileNotFoundError(f"No saved experiment metadata found at: {filepath}")

        with open(to_long_path(filepath), 'rb') as f:
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
        save_path_long = to_long_path(save_path)
        if os.path.exists(save_path_long) and renew:
            print(f"Experiment file already exists: {save_path}. Overwriting due to renew=True.")
            # we remove the existing file
            os.remove(save_path_long)
            # Save the object
            with open(save_path_long, 'wb') as f:
                pkl.dump(self, f)

            print(f"Experiment saved to: {save_path}")
        elif not os.path.exists(save_path_long):
            # Save the object
            with open(save_path_long, 'wb') as f:
                pkl.dump(self, f)
            print(f"Experiment saved to: {save_path}")
