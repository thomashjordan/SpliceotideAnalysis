import numpy as np
import pandas as pd
import scipy as sp
import pyteomics as pyteomics
import warnings

import matplotlib as mpl
import seaborn as sns

from modification_dictionary import aa_comp_custom as aa_custom

from peptide_functions import asym_peak, interpolate_uneven_data


class PeptideFragment:
    """Represents a theoretical fragment of a parent peptide after orbitrap fragmentation."""

    def __init__(self, sequence: str, ion_type: str, charge: int, position: str, mod_dict: dict):
        self.sequence = sequence
        self.ion_type = ion_type  # e.g., 'b', 'y'
        self.charge = charge
        self.position = position  # e.g., b3, y5
        self.monoisotopic_mass = pyteomics.mass.calculate_mass(sequence=sequence, ion_type=ion_type, charge=charge, aa_comp=mod_dict)

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
        parsed_sequence = pyteomics.parser.parse(
            sequence=self.sequence,
            aa_comp=mod_dict,
            show_unmodified_termini=True
        )

        # Create a Composition object from the parsed sequence
        self.composition = pyteomics.mass.Composition(parsed_sequence, aa_comp=mod_dict)

        self.monoisotopic_mass = self.composition.mass()
        self.formula = self.composition
        self.fragments = []
        self.isotopologues = {'name': [], 'mass': [], 'formula': [], 'abundance': []}


    def get_isotopologues(self, mod_dict: dict, isotope_threshold: float = 0.05, include_monoisotopic: bool = True):
        elements_with_isotopes = ['C']
        iterator = pyteomics.mass.isotopologues(composition=self.composition, report_abundance=True,
                                        overall_threshold=isotope_threshold,
                                        elements_with_isotopes=elements_with_isotopes)
        isotopologues = [(f"isotopologue {i}", pyteomics.mass.calculate_mass(composition=comp), comp, abundance) for
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
        parsed_seq = pyteomics.parser.parse(self.sequence, aa_comp=mod_dict, modifications=self.modifications)
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

        # FIX: Pivot the long-form EIC into the wide-form expected by plotting methods.
        if extracted is not None and not extracted.empty:
            self.extracted = extracted.pivot_table(
                index='time',
                columns='isotopologue',
                values='intensity_array',
                aggfunc='sum'
            ).fillna(0)
        else:
            self.extracted = pd.DataFrame()

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
        if not self.extracted.empty and self.rt_start is not None and self.rt_end is not None:
            self.zoomed_extracted = self.extracted.loc[self.rt_start:self.rt_end].copy()
            # The total_intensity column is needed for fitting and plotting the zoomed EIC
            if not self.zoomed_extracted.empty:
                self.zoomed_extracted['total_intensity'] = self.zoomed_extracted.sum(axis=1, numeric_only=True)
        else:
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
    def create_stub(cls, experiment, target, charge: int, expected_rt: float, spectrum=None):
        """Factory method to generate a placeholder object when a peak is missing."""
        # The original method failed because it used get_extracted_compound_chromatogram,
        # which does not add the 'isotopologue' column required by the __init__ method's pivot_table.
        # This new implementation correctly builds the long-form DataFrame for all isotopologues.
        all_traces = []
        if spectrum is not None and hasattr(target, 'isotopologues'):
            iso_names = target.isotopologues.get("name", [])
            iso_masses = target.isotopologues.get("mass", [])

            for name, mass in zip(iso_names, iso_masses):
                # Calculate the specific m/z for this isotopologue and charge
                mz = (mass + charge * 1.007276466812) / charge

                # Extract the chromatogram for this single m/z
                trace = spectrum.get_extracted_ion_chromatogram(
                    mz=mz,
                    filtered=False,  # Use unfiltered data for stubs to get the full picture
                    tolerance_ppm=20,
                    rt_range=None  # Get full range for the EIC plot
                )

                if trace is not None and not trace.empty:
                    trace = trace.copy()
                    # Add the crucial 'isotopologue' column that was previously missing
                    trace['isotopologue'] = name
                    all_traces.append(trace)

        # This will be a long-form DataFrame with 'time', 'mz_array', 'intensity_array', and 'isotopologue'
        if all_traces:
            extracted_chromatogram = pd.concat(all_traces, ignore_index=True)
        else:
            extracted_chromatogram = pd.DataFrame()

        stub = cls(
            experiment=experiment,
            target=target,
            charge=charge,
            extracted=extracted_chromatogram,
            raw_apex=pd.DataFrame(),
            rt=expected_rt,
            ms2_data=pd.DataFrame(),
            is_stub=True,
        )
        return stub

    def _fill_stub_defaults(self):
        """Pre-fills match structures with non-detection placeholders to protect

        downstream functions.
        """
        self.best_matches = self._match_peaks_to_target(method_number=1)
        self.process_ms1_quality()

    def _estimate_peak_area(self):
        """Estimates peak area across extracted intensity dimensions."""
        if self.zoomed_extracted is None or self.zoomed_extracted.empty:
            self.trapz_area = 0.0
            return

        zoomed = self.zoomed_extracted

        # Pass numeric_only=True to prevent adding numeric floats with string columns
        if "total_intensity" not in zoomed.columns:
            zoomed['total_intensity'] = zoomed.sum(axis=1, numeric_only=True)
        self.trapz_area = float(np.trapz(zoomed["total_intensity"], x=zoomed.index))

    def _fit_asym_gaussian(self):
        if (
            self.is_stub
            or self.zoomed_extracted.empty
            or "total_intensity" not in self.zoomed_extracted.columns
            or self.trapz_area is None):
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
                asym_peak, self.zoomed_extracted.index,
                self.zoomed_extracted["total_intensity"], p0=parguess,
                maxfev=int(1e6), bounds=(
                    (100, 0.1, 0.001, 0.01, 100),
                    (10e11, self.zoomed_extracted.index.max(), 2, 100, 10e9)),method="trf")
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

    import warnings
    import scipy.signal as sp

    def _autopique_threshold_estimation(
            self, snr_level: int = 2, initial_threshold: int = None
    ):
        if self.raw_apex.empty:
            self.raw_apex_autopique = pd.DataFrame()
            return self.raw_apex_autopique
        # 1. Quick Exit: If raw_apex is empty or has fewer than 10 points, skip AutoPique entirely
        if self.raw_apex is None or self.raw_apex.empty or len(self.raw_apex) < 10:
            self.raw_apex_autopique = self.raw_apex.copy() if (
            self.raw_apex is not None and not self.raw_apex.empty) else pd.DataFrame()
            self.autopique_threshold = 0.0
            self.autopique_lags = np.array([])
            self.autopique_autocorr = np.array([])
            return self.raw_apex_autopique

        mzs = self.raw_apex["mz_array"].values
        intensities = self.raw_apex["intensity_array"].values
        max_intensity = np.max(intensities)

        # 2. Quick Exit: Low signal or tight m/z range
        if max_intensity <= 100 or (np.max(mzs) - np.min(mzs)) < 0.5:
            self.raw_apex_autopique = self.raw_apex.copy()
            self.autopique_threshold = 0.0
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
        print(
            f"      [DEBUG AutoPique] raw_apex points={len(mzs)} | mass_spacing={self.mass_spacing} | resample_delta={resample_delta} | new_mzs size={len(new_mzs)}")

        if len(new_mzs) > 100000:
            print(f"      [WARNING] AutoPique array is massive ({len(new_mzs)} points)! SciPy correlate will hang!")

        step_size = max(max_intensity / 100.0, 0.5)
        best_threshold = 0.0
        lags_final, autocorr_final = np.array([]), np.array([])

        for i in range(100):
            threshold = step_size * i
            if threshold >= max_intensity:
                break

            filtered = np.where(new_intensities < threshold, 0, new_intensities)
            autocorr = sp.signal.correlate(filtered, filtered, method="fft")

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

            # Skip flat/constant windows where peak_widths would calculate 0 width
            if np.ptp(autocorr_window) == 0:
                continue

            prop_below = (
                    len(autocorr_window[autocorr_window <= 0]) / len(autocorr_window)
            )
            peak_index = np.argmax(autocorr_window)

            # Suppress peak property warnings locally
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                peak_widths = sp.signal.peak_widths(
                    autocorr_window, [peak_index], rel_height=0.95
                )

            if len(peak_widths[2]) > 0 and len(peak_widths[3]) > 0:
                peak_left = int(peak_widths[2][0])
                peak_right = int(peak_widths[3][0])
                width = peak_right - peak_left

                # SKIP peaks that have a width of 0
                if width <= 0:
                    continue

                prop_width = width / len(autocorr_window)

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

        max_intensity = np.max(intensities)
        if max_intensity <= 0:
            return pd.DataFrame(), mzs, intensities

        # 3. SET INTENSITY HEIGHT NOISE FLOOR
        if data_source == "autopique" and getattr(self, 'autopique_threshold', None) is not None:
            abs_height = max(height, self.autopique_threshold * 1.5)
        else:
            abs_height = max(height, max_intensity * 0.02)  # Require at least 2% of max intensity

        peaks_indices = []
        properties = {}

        # 4. PEAK FINDING
        if method == "find_peaks":
            peaks_indices, properties = sp.signal.find_peaks(
                    intensities,
                    distance=distance_points,
                    prominence=prominence,
                    height=abs_height,
                    rel_height=rel_height)

        elif method == "find_peaks_cwt" and data_source != "raw":
            R = 17500
            min_width_pts = (np.min(mzs) / R) / max(bin_size, 1e-6)
            max_width_pts = (np.max(mzs) / R) / max(bin_size, 1e-6)
            cwt_widths = np.arange(
                max(1, int(min_width_pts * 0.8)), int(max_width_pts * 1.2) + 1)

            approx_indices = sp.signal.find_peaks_cwt(
                    intensities, cwt_widths, min_snr=3, noise_perc=50)

            # Filter CWT peaks below height floor
            approx_indices = [idx for idx in approx_indices if intensities[idx] >= abs_height]

            snapped_indices = []
            window_cwt_peak = 5
            for idx in approx_indices:
                start = max(0, idx - window_cwt_peak)
                end = min(len(intensities), idx + window_cwt_peak + 1)
                local_max_idx = np.argmax(intensities[start:end]) + start
                snapped_indices.append(local_max_idx)

            peaks_indices = np.unique(snapped_indices) if snapped_indices else []

        if len(peaks_indices) == 0:
            return pd.DataFrame(), mzs, intensities

        # 5. CONSTRUCT RESULTS DATAFRAME
        peaks_df = pd.DataFrame({
            "mz_array_peak": mzs[peaks_indices],
            "intensity_array_peak": intensities[peaks_indices],
            "data_source": data_source,
            "method": method,})

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
        if self.mass_spacing is None:
            print("      [DEBUG] Mass spacing could not be estimated. Skipping MS1 peak finding.")
            return pd.DataFrame(), pd.DataFrame()
        print(f"      [DEBUG] Estimated mass spacing: {self.mass_spacing:.5f}")

        self._interpolate_peak_mzs(interp_dist=bin_size)
        if self.raw_apex_interp is None or self.raw_apex_interp.empty:
            print("Processing failed at interpolation step.")
            scipy_peaks_1 = pd.DataFrame()
            scipy_peaks_2 = pd.DataFrame()
            return scipy_peaks_1, scipy_peaks_2
        print(f"      [DEBUG] Interpolated MS1 has {len(self.raw_apex_interp)} points.")

        self._bin_peak_mzs(bin_size=bin_size)
        if self.raw_apex_bin is None or self.raw_apex_bin.empty:
            print("Processing failed at binning step.")
            scipy_peaks_1 = pd.DataFrame()
            scipy_peaks_2 = pd.DataFrame()
            return scipy_peaks_1, scipy_peaks_2
        print(f"      [DEBUG] Binned MS1 has {len(self.raw_apex_bin)} points.")

        # 2. Second step
        self._autopique_threshold_estimation(snr_level=snr_level, initial_threshold=autopique_init)
        if self.raw_apex_autopique is None or self.raw_apex_autopique.empty:
            print("Processing failed at autopique thresholding step.")
            scipy_peaks_1 = pd.DataFrame()
            scipy_peaks_2 = pd.DataFrame()
            return scipy_peaks_1, scipy_peaks_2
        print(f"      [DEBUG] Autopique threshold: {self.autopique_threshold:.2f}. Filtered MS1 has {len(self.raw_apex_autopique)} points.")

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
            fig, ax = mpl.pyplot.subplots(figsize=(8, 4))
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
            fig, ax = mpl.pyplot.subplots(figsize=(8, 4))
        ax.set_title("Zoomed Extracted Ion Chromatogram", fontsize=12)

        if self.zoomed_extracted is not None and not self.zoomed_extracted.empty:
            ax.plot(self.zoomed_extracted.index, self.zoomed_extracted["total_intensity"], color="k", label="Total EIC")
            # Display trapezoidal area using relative axis coordinates to prevent clipping.
            trapz_area = getattr(self, 'trapz_area', None)
            if trapz_area is not None:
                ax.text(0.05, 0.9, f"Trapz. Area: {trapz_area:,.0f}", transform=ax.transAxes, fontsize=9, va='center')
            else:
                ax.text(0.05, 0.9, "Trapz. Area: N/A", transform=ax.transAxes, fontsize=9, va='center')

            if self.gaussian_params is not None:
                x_fit = np.linspace(self.zoomed_extracted.index.min(), self.zoomed_extracted.index.max(), 100)
                y_fit = asym_peak(x_fit, *self.gaussian_params)
                ax.plot(x_fit, y_fit, color="red", linestyle="--", label="Fitted Asym Gaussian")
                ax.text(0.05, 0.8, f"Gaussian Area: {self.gaussian_params[0]:,.0f}", transform=ax.transAxes, fontsize=9, va='center')
            else:
                ax.text(0.05, 0.8, "Gaussian Area: N/A", transform=ax.transAxes, fontsize=9, va='center')
            ax.legend(prop={"size": 8})
        else:
            # Handle case for stubs or empty data by displaying placeholders
            ax.text(0.5, 0.5, "No Zoomed Data Available", ha='center', va='center', transform=ax.transAxes, fontsize=10, color='grey')
            ax.text(0.05, 0.9, "Trapz. Area: N/A", transform=ax.transAxes, fontsize=9, va='center')
            ax.text(0.05, 0.8, "Gaussian Area: N/A", transform=ax.transAxes, fontsize=9, va='center')

        ax.set_xlabel("Retention Time (min)")
        ax.set_ylabel("Intensity")
        return ax

    def plot_ms1_raw(self, ax=None):
        """Plots the raw profile data."""
        if ax is None:
            fig, ax = mpl.pyplot.subplots(figsize=(8, 4))
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
            fig, ax = mpl.pyplot.subplots(figsize=(8, 4))
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
            fig, ax = mpl.pyplot.subplots(figsize=(8, 4))
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
            fig, ax = mpl.pyplot.subplots(figsize=(8, 6))
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
            fig, ax = mpl.pyplot.subplots(figsize=(8, 4))
        ax.axis("off")

        # Add title with tier and score
        tier = getattr(self, 'tier', 'N/A')
        score = getattr(self, 'quality_score', float('nan'))
        ax.set_title(f"Isotope Match Summary (Tier: {tier}, Score: {score:.2f})", fontsize=10)

        if self.best_matches is None or self.best_matches.empty:
            ax.text(0.5, 0.5, "No match data available.", ha='center', va='center', transform=ax.transAxes)
            return ax

        # Define column mapping and order
        col_map = {
            "isotopologue": "isotopologue",
            "theoretical_mz": "theor. mz",
            "observed_mz": "obs. mz",
            "error_ppm": "err. ppm",
            "theoretical_abundance": "theor. abnd.",
            "observed_abundance_norm": "obs. abnd.",
            "other_peak_in_window": "other peak"
        }

        # Select, rename, and format data for the table
        table_data = self.best_matches[list(col_map.keys())].copy()
        table_data.rename(columns=col_map, inplace=True)

        # Format float columns for display
        table_data['theor. mz'] = table_data['theor. mz'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        table_data['obs. mz'] = table_data['obs. mz'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        table_data['err. ppm'] = table_data['err. ppm'].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A')
        table_data['theor. abnd.'] = table_data['theor. abnd.'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        table_data['obs. abnd.'] = table_data['obs. abnd.'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        table_data['other peak'] = table_data['other peak'].astype(str)

        table = ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)
        return ax

    def plot_peptide_sequence(self, ax=None):
        """Draws sequence strings showing structural b and y ion fragmentation assignments."""
        if ax is None:
            fig, ax = mpl.pyplot.subplots(figsize=(8, 2))
        ax.axis("off")
        ax.set_title("Peptide Sequence with Modifications", fontsize=10)

        # Assumes peptide_parser is accessible in global runtime namespace
        parsed_modx_seq = pyteomics.parser.parse(self.target.sequence)
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
        fig = mpl.pyplot.figure(constrained_layout=True, figsize=(16, 14))
        gs_main = mpl.gridspec.GridSpec(4, 2, figure=fig)

        # Construct the two-line suptitle
        exp = self.experiment
        line1_parts = [
            f"Experiment: {getattr(exp, 'experiment_id', 'N/A')}",
            f"Replicate: {getattr(exp, 'sample_id', 'N/A')}",
            f"Precursor: {getattr(exp, 'precursor_name', 'N/A')}",
            f"Maturase: {getattr(exp, 'maturase_plasmid', 'N/A')}",
            f"Temp: {getattr(exp, 'expression_temperature', 'N/A')}",
            f"Digestion: {getattr(exp, 'digestion', 'N/A')}"
        ]
        line1 = " | ".join(line1_parts)
        line2 = f"MS1 Peak Analysis for {self.target.sequence}, z={self.charge} at RT={self.rt:.2f} min"

        # Adjust suptitle `y` position to prevent it from being clipped by the layout manager.
        fig.suptitle(f"{line1}\n{line2}", fontsize=12, y=1.02)

        self.plot_ms1_eic(ax=fig.add_subplot(gs_main[0, 0]), time_window_ecc=time_window_ecc)
        self.plot_ms1_zoomed(ax=fig.add_subplot(gs_main[0, 1]))

        ax_raw = fig.add_subplot(gs_main[1, 0])
        self.plot_ms1_raw_binned(ax=ax_raw)
        self.plot_ms1_autopique(ax=fig.add_subplot(gs_main[1, 1], sharex=ax_raw))

        self.plot_ms2_mirrored(ax=fig.add_subplot(gs_main[2:, 0]))
        self.plot_summary_table(ax=fig.add_subplot(gs_main[2, 1]))
        self.plot_peptide_sequence(ax=fig.add_subplot(gs_main[3, 1]))

        # Remove tight_layout() as it can conflict with constrained_layout=True set on the figure.
        # constrained_layout is generally more robust.
        mpl.pyplot.show()

    def __repr__(self):
        return f"<PeptidePeak target='{self.target.sequence}' charge={self.charge} rt={self.rt:.2f}min>"
