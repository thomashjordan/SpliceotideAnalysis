import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Align import substitution_matrices
import os
import tempfile
import subprocess
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

PRESETS = {
    'paper': {
        'fig_dpi': 300,
        'bar_size': 0.05,
        'barchart_width': 1.4,
        'meta_width': 0.8,  # New width for left-side metadata
        'spacer': 0.15,
        'aa_fontsize': 4.5,
        'box_edgecolor': '#d3d3d3',
        'box_linewidth': 0.4,
        'label_fontsize': 8,
        'tick_fontsize': 7,
        'legend_fontsize': 7,
        'suptitle_fontsize': 10,
        'capsize': 0.08
    },
    'poster': {
        'fig_dpi': 300,
        'bar_size': 0.14,
        'barchart_width': 3.0,
        'meta_width': 2.0,  # New width for left-side metadata
        'spacer': 0.35,
        'aa_fontsize': 12.0,
        'box_edgecolor': '#b0b0b0',
        'box_linewidth': 0.7,
        'label_fontsize': 14,
        'tick_fontsize': 12,
        'legend_fontsize': 12,
        'suptitle_fontsize': 18,
        'capsize': 0.12
    },
    'poster_big': {
        'fig_dpi': 300,
        'bar_size': 0.25,
        'barchart_width': 4.00,
        'meta_width': 3.0,  # New width for left-side metadata
        'spacer': 0.5,
        'aa_fontsize': 16.0,
        'box_edgecolor': '#b0b0b0',
        'box_linewidth': 1,
        'label_fontsize': 18,
        'tick_fontsize': 14,
        'legend_fontsize': 18,
        'suptitle_fontsize': 24,
        'capsize': 0.2
    }
}

COLOR_SCHEMES = {
    'hydrophobicity': {
        'A': '#808080', 'I': '#000000', 'L': '#000000', 'M': '#000000', 'F': '#000000',
        'W': '#000000', 'Y': '#000000', 'V': '#000000', 'C': '#808080', 'P': '#808080',
        'G': '#808080', 'T': '#008000', 'S': '#008000', 'Q': '#008000', 'N': '#008000',
        'E': '#FF0000', 'D': '#FF0000', 'H': '#0000FF', 'K': '#0000FF', 'R': '#0000FF',
        '-': 'none' # Handle alignment gaps smoothly
    },
    'clustal': {
        'G': '#F0F0F0', 'P': '#F0F0F0', 'S': '#F0F0F0', 'T': '#F0F0F0', 'A': '#0000FF',
        'V': '#0000FF', 'L': '#0000FF', 'I': '#0000FF', 'M': '#0000FF', 'C': '#0000FF',
        'F': '#FF0000', 'Y': '#FF0000', 'W': '#FF0000', 'H': '#FF00FF', 'K': '#00FF00',
        'R': '#00FF00', 'Q': '#800080', 'N': '#800080', 'E': '#FF0000', 'D': '#FF0000',
        '-': 'none' # Handle alignment gaps smoothly
    }
}
# --- Data Processing Functions ---
# Your MUSCLE path here
MUSCLE_EXE_PATH = r"C:\Users\jordant\bioinformatics\muscle5_1_win64.exe"


def align_muscle(input_path: str, output_path: str) -> int:
    """Core subprocess call provided by user."""
    subprocess.check_output([MUSCLE_EXE_PATH, "-align", input_path, "-output", output_path])
    return 0


def get_muscle_msa(sequences: list, work_dir: str = ".") -> dict:
    """
    Takes a list of raw sequences, writes them to a standard FASTA file,
    runs MUSCLE 5, and returns a dict mapping {raw_sequence: aligned_sequence}.
    Uses static file names to avoid Windows tempfile permission errors.
    """
    unique_seqs = list(set([s for s in sequences if pd.notna(s)]))
    aligned_map = {}

    # Define static file paths in the current directory (or provided work_dir)
    in_fasta = os.path.join(work_dir, "muscle_temp_in.fa")
    out_fasta = os.path.join(work_dir, "muscle_temp_out.fa")

    # 1. Write unique sequences to the input FASTA
    records = [
        SeqRecord(Seq(seq), id=f"seq_{i}", description="")
        for i, seq in enumerate(unique_seqs)
    ]
    SeqIO.write(records, in_fasta, "fasta")

    # 2. Run MUSCLE
    try:
        align_muscle(in_fasta, out_fasta)
    except subprocess.CalledProcessError as e:
        print(f"MUSCLE alignment failed: {e}")
        return {s: s for s in unique_seqs}  # Fallback to unaligned on failure

    # 3. Parse output and map back to original sequences
    if os.path.exists(out_fasta):
        aligned_records = SeqIO.parse(out_fasta, "fasta")
        id_to_seq = {f"seq_{i}": seq for i, seq in enumerate(unique_seqs)}

        for rec in aligned_records:
            orig_seq = id_to_seq[rec.id]
            aligned_map[orig_seq] = str(rec.seq).upper()

    # 4. Attempt silent cleanup (ignores errors if Windows is still locking the file)
    try:
        if os.path.exists(in_fasta): os.remove(in_fasta)
        if os.path.exists(out_fasta): os.remove(out_fasta)
    except OSError:
        pass  # Just leave them to be overwritten on the next run

    return aligned_map

def find_and_parse_reports(root_dir: str) -> list:
    """
    Finds all 'plotting_data.pkl' files in a directory and loads them.
    """
    report_files = Path(root_dir).rglob('plotting_data.pkl')
    all_data = []
    for file_path in report_files:
        try:
            with open(file_path, 'rb') as f:
                data_list = pickle.load(f)
                all_data.extend(data_list)
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Warning: Could not load or parse {file_path}. Error: {e}")
            continue
    return all_data

def create_long_form_master_dataframe(all_reports_data: list) -> pd.DataFrame:
    """
    Creates a detailed, long-form master DataFrame from all collected peaks.
    """
    peak_list = []
    for report in all_reports_data:
        cut_id = report.get('cut_id')
        charge = report.get('charge')
        for peak in report.get('collected_peaks', []):
            peak_info = {
                'experiment_id': peak.experiment.spectrum_id,
                'sample_id': peak.experiment.sample_id,
                'precursor_name': peak.experiment.precursor_name,
                'peptide_sequence': peak.experiment.peptide_sequence,
                'enzyme': getattr(peak.experiment, 'maturase_name', 'N/A'),
                'cut_id': cut_id,
                'charge': charge,
                'rt': peak.rt,
                'modification_state': str(peak.target.modifications)
            }
            qc_metrics = peak.quality_metrics or {}
            peak_info['area'] = qc_metrics.get('final_trapz_area')
            peak_info['tier'] = getattr(peak, 'tier', None)
            peak_list.append(peak_info)
    return pd.DataFrame(peak_list)


def find_group_result_files(reports_root: str) -> list:
    """Finds every curated ReplicateGroupResults pickle under a reports root directory --
    the canonical per-group output produced by peak_reviewing_pipeline.py."""
    return sorted(Path(reports_root).rglob("*_manually_reviewed.pkl.gz"))


def _lookup_selected_peak(rep_results, fragment_cut, modification, charge):
    """Finds the curated PeptidePeak backing one matrix row, to pull its real quality_score/tier
    (generate_modification_state_matrix only exposes median_error_ppm as 'qc_score', not the
    actual 0-1 quality_score used elsewhere in the pipeline)."""
    for target, peaks in getattr(rep_results, "selected_peaks", {}).items():
        if target.fragment_cut != fragment_cut or str(target.modifications) != modification:
            continue
        for p in peaks:
            if p.charge == charge:
                return p
    return None


def generate_si_reports(reports_root: str, plot_mz: bool = True, plot_ms2: bool = False,
                        dpi: int = 400, rt_range: tuple = None, caption_template: str = None) -> list:
    """Generates one single-page, high-resolution A4 SI PDF per group -- best (fragment_cut,
    charge) selected via GroupedExperiments.select_best_cut_charge() (fewest stub replicates
    first, then area, then quality), plotting only manually curated peaks (stubs are noted in the
    table, not plotted) -- for every '*_manually_reviewed.pkl.gz' found under `reports_root`.

    Each PDF is written into the SAME folder as the ReplicateGroupResults pickle it came from
    (not a separate output tree), and is tagged with a sequential page number (matching its
    position in this function's discovery order, and its eventual page position once merged via
    combine_pdfs()). Returns the ordered list of PDF paths actually written -- pass that straight
    to combine_pdfs() to build one combined document in the same order.

    caption_template: a str.format() template for the figure caption printed at the bottom of
    every page -- see GroupedExperiments.generate_si_report()'s docstring (or
    analysis_containers.DEFAULT_SI_CAPTION_TEMPLATE) for the default wording and the available
    {placeholder} fields. Passed through unchanged to every group's generate_si_report() call, so
    set it once here to apply the same caption layout across the whole batch.
    """
    from analysis_containers import ReplicateGroupResults, GroupedExperiments

    group_files = find_group_result_files(reports_root)
    print(f"Found {len(group_files)} manually reviewed group file(s) under: {reports_root}")

    written = []
    for i, file_path in enumerate(group_files):
        try:
            group_results = ReplicateGroupResults.load(file_path)
            grouped_experiment = GroupedExperiments.from_replicate_group_results(group_results)
            pdf_path = grouped_experiment.generate_si_report(
                file_path.parent, plot_mz=plot_mz, plot_ms2=plot_ms2, dpi=dpi, rt_range=rt_range,
                page_number=i + 1, source_path=file_path, caption_template=caption_template,
            )
            if pdf_path is not None:
                written.append(pdf_path)
                print(f"  Wrote: {pdf_path}")
        except Exception as e:
            print(f"  Warning: Could not generate SI report for {file_path.name}. Error: {e}")

    print(f"Generated {len(written)} SI PDF report(s) out of {len(group_files)} group(s).")
    return written


def combine_pdfs(pdf_paths: list, output_path: str) -> Path | None:
    """Merges a list of PDFs, in the given order, into one combined PDF file at `output_path`.
    Pages are copied at the PDF level (no re-rendering), so vector content stays exactly as-is.
    """
    if not pdf_paths:
        print("No PDF paths given -- nothing to combine.")
        return None

    try:
        from pypdf import PdfWriter
    except ImportError:
        raise ImportError(
            "Combining PDFs requires the 'pypdf' package. Install it with:\n    pip install pypdf"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = PdfWriter()
    for pdf_path in pdf_paths:
        writer.append(str(pdf_path))

    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"Combined {len(pdf_paths)} PDF(s) into: {output_path}")
    return output_path


def generate_and_combine_si_reports(reports_root: str, combined_output_path: str, plot_mz: bool = True,
                                    plot_ms2: bool = False, dpi: int = 400, rt_range: tuple = None,
                                    caption_template: str = None) -> Path | None:
    """Convenience wrapper: generate_si_reports() then combine_pdfs() in one call."""
    pdf_paths = generate_si_reports(reports_root, plot_mz=plot_mz, plot_ms2=plot_ms2, dpi=dpi, rt_range=rt_range,
                                    caption_template=caption_template)
    return combine_pdfs(pdf_paths, combined_output_path)


def build_master_dataframe_from_groups(reports_root: str, required_grouping_cols: list = None) -> pd.DataFrame:
    """
    Builds the long-form master DataFrame directly from every group's curated
    ReplicateGroupResults object, replacing the older approach of aggregating
    '_CURATED_APPROVED.pkl' report files.

    For every '*_manually_reviewed.pkl.gz' found under `reports_root`, loads the
    ReplicateGroupResults and calls generate_modification_state_matrix() to get one row per
    (replicate, fragment_cut, charge, modification) -- including stub placeholders for anything
    never detected. Each row is then tagged with that replicate's experiment metadata
    (required_grouping_cols plus sample_id/precursor_name/peptide_sequence/core coordinates)
    pulled directly from its MSExperiment object. No separate metadata Excel merge is needed:
    that metadata already lives on the experiment objects embedded in the results, since they
    were set as attributes from the original input sheet row when the experiment was created.
    """
    from analysis_containers import ReplicateGroupResults

    if required_grouping_cols is None:
        required_grouping_cols = [
            "experiment_id",
            "precursor_plasmid",
            "maturase_name",
            "maturase_plasmid",
            "expression_temperature",
            "digestion",
        ]

    group_files = find_group_result_files(reports_root)
    print(f"Found {len(group_files)} manually reviewed group file(s) under: {reports_root}")

    df_list = []
    for file_path in group_files:
        try:
            group_results = ReplicateGroupResults.load(file_path)
        except Exception as e:
            print(f"  Warning: Could not load {file_path.name}. Error: {e}")
            continue

        df_long, _ = group_results.generate_modification_state_matrix()
        if df_long.empty:
            continue

        # Attach per-replicate experiment metadata straight from the MSExperiment objects --
        # this is what makes the required_grouping_cols (and a few other columns the downstream
        # plotting/notebook code relies on) available on every row.
        meta_records = {}
        for rep_id, rep_results in group_results.replicates.items():
            experiment = rep_results.experiment
            meta = {col: getattr(experiment, col, None) for col in required_grouping_cols}
            meta.update({
                "sample_id": getattr(experiment, "sample_id", None),
                "precursor_name": getattr(experiment, "precursor_name", None),
                "peptide_sequence": getattr(experiment, "peptide_sequence", None),
                "core_start": getattr(experiment, "core_start", 0),
                "core_end": getattr(experiment, "core_end", None),
            })
            meta_records[rep_id] = meta

        meta_df = pd.DataFrame.from_dict(meta_records, orient="index")
        meta_df.index.name = "replicate_id"
        meta_df = meta_df.reset_index()
        df_long = df_long.merge(meta_df, on="replicate_id", how="left")

        # Aliases expected by the existing plotting/aggregation helpers.
        df_long["group_name"] = group_results.group_name
        df_long["cut_id"] = df_long["fragment_cut"]
        df_long["area"] = df_long["absolute_area"]
        df_long["modification_state"] = df_long["modification"]
        df_long["target_sequence"] = df_long["peptide_sequence"]
        df_long["peptide_sequence_meta"] = df_long["peptide_sequence"]
        df_long["enzyme"] = df_long["maturase_name"] if "maturase_name" in df_long.columns else "N/A"
        df_long["_source_file"] = file_path.name

        # Pull the real (0-1, higher-is-better) quality_score/tier off each curated peak object,
        # since the matrix's own 'qc_score' column is actually just the median mass error in ppm.
        quality_scores, tiers = [], []
        for _, row in df_long.iterrows():
            rep_results = group_results.replicates[row["replicate_id"]]
            peak = _lookup_selected_peak(rep_results, row["fragment_cut"], row["modification"], row["charge"])
            quality_scores.append(getattr(peak, "quality_score", np.nan) if peak is not None else np.nan)
            tiers.append(getattr(peak, "tier", None) if peak is not None else None)
        df_long["quality_score"] = quality_scores
        df_long["tier"] = tiers

        df_list.append(df_long)

    if not df_list:
        print("No data could be aggregated -- check that reports_root contains "
              "*_manually_reviewed.pkl.gz files.")
        return pd.DataFrame()

    master_df = pd.concat(df_list, ignore_index=True)
    print(f"Master DataFrame assembled: {len(master_df)} rows across {len(df_list)} group(s).")
    return master_df


def extract_best_cut_charge(master_df: pd.DataFrame, group_col: str = "group_name") -> pd.DataFrame:
    """
    For every experimental group (one ReplicateGroupResults -- many replicates x fragment_cuts x
    charges x modifications), picks the single (cut_id, charge) combination with the best average
    quality_score, breaking ties by the highest average area -- both averaged across every
    replicate and modification state that shares that combination. Stub/undetected rows are
    excluded from the averaging since they carry no real signal (quality_score is NaN for them,
    and including their area=0 would just penalize combinations with more missing replicates
    rather than reflect the quality of what was actually detected).

    Returns a new DataFrame containing every replicate and modification row for just that winning
    (cut_id, charge) pair, per group -- i.e. one fixed cut/charge choice per group, with all
    replicates and modification states for it still intact for collapse_modification_replicates().
    """
    df = master_df.copy()
    detected = df[~df["is_stub"]]

    if detected.empty:
        print("Warning: no non-stub rows found -- cannot select a best cut/charge for any group.")
        return df.iloc[0:0].copy()

    cut_charge_stats = (
        detected.groupby([group_col, "cut_id", "charge"])
        .agg(non_stub=("is_stub", "size"), avg_quality_score=("quality_score", "mean"), avg_area=("area", "mean"))
        .reset_index()
    )

    # Rank within each group by non-stubs first, area second, quality third (both higher-is-better).
    cut_charge_stats = cut_charge_stats.sort_values(
        by=["non_stub", "avg_area", "avg_quality_score"], ascending=[False, False, False]
    )
    best_per_group = cut_charge_stats.groupby(group_col, as_index=False).first()

    merged = df.merge(
        best_per_group[[group_col, "cut_id", "charge"]],
        on=[group_col, "cut_id", "charge"],
        how="inner",
    )
    return merged


def collapse_modification_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapses every modification state down to just the unmodified peptide, per replicate, adding
    'total_modification_ratio' = 1 - (unmodified peptide's area / overall_signal_intensity) --
    i.e. the fraction of that replicate's total detected signal (across all modification states
    at its cut/charge) that is modified. Also aliased as 'mod_ratio' for compatibility with
    create_grouped_alignment_plot(), which expects that exact column name.

    Expects to be run on the output of extract_best_cut_charge() (one cut/charge per group), so
    that filtering down to the unmodified row leaves exactly one row per replicate.
    """
    df = df.copy()
    is_unmod_mask = df["modification_state"].apply(
        lambda x: "'unmodified': None" in str(x) or "unmodified" in str(x).lower()
    )
    unmod_df = df[is_unmod_mask].copy()

    unmodified_fraction = np.where(
        unmod_df["overall_signal_intensity"] > 0,
        unmod_df["area"] / unmod_df["overall_signal_intensity"],
        np.nan,
    )
    unmod_df["total_modification_ratio"] = 1 - unmodified_fraction
    unmod_df["mod_ratio"] = unmod_df["total_modification_ratio"]

    # Match the old pipeline's behavior of exposing a clean, uppercased unmodified sequence as
    # 'target_sequence' for the alignment plots.
    unmod_df["target_sequence"] = (
        unmod_df["peptide_sequence"].astype(str).str.strip().str.upper().str.replace("*", "", regex=False)
    )

    return unmod_df


def calculate_charges_mod_ratios(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates modification ratios by aggregating across charge states/clusters.
    Completely retains all original metadata columns (group_name, enzyme, is_stub, etc.)
    by copying them directly from the genuine unmodified sequence row.
    """
    df = master_df.copy()

    # 2. Define our grouping keys (primarily experiment and sample)
    group_cols = ['experiment_id', 'sample_id']
    group_cols = [c for c in group_cols if c in df.columns]

    grouped = df.groupby(group_cols)
    results = []

    for name, group in grouped:
        # Calculate areas across all rows in this sample group
        total_effective_area = group['effective_area'].sum()

        # Identify modified entries versus unmodified entries
        is_unmod_mask = group['modification_state'].apply(
            lambda x: "'unmodified': None" in str(x) or "unmodified" in str(x)
        )
        modified_group = group[~is_unmod_mask]
        modified_effective_area = modified_group['effective_area'].sum()

        mod_ratio = (modified_effective_area / total_effective_area) if total_effective_area > 0 else 0

        # --- PRESERVE ALL CRUCIAL METADATA ---
        unmod_rows = group[is_unmod_mask]

        if not unmod_rows.empty:
            # Anchor metadata to the actual unmodified record row
            reference_row = unmod_rows.iloc[0].copy()
        else:
            # Fallback if no explicit unmodified row exists in this sample chunk
            reference_row = group.iloc[0].copy()
            # Clean modx characters if forced to fallback
            raw_seq = str(reference_row['peptide_sequence'])
            reference_row['peptide_sequence'] = "".join([c for c in raw_seq if c.isupper()])

        # Standardize sequence presentation to the uppercase clean string
        clean_sequence = str(reference_row['peptide_sequence']).strip().upper()

        # 3. Build the row dictionary dynamically keeping EVERY column from the original data
        row_data = reference_row.to_dict()

        # Override specific compressed/calculated attributes
        row_data['peptide_sequence'] = clean_sequence
        # Ensure the unmodified (clean) sequence is available as 'target_sequence'
        row_data['target_sequence'] = clean_sequence
        # Preserve the original source file (if present) to help backtrace where this
        # aggregated row originated from in the master dataset
        if '_source_file' in reference_row.index:
            row_data['_source_file'] = reference_row['_source_file']

        row_data['mod_ratio'] = mod_ratio
        # the modified ratio is actually the inverse of what we have here since we group unmodified
        # 1-relative ratio = mod_ratio



        # Remove columns that don't make sense as a single scalar value post-aggregation
        columns_to_drop = ['charge', 'cluster_id', 'rt_start', 'rt_end', 'rt_consensus',
                           'area', 'effective_area', 'peak_area', 'quality_score', 'modification_state']
        for col in columns_to_drop:
            row_data.pop(col, None)

        results.append(row_data)

    return pd.DataFrame(results)

def calculate_ddg_values(mod_ratio_series: pd.Series, ref_mod_ratio: float) -> pd.Series:
    """
    Calculates ddG values for a series of modification ratios relative to a reference.
    """
    if ref_mod_ratio == 0:
        ref_mod_ratio = 1e-9  # Avoid division by zero

    # Use a small epsilon for variants with zero modification
    mod_ratio_series = mod_ratio_series.replace(0, 1e-9)

    return -np.log(mod_ratio_series / ref_mod_ratio)


def get_provenance_map(df: pd.DataFrame, sequence_column: str = 'peptide_sequence') -> dict:
    """Return a mapping sequence -> semicolon-separated provenance filenames found in '_source_file'.

    If '_source_file' is missing, returns an empty string for that sequence.
    """
    prov = {}
    if '_source_file' not in df.columns:
        return prov

    grouped = df.groupby(sequence_column)['_source_file'].unique()
    for seq, arr in grouped.items():
        # join unique file names into a short semicolon-separated list
        prov[str(seq)] = ";".join(sorted([str(x) for x in arr if pd.notna(x)]))
    return prov


def print_merge_diagnostics(df: pd.DataFrame, indicator_col: str = 'merge_source', n: int = 10):
    """Print helpful diagnostics about a pandas merge that used an indicator column.

    Shows counts and the first `n` left_only rows (with provenance columns if present).
    """
    if indicator_col not in df.columns:
        print(f"Indicator column '{indicator_col}' not found in DataFrame.")
        return

    counts = df[indicator_col].value_counts().to_dict()
    print(f"Merge indicator counts: {counts}")
    if counts.get('left_only', 0) > 0:
        cols = ['_source_file', 'sample_id', 'experiment_id']
        cols = [c for c in cols if c in df.columns]
        print(f"First {n} rows that did not match metadata (left_only):")
        print(df[df[indicator_col] == 'left_only'][cols].head(n))

# --- Plotting and Helper Functions (Adapted from enzymeanalysis/enzymeplots) ---


def format_axis(ax):
    ax.xaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    ax.yaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    ax.spines['top'].set_linewidth(0.6)
    ax.spines['right'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.6)

matrix = substitution_matrices.load("BLOSUM62")


def seq_alignment(wt_sequence, sdf, score='ddg_mean', penalties=(-15, -2)):
    """
    Safely aligns variant sequences against a reference using standard substitution matrix configurations.
    """
    from Bio import pairwise2
    from Bio.Align import substitution_matrices

    # 1. Safely load standard BLOSUM62 without manual dictionary overrides
    try:
        blosum62 = substitution_matrices.load("BLOSUM62")
    except Exception:
        # Fallback if your environment requires the older SubsMat module style
        # Import may not be available in all environments; guard and provide
        # an empty mapping as last resort so the caller can handle missing matrix.
        # Avoid additional imports as some environments may not expose the old SubsMat
        # module. If the standard load failed and no fallback is available, provide an
        # empty mapping so the caller can handle matrix lookup failures gracefully.
        blosum62 = {}

    seq_alignments, labels, labels_sparse, labels_aligned = [], [wt_sequence], [wt_sequence], [wt_sequence]

    for _, row in sdf.iterrows():
        seq = str(row['sequence']).split("*")[0].upper().strip()
        mod_efficiency = row[score]

        # Strip out any pre-existing gap hyphens inside the raw sequence string
        clean_seq = seq.replace('-', '')
        if not clean_seq:
            continue

        try:
            # 2. Execute the global sequence alignment
            # 'globalds' uses a substitution matrix dictionary AND explicit gap penalties
            alignments = pairwise2.align.globalds(
                wt_sequence,
                clean_seq,
                blosum62,  # The pure substitution matrix dictionary
                penalties[0],  # Gap open penalty (e.g., -15)
                penalties[1]  # Gap extend penalty (e.g., -2)
            )

            if not alignments:
                continue
            alignment = alignments[0]

            if alignment[1] != wt_sequence:
                seq_alignments.append((seq, alignment[1], mod_efficiency))
                labels_sparse.append("".join([i if i != w else " " for i, w in zip(alignment[1], wt_sequence)]))
                labels.append(seq)
                labels_aligned.append(alignment[1])

        except (KeyError, SystemError) as e:
            # Silently catch and skip unexpected residue/gap tracking lookup failures
            print(f"⚠️ Skipping sequence variant '{clean_seq}' due to matrix mismatch error: {e}")
            continue

    return seq_alignments, labels_sparse, labels, labels_aligned


def matrix_subplot(labels, mat_ax, color_scheme=None, aa_fontsize=6,
                   box_edgecolor='#d3d3d3', box_linewidth=0.5, show_gaps=False):
    """Draws the sequence grid. Now dynamically accommodates the longest sequence string."""
    max_columns = max([len(x) for x in labels]) if labels else 0
    max_rows = len(labels)

    # Global background grid lines
    mat_ax.hlines(y=np.arange(-0.5, max_rows), xmin=-0.5, xmax=max_columns - 0.5, color='#e0e0e0', linewidth=0.5)
    mat_ax.vlines(x=np.arange(-0.5, max_columns), ymin=-0.5, ymax=max_rows - 0.5, color='#e0e0e0', linewidth=0.5)

    for row_index, label in enumerate(labels):
        for col_index, char in enumerate(label):
            if char == " " or not char: continue

            color = 'black'
            facecolor = 'none'

            if color_scheme and char in color_scheme:
                facecolor = color_scheme[char]
                color = 'white' if facecolor not in ['#F0F0F0', 'none'] else 'black'

            # Rectangular box patch
            square = plt.Rectangle(
                xy=(col_index - 0.5, row_index - 0.5),
                width=1, height=1,
                facecolor=facecolor,
                edgecolor=box_edgecolor if facecolor == 'none' else '#e0e0e0',
                linewidth=box_linewidth
            )
            mat_ax.add_patch(square)

            # Bold centered character text
            if not show_gaps:
                if char != '-':  # Don't print the hyphen text, just leave the blank box
                    mat_ax.text(col_index, row_index, char, color=color, ha="center", va="center",
                            fontsize=aa_fontsize, weight='bold')
            elif show_gaps:
                mat_ax.text(col_index, row_index, char, color=color, ha="center", va="center",
                            fontsize=aa_fontsize, weight='bold')

    mat_ax.set_xticks([])
    mat_ax.set_yticks([])
    mat_ax.set_xlim(-0.5, max_columns - 0.5)
    mat_ax.set_ylim(-0.5, max_rows - 0.5)
    mat_ax.invert_yaxis()


def create_grouped_alignment_plot(
        processed_df: pd.DataFrame,
        ref_sequence: str,
        hue_column: str = "maturase_name",  # NEW: Flexible grouping parameter
        metadata_cols: list = None,
        color_scheme_name: str = "hydrophobicity",
        show_ddg: bool = True,
        preset: str = 'paper',
        savepath: str = None,
        use_alignment: bool = False,
        show_gaps: bool = False,
        drop_identical: bool = False,
        sequence_column: str = "peptide_sequence",
        suptitle: str = None,
        seqtitle: str = None,
        metatitle: str = None,
        plot_baseline: bool = False
):
    plot_df = processed_df.copy()
    cfg = PRESETS.get(preset.lower(), PRESETS['paper'])
    metadata_cols = metadata_cols or []

    # Validate hue_column
    if hue_column not in plot_df.columns:
        raise ValueError(f"The specified hue_column '{hue_column}' was not found in the DataFrame.")

    # --- 1. Create Unique Row Identifiers ---
    if metadata_cols:
        plot_df['row_id'] = plot_df.apply(
            lambda r: f"{r[sequence_column]}_" + "_".join([str(r[c]) for c in metadata_cols]),
            axis=1
        )
    else:
        plot_df['row_id'] = plot_df[sequence_column]

    # --- 2. Correct ddG Calculation (Grouped by the flexible hue_column) ---
    ddg_dfs = []
    for hue_val, group in plot_df.groupby(hue_column):
        ref_mask = group[sequence_column] == ref_sequence
        ref_ratio = group[ref_mask]['mod_ratio'].mean() if ref_mask.any() else group['mod_ratio'].mean()
        if pd.isna(ref_ratio) or ref_ratio == 0: ref_ratio = 1.0

        group['ddg'] = calculate_ddg_values(group['mod_ratio'], ref_ratio)
        ddg_dfs.append(group)
    plot_df = pd.concat(ddg_dfs)

    # --- 3. Sort Rows (Row IDs) ---
    sort_basis = plot_df.groupby('row_id').agg({
        'mod_ratio': 'mean',
        sequence_column: 'first'
    }).reset_index()

    ref_rows = sort_basis[sort_basis[sequence_column] == ref_sequence].sort_values(by='mod_ratio', ascending=False)
    var_rows = sort_basis[sort_basis[sequence_column] != ref_sequence].sort_values(by='mod_ratio', ascending=False)

    ordered_row_ids = ref_rows['row_id'].tolist() + var_rows['row_id'].tolist()

    # --- 4. Alignment & Matrix Labels ---
    raw_sequences_in_order = [
        plot_df[plot_df['row_id'] == rid][sequence_column].iloc[0]
        for rid in ordered_row_ids
    ]

    labels = []
    if use_alignment:
        msa_map = get_muscle_msa(raw_sequences_in_order)
        aligned_ref = msa_map.get(ref_sequence, ref_sequence)

        for raw_seq in raw_sequences_in_order:
            aligned_seq = msa_map.get(raw_seq, raw_seq)

            if drop_identical and raw_seq != ref_sequence:
                masked = "".join([
                    ' ' if (i < len(aligned_ref) and aligned_seq[i] == aligned_ref[i]) else aligned_seq[i]
                    for i in range(len(aligned_seq))
                ])
                labels.append(masked)
            else:
                labels.append(aligned_seq)
    else:
        max_len = max([len(str(s)) for s in raw_sequences_in_order])
        ref_pad = ref_sequence.ljust(max_len)

        for raw_seq in raw_sequences_in_order:
            seq_pad = str(raw_seq).ljust(max_len)

            if drop_identical and raw_seq != ref_sequence:
                masked = "".join([
                    ' ' if (i < len(ref_pad) and seq_pad[i] == ref_pad[i]) else seq_pad[i]
                    for i in range(max_len)
                ])
                labels.append(masked)
            else:
                labels.append(seq_pad)

    # --- 5. Generate Left-Axis Metadata Tags ---
    meta_labels = []
    if metadata_cols:
        for rid in ordered_row_ids:
            row_data = plot_df[plot_df['row_id'] == rid].iloc[0]
            vals = [str(row_data.get(col, '')) for col in metadata_cols]
            meta_labels.append(" | ".join([v for v in vals if v and str(v).lower() != 'nan']))

    # --- 6. Dynamic Grid Coordinate Layout Engine ---
    longest_seq = max([len(l) for l in labels]) if labels else 0
    num_variants = len(ordered_row_ids)

    barchart_width = cfg['barchart_width']
    bar_size = cfg['bar_size']
    spacer = cfg['spacer']
    meta_width = cfg['meta_width'] if metadata_cols else 0.01

    matrix_width = (longest_seq + 1) * bar_size
    matrix_height = num_variants * bar_size * 1.5

    num_charts = 2 if show_ddg else 1
    total_width = meta_width + matrix_width + (num_charts * barchart_width) + ((num_charts + 3) * spacer)
    total_height = matrix_height + 2 * spacer

    fig = plt.figure(figsize=(total_width, total_height), dpi=cfg['fig_dpi'])
    curr_x = (spacer) / total_width

    if metadata_cols:
        ax_meta = plt.axes([curr_x, (spacer) / total_height, meta_width / total_width, matrix_height / total_height],
                           frameon=False)
        curr_x += (meta_width + spacer) / total_width

        ax_meta.set_ylim(-0.5, num_variants - 0.5)
        ax_meta.invert_yaxis()
        ax_meta.set_xticks([])
        ax_meta.set_yticks([])

        if metatitle:
            ax_meta.set_xlabel(metatitle, fontsize=cfg['label_fontsize'])

        for idx, text in enumerate(meta_labels):
            ax_meta.text(1.0, idx, text, ha='right', va='center', fontsize=cfg['tick_fontsize'], color='#404040')

    ax_mat = plt.axes([curr_x, (spacer) / total_height, matrix_width / total_width, matrix_height / total_height],
                      frameon=False)
    curr_x += (matrix_width + spacer) / total_width

    ax_ratio = plt.axes([curr_x, (spacer) / total_height, barchart_width / total_width, matrix_height / total_height])
    curr_x += (barchart_width + spacer) / total_width
    active_plots = [(ax_ratio, 'mod_ratio', 'Modification Ratio', 0)]

    if show_ddg:
        ax_ddg = plt.axes([curr_x, (spacer) / total_height, barchart_width / total_width, matrix_height / total_height])
        active_plots.append((ax_ddg, 'ddg', 'ddG (relative to ref)', 1))

    # --- 7. Render Matrix and Barplots ---
    color_map = COLOR_SCHEMES.get(color_scheme_name)
    matrix_subplot(labels, ax_mat, color_scheme=color_map, aa_fontsize=cfg['aa_fontsize'],
                   box_edgecolor=cfg['box_edgecolor'], box_linewidth=cfg['box_linewidth'], show_gaps=show_gaps)
    if seqtitle:
        ax_mat.set_xlabel(seqtitle, fontsize=cfg['label_fontsize'])

    # Use the flexible hue_column for the seaborn plots
    for ax, metric, xlabel, idx in active_plots:
        sns.barplot(data=plot_df, y='row_id', x=metric, hue=hue_column, order=ordered_row_ids, orient='h',
                    ax=ax, errorbar='sd', capsize=cfg['capsize'], linewidth=1, edgecolor='k')
        sns.stripplot(data=plot_df, y='row_id', x=metric, hue=hue_column, order=ordered_row_ids, orient='h',
                      ax=ax, dodge=True, jitter=0.2, size=2.5 if preset == 'paper' else 5.5, edgecolor='k', linewidth=1)

        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=cfg['label_fontsize'])
        ax.tick_params(axis='x', labelsize=cfg['tick_fontsize'])

        if metric == 'mod_ratio': ax.set_xlim(0, 1)

        # Dynamic legend handling based on the new hue_column
        handles, leg_labels = ax.get_legend_handles_labels()
        num_hues = len(plot_df[hue_column].unique())

        if idx == (len(active_plots) - 1):
            legend_title = hue_column.replace('_', ' ').title()
            ax.legend(handles[:num_hues], leg_labels[:num_hues], title=legend_title, loc='lower right',
                      fontsize=cfg['legend_fontsize'], title_fontsize=cfg['legend_fontsize'], framealpha=0.90)
        else:
            try:
                ax.get_legend().remove()
            except Exception:
                pass

    if suptitle:
        fig.suptitle(suptitle, fontsize=cfg['suptitle_fontsize'], y=1.02)
    elif metadata_cols and (suptitle is None):
        group_title = " | ".join([c.replace('_', ' ').title() for c in metadata_cols])
        fig.suptitle(f"Condition: {group_title}", fontsize=cfg['label_fontsize'] + 2, y=1.02)


    if savepath: plt.savefig(savepath, bbox_inches='tight')
    plt.show()


def create_grouped_alignment_plot(
        processed_df: pd.DataFrame,
        ref_sequence: str,
        hue_column: str = "maturase_name",
        metadata_cols: list = None,
        color_scheme_name: str = "hydrophobicity",
        show_ddg: bool = True,
        preset: str = 'paper',
        savepath: str = None,
        use_alignment: bool = False,
        show_gaps: bool = False,
        drop_identical: bool = False,
        sequence_column: str = "peptide_sequence",
        suptitle: str = None,
        seqtitle: str = None,
        metatitle: str = None,
        plot_baseline: bool = False
):
    plot_df = processed_df.copy()
    cfg = PRESETS.get(preset.lower(), PRESETS['paper'])
    metadata_cols = metadata_cols or []

    # Validate hue_column
    if hue_column not in plot_df.columns:
        raise ValueError(f"The specified hue_column '{hue_column}' was not found in the DataFrame.")

    # --- 1. Create Unique Row Identifiers ---
    if metadata_cols:
        plot_df['row_id'] = plot_df.apply(
            lambda r: f"{r[sequence_column]}_" + "_".join([str(r[c]) for c in metadata_cols]),
            axis=1
        )
    else:
        plot_df['row_id'] = plot_df[sequence_column]

    # --- 2. Correct ddG Calculation (Grouped by the flexible hue_column) ---
    ddg_dfs = []
    for hue_val, group in plot_df.groupby(hue_column):
        ref_mask = group[sequence_column] == ref_sequence
        ref_ratio = group[ref_mask]['mod_ratio'].mean() if ref_mask.any() else group['mod_ratio'].mean()
        if pd.isna(ref_ratio) or ref_ratio == 0: ref_ratio = 1.0

        group['ddg'] = calculate_ddg_values(group['mod_ratio'], ref_ratio)
        ddg_dfs.append(group)
    plot_df = pd.concat(ddg_dfs)

    # --- 3. Sort Rows (Row IDs) ---
    sort_basis = plot_df.groupby('row_id').agg({
        'mod_ratio': 'mean',
        sequence_column: 'first'
    }).reset_index()

    ref_rows = sort_basis[sort_basis[sequence_column] == ref_sequence].sort_values(by='mod_ratio', ascending=False)
    var_rows = sort_basis[sort_basis[sequence_column] != ref_sequence].sort_values(by='mod_ratio', ascending=False)

    ordered_row_ids = ref_rows['row_id'].tolist() + var_rows['row_id'].tolist()

    # --- 4. Alignment & Matrix Labels ---
    raw_sequences_in_order = [
        plot_df[plot_df['row_id'] == rid][sequence_column].iloc[0]
        for rid in ordered_row_ids
    ]

    labels = []
    if use_alignment:
        msa_map = get_muscle_msa(raw_sequences_in_order)
        aligned_ref = msa_map.get(ref_sequence, ref_sequence)

        for raw_seq in raw_sequences_in_order:
            aligned_seq = msa_map.get(raw_seq, raw_seq)

            if drop_identical and raw_seq != ref_sequence:
                masked = "".join([
                    ' ' if (i < len(aligned_ref) and aligned_seq[i] == aligned_ref[i]) else aligned_seq[i]
                    for i in range(len(aligned_seq))
                ])
                labels.append(masked)
            else:
                labels.append(aligned_seq)
    else:
        max_len = max([len(str(s)) for s in raw_sequences_in_order])
        ref_pad = ref_sequence.ljust(max_len)

        for raw_seq in raw_sequences_in_order:
            seq_pad = str(raw_seq).ljust(max_len)

            if drop_identical and raw_seq != ref_sequence:
                masked = "".join([
                    ' ' if (i < len(ref_pad) and seq_pad[i] == ref_pad[i]) else seq_pad[i]
                    for i in range(max_len)
                ])
                labels.append(masked)
            else:
                labels.append(seq_pad)

    # --- 5. Generate Left-Axis Metadata Tags ---
    meta_labels = []
    if metadata_cols:
        for rid in ordered_row_ids:
            row_data = plot_df[plot_df['row_id'] == rid].iloc[0]
            vals = [str(row_data.get(col, '')) for col in metadata_cols]
            meta_labels.append(" | ".join([v for v in vals if v and str(v).lower() != 'nan']))

    # --- 6. Dynamic Grid Coordinate Layout Engine ---
    longest_seq = max([len(l) for l in labels]) if labels else 0
    num_variants = len(ordered_row_ids)

    barchart_width = cfg['barchart_width']
    bar_size = cfg['bar_size']
    spacer = cfg['spacer']
    meta_width = cfg['meta_width'] if metadata_cols else 0.01

    matrix_width = (longest_seq + 1) * bar_size
    matrix_height = num_variants * bar_size * 1.5

    num_charts = 2 if show_ddg else 1
    total_width = meta_width + matrix_width + (num_charts * barchart_width) + ((num_charts + 3) * spacer)
    total_height = matrix_height + 2 * spacer

    fig = plt.figure(figsize=(total_width, total_height), dpi=cfg['fig_dpi'])
    curr_x = (spacer) / total_width

    if metadata_cols:
        ax_meta = plt.axes([curr_x, (spacer) / total_height, meta_width / total_width, matrix_height / total_height],
                           frameon=False)
        curr_x += (meta_width + spacer) / total_width

        ax_meta.set_ylim(-0.5, num_variants - 0.5)
        ax_meta.invert_yaxis()
        ax_meta.set_xticks([])
        ax_meta.set_yticks([])

        if metatitle:
            ax_meta.set_xlabel(metatitle, fontsize=cfg['label_fontsize'])

        for idx, text in enumerate(meta_labels):
            ax_meta.text(1.0, idx, text, ha='right', va='center', fontsize=cfg['tick_fontsize'], color='#404040')

    ax_mat = plt.axes([curr_x, (spacer) / total_height, matrix_width / total_width, matrix_height / total_height],
                      frameon=False)
    curr_x += (matrix_width + spacer) / total_width

    ax_ratio = plt.axes([curr_x, (spacer) / total_height, barchart_width / total_width, matrix_height / total_height])
    curr_x += (barchart_width + spacer) / total_width
    active_plots = [(ax_ratio, 'mod_ratio', 'Modification Ratio', 0)]

    if show_ddg:
        ax_ddg = plt.axes([curr_x, (spacer) / total_height, barchart_width / total_width, matrix_height / total_height])
        active_plots.append((ax_ddg, 'ddg', 'ddG (relative to ref)', 1))

    # --- 7. Render Matrix and Barplots ---
    color_map = COLOR_SCHEMES.get(color_scheme_name)
    matrix_subplot(labels, ax_mat, color_scheme=color_map, aa_fontsize=cfg['aa_fontsize'],
                   box_edgecolor=cfg['box_edgecolor'], box_linewidth=cfg['box_linewidth'], show_gaps=show_gaps)
    if seqtitle:
        ax_mat.set_xlabel(seqtitle, fontsize=cfg['label_fontsize'])

    # Use the flexible hue_column for the seaborn plots
    for ax, metric, xlabel, idx in active_plots:
        sns.barplot(data=plot_df, y='row_id', x=metric, hue=hue_column, order=ordered_row_ids, orient='h',
                    ax=ax, errorbar='sd', capsize=cfg['capsize'], linewidth=1, edgecolor='k')
        sns.stripplot(data=plot_df, y='row_id', x=metric, hue=hue_column, order=ordered_row_ids, orient='h',
                      ax=ax, dodge=True, jitter=0.2, size=2.5 if preset == 'paper' else 5.5, edgecolor='k', linewidth=1)

        ax.set_ylabel("")
        ax.set_yticks([])
        ax.set_xlabel(xlabel, fontsize=cfg['label_fontsize'])
        ax.tick_params(axis='x', labelsize=cfg['tick_fontsize'])

        if metric == 'mod_ratio': ax.set_xlim(0, 1)

        # Dynamic legend mapping metrics
        handles, leg_labels = ax.get_legend_handles_labels()
        num_hues = len(plot_df[hue_column].unique())

        # --- NEW: Hue-Aligned Vertical Baseline Mapping Engine ---
        if plot_baseline and ref_sequence in plot_df[sequence_column].values:
            try:
                ref_data = plot_df[plot_df[sequence_column] == ref_sequence]
                baseline_means = ref_data.groupby(hue_column)[metric].mean()

                # Link color maps directly from seaborn legend markers
                hue_colors = {}
                for h, l in zip(handles[:num_hues], leg_labels[:num_hues]):
                    if hasattr(h, 'get_facecolor'):
                        hue_colors[str(l)] = h.get_facecolor()
                    elif hasattr(h, 'get_color'):
                        hue_colors[str(l)] = h.get_color()

                for hue_val, mean_val in baseline_means.items():
                    if not pd.isna(mean_val):
                        line_color = hue_colors.get(str(hue_val), 'gray')
                        ax.axvline(x=mean_val, color=line_color, linestyle='--', linewidth=1.2, zorder=3)
            except Exception as e:
                print(f"Warning: Baseline baseline line generation skipped: {e}")

        if idx == (len(active_plots) - 1):
            legend_title = hue_column.replace('_', ' ').title()
            ax.legend(handles[:num_hues], leg_labels[:num_hues], title=legend_title, loc='lower right',
                      fontsize=cfg['legend_fontsize'], title_fontsize=cfg['legend_fontsize'], framealpha=0.90)
        else:
            try:
                ax.get_legend().remove()
            except Exception:
                pass

    if suptitle:
        fig.suptitle(suptitle, fontsize=cfg['suptitle_fontsize'], y=1.02)
    elif metadata_cols and (suptitle is None):
        group_title = " | ".join([c.replace('_', ' ').title() for c in metadata_cols])
        fig.suptitle(f"Condition: {group_title}", fontsize=cfg['label_fontsize'] + 2, y=1.02)

    if savepath: plt.savefig(savepath, bbox_inches='tight')
    plt.show()
