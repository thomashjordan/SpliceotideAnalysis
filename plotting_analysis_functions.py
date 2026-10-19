"""Cross-group analysis and figure helpers used by the summary notebooks.

* Discovering curated group pickles and building a long-form ``master_df`` from them.
* Picking the best cut/charge per group, collapsing replicates into modification ratios and
  converting them to apparent ddG values relative to a reference.
* MUSCLE-based sequence alignment and the grouped alignment + bar-chart figures.
* Generating and concatenating SI report PDFs across many groups.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import re
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from Bio.Align import substitution_matrices
import os
import tempfile
import subprocess
from datetime import datetime
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
# MUSCLE 5 (https://drive5.com/muscle5/) is used for the multiple sequence alignments drawn next to
# the bar charts. Point the MUSCLE_EXE environment variable at the executable, or put it on PATH
# under the name `muscle`.
MUSCLE_EXE_PATH = os.environ.get("MUSCLE_EXE") or shutil.which("muscle") or "muscle"


def natural_sort_key(series: pd.Series) -> pd.Series:
    """Vectorised natural-sort key, for use directly as pandas' `sort_values(key=...)`.

    Zero-pads every run of digits inside each value to a fixed width, so ordinary lexicographic
    ordering becomes NUMERIC where numbers appear -- e.g. mutant_1, mutant_2, ..., mutant_9,
    mutant_10, mutant_11 instead of the plain-string mutant_1, mutant_10, mutant_11, ..., mutant_2.
    Purely non-numeric text is unaffected, so it's safe to apply across a mix of sort columns.

    Usage: df.sort_values(["experiment_id", "precursor_plasmid"], key=natural_sort_key)
    """
    def _one(v):
        if pd.isna(v):
            return v
        return re.sub(r"\d+", lambda m: m.group().zfill(12), str(v))
    return series.map(_one)


def align_muscle(input_path: str, output_path: str) -> int:
    """Align the FASTA at `input_path` with MUSCLE 5 and write the alignment to `output_path`."""
    subprocess.check_output([MUSCLE_EXE_PATH, "-align", input_path, "-output", output_path])
    return 0


def get_muscle_msa(sequences: list, work_dir: str = ".") -> dict:
    """
    Takes a list of raw sequences, writes them to a standard FASTA file,
    runs MUSCLE 5, and returns a dict mapping {raw_sequence: aligned_sequence}.
    Uses static file names to avoid Windows tempfile permission errors.
    """
    # Deterministic order: plain set() iteration order depends on Python's per-process hash seed,
    # so the FASTA row order (and therefore the MUSCLE alignment, which is order-sensitive) changed
    # from run to run -- alignments were not reproducible. dict.fromkeys keeps first-seen order.
    unique_seqs = list(dict.fromkeys(s for s in sequences if pd.notna(s)))
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
    except FileNotFoundError:
        print(f"MUSCLE executable not found ({MUSCLE_EXE_PATH!r}); set the MUSCLE_EXE environment "
              f"variable. Falling back to unaligned sequences.")
        return {s: s for s in unique_seqs}
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
    the canonical per-group output produced by peak_reviewing_pipeline.py.

    Skips anything under a `_metadata_backup_*` subfolder. refresh_group_experiment_metadata()
    writes a timestamped backup COPY of each `*_manually_reviewed.pkl.gz` into a
    `_metadata_backup_<timestamp>/` folder next to the original before overwriting it in place, and
    those copies match this same glob. Since rglob() recurses, including them would surface every
    refreshed group MORE THAN ONCE -- once for the live file and once per backup -- and the backups
    hold the OLD, pre-refresh metadata, so master_df / SI reports would be built partly from stale
    duplicates. Filtering on the `_metadata_backup_` folder-name prefix keeps only the live files.
    """
    return sorted(
        p for p in Path(reports_root).rglob("*_manually_reviewed.pkl.gz")
        if not any(part.startswith("_metadata_backup_") for part in p.parts)
    )


def refresh_group_experiment_metadata(group_files: list, backup_root: str = None,
                                      keys: tuple = ("precursor_plasmid", "precursor_name"),
                                      verbose: bool = True) -> list:
    """Batch-patches experiment metadata (precursor_plasmid/precursor_name by default) on every
    ALREADY-curated group in `group_files`, WITHOUT re-picking anything -- for exactly the
    situation where you've fixed the input Excel, regenerated the per-sample MSExperiment pickles
    from it (e.g. by re-running main.py), and now just need every already-manually-reviewed
    ReplicateGroupResults to pick up the corrected values. See
    ReplicateGroupResults.refresh_experiment_metadata() for what actually gets changed (just the
    named attributes on each replicate's experiment object) and how the fresh values are located
    (the same output_folder_name each replicate's own stale experiment object already points to --
    so this assumes that path hasn't itself changed).

    Deliberately does NOT rename any group's folder/filename, even though precursor_plasmid is one
    of the values group_name (and therefore the on-disk path) was originally built from -- so a
    group whose precursor_plasmid changed will have correct in-object metadata but a folder/file
    name that still reads the OLD code. Rename those yourself afterward if you want them back in
    sync.

    Before overwriting each `*_manually_reviewed.pkl.gz` in place, this makes a timestamped backup
    copy of the ORIGINAL file first (one backup subfolder per run of this function, so re-running
    it later doesn't clobber an earlier backup) -- fully reversible if a group's metadata turns out
    wrong after the fact.

    backup_root: where backups are written -- defaults to `<file's own parent>/_metadata_backup_
    <timestamp>/<filename>` (i.e. right next to each original, so backups stay easy to find) if
    left as None. Pass your own root to collect every backup in one place instead.

    Returns the ordered list of group_files paths actually updated (i.e. successfully re-saved) --
    a group that errored out is skipped and reported, not included.
    """
    from analysis_containers import ReplicateGroupResults

    group_files = [Path(f) for f in group_files]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root_path = Path(backup_root) if backup_root is not None else None

    print(f"Refreshing experiment metadata ({', '.join(keys)}) for {len(group_files)} group file(s).")

    updated = []
    for file_path in group_files:
        try:
            # Backup FIRST, before anything about the file is touched.
            if backup_root_path is not None:
                backup_path = backup_root_path / file_path.name
            else:
                backup_path = file_path.parent / f"_metadata_backup_{run_timestamp}" / file_path.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_path)

            group_results = ReplicateGroupResults.load(file_path)
            print(f"[{group_results.group_name}]")
            group_results.refresh_experiment_metadata(keys=keys, verbose=verbose)
            group_results.save(file_path)
            updated.append(file_path)
        except Exception as e:
            print(f"  Warning: Could not refresh metadata for {file_path.name}. Error: {e}")

    print(f"Refreshed {len(updated)} out of {len(group_files)} group file(s) "
         f"(backups under '_metadata_backup_{run_timestamp}' unless backup_root was given).")
    return updated


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


def generate_si_reports(group_files: list, plot_mz: bool = True, plot_ms2: bool = False,
                        dpi: int = 400, rt_range: tuple = None, caption_template: str = None) -> list:
    """Generates one single-page, high-resolution A4 SI PDF per group -- best (fragment_cut,
    charge) selected via GroupedExperiments.select_best_cut_charge() (fewest stub replicates
    first, then area, then quality), plotting only manually curated peaks (stubs are noted in the
    table, not plotted).

    group_files: an explicit, ORDERED list of '*_manually_reviewed.pkl.gz' paths -- this is the
    exact order reports are generated in, and the exact order their page numbers and the final
    combined PDF (via combine_pdfs()) come out in. This function does no sorting or reordering of
    its own, so build the list in whatever order the SI should read: pass
    find_group_result_files(reports_root) directly for plain alphabetical order, or hand-assemble
    your own list (e.g. from an Excel column, or by sorting/filtering the discovered files
    yourself) to control page order explicitly.

    Each PDF is written into the SAME folder as the ReplicateGroupResults pickle it came from
    (not a separate output tree), and is tagged with a sequential page number matching its
    position in `group_files`. Returns the ordered list of PDF paths actually written -- pass that
    straight to combine_pdfs() to build one combined document in the same order.

    caption_template: a str.format() template for the figure caption printed at the bottom of
    every page -- see GroupedExperiments.generate_si_report()'s docstring (or
    analysis_containers.DEFAULT_SI_CAPTION_TEMPLATE) for the default wording and the available
    {placeholder} fields. Passed through unchanged to every group's generate_si_report() call, so
    set it once here to apply the same caption layout across the whole batch.
    """
    from analysis_containers import ReplicateGroupResults, GroupedExperiments

    group_files = [Path(f) for f in group_files]
    print(f"Generating SI reports for {len(group_files)} group file(s), in the order given.")

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
    from analysis_containers import to_long_path   # Windows >260-char MAX_PATH safety (deep trees)

    output_path = Path(output_path)
    os.makedirs(to_long_path(output_path.parent), exist_ok=True)

    writer = PdfWriter()
    for pdf_path in pdf_paths:
        # pypdf opens each source PDF via FileIO, which hits the 260-char limit on long report
        # paths -- feed it the extended-length form so deep-tree PDFs are found.
        writer.append(to_long_path(pdf_path))

    with open(to_long_path(output_path), "wb") as f:
        writer.write(f)

    print(f"Combined {len(pdf_paths)} PDF(s) into: {output_path}")
    return output_path


def generate_and_combine_si_reports(group_files: list, combined_output_path: str, plot_mz: bool = True,
                                    plot_ms2: bool = False, dpi: int = 400, rt_range: tuple = None,
                                    caption_template: str = None) -> Path | None:
    """Convenience wrapper: generate_si_reports() then combine_pdfs() in one call, preserving
    group_files' order end-to-end -- page numbers, the combined PDF's page order, and the
    returned PDF list all match the order `group_files` was given in. See generate_si_reports()'s
    docstring for how to control that order."""
    pdf_paths = generate_si_reports(group_files, plot_mz=plot_mz, plot_ms2=plot_ms2, dpi=dpi, rt_range=rt_range,
                                    caption_template=caption_template)
    return combine_pdfs(pdf_paths, combined_output_path)


def generate_simple_si_reports(group_files: list, appendix_name: str = "SI Appendix A",
                               style: str = "white", layout: str = "alpha", dpi: int = 400,
                               rt_range: tuple = None, state_color_order: list = None,
                               break_axis: bool = False, subpanel_labels: bool = False,
                               caption_margin_cm: float = 1.5, footer: bool = True) -> list:
    """Batch driver for GroupedExperiments.plot_si_report_simple() (the "simple" SI figure style,
    distinct from generate_si_reports()/generate_si_report() above) -- generates one PDF per group
    in `group_files`, IN THAT EXACT ORDER (see generate_si_reports()'s own docstring for why order
    isn't re-derived here -- same reasoning applies: pass find_group_result_files() output
    directly, or hand-assemble/reorder your own list).

    Each group gets its own figure number ("Figure 1", "Figure 2", ... matching its position in
    `group_files`), but the PAGE numbers that show up in each page's footer are ONE running
    counter across the whole batch -- correctly advanced by however many physical pages each
    individual group actually needed (plot_si_report_simple() can spill a busy group's figure
    and/or caption onto a second page within its own PDF, and returns (path, pages_used) for
    exactly this reason -- see its own docstring), so page numbers stay consistent and gap-free
    across the eventual combined document even when not every group's PDF is the same length.

    footer=True by default here (it's off by default on plot_si_report_simple() itself) since the
    entire point of a batch run like this is that the page numbers actually show up.

    Returns the ordered list of PDF paths actually written -- pass to combine_pdfs() to build one
    combined multi-group document; the page numbers already printed in each page's footer will
    read correctly straight through it, since they were computed against this same running counter
    (see generate_and_combine_simple_si_reports() for the one-call version of that).
    """
    from analysis_containers import ReplicateGroupResults, GroupedExperiments

    group_files = [Path(f) for f in group_files]
    print(f"Generating 'simple'-style SI reports for {len(group_files)} group file(s), in the order given.")

    written = []
    page_counter = 1
    for i, file_path in enumerate(group_files):
        try:
            group_results = ReplicateGroupResults.load(file_path)
            grouped_experiment = GroupedExperiments.from_replicate_group_results(group_results)
            pdf_path, pages_used = grouped_experiment.plot_si_report_simple(
                file_path.parent, appendix_name=appendix_name, figure_name=f"Figure {i + 1}",
                style=style, layout=layout, rt_range=rt_range, dpi=dpi, page_number=page_counter,
                source_path=file_path, state_color_order=state_color_order, break_axis=break_axis,
                subpanel_labels=subpanel_labels, caption_margin_cm=caption_margin_cm, footer=footer,
            )
            if pdf_path is not None:
                written.append(pdf_path)
                print(f"  Wrote: {pdf_path}  ({pages_used} page(s), starting at page {page_counter})")
                page_counter += pages_used
        except Exception as e:
            print(f"  Warning: Could not generate simple SI report for {file_path.name}. Error: {e}")

    print(f"Generated {len(written)} 'simple' SI PDF report(s) out of {len(group_files)} group(s), "
          f"{page_counter - 1} total page(s).")
    return written


def generate_and_combine_simple_si_reports(group_files: list, combined_output_path: str,
                                           appendix_name: str = "SI Appendix A", style: str = "white",
                                           layout: str = "alpha", dpi: int = 400, rt_range: tuple = None,
                                           state_color_order: list = None, break_axis: bool = False,
                                           subpanel_labels: bool = False, caption_margin_cm: float = 1.5,
                                           footer: bool = True) -> Path | None:
    """Convenience wrapper: generate_simple_si_reports() then combine_pdfs() in one call,
    preserving group_files' order end-to-end -- see generate_simple_si_reports()'s docstring for
    how page numbering stays consistent across groups that needed more than one physical page."""
    pdf_paths = generate_simple_si_reports(
        group_files, appendix_name=appendix_name, style=style, layout=layout, dpi=dpi,
        rt_range=rt_range, state_color_order=state_color_order, break_axis=break_axis,
        subpanel_labels=subpanel_labels, caption_margin_cm=caption_margin_cm, footer=footer,
    )
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
        # Full path (not just the name) so a caller can sort/filter this DataFrame into whatever
        # row order they want and then recover the exact ordered list of source pickles straight
        # off of it -- see get_ordered_source_files() below, which is exactly what
        # generate_si_reports()'s group_files parameter expects.
        df_long["_source_path"] = str(file_path)

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


def get_ordered_source_files(df: pd.DataFrame, sort_cols: list = None, source_col: str = "_source_path",
                             ascending: bool = True) -> list:
    """Derives an ordered, de-duplicated list of group source pickle paths from a master
    DataFrame (one row per replicate/fragment_cut/charge/modification, e.g. as built by
    build_master_dataframe_from_groups()) -- for feeding straight into generate_si_reports()'s
    (or generate_and_combine_si_reports()'s) `group_files` parameter, so the SI PDFs are generated,
    numbered, and combined in exactly this order.

    sort_cols: the column(s) that define your desired SI ordering, e.g.
    ["experiment_id", "precursor_plasmid", "maturase_name"]. `df` is sorted by these first, using a
    NATURAL sort (see natural_sort_key) so numbered names order 1, 2, ..., 9, 10, 11 rather than the
    plain-string 1, 10, 11, ..., 2 -- e.g. precursor_1 ... precursor_9, precursor_10, not
    precursor_1, precursor_10, precursor_11, ..., precursor_2. Pass None to skip sorting and just
    de-duplicate in `df`'s current row order (e.g. if you already sorted/filtered/reindexed it
    yourself beforehand).

    A group spans many rows in `df` (one per replicate x fragment_cut x charge x modification), so
    this keeps only the first occurrence of each unique `source_col` value in the (sorted) row
    order -- i.e. one entry per group, in the order its first row appears.
    """
    if source_col not in df.columns:
        raise KeyError(
            f"'{source_col}' not found in DataFrame -- build it with build_master_dataframe_from_groups() "
            "(which tags every row with the full source pickle path), or pass the matching column name."
        )
    ordered_df = (df.sort_values(list(sort_cols), ascending=ascending, key=natural_sort_key)
                  if sort_cols else df)

    seen = set()
    ordered_files = []
    for path in ordered_df[source_col]:
        if path not in seen:
            seen.add(path)
            ordered_files.append(Path(path))
    return ordered_files


def extract_best_cut_charge(master_df: pd.DataFrame, group_col: str = "group_name") -> pd.DataFrame:
    """
    For every experimental group (one ReplicateGroupResults -- many replicates x fragment_cuts x
    charges x modifications), picks the single (cut_id, charge) combination detected in the most
    replicates, breaking ties by highest average area and then highest average quality_score --
    all averaged across every replicate and modification state that shares that combination. In
    other words the ranking priority is (number of non-stub detections, avg area, avg quality),
    matching the sort below; detection breadth is deliberately favoured over per-peak quality so
    the chosen cut/charge is the one most robustly seen across the cohort. Stub/undetected rows are
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

    # Guard against reporting a non-detection as confident full conversion. When the unmodified
    # peptide's own row is a stub (never detected/curated), its area is 0, so 1 - 0/overall = 1.0
    # -- indistinguishable from a genuinely fully-modified sample. That 1.0 is not a measurement:
    # we can't tell "no unmodified peptide present" from "unmodified peptide simply not picked".
    # Mark it NaN so it drops out of averages/ddG rather than silently biasing them toward 1.0.
    # `unmod_detected` is kept on the frame so a caller can choose to treat these as 1.0 explicitly.
    if "is_stub" in unmod_df.columns:
        unmod_df["unmod_detected"] = ~unmod_df["is_stub"].astype(bool)
        n_undetected = int((~unmod_df["unmod_detected"]).sum())
        if n_undetected:
            print(f"[collapse_modification_replicates] {n_undetected} of {len(unmod_df)} replicate(s) "
                  f"have an undetected (stub) unmodified peptide -> mod_ratio set to NaN "
                  f"(cannot distinguish full conversion from a missed peak).")
        unmod_df.loc[~unmod_df["unmod_detected"], ["total_modification_ratio", "mod_ratio"]] = np.nan
    else:
        unmod_df["unmod_detected"] = True

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
        plot_baseline: bool = False,
        plot_midline: bool = False
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
    if plot_midline:
        ax.axvline(x=0.5, color='red', linestyle='--', linewidth=1.2, zorder=3)
    if suptitle:
        fig.suptitle(suptitle, fontsize=cfg['suptitle_fontsize'], y=1.02)
    elif metadata_cols and (suptitle is None):
        group_title = " | ".join([c.replace('_', ' ').title() for c in metadata_cols])
        fig.suptitle(f"Condition: {group_title}", fontsize=cfg['label_fontsize'] + 2, y=1.02)

    if savepath: plt.savefig(savepath, bbox_inches='tight')
    plt.show()


# ==================================================================================================
# --- Standalone beta-style SI panels ---
# Reusable versions of the panels drawn inside GroupedExperiments.plot_si_report_simple(layout=
# 'beta'), decomposed so they can be produced (and saved) individually and driven by ONE chosen
# replicate at one (fragment_cut, charge) rather than the report's auto-selected representative.
# They call the existing GroupedExperiments panel helpers directly, so each panel is visually
# identical to the corresponding beta-layout panel.
# ==================================================================================================

def _as_grouped_experiments(grouped):
    """Accept either a GroupedExperiments (used as-is) or a ReplicateGroupResults (wrapped via
    GroupedExperiments.from_replicate_group_results), so callers can pass the loaded group directly
    without constructing a GroupedExperiments themselves."""
    if hasattr(grouped, "_get_theoretical_targets_dict"):
        return grouped
    from analysis_containers import GroupedExperiments
    return GroupedExperiments.from_replicate_group_results(grouped)


def _si_state_maps_for_replicate(grouped, replicate_id, cut_id, charge, state_color_order=None):
    """Builds the per-state plotting inputs for ONE replicate at one (fragment_cut, charge), reusing
    GroupedExperiments' own theoretical-target lookup, canonical state ordering and color mapping so
    the panels match plot_si_report_simple(layout='beta').

    Returns (target_dict, ordered_states, state_labels, color_map, rep_peaks, isoform_peaks_by_state):
      * target_dict           -- {state_key: TargetPeptide} for EVERY theoretical state at this cut
                                 (drives the m/z-change shaded regions / EIC "not detected" entries),
      * ordered_states        -- state_keys in the canonical (color) order,
      * state_labels          -- {state_key: nice label},
      * color_map             -- {nice label: RGB},
      * rep_peaks             -- {state_key: PeptidePeak} this replicate actually selected (non-Reject),
      * isoform_peaks_by_state-- {state_key: [PeptidePeak]} (single-replicate here) for the MS2 panels.
    """
    from collections import defaultdict
    from analysis_containers import format_modification_label, get_state_color_map

    grouped = _as_grouped_experiments(grouped)
    charge = int(charge)
    target_dict = grouped._get_theoretical_targets_dict(cut_id)
    all_states = sorted(target_dict.keys())
    state_labels = {sk: format_modification_label(target_dict[sk].modifications) for sk in all_states}
    ordered_states = grouped._order_states_canonically(all_states, state_labels, state_color_order)
    color_map = get_state_color_map([state_labels[s] for s in ordered_states], state_color_order)

    rep_peaks = {}
    isoform_peaks_by_state = defaultdict(list)
    for sk in ordered_states:
        tobj = target_dict[sk]
        peak = grouped._find_selected_peak(replicate_id, cut_id, str(tobj.modifications), charge,
                                           sequence=tobj.sequence)
        if peak is not None and getattr(peak, "tier", None) != "Reject":
            rep_peaks[sk] = peak
            isoform_peaks_by_state[sk].append(peak)
    return target_dict, ordered_states, state_labels, color_map, rep_peaks, isoform_peaks_by_state


def plot_si_overview_panels(grouped, replicate_id, cut_id, charge, rt_range=(3, 15),
                            break_axis=True, state_color_order=None, figsize=(9.0, 3.2)):
    """FUNCTION 1 -- one figure, two columns (the top row of plot_si_report_simple layout='beta'):
    the zoomed extracted-ion chromatogram overlay (left) and the m/z-change panel (right), the
    m/z-change panel given MORE width via a 5-column gridspec (EIC = 2 cols, m/z-change = 3), exactly
    as in beta. Driven by one replicate's curated peaks at (cut_id, charge). Returns the Figure."""
    grouped = _as_grouped_experiments(grouped)
    (target_dict, ordered_states, state_labels, color_map,
     rep_peaks, _) = _si_state_maps_for_replicate(grouped, replicate_id, cut_id, charge, state_color_order)

    charge = int(charge)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 6, left=0.08, right=0.97, top=0.90, bottom=0.20, wspace=0.55)
    ax_eic = fig.add_subplot(gs[0, 0:2])                         # narrower: zoomed EIC overlay
    grouped._plot_eic_representative(ax_eic, ordered_states, state_labels, rep_peaks, color_map,
                                     charge, rt_range=rt_range)
    grouped._plot_mz_change_plot_representative(fig, gs[0, 2:6], ordered_states, state_labels,   # wider
                                                rep_peaks, target_dict, charge, color_map,
                                                break_axis=break_axis)
    return fig


def plot_si_ms2_panels(grouped, isoform_peaks, state_label, residues_per_line=None,
                       fig_width=9.0, base_height=3.2):
    """FUNCTION 2 -- one figure for a SINGLE modification state, two equal columns (a beta MS2 row):
    the head-to-tail MS2 mirror plot (left) and the annotated sequence-ladder plot (right). The
    figure grows taller when the sequence has to wrap onto multiple lines, mirroring how the SI
    report grows that state's row. Returns the Figure."""
    import pyteomics
    from analysis_containers import _seq_residues_per_line

    grouped = _as_grouped_experiments(grouped)
    if not isoform_peaks:
        raise ValueError("plot_si_ms2_panels: isoform_peaks is empty (no curated peak for this state).")

    passed = [p for p in isoform_peaks if p.quality_metrics.get('overall_pass', False)]
    rep_peak = passed[0] if passed else isoform_peaks[0]

    # Grow the figure height by the number of wrapped sequence lines (sized the SAME way the report
    # does), so a long peptide's ladder stays legible in a standalone figure instead of cramming.
    seq_panel_w = fig_width * 0.5
    per_line = int(residues_per_line) if residues_per_line else _seq_residues_per_line(seq_panel_w)
    try:
        seq_len = len(pyteomics.parser.parse(rep_peak.target.sequence))
    except Exception:
        seq_len = 0
    n_lines = max(1, int(np.ceil(seq_len / per_line))) if seq_len else 1

    fig = plt.figure(figsize=(fig_width, base_height * n_lines))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.97, top=0.88, bottom=0.14, wspace=0.35)
    ax_mirror = fig.add_subplot(gs[0, 0])
    grouped._plot_ms2_mirror(ax_mirror, isoform_peaks)
    ax_seq = fig.add_subplot(gs[0, 1])
    b_ions, y_ions = grouped._compute_ms2_matches(isoform_peaks)
    grouped._plot_si_sequence_annotations(ax_seq, rep_peak, b_ions, y_ions, state_label,
                                          residues_per_line=per_line)
    return fig


def plot_replicate_si_panels(grouped, replicate_id, cut_id, charge, output_dir="summary_data_files",
                             name_prefix=None, rt_range=(3, 15), break_axis=True,
                             state_color_order=None, dpi=300, ext="png"):
    """DRIVER -- for ONE replicate at (cut_id, charge), build and SAVE beta-style panels into
    `output_dir` (default summary_data_files/):
        1. one overview figure  (zoomed EIC + m/z-change)          via plot_si_overview_panels(),
        2. one mirror+sequence figure PER detected modification state via plot_si_ms2_panels().

    replicate_id/cut_id/charge are exactly the values in a collect_peaks() row -- e.g. from a
    filtered `to_plot` frame:
        rep = to_plot['replicate'].iloc[0]; cut = to_plot['fragment_cut'].iloc[0]
        z   = int(to_plot['charge'].iloc[0])
        plot_replicate_si_panels(grouped_experiment, rep, cut, z)

    Returns the list of saved Paths.
    """
    from analysis_containers import to_long_path

    grouped = _as_grouped_experiments(grouped)
    charge = int(charge)
    output_dir = Path(output_dir)
    os.makedirs(to_long_path(output_dir), exist_ok=True)
    if name_prefix is None:
        name_prefix = getattr(getattr(grouped, "group_results", None), "group_name", "group")

    (target_dict, ordered_states, state_labels, color_map,
     rep_peaks, isoform_peaks_by_state) = _si_state_maps_for_replicate(
        grouped, replicate_id, cut_id, charge, state_color_order)

    safe_cut = re.sub(r"[^0-9A-Za-z]+", "-", str(cut_id)).strip("-")
    rep_tag = re.sub(r"[^0-9A-Za-z]+", "_", str(replicate_id).split('_Rep_')[-1])
    saved = []

    # (1) overview figure
    fig = plot_si_overview_panels(grouped, replicate_id, cut_id, charge, rt_range=rt_range,
                                  break_axis=break_axis, state_color_order=state_color_order)
    op = output_dir / f"{name_prefix}_Rep{rep_tag}_overview_Cut_{safe_cut}_z{charge}.{ext}"
    fig.savefig(to_long_path(op), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    saved.append(op)

    # (2) one MS2 (mirror + sequence) figure per detected modification state
    for sk in [s for s in ordered_states if isoform_peaks_by_state.get(s)]:
        label = state_labels[sk]
        fig = plot_si_ms2_panels(grouped, isoform_peaks_by_state[sk], label)
        safe_label = re.sub(r"[^0-9A-Za-z]+", "_", label).strip("_")
        op = output_dir / f"{name_prefix}_Rep{rep_tag}_ms2_{safe_label}_Cut_{safe_cut}_z{charge}.{ext}"
        fig.savefig(to_long_path(op), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(op)

    print(f"Saved {len(saved)} figure(s) to {output_dir}:")
    for pth in saved:
        print("   ", pth.name)
    return saved


def plot_si_ms2_multipanel(grouped, replicate_id, cut_id, charge, output_dir="summary_data_files",
                           name_prefix=None, state_color_order=None, dpi=300, ext="png",
                           fig_width=9.0, row_unit_height=2.3, save=True, suptitle=None,
                           include_undetected=False):
    """Combine the per-state MS2 panels for ONE replicate at (cut_id, charge) into a SINGLE
    multipanel figure: one row per DETECTED modification state, each row holding the MS2 mirror plot
    (left) and the annotated sequence-ladder plot (right), and titled with a bold panel letter + the
    state's nice label in that state's own color. Rows whose sequence has to wrap are made
    proportionally taller (as in the SI report). This is the same content plot_replicate_si_panels()
    saves as separate MS2 files, gathered into one labeled figure instead.

    Saves one image into `output_dir` (default summary_data_files/) when save=True. Returns
    (fig, saved_path) -- saved_path is None when save=False.

    include_undetected=True adds a row for EVERY theoretical modification state at this cut (not just
    the ones this replicate detected); undetected states are drawn as "No MS2 detected" placeholders.
    Use it when you want a fixed set of rows (e.g. all four states of the group) regardless of which
    were curated for this particular replicate/charge.
    """
    import types
    import pyteomics
    from analysis_containers import _seq_residues_per_line, to_long_path

    grouped = _as_grouped_experiments(grouped)
    charge = int(charge)
    (target_dict, ordered_states, state_labels, color_map,
     rep_peaks, isoform_peaks_by_state) = _si_state_maps_for_replicate(
        grouped, replicate_id, cut_id, charge, state_color_order)

    detected = [s for s in ordered_states if isoform_peaks_by_state.get(s)]
    states = ordered_states if include_undetected else detected
    if not states:
        raise ValueError("plot_si_ms2_multipanel: no modification states to plot for this "
                         f"replicate / cut {cut_id} / charge {charge}.")

    # residues-per-line for the sequence panels (right column ~ half the figure width); used both to
    # size each row's height (wrapped sequences need more rows) and passed to the drawing so the two
    # agree -- identical to plot_si_ms2_panels().
    per_line = _seq_residues_per_line(fig_width * 0.5)

    def _rep_peak(sk):
        """The peak whose sequence/matches drive this state's row. For an undetected state (no
        curated peak) a lightweight stand-in carrying just the theoretical target is used, so the
        sequence panel can still draw the bare peptide + a 'No MS2 detected' note."""
        peaks = isoform_peaks_by_state.get(sk)
        if not peaks:
            return types.SimpleNamespace(target=target_dict[sk])
        passed = [p for p in peaks if p.quality_metrics.get('overall_pass', False)]
        return passed[0] if passed else peaks[0]

    def _wrap_lines(sk):
        try:
            n = len(pyteomics.parser.parse(_rep_peak(sk).target.sequence))
        except Exception:
            n = 0
        return max(1, int(np.ceil(n / per_line))) if n else 1

    # Grid: for each state a thin title row (spanning both columns) then its panel row, so titles
    # never collide with the (possibly tall, wrapped) panels.
    TITLE_H = 0.28
    row_lines = {sk: _wrap_lines(sk) for sk in states}
    height_ratios = []
    for sk in states:
        height_ratios += [TITLE_H, row_lines[sk]]
    fig_height = sum(height_ratios) * row_unit_height

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(2 * len(states), 2, height_ratios=height_ratios,
                          left=0.07, right=0.97, top=0.955, bottom=0.03, hspace=0.45, wspace=0.32)

    for i, sk in enumerate(states):
        label = state_labels[sk]
        letter = chr(ord('A') + i) if i < 26 else f"#{i + 1}"
        peaks = isoform_peaks_by_state.get(sk, [])
        # --- row title (spans both columns), colored to match the state everywhere else ---
        ax_title = fig.add_subplot(gs[2 * i, :])
        ax_title.axis('off')
        title = f"{letter}) {label}" + ("" if peaks else "  (not detected)")
        ax_title.text(0.0, 0.15, title, ha='left', va='center',
                      fontsize=12, weight='bold', color=color_map[label])
        # --- mirror (left) + sequence (right) ---
        ax_mirror = fig.add_subplot(gs[2 * i + 1, 0])
        grouped._plot_ms2_mirror(ax_mirror, peaks)              # empty list -> "No MS2 detected"
        ax_seq = fig.add_subplot(gs[2 * i + 1, 1])
        b_ions, y_ions = grouped._compute_ms2_matches(peaks) if peaks else (None, None)
        grouped._plot_si_sequence_annotations(ax_seq, _rep_peak(sk), b_ions, y_ions, label,
                                              residues_per_line=per_line)

    if suptitle is None:
        group_name = getattr(getattr(grouped, "group_results", None), "group_name", "")
        rep_tag = str(replicate_id).split('_Rep_')[-1]
        suptitle = f"{group_name}  |  Rep {rep_tag}  |  cut {cut_id}, z={charge}  |  MS2 fragmentation"
    fig.suptitle(suptitle, fontsize=10, y=0.985)

    saved_path = None
    if save:
        import re as _re
        output_dir = Path(output_dir)
        os.makedirs(to_long_path(output_dir), exist_ok=True)
        if name_prefix is None:
            name_prefix = getattr(getattr(grouped, "group_results", None), "group_name", "group")
        safe_cut = _re.sub(r"[^0-9A-Za-z]+", "-", str(cut_id)).strip("-")
        rep_tag = _re.sub(r"[^0-9A-Za-z]+", "_", str(replicate_id).split('_Rep_')[-1])
        saved_path = output_dir / f"{name_prefix}_Rep{rep_tag}_ms2_multipanel_Cut_{safe_cut}_z{charge}.{ext}"
        fig.savefig(to_long_path(saved_path), dpi=dpi, bbox_inches="tight")
        print(f"Saved MS2 multipanel ({len(states)} state row(s), {len(detected)} detected) to: {saved_path}")

    return fig, saved_path
