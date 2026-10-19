#!/usr/bin/env python
"""Step 1 of the workflow: automated peak picking for every experimental group.

Reads a sample sheet (one row per LC-MS run, see README.md for the column layout), groups the
rows into biological-replicate cohorts, and for each cohort:

  1. loads/parses each replicate's mzML into a cached MassSpectrum,
  2. expands the precursor sequence into digest fragments (unmodified + modified states),
  3. extracts EICs for every fragment/charge, clusters candidate peaks across replicates with
     DBSCAN, scores them (MS1 isotope pattern, ppm error, MS2 fragment coverage), and
  4. writes a `{group}_cohort.pkl.gz` ReplicateGroupResults archive under --output_dir.

The cohorts are then curated interactively with peak_reviewing_pipeline.py.

Example:
    python peak_picking_pipeline.py -i input_excel_files/96_well_plate_expressions.xlsx -o reports/plates
"""
import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib

# Figures are only ever written to disk, never shown, so use the headless backend.
matplotlib.use('Agg')

# Local project imports
from pipeline_functions import run_new_analysis_for_group


def load_and_group_samples(excel_path: Path) -> pd.core.groupby.DataFrameGroupBy:
    """Loads the sample sheet from an Excel file and groups it for processing."""
    if not excel_path.exists():
        print(f"ERROR: Input Excel file not found at: {excel_path}")
        raise FileNotFoundError(f"Input Excel file not found at: {excel_path}")

    print(f"Loading samples from: {excel_path}")
    df_samples = pd.read_excel(excel_path)

    required_grouping_cols = [
        'experiment_id',
        'precursor_plasmid',
        'maturase_name',
        'maturase_plasmid',
        'expression_temperature',
        'digestion'
    ]

    for col in required_grouping_cols:
        if col not in df_samples.columns:
            raise ValueError(f"Missing required grouping column in Excel file: '{col}'")

    # Grouping across all 6 metadata attributes ensures group_info is a 6-value tuple
    grouped = df_samples.groupby(required_grouping_cols)
    return grouped


def main():
    """Main function to parse CLI arguments and run the LC-MS/MS peak picking pipeline."""
    parser = argparse.ArgumentParser(
        description="Run grouped LC-MS/MS analysis pipeline from pre-processed experiment files."
    )
    parser.add_argument(
        "-i", "--input_excel",
        required=True,
        type=Path,
        help="Path to the input Excel file describing the samples and experiments."
    )
    parser.add_argument(
        "-o", "--output_dir",
        required=True,
        type=Path,
        help="Path to the main output directory where all final group analysis files will be saved."
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="If set, forces re-parsing of raw .mzML files, ignoring any cached data."
    )
    parser.add_argument(
        "--force_reanalyze",
        action="store_true",
        help="If set, forces re-running of peak picking and scoring, ignoring cached analysis results."
    )
    parser.add_argument(
        "--manual_review",
        action="store_true",
        help="Deprecated and ignored: manual review is now done with peak_reviewing_pipeline.py."
    )
    args = parser.parse_args()

    try:
        # Ensure root output directory exists
        args.output_dir.mkdir(parents=True, exist_ok=True)

        grouped_samples = load_and_group_samples(args.input_excel)
        total_groups = len(grouped_samples)

        print(f"Found {total_groups} distinct experimental groups to process.")

        for i, (group_info, group_df) in enumerate(grouped_samples):
            # Format readable group display string
            group_label = " | ".join([str(val) for val in group_info])
            print(f"\n==================================================")
            print(f"Processing Group {i + 1}/{total_groups}: {group_label}")
            print(f"Replicates in group: {len(group_df)}")
            print(f"==================================================")

            # Call core driver with aligned keyword arguments
            cohort_results = run_new_analysis_for_group(
                group_info=group_info,
                group_df=group_df,
                report_output_dir=args.output_dir,  # Aligned parameter name
                force_reload=args.force_reload,
                manual_review=args.manual_review,
                force_reanalyze=args.force_reanalyze
            )

            # Terminal diagnostic printout upon group completion
            if cohort_results is not None:
                _, df_summary = cohort_results.generate_modification_state_matrix()
                if not df_summary.empty:
                    print("\n--- Group Summary Overview ---")
                    cols_to_show = [c for c in ['fragment_cut', 'charge', 'modification',
                                              'avg_retention_time', 'avg_relative_area',
                                              'cohort_detection_frequency'] if c in df_summary.columns]
                    print(df_summary[cols_to_show].to_string(index=False))

        print("\n--- Pipeline finished successfully! ---")

    except Exception as e:
        print(f"\nA critical error stopped the pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
