import sys
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg') # Or 'TkAgg' if you don't have PyQt installed
import matplotlib.pyplot as plt

# Local project imports
# This assumes the script is run from the project's root directory.
from pipeline_functions import run_analysis_for_group

def load_and_group_samples(excel_path: Path) -> pd.core.groupby.DataFrameGroupBy:
    """Loads the sample sheet from an Excel file and groups it for processing."""
    if not excel_path.exists():
        raise FileNotFoundError(f"Input Excel file not found at: {excel_path}")

    df_samples = pd.read_excel(excel_path)

    # Group by the unique experimental conditions to process replicates together
    required_grouping_cols = ['experiment_id', 'precursor_plasmid', 'maturase_name']
    for col in required_grouping_cols:
        if col not in df_samples.columns:
            raise ValueError(f"Missing required grouping column in Excel file: '{col}'")

    grouped = df_samples.groupby(required_grouping_cols)
    return grouped


def main():
    """Main function to parse arguments and run the plate analysis pipeline."""
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
        help="Path to the main output directory where all results, reports, and logs will be saved."
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="If set, forces re-parsing of raw .mzML files, ignoring any cached data."
    )
    parser.add_argument(
        "--force_reanalyze",
        action="store_true",
        help="If set, forces re-running of the peak picking and scoring, ignoring cached analysis results."
    )
    parser.add_argument(
        "--manual_review",
        action="store_true",
        help="If set, activates interactive manual review mode for each replicate, showing plots for user validation."
    )
    args = parser.parse_args()

    try:
        grouped_samples = load_and_group_samples(args.input_excel)

        total_groups = len(grouped_samples)
        for i, (group_info, group_df) in enumerate(grouped_samples):
            print(f"--- Processing Group {i + 1}/{total_groups} ---")
            # The core logic is now in a separate function for clarity
            run_analysis_for_group(
                group_info=group_info,
                group_df=group_df,
                main_output_dir=args.output_dir,
                force_reload=args.force_reload,
                manual_review=args.manual_review,
                force_reanalyze=args.force_reanalyze
            )

        print("--- Pipeline finished successfully! ---")

    except Exception as e:
        print(f"A critical error stopped the pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
