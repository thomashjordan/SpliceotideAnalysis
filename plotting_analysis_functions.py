import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import pairwise2
from Bio.Align import substitution_matrices

# --- Data Processing Functions ---

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

def calculate_charges_mod_ratios(master_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates modification ratios from the master DataFrame, grouping by experiment
    but summing across charge states.
    """
    df = master_df.copy()
    df['effective_area'] = df.apply(
        lambda row: 0 if row['tier'] in ['Reject', 'Bronze'] else row['area'],
        axis=1
    ).fillna(0)

    grouped = df.groupby(['experiment_id', 'sample_id'])
    results = []
    for name, group in grouped:
        total_effective_area = group['effective_area'].sum()
        modified_group = group[group['modification_state'] != "{'unmodified': None}"]
        modified_effective_area = modified_group['effective_area'].sum()
        mod_ratio = (modified_effective_area / total_effective_area) if total_effective_area > 0 else 0
        
        first_row = group.iloc[0]

        results.append({
            'experiment_id': name[0],
            'sample_id': name[1],
            'peptide_sequence': first_row['peptide_sequence'],
            'cut_id': first_row['cut_id'],
            'enzyme': first_row['enzyme'],
            'precursor_name': first_row['precursor_name'],
            'mod_ratio': mod_ratio
        })
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

# --- Plotting and Helper Functions (Adapted from enzymeanalysis/enzymeplots) ---

# Define color schemes
COLOR_SCHEMES = {
    'hydrophobicity': {
        'A': '#808080', 'I': '#000000', 'L': '#000000', 'M': '#000000', 'F': '#000000',
        'W': '#000000', 'Y': '#000000', 'V': '#000000', 'C': '#808080', 'P': '#808080',
        'G': '#808080', 'T': '#008000', 'S': '#008000', 'Q': '#008000', 'N': '#008000',
        'E': '#FF0000', 'D': '#FF0000', 'H': '#0000FF', 'K': '#0000FF', 'R': '#0000FF'
    },
    'clustal': {
        'G': '#F0F0F0', 'P': '#F0F0F0', 'S': '#F0F0F0', 'T': '#F0F0F0', 'A': '#0000FF',
        'V': '#0000FF', 'L': '#0000FF', 'I': '#0000FF', 'M': '#0000FF', 'C': '#0000FF',
        'F': '#FF0000', 'Y': '#FF0000', 'W': '#FF0000', 'H': '#FF00FF', 'K': '#00FF00',
        'R': '#00FF00', 'Q': '#800080', 'N': '#800080', 'E': '#FF0000', 'D': '#FF0000'
    }
}

def format_axis(ax):
    ax.xaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    ax.yaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    ax.spines['top'].set_linewidth(0.6)
    ax.spines['right'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.6)

matrix = substitution_matrices.load("BLOSUM62")
def seq_alignment(wt_sequence, sdf, score='ddg', penalties=(-15, -2)):
    seq_alignments, labels, labels_sparse, labels_aligned = [], [wt_sequence], [wt_sequence], [wt_sequence]
    for _, row in sdf.iterrows():
        seq, mod_efficiency = row['sequence'], row[score]
        alignments = pairwise2.align.globalds(wt_sequence, seq.split("*")[0], matrix, penalties[0], penalties[1])
        if not alignments: continue
        alignment = alignments[0]
        
        if alignment[1] != wt_sequence:
            seq_alignments.append((seq, alignment[1], mod_efficiency))
            labels_sparse.append("".join([i if i != w else " " for i, w in zip(alignment[1], wt_sequence)]))
            labels.append(seq)
            labels_aligned.append(alignment[1])
    return seq_alignments, labels_sparse, labels, labels_aligned

def matrix_subplot(labels, mat_ax, color_scheme=None):
    max_columns = max([len(x) for x in labels]) if labels else 0
    max_rows = len(labels)
    
    mat_ax.hlines(y=np.arange(-0.5, max_rows), xmin=-0.5, xmax=max_columns-0.5, color='0.8', linewidth=0.6)
    mat_ax.vlines(x=np.arange(-0.5, max_columns), ymin=-0.5, ymax=max_rows-0.5, color='0.8', linewidth=0.6)

    for row_index, label in enumerate(labels):
        for col_index, char in enumerate(label):
            if char == " ": continue
            
            color = 'black'
            facecolor = 'none'
            
            if color_scheme and char in color_scheme:
                facecolor = color_scheme[char]
                color = 'white' if facecolor != '#F0F0F0' else 'black' # Make text visible on dark backgrounds

            if facecolor != 'none':
                 square = plt.Rectangle(xy=(col_index-0.5, row_index-0.5), width=1, height=1, facecolor=facecolor, edgecolor='0.8', linewidth=0.6)
                 mat_ax.add_patch(square)

            mat_ax.text(col_index, row_index, char, color=color, ha="center", va="center", fontsize=6, weight='bold')

    mat_ax.set_xticks([])
    mat_ax.set_yticks([])
    mat_ax.set_xlim(-0.5, max_columns - 0.5)
    mat_ax.set_ylim(-0.5, max_rows - 0.5)
    mat_ax.invert_yaxis() # Puts the reference sequence at the top

def create_summary_alignment_plot(processed_df: pd.DataFrame, ref_sequence: str, color_scheme_name: str = None, sort_decreasing = True):
    """
    Creates a three-panel summary plot with sequence alignment, modification ratio, and ddG.
    """
    plot_df = processed_df.copy()
    
    ref_ratio = plot_df[plot_df['peptide_sequence'] == ref_sequence]['mod_ratio'].mean()
    plot_df['ddg'] = calculate_ddg_values(plot_df['mod_ratio'], ref_ratio)

    agg_df = plot_df.groupby('peptide_sequence').agg(
        mod_ratio_mean=('mod_ratio', 'median'), mod_ratio_std=('mod_ratio', 'std'),
        ddg_mean=('ddg', 'median'), ddg_std=('ddg', 'std'),
    ).reset_index().sort_values("mod_ratio_mean", ascending=not(sort_decreasing)).rename(columns={'peptide_sequence': 'sequence'})

    _, labels, label_order, _ = seq_alignment(ref_sequence, agg_df, score='ddg_mean')
    agg_df_ordered = agg_df.set_index('sequence').reindex(label_order).reset_index()

    # --- DYNAMIC PLOT SIZING ---
    longest_seq = len(ref_sequence)
    num_variants = len(label_order)
    barchart_width = 0.8
    bar_size = 0.06
    spacer = 0.15
    matrix_width = (longest_seq + 1) * bar_size
    matrix_height = num_variants * bar_size * 1.5
    total_width = matrix_width + (2 * barchart_width) + (4 * spacer)
    total_height = matrix_height + 2 * spacer
    
    fig = plt.figure(figsize=(total_width, total_height), dpi=300)

    # --- Create Axes using plt.axes with normalized coordinates ---
    ax_mat = plt.axes([(spacer)/total_width, 
                           (spacer)/total_height, 
                           matrix_width/total_width, 
                           matrix_height/total_height], frameon=False)
    
    ax_ratio = plt.axes([(matrix_width + 2*spacer)/total_width, 
                             (spacer)/total_height, 
                             barchart_width/total_width, 
                             matrix_height/total_height])

    ax_ddg = plt.axes([(matrix_width + barchart_width + 3*spacer)/total_width,
                           (spacer)/total_height,
                           barchart_width/total_width,
                           matrix_height/total_height])

    # --- Plot Matrix ---
    color_map = COLOR_SCHEMES.get(color_scheme_name)
    matrix_subplot(labels, ax_mat, color_scheme=color_map)

    # --- Plot Modification Ratio ---
    sns.barplot(data=agg_df_ordered, y='sequence', x='mod_ratio_mean', color='0.7', ax=ax_ratio, orient='h', xerr=agg_df_ordered['mod_ratio_std'])
    sns.stripplot(data=plot_df, y='peptide_sequence', x='mod_ratio', order=label_order, color='0.0', ax=ax_ratio, orient='h', jitter=0.2, size=3)
    ax_ratio.set_xlim(0, 1)
    ax_ratio.set_xlabel("Mod. Ratio")
    ax_ratio.set_ylabel("")
    ax_ratio.set_yticks([])
    ax_ratio.set_yticklabels([])

    # --- Plot ddG ---
    sns.barplot(data=agg_df_ordered, y='sequence', x='ddg_mean', color='0.7', ax=ax_ddg, orient='h', xerr=agg_df_ordered['ddg_std'])
    sns.stripplot(data=plot_df, y='peptide_sequence', x='ddg', order=label_order, color='0.0', ax=ax_ddg, orient='h', jitter=0.2, size=3)
    ax_ddg.set_xlabel("ddG [cal/mol]")
    ax_ddg.set_ylabel("")
    ax_ddg.set_yticks([])
    ax_ddg.set_yticklabels([])

    fig.tight_layout()
    plt.show()
