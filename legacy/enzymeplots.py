import enzymeanalysis as da
#import lcmsanalysis as la
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd

matplotlib.rcParams['pdf.fonttype'] = 42
big_font = {'family' : 'sans-serif', 'size'   : 6}
small_font = {'family' : 'sans-serif', 'size' : 4}
matplotlib.rc('font', **small_font)

def ddgi_heatmap(wt_sequence, scores, core_length=0, norm=[0, 2.0]):

    fig = plt.figure(figsize=(0.1*(len(wt_sequence)-core_length), 0.1), dpi=600)

    black_blue = matplotlib.colors.LinearSegmentedColormap.from_list('black_blue', [(0,0,0),(0.5,0.9,1)], N=100)
    cNorm  = matplotlib.colors.Normalize(vmin=norm[0], vmax=norm[1])
    
    ax = plt.axes([0,0,1,1])

    for position, score in enumerate(scores):
        if np.isnan(score):
            square = plt.Rectangle(xy=(position, 0), width=1, height=1, color='0.6', ec=None)
        else:
            c = black_blue(int(cNorm(score)*100) - 1)
            square = plt.Rectangle(xy=(position, 0), width=1, height=1, color=c, ec=None)
        ax.add_patch(square)
            
    ax.set_xlim(0,len(wt_sequence)-core_length)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.array(range(len(wt_sequence)-core_length)) + 0.5)
    ax.set_xticklabels(wt_sequence, fontdict=big_font, ha="center")
    ax.tick_params(axis='x', pad=-3, bottom=False)
    ax.set_yticks([])
    ax.set_frame_on(False)
    return

def format_axis(ax):
    ax.xaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    ax.yaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    ax.spines['top'].set_linewidth(0.6)
    ax.spines['right'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.spines['left'].set_linewidth(0.6)
    
def fm_subplot(extract_df, fraction_ax, label_order=None, y='sequence', x='mod_fraction'):
    sns.barplot(data=extract_df, y=y, x=x, ax=fraction_ax,
                color='0.7', linewidth=0.6, edgecolor="0.0", ci=None, order=label_order)
    sns.stripplot(data=extract_df, y=y, x=x, ax=fraction_ax,
                  order=label_order, color='0.0', size=1.2, jitter=0.2)
    fraction_ax.set_xlabel("Fraction Modified", fontdict=big_font)
    fraction_ax.set_yticks([])
    fraction_ax.set_yticklabels([])
    fraction_ax.set_ylabel("")
    fraction_ax.set_xlim([0.0, 1.0])
    fraction_ax.set_xticks([0.0, 0.5, 1.0])
    fraction_ax.set_xticklabels([0.0, 0.5, 1.0], fontdict=big_font)
    format_axis(fraction_ax)

def ddg_subplot(extract_df, ddg_ax, label_order=None):
    sns.barplot(data=extract_df.groupby('sequence', group_keys=False).agg({'ddg':'mean'}).reset_index(),
                y='sequence', x='ddg', ax=ddg_ax, color='0.7', linewidth=0.6, edgecolor="0.0",
                ci=None, order=label_order)
    sns.stripplot(data=extract_df, y='sequence', x='ddg', ax=ddg_ax,
                  order=label_order, color='0.0', size=1.2, jitter=0.2)
    ddg_ax.set_xlabel("DDGn (kJ/mol)", fontdict=big_font)
    ddg_ax.set_ylabel('')
    ddg_ax.set_yticks([])
    ddg_ax.set_yticklabels([])
    xmin = extract_df['ddg'].min()
    xmax = extract_df['ddg'].max()
    jumps = 5 if xmax-xmin < 20 else 10
    xmin = int(xmin/jumps)
    xmax = int(xmax/jumps)
    xmin = xmin - 1 if xmin <= 0 else xmin
    xmax = xmax + 1 if xmax >= 0 else xmax
    ddg_ax.set_xlim([jumps*xmin, jumps*xmax])
    ddg_ax.set_xticks([jumps * i for i in range(xmin, xmax+1)])
    ddg_ax.set_xticklabels([jumps * i for i in range(xmin, xmax+1)], fontdict=big_font)
    format_axis(ddg_ax)
    
def matrix_subplot(labels, mat_ax, max_columns=None, plasmid_numbers=[]):
    if not max_columns:
        max_columns = max([len(x) for x in labels])
    max_rows = len(labels)
    mat_ax.hlines(y=-0.5, xmin=-0.5, xmax=max_columns-0.5, color='0.8', linewidth=0.6)
    mat_ax.vlines(x=-0.5, ymin=-0.5, ymax=max_rows-0.5, color='0.8', linewidth=0.6)
    for col_index in range(max_columns):
        mat_ax.vlines(x=col_index+0.5, ymin=-0.5, ymax=max_rows-0.5, color='0.8', linewidth=0.6)
        pass
    for row_index in range(max_rows):
        mat_ax.hlines(y=row_index+0.5, xmin=-0.5, xmax=max_columns-0.5, color='0.8', linewidth=0.6)
        for col_index in range(max_columns):
            if col_index >= len(labels[row_index]):
                break
            elif labels[row_index][col_index] == " ":
                continue
            text = mat_ax.text(col_index, len(labels)-row_index-1.1, labels[row_index][col_index],
                               fontdict=small_font, ha="center", va="center")
    mat_ax.set_xticks([])
    mat_ax.set_xticklabels([])
    mat_ax.set_yticks(list(range(len(plasmid_numbers))))
    mat_ax.set_yticklabels(plasmid_numbers, fontdict=small_font)
    mat_ax.set_xlim((-1,max_columns))
    mat_ax.set_ylim((-0.5, max_rows-0.5))
    mat_ax.yaxis.set_tick_params(pad=0, left=False)

def alanine_block_plot(extract_df, wt_sequence, ddgi_scores, core_length=0):
    variants_ddgn = extract_df.groupby('sequence', group_keys=False).agg({'ddg':'mean', 'pep_plasmid':'first'}).reset_index()
    
    seq_alignments, labels, label_order, labels_aligned = da.seq_alignment(wt_sequence, variants_ddgn, score='ddg')
    
    variants_ddgn['order'] = variants_ddgn['sequence'].apply(lambda x: label_order.index(x))
    plasmid_numbers = variants_ddgn.sort_values('order', ascending=False)['pep_plasmid'].astype('int')
    
    binary_df, ddg_scores = da.aln2binary_df(wt_sequence, seq_alignments, invert=True)
    
    ddgi_barchart_height = 1
    side_barchart_width = 0.6
    bar_size = 0.05
    spacer = 0.1
    matrix_width = (len(wt_sequence)-core_length+1.5)*bar_size
    matrix_height = len(label_order)*bar_size
    total_width = (2*side_barchart_width+3*spacer) + matrix_width
    total_height = ddgi_barchart_height + matrix_height + spacer
    
    plt.figure(figsize=(total_width, total_height), dpi=600)
    
    hline  = (matrix_height + spacer)                        / total_height
    vline1 = (matrix_width  + spacer)                        / total_width
    vline2 = (matrix_width + side_barchart_width + 3*spacer) / total_width

    ddgi_ax =     plt.axes([0     , hline, matrix_width/total_width       , ddgi_barchart_height/total_height])
    fraction_ax = plt.axes([vline1, 0    , side_barchart_width/total_width, matrix_height/total_height])
    ddg_ax =      plt.axes([vline2, 0    , side_barchart_width/total_width, matrix_height/total_height])
    mat_ax =      plt.axes([0     , 0    , matrix_width/total_width       , matrix_height/total_height], frame_on=False)
    
    ### Section for plotting top part DDGi scores
    ddgi_ax.bar(list(range(len(ddgi_scores))), ddgi_scores, color='0.7', edgecolor='0.0', linewidth=0.6)
    ddgi_ax.set_xlim(-1,len(wt_sequence)-core_length)
    ddgi_ax.set_xticks([])
    ddgi_ax.set_ylabel('DDGi (kJ/mol)', fontdict=big_font)
    ymin = min(ddgi_scores)
    ymin = int(ymin - 1) if ymin < 0 else int(ymin)
    ymax = int(max(ddgi_scores)) + 1
    jumps = max([1, int((ymax-ymin) / 4)])
    ymin = int(ymin/jumps) - 1 if (ymin <=0) and (ymin % jumps > 0) else int(ymin/jumps)
    ymax = int(ymax/jumps) + 1 if (ymax >=0) and (ymax % jumps > 0) else int(ymax/jumps)
    ddgi_ax.set_ylim([ymin*jumps, ymax*jumps])
    ddgi_ax.set_yticks([i*jumps for i in range(ymin, ymax+1)])
    ddgi_ax.set_yticklabels([i*jumps for i in range(ymin, ymax+1)], fontdict=big_font)
    ddgi_ax.set_xlim((-1,len(wt_sequence)-core_length))
    format_axis(ddgi_ax)

    ### Section for side plots
    fm_subplot(extract_df, fraction_ax, label_order=label_order)
    ddg_subplot(extract_df, ddg_ax, label_order=label_order)
    
    ### Section for plotting matrix
    matrix_subplot(labels, mat_ax, max_columns=len(wt_sequence) - core_length, plasmid_numbers=plasmid_numbers)
    return

def spring_plot(extract_df, optimal_plasmid, k=(None, None), plot=False, print_stats=False):
    """Function takes an extract dataframe and the plasmid number with optimal modification.
    Fits spring constants k1 and k2 to the data (or optionally plots the data with provided
    k constants).
    """
    optimal_sequence = extract_df[extract_df['pep_plasmid'] == optimal_plasmid]['sequence'].any()
    extract_df['spacing'] = extract_df['sequence'].apply(lambda x: len(x)-len(optimal_sequence))
    
    da.detection_threshold_adjust(extract_df)
    da.wt_normalize(optimal_plasmid, extract_df)
    da.calculate_ddg(extract_df)
    
    if k == (None, None):
        k1, k2 = da.fit_spring(extract_df)
    else:
        k1, k2 = k
    if print_stats:
        print("k1", k1, "k2", k2)
    
    if plot:
        xvals = np.linspace(-20, 20, 1000)
        yvals = da.spring_eq(xvals, k1, k2)

        ymax = max(int(max(extract_df['ddg'])/5)+1,3)

        figure = plt.figure(figsize=(0.9,0.1125*(ymax+1)),dpi=600)
        ax = plt.axes([0,0,1,1])
        ax.scatter(x=extract_df['spacing']*-1, y=extract_df['ddg'], color='black', s=0.4)
        ax.plot(xvals*-1, yvals, lw=0.6, color='black', zorder=0)

        ax.set_ylabel('DDGn (kJ/mol)', fontdict=big_font)
        ax.set_xlabel('d (residues)', fontdict=big_font)

        ax.set_yticks([i for i in range(-5, (5*ymax)+1, 5)])
        ax.set_yticklabels([i for i in range(-5, (5*ymax)+1, 5)], fontdict=big_font)
        ax.set_yticks(range(-5,(5*ymax)+1,1), minor=True)
        ax.set_ylim(-5,5*ymax)
        ax.set_xlim(-20,20)
        ax.set_xticks([-20,-10,0,10,20])
        ax.set_xticklabels([-20,-10,0,10,20], fontdict=big_font)
        ax.set_xticks(range(-20,21,1), minor=True)
        ax.xaxis.set_tick_params(width=0.6, which='minor', length=1.5, pad=1.5)
        ax.yaxis.set_tick_params(width=0.6, which='minor', length=1.5, pad=1.5)
        ax.xaxis.set_tick_params(width=0.6, which='major', length=2.5, pad=1.5)
        ax.yaxis.set_tick_params(width=0.6, which='major', length=2.5, pad=1.5)
        ax.spines['top'].set_linewidth(0.6)
        ax.spines['right'].set_linewidth(0.6)
        ax.spines['bottom'].set_linewidth(0.6)
        ax.spines['left'].set_linewidth(0.6)
    return k1, k2

def alignment_plot(extract_df, wt_plasmid):
    da.detection_threshold_adjust(extract_df)
    da.wt_normalize(wt_plasmid, extract_df)
    da.calculate_ddg(extract_df)
    
    variants_ddgn = extract_df.groupby('sequence', group_keys=False).agg({'ddg':'mean', 'pep_plasmid':'first'}).reset_index()
    
    wt_sequence = extract_df[extract_df['pep_plasmid'] == wt_plasmid]['sequence'].any()
    seq_alignments, labels, label_order, labels_aligned = da.seq_alignment(wt_sequence, variants_ddgn, score='ddg')
    
    variants_ddgn['order'] = variants_ddgn['pep_plasmid'].astype('int')
    variants_ddgn.at[variants_ddgn[variants_ddgn['pep_plasmid']==wt_plasmid].index, 'order'] = 0
    plasmid_numbers = variants_ddgn.sort_values('order', ascending=False)['pep_plasmid'].astype('int')
    
    binary_df, ddg_scores = da.aln2binary_df(wt_sequence, seq_alignments, invert=True)
    
    longest_seq = max([len(l) for l in label_order])
    
    barchart_width = 0.6
    bar_size = 0.05
    spacer = 0.1
    matrix_width  = (longest_seq+1.5)*bar_size
    matrix_height = len(label_order)*bar_size
    total_width   = (2*barchart_width+3*spacer) + matrix_width
    total_height  = matrix_height
    
    plt.figure(figsize=(total_width, total_height), dpi=600)
    
    vline1 = (matrix_width + spacer)                        / total_width
    vline2 = (matrix_width + barchart_width + 3*spacer)/ total_width
    
    fraction_ax = plt.axes([vline1, 0    , barchart_width/total_width, 1])
    ddg_ax =      plt.axes([vline2, 0    , barchart_width/total_width, 1])
    mat_ax =      plt.axes([0     , 0    , matrix_width/total_width  , 1], frame_on=False)

    ### Section for side plots
    fm_subplot(extract_df, fraction_ax, label_order=label_order)
    ddg_subplot(extract_df, ddg_ax, label_order=label_order)
    
    ### Section for plotting matrix
    matrix_subplot(labels_aligned, mat_ax, plasmid_numbers=plasmid_numbers)
    
def core_ordering(pep):
    pep = str(pep)
    if "-" in pep:
        pep, col = list(map(int, pep.split("-")))
    else:
        pep = int(pep)
        col = 0
    return int(str(pep)[0])*-1, pep, col
    
def core_matrix_plot(extract_df):
    label_order = list(extract_df.groupby('display_core', group_keys=False).mean().reset_index().
                       sort_values('peak_fraction', ascending=False)['display_core'])
    
    wt_display = " "*len(label_order[0])
    wt_index = label_order.index(wt_display)
    label_order.pop(wt_index)
    label_order.insert(0, wt_display)

    actual_labels = label_order.copy()
    actual_labels[0] = extract_df['wt'].iloc[0]
    
    barchart_width = 1
    bar_size = 0.05
    spacer = 0.1
    matrix_width  = (len(actual_labels[0])+1.5)*bar_size
    matrix_height = len(actual_labels)*bar_size
    total_width   = barchart_width + matrix_width
    total_height  = matrix_height
    
    plt.figure(figsize=(total_width, total_height), dpi=600)
    
    vline = matrix_width / total_width
    
    fraction_ax = plt.axes([vline, 0    , barchart_width/total_width, 1])
    mat_ax =      plt.axes([0    , 0    , matrix_width/total_width  , 1], frame_on=False)

    ### Section for side plots
    fm_subplot(extract_df, fraction_ax, label_order=label_order, x='peak_fraction', y='display_core')
    
    ppl = extract_df.groupby('display_core').agg({'pep_plasmid':
        lambda x: "|".join(map(str, sorted(set(x), key=core_ordering)))}).reset_index()
    plasmid_numbers = [ppl[ppl['display_core'] == i]['pep_plasmid'].iloc[0] for i in label_order]
    ### Section for plotting matrix
    matrix_subplot(actual_labels, mat_ax, plasmid_numbers=plasmid_numbers[::-1])
    
groups = {'positive':['R', 'H', 'K'], 'negative':['D', 'E'],
          'polar':['C', 'S', 'T', 'N', 'Q'], 'aliphatic': ['A', 'M', 'V', 'I', 'L'],
          'aromatic': ['F', 'Y', 'W'], 'misc': ['G', 'P']}
group_colors = {'positive':'tab:red',
                'negative':'tab:blue',
                'polar':'tab:green',
                'aliphatic':'tab:purple',
                'aromatic':'tab:olive',
                'misc':'none'}
group_lookup = dict([(aa, k) for k,v in groups.items() for aa in v])
color_lookup = dict([(aa, group_colors[g]) for aa, g in group_lookup.items()])

from Bio.Align import substitution_matrices
blosum80 = substitution_matrices.load("BLOSUM80")
raw_blosum = {tuple(sorted(pair)): value for pair, value in blosum80.items() if pair[0] != pair[1]}
maxval = max(raw_blosum.values())
minval = min(raw_blosum.values())

blosum_dict = {}
all_aas = set(p[0] for p in raw_blosum.keys()) | set(p[1] for p in raw_blosum.keys())

for aa1 in all_aas:
    blosum_dict[aa1] = {'-': 1.0}
    for aa2 in all_aas:
        pair = tuple(sorted((aa1, aa2)))
        if aa1 == aa2:
            blosum_dict[aa1][aa2] = 0.0
        elif pair in raw_blosum:
            score = raw_blosum[pair]
            normalized_score = 1 - ((score - minval) / (maxval - minval))
            blosum_dict[aa1][aa2] = normalized_score

def wt_sorter(wt_char, aa, all_aas):
    blosums = [[c, blosum_dict[c][wt_char],group_lookup[c]] for c in all_aas]
    for c_info in blosums:
        same_cat = [other_info for other_info in blosums if other_info[2] == c_info[2]]
        c_info.append(max([sc[1] for sc in same_cat]))
        c_info.append(np.mean([sc[1] for sc in same_cat]))
        c_info.append(1 if c_info[2] == group_lookup[wt_char] else 0)
        c_info.append(0 if c_info[2] == 'misc' else 1)
    #0 is the char
    #1 is the blosum score
    #2 is the category
    #3 is the max blosum score for that category
    #4 is the mean blosum score for that category
    #5 is 0/1 based on if it matches into aa category
    #6 is if its NOT category 'misc'
    for aa_info in blosums:
        if aa_info[0] == aa:
            break
    
    return (aa_info[5], aa_info[6], aa_info[3], aa_info[4], aa_info[1])

empty_df = pd.DataFrame(columns=['mutation', 'mut_position'])
    
def motif_plot(wt_core, good_aas=empty_df.copy(), bad_aas=empty_df.copy()):
    
    max_goods = 0 if len(good_aas) == 0 else good_aas.groupby('mut_position', group_keys=False).size().reset_index()[0].max()
    max_bads = 0 if len(bad_aas) == 0 else bad_aas.groupby('mut_position', group_keys=False).size().reset_index()[0].max()
    
    fig_height = max_goods + max_bads + 1
    char_spacing = 0.1
    fig = plt.figure(figsize=(char_spacing*len(wt_core), char_spacing*fig_height), dpi=600)
    ax = plt.axes([0,0,1,1], frame_on=False)

    for i, wt_char in enumerate(wt_core):
        text = ax.text(i*char_spacing, 0, wt_char, fontdict=big_font, ha="center", va="center_baseline")

        rect = matplotlib.patches.Rectangle(((i*char_spacing)-(char_spacing/2),0-(char_spacing/2)),
                                     char_spacing, char_spacing, 
                             edgecolor='none', facecolor=color_lookup[wt_char], alpha=0.3)
        ax.add_patch(rect)

        good_aas_position = list(good_aas[good_aas['mut_position'] == i]['mutation'])
        good_aas_position = sorted(good_aas_position, key=lambda x: wt_sorter(wt_char, x, good_aas_position), reverse=True)

        for j, aa in enumerate(good_aas_position):
            text = ax.text(i*char_spacing, -(j+1)*char_spacing, aa, fontdict=big_font, ha="center", va="center_baseline")
            rect = matplotlib.patches.Rectangle(((i*char_spacing)-(char_spacing/2), (-(j+1)*char_spacing)-(char_spacing/2)),
                                     char_spacing, char_spacing, edgecolor='none', facecolor=color_lookup[aa], alpha=0.3)
            ax.add_patch(rect)


        bad_aas_position = list(bad_aas[bad_aas['mut_position'] == i]['mutation'])
        bad_aas_position = sorted(bad_aas_position, key=lambda x: wt_sorter(wt_char, x, bad_aas_position), reverse=True)
        for j, aa in enumerate(bad_aas_position):
            text = ax.text(i*char_spacing, (j+1)*char_spacing, aa, fontdict=big_font, ha="center", va="center_baseline")
            rect = matplotlib.patches.Rectangle(((i*char_spacing)-(char_spacing/2),
                                      ((j+1)*char_spacing)-(char_spacing/2)),
                                     char_spacing, char_spacing, 
                             edgecolor='none', facecolor=color_lookup[aa], alpha=0.3)
            ax.add_patch(rect)
    rect = matplotlib.patches.Rectangle((-char_spacing/2,-char_spacing/2),
                                     (len(wt_core)*char_spacing),
                             char_spacing, 
                             edgecolor='black', facecolor='none', lw=0.6)
    ax.add_patch(rect)

    ax.set_ylim((-(max_goods+0.5)*char_spacing, (max_bads+0.5)*char_spacing))
    ax.set_xlim((-char_spacing/2, (len(wt_core)+0.5)*char_spacing))
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.xaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    ax.yaxis.set_tick_params(width=0.6, which='both', length=2, pad=1.5)
    return

def get_lims(max_ever, adj=10000000, scale=1):
    y_lim = (int(max_ever/adj)+1)*adj
    if y_lim < (4*adj):
        y_ticks = list(range(0,y_lim+1,adj))
        y_labels = list(range(0,int(y_lim/adj)+1,1))
    elif y_lim < (8*adj):
        y_lim = (int(max_ever/(2*adj))+1)*(2*adj)
        y_ticks = list(range(0,y_lim+1,(adj*scale)))
        y_labels = list(range(0,int(y_lim/adj)+1,1*scale))
    elif y_lim < (16*adj):
        y_lim = (int(max_ever/(4*adj))+1)*(4*adj)
        y_ticks = list(range(0,y_lim+1,(2*adj*scale)))
        y_labels = list(range(0,int(y_lim/adj)+1,2*scale))
    elif y_lim < (32*adj):
        y_lim = (int(max_ever/(8*adj))+1)*(8*adj)
        y_ticks = list(range(0,y_lim+1,(4*adj*scale)))
        y_labels = list(range(0,int(y_lim/adj)+1,4*scale))
    elif y_lim < (64*adj):
        y_lim = (int(max_ever/(16*adj))+1)*(16*adj)
        y_ticks = list(range(0,y_lim+1,(4*adj*scale)))
        y_labels = list(range(0,int(y_lim/adj)+1,8*scale))
    else:
        print('huge peak', lcd['extract'])
    return y_ticks, y_labels

def plot_cm(msdf, ax, color='black'):
    ax.plot(msdf['rt'], msdf['int'], color=color, lw=1)
    ax.set_xlabel('Retention time (min)', fontdict=small_font, labelpad=2)
    ax.set_ylabel('Intensity (x$10^7$)', fontdict=small_font, labelpad=2)

def format_cm(ax, yticks=None, yticklabels=None, rt_min=2, rt_max=5.5,
              xlabel='Retention time (min)', ylabel='Intensity (x$10^7$)',
              xticks=None, xticklabels=None):
    if not xticks:
        xticks = np.array(range(int(rt_min*2),int(rt_max*2)+1)) / 2
        xlabels = xticks
    if yticks:
        ax.set_ylim([0, yticks[-1]])
        ax.set_yticks(yticks)
    if yticklabels:
        ax.set_yticklabels(yticklabels, fontdict=small_font)
    else:
        ax.set_yticklabels([])
    ax.set_xlim([rt_min, rt_max])
    ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels, fontdict=small_font)
    else:
        ax.set_xticklabels([])
    if xlabel:
        ax.set_xlabel(xlabel, fontdict=small_font,  labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2, fontdict=small_font)
    
    format_axis(ax)
    
def plot_spec(msdf, ax, color='black', width=1, adj=1000000, scale=1):
    ax.bar(msdf['mz'], msdf['int'], width=width, lw=0, color=color)

    y_ticks, y_labels = get_lims(msdf['int'].max(), adj=adj, scale=scale)
    ax.set_ylim([0, y_ticks[-1]])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontdict=small_font)
    ax.tick_params(axis='both', which='both', labelsize=4)
    
def mod_type_color(pt, count_others=False):
    if pt == 'mod':
        color='red'
    elif pt == 'unmod':
        color='green'
    elif pt == 'other' and count_others == False:
        color='blue'
    elif pt == 'other' and count_others == True:
        color='red'
    else:
        color='black'
    return color

def validation_plots(extract_df, count_others=False, rt_min=2, rt_max=5.5, save=True):
    sdf = da.get_full_sdf(extract_df, group=True)

    for ex, e_df in sdf.groupby('extract'):
        fig = plt.figure(figsize=(6, 2.5), dpi=2000)

        spec_ax =      plt.axes([0   , 0   , 0.48, 0.42])
        spec_ax_zoom = plt.axes([0.52, 0   , 0.48, 0.42])
        ecc_ax =       plt.axes([0.52, 0.58, 0.48, 0.42])
        tic_ax =       plt.axes([0   , 0.58, 0.48, 0.42])

        tic_df = e_df['lcd'].iloc[0].tic()
        tic_df = tic_df[(tic_df['rt'] < rt_max) & (tic_df['rt'] > rt_min)]
        
        y_ticks, y_labels = get_lims(tic_df['int'].max())
        plot_cm(tic_df, tic_ax, color='0.0')
        format_cm(tic_ax, yticks=y_ticks, yticklabels=y_labels, rt_min=rt_min, rt_max=rt_max)
        tic_ax.set_title("Total Ion Chromatogram", fontsize=6, pad=3)

        max_ever = 0
        rt_pairs = []
        expected_masses = []
        observed_charges = set()
        for pt, h in e_df.iterrows():
            pt = h['peak_type']
            color = mod_type_color(pt, count_others=count_others)
            
            msdf = h['msdf'].copy()
            msdf = msdf[(msdf['rt'] > rt_min) & (msdf['rt'] < rt_max)]
            msdf = msdf.groupby('rt', group_keys=False).agg({'int':'sum'}).reset_index()

            max_ever = msdf['int'].max() if msdf['int'].max() > max_ever else max_ever

            plot_cm(msdf, ecc_ax, color=color)
            
            expected_masses.append((h['mass'], color))
            charges = h['msdf'].groupby('charge').agg({'int':'sum', 'mz':'mean',
                                                       'eic_mz':'first',
                                                       'eic_dmz':'first'}).reset_index()
            observed_charges = observed_charges.union(set(charges['charge']))
            charge_of_interest = charges[charges['mz'] > 1000].sort_values('mz').iloc[0]

            if h['fit_area'] > 1:
                start_rt = h['fit_rt'] - 2*h['fit_width']
                end_rt = h['fit_rt'] + 2*h['fit_width'] + h['fit_skew']
                rt_pairs.append((start_rt, end_rt))
                tic_ax.add_patch(matplotlib.patches.Rectangle((start_rt, 0), end_rt-start_rt, y_ticks[-1],
                                                    edgecolor='none', facecolor='0.9'))

        y_ticks, y_labels = get_lims(max_ever)
        format_cm(ecc_ax, yticks=y_ticks, yticklabels=y_labels, rt_min=rt_min, rt_max=rt_max, ylabel=None)
        ecc_ax.set_title("Extracted Compound Chromatograms", fontsize=6, pad=3)

        msdf = e_df['lcd'].iloc[0]._df.copy()

        query = "|".join(["(rt > {} & rt < {})".format(rt[0], rt[1]) for rt in rt_pairs])
        spectra_df = msdf.query(query)
        spectra_df = spectra_df[(spectra_df['mz'] < 2000) & (spectra_df['mz'] > 500)]

        plot_spec(spectra_df, spec_ax, color='black')
        spec_ax.set_xlim([500, 2000])
        spec_ax.set_xticks([500, 750, 1000, 1250, 1500, 1750, 2000])
        spec_ax.set_xticklabels([500, 750, 1000, 1250, 1500, 1750, 2000], fontdict=small_font)
        spec_ax.set_xlabel('$m/z$', fontdict=small_font, labelpad=2)
        spec_ax.set_ylabel('Intensity (x$10^6$)',  fontdict=small_font, labelpad=2)
        spec_ax.set_title("Spectrum", fontsize=6, pad=3)
        format_axis(spec_ax)
        
        charge_of_interest = charge_of_interest['charge']
        expected_masses = [((mass+charge_of_interest) / charge_of_interest, color) for mass, color in expected_masses]
        mzmin = (int(min([m for m,c in expected_masses]))) - 5
        mzmax = (int(max([m for m,c in expected_masses]))) + 5

        spectra_df = spectra_df[(spectra_df['mz'] < mzmax) & (spectra_df['mz'] > mzmin)]


        for mz, color in expected_masses:
            spec_ax_zoom.add_patch(matplotlib.patches.Rectangle(
                    (mz-(5/charge_of_interest),0),
                    2*(5/charge_of_interest),
                    y_ticks[-1],
                    edgecolor='none',
                    facecolor=color,
                    alpha=0.4))
        
        plot_spec(spectra_df, spec_ax_zoom, color='black', width=0.03)
        spec_ax_zoom.set_xlim([mzmin, mzmax])
        spec_ax_zoom.set_xlabel('$m/z$', fontdict=small_font, labelpad=2)

        format_axis(spec_ax_zoom)
        
        c = str(int(charge_of_interest))
        spec_ax_zoom.set_title("Spectrum (Adduct: [M+" + c + "H$^+]^{" + c + "+}$)", fontsize=6, pad=3)
        
        plt.text(0.02, 0.58, 'a', fontdict=big_font, transform=plt.gcf().transFigure)
        if save:
            plt.savefig("./matplotlib/{}_{}.png".format(e_df['pep_plasmid'].iloc[0], e_df['extract'].iloc[0]),
                        bbox_inches='tight', pad_inches=0)

def wrap_list(pep_list, length=20):
    if type(length) == int:
        length = [length] * (int(len(pep_list) / length) + 1)
    start = 0
    for end in length:
        end = min([start+end, len(pep_list)])
        yield pep_list[start:end]
        start = end
        if start == len(pep_list):
            break
'''        
def variant_raw_plots(pep_plasmids, sdf, count_others=False, rt_min=2, rt_max=5.5, sort=False,
                      wrap=20, save_prefix="", save_index=None, plot_replicate=1, save=True):
    
    pep_plasmids = [pp if type(pp) == int else pp if "-" in pp else int(pp) for pp in pep_plasmids]
    if sort:
        pep_plasmids = sorted(pep_plasmids, key=core_ordering)
    if wrap:
        if len(pep_plasmids) > wrap:
            for i, pp in enumerate(wrap_list(pep_plasmids, length=wrap)):
                variant_raw_plots(pp, sdf, count_others=count_others, rt_min=rt_min,
                                  rt_max=rt_max, sort=sort, wrap=wrap, save_prefix=save_prefix,
                                  save_index=i+1, plot_replicate=plot_replicate, save=save)
            return
    extract_df = da.sub_pivot_df(pep_plasmids, sdf, group=False)
    sdf = da.get_full_sdf(extract_df, group=False)
    sub_df = la.lcms_df_processor(sdf, group_peaks=False, mod_other=False)
        
    y_dist = 0.42
    total_y = y_dist * len(pep_plasmids)
    y_fraction = y_dist / total_y
    y_height = (y_dist-0.12) / total_y
    fig = plt.figure(figsize=(2.9,total_y), dpi=2000)

    rt_min = 2
    rt_max = 5.5
    
    i=1
    for pp in pep_plasmids:
        g = sub_df[sub_df['pep_plasmid'] == pp]
        if len(set(g['extract'])) < 3:
            print(pp, set(g['extract']))
            continue
        extract = sorted(set(g['extract']))[plot_replicate]
        g = g[g['extract'] == extract].sort_values('fit_area')

        y_spot = (len(pep_plasmids)-i)*y_fraction
        unmod_ax    = plt.axes([0   , y_spot, 0.28, y_height])
        mod_ax      = plt.axes([0.31, y_spot, 0.28, y_height])
        spectrum_ax = plt.axes([0.65, y_spot, 0.35, y_height])
        i += 1

        max_ever = 0
        unmod_area = g[g['peak_type'] == 'unmod']['fit_area'].iloc[0]
        mod_area = g[g['peak_type'] == 'mod']['fit_area'].iloc[0]
        others = 0
        nothing_worked=True
        for pt, h in g.iterrows():
            pt = h['peak_type']
            color = mod_type_color(pt, count_others=count_others)
            
            msdf = h['msdf'].copy()
            msdf = msdf[(msdf['rt'] > rt_min) & (msdf['rt'] < rt_max)]
            msdf = msdf.groupby('rt', group_keys=False).agg({'int':'sum'}).reset_index()

            max_ever = max_ever if max_ever > msdf['int'].max() else msdf['int'].max()

            if pt == 'unmod':
                unmod_ax.plot(msdf['rt'], msdf['int'], color=color, lw=1)
            else:
                mod_ax.plot(msdf['rt'], msdf['int'], color=color, lw=1)

            peak_area = h['fit_area']
            peak_rt = h['fit_rt']
            peak_width = h['fit_width']
            peak_skew = h['fit_skew']
            peak_baseline = h['fit_baseline']
            if (pt == 'mod') and (h['mod_area'] > 1):
                start_rt = peak_rt - 2*peak_width
                end_rt = peak_rt + 2*peak_width
            elif (pt == 'unmod') and (mod_area == 0):
                start_rt = peak_rt - 2*peak_width
                end_rt = peak_rt + 2*peak_width

            if h['mod_area'] > 1:
                x = np.array(range(int(rt_min*60), int(rt_max*60)+1)) / 60
                y = la.asym_peak(x, peak_area, peak_rt, peak_width, peak_skew, peak_baseline)
                text_offset = 0.2
                if pt == 'unmod':
                    unmod_ax.plot(x, y, color='black', lw=0.6)
                    unmod_ax.text(0.96, 0.78, round(peak_area/100000)/10, color=color,
                             transform=unmod_ax.transAxes, ha='right', fontsize=4)
                elif pt == 'mod':
                    mod_ax.plot(x, y, color='black', lw=0.6)
                    nothing_worked=False
                    mod_ax.text(0.96, 0.78, round(peak_area/100000)/10, color=color,
                             transform=mod_ax.transAxes, ha='right', fontsize=4)
                elif pt == 'other':
                    y_position = 30000000 - 8000000*others
                    mod_ax.plot(x, y, color='black', lw=0.6)
                    others += 1
                    mod_ax.text(0.96, 0.78-(0.2*others), round(peak_area/100000)/10,
                             color=color, transform=mod_ax.transAxes, ha='right',
                             fontsize=4)

        y_ticks, y_labels = get_lims(max_ever, scale=2)

        if i == len(pep_plasmids)+1:
            format_cm(unmod_ax, yticks=y_ticks, yticklabels=y_labels, rt_min=rt_min, rt_max=rt_max,
                  ylabel="{}".format(h['pep_plasmid']), xticks=[2,3,4,5], xticklabels=[2,3,4,5],
                  xlabel='Retention time (min)')
            format_cm(mod_ax, yticks=y_ticks, rt_min=rt_min, rt_max=rt_max,
                  ylabel='', xticks=[2,3,4,5], xticklabels=[2,3,4,5], xlabel='Retention time (min)')
        else:
            format_cm(unmod_ax, yticks=y_ticks, yticklabels=y_labels, rt_min=rt_min, rt_max=rt_max,
                  ylabel="{}".format(h['pep_plasmid']), xlabel='', xticks=[2,3,4,5])
            format_cm(mod_ax, yticks=y_ticks, rt_min=rt_min, rt_max=rt_max, 
                      xlabel='', ylabel='', xticks=[2,3,4,5])

            
        if not nothing_worked:
            spectra_df = g.iloc[0]['lcd']._df.copy()
            spectra_df = spectra_df[(spectra_df['rt'] > start_rt) & (spectra_df['rt'] < end_rt)]
            spectra_df = spectra_df.groupby('mz', group_keys=False).agg({'int':'sum'}).reset_index()
            plot_spec(spectra_df, spectrum_ax, adj=10000000, scale=2)
        else:
            spectrum_ax.set_ylim([0  ,10000000])
            spectrum_ax.set_yticks([0,10000000])
            spectrum_ax.set_yticklabels([0,1], fontdict=small_font)
            spectrum_ax.tick_params(axis='both', which='both', labelsize=4)

        spectrum_ax.set_xlim(500, 2000)
        if i == len(pep_plasmids)+1:
            spectrum_ax.set_xticks([500, 1000, 1500, 2000])
            spectrum_ax.set_xticklabels([500, 1000, 1500, 2000], fontdict=small_font)
            spectrum_ax.set_xlabel('Mass-to-charge', fontdict=small_font, labelpad=1)
        else:
            spectrum_ax.set_xticks([500, 1000, 1500, 2000])
            spectrum_ax.set_xticklabels([])
        format_axis(spectrum_ax)
    if save:
        if type(save_index) == int:
            plt.savefig("./matplotlib/{}_{}.png".format(save_prefix, save_index), bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig("./matplotlib/{}.png".format(save_prefix), bbox_inches='tight', pad_inches=0)

'''