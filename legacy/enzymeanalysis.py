from Bio import pairwise2
from Bio.Align import substitution_matrices
matrix = substitution_matrices.load("BLOSUM62")

import numpy as np
import scipy
import pandas as pd
import regex as re
import pickle

def sub_pivot_df(pps, sdf, group=True):
    """function takes a long form datatable of extracts and peaks (input sdf), filters
    for peptide plasmids of interest (input pps) and outputs a datatable with
    one row per extract, with columns for 'unmod' and 'mod' (or any other peak type)
    with the respective peak area. group option specifies if replicates should be grouped
    (by peptide sequence), with"""
    #filter for a sub-dataframe that includes just the peptide plasmids of interest
    sub_df = sdf[sdf['pep_plasmid'].isin(pps)]

    #Grab the set of sequences of interest (set to make non-redundant)
    sequences = set(sub_df['sequence'])

    #grab just the modification information (%mod fractions) for each extract
    stats_df = sub_df.pivot_table(index='extract', columns='peak_type',
                                values='mod_area', fill_value=0).reset_index()
    #metadata for all of the extracts
    meta_df = sub_df.groupby('extract', group_keys=False).first().reset_index().sort_values('extract')
    #merge metadata with stats data based on extract
    extract_df = meta_df.merge(stats_df, on='extract', how='inner')
    #if include_other:
    #    sub_data['mod'] = sub_data['mod'] + sub_data['other']
    if group:
        extract_df['replicate'] = 1
        return extract_df.groupby(
            ['sequence', 'mod_plasmid', 'modification description'], group_keys=False).agg(
            {'media':'first','ms':'first', 'pep_plasmid':'first', 'replicate':'sum', 'total_area':'mean',
             'mod':'mean','unmod':'mean', 'extract':'first'}).reset_index().sort_values('mod', ascending=False)
    else:
        return extract_df
    
def seq_alignment(wt_sequence, sdf, score='ddg', penalties=(-15, -2)):
    """Function takes a wild-type sequence and a dataframe of extracts of sequence variants to align to.
    Returns four lists, each list having one element per row of the input dataframe:
        seq_alignments - a list of tuples. Each tuple is the variant sequence, it's alignment to the
            wild-type sequence, and it's modification score (the type of score specified in 'score' input).
        labels_sparse - the variant sequence aligned to the wild-type sequence, positions that match
            wild-type are blank (space), positions that are mutated are the mutant amino acid (or '-' for
            gap). Note that for the wild-type sequence, the full sequence is here, no spaces, as a reference.
        labels - the variant sequence, unchanged/unaligned.
        labels_aligned - the variant sequence, aligned (with gaps)
    """
    seq_alignments = []
    labels = [wt_sequence]
    labels_sparse = [wt_sequence]
    labels_aligned = [wt_sequence]
    for ind, row in enumerate(sdf.iterrows()):
        #get rid of the index
        row = row[1]
        seq = row['sequence']
        mod_efficiency = row[score]
        #align the sequences, this will be a list of alignments, we just take the first one, since they are all
        # functionally equivalent for our purposes 
        alignments = pairwise2.align.globalds(wt_sequence, seq.split("*")[0], matrix, penalties[0], penalties[1])[0]
        #skip the wt sequence for the labels/order, so we added it at the beginning
        if alignments[1] == wt_sequence:
            seq_alignments.append((seq, alignments[1], mod_efficiency))
        else:
            seq_alignments.append((seq, alignments[1], mod_efficiency))
            labels_sparse.append("".join([i if i != w else " " for i, w in zip(alignments[1], wt_sequence)]))
            labels.append(seq)
            labels_aligned.append(alignments[1])
    return seq_alignments, labels_sparse, labels, labels_aligned

def aln2binary_df(wt_sequence, seq_alignments, invert=False):
    """function takes a wild-type sequence, and a list of sequence alignments from the seq_alignment function
    (list should be a list of tuples, one tuple per variant: (variant sequence, it's alignment to the
    wild-type sequence, and it's modification score)
    
    Returns a new dataframe that is one row per variant, and one column per amino acid position. At each
    position, the number 1 means that the variant sequence matches wild-type, 0 means the variant sequence
    does not match wild-type
    
    If invert, then the 1/0 assignment is switched.
    
    DOES NOT WORK IF THERE ARE GAPS (or rather, it just assumes that a gap is not a match, it is not recorded
    specially)
    """
    #Making a new dataframe (seq_df) that has a column for each amino acid
    indexes = [i for i in range(len(wt_sequence))]
    #temporary list, 1 element for each variant
    new_form = []
    mod_scores = []
    for variant_seq, aligned_seq, mod_eff in seq_alignments:
        binary_seq = []
        for s,w in zip(aligned_seq, wt_sequence):
            if s == w:
                binary_seq.append(0 if invert else 1)
            else:
                binary_seq.append(1 if invert else 0)
        new_form.append(binary_seq)
        mod_scores.append(mod_eff)
    binary_df = pd.DataFrame(new_form, columns = indexes)
    #convert modification scores into a numpy array and then into delta delta G for each variant
    mod_scores = np.array(mod_scores)
    return binary_df, mod_scores

def detection_threshold_adjust(extract_df, qqq_threshold=10000, qtof_threshold=1000):
    """Function takes a dataframe of extracts (each row is an extract) and adjusts for the noise level
    of the lcms. If modified and unmodified peptide are unobserved, the extract is removed. If
    unmodified or modified peptide is unobserved, it's peak area is set to the detection threshold
    so that the modified ratio or DDG of modification are real numbers.
    
    Requires the following columns to be in the dataframe:
    mod - the area of the peak corresponding to modified peptide in the extract
    total_area - the sum of all modification state peak areas in the extract
    ms - the mass spectrometer used
    
    Adds the following columns to the dataframe:
    mod_area - equal to the column 'mod'
    mod_fraction - mod_area / total_area
    mod_area_capped - the new mod_area, adjusted for the threshold
    total_area_capped - the new total_area, adjusted for the threshold
    mod_fraction_capped - mod_area_capped / total_area_capped
    mod_ratio_capped - mod_area_capped / (total_area_capped - mod_area_capped)
    """
    extract_df['mod_area'] = extract_df['mod']
    extract_df['mod_fraction'] = extract_df['mod_area'] / extract_df['total_area']
    extract_df['mod_area_capped'] = extract_df['mod_area']
    extract_df['total_area_capped'] = extract_df['total_area']
    #print(sub_df)
    for eind, extract in extract_df.iterrows():
        #if mod and total are zero, no peptide was observed, extract is removed since nothing
        # can be said about modification.
        if extract['mod_area'] == 0 and extract['total_area'] == 0:
            extract_df.drop(eind, inplace=True)
        #if mod was not observed, but unmod was, set the mod area to be the detection threshold
        elif extract['mod_area'] == 0:
            e_a = None
            if extract['ms'] == 'qtof':
                e_a = qtof_threshold
            elif extract['ms'] == 'qqq':
                e_a = qqq_threshold
            #change the mod area, and the total area to match
            extract_df.set_value(eind, 'mod_area_capped', e_a)
            extract_df.set_value(eind, 'total_area_capped', extract['total_area_capped'] + e_a)
        #if unmod was not observed, but mod was, set the unmod area to be the detection threshold
        if extract['mod_area'] == extract['total_area']:
            e_a = None
            if extract['ms'] == 'qtof':
                e_a = qtof_threshold
            elif extract['ms'] == 'qqq':
                e_a = qqq_threshold
            extract_df.set_value(eind, 'total_area_capped', extract['total_area_capped'] + e_a)

    extract_df['mod_fraction_capped'] = extract_df['mod_area_capped'] / extract_df['total_area_capped']
    extract_df['mod_ratio_capped']    = extract_df['mod_area_capped'] / (extract_df['total_area_capped'] -
                                                             extract_df['mod_area_capped'])
    
def wt_normalize(wt_plasmid, extract_df):
    #Grab the wild-type amino acid sequence
    wt_extracts = set(extract_df[extract_df['pep_plasmid'] == wt_plasmid]['extract'])
    #Get the wild-type modification efficiency to normalize by
    wt_mod_ratio = scipy.stats.gmean(extract_df[extract_df['extract'].isin(wt_extracts)]['mod_ratio_capped'])

    extract_df['mod_ratio_normalized'] = extract_df['mod_ratio_capped'] / float(wt_mod_ratio)
    
def calculate_ddg(extract_df):
    extract_df['ddg'] = (-(1.38*10**-23*310)*np.log(extract_df['mod_ratio_normalized'])*6.022*10**23)/1000
    extract_df['ddg'] = extract_df['ddg'].astype('float')
    
def ddgi(wt, extract_df):
    """function takes the wild-type precursor peptide plasmid number, a list of plasmid
    numbers that correspond to alanine block scan mutants, and peak dataframe.
    """
    detection_threshold_adjust(extract_df)
    wt_normalize(wt, extract_df)
    calculate_ddg(extract_df)
    
    variants_ddgn = extract_df.groupby('sequence', group_keys=False).agg({'ddg':'mean'}).reset_index()
    
    wt_sequence = extract_df[extract_df['pep_plasmid'] == wt]['sequence'].any()
    seq_alignments, labels, _, _ = seq_alignment(wt_sequence, variants_ddgn, score='ddg')
    
    binary_df, ddg_scores = aln2binary_df(wt_sequence, seq_alignments, invert=True)

    #get individual DDGi scalars for each variant based on the number of muated residues
    ddgi_scalar = [s/d if d!=0 else 0 for
                   s,d in zip(ddg_scores, binary_df.sum(axis=1))]
    #multiply that onto the binary_df to get the score contribution of each mutation
    ddgi_scores = binary_df.multiply(ddgi_scalar, axis=0)
    
    #replace with nan so 0 doesn't affect the mean, then take the mean to get mean ddgi per position across
    # all the variants to initialize the scores
    ddgi_scores = ddgi_scores.replace(0, np.nan).mean(axis=0)
    
    moved = 1
    while moved > 0.001:
        moved = 0
        movement = np.zeros(len(ddgi_scores))
        
        #multiply score at each position onto mutated positions in the binary_df, then sum each variant's
        # ddgi to get the full variant ddg. The difference between summed ddgi ('sum') and measured ddg ('ddg')
        # is what will be fixed in the iteration.
        score_df = binary_df.replace(0, np.nan).multiply(ddgi_scores, axis=1)
        score_df['sum'] = score_df.sum(axis=1)
        score_df['ddg'] = ddg_scores
        for position in binary_df.columns:
            if all(score_df[position].isnull()):
                #if there are no variants with mutations at this position, then continue
                continue
            mutated_df = score_df[score_df[position].notnull()]
            wrong_by = np.array(list(mutated_df['ddg'] - mutated_df['sum'])).mean()
            #Adding a scaler to the wrong by amount that is one-third the value of the ddgi value of that
            # position to discourage unlimited growth at each position.
            wrong_by = wrong_by - (ddgi_scores[position]/3.0)
            #move 1% of the total "wrong by" amount
            to_move = wrong_by / 100.0
               
            #sanity/bounding checks
            if ddgi_scores[position]+to_move < 0:
                if all(mutated_df['ddg']>0):
                    #don't allow a negative ddgi, if all variant ddg values are positive
                    to_move = 0
                    if ddgi_scores[position] < 0:
                        to_move = -ddgi_scores[position]
            elif ddgi_scores[position]+to_move > 0:
                if all(mutated_df['ddg'] < 0):
                    #don't allow a positive ddgi, if all variant ddg values are negative
                    to_move = 0
                    if ddgi_scores[position] > 0:
                        to_move = -ddgi_scores[position]
            for ddg in mutated_df['ddg']:
                #don't allow a ddgi value to get bigger than the variant ddg value
                if ddgi_scores[position]+to_move > ddg and ddg > 0:
                    to_move = 0
                    if ddgi_scores[position] > ddg:
                        #hit a maximum of ddg/2 for any given ddgi
                        to_move = (ddg/2)-ddgi_scores[position]
                elif ddgi_scores[position]+to_move < ddg and ddg < 0:
                    to_move = 0
                    if ddgi_scores[position] < ddg:
                        #hit a maximum of ddg/2 for any given ddgi
                        to_move = (ddg/2)-ddgi_scores[position]
            movement[position] = to_move
            
        moved = np.abs(movement).sum()
        ddgi_scores = np.add(ddgi_scores, movement)

    return wt_sequence, ddgi_scores

def spring_eq(x, k1, k2):
    delta = np.array([1 if xi > 0 else 0 for xi in x])
    return (((k1*delta*(x**2) + k2*(1-delta)*(x**2))/2)/1000)
    
def fit_spring(extract_df):
    (k1, k2), _ = scipy.optimize.curve_fit(f=spring_eq,
                                           xdata=extract_df['spacing'],
                                           ydata=extract_df['ddg'],
                                           bounds=[(0, 0),(100000, 100000)])
    return k1, k2

def fuzzy(core, query, mismatches=1):
    
    r = r"(" + core+"){s<=" + str(mismatches)+ "}"
    
    matches = re.findall(r, query)
    if matches:
        return matches[0]
    
def core_mutation_stats(subset, core, leader=""):
    subset['wt'] = core
    if leader:
        subset['leader'] = subset.apply(lambda x: x['sequence'][:len(x['sequence'])-len(x['core'])], axis=1)
    
    subset['display_core'] = subset.apply(
                lambda x: "".join([c if c != w else " " for c,w in zip(x['core'], x['wt'])]), axis=1)
    subset['mutation'] = subset['display_core'].apply(lambda x: x.strip())
    subset['mut_positions'] = subset['display_core'].apply(lambda x: 
                                                          [i for i, c in enumerate(x)
                                                          if c != " "])
    subset['num_mutations'] = subset['mut_positions'].apply(lambda x: len(x))
    
def get_seq_df(sdf, wt_plasmid, group_sequences=True):
    seq_df = sdf[sdf['peak_type'] == 'mod'].groupby('pep_plasmid', group_keys=False).agg(
                {'mod_plasmid':'first', 'sequence': 'first', 'peak_fraction':'std'}).reset_index()
    mod_plasmid = seq_df[seq_df['pep_plasmid'] == wt_plasmid]['mod_plasmid'].iloc[0]
    seq_df = seq_df[seq_df['mod_plasmid'] == mod_plasmid]
    wt_df  = seq_df[seq_df['pep_plasmid'] == wt_plasmid]
    seq_df = seq_df[seq_df['pep_plasmid'] != wt_plasmid]
    seq_df = seq_df[seq_df['peak_fraction'] < 0.5]
    if group_sequences:
        seq_df = seq_df.groupby('sequence').agg({'pep_plasmid':'last', 'mod_plasmid':'first'}).reset_index()
        wt_df = wt_df.groupby('sequence').agg({'pep_plasmid':'last', 'mod_plasmid':'first'}).reset_index()
    seq_df = pd.concat([seq_df, wt_df])
    return seq_df

def grab_peptides(sdf, wt_plasmid, core, leader="", mismatches=[0,4]):
    seq_df = get_seq_df(sdf, wt_plasmid, group_sequences=True)
    min_mismatches = mismatches[0]
    max_mismatches = mismatches[1]
    subset = seq_df
    if leader:
        subset = seq_df[seq_df['sequence'].str.contains(leader)]
    subset = subset[subset['sequence'].str.len() == (len(leader) + len(core))]
    subset['sequence'] = subset['sequence'].str.strip("*")
    subset = subset[~subset['sequence'].str.contains("*", regex=False)]
    subset['core'] = subset['sequence'].apply(lambda x: fuzzy(core, x, mismatches=max_mismatches))
    subset = subset[~subset['core'].isnull()]
    
    extract_df = sub_pivot_df(set(subset['pep_plasmid']), sdf, group=False)
    
    extract_df['sequence'] = extract_df['sequence'].str.strip("*")
    extract_df['core'] = extract_df['sequence'].apply(lambda x: fuzzy(core, x, mismatches=max_mismatches))
        
    core_mutation_stats(extract_df, core, leader=leader)
    
    extract_df = extract_df[(extract_df['num_mutations'] >=min_mismatches) |
                            (extract_df['pep_plasmid'] == wt_plasmid)]
    if leader:
        extract_df = extract_df[extract_df['leader'] == leader] 
    return extract_df

def grab_peptides_regex(sdf, regy, wt_plasmid, core, leader="", min_mismatches=1):
    seq_df = get_seq_df(sdf, wt_plasmid, group_sequences=True)
    
    subset = seq_df
    if leader:
        subset = seq_df[seq_df['sequence'].str.contains(leader)]
    subset = subset[subset['sequence'].str.len() == (len(leader) + len(core))]
    subset['sequence'] = subset['sequence'].str.strip("*")
    subset = subset[~subset['sequence'].str.contains("*", regex=False)]
    subset['core'] = subset['sequence'].apply(lambda x: re.findall(regy, x)[0] if len(re.findall(regy, x)) else np.nan)
    subset = subset[~subset['core'].isnull()]
    
    extract_df = sub_pivot_df(set(subset['pep_plasmid']), sdf, group=False)
    
    extract_df['sequence'] = extract_df['sequence'].str.strip("*")
    extract_df['core'] = extract_df['sequence'].apply(lambda x: re.findall(regy, x)[0] if len(re.findall(regy, x)) else np.nan)
        
    core_mutation_stats(extract_df, core, leader=leader)
    
    extract_df = extract_df[(extract_df['num_mutations'] >=min_mismatches) |
                            (extract_df['pep_plasmid'] == wt_plasmid)]
    if leader:
        extract_df = extract_df[extract_df['leader'] == leader] 
    return extract_df
    
def generate_motif(extract_df):
    
    wt_core = extract_df['wt'].iloc[0]
    wt_mf = extract_df[extract_df['mutation'] == ""]['peak_fraction'].mean()
    
    motif_df = extract_df.groupby('display_core', group_keys=False).agg({'peak_fraction':'mean',
                            'core':'first', 'mutation':'first', 'mut_positions':'first'}).reset_index()

    mutations = []
    for i, cv in motif_df.iterrows():
        #don't add variants that have multiple mutations with a below threshold fraction modified
        if (len(cv['mutation'].replace(" ", "")) > 1) and (cv['peak_fraction'] < 0.5*wt_mf):
            continue
        for mut, mut_pos in zip(cv['mutation'].replace(" ",""), cv['mut_positions']):
            mutations.append({'display_core': cv['display_core'], 'peak_fraction': cv['peak_fraction'],
                        'mutation': mut, 'mut_position': mut_pos})
    motif_df = pd.DataFrame(mutations)
    motif_df = motif_df.groupby(['mutation', 'mut_position'], group_keys=False).\
                    agg({'peak_fraction':'max'}).reset_index()
    
    good_aas = motif_df[motif_df['peak_fraction']>=0.5*wt_mf]
    good_aas = good_aas[good_aas['mutation'] != ""]
    bad_aas = motif_df[motif_df['peak_fraction']<0.5*wt_mf]
    
    return wt_core, good_aas, bad_aas

def get_full_sdf(e_df, group=True):
    if group:
        extracts = set(e_df.groupby('pep_plasmid').first().reset_index()['extract'])
    else:
        extracts = set(e_df['extract'])

    df_list = []
    for extract in extracts:
        with open('./extract_dataframes/{}.pickle'.format(int(extract)), 'rb') as f:
            df_list.append(pickle.load(f))
    sdf = pd.concat(df_list)

    return sdf