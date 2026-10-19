# This is the python script performing the analysis of many LC-MS files for RiPP modifications
import sys
import os
import pickle
import argparse
import ast


# science imports
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from pyteomics import mzml, mass

from classes_spectrometry import *
from modification_dictionary import aa_comp_custom
import peptide_functions as peptide_functions


parser = argparse.ArgumentParser(
    description='call the main script for RiPP modification analysis on a specified input excel file or csv file')
parser.add_argument('input', type=str, help='Path to the input excel or csv file containing the data')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Increase output verbosity, keeps track of all the intermediate steps')
parser.add_argument('-o', '--override', action='store_true', help='Override existing output files for the spectra, reparses spectrum')
parser.add_argument('-r', '--renew', action='store_true', help='Overwrite the experiment and associated output files')
parser.add_argument('-m', '--minimum_intensity', type=int, default=1000,
                    help='Minimum intensity threshold for peak detection (default: 1000)')
parser.add_argument('-d', '--modification_dictionary', type=str, default="None",
                    help='Path to a pickle file containing a custom modification dictionary')



# we define a main function to run the analysis, the main fucntion takes the arguments from the command line such as the input excel file
def main(input_file, verbose, override, renew, minimum_intensity, mod_dict):

    # we read the input excel or csv file
    if input_file.endswith('.xlsx') or input_file.endswith('.xls'):
        df = pd.read_excel(input_file)
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        print("Input file must be an excel (.xlsx or .xls) or csv (.csv) file")
        sys.exit(1)

    # we check that the required columns are present in the dataframe
    required_columns = ['sample_id', 'experiment_id', 'precursor_name', 'input_mzml_folder',
            'input_mzml', 'output_folder_name', 'peptide_sequence', 'digestion']

    for col in required_columns:
        if col not in df.columns:
            print(f"Missing required column: {col}")
            sys.exit(1)

    # we loop over each row in the dataframe
    for index, row in df.iterrows():
        ################################################################################
        # we create the paths for the mzml file and the output folder
        # we generate the MSExperiment object by feeding it the row of the dataframe as a dictionary
        if verbose:
            print(f"Processing row {index+1}/{len(df)}: {row['sample_id']} - {row['experiment_id']}")
        try:
            experiment = MSExperiment(**row.to_dict())
            if verbose:
                print(f"Experiment object created for {experiment.sample_id} - {experiment.experiment_id}")
                print(f"Creating MassSpectrum object for {experiment.sample_id} - {experiment.experiment_id}")
            # Accessing experiment.mzml_path here will trigger the FileNotFoundError if the file is missing
            spectrum = MassSpectrum(mzml_path=experiment.mzml_path, output_path=experiment.output_path, spectrum_id=experiment.spectrum_id)
        except FileNotFoundError as e:
            print(f"\nWarning: Skipping row {index+1} ({row['sample_id']}) because a file was not found: {e}\n")
            continue

        if verbose:
            print(f"MassSpectrum object created for {experiment.sample_id} - {experiment.experiment_id}")
            print(f"Loading spectrum data for {experiment.sample_id} - {experiment.experiment_id}")
        spectrum.load(override=override,minimum_intensity=minimum_intensity)
        if spectrum.ms1 is not None:
            print(f"Spectrum data already processed for {experiment.sample_id} - {experiment.experiment_id}, skipping to next row.")
        if verbose:
            print(f"Spectrum data loaded for {experiment.sample_id} - {experiment.experiment_id}")
            print(f"Creating Peptide object for {experiment.sample_id} - {experiment.experiment_id}")

        # we cleave the peptide_sequence, if experiment does not have core_start and core_end, we set them to 0 and None
        if not hasattr(experiment, 'core_start'):
            experiment.core_start = 0
        if not hasattr(experiment, 'core_end'):
            experiment.core_end = None
        # we sanitize any stop codons
        experiment.peptide_sequence = experiment.peptide_sequence.replace('*', '')
        # we generate the peptide targets for the unmodified peptide
        peptides = peptide_functions.generate_targets(peptide_sequence=experiment.peptide_sequence, digestion=experiment.digestion, mod_dict=mod_dict,
                                    missed_cleavages=1, min_length=6, max_length=30, core_only=True, core_start=experiment.core_start, core_end=experiment.core_end)
        if verbose:
            print(f"Peptide Targets from {experiment.peptide_sequence}:\n{peptides}")
        # we generate the Peptide objects for each peptide
        target_objects = []
        for tgt in peptides:
            tgt_object = TargetPeptide(sequence=tgt[1], mod_dict=mod_dict, fragment_cut=(tgt[0], tgt[2]), modifications={} )
            target_objects.append(tgt_object)

        # we generate the isotope patterns for each peptide
        for tgt in target_objects:
            tgt.get_isotopologues(mod_dict=aa_comp_custom, isotope_threshold=0.05, include_monoisotopic=True)
            # we generate the fragments for each peptide (we consider only b and y ions for now)
            tgt.generate_fragments(mod_dict=aa_comp_custom, types=['b', 'y'], max_charge=1)

        # we save the peptide objects to the experiment object as the attribute target_peptides
        experiment.target_peptides = target_objects

        # we check if modified_sequence is present as an attribute of experiment, if yes it's either a list or a string
        if hasattr(experiment, 'modified_sequences'):
            mod_seq_input = experiment.modified_sequences

            # Ensure input is a list of strings
            if isinstance(mod_seq_input, str):
                # Split comma-separated strings, handling potential spaces
                sequences_list = [seq.strip() for seq in mod_seq_input.split(',')]
            elif isinstance(mod_seq_input, list):
                sequences_list = mod_seq_input
            else:
                print(f"modified_sequence must be a string or a list of strings")
                sys.exit(1)

            # Sanitize all sequences in the list
            modified_sequences = [seq.strip().replace('*', '') for seq in sequences_list]

            if hasattr(experiment, 'modifications'):
                mods_input = experiment.modifications
                if isinstance(mods_input, str):
                    # To safely parse a string like "('a', 1), ('b', 2)", we wrap it in brackets
                    # to make it a valid list literal, then use ast.literal_eval.
                    # This is much safer and more robust than using eval() or split().
                    try:
                        # The final structure should be a list of lists,
                        # one list of modifications for each modified sequence.
                        # e.g., [('Dehydration', 1), ('Methylation', 2), ('Oxidation', 1)] is one such inner list.
                        modifications = ast.literal_eval(f"[{mods_input}]")
                    except (ValueError, SyntaxError) as e:
                        print(f"Error: Could not parse 'modifications' string: '{mods_input}'")
                        print(f"Details: {e}")
                        sys.exit(1)
                else:
                    print(f"Error: 'modifications' must be a string. Found: {type(mods_input)}")
                    sys.exit(1)
            # we renew the modified_sequences attribute of the experiment class
            experiment.modified_sequences = modified_sequences
            experiment.modifications = modifications
            print(f"Modified Sequences: {experiment.modified_sequences}, Modifications: {experiment.modifications}")
            # we loop over each modified sequence and generate the target peptides
            mod_target_objects = []
            for mod_seq, mods in zip(experiment.modified_sequences, experiment.modifications):
                print(f"Processing modified sequence: {mod_seq} with modifications: {mods}")
                mod_seq = mod_seq.replace('*', '')
                mod_peptides = peptide_functions.generate_targets(peptide_sequence=mod_seq, digestion=experiment.digestion, mod_dict=mod_dict,
                                                missed_cleavages=1, min_length=6, max_length=30, core_only=True, core_start=experiment.core_start, core_end=experiment.core_end)
                if verbose:
                    print(f"Modified Peptide Targets from {mod_seq}:\n{mod_peptides}")
                # we generate the Peptide objects for each modified peptide


                # since we do not know the position of the modifications in this cas, they are added as variable
                # modifications by passing them in a dictionary to the TargetPeptide object
                print(f"Modifications to apply: {mods}")
                mod_description = {mod[0]: mod[1] for mod in mods} if type(mods) == list else {mods[0]: mods[1]}
                temp_list = []
                for tgt in mod_peptides:
                    tgt_object = TargetPeptide(sequence=tgt[1], mod_dict=mod_dict, fragment_cut=(tgt[0], tgt[2]), modifications=mod_description)
                    temp_list.append(tgt_object)

                # we generate the isotope patterns for each modified peptide
                for tgt in temp_list:
                    tgt.get_isotopologues(mod_dict=aa_comp_custom, isotope_threshold=0.05, include_monoisotopic=True)
                    # we generate the fragments for each peptide (we consider only b and y ions for now)
                    tgt.generate_fragments(mod_dict=aa_comp_custom, types=['b', 'y'], max_charge=1)
                mod_target_objects += temp_list
            print(mod_target_objects)
            # we add the modified target objects to the experiment target peptides
            experiment.target_peptides = experiment.target_peptides + mod_target_objects
        elif hasattr(experiment, 'modifications'):
            # WORK IN PROGRESS
            modifications = experiment.modifications

        elif hasattr(experiment, 'possible_modifications'):
            # modifications are given as a list of tuples ("modification type", "modification number")
            # we do not cover cyclizations in this code yet, suports only linear modifications
            modifications = experiment.possible_modifications
            # since we do not know the position of the modifications in this cas, they are added as variable
            # modifications by passing them in a dictionary to the TargetPeptide object

            ########### WORK IN PROGRESS #############

        # we save the experiment object to the output folder
        if verbose:
            print(f"Peptide objects created for {experiment.sample_id} - {experiment.experiment_id}")
            print(f"Saving experiment object for {experiment.sample_id} - {experiment.experiment_id}")
        experiment.save(renew=renew)

if __name__ == "__main__":
    args = parser.parse_args()

    mod_dict_to_use = aa_comp_custom
    if args.modification_dictionary != "None":
        try:
            with open(args.modification_dictionary, 'rb') as f:
                mod_dict_to_use = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: Modification Dictionary file '{args.modification_dictionary}' not found.")
            sys.exit(1)

    main(input_file=args.input,
         verbose=args.verbose,
         override=args.override,
         renew=args.renew,
         minimum_intensity=args.minimum_intensity,
         mod_dict=mod_dict_to_use)
