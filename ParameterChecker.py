import os
import argparse
import tomllib
import random
import sys
import pandas as pd
import pickle

import numpy as np

from Colors import Colors

class ParameterChecker:
    '''A class to check the validity of parameters for TSP rule generation.'''

    def __init__(self):
        pass

    def set_up_parser(self):
        parser = argparse.ArgumentParser(description='Suite of methods to find features, train a classifier and/or apply a classifier to experimental data.')

        # Required files.
        parser.add_argument("-c", "--config", required=False, default="config.toml", help="Path to the config file.")

        return parser

    def check_arguments(self, args):
        if not os.path.isfile(args.config):
            print(f"{Colors.ERROR}ERROR: Config file '{args.config}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        if not args.config.endswith('.toml'):
            print(f"{Colors.ERROR}ERROR: Config file '{args.config}' does not have a valid file extension. Expected '.toml'.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        try:
            print(f" - READING IN {args.config}", file=sys.stderr, flush=True)
            with open(args.config, "rb") as f:
                configs = tomllib.load(f)
            return configs
        except tomllib.TOMLDecodeError as e:
            print(f"{Colors.ERROR}ERROR decoding TOML: {e}{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        except Exception as e:
            print(f"{Colors.ERROR}ERROR: {e}{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

    def read_tsv(self, tsv_file_path):
        try:
            print(f"    - READING IN {tsv_file_path}", file=sys.stderr, flush=True)
            df = pd.read_csv(tsv_file_path, sep='\t')
            return df
        except pd.errors.ParserError as e:
            print(f"{Colors.ERROR}ERROR parsing TSV '{tsv_file_path}': {e}{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        except Exception as e:
            print(f"{Colors.ERROR}ERROR: {e}{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

    def read_pkl(self, pkl_file_path):
        try:
            print(f"    - READING IN {pkl_file_path}", file=sys.stderr, flush=True)
            with open(pkl_file_path, 'rb') as f:
                model_with_metadata = pickle.load(f)
                model = model_with_metadata['model']
                model._sklearn_version = model_with_metadata['sklearn_version']
            return model
        except pickle.UnpicklingError as e:
            print(f"{Colors.ERROR}ERROR unpickling the model '{pkl_file_path}': {e}{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        except Exception as e:
            print(f"{Colors.ERROR}ERROR: {e}{Colors.END}", file=sys.stderr, flush=True)
            print(f"{Colors.ERROR} Only models generated with this pipeline can be passed in through 'model_file'.")
            raise SystemExit(1)

    def check_configurations_project_settings(self, configs):
        print(" - CHECKING PROJECT SETTINGS", file=sys.stderr, flush=True)
        # check project settings
        # all but seed must be booleans
        # all but seed cannot be False
        if not isinstance(configs['find_features'], bool):
            print(f"{Colors.ERROR}ERROR: 'find_features' must be a boolean. Type of 'find_features' is: {type(configs['find_features'])}.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        if not isinstance(configs['train_model'], bool):
            print(f"{Colors.ERROR}ERROR: 'train_model' must be a boolean. Type of 'train_model' is: {type(configs['train_model'])}.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        if not isinstance(configs['apply_model'], bool):
            print(f"{Colors.ERROR}ERROR: 'apply_model' must be a boolean. Type of 'apply_model' is: {type(configs['apply_model'])}.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        if configs['find_features'] is False and configs['train_model'] is False and configs['apply_model'] is False:
            print(f"{Colors.ERROR}ERROR: at least one of 'find_features', 'train_model', and 'apply_model' must be true.{Colors.END}")
            raise SystemExit(1)

        # seed must be "random" or int
        if configs['seed'] == "random":
            configs['seed'] = None
        elif isinstance(configs['seed'], bool) or not isinstance(configs['seed'], int):
            print(f"{Colors.WARNING}WARNING: 'seed' {configs['seed']} is not type 'int'. Changing 'seed' to 'random'.{Colors.END}", file=sys.stderr, flush=True)
            configs['seed'] = None
        else:
            random.seed(configs['seed'])
            np.random.seed(configs['seed'])
            print(f"{Colors.INFO}INFO: Using fixed seed {configs['seed']}.{Colors.END}", file=sys.stderr, flush=True)

        # input_files can be "reference" or "individual"
        if configs['input_files'] != "reference" and configs['input_files'] != "individual":
            print(f"{Colors.ERROR}ERROR: 'input_files' must be \"reference\" or \"individual\", got {configs['input_files']}.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

    def check_configurations_files(self, configs):
        print(" - CHECKING FILES", file=sys.stderr, flush=True)
        # check files
        # check that the FS, train, and validate paths are not the same if they're not all empty
        if configs['input_files'] == "individual" and configs['train_model']:
            identical_files = 0
            if (configs['feature_quant_file'] + configs['feature_meta_file']) == (configs['train_quant_file'] + configs['train_meta_file']) and (configs['feature_quant_file'] + configs['feature_meta_file']) != "":
                identical_files += 1
            if (configs['feature_quant_file'] + configs['feature_meta_file']) == (configs['validate_quant_file'] + configs['validate_meta_file']) and (configs['feature_quant_file'] + configs['feature_meta_file']) != "":
                identical_files += 1
            if (configs['train_quant_file'] + configs['train_meta_file']) == (configs['validate_quant_file'] + configs['validate_meta_file']) and (configs['train_quant_file'] + configs['train_meta_file']) != "":
                identical_files += 1

            if identical_files > 0:
                print(f"{Colors.ERROR}ERROR: 'feature_quant_file'+'feature_meta_file', 'train_quant_file'+'train_meta_file', and 'validate_quant_file'+'validate_meta_file' must all be different, unless empty.{Colors.END}", file=sys.stderr, flush=True)
                print(f"{Colors.ERROR}        If you would like to use one reference dataset to both select features and train a model, use 'input_files = \"reference\"' and 'reference_quant_file' and 'reference_meta_file'.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)
        
        # if path is required based on project settings:
        #   - must be valid path
        #   - must have valid extension
        configs['split_for_FS'] = False
        configs['split_for_train'] = False
        configs['split_for_validate'] = False

        if configs['find_features']:
            # required paths:
            #   - reference_quant_file
            #   - reference_meta_file
            #   OR
            #   - feature_quant_file
            #   - feature_meta_file
            if configs['input_files'] == "reference":
                configs['split_for_FS'] = True

                if configs['feature_quant_file'] != "" or configs['feature_meta_file'] != "":
                    print(f"{Colors.WARNING}WARNING: 'input_files' set to \"reference\" but 'feature_quant_file' or 'feature_meta_file' not empty; using only specified reference files.{Colors.END}", file=sys.stderr, flush=True)
            else:
                if not os.path.isfile(configs['feature_quant_file']):
                    print(f"{Colors.ERROR}ERROR: 'feature_quant_file' '{configs['feature_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not os.path.isfile(configs['feature_meta_file']):
                    print(f"{Colors.ERROR}ERROR: 'feature_meta_file' '{configs['feature_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['feature_quant_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'feature_quant_file' '{configs['feature_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['feature_meta_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'feature_meta_file' '{configs['feature_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                configs['feature_quant_table'] = self.read_tsv(configs['feature_quant_file'])
                configs['feature_meta_table'] = self.read_tsv(configs['feature_meta_file'])
        else:
            # required paths:
            #   - feature_file
            if not os.path.isfile(configs['feature_file']):
                print(f"{Colors.ERROR}ERROR: 'feature_file' '{configs['feature_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            if not configs['feature_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'feature_file' '{configs['feature_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            configs['feature_table'] = self.read_tsv(configs['feature_file'])

        if configs['train_model']:
            # required paths:
            #   - reference_quant_file
            #   - reference_meta_file
            #   OR
            #   - train_quant_file
            #   - train_meta_file
            #   - validate_quant_file
            #   - validate_meta_file
            if configs['input_files'] == "reference":
                configs['split_for_train'] = True
                configs['split_for_validate'] = True

                if configs['train_quant_file'] != "" or configs['train_meta_file'] != "":
                    print(f"{Colors.WARNING}WARNING: 'input_files' set to \"reference\" but 'train_quant_file' or 'train_meta_file' not empty; using only specified reference files.{Colors.END}", file=sys.stderr, flush=True)

                if configs['validate_quant_file'] != "" or configs['validate_meta_file'] != "":
                    print(f"{Colors.WARNING}WARNING: 'input_files' set to \"reference\" but 'validate_quant_file' or 'validate_meta_file' not empty; using only specified reference files.{Colors.END}", file=sys.stderr, flush=True)
            else:
                # train/test dataset
                if not os.path.isfile(configs['train_quant_file']):
                    print(f"{Colors.ERROR}ERROR: 'train_quant_file' '{configs['train_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not os.path.isfile(configs['train_meta_file']):
                    print(f"{Colors.ERROR}ERROR: 'train_meta_file' '{configs['train_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['train_quant_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'train_quant_file' '{configs['train_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['train_meta_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'train_meta_file' '{configs['train_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                configs['train_quant_table'] = self.read_tsv(configs['train_quant_file'])
                configs['train_meta_table'] = self.read_tsv(configs['train_meta_file'])

                # validation dataset
                if not os.path.isfile(configs['validate_quant_file']):
                    print(f"{Colors.ERROR}ERROR: 'validate_quant_file' '{configs['validate_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not os.path.isfile(configs['validate_meta_file']):
                    print(f"{Colors.ERROR}ERROR: 'validate_meta_file' '{configs['validate_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['validate_quant_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'validate_quant_file' '{configs['validate_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['validate_meta_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'validate_meta_file' '{configs['validate_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                configs['validate_quant_table'] = self.read_tsv(configs['validate_quant_file'])
                configs['validate_meta_table'] = self.read_tsv(configs['validate_meta_file'])
        else:
            if configs['apply_model']:
                # required paths:
                #   - model_file
                if not os.path.isfile(configs['model_file']):
                    print(f"{Colors.ERROR}ERROR: 'model_file' '{configs['model_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['model_file'].endswith('.pkl'):
                    print(f"{Colors.ERROR}ERROR: 'model_file' '{configs['model_file']}' does not have a valid file extension. Expected '.pkl'.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

                if not configs['find_features']:
                    configs['model'] = self.read_pkl(configs['model_file'])
                else:
                    print(f"{Colors.ERROR}ERROR: 'find_features' and 'apply_model' enabled while 'train_model' disabled. Generated features may not match loaded model.{Colors.END}", file=sys.stderr, flush=True)
                    raise SystemExit(1)

        if configs['apply_model']:
            # required paths:
            #   - experimental_quant_file
            if not os.path.isfile(configs['experimental_quant_file']):
                print(f"{Colors.ERROR}ERROR: 'experimental_quant_file' '{configs['experimental_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            if not configs['experimental_quant_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'experimental_quant_file' '{configs['experimental_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            configs['experimental_quant_table'] = self.read_tsv(configs['experimental_quant_file'])

        if configs['input_files'] == "reference":
            if not os.path.isfile(configs['reference_quant_file']):
                print(f"{Colors.ERROR}ERROR: 'reference_quant_file' '{configs['reference_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            if not os.path.isfile(configs['reference_meta_file']):
                print(f"{Colors.ERROR}ERROR: 'reference_meta_file' '{configs['reference_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            if not configs['reference_quant_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'reference_quant_file' '{configs['reference_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            if not configs['reference_meta_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'reference_meta_file' '{configs['reference_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)

            configs['reference_quant_table'] = self.read_tsv(configs['reference_quant_file'])
            configs['reference_meta_table'] = self.read_tsv(configs['reference_meta_file'])

        if configs['output_dir'] == "cwd":
            configs['output_dir'] = os.getcwd()
            print(f"{Colors.INFO}INFO: Setting output directory to '{configs['output_dir']}'.{Colors.END}", file=sys.stderr, flush=True)
        if not os.path.isdir(configs['output_dir']):
            print(f"{Colors.ERROR}ERROR: 'output_dir' '{configs['output_dir']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

    def check_configurations_feature_selection(self, configs):
        # check find features settings
        if configs['find_features']:
            print(" - CHECKING FEATURE SELECTION SETTINGS", file=sys.stderr, flush=True)
            if isinstance(configs['k_rules'], bool) or not isinstance(configs['k_rules'], int) or not (0 < configs['k_rules'] <= 50):
                print(f"{Colors.WARNING}WARNING: 'k_rules' must be a positive integer between 1 and 50. Changing 'k_rules' to 15.{Colors.END}", file=sys.stderr, flush=True)
                configs['k_rules'] = 15

            if isinstance(configs['missingness_cutoff'], bool) or not isinstance(configs['missingness_cutoff'], (int, float, complex)) or not (0.0 <= configs['missingness_cutoff'] <= 1.0):
                print(f"{Colors.WARNING}WARNING: 'missingness_cutoff' must be a number between 0.0 and 1.0. Changing 'missingness_cutoff' to 0.5.{Colors.END}", file=sys.stderr, flush=True)
                configs['missingness_cutoff'] = 0.5

            if not isinstance(configs['disjoint'], bool):
                print(f"{Colors.WARNING}WARNING: 'disjoint' must be a boolean. Type of 'disjoint' is: {type(configs['disjoint'])}. Changing 'disjoint' to False.{Colors.END}", file=sys.stderr, flush=True)
                configs['disjoint'] = False
            if configs['disjoint']:
                print(f"{Colors.INFO}INFO: Disjoint filtering enabled.{Colors.END}", file=sys.stderr, flush=True)

            if not isinstance(configs['mutual_information'], bool):
                print(f"{Colors.WARNING}WARNING: 'mutual_information' must be a boolean. Type of 'mutual_information' is: {type(configs['mutual_information'])}. Changing 'mutual_information' to True.{Colors.END}", file=sys.stderr, flush=True)
                configs['mutual_information'] = True
            if not configs['mutual_information']:
                print(f"{Colors.WARNING}WARNING: Mutual information filtering disabled.{Colors.END}", file=sys.stderr, flush=True)

            if isinstance(configs['mutual_information_cutoff'], bool) or not isinstance(configs['mutual_information_cutoff'], (int, float, complex)) or not (0 <= configs['mutual_information_cutoff'] <= 1):
                print(f"{Colors.WARNING}WARNING: 'mutual_information_cutoff' must be a number between 0.0 and 1.0. Changing 'mutual_information_cutoff' to 0.7.{Colors.END}", file=sys.stderr, flush=True)
                configs['mutual_information_cutoff'] = 0.7

    def check_configurations_model_training(self, configs):
        # check train model settings
        if configs['train_model']:
            print(" - CHECKING MODEL TRAINING SETTINGS", file=sys.stderr, flush=True)
            if not isinstance(configs['impute_NA_missing'], bool):
                print(f"{Colors.WARNING}WARNING: 'impute_NA_missing' must be a boolean. Type of 'impute_NA_missing' is: {type(configs['impute_NA_missing'])}. Changing 'impute_NA_missing' to True.{Colors.END}", file=sys.stderr, flush=True)
                configs['impute_NA_missing'] = True

            if isinstance(configs['cross_val'], bool) or not isinstance(configs['cross_val'], int) or not (0 < configs['cross_val'] <= 20):
                print(f"{Colors.WARNING}WARNING: 'cross_val' must be a positive integer between 1 and 20. Changing 'cross_val' to 5.{Colors.END}", file=sys.stderr, flush=True)
                configs['cross_val'] = 5

            if configs['model_type'] not in ['RF', 'SVM']:
                print(f"{Colors.WARNING}WARNING: 'model_type' must be one of ('RF', 'SVM'). Got {configs['model_type']}. Changing 'model_type' to 'RF'.{Colors.END}", file=sys.stderr, flush=True)
                configs['model_type'] = 'RF'

            if configs['autotune_hyperparameters'] == "":
                configs['autotune_hyperparameters'] = None

            if configs['autotune_hyperparameters'] not in [None, 'random', 'grid']:
                print(f"{Colors.WARNING}WARNING: 'autotune_hyperparameters' must be one of ('', 'random', 'grid'). Got {configs['autotune_hyperparameters']}. Changing 'autotune_hyperparameters' to '' (no auto-tuning).{Colors.END}", file=sys.stderr, flush=True)
                configs['autotune_hyperparameters'] = None

            if configs['autotune_hyperparameters'] in ['random', 'grid']:
                print(f"{Colors.WARNING}WARNING: Auto-tuning hyperparameters will increase computational complexity and runtime.{Colors.END}",file=sys.stderr, flush=True)

            if isinstance(configs['autotune_n_iter'], bool) or not isinstance(configs['autotune_n_iter'], int) or not (0 < configs['autotune_n_iter'] <= 100):  # TODO decide if this is a good max
                print(f"{Colors.WARNING}WARNING: 'autotune_n_iter' must be a positive integer. Changing 'autotune_n_iter' to 20.{Colors.END}", file=sys.stderr, flush=True)
                configs['autotune_n_iter'] = 20

            if isinstance(configs['verbose'], bool) or not isinstance(configs['verbose'], int) or configs['verbose'] not in [0, 1, 2, 3, 4]:
                print(f"{Colors.WARNING}WARNING: 'verbose' must be one of (0, 1, 2, 3, 4). Got {configs['verbose']}. Changing 'verbose' to 0.{Colors.END}", file=sys.stderr, flush=True)
                configs['verbose'] = 0

    def check_configurations_experimental_classification(self, configs):
        # check apply model settings.
        if configs['apply_model']:
            print(" - CHECKING EXPERIMENTAL CLASSIFICATION SETTINGS", file=sys.stderr, flush=True)
            if configs['prediction_format'] not in ['classes', 'probabilities']:
                print(f"{Colors.WARNING}WARNING: 'prediction_format' must be one of ('classes', 'probabilities'). Got {configs['prediction_format']}. Changing 'prediction_format' to 'classes'.{Colors.END}", file=sys.stderr, flush=True)
                configs['prediction_format'] = 'classes'

    def run_paramater_checker(self):
        print("PARSING PARAMETERS", file=sys.stderr, flush=True)
        paramater_parser = self.set_up_parser()
        args = paramater_parser.parse_args()
        configs = self.check_arguments(args)

        print("CHECKING PARAMETERS", file=sys.stderr, flush=True)
        self.check_configurations_project_settings(configs)
        self.check_configurations_files(configs)
        self.check_configurations_feature_selection(configs)
        self.check_configurations_model_training(configs)
        self.check_configurations_experimental_classification(configs)

        return configs
    