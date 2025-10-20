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
            sys.exit(1)

        if not args.config.endswith('.toml'):
            print(f"{Colors.ERROR}ERROR: Config file '{args.config}' does not have a valid file extension. Expected '.toml'.{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)

        try:
            with open(args.config, "rb") as f:
                configs = tomllib.load(f)
            return configs
        except tomllib.TOMLDecodeError as e:
            print(f"{Colors.ERROR}ERROR decoding TOML: {e}{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.ERROR}ERROR: {e}{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)

    def read_tsv(self, tsv_file_path):
        try:
            print(f" - READING IN {tsv_file_path}", file=sys.stderr, flush=True)
            df = pd.read_csv(tsv_file_path, sep='\t')
            return df
        except pd.errors.ParserError as e:
            print(f"{Colors.ERROR}ERROR parsing TSV '{tsv_file_path}': {e}{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.ERROR}ERROR: {e}{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)

    def read_pkl(self, pkl_file_path):
        try:
            print(f" - READING IN {pkl_file_path}", file=sys.stderr, flush=True)
            with open(pkl_file_path, 'rb') as f:
                model = pickle.load(f)
            return model
        except pickle.UnpicklingError as e:
            print(f"{Colors.ERROR}ERROR unpickling the model '{pkl_file_path}': {e}{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"{Colors.ERROR}ERROR: {e}{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)

    def check_configurations(self, configs):
        # check project settings
        # all but seed must be booleans
        # all but seed cannot be False
        if not isinstance(configs['find_features'], bool):
            print(f"{Colors.ERROR}ERROR: 'find_features' must be a boolean. Type of 'find_features' is: {type(configs['find_features'])}.{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)

        if not isinstance(configs['train_model'], bool):
            print(f"{Colors.ERROR}ERROR: 'train_model' must be a boolean. Type of 'train_model' is: {type(configs['train_model'])}.{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)

        if not isinstance(configs['apply_model'], bool):
            print(f"{Colors.ERROR}ERROR: 'apply_model' must be a boolean. Type of 'apply_model' is: {type(configs['apply_model'])}.{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)

        if configs['find_features'] is False and configs['train_model'] is False and configs['apply_model'] is False:
            print(f"{Colors.ERROR}ERROR: at least one of 'find_features', 'train_model', and 'apply_model' must be true.{Colors.END}")

        # seed must be "random" or int
        if configs['seed'] == "random":
            configs['seed'] = None
        elif not isinstance(configs['seed'], int):
            print(f"{Colors.WARNING}WARNING: 'seed' {configs['seed']} is not type 'int'. Changing 'seed' to 'random'.{Colors.END}", file=sys.stderr, flush=True)
            configs['seed'] = None
        else:
            random.seed(configs['seed'])
            np.random.seed(configs['seed'])
            print(f"{Colors.INFO}INFO: Using fixed seed {configs['seed']}.{Colors.END}", file=sys.stderr, flush=True)

        # input_files can be "reference" or "individual"
        if configs['input_files'] != "reference" and configs['input_files'] != "individual":
            print(f"{Colors.ERROR}ERROR: 'input_files' must be \"reference\" or \"individual\", got {configs['input_files']}.{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)


        # check files
        # check that the FS, train, and validate paths are not the same if they're not all empty
        if configs['input_files'] == "individual":
            identical_files = 0
            if (configs['feature_quant_file'] + configs['feature_meta_file']) == (configs['train_quant_file'] + configs['train_meta_file']) and (configs['feature_quant_file'] + configs['feature_meta_file']) != "":
                identical_files += 1
            if (configs['feature_quant_file'] + configs['feature_meta_file']) == (configs['validate_quant_file'] + configs['validate_meta_file']) and (configs['feature_quant_file'] + configs['feature_meta_file']) != "":
                identical_files += 1
            if (configs['train_quant_file'] + configs['train_meta_file']) == (configs['validate_quant_file'] + configs['validate_meta_file']) and (configs['train_quant_file'] + configs['train_meta_file']) != "":
                identical_files += 1

            if identical_files > 0:
                print(f"{Colors.ERROR}ERROR: 'feature_quant_file'+'feature_meta_file', 'train_quant_file'+'train_meta_file', and 'validate_quant_file'+'validate_meta_file' must all be different, unless empty.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)
        
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
                    sys.exit(1)

                if not os.path.isfile(configs['feature_meta_file']):
                    print(f"{Colors.ERROR}ERROR: 'feature_meta_file' '{configs['feature_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['feature_quant_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'feature_quant_file' '{configs['feature_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['feature_meta_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'feature_meta_file' '{configs['feature_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                configs['feature_quant_table'] = self.read_tsv(configs['feature_quant_file'])
                configs['feature_meta_table'] = self.read_tsv(configs['feature_meta_file'])
        else:
            # required paths:
            #   - feature_file
            if not os.path.isfile(configs['feature_file']):
                print(f"{Colors.ERROR}ERROR: 'feature_file' '{configs['feature_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

            if not configs['feature_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'feature_file' '{configs['feature_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

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
                    sys.exit(1)

                if not os.path.isfile(configs['train_meta_file']):
                    print(f"{Colors.ERROR}ERROR: 'train_meta_file' '{configs['train_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['train_quant_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'train_quant_file' '{configs['train_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['train_meta_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'train_meta_file' '{configs['train_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                configs['train_quant_table'] = self.read_tsv(configs['train_quant_file'])
                configs['train_meta_table'] = self.read_tsv(configs['train_meta_file'])

                # validation dataset
                if not os.path.isfile(configs['validate_quant_file']):
                    print(f"{Colors.ERROR}ERROR: 'validate_quant_file' '{configs['validate_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not os.path.isfile(configs['validate_meta_file']):
                    print(f"{Colors.ERROR}ERROR: 'validate_meta_file' '{configs['validate_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['validate_quant_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'validate_quant_file' '{configs['validate_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['validate_meta_file'].endswith('.tsv'):
                    print(f"{Colors.ERROR}ERROR: 'validate_meta_file' '{configs['validate_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                configs['validate_quant_table'] = self.read_tsv(configs['validate_quant_file'])
                configs['validate_meta_table'] = self.read_tsv(configs['validate_meta_file'])
        else:
            if configs['apply_model']:
                # required paths:
                #   - model_file
                if not os.path.isfile(configs['model_file']):
                    print(f"{Colors.ERROR}ERROR: 'model_file' '{configs['model_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['model_file'].endswith('.pkl'):
                    print(f"{Colors.ERROR}ERROR: 'model_file' '{configs['model_file']}' does not have a valid file extension. Expected '.pkl'.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

                if not configs['find_features']:
                    configs['model'] = self.read_pkl(configs['model_file'])
                else:
                    print(f"{Colors.ERROR}ERROR: 'find_features' and 'apply_model' enabled while 'train_model' disabled. Generated features may not match loaded model.{Colors.END}", file=sys.stderr, flush=True)
                    sys.exit(1)

        if configs['apply_model']:
            # required paths:
            #   - experimental_quant_file
            if not os.path.isfile(configs['experimental_quant_file']):
                print(f"{Colors.ERROR}ERROR: 'experimental_quant_file' '{configs['experimental_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

            if not configs['experimental_quant_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'experimental_quant_file' '{configs['experimental_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

            configs['experimental_quant_table'] = self.read_tsv(configs['experimental_quant_file'])

        if configs['input_files'] == "reference":
            if not os.path.isfile(configs['reference_quant_file']):
                print(f"{Colors.ERROR}ERROR: 'reference_quant_file' '{configs['reference_quant_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

            if not os.path.isfile(configs['reference_meta_file']):
                print(f"{Colors.ERROR}ERROR: 'reference_meta_file' '{configs['reference_meta_file']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

            if not configs['reference_quant_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'reference_quant_file' '{configs['reference_quant_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

            if not configs['reference_meta_file'].endswith('.tsv'):
                print(f"{Colors.ERROR}ERROR: 'reference_meta_file' '{configs['reference_meta_file']}' does not have a valid file extension. Expected '.tsv'.{Colors.END}", file=sys.stderr, flush=True)
                sys.exit(1)

            configs['reference_quant_table'] = self.read_tsv(configs['reference_quant_file'])
            configs['reference_meta_table'] = self.read_tsv(configs['reference_meta_file'])

        if configs['output_dir'] == "cwd":
            configs['output_dir'] = os.getcwd()
            print(f"{Colors.INFO}INFO: Setting output directory to '{configs['output_dir']}'.{Colors.END}", file=sys.stderr, flush=True)
        if not os.path.isdir(configs['output_dir']):
            print(f"{Colors.ERROR}ERROR: 'output_dir' '{configs['output_dir']}' does not exist.{Colors.END}", file=sys.stderr, flush=True)
            sys.exit(1)


        # check find features settings
        if configs['find_features']:
            if not isinstance(configs['k'], int) or not (0 < configs['k'] <= 50):
                print(f"{Colors.WARNING}WARNING: 'k' must be a positive integer between 1 and 50. Changing 'k' to 50.{Colors.END}", file=sys.stderr, flush=True)
                configs['k'] = 50

            if not isinstance(configs['mc'], (int, float, complex)) or not (0.0 <= configs['mc'] <= 1.0):
                print(f"{Colors.WARNING}WARNING: 'mc' must be between 0.0 and 1.0. Changing 'mc' to 0.5.{Colors.END}", file=sys.stderr, flush=True)
                configs['mc'] = 0.5

            if not isinstance(configs['d'], bool):
                print(f"{Colors.ERROR}WARNING: 'd' must be a boolean. Type of 'd' is: {type(configs['d'])}. Changing 'd' to False.{Colors.END}", file=sys.stderr, flush=True)
                configs['d'] = False
            if configs['d']:
                print(f"{Colors.INFO}INFO: Disjoint filtering enabled.{Colors.END}", file=sys.stderr, flush=True)

            if not isinstance(configs['mi'], bool):
                print(f"{Colors.ERROR}WARNING: 'mi' must be a boolean. Type of 'mi' is: {type(configs['mi'])}. Changing 'mi' to True.{Colors.END}", file=sys.stderr, flush=True)
                configs['mi'] = True
            if not configs['mi']:
                print(f"{Colors.WARNING}WARNING: Mutual information filtering disabled.{Colors.END}", file=sys.stderr, flush=True)

            if not isinstance(configs['mic'], (int, float, complex)) or not (0 <= configs['mic'] <= 1):
                print(f"{Colors.WARNING}WARNING: 'mic' must be between 0.0 and 1.0. Changing 'mic' to 0.7.{Colors.END}", file=sys.stderr, flush=True)
                configs['mic'] = 0.7

        
        # check train model settings
        # TODO
        if configs['train_model']:
            pass


        # check apply model settings
        # TODO
        if configs['apply_model']:
            pass

    def run_paramater_checker(self):
        print("PARSING PARAMETERS", file=sys.stderr, flush=True)
        paramater_parser = self.set_up_parser()
        args = paramater_parser.parse_args()
        configs = self.check_arguments(args)

        print("CHECKING PARAMETERS", file=sys.stderr, flush=True)
        self.check_configurations(configs)

        return configs
    