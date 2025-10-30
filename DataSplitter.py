import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from Colors import Colors

class DataSplitter:

    def __init__(self):
        pass

    def split_table(self, quant_df: pd.DataFrame, meta_df: pd.DataFrame, proportions: tuple, seed):
        """
        Splits a paired reference quant and reference meta table into 2 or 3 separate, paired quant and meta tables.
        No overlap between samples in the splits, all samples in the original tables are found in one of the split tables.
        Stratifies splits based on the classification labels in the meta_df.

        Args:
            quant_df: A pandas DataFrame with sample_id as the index, proteins as columns, quant values as values.
            meta_df: A pandas DataFrame with sample_id as the index (same as quant_df), classification_label as column, class labels as values.
            proportions: A tuple containing the fraction of the original input that each split should contain; should add up to 1.
            seed: A fixed seed to be used in the splitting process (for reproducibility), or None for a random seed.

        Returns:
            A tuple of 2 or 3 pairs of quant and meta DataFrames, each a unique subset of the original paired input.

        Raises:
            SystemExit(1): If the number of samples in the splits does not equal the number of samples in the original 
                           reference (i.e., some samples didn't make it into a subset DataFrame).
        """
        try:
            split_dfs = ()

            if len(proportions) == 2:
                if seed is not None:
                    quant_1, quant_2, meta_1, meta_2 = train_test_split(quant_df, meta_df, test_size=(proportions[1]), stratify=meta_df['classification_label'], random_state=seed)
                else:
                    quant_1, quant_2, meta_1, meta_2= train_test_split(quant_df, meta_df, test_size=(proportions[1]), stratify=meta_df['classification_label'])

                split_dfs += (quant_1.reset_index(), meta_1.reset_index(), quant_2.reset_index(), meta_2.reset_index())
            elif len(proportions) == 3:
                if seed is not None:
                    quant_1, quant_rest, meta_1, meta_rest = train_test_split(quant_df, meta_df, test_size=(round(1-proportions[0], 2)), stratify=meta_df['classification_label'], random_state=seed)

                    val_ratio = proportions[2] / (proportions[1] + proportions[2])
                    quant_2, quant_3, meta_2, meta_3 = train_test_split(quant_rest, meta_rest, test_size=val_ratio, stratify=meta_rest['classification_label'], random_state=seed)
                else:
                    quant_1, quant_rest, meta_1, meta_rest = train_test_split(quant_df, meta_df, test_size=(round(1-proportions[0], 2)), stratify=meta_df['classification_label'])

                    val_ratio = proportions[2] / (proportions[1] + proportions[2])
                    quant_2, quant_3, meta_2, meta_3 = train_test_split(quant_rest, meta_rest, test_size=val_ratio, stratify=meta_rest['classification_label'])

                split_dfs += (quant_1.reset_index(), meta_1.reset_index(), quant_2.reset_index(), meta_2.reset_index(), quant_3.reset_index(), meta_3.reset_index())
            else:
                print(f"{Colors.ERROR}ERROR: Number of proportions provided must be 2 or 3, got {len(proportions)} (please submit issue on GitHub, this is an internal problem).{Colors.END}", 
                    file=sys.stderr, flush=True)
                raise SystemExit(1)

            # sanity check
            total_length = sum(len(df) for df in split_dfs[::2])  # count only quant_dfs
            if total_length != len(quant_df):
                print(f"{Colors.ERROR}ERROR: Number of rows in the split data tables does not match the number of rows in the reference data table (please submit issue on GitHub, this is an internal problem).{Colors.END}", 
                    file=sys.stderr, flush=True)
                raise SystemExit(1)

            # return a tuple of all the dataframes, quant then meta, in order of proportions
            return split_dfs
        except Exception as e:
            print(f"{Colors.ERROR}ERROR splitting data: {e}{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

    def run_data_splitter(self, configs):
        """
        Runner function that calls split_table based on the user input stored in configs and stores the new tables under the appropriate key in the dictionary.

        Args:
            configs: A dictionary storing user configurations and data structures related to the pipeline.

        Returns:
            None

        Raises:
            SystemExit(1): If 'split_for_train' or 'split_for_validate' are ever different (should both be True or both False, never split).
        """
        num_samples_before_split = len(configs['reference_quant_table'])
        print(f"{Colors.INFO}INFO: {num_samples_before_split} reference samples before splitting.", file=sys.stderr, flush=True)

        if configs['split_for_FS'] and not configs['split_for_train'] and not configs['split_for_validate']:
            configs['feature_quant_table'] = configs['reference_quant_table'].reset_index()
            configs['feature_meta_table'] = configs['reference_meta_table'].reset_index()

            num_samples_FS = len(configs['feature_quant_table'])
            print(f"{Colors.INFO}INFO: {num_samples_FS} samples for feature selection after splitting.", file=sys.stderr, flush=True)
        elif not configs['split_for_FS'] and configs['split_for_train'] and not configs['split_for_validate']:  # should never happen
            print(f"{Colors.ERROR}ERROR: 'split_for_train' and 'split_for_validate' must both be True or False (please submit issue on GitHub, this is an internal problem).{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif not configs['split_for_FS'] and not configs['split_for_train'] and configs['split_for_validate']:  # should never happen
            print(f"{Colors.ERROR}ERROR: 'split_for_train' and 'split_for_validate' must both be True or False (please submit issue on GitHub, this is an internal problem).{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif configs['split_for_FS'] and configs['split_for_train'] and not configs['split_for_validate']:  # should never happen
            print(f"{Colors.ERROR}ERROR: 'split_for_train' and 'split_for_validate' must both be True or False (please submit issue on GitHub, this is an internal problem).{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif configs['split_for_FS'] and not configs['split_for_train'] and configs['split_for_validate']:  # should never happen
            print(f"{Colors.ERROR}ERROR: 'split_for_train' and 'split_for_validate' must both be True or False (please submit issue on GitHub, this is an internal problem).{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif not configs['split_for_FS'] and configs['split_for_train'] and configs['split_for_validate']:
            configs['train_quant_table'], configs['train_meta_table'], configs['validate_quant_table'], configs['validate_meta_table'] = self.split_table(quant_df=configs['reference_quant_table'], 
                                                                                                                                                          meta_df=configs['reference_meta_table'], 
                                                                                                                                                          proportions=(0.7, 0.3),  # TODO: finalize these proportions
                                                                                                                                                          seed=configs['seed'])
            
            num_samples_train = len(configs['train_quant_table'])
            num_samples_validate = len(configs['validate_quant_table'])
        else:  # all three are True, checked for all three False in ParameterChecker
            configs['feature_quant_table'], configs['feature_meta_table'], configs['train_quant_table'], configs['train_meta_table'], configs['validate_quant_table'], configs['validate_meta_table'] = self.split_table(quant_df=configs['reference_quant_table'], 
                                                                                                                                                                                                                   meta_df=configs['reference_meta_table'], 
                                                                                                                                                                                                                   proportions=(0.15, 0.65, 0.2),  # TODO: finalize these proportions
                                                                                                                                                                                                                   seed=configs['seed'])
            
            num_samples_FS = len(configs['feature_quant_table'])
            num_samples_train = len(configs['train_quant_table'])
            num_samples_validate = len(configs['validate_quant_table'])
            print(f"{Colors.INFO}INFO: {num_samples_FS} samples for feature selection after splitting.", file=sys.stderr, flush=True)
            print(f"{Colors.INFO}INFO: {num_samples_train} samples for model training after splitting.", file=sys.stderr, flush=True)
            print(f"{Colors.INFO}INFO: {num_samples_validate} samples for model validation after splitting.", file=sys.stderr, flush=True)

