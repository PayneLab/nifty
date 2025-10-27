import sys
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from Colors import Colors

class DataSplitter:

    def __init__(self):
        pass

    def split_table(self, quant_df: pd.DataFrame, meta_df: pd.DataFrame, proportions: tuple, seed):
        try:
            split_dfs = ()

            if len(proportions) == 2:
                if seed is not None:
                    quant_1, quant_2, meta_1, meta_2 = train_test_split(quant_df, meta_df, test_size=(proportions[1]), stratify=meta_df['classification_label'], random_state=seed)
                else:
                    quant_1, quant_2, meta_1, meta_2= train_test_split(quant_df, meta_df, test_size=(proportions[1]), stratify=meta_df['classification_label'])

                split_dfs += (quant_1, meta_1, quant_2, meta_2)
            elif len(proportions) == 3:
                if seed is not None:
                    quant_1, quant_rest, meta_1, meta_rest = train_test_split(quant_df, meta_df, test_size=(round(1-proportions[0], 2)), stratify=meta_df['classification_label'], random_state=seed)

                    val_ratio = proportions[2] / (proportions[1] + proportions[2])
                    quant_2, quant_3, meta_2, meta_3 = train_test_split(quant_rest, meta_rest, test_size=val_ratio, stratify=meta_rest['classification_label'], random_state=seed)
                else:
                    quant_1, quant_rest, meta_1, meta_rest = train_test_split(quant_df, meta_df, test_size=(round(1-proportions[0], 2)), stratify=meta_df['classification_label'])

                    val_ratio = proportions[2] / (proportions[1] + proportions[2])
                    quant_2, quant_3, meta_2, meta_3 = train_test_split(quant_rest, meta_rest, test_size=val_ratio, stratify=meta_rest['classification_label'])

                split_dfs += (quant_1, meta_1, quant_2, meta_2, quant_3, meta_3)
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
        if configs['split_for_FS'] and not configs['split_for_train'] and not configs['split_for_validate']:
            configs['feature_quant_table'] = configs['reference_quant_table']
            configs['feature_meta_table'] = configs['reference_meta_table']
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
        else:  # all three are True
            configs['feature_quant_df'], configs['feature_meta_df'], configs['train_quant_table'], configs['train_meta_table'], configs['validate_quant_table'], configs['validate_meta_table'] = self.split_table(quant_df=configs['reference_quant_table'], 
                                                                                                                                                                                                                   meta_df=configs['reference_meta_table'], 
                                                                                                                                                                                                                   proportions=(0.15, 0.65, 0.2),  # TODO: finalize these proportions
                                                                                                                                                                                                                   seed=configs['seed'])

