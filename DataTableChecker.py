import numpy as np
import pandas as pd
import sys


class DataTableChecker:
    """
    Error Codes:
    0 = Ok
    1 = Files do not have two columns.
    2 = First column is not 'sample_id' column.
    3 = Second column is not 'classification_label' column.
    4 = Different number of rows in both files.
    5 = Different sample_ids in both files.
    6 = The Quant data file only contains NA.
    7 = The Quant data file contains non-numerical or NA values.
    8 = The Quant data file has duplicate protein names.
    9 = Duplicate sample_id in quant file.
    10 = All proteins were filtered out.
    11 = Not enough samples per class.
    12 = Not enough proteins in quantification data.
    13 = Meta data file contains NA values.
    14 = Duplicate sample_id in meta data file.
    """

    def __init__(self):
        pass

    def check_meta_file(self, meta_df):
        # Check if there are two columns in the meta file.
        header = list(meta_df.columns)
        # The metadata file has to have two columns
        if len(header) != 2:
            return 1
        # The metadata file's first column has to be named sample_id
        if header[0].strip() != "sample_id":
            return 2
        if header[1].strip() != "classification_label":
            return 3
        # Ensure there are no NA values in meta file
        if meta_df.isna().any().any():
            return 13
        return 0
    
    def check_samples(self, quant_df, meta_df):
        # Check if there is the same number of samples in both files.
        # Quant df and meta df do not have the same number of rows.
        if len(quant_df) != len(meta_df):
            return 4

        meta_ids = sorted(list(set(meta_df["sample_id"].astype(str).str.strip())))
        quant_ids = sorted(list(set(quant_df["sample_id"].astype(str).str.strip())))

        # Quant df and meta df do not have the same sample IDs.
        if quant_ids != meta_ids:
            return 5

        return 0
    '''
    def check_samples(self, quant_df, meta_df):
        if "sample_id" not in quant_df.columns:
            return 2
        if "sample_id" not in meta_df.columns:
            return 2
        if len(quant_df) != len(meta_df):
            return 4
        meta_ids = sorted(set(meta_df["sample_id"].astype(str).str.strip()))
        quant_ids = sorted(set(quant_df["sample_id"].astype(str).str.strip()))
        if quant_ids != meta_ids:
            return 5
        return 0'''

    def sort_data(self, quant_df, meta_df):
        quant_df = quant_df.sort_values(by="sample_id").reset_index(drop=True)
        meta_df = meta_df.sort_values(by="sample_id").reset_index(drop=True)
        #Add and error?
        return quant_df, meta_df

    def check_quant_data(self, quant_df):
        ''' Ensures values of quant table are either numeric or NA'''
        quant_header = list(quant_df.columns)
        # Same as the previous function, but in the quant_df.
        if quant_header[0].strip() != "sample_id":
            return 2

        quant_df_values = quant_df.drop(columns=['sample_id'], errors='ignore')

        quant_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        nan_before = quant_df_values.isna().sum().sum()
        coerced = quant_df_values.apply(pd.to_numeric, errors="coerce")
        nan_after = coerced.isna().sum().sum()

        if coerced.isna().all().all():
            return 6

        if nan_after > nan_before:
            return 7

        quant_df.iloc[:, 1:] = coerced
        return 0

    def check_duplicate_proteins(self, quant_df):
        # Check for no duplicate protein names in quant files.
        if any(quant_df.columns.duplicated()):
            return 8
        return 0

    def check_duplicate_samples(self, quant_df, meta_df):
        # Check that there are no duplicate sample IDs in both files.
        if quant_df["sample_id"].duplicated().any():
            return 9
        if meta_df["sample_id"].duplicated().any():
            return 14
        return 0
    
    def filter_proteins(self, quant_df, fraction_na):
        ''' Filter out proteins that have more than fraction_na of their values as NaN'''
        sample_col = quant_df.iloc[:, 0]
        protein_data = quant_df.iloc[:, 1:]
        # Calculate the fraction of NaN values for each protein
        na_fractions = protein_data.isna().mean()

        # Filter proteins based on the specified fraction of NaN values
        filtered_proteins = protein_data.loc[:, na_fractions <= fraction_na]

        # Construct df to return
        filtered_df = pd.concat([sample_col, filtered_proteins], axis=1)

        # Check if filtered_df is empty
        if filtered_df.shape[1] <= 1:  # only sample_id column left
            print("No proteins left after filtering. Please adjust the fraction_na parameter.")
            return 10
        return filtered_df

    def filter_proteins_by_class(self, quant_df, class_labels, fraction_na, proteins_to_keep=[]):
        ''' Filter out proteins that have more than fraction_na of their values as NaN'''
        quant_labels_df = quant_df.join(class_labels, how='inner')
        # print(quant_labels_df.shape)
        quant_labels_df = quant_labels_df.dropna(subset=[class_labels.columns[0]])
        # print(quant_labels_df.shape)

        label_col = class_labels.columns[0]

        classes = class_labels['classification_label'].unique()

        proteins_to_drop = []

        for col in quant_df.columns:
            drop = True

            if col in proteins_to_keep:
                drop = False
            else:
                for cls in classes:
                    class_subset = quant_labels_df[quant_labels_df[label_col] == cls]

                    nan_ratio = class_subset[col].isna().mean()

                    if nan_ratio <= fraction_na:
                        drop = False
                        break

            if drop:
                proteins_to_drop.append(col)

        filtered_df = quant_df.drop(columns=proteins_to_drop)

        # Check if filtered_df is empty
        ## TODO: change fraction_na ourselves if needed, make less stringent until things work
        if filtered_df.shape[1] <= 1:  # only sample_id column left
            return 10

        return filtered_df

    def check_enough_samples(self, meta_df, min_samples):
        ''' Ensures there are enough samples per class '''
        # Create series with counts of each label
        label_counts = meta_df['classification_label'].value_counts()

        for label, count in label_counts.items():
            if count < min_samples:
                print(f"ERROR: Not enough samples for label '{label}': {count} samples found, minimum required is {min_samples}.",
                    file=sys.stderr, flush=True)
                return 11

        return 0

    def check_protein_amount(self, quant_df, min_proteins=2):
        ''' Ensures there are enough proteins in the quantification data '''
        protein_count = quant_df.shape[1] - 1
        if protein_count < min_proteins:
            return 12
        return 0
    
    def set_index(self, df):
        df = df.set_index('sample_id')
        return df

    def run_data_table_checker(self, args, quant_df, meta_df):

        check_meta_file_return = self.check_meta_file(meta_df)
        if check_meta_file_return == 1:
            print(f"ERROR: Meta data file must have 2 columns, got {len(meta_df.columns)}.", file=sys.stderr,
                  flush=True)
            sys.exit(1)
        elif check_meta_file_return == 2:
            print(f"ERROR: First column of meta data file must be named 'sample_id', but found '{meta_df.columns[0]}'.",
                  file=sys.stderr, flush=True)
            sys.exit(1)
        elif check_meta_file_return == 3:
            print(f"ERROR: Second column of meta data file must be named 'classification_label', but found '{meta_df.columns[1]}'.",
                file=sys.stderr, flush=True)
            sys.exit(1)
        elif check_meta_file_return == 13:
            print("ERROR: Meta data file contains NA values.")
            sys.exit(1)

        check_quant_data_return = self.check_quant_data(quant_df)
        if check_quant_data_return == 2:
            print(
                f"ERROR: First column of quant data file must be named 'sample_id', but found '{quant_df.columns[0]}'.",
                file=sys.stderr, flush=True)
            sys.exit(1)
        elif check_quant_data_return == 6:
            print("ERROR: Quant data file is empty or contains only NaN values.", file=sys.stderr, flush=True)
            sys.exit(1)
        elif check_quant_data_return == 7:
            print(f"ERROR: Found non-numeric, non-NA value in quant data.", file=sys.stderr, flush=True)
            sys.exit(1)

        quant_df, meta_df = self.sort_data(quant_df, meta_df)

        check_protein_amount_return = self.check_protein_amount(quant_df)
        if check_protein_amount_return == 12:
            print(f"ERROR: Not enough proteins in quant data file: {quant_df.shape[1] - 1} proteins found, minimum required is 2.")

        check_enough_samples_return = self.check_enough_samples(meta_df, args.min_sample_per_class)
        if check_enough_samples_return == 11:
            sys.exit(1)

        check_samples_return = self.check_samples(quant_df, meta_df)
        if check_samples_return == 4:
            print(f"ERROR: Number of samples in quant data file {len(quant_df)} does not match number of samples meta data file {len(meta_df)}.",
                file=sys.stderr, flush=True)
            sys.exit(1)
        elif check_samples_return == 5:
            print(f"ERROR: Sample IDs in quant data file do not match Sample IDs in meta data file.", file=sys.stderr,
                  flush=True)
            sys.exit(1)

        check_duplicate_samples_return = self.check_duplicate_samples(quant_df, meta_df)
        if check_duplicate_samples_return == 9:
            print("ERROR: Duplicate sample ID(s) in quant data file.", file=sys.stderr, flush=True)
            sys.exit(1)
        elif check_duplicate_samples_return == 14:
            print("ERROR: Duplicate sample ID(s) in meta data file.", file=sys.stderr, flush=True)
            sys.exit(1)

        # Set index to sample_id for filtering   
        quant_df = self.set_index(quant_df)
        meta_df = self.set_index(meta_df)

        print(f"INFO: {len(quant_df.columns)} proteins before filtering.", file=sys.stderr, flush=True)
        filtered_quant_df = self.filter_proteins_by_class(quant_df, meta_df, args.missing_cutoff)
        if filtered_quant_df == 10:
            print("ERROR: No proteins left after filtering. Please adjust the fraction_na parameter.", file=sys.stderr,
                  flush=True)
            sys.exit(1)
        print(f"INFO: {len(filtered_quant_df.columns)} proteins after filtering.", file=sys.stderr, flush=True)

        check_duplicate_proteins_return = self.check_duplicate_proteins(filtered_quant_df)
        if check_duplicate_proteins_return == 8:
            print("ERROR: Duplicate protein names in quant data file.", file=sys.stderr, flush=True)
            sys.exit(1)

        return filtered_quant_df, meta_df
