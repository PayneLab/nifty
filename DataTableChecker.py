import numpy as np
import pandas as pd

class DataTableChecker:
    """
    Error Codes:
    0 = Ok
    1 = Files do not have two columns.
    2 = First column is not 'sample_id' column.
    3 = Second column is not 'classification_label' column.
    4 = Different number of rows in both files.
    5 = Different sample_ids in both files.
    6 = Different sorting order in both files.
    7 = The Quant data file only contains NA.
    8 = The Quant data file contains non-numerical or NA values.
    9 = The Quant data file has duplicate protein names.
    10 = Duplicate sample_id in file.
    11 = All proteins were filtered out.
    12 = Not enough samples per class.

    """

    def __init__(self):
        pass

    # TO-DO: Return error values across the class.

    def check_meta_file(self, meta_df):
        # Check if there are two columns in the meta file.
        header = list(meta_df.columns)
        # The metadata file has to have two columns
        if len(header) != 2:
            print(f"Meta data file must have 2 columns, got {len(header)}.")
            return 1
        # The metadata file's first column has to be named sample_id
        if header[0].strip() != "sample_id":
            print(f"First column of meta data file must be named 'sample_id', but found '{header[0]}'.")
            return 2
        if header[1].strip() != "classification_label":
            print(f"Second column of meta data file must be named 'classification_label', but found '{header[1]}'.")
            return 3
        return 0

    def check_samples(self, quant_df, meta_df):
        # Check if there is the same number of samples in both files.
        quant_header = list(quant_df.columns)
        # Same as the previous function, but in the quant_df.
        if quant_header[0].strip() != "sample_id":
            print(f"First column of quant data file must be named 'sample_id', but found '{quant_header[0]}'.")
            return 2
        # Quant df and meta df do not have the same number of rows.
        if len(quant_df) != len(meta_df):
            print(f"Number of rows in quant data file {len(quant_df)} does not match number of meta data file {len(meta_df)}")
            return 4

        # TO DO
        # Delete the set.
        # Put in ia list and make sure they are the same.

        meta_ids = set(meta_df["sample_id"].astype(str).str.strip())
        quant_ids = set(quant_df["sample_id"].astype(str).str.strip())

        # Quant df and meta df do not have the same sample IDs.
        if quant_ids != meta_ids:
            print(f"IDs in quant data file does not match IDs in meta data file.")
            return 5
        # Sort them
        # MOVE!
        # Take them out of the set and sort them into a list.
        sorted_meta_df = meta_df.sort_values("sample_id").reset_index(drop=True)
        sorted_quant_df = quant_df.sort_values("sample_id").reset_index(drop=True)

        if not sorted_meta_df["sample_id"].equals(sorted_quant_df["sample_id"]):
            print(f"IDs sorting in meta data file does not match the sorting in quant data file.")
            return 6

        return 0

    def check_quant_data(self, quant_df):
        ''' Ensures values of quant table are either numeric or NA'''
        #replace empty cells or empty strings with NaN
        quant_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

        #ensures df is not empty
        quant_df_values = quant_df.drop(columns=['sample_id'], errors='ignore')
        if quant_df_values.isna().all().all():
            print("Quant data file is empty or contains only NaN values.")
            return 7
        
        #check for non-numeric values and NaN values
        def is_valid(x):
            return pd.isna(x) or isinstance(x, (int, float, np.integer, np.floating))

        #check each value in df
        data = quant_df_values.iloc[:,1:]
        invalid_mask = ~data.applymap(is_valid)

        if invalid_mask.any().any():
            invalid_value = quant_df_values[invalid_mask].stack().iloc[0]
            print(f"Error: Found non-numeric, non-NA value in quant data: '{invalid_value}'")
            return 8
        return 0

    def check_duplicate_proteins(self, quant_df):
        # Check for no duplicate protein names in quant files.
        if any(quant_df.columns.duplicated()):
            print("Duplicate protein names in quant data file.")
            return 9
        return 0

    def check_duplicate_samples(self, quant_df, meta_df):
        # Check that there are no duplicate sample IDs in both files.
        if quant_df["sample_id"].duplicated().any():
            print("Duplicate samples ID in quant data file.")
            return 10
        if meta_df["sample_id"].duplicated().any():
            print("Duplicate samples ID in meta data file.")
            return 10
        return 0

    def filter_proteins(self, quant_df, fraction_na=0.5):
        ''' Filter out proteins that have more than fraction_na of their values as NaN'''
        sample_col = quant_df.iloc[:, 0]
        protein_data = quant_df.iloc[:, 1:]
        # Calculate the fraction of NaN values for each protein
        na_fractions = protein_data.isna().mean()

        # Filter proteins based on the specified fraction of NaN values
        filtered_proteins = protein_data.loc[:, na_fractions <= fraction_na]

        # Construct df to return
        filtered_df = pd.concat([sample_col, filtered_proteins], axis=1)

        #check if filtered_df is empty
        if filtered_df.shape[1] <= 1:  # only sample_id column left
            print("No proteins left after filtering. Please adjust the fraction_na parameter.")
            return 11
        return filtered_df

    def check_enough_samples(self, meta_df, min_samples = 15):
        ''' Ensures there are enough samples per class '''

        #Create series with counts of each label
        label_counts = meta_df['classification_label'].value_counts()

        for label, count in label_counts.items():
            if count < min_samples:
                print(f"Not enough samples for label '{label}': {count} samples found, minimum required is {min_samples}.")
                return 12
            
        return 0
