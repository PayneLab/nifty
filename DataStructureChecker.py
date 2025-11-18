import sys

import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.base import BaseEstimator

from Colors import Colors

class DataStructureChecker:
    """
    Error Codes:
    0 = Ok
    1 = Files do not have two columns.
    2 = First column is not 'sample_id' column.
    3 = Second column is not 'classification_label' column.
    4 = Different number of rows in both files.
    5 = Different sample_ids in both files.
    6 = The Quant data table only contains NA.
    7 = The Quant data table contains non-numerical or NA values.
    8 = The Quant data table has duplicate protein names.
    9 = Duplicate sample_id in file.
    10 = All proteins were filtered out.
    11 = Not enough samples per class.
    12 = Not enough proteins in quantification data.
    13 = Meta data table contains NA values.
    14 = Number of classification labels != 2 (currently force binary classification)
    """

    def __init__(self):
        pass

    def check_meta_file(self, meta_df):
        # Check if there are two columns in the meta file.
        header = list(meta_df.columns)
        # The metadata table has to have two columns
        if len(header) != 2:
            return 1
        # The metadata table's first column has to be named sample_id
        if "sample_id" not in header:
            return 2
        if "classification_label" not in header:
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

    def sort_data(self, quant_df, meta_df):
        quant_df = quant_df.sort_values(by="sample_id").reset_index(drop=True)
        meta_df = meta_df.sort_values(by="sample_id").reset_index(drop=True)
        return quant_df, meta_df

    def check_quant_data(self, quant_df):
        ''' Ensures values of quant table are either numeric or NA'''
        quant_header = list(quant_df.columns)
        # Same as the previous function, but in the quant_df.
        if "sample_id" not in quant_header:
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

        return 0

    def check_duplicate_proteins(self, quant_df):
        # Check for no duplicate protein names in quant files.
        if any(quant_df.columns.duplicated()):
            return 8
        return 0

    def check_duplicate_samples(self, df):
        # Check that there are no duplicate sample IDs in both files.
        if df["sample_id"].duplicated().any():
            return 9
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
        ''' Filter out proteins that have more than fraction_na of their values as NaN in both classes.'''
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
        ## TODO: change fraction_na ourselves if needed, make less stringent until things work?
        if filtered_df.shape[1] <= 1:  # only sample_id column left
            return 10

        return filtered_df

    def check_enough_samples(self, meta_df, min_samples):
        ''' Ensures there are enough samples per class '''
        # Create series with counts of each label
        label_counts = meta_df['classification_label'].value_counts()

        if len(label_counts) != 2:
            print(f"{Colors.ERROR}ERROR: Wrong number of classification labels; must have 2, found {len(label_counts)}.{Colors.END}",
                    file=sys.stderr, flush=True)
            return 14

        for label, count in label_counts.items():
            if count < min_samples:
                print(f"{Colors.ERROR}ERROR: Not enough samples for label '{label}': {count} samples found, minimum required is {min_samples}.{Colors.END}",
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

    def coerce_to_numeric(self, df):
        df = df.apply(pd.to_numeric, errors='coerce')
        return df
    
    def balance_classes(self, quant_df, meta_df):
        # find the class with the least number of samples, that becomes the number of samples per class
        label_counts = meta_df['classification_label'].value_counts()
        min_samples = -1

        for label, count in label_counts.items():
            if min_samples == -1:
                min_samples = count
            elif count < min_samples:
                min_samples = count

        # randomly select that number of samples for each class
        filtered_samples = []
        for label in label_counts.keys():
            samples = meta_df[meta_df['classification_label'] == label]['sample_id'].tolist()
            samples_to_keep = random.sample(samples, min_samples)

            filtered_samples += samples_to_keep

        # filter the dfs to those samples
        filtered_quant_df = quant_df[quant_df['sample_id'].isin(filtered_samples)]
        filtered_meta_df = meta_df[meta_df['sample_id'].isin(filtered_samples)]

        print(f"{Colors.INFO}INFO: {len(filtered_meta_df)} samples for model validation after balancing classes.{Colors.END}", file=sys.stderr, flush=True)

        return filtered_quant_df, filtered_meta_df

    def check_paired_quant_and_meta_tables(self, configs, quant_df, meta_df, min_samples, balance):
        print(" - CHECKING META DATA TABLE", file=sys.stderr, flush=True)
        check_meta_file_return = self.check_meta_file(meta_df)
        if check_meta_file_return == 1:
            print(f"{Colors.ERROR}ERROR: Meta data table must have 2 columns, got {len(meta_df.columns)}.{Colors.END}", file=sys.stderr,
                  flush=True)
            raise SystemExit(1)
        elif check_meta_file_return == 2:
            print(f"{Colors.ERROR}ERROR: Meta data table must have 'sample_id' column.{Colors.END}",
                  file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif check_meta_file_return == 3:
            print(f"{Colors.ERROR}ERROR: Meta data table must have 'classification_label' column.{Colors.END}",
                file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif check_meta_file_return == 13:
            print(f"{Colors.ERROR}ERROR: Meta data table contains NA values.{Colors.END}")
            raise SystemExit(1)

        print(" - CHECKING QUANT DATA TABLE", file=sys.stderr, flush=True)
        check_quant_data_return = self.check_quant_data(quant_df)
        if check_quant_data_return == 2:
            print(
                f"{Colors.ERROR}ERROR: Quant data table must have 'sample_id' column.{Colors.END}",
                file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif check_quant_data_return == 6:
            print(f"{Colors.ERROR}ERROR: Quant data table is empty or contains only NaN values.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif check_quant_data_return == 7:
            print(f"{Colors.ERROR}ERROR: Found non-numeric, non-NA value in quant data.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        print(" - CHECKING PROTEINS", file=sys.stderr, flush=True)
        check_protein_amount_return = self.check_protein_amount(quant_df)
        if check_protein_amount_return == 12:
            print(f"{Colors.ERROR}ERROR: Not enough proteins in quant data table: {quant_df.shape[1] - 1} proteins found, minimum required is 2.{Colors.END}")
            raise SystemExit(1)

        check_duplicate_proteins_return = self.check_duplicate_proteins(quant_df)
        if check_duplicate_proteins_return == 8:
            print(f"{Colors.ERROR}ERROR: Duplicate protein names in quant data table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        print(" - CHECKING SAMPLES", file=sys.stderr, flush=True)
        check_enough_samples_return = self.check_enough_samples(meta_df, min_samples)
        if check_enough_samples_return == 11:
            # not enough samples per class
            raise SystemExit(1)
        if check_enough_samples_return == 14:
            # more than two classes
            raise SystemExit(1)
        
        # balance the classes if true
        if balance:
            quant_df, meta_df = self.balance_classes(quant_df, meta_df)

        quant_df, meta_df = self.sort_data(quant_df, meta_df)

        check_samples_return = self.check_samples(quant_df, meta_df)
        if check_samples_return == 4:
            print(f"{Colors.ERROR}ERROR: Number of samples in quant data table {len(quant_df)} does not match number of samples meta data table {len(meta_df)}.{Colors.END}",
                file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif check_samples_return == 5:
            print(f"{Colors.ERROR}ERROR: Sample IDs in quant data table do not match Sample IDs in meta data table.{Colors.END}", file=sys.stderr,
                  flush=True)
            raise SystemExit(1)

        check_duplicate_samples_return_quant = self.check_duplicate_samples(quant_df)
        if check_duplicate_samples_return_quant == 9:
            print(f"{Colors.ERROR}ERROR: Duplicate sample ID(s) in quant data table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        check_duplicate_samples_return_meta = self.check_duplicate_samples(meta_df)
        if check_duplicate_samples_return_meta == 9:
            print(f"{Colors.ERROR}ERROR: Duplicate sample ID(s) in meta data table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        
        # Set index to sample_id for filtering   
        quant_df = self.set_index(quant_df)
        meta_df = self.set_index(meta_df)

        # Coerce quant df to numeric before filtering
        quant_df = self.coerce_to_numeric(quant_df)

        return quant_df, meta_df
    
    def check_quant_table(self, configs, quant_df):
        print(" - CHECKING QUANT DATA TABLE", file=sys.stderr, flush=True)
        check_quant_data_return = self.check_quant_data(quant_df)
        if check_quant_data_return == 2:
            print(
                f"{Colors.ERROR}ERROR: Quant data table must have 'sample_id' column.{Colors.END}",
                file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif check_quant_data_return == 6:
            print(f"{Colors.ERROR}ERROR: Quant data table is empty or contains only NaN values.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)
        elif check_quant_data_return == 7:
            print(f"{Colors.ERROR}ERROR: Found non-numeric, non-NA value in quant data.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        print(" - CHECKING PROTEINS", file=sys.stderr, flush=True)
        check_protein_amount_return = self.check_protein_amount(quant_df)
        if check_protein_amount_return == 12:
            print(f"{Colors.ERROR}ERROR: Not enough proteins in quant data table: {quant_df.shape[1] - 1} proteins found, minimum required is 2.{Colors.END}")
            raise SystemExit(1)

        check_duplicate_proteins_return = self.check_duplicate_proteins(quant_df)
        if check_duplicate_proteins_return == 8:
            print(f"{Colors.ERROR}ERROR: Duplicate protein names in quant data table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        print(" - CHECKING SAMPLES", file=sys.stderr, flush=True)
        check_duplicate_samples_return_quant = self.check_duplicate_samples(quant_df)
        if check_duplicate_samples_return_quant == 9:
            print(f"{Colors.ERROR}ERROR: Duplicate sample ID(s) in quant data table.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        quant_df = self.set_index(quant_df)
        quant_df = self.coerce_to_numeric(quant_df)

        return quant_df
    
    def filter_quant_table(self, configs, quant_df, meta_df):
        print("FILTERING PROTEINS", file=sys.stderr, flush=True)
        print(f"{Colors.INFO}INFO: {len(quant_df.columns)} proteins before filtering.{Colors.END}", file=sys.stderr, flush=True)
        filtered_quant_df = self.filter_proteins_by_class(quant_df, meta_df, configs['missingness_cutoff'])
        if isinstance(filtered_quant_df, int) and filtered_quant_df == 10:
            print(f"{Colors.ERROR}ERROR: No proteins left after filtering. Please adjust the 'missingness_cutoff' parameter.{Colors.END}", file=sys.stderr,
                  flush=True)
            raise SystemExit(1)
        print(f"{Colors.INFO}INFO: {len(filtered_quant_df.columns)} proteins after filtering.{Colors.END}", file=sys.stderr, flush=True)

        return filtered_quant_df
    
    def check_feature_table(self, feature_df):
        if "Protein1" not in feature_df.columns or "Protein2" not in feature_df.columns:
            print(f"{Colors.ERROR}ERROR: Feature table must have columns (\"Protein1\", \"Protein2\").{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        if len(feature_df) < 1:
            print(f"{Colors.ERROR}ERROR: Feature table must have at least 1 rule, found {len(feature_df)}.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

    def check_model(self, configs, model, feature_df):
        # Check that the model is a scikit-learn model
        if not isinstance(model, BaseEstimator):
            print(f"{Colors.ERROR}ERROR: Loaded model is not a valid scikit-learn model.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        # check that the scikit-learn version for the model matches the one that loaded the model
        sklearn_version = sklearn.__version__

        if hasattr(model, "_sklearn_version"):
            model_version = model._sklearn_version

            if model_version != sklearn_version:
                print(f"{Colors.ERROR}ERROR: Scikit-learn version used to train the model is not the same as the scikit-learn version being used to load the model.{Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)
        else:
            print(f"{Colors.ERROR}ERROR: Loaded model does not have '_sklearn_version' attribute. Cannot check that the scikit-learn version used to train the model is the same as the scikit-learn version being used to load the model. The following code should be executed after fitting the model: 'model._sklearn_version = sklearn.__version__'.{Colors.END}", file=sys.stderr, flush=True)
            raise SystemExit(1)

        # check that the model can predict
        if not hasattr(model, "predict"):
            print(f"{Colors.ERROR}ERROR: Loaded model does not have 'predict()' method.{Colors.END}", file=sys.stderr, flush=True)
            SystemExit(1)

        # check that the model can predict_proba
        if configs['prediction_format'] == "probabilities":
            if not hasattr(model, "predict_proba"):
                print(f"{Colors.ERROR}ERROR: Loaded model does not have 'predict_proba()' method. Some models need to be generated with 'probability=True' to have this method.{Colors.END}", file=sys.stderr, flush=True)
                SystemExit(1)

        # check that the model has features
        n_features_experimental = len(feature_df)
        if hasattr(model, 'n_features_in_'):
            # check that the number of features in the model match the number of features in the feature_df
            if model.n_features_in_ != n_features_experimental:
                print(f"{Colors.ERROR}ERROR: Loaded model does not have the same number of features as the feature table. {Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)
        else:
            print(f"{Colors.ERROR}ERROR: Loaded model does not have 'n_features_in_' attribute. Cannot check for feature alignment between the model and the feature table.", file=sys.stderr, flush=True)
            raise SystemExit(1)

        # check that the model has feature names
        features_experimental = list(zip(feature_df['Protein1'].tolist(), feature_df['Protein2'].tolist()))
        features_experimental = [">".join(pair) for pair in features_experimental]
        if hasattr(model, "feature_names_in_"):
            # check that the feature names match those in the feature_df
            if sorted(model.feature_names_in_) != sorted(features_experimental):
                print(f"{Colors.ERROR}ERROR: Loaded model does not have the same feature names as the feature table. {Colors.END}", file=sys.stderr, flush=True)
                raise SystemExit(1)
        else:
            print(f"{Colors.ERROR}ERROR: Loaded model does not have 'feature_names_in_' attribute. Cannot check for feature alignment between the model and the feature table. Model needs to be fitted on a pandas.DataFrame.", file=sys.stderr, flush=True)
            raise SystemExit(1)
        
        # check that the model has classes
        if not hasattr(model, 'classes_'):
            print(f"{Colors.ERROR}ERROR: Loaded model does not have 'classes_' attribute.", file=sys.stderr, flush=True)
            raise SystemExit(1)
