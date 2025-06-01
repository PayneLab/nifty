class DataTableChecker:
    def __init__(self):
        pass

    def check_meta_file(self, meta_df):
        header = list(meta_df.columns)
        # Meta data file has to have two columns
        if len(header) != 2:
            print(f"Meta data file must have 2 columns, got {len(header)}.")
            return 1
        # Meta data file's first column has to be named sample_id
        if header[0].strip() != "sample_id":
            print(f"First column of meta data file must be named 'sample_id', but found '{header[0]}'.")
            return 2
        if header[1].strip() != "classification_label":
            print(f"Second column of meta data file must be named 'classification_label', but found '{header[1]}'.")
            return 3
        return 0

    def check_samples(self, quant_df, meta_df):
        quant_header = list(quant_df.columns)

        # Same as the previous function, but in the quant_df.
        if quant_header[0].strip() != "sample_id":
            print(f"First column of quant data file must be named 'sample_id', but found '{quant_header[0]}'.")
            return 2
        # Quant df and meta df do not have the same number of rows.
        if len(quant_df) != len(meta_df):
            print(f"Number of rows in quant data file {len(quant_df)} does not match number of meta data file {len(meta_df)}")
            return 3
        return 0

    def check_quant_data(self):
        pass

    def check_duplicate_proteins(self):
        pass

    def check_duplicate_samples(self):
        pass
