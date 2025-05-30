class DataTableChecker:
    def __init__(self, quant_file: str, meta_file: str):
        self.quant_file = quant_file
        self.meta_file = meta_file

    def check_meta_file(self):
        # QUESTION: What method should I use to read the metadata lines?
        header = lines[0]
        if len(header) != 2:
            raise ValueError(f"Meta data file must have 2 columns, got {len(header)}.")
        if header[0].strip() != "sample_id":
            raise ValueError(f"First column of meta data file must be named 'sample_id', but found '{header[0]}'.")
        if header[1].strip() != "classification_label":
            raise ValueError(f"Second column of meta data file must be named 'classification_label', but found '{header[1]}'.")
        pass
    
    def check_samples(self):
        # QUESTION: What method should I use to read the metadata lines?
        header = lines[0]
        if header[0].strip() != "sample_id":
            raise ValueError(f"First column of quant data file must be named 'sample_id', but found '{header[0]}'.")
        pass

    def check_quant_data(self):
        pass

    def check_duplicate_proteins(self):
        pass

    def check_duplicate_samples(self):
        pass