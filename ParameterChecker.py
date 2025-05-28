import os

class ParameterChecker:
    '''A class to check the validity of parameters for TSP rule generation.'''

    def __init__(self, quant_path, meta_path):
        self.quant_path = quant_path
        self.meta_path = meta_path

    def check_file_format(self):
        '''Check if the file format is valid based on whether the extension is "csv" or "tsv".'''
        lower_name = os.path.basename(self.quant_path).lower().split('.')

        if lower_name[1] not in ['csv', 'tsv']:
            raise ValueError(f"Could not determine file type: '{self.file_path}'. Expected 'csv' or 'tsv' file extension.")
        
        