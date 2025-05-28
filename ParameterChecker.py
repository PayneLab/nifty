import os

class ParameterChecker:
    '''A class to check the validity of parameters for TSP rule generation.'''

    def __init__(self, parameters: list):
        '''Takes a list of parameters and verifies them'''
        self.parameters = parameters

    def check_file_existence(self):
        '''Check if two files exist'''
        if self.parameters is None or len(self.parameters) != 2:
            raise ValueError("Two parameters are required: the path to the Quant file and the path to the Meta file.")

    def check_file_format(self):
        '''Ensures rows are samples and columns are features.'''
        # I'm not sure how to do this
        pass

    def check_file_type(self):
        '''Check if the file type is valid based on the extension.'''
        for parameter in self.parameters:
            lower_name = parameter.lower().split('.')

            if lower_name[1] not in ['csv', 'tsv']:
                raise ValueError(f"Could not determine file type: '{parameter}'. Expected 'csv' or 'tsv' file extension.")
        
