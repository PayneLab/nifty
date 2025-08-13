import os
import argparse


class ParameterChecker:
    '''A class to check the validity of parameters for TSP rule generation.'''

    def __init__(self, args):
        '''Takes a list of parameters and verifies them'''
        self.args = self.parse_arguments()
        self.quant_file = self.args.quant
        self.meta_file = self.args.meta
        self.k = self.args.k
        self.disjoint = self.args.disjoint

    def parse_arguments(self):
        parser = argparse.ArgumentParser(description='Generate list of top scoring pairs without data imputation.')

        # Required files.
        parser.add_argument("--quant", required=True, help="Path to the Quant file (tsv).")
        parser.add_argument("--meta", required=True, help="Path to the Meta file (tsv).")

        parser.add_argument("--k", type=int, default=50, help="Number of pairs to return. Default is 50.")
        #parser.add_argument("--optimize-k", action="store_true", help="Optimize K internally. Default is False.")
        parser.add_argument("--disjoint", action="store_true", default=True, help="Whether to return disjoint pairs.")

        args = parser.parse_args()

        for file in [args.quant, args.meta]:
            if not os.path.isfile(file):
                raise FileNotFoundError(f"File not found: '{file}'.")
            ext = file.split('.')[-1]
            if ext not in ['csv', 'tsv']:
                raise ValueError(f"Could not determine file type: '{file}'. Expected 'csv' or 'tsv' file extension.")
        if args.k <= 0:
            raise ValueError("K must be a positive integer.")
        return args


if __name__ == "__main__":
    checker = ParameterChecker()
    print("All parameters are valid.")

    # def check_file_existence(self):
    #     '''Check if two files exist'''
    #     if self.parameters is None or len(self.parameters) != 2:
    #         raise ValueError("Two parameters are required: the path to the Quant file and the path to the Meta file.")
    #
    # def check_file_type(self):
    #     '''Check if the file type is valid based on the extension.'''
    #     for parameter in self.parameters:
    #         lower_name = parameter.lower().split('.')
    #
    #         if lower_name[1] not in ['csv', 'tsv']:
    #             raise ValueError(f"Could not determine file type: '{parameter}'. Expected 'csv' or 'tsv' file extension.")
