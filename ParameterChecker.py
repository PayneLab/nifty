import os
import argparse
import sys


class ParameterChecker:
    '''A class to check the validity of parameters for TSP rule generation.'''

    def __init__(self):
        pass

    def set_up_parser(self):
        parser = argparse.ArgumentParser(description='Generate list of top scoring pairs without data imputation.')

        # Required files.
        parser.add_argument("-q", "--quant", required=True, help="Path to the Quant file (tsv).")
        parser.add_argument("-m", "--meta", required=True, help="Path to the Meta file (tsv).")

        parser.add_argument("-k", type=int, default=50, help="Number of top-scoring pairs to return. Default: 50.")
        # parser.add_argument("-optimize-k", action="store_true", help="Optimize K internally. Default is False.")
        parser.add_argument("-nd", "--no-disjoint", action="store_false", default=True, help="Disable disjoint "
                                                                                              "filtering. Default: "
                                                                                              "Disjoint enabled).")
        return parser

    def check_arguments(self, args):
        if args.k <= 0:
            print("Error: K must be a positive integer. Defaulting to 50.", file=sys.stderr, flush=True)

        return args


if __name__ == "__main__":
    checker = ParameterChecker()
    print("All parameters are valid.")

    # parser = handler.set_up_parser()
    # args = parser.parse_args()
    # handler.check_arguments(args)

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
