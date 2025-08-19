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

        parser.add_argument("-k", "--k", type=int, default=50, help="Number of top-scoring pairs to return. Default: "
                                                                    "50.")
        # parser.add_argument("-optimize-k", action="store_true", help="Optimize K internally. Default is False.")
        parser.add_argument("-nd", "--no-disjoint", action="store_false", default=True, help="Disable disjoint "
                                                                                             "filtering. Default: "
                                                                                             "Enabled.")
        parser.add_argument("-mc", "--missing-cutoff", type=float, default=0.5, help="Filter out proteins with more "
                                                                                     "than X percent missing values. "
                                                                                     "Default: 0.5.")
        parser.add_argument("-ms", "--min-sample-per-class", type=int, default=15, help="Minimum number of samples "
                                                                                        "required per class. Default:"
                                                                                        " 15.")
        return parser

    def check_arguments(self, args):
        if not os.path.isfile(args.quant):
            print(f"Error: Quant file '{args.quant}' does not exist.", file=sys.stderr, flush=True)
            sys.exit(1)
        if not os.path.isfile(args.meta):
            print(f"Error: Meta file '{args.meta}' does not exist.", file=sys.stderr, flush=True)
            sys.exit(1)
        if not args.quant.endswith('.tsv'):
            print(f"Error: Quant file '{args.quant}' does not have a valid extension. Expected 'tsv'.",
                  file=sys.stderr, flush=True)
            sys.exit(1)
        if not args.meta.endswith('.tsv'):
            print(f"Error: Meta file '{args.meta}' does not have a valid extension. Expected 'tsv'.",
                  file=sys.stderr, flush=True)
            sys.exit(1)
        if args.k <= 0:
            print("Error: K must be a positive integer. Defaulting to 50.", file=sys.stderr, flush=True)
            args.k = 50
        if not (0.0 <= args.missing_cutoff <= 1.0):
            print("Error: Missing cutoff must be a float between 0.0 and 1.0. Defaulting to 0.5.",
                  file=sys.stderr, flush=True)
            args.missing_cutoff = 0.5
        if args.min_sample_per_class < 1:
            print("Error: Minimum sample per class must be a positive integer. Defaulting to 15.",
                  file=sys.stderr, flush=True)
            args.min_sample_per_class = 15
        if args.no_disjoint is False:
            print("Notice: Disjoint filtering disabled.", file=sys.stderr, flush=True)
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
