import os
import argparse
import random
import sys
import numpy as np


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
        #K cannot be more than 50.
        parser.add_argument("-mi", "--mutual-info", action="store_false", default=True, help="Disable mutual information "
                                                                                             "filtering. Default: "
                                                                                             "Enabled.")
        parser.add_argument("-mc", "--missing-cutoff", type=float, default=0.5, help="Filter out proteins with more "
                                                                                     "than X percent missing values. "
                                                                                     "Default: 0.5.")
        parser.add_argument("-ms", "--min-sample-per-class", type=int, default=15, help="Minimum number of samples "
                                                                                        "required per class. Default:"
                                                                                        " 15.")
        parser.add_argument("-o", "--output", required=False, default=None, help="Path to the output directory. "
                                                                                 "Default: current working directory"
                            )
        parser.add_argument("-s", "--seed", type=int, default=None, help="Random seed for reproducibility. Default: "
                                                                         "Randomly generated."
        )
        return parser

    def check_arguments(self, args):
        if not os.path.isfile(args.quant):
            print(f"ERROR: Quant file '{args.quant}' does not exist.", file=sys.stderr, flush=True)
            sys.exit(1)
        if not os.path.isfile(args.meta):
            print(f"ERROR: Meta file '{args.meta}' does not exist.", file=sys.stderr, flush=True)
            sys.exit(1)
        if not args.quant.endswith('.tsv'):
            print(f"ERROR: Quant file '{args.quant}' does not have a valid extension. Expected 'tsv'.",
                  file=sys.stderr, flush=True)
            sys.exit(1)
        if not args.meta.endswith('.tsv'):
            print(f"ERROR: Meta file '{args.meta}' does not have a valid extension. Expected 'tsv'.",
                  file=sys.stderr, flush=True)
            sys.exit(1)
        if args.k <= 0:
            print("WARNING: K must be a positive integer. Defaulting to 50.", file=sys.stderr, flush=True)
            args.k = 50
        if not (0.0 <= args.missing_cutoff <= 1.0):
            print("WARNING: Missing cutoff must be between 0.0 and 1.0. Defaulting to 0.5.", file=sys.stderr, flush=True)
            args.missing_cutoff = 0.5
        if args.min_sample_per_class < 1:
            print("WARNING: Minimum sample per class must be a positive integer. Defaulting to 15.", file=sys.stderr, flush=True)
            args.min_sample_per_class = 15
        if args.mutual_info is False:
            print("WARNING: No rule filtering enabled.", file=sys.stderr, flush=True)
        if args.seed is not None:
            print(f"INFO: Using fixed seed {args.seed}", file=sys.stderr, flush=True)
        if args.output is None:
            args.output = os.getcwd()
            print(f"INFO: Setting output directory to '{args.output}'.", file=sys.stderr, flush=True)
        if not os.path.isdir(args.output):
            print(f"ERROR: Output directory '{args.output}' does not exist.", file=sys.stderr, flush=True)
            sys.exit(1)
        if args.seed is not None:
            random.seed(args.seed)
            np.random.seed(args.seed)

        return args

    def run_paramater_checker(self):
        paramater_parser = self.set_up_parser()
        args = paramater_parser.parse_args()
        args = self.check_arguments(args)

        return args