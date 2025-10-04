import pandas as pd

from ParameterChecker import ParameterChecker
from DataTableChecker import DataTableChecker
from GenerateRules import GenerateRules
from EvaluateRules import EvaluateRules




def main():
    # Check Parameters
    param_checker = ParameterChecker()
    parser = param_checker.set_up_parser()
    args = param_checker.check_arguments(parser.parse_args())

    # Set random seed if not given
    if args.seed is not None:
        import random, numpy as np
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Read in files
    meta_df = pd.read_csv(args.quant, sep="\t")
    quant_df = pd.read_csv(args.meta, sep="\t")

    # Check Data Tables
    data_table_checker = DataTableChecker()
    filtered_quant_df, meta_df = data_table_checker.run_data_table_checker(args, quant_df, meta_df)

    # Generate Rules
    rule_generator = GenerateRules()
    rules = rule_generator.generate_rule_pairs(filtered_quant_df)

    # Evaluate Rules
    rule_evaluator = EvaluateRules()
    rule_evaluator.seed = args.seed

    true_scores, summary_df, filtered_df = rule_evaluator.evaluate_buckets_wrapper(
    pairs=rules,
    quant_df=filtered_quant_df,
    meta_df=meta_df,
    k_value=args.k,
    output_file_path=args.output,
    disjoint=args.no_disjoint,   # <- maps your -nd flag
)

if __name__ == "__main__":
    main()
