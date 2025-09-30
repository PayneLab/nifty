import pandas as pd

from ParameterChecker import ParameterChecker
from DataTableChecker import DataTableChecker
from GenerateRules import GenerateRules
from EvaluateRules import EvaluateRules




def main():
    # Check Parameters
    param_checker = ParameterChecker()
    args = []  # TODO

    # Read in files
    meta_df = pd.read_csv("meta.tsv", sep="\t")
    quant_df = pd.read_csv("quant.tsv", sep="\t")

    # Check Data Tables
    data_table_checker = DataTableChecker()
    filtered_quant_df, meta_df = data_table_checker.run_data_table_checker(args, quant_df, meta_df)

    # Generate Rules
    rule_generator = GenerateRules()
    rules = rule_generator.generate_rule_pairs(filtered_quant_df)

    # Evaluate Rules
    rule_evaluator = EvaluateRules()
    true_scores, summary_df, filtered_df = rule_evaluator.evaluate_buckets_wrapper(args, rules, filtered_quant_df, meta_df)


if __name__ == "__main__":
    main()
