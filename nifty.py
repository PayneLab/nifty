import pandas as pd

from ParameterChecker import ParameterChecker
from DataTableChecker import DataTableChecker
from GenerateRules import GenerateRules
from EvaluateRules import EvaluateRules


def main():
    # Check Parameters
    print("---CHECKING PARAMETERS---", file=sys.stderr, flush=True)
    param_checker = ParameterChecker()
    args = param_checker.run_paramater_checker()

    # Read in files
    print("", file=sys.stderr, flush=True)
    print("---READING IN FILES---", file=sys.stderr, flush=True)
    meta_df = pd.read_csv(args.quant, sep="\t")
    quant_df = pd.read_csv(args.meta, sep="\t")

    # Check Data Tables
    print("", file=sys.stderr, flush=True)
    print("---CHECKING DATA TABLES---", file=sys.stderr, flush=True)
    data_table_checker = DataTableChecker()
    filtered_quant_df, meta_df = data_table_checker.run_data_table_checker(args=args, 
                                                                           quant_df=quant_df, 
                                                                           meta_df=meta_df)

    # Generate Rules
    print("", file=sys.stderr, flush=True)
    print("---GENERATING RULES---", file=sys.stderr, flush=True)
    rule_generator = GenerateRules()
    rules = rule_generator.generate_rule_pairs(filtered_quant_df)

    # Evaluate Rules
    print("", file=sys.stderr, flush=True)
    print("---EVALUATING RULES---", file=sys.stderr, flush=True)
    rule_evaluator = EvaluateRules(args.seed)
    true_scores, summary_df, filtered_df = rule_evaluator.run_rule_evaluator(args=args, 
                                                                             pairs=rules,
                                                                             quant_df=filtered_quant_df,
                                                                             meta_df=meta_df)

if __name__ == "__main__":
    main()
