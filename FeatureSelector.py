import sys

from GenerateRules import GenerateRules
from EvaluateRules import EvaluateRules

class FeatureSelector:

    def __init__(self):
        pass

    def find_features(self, configs):
        # Generate Rules
        print("GENERATING RULES", file=sys.stderr, flush=True)
        rule_generator = GenerateRules()
        rules = rule_generator.generate_rule_pairs(configs['filtered_feature_quant_table'])

        # Evaluate Rules
        print("EVALUATING RULES", file=sys.stderr, flush=True)
        rule_evaluator = EvaluateRules(configs['seed'])
        true_scores, all_evaluated_rules, top_k_rules = rule_evaluator.run_rule_evaluator(configs=configs,
                                                                                          pairs=rules,
                                                                                          quant_df=configs['filtered_feature_quant_table'],
                                                                                          meta_df=configs['feature_meta_table'])
        
        return rules, true_scores, all_evaluated_rules, top_k_rules
