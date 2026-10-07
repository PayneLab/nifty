import sys
import os
import psutil
import numpy as np
import math

from GenerateRules import GenerateRules
from EvaluateRules import EvaluateRules
from Colors import Colors

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

        # check for number of cells final rule table would have
        # max_mem = (psutil.virtual_memory().available / (1024 ** 3)) - 8  # TODO: add user parameter that allws them to set the max RAM allotment in GB and store in configs
        max_mem = 100 - 8  # TODO: add user parameter that allws them to set the max RAM allotment in GB and store in configs
        mem_avail = max_mem - (psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3))
        print(f"{Colors.INFO}INFO: Memory available: {mem_avail:.2f} GB{Colors.END}", file=sys.stderr, flush=True)

        # max_cols = math.ceil(mem_avail / ((len(configs['feature_meta_table']) * np.dtype(np.int8).itemsize) / (1024 ** 3)))
        max_cols = 1000000
        print(f"{Colors.INFO}INFO: Maximum number of rules to evaluate: {max_cols}{Colors.END}", file=sys.stderr, flush=True)

        # if number of cells exceeds memory limit, split rules into subsets
        if len(rules) > max_cols:
            num_rules = len(rules)
            print(num_rules)
            filtered_rules = rules
            while num_rules > max_cols:
                # segment rules list
                num_segments = math.ceil(num_rules / max_cols)
                segment_size = math.ceil(num_rules / num_segments)

                print(f"Evaluating rules in subsets of size {segment_size}.", file=sys.stderr, flush=True)

                new_rules_subset = []
                segment_num = 1

                # loop through segments and get top k rules
                for i in range(0, num_rules, segment_size):
                    print(f"Evaluating subset #{segment_num}.", file=sys.stderr, flush=True)
                    
                    segment = filtered_rules[i:i + segment_size]

                    top_k_rules = rule_evaluator.run_rule_evaluator(configs=configs,
                                                                    pairs=segment,
                                                                    quant_df=configs['filtered_feature_quant_table'],
                                                                    meta_df=configs['feature_meta_table'], 
                                                                    save=False)
                    #combine top k rules into list
                    new_rules_subset.extend(top_k_rules["Gene_Pair"].tolist())

                    segment_num += 1

                filtered_rules = new_rules_subset
                num_rules = len(filtered_rules)

            # run filter one final time on final top k rules list for real top k rules
            true_scores, all_evaluated_rules, top_k_rules = rule_evaluator.run_rule_evaluator(configs=configs,
                                                                                                          pairs=filtered_rules,
                                                                                                          quant_df=configs['filtered_feature_quant_table'],
                                                                                                          meta_df=configs['feature_meta_table'])
        else:
            true_scores, all_evaluated_rules, top_k_rules = rule_evaluator.run_rule_evaluator(configs=configs,
                                                                                              pairs=rules,
                                                                                              quant_df=configs['filtered_feature_quant_table'],
                                                                                              meta_df=configs['feature_meta_table'])
        
        return rules, true_scores, all_evaluated_rules, top_k_rules
