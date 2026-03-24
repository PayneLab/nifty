import sys
import time

from ParameterChecker import ParameterChecker
from DataStructureChecker import DataStructureChecker
from DataSplitter import DataSplitter
from FeatureSelector import FeatureSelector
from ModelGenerator import ModelGenerator
from ExperimentalClassifier import ExperimentalClassifier
from Colors import Colors


MIN_SAMPLES_FEATURES = 15  
MIN_SAMPLES_TRAIN = 35  
MIN_SAMPLES_VALIDATE = 15


def main():
    start_time = time.time()

    # CHECK PARAMS
    print(f"{Colors.HEADER}---CHECKING PARAMETERS AND READING IN FILES---{Colors.END}", file=sys.stderr, flush=True)
    param_checker = ParameterChecker()
    configs = param_checker.run_paramater_checker()

    check_param_end = time.time()
    check_param_time = (check_param_end - start_time) / 60
    print(f"{Colors.INFO}INFO: Parameters checked and files read in {check_param_time:.2f} minutes.", file=sys.stderr, flush=True)


    # SPLIT REFERENCE
    if configs['input_files'] == "reference":
        print("", file=sys.stderr, flush=True)
        print(f"{Colors.HEADER}---SPLITTING REFERENCE DATA---{Colors.END}", file=sys.stderr, flush=True)
        
        print("CHECKING REFERENCE TABLES", file=sys.stderr, flush=True)
        data_structure_checker = DataStructureChecker()
        configs['reference_quant_table'], configs['reference_meta_table'] = data_structure_checker.check_paired_quant_and_meta_tables(configs=configs, 
                                                                                                                                  quant_df=configs['reference_quant_table'], 
                                                                                                                                  meta_df=configs['reference_meta_table'], 
                                                                                                                                  min_samples=MIN_SAMPLES_VALIDATE, 
                                                                                                                                  balance=False)  # TODO: is this the min we want here?

        print("SPLITTING REFERENCE TABLES", file=sys.stderr, flush=True)
        data_table_splitter = DataSplitter()
        data_table_splitter.run_data_splitter(configs=configs)

        split_reference_end = time.time()
        split_reference_time = (split_reference_end - check_param_end) / 60
        print(f"{Colors.INFO}INFO: Reference split in {split_reference_time:.2f} minutes.", file=sys.stderr, flush=True)
    else:
        split_reference_end = time.time()


    # FIND FEATURES
    if configs['find_features']:
        print("", file=sys.stderr, flush=True)
        print(f"{Colors.HEADER}---FINDING FEATURES---{Colors.END}", file=sys.stderr, flush=True)

        print("CHECKING FEATURE TABLES", file=sys.stderr, flush=True)
        data_structure_checker = DataStructureChecker()
        configs['feature_quant_table'], configs['feature_meta_table'] = data_structure_checker.check_paired_quant_and_meta_tables(configs=configs, 
                                                                                        quant_df=configs['feature_quant_table'], 
                                                                                        meta_df=configs['feature_meta_table'], 
                                                                                        min_samples=MIN_SAMPLES_FEATURES, 
                                                                                        balance=False)
        configs['filtered_feature_quant_table'] = data_structure_checker.filter_quant_table(configs=configs, 
                                                                                        quant_df=configs['feature_quant_table'], 
                                                                                        meta_df=configs['feature_meta_table'])

        feature_finder = FeatureSelector()
        configs['rules'], configs['true_scores'], configs['all_evaluated_rules'], configs['feature_table'] = feature_finder.find_features(configs=configs)

        find_feature_end = time.time()
        find_feature_time = (find_feature_end - split_reference_end) / 60
        print(f"{Colors.INFO}INFO: Features found in {find_feature_time:.2f} minutes.", file=sys.stderr, flush=True)
    else:
        find_feature_end = time.time()


    # TRAIN MODEL
    if configs['train_model']:
        print("", file=sys.stderr, flush=True)
        print(f"{Colors.HEADER}---TRAINING MODEL---{Colors.END}", file=sys.stderr, flush=True)

        print("CHECKING TRAIN TABLES", file=sys.stderr, flush=True)
        data_structure_checker = DataStructureChecker()
        configs['train_quant_table'], configs['train_meta_table'] = data_structure_checker.check_paired_quant_and_meta_tables(configs=configs, 
                                                                                                                          quant_df=configs['train_quant_table'], 
                                                                                                                          meta_df=configs['train_meta_table'], 
                                                                                                                          min_samples=MIN_SAMPLES_TRAIN, 
                                                                                                                          balance=False)
        
        print("CHECKING VALIDATE TABLES", file=sys.stderr, flush=True)
        configs['validate_quant_table'], configs['validate_meta_table'] = data_structure_checker.check_paired_quant_and_meta_tables(configs=configs, 
                                                                                                                                 quant_df=configs['validate_quant_table'], 
                                                                                                                                 meta_df=configs['validate_meta_table'], 
                                                                                                                                 min_samples=MIN_SAMPLES_VALIDATE, 
                                                                                                                                 balance=True)
        
        print("CHECKING FEATURE TABLE", file=sys.stderr, flush=True)
        data_structure_checker.check_feature_table(feature_df=configs['feature_table'])

        model_generator = ModelGenerator()
        configs['model'], configs['model_information'] = model_generator.run_model_generator(configs=configs)

        train_model_end = time.time()
        train_model_time = (train_model_end - find_feature_end) / 60
        print(f"{Colors.INFO}INFO: Model trained in {train_model_time:.2f} minutes.", file=sys.stderr, flush=True)
    else:
        train_model_end = time.time()

    
    # APPLY MODEL
    if configs['apply_model']:
        print("", file=sys.stderr, flush=True)
        print(f"{Colors.HEADER}---APPLYING MODEL---{Colors.END}", file=sys.stderr, flush=True)

        print("CHECKING EXPERIMENTAL TABLE", file=sys.stderr, flush=True)
        data_structure_checker = DataStructureChecker()
        configs['experimental_quant_table'] = data_structure_checker.check_quant_table(configs=configs, 
                                                                                      quant_df=configs['experimental_quant_table'])   
        
        print("CHECKING FEATURE TABLE", file=sys.stderr, flush=True)
        data_structure_checker.check_feature_table(feature_df=configs['feature_table'])

        print("CHECKING MODEL", file=sys.stderr, flush=True)
        data_structure_checker.check_model(configs=configs, model=configs['model'], feature_df=configs['feature_table'])

        experimental_classifier = ExperimentalClassifier()
        configs['experimental_classification'] = experimental_classifier.run_experimental_classifier(configs)

        apply_model_end = time.time()
        apply_model_time = (apply_model_end - train_model_end) / 60
        print(f"{Colors.INFO}INFO: Samples classified in {apply_model_time:.2f} minutes.", file=sys.stderr, flush=True)
        

    end_time = time.time()
    total_time = (end_time - start_time) / 60
    print("", file=sys.stderr, flush=True)
    print(f"{Colors.HEADER}---TOTAL RUNTIME: {total_time:.2f} MINUTES---{Colors.END}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
