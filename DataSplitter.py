## TODO

class DataSplitter:

    def __init__(self):
        pass

    def split_table(self):
        # TODO: write this function
        pass

    def run_data_splitter(self, configs):
        # TODO: split reference as needed and store splits to the correct config locations
        
        if configs['split_for_FS'] and not configs['split_for_train'] and not configs['split_for_validate']:
            configs['feature_quant_table'] = configs['reference_quant_table']
            configs['feature_meta_table'] = configs['reference_meta_table']
        elif not configs['split_for_FS'] and configs['split_for_train'] and not configs['split_for_validate']:  # should never happen
            configs['train_quant_table'] = configs['reference_quant_table']
            configs['train_meta_table'] = configs['reference_meta_table']
        elif not configs['split_for_FS'] and not configs['split_for_train'] and configs['split_for_validate']:  # should never happen
            configs['validate_quant_table'] = configs['reference_quant_table']
            configs['validate_meta_table'] = configs['reference_meta_table']
        elif configs['split_for_FS'] and configs['split_for_train'] and not configs['split_for_validate']:  # should never happen
            # TODO split into FS and train
            pass
        elif configs['split_for_FS'] and not configs['split_for_train'] and configs['split_for_validate']:  # should never happen
            # TODO split into FS and validate
            pass
        elif not configs['split_for_FS'] and configs['split_for_train'] and configs['split_for_validate']:
            # TODO: split for train and validate
            pass
        else:
            # TODO: split for FS, train, and validate
            pass

