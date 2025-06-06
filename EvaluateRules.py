import numpy as np

class GenerateRules:
    def __init__(self):
        pass

    def vectorize_pair(pair: list, quant_df):
        '''Gets all values for two proteins of a pair, compares them and returns a boolean vector'''
        prot1 = pair[0]
        prot2 = pair[1]

        prot1_values = quant_df[prot1].values
        prot2_values = quant_df[prot2].values

        #Check for NA values in either prot1_values or prot2_values using numpy
        for i in range(len(prot1_values)):
            # Prot1 is NA and Prot2 is not NA
            if np.isnan(prot1_values[i]) and not np.isnan(prot2_values[i]):
                prot1_values[i] = 0
                prot2_values[i] = 10
            # Prot2 is NA and Prot1 is not NA
            elif np.isnan(prot2_values[i]) and not np.isnan(prot1_values[i]):
                prot2_values[i] = 0
                prot1_values[i] = 10
            # Both are NA, result in bool vector will be False
            elif np.isnan(prot1_values[i]) and np.isnan(prot2_values[i]):
                prot1_values[i] = 0
                prot2_values[i] = 10

        bool_vector = prot1_values > prot2_values

        return bool_vector
    
    def score_pair(pair: list, quant_df, meta_df):
        '''Scores a pair of proteins based on how well they separate the classes in the meta data'''
        bool_vector = GenerateRules.vectorize_pair(pair, quant_df)
        class_labels = meta_df['classification_label'].values

        # Change labels to 0 and 1
        class_labels = np.array([1 if label == class_labels[0] else 0 for label in class_labels])

        # Find TP and FP values
        TP = np.sum((bool_vector == 1) & (class_labels == 1))
        FP = np.sum((bool_vector == 1) & (class_labels == 0))

        # Find proportion of TP and FP
        TP_prop = TP / np.sum(class_labels == 1)
        FP_prop = FP / np.sum(class_labels == 0)

        # Calculate Score
        score = abs(TP_prop - FP_prop)

        return score
    
    def evaluate_pairs(self, pairs: list, quant_df, meta_df) -> list:
        '''Evaluates all pairs of proteins and returns a list of tuples with the pair and its score'''
        scored_pairs = []
        for pair in pairs:
            score = GenerateRules.score_pair(pair, quant_df, meta_df)
            scored_pairs.append((pair, score))
        
        return scored_pairs
        
