### Feature File Format:
| Protein_Pair        | Protein1 | Protein2 | Score | P_Value |
|---------------------|----------|----------|-------|---------|
| ('P1','P2')         | P1       | P2       | 1.00  | 0.0     |
| ('P3','P2')         | P3       | P2       | 0.88  | 0.0     |
| ('P4','P5')         | P4       | P5       | 0.88  | 0.0     |
| ('P6','P7')         | P6       | P7       | 0.85  | 0.0     |

Above is the ouput format that NIFty produces when making the feature file. If features are given to NIFty, the above format is accepted as well as a simple 2 column dataframe with Protein1 and Protein2 being columns, implying that Protein1 < Protein2.

| Protein1 | Protein2 |
|----------|----------|
| P1       | P2       |
| P3       | P2       |
| P4       | P5       |
| P6       | P7       |

### Model File Description:
This is a binary pickle file containing everything needed to apply a trained NIFty model to new data.

It includes:
- the trained classifier
- the selected rule-based features
- metadata for transforming new samples
- training details (hyperparameters, CV scores, seed)

Use this file by setting model_file in the config when apply_model = true.

### Output Files:

### Feature File Format:
| Protein_Pair        | Protein1 | Protein2 | Score | P_Value |
|---------------------|----------|----------|-------|---------|
| ('P1','P2')         | P1       | P2       | 1.00  | 0.0     |
| ('P3','P2')         | P3       | P2       | 0.88  | 0.0     |
| ('P4','P5')         | P4       | P5       | 0.88  | 0.0     |
| ('P6','P7')         | P6       | P7       | 0.85  | 0.0     |

Above is the ouput format that NIFty produces when making the feature file. The same format is expected when a feature file is given.

### Model File Description:
This is a binary pickle file containing everything needed to apply a trained NIFty model to new data. 

This file is saved as the following:
trained_model_and_model_metadata.pkl

It includes:
- the trained classifier
- the selected rule-based features
- metadata for transforming new samples
- training details (hyperparameters, CV scores, seed)

This file can also be used as input by setting model_file in the config when apply_model = true.

### Model Information File (model_information.txt):
This file provides a human-readable summary of the trained model. It includes:

1. Model Parameters:
All hyperparameters used by the final trained classifier (e.g., number of trees, max depth, random seed).

2. Cross-Validation Performance:
Mean and standard deviation for metrics such as accuracy, precision, and recall computed during training.

3. Validation Set Results:
Final performance metrics on the held-out validation dataset (accuracy, precision, recall).


This file is meant for inspection and reporting—it does not contain the model itself.