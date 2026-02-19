# File Formats and Descriptions

## Input Files:

### Config File Description:

#### Project Settings:
| Setting         | Type                            | Description                                                                                                                                                                          |
| --------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `find_features` | bool                            | Runs feature generation to produce the top k rules. Required.                                                                                                                        |
| `train_model`   | bool                            | Trains a machine learning classifier using selected rules. Required.                                                                                                                 |
| `apply_model`   | bool                            | Applies a trained model to unlabeled data. Required.                                                                                                                                 |
| `seed`          | int or `"random"`               | Random seed for reproducibility. Optional (default: `"random"`).                                                                                                                     |
| `input_files`   | `"reference"` or `"individual"` | Defines whether you supply one reference dataset or separate individual datasets for each stage. Required if `find_features = true` or `train_model = true`. Default: `"reference"`. |

#### File Paths:
| Setting                   | Required When                                                          | Description                                                 |
| ------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------- |
| `output_dir`              | optional                                                               | Directory to write results. `"cwd"` uses current directory. |
| `reference_quant_file`    | (`find_features` or `train_model`) **and** `input_files = "reference"` | Quantification table for the reference dataset.             |
| `reference_meta_file`     | same as above                                                          | Metadata table for the reference dataset.                   |
| `feature_quant_file`      | `find_features = true` **and** `input_files = "individual"`            | Quant table used for feature selection only.                |
| `feature_meta_file`       | same as above                                                          | Metadata used for feature selection only.                   |
| `feature_file`            | `find_features = false`                                                | Existing feature file to use instead of generating rules.   |
| `train_quant_file`        | `train_model = true` **and** `input_files = "individual"`              | Training quantification table.                              |
| `train_meta_file`         | same as above                                                          | Training metadata.                                          |
| `validate_quant_file`     | same as above                                                          | Validation quant table.                                     |
| `validate_meta_file`      | same as above                                                          | Validation metadata.                                        |
| `model_file`              | `train_model = false` **and** `apply_model = true`                     | Trained model pickle file.                                  |
| `experimental_quant_file` | `apply_model = true`                                                   | Unlabeled samples for prediction.                           |

#### Feature Selection Settings:
| Setting                     | Type  | Description                                                          |
| --------------------------- | ----- | -------------------------------------------------------------------- |
| `k_rules`                   | int   | Number of top rules to keep. Optional, default = 15 (max = 50).      |
| `missingness_cutoff`        | float | Removes candidate features with too much missingness. Default = 0.5. |
| `disjoint`                  | bool  | If true, selected rules cannot share features. Default = false.      |
| `mutual_information`        | bool  | Whether to score rules using mutual information. Default = true.     |
| `mutual_information_cutoff` | float | Minimum mutual information required for a rule. Default = 0.7.       |

#### Model Training Settings:
| Setting                    | Type   | Description                                                                   |
| -------------------------- | ------ | ----------------------------------------------------------------------------- |
| `impute_NA_missing`        | bool   | Whether to impute missing values *after* rule transformation. Default = true. |
| `cross_val`                | int    | Number of folds for cross-validation.                                         |
| `model_type`               | string | Type of classifier to train. Default = `"RF"` (Random Forest).                |
| `autotune_hyperparameters` | string | `"random"` or `"grid"` for hyperparameter tuning. Blank = no tuning.          |
| `autotune_n_iter`          | int    | Number of random-search iterations for tuning. Default = 20.                  |
| `verbose`                  | int    | Level of console output (0–4). Default = 0.                                   |

#### Model Application Settings:
| Setting             | Type                             | Description                                                             |
| ------------------- | -------------------------------- | ----------------------------------------------------------------------- |
| `prediction_format` | `"classes"` or `"probabilities"` | Output predicted classes or probability vectors. Default = `"classes"`. |


### Quantification File Format:
| sample_id| ProteinA | ProteinB | ProteinC |
|----------|----------|----------|----------|
| S1       | 8.3      | NA       | 0.54     |
| S2       | 7.1      | 1.88     | NA       |
| S3       | NA       | 0.92     | 0.44     |

#### Requirements:
- Samples must be rows
- Proteins/features must be columns
- Missing values allowed (NA or empty cell)
- First column must contain unique sample identifiers
- File must be TSV

### Metadata File Format:
| sample_id | classification_label  |
|-----------|-----------------------|
| S1        | A                     |
| S2        | B                     |
| S3        | A                     |


#### Requirements:
- Rows must correspond exactly to quantification file sample names
- Must contain a column representing the class label
- SampleID must match the quantification file’s SampleID exactly
- Order does not need to match (NIFty aligns automatically)

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
