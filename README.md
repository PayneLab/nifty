# NIFty
Never Impute Features (thank you).

NIFty is a python program for feature selection (including generation and scoring), model generation, and experimental classification that does not require missing-value imputation, avoids common circular analysis pitfalls by default, and overcomes batch effects. 
The primary application is large molecular data, like proteomics. 
We assume input to be two tables: (1) a table with proteins (or other data) as the columns and samples as the rows, and (2) a table that has the label (class) for each sample. 
The output from our program depends on which functionalty the user would like to run. 
In the 'find_features' mode, the output is a list of the k top features that can be used to train a machine learning classifier to annotate samples. 
In the 'train_model' mode, the output is a trained machine learning model on the selected features. 
In the 'apply_model' mode, the output is a list of sample classifications from applying the trained model on experimental, unlabeled data.
The important thing is that we never impute. We can deal with null values.

Run this program with the following command on the commandline (assuming config.toml exists in the same directory):
> python nifty.py

Run this program with the following command on the commandline (with a custom config filepath):
> python nifty.py -c <config/file/path>

The codebase functions as follows:
![NIFty Flowchart](images/Pipeline_flow.png)

## How it works
NIFTfy can be executed in several modes depending on which steps of the pipeline you want to run:
1. **find_features**: Generate and score rules (features) to find the top k features for classification.
2. **train_model**: Train a machine learning classifier using the selected features.
3. **apply_model**: Apply the trained classifier on experimental, unlabeled data.

You can control this behavoir using the config.toml file.
Here are all valid combinations of the three main flags:

## Tutorials:
- [Full Pipeline](tutorials/full_pipeline_tutorial.md)
- 


## Full Pipeline:
The most common use of our tool will be to run the full pipeline. This means that the program will 1. find features, 2. train a model and 3. apply the model. The code and descriptive examples below will help you run the full pipeline.
LINK TO REGRESSION TEST FOLDR
LINK TO REGRESSION TEST CONFIG

### Configuration:
- find_features = true 
- train_model = true 
- apply_model = true
### Requires
- reference_quant_file / feature_quant_file
- reference_meta_file / feature_meta_file
- train_quant_file / reference split
- train_meta_file / reference split
- validate_quant_file / reference split
- validate_meta_file / reference split
- experimental_quant_file
### Produces
- top_k_rules.csv
- trained_model_and_model_metadata.pkl
- model_information.txt
- predicted_classes.tsv


## Feature Selection Only:
Use when you only want the top k rules to later train a model manually or in another run.
### Configuration:
- find_features = true
- train_model = false
- apply_model = false
### Requires
- reference_quant_file / feature_quant_file
- reference_meta_file / feature_meta_file
### Produces
- top_k_rules.csv


## Train Model Only:
Use when you already have a feature file (rules) and want to train a model.
### Configuration:
- find_features = false
- train_model = true
- apply_model = false
### Requires
- feature_file
- train_quant_file
- train_meta_file
- validate_quant_file
- validate_meta_file
### Produces
- trained_model_and_model_metadata.pkl
- model_information.txt


## Apply Model Only:
Use when you already have a trained model and want to classify experimental samples.
### Configuration:
- find_features = false 
- train_model = false 
- apply_model = true
### Requires
- model_file
- experimental_quant_file
### Produces
- predicted_classes.tsv


## Find Features and Train Model:
Use when you want to find the top rules and then train a model using those rules in one run.
### Configuration:
- find_features = true
- train_model = true
- apply_model = false
### Requires
- reference_quant_file / feature_quant_file
- reference_meta_file / feature_meta_file
- train_quant_file / reference split
- train_meta_file / reference split
- validate_quant_file / reference split
- validate_meta_file / reference split
### Produces
- top_k_rules.csv
- trained_model_and_model_metadata.pkl
- model_information.txt


## Train Model and Apply Model:
Use when you want to train a model and then immediately apply it to unlabeled experimental samples.
### Configuration:
- find_features = false
- train_model = true
- apply_model = true
### Requires
- feature_file
- train_quant_file
- train_meta_file
- validate_quant_file
- validate_meta_file
- experimental_quant_file
### Produces
- trained_model_and_model_metadata.pkl
- model_information.txt
- predicted_classes.tsv


# File Formats

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
| SampleID | ProteinA | ProteinB | ProteinC |
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
| SampleID | Class   |
|----------|---------|
| S1       | A       |
| S2       | B       |
| S3       | A       |


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

Above is the ouput format that NIFty produces when making the feature file. The same format is expected when a feature file is given.

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
