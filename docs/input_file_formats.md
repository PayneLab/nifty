# Input File Formats and Descriptions

NIFty requires three main input files/formats:
1. [Configuration File](#configuration-file): this file contains all of the configuration settings to run NIFty, including paths to the other two required input files.
2. [Quantification File](#quantification-files): this file contains quantification information for all samples. 
3. [Metadata File](#metadata-files): this file contains required metadata to map samples to class labels.

Below are descriptions of the required information and formats for each of the input files.

## Configuration File

The configuration file is a `.toml` file containing all of the configuration settings to run NIFty.
For information on the TOML file syntax and types, see [[Tom's Obvious Minimal Language]](https://toml.io/en/).

There are five sets of settings in NIFty's config file:
1. [Project Settings](#project-settings)
2. [File Paths](#file-paths)
3. [Feature Selection Settings](#feature-selection-settings)
4. [Model Training Settings](#model-training-settings)
5. [Model Application Settings](#model-application-settings)

A baseline configuration file to run NIFty can be found in the main repository directory under `config.toml`. 
*NOTE: This file has default settings in place where possible, but there are a number of settings that **must** be provided by the user before NIFty can be run.
Required settings with no default value are left blank.*

Below are descriptions of each customizable setting in the configuration file.

### Project Settings

#### Example TOML format for NIFty project settings:

```toml
# project settings
find_features =   # required
train_model =   # required
apply_model =   # required
seed = "random"  # optional, default is "random"

input_files = "reference"  # required if find_features or train_model are true; options: "reference", "individual"; default is "reference"
```

#### Description of each project setting:

| Key | Status | Value | Description | Default |
| --- | ------ | ----- | ----------- | ------- | 
| `find_features` | Required | bool | Enables feature generation and selection and outputs the best *k* features. | |
| `train_model` | Required | bool | Enables machine learning classifier training using selected rules. | |
| `apply_model` | Required | bool | Enables trained model application on unlabeled data. | |
| `seed` | Optional |int or `"random"` | Sets seed for reproducibility or uses random seed. | `"random"` |
| `input_files` | Optional | `"reference"` or `"individual"` | If `find_features = true` or `train_model = true`, indicates whether quant and meta data files are one, reference set to be split for each mode or whether individual quant and metadata files will be provided for each mode. | `"reference"` |

### File Paths

#### Example TOML format for NIFty project settings:

```toml
# file paths
output_dir = "cwd"  # Optional, default is "cwd"

reference_quant_file = ""  # required if find_features or train_model are true and input_files = "reference"
reference_meta_file = ""  # required if find_features or train_model are true and input_files = "reference"

feature_quant_file = ""  # required if find_features = true and input_files = "individual"
feature_meta_file = ""  # required if find_features = true and input_files = "individual"
feature_file = ""  # required if find_features = false

train_quant_file = ""  # required if train_model = true and input_files = "individual"
train_meta_file = ""  # required if train_model = true and input_files = "individual"
validate_quant_file = ""  # required if train_model = true and input_files = "individual"
validate_meta_file = ""  # required if train_model = true and input_files = "individual"
model_file = ""  # required if train_model = false and apply_model = true

experimental_quant_file = ""  # required if apply_model = true
```

#### Description of each project setting:

| Key | Status | Value | Description | Default |
| --- | ------ | ----- | ----------- | ------- | 
| `output_dir` | Optional | basic string | Path to directory where NIFty should write results. `"cwd"` uses current directory. | `"cwd"` |
| `reference_quant_file` | Required if (`find_features = true` or `train_model = true`) **and** `input_files = "reference"` | basic string | Path to quantification table for the reference dataset. | |
| `reference_meta_file` |  Required if (`find_features = true` or `train_model = true`) **and** `input_files = "reference"` | basic string | Path to metadata table for the reference dataset. | |
| `feature_quant_file` | Required if `find_features = true` **and** `input_files = "individual"` | basic string | Path to quantification table for feature selection. | |
| `feature_meta_file` | Required if `find_features = true` **and** `input_files = "individual"` | basic string | Path to metadata table for feature selection. | |
| `feature_file` | Required if `find_features = false` | basic string | Path to `selected_features.tsv` file previously generated by NIFty's `find_features` mode. | |
| `train_quant_file` | Required if `train_model = true` **and** `input_files = "individual"` | basic string | Path to quantification table for model training. | |
| `train_meta_file` | Required if `train_model = true` **and** `input_files = "individual"` | basic string | Path to metadata table for model training. | |
| `validate_quant_file` | Required if `train_model = true` **and** `input_files = "individual"` | basic string | Path to quantification table for model validation. | |
| `validate_meta_file` | Required if `train_model = true` **and** `input_files = "individual"` | basic string | Path to metadata table for model validation. | |
| `model_file` | Required if `train_model = false` **and** `apply_model = true` | bastic string | Path to `trained_model_and_model_metadata.pkl` file previously generated by NIFty's `train_model` mode. | |
| `experimental_quant_file` | Required if `apply_model = true` | basic string | Path to quantification table for model application. | |

### Feature Selection Settings

#### Example TOML format for NIFty project settings:

```toml
# find_features settings
k_rules = 15  # optional, default is 15, max is 50
missingness_cutoff = 0.5  # optional, default is 0.5
disjoint = false  # optional, default is false
mutual_information = true  # optional, default is true
mutual_information_cutoff = 0.7  # optional, default is 0.7
```

#### Description of each project setting:

| Key | Status | Value | Description | Default |
| --- | ------ | ----- | ----------- | ------- | 
| `k_rules` | Optional | int | Number (between 1 and 50) of top rules to keep. | 15 |
| `missingness_cutoff` | Optional | float | Data-missingness threshhold (between 0.0 and 1.0); if the proportion of missing values in a quant column is above this threshhold in all classes, the column is dropped. | 0.5 |
| `disjoint` | Optional | bool  | Enables disjoint filtering for feature selection. When enabled, selected features cannot share proteins. | false |
| `mutual_information` | Optional | bool | Enables mutual information filtering for feature selection. When enabled, the best *k* features that share less mutual information than the `mutual_information_cutoff` are selected. | true |
| `mutual_information_cutoff` | Optional | float | Mutual information theshhold (between 0.0 and 1.0); if the amount of mututal information shared between a prospective top feature and any other already-selected feature is above the threshhold, the feature is not selected. | 0.7 |

### Model Training Settings

#### Example TOML format for NIFty project settings:

```toml
# train_model settings
impute_NA_missing = true  # optional, default is true
cross_val = 5
model_type = "RF"  # optional, default is "RF"
autotune_hyperparameters = ""  # if blank, no autotune will be done. User inputs either "random" or "grid"
autotune_n_iter = 20  # optional, default is 20 for random search
verbose = 0  # optional, default is 0, allowed values include [0, 1, 2, 3, 4]
```

#### Description of each project setting:

| Key | Status | Value | Description | Default |
| --- | ------ | ----- | ----------- | ------- | 
| `impute_NA_missing` | Optional | bool | When enabled, missing quant columns required for data transformation based on selected features are created and filled with *NA*. | true |
| `cross_val` | Opteional | int | Number (>0) of folds for model train/test cross-validation. | 5 |
| `model_type` | Optional | `"RF"` or `"SVM"` | Type of classifier trained, random forest (RF) or SVM. | `"RF"` |
| `autotune_hyperparameters` | Optional | `"random"`, `"grid"`, or `""` | If `"random"`, randomized search is used for model hyperparameter tuning. If `"grid"`, grid search is used for model hyperparameter turning. If `""`, no model hyperparameter tuning is performed (default parameters used). | `""` |
| `autotune_n_iter` | Optional | int | Number of randomized search iterations for model hyperparameter tuning. | 20 |
| `verbose` | Optional | int | Number (0-4) indicating the level of console output from Scikit-learn model training process. | 0 |                 |

### Model Application Settings

#### Example TOML format for NIFty project settings:

```toml
# apply_model settings
prediction_format = "classes"  # can be 'classes' or 'probabilities', default is 'classes'
```

#### Description of each project setting:

| Key | Status | Value | Description | Default |
| --- | ------ | ----- | ----------- | ------- | 
| `prediction_format` | Optional | `"classes"` or `"probabilities"` | Specifies whether output classification predictions are classes or probability vectors. | `"classes"` |


## Quantification File(s)

Quantification files must always be paired with a matching metadata file.

### Requirements
Quantification files in NIFty have the following structural requirements:
* Quant files must be in `.tsv` format
* Rows represent samples, one row per unique sample ID
* Columns represent proteins, or other quantified molecular identifiers, one column per unique molecular identifier
    * One column must be called `sample_id` and contain unique identifiers for the samples
    * All other columns must contain numeric or *NA* (empty) values

*NOTE: Order of rows does not matter, NIFty automatically sorts quantification tables on the `sample_id` column.*

### Example Structure
Example quantification file structure (`.tsv`):
```
sample_id\tProteinA\tProteinB\tProteinC\n
S1\t8.3\t\t0.54\n
S2\t7.1\t1.88\t\n
S3\t\t0.92\t0.44\n
```

Example quantification file structure (dataframe):
| sample_id| ProteinA | ProteinB | ProteinC |
|----------|----------|----------|----------|
| S1       | 8.3      | *NaN*    | 0.54     |
| S2       | 7.1      | 1.88     | *NaN*    |
| S3       | *NaN*    | 0.92     | 0.44     |

## Metadata File(s)

Metadata files must always be paired with a matching quantification file.

### Requirements
Metadata files in NIFty have the following structural requirements:
* One column must be called `sample_id` and contain unique identifiers for the samples; IDs in this column must match exactly to those in the `sample_id` column in the quant table
* One column must be called `classification_label` and contain class labels corresponding to the samples

*NOTE: Order of rows does not matter, NIFty automatically sorts metadata tables on the `sample_id` column.*

### Example Structure
Example quantification file structure (`.tsv`):
```
sample_id\tclassification_label\n
S1\tA\n
S2\tB\n
S3\tA\n
```

Example metadata file structure (dataframe):
| sample_id | classification_label  |
|-----------|-----------------------|
| S1        | A                     |
| S2        | B                     |
| S3        | A                     |


