# Using NIFty to Train a Classification Model

It is possible to just train a classification model when running NIFty.
Running NIFty in this mode presupposes that, at some point, you have used NIFty to generate relevant features.

The code and descriptive examples below will help you run NIFty is find_features only mode:
* [Model Training Regression Test](../Regression%20Tests/ModelGenOnlyTest)
* [Model Training Config File Example](../Regression%20Tests/ModelGenOnlyTest/config.toml)

## Minimum Required Changes to Configurations:
To run NIFty in this mode, the following are the minimum required changes to the default configuration file:
* find_features = false
* train_model = true
* apply_model = false
* reference_quant_file = "your/path/to/reference/quant.tsv"
* reference_meta_file = "your/path/to/reference/meta.tsv"
* feature_file = "your/path/to/selected_features.tsv"

*NOTE: NIFty, by default, only requires one reference dataset to both generate features and train and validate a classifier. 
When run in this mode, NIFty internally splits the reference dataset into three, non-overlapping sets: a feature selection set, a training/testing set, and a validation set. 
Alternatively, if you want to split your reference data yourself (or use multiple reference datasets for different portions of the pipeline), you can make these additional changes to the configuration file:*
* input_files = "individual"
* train_quant_file = "your/path/to/training/testing/quant.tsv"
* train_meta_file = "your/path/to/training/testing/meta.tsv"

## Expected Output Files:
* `trained_model_and_model_metadata.pkl`
* `model_information.txt`
