# Running the Full NIFty Pipeline

When running the full pipeline, NIFty will:
1. generate and select TSP features, 
2. train a classification model, and 
3. apply the model to unlabeled data.

The code and descriptive examples below will help you run the full pipeline:
* [Full Pipeline Regression Test](../Regression%20Tests/ApplyClassifierTest/)
* [Full Pipeline Config File Example](../Regression%20Tests/ApplyClassifierTest/config.toml)

## Minimum Required Changes to Configurations:
To run NIFty in this mode, the following are the minimum required changes to the default configuration file:
* find_features = true
* train_model = true
* apply_model = true
* reference_quant_file = "your/path/to/reference/quant.tsv"
* reference_meta_file = "your/path/to/reference/meta.tsv"
* experimental_quant_file = "your/path/to/unlabeled/experimental/quant.tsv"

*NOTE: NIFty, by default, only requires one reference dataset to both generate features and train and validate a classifier. 
When run in this mode, NIFty internally splits the reference dataset into three, non-overlapping sets: a feature selection set, a training/testing set, and a validation set. 
Alternatively, if you want to split your reference data yourself (or use multiple reference datasets for different portions of the pipeline), you can make these additional changes to the configuration file:*
* input_files = "individual"
* feature_quant_file = "your/path/to/feature/selection/quant.tsv"
* feature_meta_file = "your/path/to/feature/selection/meta.tsv"
* train_quant_file = "your/path/to/training/testing/quant.tsv"
* train_meta_file = "your/path/to/training/testing/meta.tsv"
* validate_quant_file = "your/path/to/validation/quant.tsv"
* validate_meta_file = "your/path/to/validation/meta.tsv"

## Expected Output Files:
* `selected_features.tsv`
* `trained_model_and_model_metadata.pkl`
* `model_information.txt`
* `predicted_classes.tsv`
