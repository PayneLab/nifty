# Using NIFty to Generate and Select Features

It is possible to just generate and select the best *k* TSP features when running NIFty.

The code and descriptive examples below will help you run NIFty is find_features only mode:
* [Feature Selection Regression Test](../Regression%20Tests/FeatureSelectionTest/)
* [Feature Selection Config File Example](../Regression%20Tests/FeatureSelectionTest/config.toml)

## Minimum Required Changes to Configurations:
To run NIFty in this mode, the following are the minimum required changes to the default configuration file:
* find_features = true
* train_model = false
* apply_model = false
* reference_quant_file = "your/path/to/reference/quant.tsv"
* reference_meta_file = "your/path/to/reference/meta.tsv"

*NOTE: NIFty, by default, only requires one reference dataset to both generate features and train and validate a classifier. 
When run in this mode, NIFty internally splits the reference dataset into three, non-overlapping sets: a feature selection set, a training/testing set, and a validation set. 
Alternatively, if you want to split your reference data yourself (or use multiple reference datasets for different portions of the pipeline), you can make these additional changes to the configuration file:*
* input_files = "individual"
* feature_quant_file = "your/path/to/feature/selection/quant.tsv"
* feature_meta_file = "your/path/to/feature/selection/meta.tsv"

## Expected Output Files:
* `selected_features.tsv`
