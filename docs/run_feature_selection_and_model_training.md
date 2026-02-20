# Using NIFty to Generate and Select Features and Train a Classification Model

It is possible to generate and select the best *k* TSP features **and** train a matching classification model when running NIFty.

The code and descriptive examples below will help you run the full pipeline:
* [Feature Selection and Model Training Regression Test](../Regression%20Tests/ModelGeneratorTest/)
* [Feature Selection and Model Training Config File Example](../Regression%20Tests/ModelGeneratorTest/config.toml)

## Minimum Required Changes to Configurations:
To run NIFty in this mode, the following are the minimum required changes to the default configuration file:
* find_features = true
* train_model = true
* apply_model = false
* reference_quant_file = "your/path/to/reference/quant.tsv"
* reference_meta_file = "your/path/to/reference/meta.tsv"

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
