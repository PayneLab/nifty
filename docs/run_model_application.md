# Using NIFty to Classify Unlabaled Data

It is possible to just apply a classification model to unlabeled data when running NIFty.
Running NIFty in this mode presupposes that, at some point, you have used NIFty to generate relevant features **and** train a matching model.

The code and descriptive examples below will help you run NIFty is find_features only mode:
* [Model Application Regression Test](../Regression%20Tests/ApplyClassifierOnlyTest/)
* [Model Application Config File Example](../Regression%20Tests/ApplyClassifierOnlyTest/config.toml)

## Minimum Required Changes to Configurations:
To run NIFty in this mode, the following are the minimum required changes to the default configuration file:
* find_features = false
* train_model = false
* apply_model = true
* experimental_quant_file = "your/path/to/unlabeled/experimental/quant.tsv"
* feature_file = "your/path/to/selected_features.tsv"
* model_file = "your/path/to/trained_model_and_model_metadata.pkl"

## Expected Output Files:
* `predicted_classes.tsv`
