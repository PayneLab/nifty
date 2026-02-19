# NIFty
Never Impute Features (thank you).

The pre-print manuscript associated with this tool can be found here: <TBA>

NIFty is a python program for feature selection (including generation and scoring), model generation, and experimental classification that does not require missing-value imputation, avoids common circular analysis pitfalls by default, and overcomes batch effects. 
The primary application is large molecular data, like proteomics. 
We assume input to be two tables: 

1. a table with quantification data, proteins (or some other data type) as the columns and samples as the rows; and
2. a table that has the label (class) for each sample. 

The output from our program depends on which functionalty the user would like to run. 
In the 'find_features' mode, the output is a list of the *k* best features that can be used to train a machine learning classifier for sample annotation. 
In the 'train_model' mode, the output is a trained machine learning model on the selected features. 
In the 'apply_model' mode, the output is a list of predicted sample labels or label probabilities from applying the trained model on experimental, unlabeled data.

The important thing is that we never impute; we can deal with null values.

After downloading this repository, run NIFty on your own data with the following command on the commandline (assuming config.toml exists in the same directory):
> python nifty.py

After downloading this repository, run NIFty on your own data with the following command on the commandline (with a custom config filepath):
> python nifty.py -c <config/file/path>

## How NIFty Generates and Selects Features

NIFty's implemented feature selection method is what allows it to overcome missing values in the input, double dipping issues associated with sample classification, and large batch effects in all stages of classification (feature selection, model generation, and model application). 
Briefely, ...

For a more detailed description, please visit <TBA>.

## Requirements

NIFty requires Python (version ?? or greater) and the following Python packages:

## Codebase Structure
The codebase functions as follows:
![NIFty Flowchart](images/Pipeline_flow.png)

## Run Modes
NIFty can be executed in several modes depending on which steps of the pipeline you want to run:
1. **find_features**: Generate and score features to find the best *k* features for classification.
2. **train_model**: Train a machine learning classifier using the selected features.
3. **apply_model**: Apply the trained classifier on experimental, unlabeled data.

You can control this behavoir using a `.toml` configuration file.

## File Formats and Descriptions
* A description of all necessary input files and their required formats can be found [here](docs/input_file_formats.md).
* A description of all output files can be found [here](docs/output_file_formats.md).

## Use Cases
Each of the use-case documents below contain the following information: (1) a brief description about when to run a particular use case of NIFty; and (2) changes to default configurations needed to run that particular use case (to see a default configuration file, see [**File Formats and Descriptions**](#file-formats-and-descriptions)).

* [Full Pipeline (Feature Selection, Model Training, and Model Application)](docs/run_full_pipeline.md)
* [Feature Selection](docs/run_feature_selection.md)
* [Model Training](docs/run_model_training.md)
* [Model Application](docs/run_model_application.md)
* [Feature Selection and Model Training](docs/run_feature_selection_and_model_training.md)
* [Model Training and Model Application](docs/run_model_training_and_application.md)

## Citing NIFty

<TBA>
