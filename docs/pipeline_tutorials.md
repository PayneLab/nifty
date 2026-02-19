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