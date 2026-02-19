## Full Pipeline:

Why would you want to run the full pipeline?

TO-DO 

The most common use of our tool will be to run the full pipeline. This means that the program will:
1. Find features
2. Train a model
3. Apply the model

The code and descriptive examples below will help you run the full pipeline.

- [Full Pipeline Regression Test](../Regression%20Tests/ApplyClassifierTest/)
- [Full Pipeline Config File Example](../Regression%20Tests/ApplyClassifierTest/config.toml)

### Required Configuration:
- find_features = true
- train_model = true
- apply_model = true
- seed = 42
- input_files = "individual" (default is "reference")

---

### Requires
Because `input_files = "individual"` is used, the following files are required:

#### Feature Selection
- `feature_quant_file`
- `feature_meta_file`

#### Model Training
- `train_quant_file`
- `train_meta_file`
- `validate_quant_file`
- `validate_meta_file`

#### Model Application
- `experimental_quant_file`

---

### Produces
- `selected_features.tsv`
- `trained_model_and_model_metadata.pkl`
- `model_information.txt`
- `predicted_classes.tsv`

### Full Pipeline in Reference Mode

Why would you want to run the full pipeline in Reference Mode?

TODO

Use this mode when you want to provide a single reference dataset and let NIFty automatically split it internally.

### Required Configuration
- find_features = true  
- train_model = true  
- apply_model = true  
- seed = 42  
- input_files = "reference"

---

### Requires
Because `input_files = "reference"` is used, the following files are required:

- `reference_quant_file`
- `reference_meta_file`
- `experimental_quant_file`

---

### Produces
- `selected_features.tsv`
- `trained_model_and_model_metadata.pkl`
- `model_information.txt`
- `predicted_classes.tsv`