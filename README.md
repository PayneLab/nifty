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
