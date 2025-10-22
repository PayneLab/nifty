# NIFTY
Never Impute Features (thank you).

NIFTY is a python program for feature generation, feature scoring and feature selection. The primary application is large molecular data, like proteomics. We assume input to be a table with samples as the columns and protein (or other data) as the rows. We also require an input file that has the label (class) for each sample. The output from our program is the list of k top features that can be used to train a ML classifier to annotate samples. 
The important thing is that we never impute. We can deal with null values.

Run this program with the following command line
> python main.py -q [quant table file] -m [meta data containing sample labels]



![NIFty train model flowchart](images/Pipeline flow(1).png)
![NIFty apply model flowchart](images/Pipeline flow(2).png)