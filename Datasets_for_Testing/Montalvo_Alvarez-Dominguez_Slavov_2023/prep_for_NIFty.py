#!/bin/usr/env python
import os
import pandas as pd

script_path = os.path.abspath(__file__)
script_directory = os.path.dirname(script_path)

# read in the imputed data
imputed_raw_path = os.path.join(script_directory, "Original_Data", "Protein_imputed.csv")
imputed_raw = pd.read_csv(imputed_raw_path)

# transpose the df so proteins are column names and sample ids are rows in a column called 'sample_id'
imputed = imputed_raw.T.reset_index()
imputed.columns = imputed.iloc[0]
imputed = imputed.drop(0).rename(columns={'Unnamed: 0': 'sample_id'})



# read in the unimputed data
unimputed_raw_path = os.path.join(script_directory, "Original_Data", "Protein_uniputed.csv")
unimputed_raw = pd.read_csv(unimputed_raw_path)

# transpose the df so proteins are column names and sample ids are rows in a column called 'sample_id'
unimputed = unimputed_raw.T.reset_index()
unimputed.columns = unimputed.iloc[0]
unimputed = unimputed.drop(0).rename(columns={'Unnamed: 0': 'sample_id'})



# read in the meta data
meta_raw_path = os.path.join(script_directory, "Original_Data", "meta.csv")
meta_raw = pd.read_csv(meta_raw_path)

# filter the meta data file to just the cell types (ko and wt)
meta = meta_raw[(meta_raw['celltype'] == "ko") | (meta_raw['celltype'] == "wt")]

# format the meta table to two columns: sample_id and classification_label
meta = meta[['id', 'celltype']].rename(columns={'id':'sample_id', 
                                                'celltype':'classification_label'})



# filter dfs to the samples in the meta df
samples_to_keep = sorted(list(set(meta['sample_id'].tolist()) & set(imputed['sample_id'].tolist()) & set(unimputed['sample_id'].tolist())))

imputed_final = imputed[imputed['sample_id'].isin(samples_to_keep)]
unimputed_final = unimputed[unimputed['sample_id'].isin(samples_to_keep)]
meta_final = meta[meta['sample_id'].isin(samples_to_keep)]



# save the new files
output_directory = os.path.join(script_directory, "NIFty_Ready_Data")
imputed_final.to_csv(os.path.join(output_directory, "quant_imputed.tsv"), sep="\t", index=False)
unimputed_final.to_csv(os.path.join(output_directory, "quant_unimputed.tsv"), sep="\t", index=False)
meta_final.to_csv(os.path.join(output_directory, "meta.tsv"), sep="\t", index=False)


