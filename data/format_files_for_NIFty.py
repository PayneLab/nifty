#!/bin/usr/env python
import os
import pandas as pd



def format_df(df):
    df.set_index("Unnamed: 0", inplace=True)
    df_transposed = df.transpose()
    df_transposed.reset_index(inplace=True)
    df_transposed.rename(columns={"index":"sample_id", "label":"classification_label"}, inplace=True)
    
    quant_df = df_transposed.drop('classification_label', axis=1)
    meta_df = df_transposed.filter(items=['sample_id', 'classification_label'])

    return quant_df, meta_df

def save_df(df, output_file_path, output_file_name):
    output_file_path = os.path.join(output_file_path, output_file_name)
    df.to_csv(output_file_path, index=False, sep="\t")

# # easy dataset - no imputation
# easy_dataset_na = "na-allow-testing-dfs\\proteinXsinglecell_unimputed_labeled_corrected.csv"
# easy_file_path_na = "na-allow-testing-dfs\\Leduc_et_al_2022"

# # hard dataset  - no imputation
# hard_dataset_na = "na-allow-testing-dfs\\Protein_unimputed_labeled.csv"
# hard_file_path_na = "na-allow-testing-dfs\\Khan_Elcheikhali_et_al_2024"

# easy dataset - imputation
easy_dataset_no_na = os.path.join("no-na-testing-dfs", "proteinXsinglecell_imputed_labeled_corrected.csv")
easy_file_path_no_na = os.path.join("no-na-testing-dfs", "Leduc_et_al_2022")

# # hard dataset  - imputation
# hard_dataset_no_na = "no-na-testing-dfs\\Protein_imputed_labeled.csv"
# hard_file_path_no_na = "no-na-testing-dfs\\Khan_Elcheikhali_et_al_2024"


# ## EASY UNIMPUTED
# # read in the data
# easy_dataset_na = pd.read_csv(easy_dataset_na)
# quant_df, meta_df = format_df(easy_dataset_na)

# # save the data
# save_df(quant_df, easy_file_path_na, "quant_table_unimputed.tsv")
# save_df(meta_df, easy_file_path_na, "meta_table_unimputed.tsv")



## EASY IMPUTED
# read in the data
easy_dataset_no_na = pd.read_csv(easy_dataset_no_na)
quant_df, meta_df = format_df(easy_dataset_no_na)

# save the data
save_df(quant_df, easy_file_path_no_na, "quant_table_imputed.tsv")
save_df(meta_df, easy_file_path_no_na, "meta_table_imputed.tsv")



# ## HARD UNIMPUTED
# # read in the data
# hard_dataset_na = pd.read_csv(hard_dataset_na)
# quant_df, meta_df = format_df(hard_dataset_na)

# # save the data
# save_df(quant_df, hard_file_path_na, "quant_table_unimputed.tsv")
# save_df(meta_df, hard_file_path_na, "meta_table_unimputed.tsv")



# ## HARD IMPUTED
# # read in the data
# hard_dataset_no_na = pd.read_csv(hard_dataset_no_na)
# quant_df, meta_df = format_df(hard_dataset_no_na)

# # save the data
# save_df(quant_df, hard_file_path_no_na, "quant_table_imputed.tsv")
# save_df(meta_df, hard_file_path_no_na, "meta_table_imputed.tsv")




