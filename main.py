import pandas as pd

from DataTableChecker import DataTableChecker


def main():
    meta_df = pd.read_csv("meta.tsv", sep="\t")
    quant_df = pd.read_csv("quant.tsv", sep="\t")

    data_checker = DataTableChecker()


if __name__ == "__main__":
    main()
