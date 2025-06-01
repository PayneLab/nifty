import pandas as pd


def main():
    meta_df = pd.read_csv("meta.tsv", sep="\t")
    quant_df = pd.read_csv("quant.tsv", sep="\t")


if __name__ == "__main__":
    main()
