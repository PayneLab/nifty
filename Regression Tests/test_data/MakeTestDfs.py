#!/usr/bin/env python3
import os
import random
import pandas as pd

# -------- CONFIG --------
QUANT_FILENAME = "/Users/blakemcgee/Desktop/TSP Classifier/github/nifty/data/na-allow-testing-dfs/Leduc_et_al_2022/quant_table_unimputed.tsv"
META_FILENAME = "/Users/blakemcgee/Desktop/TSP Classifier/github/nifty/data/na-allow-testing-dfs/Leduc_et_al_2022/meta_table_unimputed.tsv"

N_PARTS = 4
PER_CLASS_PER_PART = 50  # 50 of each class per split
CLASS_COL = "classification_label"
ID_COL = "sample_id"
PART_NAMES = ["FS", "Train_Test", "Val", "Experimental"]
RANDOM_SEED = 42
# ------------------------


def main():
    random.seed(RANDOM_SEED)

    # Get directory of this script / where files live
    base_dir = os.path.abspath(os.path.dirname(__file__))
    quant_path = os.path.join(base_dir, QUANT_FILENAME)
    meta_path = os.path.join(base_dir, META_FILENAME)

    # Load data
    quant = pd.read_csv(quant_path, sep="\t")
    meta = pd.read_csv(meta_path, sep="\t")

    # Basic checks
    if ID_COL not in quant.columns:
        raise ValueError(f"'{ID_COL}' column not found in quant table.")
    if ID_COL not in meta.columns:
        raise ValueError(f"'{ID_COL}' column not found in meta table.")
    if CLASS_COL not in meta.columns:
        raise ValueError(f"'{CLASS_COL}' column not found in meta table.")

    # Align by sample_id, and use only common samples (intersection)
    quant_i = quant.set_index(ID_COL)
    meta_i = meta.set_index(ID_COL)

    common_ids = quant_i.index.intersection(meta_i.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping sample_ids between quant and meta tables.")

    quant_i = quant_i.loc[common_ids]
    meta_i = meta_i.loc[common_ids]

    # Check there are exactly 2 classes
    class_values = meta_i[CLASS_COL].unique()
    if len(class_values) != 2:
        raise ValueError(
            f"Expected exactly 2 classes in '{CLASS_COL}', found: {class_values}"
        )

    # For each class, shuffle sample_ids and split into chunks of size PER_CLASS_PER_PART
    idx_by_class = {}
    for c in class_values:
        ids = meta_i.index[meta_i[CLASS_COL] == c].tolist()
        random.shuffle(ids)

        needed = N_PARTS * PER_CLASS_PER_PART
        if len(ids) < needed:
            raise ValueError(
                f"Not enough samples for class '{c}'. "
                f"Needed at least {needed}, found {len(ids)}."
            )

        # Take only as many as needed and split into N_PARTS chunks
        ids = ids[:needed]
        idx_by_class[c] = [
            ids[i * PER_CLASS_PER_PART : (i + 1) * PER_CLASS_PER_PART]
            for i in range(N_PARTS)
        ]

    # Build each split and save
    for part_idx, part_name in enumerate(PART_NAMES):
        part_ids = []
        # Add 50 from each class for this split
        for c in class_values:
            part_ids.extend(idx_by_class[c][part_idx])

        # Shuffle within the split (optional; can comment out if you want class-blocked)
        random.shuffle(part_ids)

        quant_part = quant_i.loc[part_ids].reset_index()  # bring sample_id back as column
        meta_part = meta_i.loc[part_ids].reset_index()

        # Output paths (same directory as input files)
        quant_out = os.path.join(base_dir, f"{part_name}_quant.tsv")
        meta_out = os.path.join(base_dir, f"{part_name}_meta.tsv")

        quant_part.to_csv(quant_out, sep="\t", index=False)
        meta_part.to_csv(meta_out, sep="\t", index=False)

        print(
            f"Saved {part_name}_quant.tsv and {part_name}_meta.tsv "
            f"with {len(quant_part)} samples "
            f"({PER_CLASS_PER_PART} per class)."
        )


if __name__ == "__main__":
    main()
