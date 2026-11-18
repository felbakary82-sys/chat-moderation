"""Step 1 - download the dataset and save train / validation / test as CSV files.

Dataset: OLID (OffensEval 2019, English tweets), from the TweetEval benchmark on
Hugging Face. Labels: 0 = clean (not offensive), 1 = offensive.
The official test split is our single held-out test set for ALL models.
"""
import pandas as pd
from datasets import load_dataset

import config


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    ds = load_dataset(config.HF_DATASET, config.HF_CONFIG)
    for split in ("train", "validation", "test"):
        df = pd.DataFrame({"text": ds[split]["text"], "label": ds[split]["label"]})
        df.to_csv(config.DATA_DIR / f"{split}.csv", index=False)
        share = df["label"].mean() * 100
        print(f"{split:10s} {len(df):6d} messages   offensive: {share:4.1f}%")
    print(f"Saved to {config.DATA_DIR}")


if __name__ == "__main__":
    main()
