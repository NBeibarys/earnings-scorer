"""
download_transcripts.py
Downloads earnings call transcripts from HuggingFace and saves as parquet.

Pinned to a specific dataset commit for full reproducibility.

Usage:
    python data/download_transcripts.py

Output:
    data/transcripts/transcripts.parquet
"""

import os
from datasets import load_dataset
from config import TRANSCRIPT_DATASET, TRANSCRIPT_DATASET_REVISION, TRANSCRIPTS


def main():
    os.makedirs(TRANSCRIPTS, exist_ok=True)
    output = os.path.join(TRANSCRIPTS, "transcripts.parquet")

    print(f"Dataset:  {TRANSCRIPT_DATASET}")
    print(f"Revision: {TRANSCRIPT_DATASET_REVISION}\n")

    ds = load_dataset(
        TRANSCRIPT_DATASET,
        revision=TRANSCRIPT_DATASET_REVISION,
        split="train"
    )

    df = ds.to_pandas()
    print(f"Total transcripts: {len(df):,}")
    print(f"Unique companies:  {df['symbol'].nunique():,}")

    df.to_parquet(output, index=False)
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
