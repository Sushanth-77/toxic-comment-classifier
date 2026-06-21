"""One-time script: split raw train.csv into train/val, apply both
preprocessing modes, and save to data/processed/.

Run with: python scripts/build_processed_data.py
"""

import logging

from toxic_clf.config import load_params
from toxic_clf.data import load_raw_train, train_val_split
from toxic_clf.preprocessing import clean_dataframe

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    params = load_params()
    df = load_raw_train()
    train_df, val_df = train_val_split(
        df,
        val_size=params["split"]["val_size"],
        random_state=params["split"]["random_state"],
    )

    for name, split_df in [("train", train_df), ("val", val_df)]:
        split_df = clean_dataframe(split_df, mode="heavy", output_col="clean_heavy")
        split_df = clean_dataframe(split_df, mode="light", output_col="clean_light")
        out_path = f"data/processed/{name}.csv"
        split_df.to_csv(out_path, index=False)
        logger.info("Saved %s (%d rows) to %s", name, len(split_df), out_path)


if __name__ == "__main__":
    main()
