"""CLI entrypoint for fine-tuning DistilBERT.

Run with --smoke-test for a fast, CPU-friendly pipeline correctness check
on a tiny data subset before committing to a real GPU training run.
"""

import argparse
import logging

from toxic_clf.train_distilbert import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Tiny subset, 1 epoch -- pipeline check, not meaningful results.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(smoke_test=args.smoke_test)
