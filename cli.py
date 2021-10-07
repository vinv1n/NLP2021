#!/usr/bin/env python3

import argparse
import sys
import logging

from nlp import WebSimilarity

logging.basicConfig(
    format="%(asctime)s:%(name)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "words", nargs="*", help="Words to which web similarity is computed"
    )

    args = parser.parse_args()
    if len(args.words) != 2:
        logger.error("Only two words are accepted %s provided", len(args.words))
        sys.exit(1)

    similarity = WebSimilarity()
    similarity.web_similarity(*args.words)
