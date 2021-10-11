import argparse
import sys
import logging

from .similarity import WebSimilarity

logging.basicConfig(
    format="%(asctime)s:%(name)s:%(levelname)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__package__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "words", nargs="*", help="Words to which web similarity is computed"
    )
    parser.add_argument(
        "--wordlist",
        help="Path to the wordlist json file",
        type=str,
        dest="wordlist"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="More verbose output",
        dest="verbose",
        action="store_true"
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if len(args.words) != 2 and not args.wordlist:
        logger.error("Only two words or wordlist path are accepted %s provided", len(args.words))
        sys.exit(1)

    similarity = WebSimilarity(wordlist=args.wordlist)
    similarity.construct_result_table(args.words)
