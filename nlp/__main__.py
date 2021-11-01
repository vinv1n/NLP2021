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
        "--wordlist", help="Path to the wordlist json file", type=str, dest="wordlist"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        help="More verbose output",
        dest="verbose",
        action="store_true",
    )
    parser.add_argument(
        "--single",
        "-s",
        help="Single Similarity metric",
        choices=["webjaccard"],
        dest="single",
    )
    parser.add_argument(
        "--snippet",
        "-p",
        help="Fecth snippets based on word",
        dest="snippet",
        action="store_true",
    )
    parser.add_argument(
        "--task",
        choices=[2, 3, 4, 5, 6, 7, 8],
        type=int,
        help="Select task to executute",
        dest="task",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="Path to the annotated data",
        dest="data",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    if len(args.words) != 2 and not args.wordlist:
        logger.error(
            "Only two words or wordlist path are accepted %s provided", len(args.words)
        )
        sys.exit(1)

    similarity = WebSimilarity(wordlist=args.wordlist)

    results = None
    if args.task == 2:
        results = similarity.get_correlation_between_words()
    elif args.task == 3:
        results = similarity.construct_result_table(args.words)
    elif args.task == 4:
        similarity.run_task4(*args.words)
    elif args.task == 5:
        results = similarity.sim_snippet1(*args.words)
    elif args.task == 6:
        results = similarity.execute_sim_snippet2()
    elif args.task == 7:
        results = similarity.compare_tasks()
    elif args.task == 8:
        if not args.data:
            logger.critical("Please define data path with --data argument")
            sys.exit(1)

        similarity.compute_correlation_with_annotated_data(args.data)

    if results is not None and not results.empty:
        with open(f"results-task-{str(args.task)}.html", "w") as fd:
            fd.write(results.to_html())
