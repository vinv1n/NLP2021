import logging
import os
import requests
import json
import pprint
import nltk
import re
import pandas as pd

from queue import Queue
from pathlib import Path
from itertools import product
from collections import defaultdict

from pathlib import Path
from typing import Optional, Iterator, Tuple, Any, Dict, List, DefaultDict

from nltk.corpus import wordnet
from multiprocessing import Lock


logger = logging.getLogger(__name__)

"""
I don't explain here how to get these, please see links below

https://developers.google.com/custom-search/v1/using_rest
https://developers.google.com/custom-search/v1/introduction#identify_your_application_to_google_with_api_key
https://stackoverflow.com/questions/4082966/what-are-the-alternatives-now-that-the-google-web-search-api-has-been-deprecated
"""

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CX_ID = os.environ.get("GOOGLE_CX_ID", "")
BASE_SEARCH_URL = (
    f"https://customsearch.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}"
)
DEBUG = False  # set True to allow search json to be written to disk


class WebSimilarityError(Exception):
    pass


def _permutate_words(word1: str, word2: str) -> Iterator[Tuple[str, str]]:
    for query in (
        f"{word1}&hq={word2}",  # word1 and word2
        f"{word1}&excludeTerms={word2}",  # word 1 and not word2
        f"{word2}&excludeTerms={word1}",  # word2 and not word1
    ):
        yield query


class WebSimilarity:
    def __init__(self, wordlist: str) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Maconstructed_url = template[:]c OS X 10.9; rv:40.0) Gecko/20100101 Firefox/40.0"
            }
        )

        self.snippets: DefaultDict[int, str] = defaultdict(str)
        self._loaded = False
        if wordlist:
            self.wordlist = self._load_wordlist(Path(wordlist))
        else:
            self.wordlist = []

    @staticmethod
    def _dump_search_results(entry):
        filepath = Path("samples", f"{word1}-{word2}.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(json.dumps(result, indent=4))

    @property
    def _load_wordnet(self):
        # this is kind stupid feature as I was not able to find
        # to check if wordnet is already downloaded
        if not self._loaded:
            self._loaded = nltk.download("wordnet")

    def _load_wordlist(self, wordlist_path: Path) -> List[Tuple[str, str, str, int]]:
        if not wordlist_path.exists():
            logger.warning(
                "Path %s to the wordlist does not exist", wordlist_path.as_posix()
            )
            return (
                []
            )  # make a copy of the list as we do not want to mess the original url

        data = {}
        with open(wordlist_path, "r") as fd:
            try:
                data = json.load(fd)
            except (json.JSONDecodeError, ValueError, OSError, IOError) as e:
                logger.warning("Failed to load wordlist, error %s", e)
                return []

        return [tuple(x) for _, x in data.items()]

    def _get_search_pages(self, word1: str, word2: str) -> List[int]:
        if not GOOGLE_API_KEY:
            logger.critical("Could not get page count, missing key")
            return []

        request_url_base = f"{BASE_SEARCH_URL}&cx={CX_ID}q"

        results = []
        for query in _permutate_words(word1, word2):
            request_url = f"{request_url_base}={query}"
            response = self.session.get(request_url)
            if response.status_code != 200:
                logger.critical(
                    "Return code %s from google api %s",
                    response.status_code,
                    response.text,
                )

                return []

            try:
                result = response.json()
            except (ValueError, IOError, json.decoder.JSONDecodeError) as e:
                logger.warning(
                    "Response from url %s was not json, error %s", request_url, e
                )
                continue

            logger.info(
                "Recieved response with content %s from google search api", request_url
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Response from api %s", pprint.pformat(result))

            if DEBUG:
                self._dump_search_results(result)

            results.append(result)

        return results

    def compute_web_jaccard_similarity(self, word1: str, word2: str) -> float:
        return self._compute_web_jaccard_similarity_by_search_results(word1, word2)

    @staticmethod
    def _construct_url_from_openseach_template(
        template: str, entry: Dict[str, Any]
    ) -> str:

        constructed_url = template[:]
        for key, value in entry.items():
            if key == "title":
                continue

            if key == "searchTerms":
                query = f"{{{key}}}"
            else:
                query = f"{{{key}\\?}}"

            constructed_url = re.sub(query, str(value), constructed_url)

        url_parts = []
        for part in constructed_url.split("&"):
            # this as really hacky way to do this, but I don't bother
            # to make this cleaner
            if part.endswith("}") and part.find("=") != -1:
                continue

            url_parts.append(part)

        return "&".join(url_parts)

    def fetch_search_snippets(self, word1: str, word2: str) -> float:
        """
        Task 4. implementation, relies on the previous fetch search result from custom
        search api implementation
        """

        # NOTE: from google api ducumentation
        # due to this limitation 100 is absolute maximum for snippet
        #
        # The index of the first result to return. The default number of results per page is 10,
        # so &start=11 would start at the top of the second page of results. Note: The JSON API will
        # never return more than 100 results, even if more than 100 documents match the query, so setting the sum of start + num
        # to a number greater than 100 will produce an error. Also note that the maximum value for num is 10.
        logger.info("Fetching snippets for search words %s %s", word1, word2)

        # setup queue to keep next pages in order
        queue = Queue()

        # add initial search pages to the queue
        response = self._get_search_pages(word1, word2)
        for resp in response:
            queue.put(resp)

        lock = Lock()
        while not queue.empty():
            search_page = queue.get()
            if not search_page:
                continue

            if next_page := search_page.get("queries", {}).get("nextPage", []):
                # for some reason this is a list, not sure why
                template = search_page.get("url", {}).get("template", "")
                for pages in next_page:
                    next_url = self._construct_url_from_openseach_template(
                        template, pages
                    )
                    if not next_url:
                        logger.warning("Could not construct url from entry %s", pages)
                        continue

                    response = self.session.get(next_url)
                    if not response or response.status_code != 200:
                        logger.warning(
                            "Could not fetch next page, reason %s %s",
                            response.status_code,
                            response.text,
                        )
                        continue

                    try:
                        content = response.json()
                    except (ValueError, json.decoder.JSONDecodeError) as e:
                        logger.warning("Could not decode json, error %s", e)
                        continue

                    # add new entry to queue
                    queue.put(content)

            for item in search_page.get("items", []):
                # items is a list of dicts
                snippet = item.get("snippet", "")
                if not snippet:
                    logger.warning("Search result entry without snippet, ignoring")
                    continue

                with lock:
                    self.snippets[hash(snippet)] = snippet

    def _compute_web_jaccard_similarity_by_search_results(
        self, word1: str, word2: str
    ) -> float:
        logger.info(
            "Using words %s and %s for WebJaccard similarity by search results",
            word1,
            word2,
        )

        search_results = self._get_search_pages(word1, word2)

        lengths = []
        for result in search_results:
            try:
                search_results = int(
                    result.get("searchInformation", {}).get("totalResults", "0")
                )
            except ValueError:
                search_results = 0

            logger.info("Found %s search results", search_results)

            # api has 10 results per "page" so that is what we use
            # TODO: make this less hard coded and more user defined
            count = search_results // 10
            if search_results % 10 != 0:  # one leftover so increase page cout by one
                count += 1

            lengths.append(count)

        if not lengths:
            logger.warning("No results with words %s and %s", word1, word2)
            return 0.0

        if (amount := len(lengths)) != 3:
            logger.warning(
                "Incorrect amount of results returned, %s should have been 3", amount
            )
            return 0.0

        try:
            similarity = lengths[0] / sum(lengths)
        except ZeroDivisionError:
            logger.warning("Sum of page lenghts is 0, cannot compute similarity")
            return 0.0

        logger.info("Similarity between word %s and %s is %s", word1, word2, similarity)
        return similarity

    @staticmethod
    def _get_correct_word_pair(word1_entry, word2_entry, word1, word2):
        """
        This makes my eyes bleed

        """
        for w1, w2 in product(word1_entry, word2_entry):
            if w1.pos() == w2.pos():
                return w1, w2
        return None, None

    def compute_semantic_similarity(
        self, word1: str, word2: str
    ) -> Tuple[float, float, int]:
        """
        Computes leacock chordorow (LCH) Similarity, Path length and Wu Palmer similarities

        :return: Tuple containing Wu palmer similarity, lch similarity and path length
        """
        # load wordnet on demand
        self._load_wordnet

        try:
            # this is literally crime against humanity to implement query system like this
            # the query pattern lemmas.pos.nn is a HUGE antipattern and should not exist
            word1_entry_list = filter(
                lambda x: x if x.name().startswith(word1) else None,
                wordnet.synsets(word1),
            )
            word2_entry_list = filter(
                lambda x: x if x.name().startswith(word2) else None,
                wordnet.synsets(word2),
            )
        except (IndexError, nltk.corpus.reader.wordnet.WordNetError) as e:
            logger.warning(
                "Could not find words %s and %s from wordnet, error %s", word1, word2, e
            )
            return 0.0, 0.0, 0.0

        word1_entry, word2_entry = self._get_correct_word_pair(
            word1_entry_list, word2_entry_list, word1, word2
        )
        if not all(
            (
                word1_entry,
                word2_entry,
            )
        ):
            logger.warning(
                "Could not find matchig word pair for %s and %s found %s and %s",
                word1,
                word2,
                word1_entry_list,
                word2_entry_list,
            )
            return 0.0, 0.0, 0.0

        logger.debug(
            "Selected wordnet entries word1: %s word2: %s", word1_entry, word2_entry
        )

        wu_palmer = word1_entry.wup_similarity(word2_entry)
        path_similarity = word1_entry.path_similarity(word2_entry)
        lch_similarity = word1_entry.lch_similarity(word2_entry)

        logger.debug(
            "Metrics for words %s and %s: Wu palmer: %s, path length: %s, leacock chordorow: %s",
            word1,
            word2,
            wu_palmer,
            path_similarity,
            lch_similarity,
        )
        return wu_palmer, lch_similarity, path_similarity

    def construct_result_table(self, words: List[str]):
        """
        Construct result table from provided wordlist
        """
        table = pd.DataFrame()
        for word1, word2, relation in self.wordlist:
            web_similarity = self.compute_web_jaccard_similarity(word1, word2)
            wu_palmer, path_length, lch = self.compute_semantic_similarity(word1, word2)

            series = pd.Series(
                [web_similarity, wu_palmer, path_length, lch],
                index=[
                    "Web similarity",
                    "Wu & Palmer",
                    "Path similarity",
                    "Leacock Chordorow",
                ],
                name=f"{word1} <-> {word2}",
            )

            # join new series next to exiting table
            table = pd.concat([table, series], axis=1)

        logger.info("Resulting table:\n%s", table)
