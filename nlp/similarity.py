import logging
import os
import requests_cache
import json
import pprint
import nltk
import re
import pandas as pd
import csv
import random
import time
import string

from fuzzywuzzy import fuzz
from datetime import timedelta
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from queue import SimpleQueue
from queue import Empty
from pathlib import Path
from itertools import product, combinations
from collections import defaultdict

from pathlib import Path
from typing import Optional, Iterator, Tuple, Any, Dict, List, DefaultDict

from nltk.corpus import wordnet
from multiprocessing import Lock


logger = logging.getLogger(__name__)

# setup seed for random based on current time
random.seed(time.time())

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
        # avoid spamming api as little as possible
        self.session = requests_cache.CachedSession(
            "api_cache",
            backend="sqlite",
            # invalidate cache after two weeks
            # this has to be done as rate limit of free queries in google search api
            # and parsing html with selenium is just pure stupid
            expire_after=timedelta(days=14),
            allowable_codes=(200,),
            stale_if_error=True,
        )
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:40.0) Gecko/20100101 Firefox/40.0"
            }
        )

        self._loaded = False
        self._stopwords = None
        self._punkt = None

        self.stemmer = PorterStemmer()

        if wordlist:
            self.wordlist = self._load_wordlist(Path(wordlist))
        else:
            self.wordlist = []

    @staticmethod
    def _dump_search_results(entry: Dict[str, Any], word1: str, word2: str()) -> None:
        filepath = Path("samples", f"{word1}-{word2}.json")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write(json.dumps(entry, indent=4))

    @property
    def _load_wordnet(self) -> None:
        # this is kind stupid feature as I was not able to find
        # to check if wordnet is already downloaded
        if not self._loaded:
            self._stopwords = nltk.download("stopwords")
            self._punkt = nltk.download("punkt")
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

    def _fetch_search_api_result(self, query: str) -> Dict[str, Any]:
        request_url = f"{BASE_SEARCH_URL}&cx={CX_ID}&q={query}"
        response = self.session.get(request_url)
        if response.status_code != 200:
            logger.critical(
                "Return code %s from google api %s",
                response.status_code,
                response.text,
            )
            return {}

        try:
            result = response.json()
        except (ValueError, IOError, json.decoder.JSONDecodeError) as e:
            logger.warning(
                "Response from url %s was not json, error %s", request_url, e
            )
            return {}

        logger.info(
            "Recieved response with content %s from google search api", request_url
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Response from api %s", pprint.pformat(result))

        if DEBUG:
            self._dump_search_results(result)

        return result

    def _get_search_pages(self, word1: str, word2: str) -> List[int]:
        if not GOOGLE_API_KEY:
            logger.critical("Could not get page count, missing key")
            return []

        results = []
        for query in _permutate_words(word1, word2):
            result = self._fetch_search_api_result(query)
            if not result:
                continue

            results.append(result)

        return results

    def compute_web_jaccard_similarity(self, word1: str, word2: str) -> float:
        """
        Wrapper for task 1
        """
        return self._compute_web_jaccard_similarity_by_search_results(word1, word2)

    @staticmethod
    def _construct_url_from_openseach_template(
        template: str, entry: Dict[str, Any]
    ) -> str:
        """
        Helper function to construct urls from opensearch templates
        """

        try:
            start_value = int(entry.get("startIndex", "0"))
            if start_value > 100:
                logger.info("Search limit encountered, exiting")
                return ""

        except ValueError:
            logger.warning("Could not convert %s to int", value)
            return ""

        # add api key to the dict
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

        url_parts.insert(len(url_parts) - 1, f"key={GOOGLE_API_KEY}")
        return "&".join(url_parts)

    def fetch_search_snippets(
        self, word: str, limit: int = -1, clean: bool = False
    ) -> List[str]:
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
        logger.info("Fetching snippets for search word %s", word)

        # setup queue to keep next pages in order
        queue = SimpleQueue()
        lock = Lock()

        # add initial search pages to the queue
        response = self._fetch_search_api_result(word)
        if response:
            queue.put(response, block=False)

        counter = 0
        snippets: List[str] = []
        while queue.qsize() > 0:
            if limit != -1 and counter > limit:
                logger.info("Search limit %s reached exiting", limit)
                break

            try:
                search_page = queue.get(timeout=10, block=False)
            except Empty:
                break

            if not search_page:
                logger.info("No more items in a queue exiting")
                break

            if next_page := search_page.get("queries", {}).get("nextPage", []):
                # for some reason this is a list, not sure why
                template = search_page.get("url", {}).get("template", "")
                for pages in next_page:
                    next_url = self._construct_url_from_openseach_template(
                        template, pages
                    )
                    if not next_url:
                        logger.warning("Could not construct url from entry %s", pages)
                        break

                    logger.info("Next url is %s", next_url)
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

                    # add new entry to queue and keep index
                    queue.put(content, block=False)

            for item in search_page.get("items", []):
                # items is a list of dicts
                snippet = item.get("snippet", "")
                if not snippet:
                    logger.warning("Search result entry without snippet, ignoring")
                    continue

                counter += 1
                if clean:
                    tokens = [
                        self.stemmer.stem(x)
                        for x in word_tokenize(snippet.lower())
                        if all((x not in string.punctuation, x.isalpha()))
                    ]
                    snippets.append(" ".join(tokens))
                else:
                    snippets.append(snippet)

        return snippets

    def _get_all_words(self, words: List[str]) -> List[str]:
        """
        ravel all nested lists
        """
        result = []
        for word_array in words:
            result.extend(word_array)
        return result

    def sim_snippet1(self, *args) -> pd.DataFrame:
        """
        In python styleguide only class names are in CamelCase, funcs are _always_ in lowecase :)

        However, this is task 5 implementation
        """
        self._load_wordnet
        if len(args) > 0:
            wordlist = list(combinations(args, 2))
        else:
            # remove types from the list as we do not need them
            wordlist = [(x, y) for x, y, _ in self.wordlist]

        similarities = pd.DataFrame()
        for word1, word2 in wordlist:
            snippets1 = self.fetch_search_snippets(word1, limit=5)
            snippets2 = self.fetch_search_snippets(word2, limit=5)
            if not all((snippets1, snippets2)):
                logger.warning("Could not fetch snippets for words %s %s", word1, word2)
                continue

            # get tokens for snippets
            snippet1_tokens = self._get_all_words(
                [word_tokenize(x.lower()) for x in snippets1]
            )
            snippet2_tokens = self._get_all_words(
                [word_tokenize(x.lower()) for x in snippets2]
            )

            # and then stemming, note that noise removal is not really needed
            # as these are quite clean samples anyway
            stem_word1 = [
                self.stemmer.stem(x)
                for x in snippet1_tokens
                if all((x not in string.punctuation, x.isalpha()))
            ]
            stem_word2 = [
                self.stemmer.stem(x)
                for x in snippet2_tokens
                if all((x not in string.punctuation, x.isalpha()))
            ]

            try:
                similarity = len(set(stem_word1).intersection(set(stem_word2))) / (
                    len(stem_word1) + len(stem_word1)
                )
            except ZeroDivisionError:
                logger.warning("Length of all words in snippets is 0")
                continue

            (
                wu_palmer,
                lch_similarity,
                path_similarity,
            ) = self.compute_semantic_similarity(word1, word2)
            frame = pd.Series(
                [similarity, wu_palmer, lch_similarity, path_similarity],
                index=[
                    "Task 5 similarity",
                    "Wu Palmer similarity",
                    "LCH similarity",
                    "Path similarity",
                ],
                name=f"{word1} and {word2}",
            )
            similarities = pd.concat([similarities, frame], axis=1)

        logger.info("Resulting similarities %s", similarities)
        return similarities

    def sim_snippet2(
        self, wordlist: List[Tuple[str, str, int]], limit: int = 0, clean: bool = False
    ) -> pd.DataFrame:
        """
        Actual implementation of task 6. as this is needed again on task 7.
        with different params we need to do this
        """

        def compute_similarity_of_words(word1, word2):
            return fuzz.token_set_ratio(word1, word2)

        results = []
        for word1, word2, _ in wordlist:
            snippets1 = self.fetch_search_snippets(word1, limit=limit, clean=clean)
            snippets2 = self.fetch_search_snippets(word2, limit=limit, clean=clean)

            # we assume that every snippet in a set is unique
            shared = len(
                set([x.lower() for x in snippets1]).intersection(
                    set([x.lower() for x in snippets2])
                )
            )

            doc1 = "".join(snippets1)
            doc2 = "".join(snippets2)
            overlapping = compute_similarity_of_words(doc1, doc2)

            logger.info(
                "Amount of shared snippets between word %s and word %s is %s and overlapping snippets %s",
                word1,
                word2,
                shared,
                overlapping,
            )
            results.append((f"{word1} - {word2}", shared, overlapping))

        frame = pd.DataFrame(
            results, columns=["words", "shared snippets", "overlapping token ratio"]
        )
        return frame

    def execute_sim_snippet2(self) -> pd.DataFrame:
        """
        Implementation of Task 6.
        """
        # group (index 2) value 2 means that words have opposite meaning
        words = list(filter(lambda x: x if x[2] == 2 else None, self.wordlist))

        # take max 3 pairs from wordlist
        frame = self.sim_snippet2(words[:3], limit=10, clean=True)
        logger.info("Task 6 results are %s", frame)
        return frame

    def compare_tasks(self, *args) -> pd.DataFrame:
        """
        Task 7 implementation
        """
        if len(args) > 0:
            wordlist = list(combinations(args, 2))
        else:
            wordlist = self.wordlist

        frame = self.sim_snippet2(wordlist, limit=5, clean=True)
        logger.info("Result for task 7 is %s", frame)
        return frame

    def _compute_web_jaccard_similarity_by_search_results(
        self, word1: str, word2: str
    ) -> float:
        """
        Implementation for task 1
        """
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
        Task 3. implementation

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

    def get_correlation_between_words(self):
        """
        Implementation of task 2
        """
        results = defaultdict(list)
        for word1, word2, label in self.wordlist:
            if label == 1:
                name = "synonym"
            elif label == 2:
                name = "antonym"
            else:
                name = "hyponym"

            logger.info(
                "Computing WebJaccard similarity for words %s %s classification %s",
                word1,
                word2,
                name,
            )
            similarity = self.compute_web_jaccard_similarity(word1, word2)
            results[name].append((similarity, f"{word1} - {word2}"))

        for wordtype, similarities in results.items():
            entry = pd.Series(
                [x[0] for x in similarities], index=[x[1] for x in similarities]
            )
            logger.info("Similarities for word type %s are %s", wordtype, entry)
            logger.info(
                "Mean for similarities is %s and std %s", entry.mean(), entry.std()
            )

    def construct_result_table(self, words: List[str]):
        """
        Construct result table from provided wordlist, task 3. wrapper
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
        return table

    def compute_correlation_with_annotated_data(
        self, annotated_data_path: str
    ) -> float:
        """
        Implementation of task 8.
        """
        data_path = Path(annotated_data_path)
        if not data_path.exists():
            logger.critical(
                "Path %s to the annotated data csv does not exist", data_path.as_posix()
            )
            return 0.0

        data = []
        with open(data_path, "r") as fd:
            reader = csv.reader(fd, delimiter=";", quotechar="|")
            for row in reader:
                data.append(tuple(row))

        if not data:
            logger.critical("Empty csv provided")
            return 0.0

        logger.info("Computing web jaccard similarity and comparing 10 samples")

        results = []
        # take 10 random samples from the data helps to reduce rate limit
        for word1, word2, similarity in data:
            logger.info(
                "Fetching WebJaccard similarity for words %s and %s", word1, word2
            )
            web_jaccard = self.compute_web_jaccard_similarity(word1, word2)
            results.append((float(web_jaccard), float(similarity)))

        if not results:
            logger.warning("Similarity results are empty, this should not happen")
            return 0.0

        # make pd series out of the results
        web_jaccard_similarity = pd.Series([x[0] for x in results])
        human_determined_similarity = pd.Series([x[1] for x in results])

        correlation = web_jaccard_similarity.corr(human_determined_similarity)
        logger.info(
            "Correlation between human determined similarities and WebJaccard similaries is %s",
            correlation,
        )
