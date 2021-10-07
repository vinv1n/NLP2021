import logging
import os
import requests
import json
import pprint

from typing import Optional, Iterator, Tuple, Any, Dict, List


logger = logging.getLogger(__name__)

"""
I don't explain here how to get these, please see links below

https://developers.google.com/custom-search/v1/using_rest
https://developers.google.com/custom-search/v1/introduction#identify_your_application_to_google_with_api_key
https://stackoverflow.com/questions/4082966/what-are-the-alternatives-now-that-the-google-web-search-api-has-been-deprecated
"""

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CX_ID = os.environ.get("GOOGLE_CX_ID", "")


def _permutate_words(word1: str, word2: str) -> Iterator[Tuple[str, str]]:
    for param1, param2 in zip((word1, word1, "",), (word2, "", word2,)):
        yield f"{param1},{param2}"


class WebSimilarity:

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:40.0) Gecko/20100101 Firefox/40.0"
            }
        )

    def _get_search_pages(self, word1: str, word2: str) -> List[int]:
        if not GOOGLE_API_KEY:
            logger.critical("Could not get page count, missing key")
            return []

        request_url_base = f"https://customsearch.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={CX_ID}&q"

        results = []
        for query in _permutate_words(word1, word2):
            request_url = f"{request_url_base}={query}"
            response = self.session.get(request_url)
            if response.status_code != 200:
                logger.critical("Return code %s from google api %s", response.status_code, response.text)
                return []

            try:
                result = response.json()
            except (
                ValueError,
                IOError,
                json.decoder.JSONDecodeError
            ) as e:
                logger.warning("Response from url %s was not json, error %s", request_url, e)
                continue

            logger.info("Recieved response with content %s from google search api", request_url)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Response from api %s", pprint.pformat(result))

            try:
                search_results = int(
                    result.get("searchInformation", {}).get("totalResults", "0")
                )
            except ValueError:
                search_results = 0

            logger.info("With query %s found %s search results", query, search_results)
            results.append(search_results)

        return results

    def web_similarity(self, word1: str, word2: str):
        logger.info("Using words %s and %s for the search", word1, word2)

        lengths = self._get_search_pages(word1, word2)
        if not lengths:
            logger.warning("No results with words %s and %s", word1, word2)
            return 0.0

        if (amount := len(lengths)) != 3:
            logger.warning("Incorrect amount of results returned, %s should have been 3", amount)
            return 0.0

        try:
            similarity = lengths[0] / sum(lengths)
        except ZeroDivisionError:
            logger.warning("Sum of page lenghts is 0, cannot compute similarity")
            return 0.0

        logger.info("Similarity between word %s and %s is %s", word1, word2, similarity)
        return similarity
