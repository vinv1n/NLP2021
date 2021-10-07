import logging
import os
import requests
import json

from typing import Optional, Iterator, Tuple, Any, Dict, List


logger = logging.getLogger(__name__)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
CX_ID = os.environ.get("GOOGLE_CX_ID", "")


def _permutate_words(word1: str, word2: str) -> Iterator[Tuple[str, str]]:
    for param1, param2 in zip((word1, word1, "",), (word2, "", word2,)):
        yield f"{param1} {param2}"


class WebSimilarity:

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Custom user agent"
            }
        )

    def _get_search_pages(self, word1: str, word2: str) -> List[int]:
        if not GOOGLE_API_KEY:
            logger.critical("Could not get page count, missing key")
            return []

        request_url_base = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={CX_ID}:omuauf_lfve&q"

        results = []
        for query in _permutate_words(word1, word2):
            request_url = f"{request_url_base}={query}"
            response = self.session.get(request_url)
            if response.status_code != 200:
                logger.critical("Return code %s from google api, exiting", response.status_code)
                return []

            try:
                result = response.json()
            except (ValueError, IOError) as e:
                logger.warning("Response from url %s was not json, error %s", request_url, e)
                continue

            logger.info("Recieved response with content %s from google search api %s", request_url, result)

            # page results not yet computed as I am not sure what they contant
            results.append(result)

        return results

    def web_similarity(self, word1: str, word2: str):
        lengths = self._get_search_pages(word1, word2)
        if not lengths:
            logger.warning("No results with words %s and %s", word1, word2)
            return 0.0

        if len(lengths) != 3:
            logger.warning("Incorrect amount of results returned, %s should have been 3". len(lengths))
            return 0.0

        try:
            return lengths[0] / sum(lengths)
        except ZeroDivisionError:
            logger.warning("Sum of page lenghts is 0, cannot compute similarity")
            return 0.0