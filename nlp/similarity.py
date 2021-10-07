import logging
import os
import requests
import json

from typing import Optional, Iterator, Tuple, Any, Dict


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

    def _get_search_pages(self, word1: str, word2: str) -> Optional[int]:
        if not GOOGLE_API_KEY:
            logger.critical("Could not get page count, missing key")
            return None

        request_url_base = f"https://www.googleapis.com/customsearch/v1?key={GOOGLE_API_KEY}&cx={CX_ID}:omuauf_lfve&q"

        results = []
        for query in _permutate_words(word1, word2):
            request_url = f"{request_url_base}={query}"
            response = self.session.get(request_url)
            if response.status_code != 200:
                logger.critical("Return code %s from google api, exiting", response.status_code)
                return None

            try:
                result = response.json()
            except (ValueError, IOError) as e:
                logger.warning("Response from url %s was not json, error %s", request_url, e)
                continue

            logger.info("Recieved response with content %s from google search api %s", request_url, result)

            # page results not yet computed as I am not sure what they contant
            results.append(result)

    def web_similarity(self, word1: str, word: str):
        pass
