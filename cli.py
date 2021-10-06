import logging


logging.basicConfig(
    format="%(asctime)s:%(name)s: %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    logger.info("%s", "Hello World")
