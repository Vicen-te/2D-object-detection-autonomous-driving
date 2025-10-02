# config_logging.py
import logging
import sys

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%d-%m %H:%M:%S",
        stream=sys.stdout
    )

logger = logging.getLogger()