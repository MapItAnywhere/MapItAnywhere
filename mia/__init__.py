import os, sys

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import logging
from dataclasses import dataclass

@dataclass
class Colors:
    grey: str = "\x1b[38;20m"
    blue: str = "\x1b[34;20m"
    bold_blue: str = "\x1b[34;1m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"


class ColorFormatter(logging.Formatter):

    colors = Colors()
    format = "[%(asctime)s %(name)s %(levelname)s] %(message)s"
    datefmt="%Y-%m-%d %H:%M:%S"

    FORMATS = {
        logging.DEBUG: colors.grey + format + colors.reset,
        logging.INFO: colors.grey + format + colors.reset,
        logging.WARNING: colors.yellow + format + colors.reset,
        logging.ERROR: colors.red + format + colors.reset,
        logging.CRITICAL: colors.bold_red + format + colors.reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt=self.datefmt)
        return formatter.format(record)

formatter = logging.Formatter(
    fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

handler = logging.StreamHandler()
handler.setFormatter(ColorFormatter())
handler.setLevel(logging.INFO)

logger = logging.getLogger("mia")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False