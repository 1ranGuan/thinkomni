import logging
import sys
from typing import Union
import os
from datetime import datetime
import socket
import warnings

__all__ = ["setup_logger"]

def _clear_handlers(logger: logging.Logger) -> None:
    """Remove handlers that might already be attached to the logger to avoid
    duplicate log entries when `setup_logger` is called multiple times.
    """
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

# first time logger only
# other files use `logger = logging.getLogger(__name__)`
def setup_logger(
    name: str = __name__,
    *,
    log_dir: str = "logs",
    log_file: str = None,
    level: Union[int, str] = logging.INFO,
    stream: bool = True,
    warn_for_once: bool = True,
) -> logging.Logger:
    """
    Return a configured :pyclass:`logging.Logger`.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Make sure we don't accumulate handlers on repeated calls
    _clear_handlers(logger)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # make dirs if not exist
    os.makedirs(log_dir, exist_ok=True)
    
    if log_file is None:
        # use time stamp & hostname as log file name
        log_file = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{socket.gethostname()}.log"
    else:
        log_file = log_file.replace(".log", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{socket.gethostname()}.log")
        
    log_file = os.path.join(log_dir, log_file)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    if stream:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
    
    if warn_for_once:
        # Suppress warnings about duplicate log entries
        logging.captureWarnings(True)
        warnings.filterwarnings("once", category=UserWarning)

    return logger
