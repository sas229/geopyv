import logging
import os
import sys
import platform

class CustomFormatter(logging.Formatter):
    """Function to format the std log output such that INFO logs provide
    just the log message and other levels also provide the level type."""

    # Colours.
    white = '\u001b[37m'
    yellow = '\x1b[38;5;226m'
    red = '\x1b[38;5;196m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'

    def __init__(self, fmt, fmt_INFO):
        super().__init__()
        self.fmt = fmt
        self.fmt_INFO = fmt_INFO
        self.FORMATS = {
            logging.DEBUG: self.white + self.fmt + self.reset,
            logging.INFO: self.white + self.fmt_INFO + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def initialise(level):
    """Function to initialise the log file."""
    # Get platform and define destination for the logging file.
    operating_system = platform.system()
    home_dir = os.path.expanduser( '~' )
    if operating_system == "Windows":
        log_dir = os.path.abspath(os.path.join(home_dir,"AppData/geopyv"))
        isdir = os.path.isdir(log_dir) 
        if isdir == False:
            os.mkdir(log_dir)
        log_file = os.path.abspath(os.path.join(home_dir,"AppData/geopyv/geopyv.log"))
    elif operating_system == "Linux":
        log_dir = os.path.abspath(os.path.join(home_dir,".geopyv"))
        isdir = os.path.isdir(log_dir) 
        if isdir == False:
            os.mkdir(log_dir)
        log_file = os.path.abspath(os.path.join(home_dir,".geopyv/geopyv.log"))

    # Delete log file if already in existence.
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Log format.
    format = (
        '%(levelname)s - '
        '%(name)s - '
        '%(funcName)s - '
        '%(message)s'
    )

    # INFO log format for console output.
    format_INFO = (
        '%(message)s'
    )

    # Basic configuration.
    logger = logging.getLogger(__name__)

    # Output full log.
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(format))
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(CustomFormatter(format, format_INFO))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(ch)

    return

def set_level(level):
    """Function to set the log level after initialisation."""
    log = logging.getLogger(__name__)
    log.setLevel(level)