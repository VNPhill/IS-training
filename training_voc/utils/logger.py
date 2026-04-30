"""
utils/logger.py — Terminal output mirroring.

Redirects sys.stdout so that every print() call writes to BOTH the terminal
and a plain-text log file simultaneously.  No existing print statements need
to change.

Usage:
    from utils.logger import setup_logging
    setup_logging(log_dir='logs/retinanet', filename='train.log')
    # everything printed after this line is also written to the file

The log file is opened in append mode, so resuming a run adds to the same
file rather than overwriting it.  A timestamped header is written each time
the script starts so different sessions are easy to tell apart.
"""

import os
import sys
from datetime import datetime


class _Tee:
    """
    Wraps the current sys.stdout so writes go to both the terminal and a file.

    Attributes:
        terminal : the original sys.stdout
        logfile  : open file handle for the log
    """

    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile  = logfile

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)
        self.logfile.flush()          # flush after every write so the file is
                                      # readable even while the script is running

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        # Some libraries check this; delegate to the real terminal
        return self.terminal.isatty()


def setup_logging(log_dir: str, filename: str = 'run.log') -> str:
    """
    Start mirroring all stdout output to a log file.

    Creates `log_dir` if it does not exist.  Opens the log file in append
    mode and writes a timestamped session header before handing control back.

    Args:
        log_dir  : directory where the log file will be saved
                   (e.g. 'logs/retinanet')
        filename : name of the log file  (default: 'run.log')

    Returns:
        Full path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    logfile = open(log_path, 'a', encoding='utf-8')

    # Write a visible separator so multiple sessions inside one file are clear
    header = (
        f"\n{'='*65}\n"
        f"  Session started : {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}\n"
        f"  Log file        : {log_path}\n"
        f"{'='*65}\n"
    )
    logfile.write(header)
    logfile.flush()

    sys.stdout = _Tee(sys.__stdout__, logfile)

    # Also mirror stderr so TF warnings / stack traces are captured
    sys.stderr = _Tee(sys.__stderr__, logfile)

    # Print the header to terminal too (stdout is now the Tee, so one write
    # covers both)
    print(header, end='')

    return log_path


def teardown_logging():
    """
    Restore sys.stdout and sys.stderr to their originals and close the file.
    Call this at the very end of a script if you need clean shutdown.
    Omitting it is safe — Python will close the file on interpreter exit.
    """
    if isinstance(sys.stdout, _Tee):
        sys.stdout.logfile.close()
        sys.stdout = sys.__stdout__
    if isinstance(sys.stderr, _Tee):
        sys.stderr = sys.__stderr__
        
        