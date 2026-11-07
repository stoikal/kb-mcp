"""Logging setup. Logs go to stderr (stdout carries the stdio protocol) and, optionally,
a rotating file. Bearer tokens and database passwords are redacted from every record."""

import logging
import logging.handlers
import os
import re
import sys

_BEARER = re.compile(r"(?i)(bearer\s+)[^\s'\"]+")
_URL_PASSWORD = re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^:/@\s]+:)[^@\s]+@")
_TOKEN_ENV = re.compile(r"(?i)(RAGMCP_(?:ADMIN|READER)_TOKENS\s*[=:]\s*)\S+")


def redact(text: str) -> str:
    text = _BEARER.sub(r"\1[redacted]", text)
    text = _URL_PASSWORD.sub(r"\1[redacted]@", text)
    return _TOKEN_ENV.sub(r"\1[redacted]", text)


class RedactFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info:  # fold the traceback in so it is redacted too
            record.msg = redact(f"{record.msg}\n{logging.Formatter().formatException(record.exc_info)}")
            record.exc_info = None
        return True


def setup_logging(level: str | None = None, log_file: str | None = None) -> None:
    level = (level or os.environ.get("RAGMCP_LOG_LEVEL", "INFO")).upper()
    log_file = log_file if log_file is not None else os.environ.get("RAGMCP_LOG_FILE", "")
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.handlers.RotatingFileHandler(log_file, maxBytes=5_000_000, backupCount=5))
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(RedactFilter())
        root.addHandler(h)
    root.setLevel(getattr(logging, level, logging.INFO))
    # uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
