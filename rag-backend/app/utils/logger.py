"""Shared logger factory.

Call ``get_logger(__name__)`` at module top to get a namespaced logger that
writes to stdout with a consistent format. Configured once, idempotently.
"""

import logging
import sys

_CONFIGURED = False
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root = logging.getLogger("rag")
    root.setLevel(logging.INFO)
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    # Namespace everything under "rag" so third-party noise stays separate.
    short = name.split(".")[-1]
    return logging.getLogger(f"rag.{short}")
