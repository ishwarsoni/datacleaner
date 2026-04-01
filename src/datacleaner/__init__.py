"""datacleaner package."""

from .core import clean
from .analysis import analyze
from .target_handling import handle_target

__version__ = "0.1.0"

__all__ = ["clean", "analyze", "handle_target"]
