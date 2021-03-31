"""otmarkov module."""
from .MarkovChain import MarkovChain
from .MarkovChainRandomVector import MarkovChainRandomVector
from .MarkovProcess import MarkovProcess
from .MarkovChainResult import MarkovChainResult

__all__ = [
    "MarkovChain",
    "MarkovChainRandomVector",
    "MarkovProcess",
    "MarkovChainResult"
]
__version__ = "0.1"
