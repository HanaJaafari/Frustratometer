"""Calculates single residue frustration, and mutational frustration of proteins."""


# Add imports here
from .classes import *
from .utils import _path
from . import utils
from . import pfam
from . import pdb
from . import filter
from . import dca
from . import map
from . import align
from . import frustration
from typing import TYPE_CHECKING

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

from . import _version
__version__ = _version.get_versions()['version']

def __getattr__(name: str):
    if name == "optimization":
        mod = import_module(".optimization", __name__)
        globals()["optimization"] = mod  # cache for subsequent lookups
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Optional: make tab-complete show 'optimization'
def __dir__():
    return sorted(list(globals().keys()) + ["optimization"])

# Optional: help static type checkers/IDE
if TYPE_CHECKING:
    from . import optimization
