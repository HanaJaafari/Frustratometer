"""Calculates single residue frustration, and mutational frustration of proteins."""

# Add imports here
from .utils import *
from .download import *
from .dca_frustratometer import *

#Define a path
_path = Path(__file__).parent.absolute()

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions

from . import _version
__version__ = _version.get_versions()['version']
