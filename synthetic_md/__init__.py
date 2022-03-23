"""A short description of the project (less than one line)."""

# Add imports here
from .synthetic_md import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
