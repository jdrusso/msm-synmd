"""
Unit and regression test for the synthetic_md package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import synthetic_md


def test_synthetic_md_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "synthetic_md" in sys.modules
