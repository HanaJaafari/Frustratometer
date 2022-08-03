"""
Unit and regression test for the dca_frustratometer package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import dca_frustratometer


def test_dca_frustratometer_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "dca_frustratometer" in sys.modules
