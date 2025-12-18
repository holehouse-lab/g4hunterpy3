"""
Unit and regression test for the g4hunterpy3 package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import g4hunterpy3


def test_g4hunterpy3_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "g4hunterpy3" in sys.modules
