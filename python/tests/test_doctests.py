"""Every example in the package's docstrings runs and prints what it says."""

import doctest
import importlib

import pytest


@pytest.mark.parametrize("module", ["qumo", "qumo.circuit", "qumo.sim", "qumo.compiler", "qumo.algorithms",
                                    "qumo.qec", "qumo.errors"])
def test_docstring_examples(module):
    result = doctest.testmod(importlib.import_module(module), optionflags=doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0
    if module != "qumo.errors":
        assert result.attempted > 0
