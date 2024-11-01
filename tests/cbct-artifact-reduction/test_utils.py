import numpy as np
import pytest

import cbct_artifact_reduction.utils as utils


def test_get_scanner_from_num():
    assert utils.get_scanner_from_num(1) == (1, "axeos", "ti", 3, "small")
