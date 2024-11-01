import numpy as np
import pytest

import cbct_artifact_reduction.utils as utils


def test_get_scanner_from_num():
    assert utils.get_scanner_from_num(1) == (1, "axeos", "ti", 3, "small")


def test_min_max_normalize():
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert (
        utils.min_max_normalize(data) == np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ).all()
