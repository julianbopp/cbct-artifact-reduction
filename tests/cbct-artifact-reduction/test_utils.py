import cbct_artifact_reduction.utils as utils


def test_get_scanner_from_num():
    assert utils.get_scanner_from_num(1) == (1, "axeos", "ti", 3, "small")


def testGetAllControlIDs():
    controlIDs = utils.getAllControlIDs()
    control_list = [f"{f}" for f in range(0, 401) if f % 10 == 0]

    assert controlIDs == control_list
