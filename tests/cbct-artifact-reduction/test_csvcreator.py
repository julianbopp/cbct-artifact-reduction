import cbct_artifact_reduction.csvcreator as cc
import pytest


def testGetAllControlIDs():
    controlIDs = cc.getAllControlIDs()
    control_list = [f"{f}" for f in range(0, 401) if f % 10 == 0]

    assert controlIDs == control_list


def testCreateSliceMaskCSV(tmp_path):
    slices = ["slice1.nii.gz", "slice2.nii.gz", "slice3.nii.gz"]
    masks = ["mask1.nii.gz", "mask2.nii.gz", "mask3.nii.gz"]

    cc.createSliceMaskCSV(slices, masks, tmp_path / "test.csv")

    with open(tmp_path / "test.csv", "r") as file:
        csv_content = file.read()

        assert (
            csv_content
            == "slice,mask\nslice1.nii.gz,mask1.nii.gz\nslice2.nii.gz,mask2.nii.gz\nslice3.nii.gz,mask3.nii.gz\n"
        )


def testCreateSliceMaskCSVDifferentLengths(tmp_path):
    slices = ["slice1.nii.gz", "slice2.nii.gz", "slice3.nii.gz"]
    masks = ["mask1.nii.gz", "mask2.nii.gz"]
    with pytest.raises(AssertionError):
        cc.createSliceMaskCSV(slices, masks, tmp_path / "test.csv")
