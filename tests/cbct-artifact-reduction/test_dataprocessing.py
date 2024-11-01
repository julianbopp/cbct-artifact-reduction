import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

import cbct_artifact_reduction.dataprocessing as df
from cbct_artifact_reduction.dataprocessing import (
    filename_without_extension,
    nifti_vol_to_frames,
    resize_single_file,
    single_nifti_to_numpy,
    tif_to_nifti,
)
from cbct_artifact_reduction.utils import ROOT_DIR


def test_datafolder():
    """Test creation of data folder. If this doesn't work, the rest of the tests will fail aswell."""
    data_folder = df.DataFolder("data", ".nii.gz")


def test_list_files():
    """Test listing of files in the data folder."""
    data_folder_path = os.path.join(ROOT_DIR, "sample_data")
    data_folder = df.DataFolder(data_folder_path, ".nii.gz")
    data_list = data_folder.list_filenames()
    assert len(data_list) == 5
    assert "80_0.nii.gz" in data_list
    assert "80_1.nii.gz" in data_list
    assert "80_2.nii.gz" in data_list
    assert "80_3.nii.gz" in data_list
    assert "80_4.nii.gz" in data_list


def test_single_nifti_to_numpy():
    """Test conversion of a single nifti file to numpy array."""
    data_folder_path = os.path.join(ROOT_DIR, "sample_data")
    nifti_data_folder = df.NiftiDataFolder(data_folder_path)
    nifti_path = os.path.join(data_folder_path, "80_0.nii.gz")
    nib_object = nib.Nifti1Image.from_filename(nifti_path)
    old_shape = nib_object.shape
    nifti_nparray = single_nifti_to_numpy(nifti_path)
    assert nifti_nparray.shape == old_shape


def test_resize_single_file(tmp_path):
    """Test if resized file has the correct shape."""
    data_folder_path = os.path.join(ROOT_DIR, "sample_data")
    nifti_data_folder = df.NiftiDataFolder(data_folder_path)
    nifti_path = os.path.join(data_folder_path, "80_4.nii.gz")
    resized_nifti_path = os.path.join(tmp_path, "resized.nii.gz")
    new_shape = (5, 5)
    resize_single_file(nifti_path, new_shape, resized_nifti_path)
    resized_nifti = nib.Nifti1Image.from_filename(resized_nifti_path)
    assert resized_nifti.shape == new_shape


def test_resize_all_files(tmp_path):
    """Test if all files in the folder are resized."""
    data_folder_path = os.path.join(ROOT_DIR, "sample_data")
    nifti_data_folder = df.NiftiDataFolder(data_folder_path)
    new_shape = (3, 3)
    nifti_data_folder.resize_all_files(tmp_path, new_shape, overwrite_files=True)
    resized_files = os.listdir(tmp_path)
    assert len(resized_files) == 5
    for file in resized_files:
        resized_nifti = nib.Nifti1Image.from_filename(os.path.join(tmp_path, file))
        assert resized_nifti.shape == new_shape


def test_filename_without_extension():
    assert filename_without_extension("test.nii.gz") == "test"
    assert filename_without_extension("test.nii") == "test"
    assert filename_without_extension("test.gz") == "test"
    assert filename_without_extension("test") == "test"


def test_nifti_vol_to_frames(tmp_path):
    numpy_data = np.random.rand(10, 10, 5)
    shape = numpy_data.shape
    nib_data = nib.Nifti1Image(numpy_data, np.eye(4))
    nib.save(nib_data, tmp_path / "test.nii.gz")
    nifti_vol_to_frames(tmp_path / "test.nii.gz", tmp_path)

    for i in range(shape[2]):
        assert (tmp_path / f"test_{i}.nii.gz").exists()
        assert np.allclose(
            nib.Nifti1Image.load(tmp_path / f"test_{i}.nii.gz").get_fdata(),
            numpy_data[:, :, i],
        )


def test_tif_to_nifti_on_non_existing_input():
    with pytest.raises(AssertionError):
        tif_to_nifti("_______non_existing______.tif", "output.nii")
