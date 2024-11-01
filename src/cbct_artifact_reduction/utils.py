import mimetypes
import os
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

# from brainglobe_utils.IO.image import load, save

# Save project root directory
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CODE_DIR = os.path.join(ROOT_DIR, "code")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
FRAME_DIR = os.path.join(ROOT_DIR, "output", "frames")


def get_scanner_from_num(num: int):
    orig_num = num
    scanner = ""
    material = ""
    implants = ""
    fov = ""

    if ((num - 1) // 20) % 4 == 0:
        scanner = "axeos"
    elif ((num - 1) // 20) % 4 == 1:
        scanner = "accuitomo"
    elif ((num - 1) // 20) % 4 == 2:
        scanner = "planmeca"
    elif ((num - 1) // 20) % 4 == 3:
        scanner = "x800"

    if num % 10 == 0:
        material = ""
        implants = 0
    elif (num % 10) % 3 == 1:
        implants = 3
    elif (num % 10) % 3 == 2:
        implants = 2
    elif (num % 10) % 3 == 0:
        implants = 1

    if ((num - 1) % 10) // 3 == 0:
        material = "ti"
    elif ((num - 1) % 10) // 3 == 1:
        material = "tizr"
    elif ((num - 1) % 10) // 3 == 2:
        material = "zr"

    if ((num - 1) // 10) % 2 == 0:
        fov = "small"
    elif ((num - 1) // 10) % 2 == 1:
        fov = "large"

    return orig_num, scanner, material, implants, fov


def extract_tar_gz(tar_gz_path: str, output_path: str):
    file = tarfile.open(tar_gz_path)
    file.extractall(output_path)
    file.close()


def tif_to_nifti(input_path: str, output_path: str):
    assert os.path.exists(input_path), f"input path {input_path} does not exist"
    assert os.path.exists(output_path), f"output path {output_path} does already exist"

    # Load the data
    data = nib.nifti1.Nifti1Image.from_filename(input_path)
    # Save the data
    nib.nifti1.Nifti1Image.to_filename(data, output_path)


def min_max_normalize(img):
    # Function used to normalize image before visualization with vizdom

    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def nifti_to_numpy(nifti_path: str) -> NDArray[np.float64]:
    assert os.path.exists(nifti_path), f"{nifti_path} does not exist"
    image = nib.Nifti1Image.from_filename(nifti_path)
    np_array = np.array(image.dataobj)
    return np_array


def nifti_vol_to_frames(nifti_path: str, output_dir: str, overwrite=False):
    """Extract frames from a 3d nifti volume and save them as individual 2d nifti files.
    Input dimensions would be NxMxT, where T is the number of frames."""

    assert os.path.exists(nifti_path), f"{nifti_path} does not exist"
    assert os.path.exists(output_dir), f"{output_dir} does not exist"
    image = nib.Nifti1Image.from_filename(nifti_path)
    np_array = np.array(image.dataobj)

    base_filename = filename_without_extension(os.path.basename(nifti_path))
    for i in range(np_array.shape[2]):
        if not overwrite and os.path.exists(
            os.path.join(output_dir, f"{base_filename}_{i}.nii.gz")
        ):
            print(f"Skipping {base_filename}_{i}.nii.gz as it already exists")
            continue
        frame = np_array[:, :, i]
        nib_frame = nib.Nifti1Image(frame, image.affine)
        nib.save(nib_frame, os.path.join(output_dir, f"{base_filename}_{i}.nii.gz"))


def guess_extensions(filename: str):
    mimetypes.add_type("image/nifti", ".nii")
    mimetypes.add_type("file/archive", ".gz")
    return [s for s in Path(filename).suffixes if s in mimetypes.types_map]


def filename_without_extension(filename: str):
    extensions = guess_extensions(filename)

    for i in extensions:
        filename = filename.replace(i, "")

    return filename
