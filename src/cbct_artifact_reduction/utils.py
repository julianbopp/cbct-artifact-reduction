import os
import tarfile
import nibabel as nib
from brainglobe_utils.IO.image import load, save
import numpy as np

# Save project root directory
ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
DATA_DIR = os.path.join(ROOT_DIR, "data")
CODE_DIR = os.path.join(ROOT_DIR, "code")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
print(ROOT_DIR)


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
    # Load the data
    data = load.load_img_stack(input_path, 1, 1, 1)
    # Save the data
    save.to_nii(data, output_path)


def min_max_normalize(img):
    # Function used to normalize image before visualization with vizdom

    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


def nifti_to_numpy(nifti_path):
    assert os.path.exists(nifti_path), f"{nifti_path} does not exist"
    image = nib.Nifti1Image.from_filename(nifti_path)
    np_array = np.array(image.dataobj)
    return np_array
