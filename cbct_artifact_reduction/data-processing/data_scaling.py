import os
from utils import OUTPUT_DIR, min_max_normalize, nifti_to_numpy, get_scanner_from_num
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as skTrans

processed_data = os.path.join(OUTPUT_DIR, "rotated")
list_scans = sorted([f for f in os.listdir(processed_data) if f.endswith(".nii.gz")])


for scan in list_scans:
    _, _, _, num_of_transplants, _ = get_scanner_from_num(int(scan[:-7]))
    if num_of_transplants != 0:
        print(f"{scan} is a transplant scan. Skipping.")
        continue
    else:
        print(f"{scan} is a control scan. Processing.")
    if os.path.exists(os.path.join(OUTPUT_DIR, "scaled", scan)):
        print(f"{scan} already exists. Skipping.")
        continue
    scan_path = os.path.join(processed_data, scan)
    scan_nparray = nifti_to_numpy(scan_path)
    print(f"Image shape: {scan_nparray.shape}")

    result1 = skTrans.resize(
        scan_nparray, (256, 256, scan_nparray.shape[2]), preserve_range=True
    )

    print(f"Result shape: {result1.shape}")
    scaledNiftiImage = nib.Nifti1Image(result1, affine=None)

    nib.save(scaledNiftiImage, os.path.join(OUTPUT_DIR, "scaled", scan))
