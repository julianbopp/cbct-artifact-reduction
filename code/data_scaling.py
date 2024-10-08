import os
from utils import get_scanner_from_num, ROOT_DIR, DATA_DIR, OUTPUT_DIR
import nibabel as nib
import numpy as np

import skimage.transform as skTrans


processed_data = os.path.join(OUTPUT_DIR, "data")
list_scans = os.listdir(processed_data)

for scan in list_scans:
    scan_path = os.path.join(processed_data, scan)
    img = nib.load(scan_path).get_fdata()  # type: ignore
    print(img.shape)
    result1 = skTrans.resize(
        img, (256, 256, img.shape[2]), order=1, preserve_range=True
    )
    nibObject = nib.Nifti1Image(result1, np.eye(4))
    nib.save(nibObject, os.path.join(OUTPUT_DIR, "scaled", scan))
