import nibabel as nib
import os

import utils

img = nib.load(os.path.join(utils.OUTPUT_DIR, "201.nii.gz"))
