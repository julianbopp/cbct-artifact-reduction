import ants
import os
from utils import (
    OUTPUT_DIR,
    nifti_to_numpy,
)
import nibabel as nib
from skimage import transform as skTrans

type = "fixed"
affine_registered_path = os.path.join(OUTPUT_DIR, "rotated", f"1.nii.gz")
control_path = os.path.join(OUTPUT_DIR, "rotated", "10.nii.gz")

affine_registered_numpy = nifti_to_numpy(affine_registered_path)
control_numpy = nifti_to_numpy(control_path)

control_numpy = skTrans.resize(
    control_numpy,
    affine_registered_numpy.shape,
)
diff_numpy = control_numpy - affine_registered_numpy
diff_nifti = nib.Nifti1Image(diff_numpy, affine=None)

nib.save(diff_nifti, os.path.join(OUTPUT_DIR, "registered", f"control_diff.nii.gz"))
