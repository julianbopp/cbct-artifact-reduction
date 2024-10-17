from tarfile import data_filter
import nibabel as nib
import os
import matplotlib.pyplot as plt
import utils
import numpy as np

path = os.path.join(utils.OUTPUT_DIR, "data", "1.nii.gz")
np_array = utils.nifti_to_numpy(path)
nibCorrected = np.rot90(np.swapaxes(np_array, 0, 2), 3)
nib.save(
    nib.Nifti1Image(nibCorrected, np.eye(4)),
    os.path.join(utils.OUTPUT_DIR, "1.nii.gz"),
)
plt.imshow(nibCorrected[:, :, 0])
plt.show()
