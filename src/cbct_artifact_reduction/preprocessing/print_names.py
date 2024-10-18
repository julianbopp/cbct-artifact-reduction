import os
from cbct_artifact_reduction.utils import OUTPUT_DIR
import numpy as np

array = np.zeros((1, 1400))
for index, f in enumerate(os.listdir(os.path.join(OUTPUT_DIR, "rotated"))):
    if f.endswith(".nii.gz"):
        array[0, int(f[0:-7])] = 1

breakpoint()
