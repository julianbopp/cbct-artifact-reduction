from lakefs_spec import LakeFSFileSystem
from utils import OUTPUT_DIR
import os

fs = LakeFSFileSystem()

remote_path = "lakefs://cbct-pig-jaws/processed_data/101_2_accuitomo_ti_3im_sf_r.nii.gz"
local_path = os.path.join(OUTPUT_DIR, "lakefs_test.nii.gz")

