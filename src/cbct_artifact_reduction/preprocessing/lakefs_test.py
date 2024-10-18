from lakefs.client import Client
from utils import OUTPUT_DIR
import os

# import utils
clt = Client(
    host="https://dbe-lakefs.dbe.unibas.ch:8000",
)

remote_path = "lakefs://cbct-pig-jaws/processed_data/101_2_accuitomo_ti_3im_sf_r.nii.gz"
local_path = os.path.join(OUTPUT_DIR, "lakefs_test.nii.gz")
