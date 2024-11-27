import os
import random

import cbct_artifact_reduction.csvcreator
import cbct_artifact_reduction.utils as utils
import pandas as pd

path_to_data_csv = os.path.join(utils.ROOT_DIR, "data.csv")
path_to_shape_csv = os.path.join(utils.ROOT_DIR, "data_shape.csv")

data_csv = pd.read_csv(path_to_data_csv)
shape_csv = pd.read_csv(path_to_shape_csv)


A = data_csv
B = shape_csv.drop("id", axis=1)

control_ids = data_csv[data_csv["implants"] == 0]["id"]

slice_ids = [i for i in range(100, 200)]

slice_paths = []

for id in control_ids:
    for slice_id in slice_ids:
        slice_paths.append(f"{id}_{slice_id}.nii.gz")

length = len(slice_paths)
mask_ids = random.choices([i for i in range(1, 1000)], k=length)

mask_paths = [f"mask_{id}.nii.gz" for id in mask_ids]

print(len(slice_paths), len(mask_paths))
csvCreator = cbct_artifact_reduction.csvcreator.createSliceMaskCSV(
    slice_paths, mask_paths, os.path.join(utils.ROOT_DIR, "training_data.csv")
)
