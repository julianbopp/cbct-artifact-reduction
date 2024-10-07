import os
from utils import ROOT_DIR, DATA_DIR, CODE_DIR, OUTPUT_DIR
from brainglobe_utils.IO.image import load, save

import numpy as np


planmeca_path = os.path.join(
    DATA_DIR,
    "Projections_Planmeca_SmallFOV_part1_B/201/pm3DData/20240117075947/corrected/",
)

np_array = np.fromfile(planmeca_path + "frame-100.raw", dtype=np.uint16)
print(np_array.shape)


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min) / (_max - _min)
    return normalized_img


vol = np.zeros((840, 732, len(list_names)))

cnt = 0


print("ok")
