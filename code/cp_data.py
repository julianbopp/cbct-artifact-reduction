from utils import get_scanner_from_num, OUTPUT_DIR
import os
import shutil

data_path = os.path.join(OUTPUT_DIR, "data")
destination_path = os.path.join(OUTPUT_DIR, "rotated")

# Get list of folders in data_dir
files = sorted([f for f in os.listdir(data_path) if f.endswith("nii.gz")])


for file in files:

    _, scanner, _, _, _ = get_scanner_from_num(int(file[0:-7]))
    if scanner == "planmeca":
        print(
            f"Copyying planmeca data {file[0:-7]} from {data_path} to {destination_path}"
        )
        # Copy file to destination
        shutil.copyfile(
            os.path.join(data_path, file), os.path.join(destination_path, file)
        )
    else:
        continue
