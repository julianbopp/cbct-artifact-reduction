import os
import shutil

# Path to the directory containing the mandible1 files

script_dir = os.path.dirname(__file__)
output_dir = os.path.join(script_dir, "../output/mandible1")
data_dir = os.path.join(script_dir, "../data")
mandible_projections_dir = os.path.join(script_dir, "../data/mandible1/projections/")

proj_parent_dirs = os.listdir(mandible_projections_dir)


print("Copying mandible1 files to output directory")
for folder in proj_parent_dirs:
    if folder == ".DS_Store":
        continue
    subfolder = os.listdir(os.path.join(mandible_projections_dir, folder))
    for item in subfolder:
        if item == ".DS_Store":
            continue
        # copy item to output directory
        shutil.copytree(
            os.path.join(mandible_projections_dir, folder, item),
            os.path.join(output_dir, item),
            dirs_exist_ok=True,
        )
print("Done copying mandible1 files to output directory")
