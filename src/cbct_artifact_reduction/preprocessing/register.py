import os
from cbct_artifact_reduction.utils import (
    OUTPUT_DIR,
)
import ants

overwrite = True


fixed_path = os.path.join(OUTPUT_DIR, "rotated", "10.nii.gz")
moving_paths = [
    os.path.join(OUTPUT_DIR, "rotated", f"{index}.nii.gz") for index in range(1, 2)
]
moving_path_1 = moving_paths[0]

fixedAntsImage = ants.image_read(fixed_path)
fixed_np_array = fixedAntsImage.numpy()  # [:, 0 : 324 - 7, :]
fixedAntsImage = ants.from_numpy(fixed_np_array)
movingAntsImage = ants.image_read(moving_path_1)
movingAntsImage = ants.from_numpy(movingAntsImage.numpy())

filename = "SyNRA"
if (
    os.path.exists(os.path.join(OUTPUT_DIR, "registered", f"{filename}.nii.gz"))
    and not overwrite
):
    print("Registered image already exists")
    registeredAntsImage = ants.image_read(
        os.path.join(OUTPUT_DIR, "registered", "10.nii.gz")
    )
else:
    registration = ants.registration(
        fixed=fixedAntsImage,
        moving=movingAntsImage,
        type_of_transform=f"{filename}",
    )
    registeredFixedAntsImage = registration["warpedfixout"]
    registeredMovingAntsImage = registration["warpedmovout"]
    ants.image_write(
        registeredFixedAntsImage,
        os.path.join(OUTPUT_DIR, "registered", f"{filename}_fixed.nii.gz"),
    )
    ants.image_write(
        registeredMovingAntsImage,
        os.path.join(OUTPUT_DIR, "registered", f"{filename}_moving.nii.gz"),
    )
