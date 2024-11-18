import csv

from cbct_artifact_reduction.utils import get_scanner_from_num


def getAllControlIDs(exludeIDs: list[int] | None = [41, 208]) -> list[str]:
    """Find all scan ID's that correspond to control images without implants in the CBCT pig jaw data.

    The CBCT pig jaw dataset consists of 398 scans. The scans with the ID's 41 and 208 are missing.

    Args:
        exludeIDs (list[int], optional): List of scan ID's to exclude. Defaults to missing ID's of CBCT pig jaw data.
    Returns:
        list[str]: List of scan ID's that correspond to control images without implants.
    """

    if exludeIDs is None:
        possibleIDs = [f"{f}" for f in range(0, 401)]
    else:
        possibleIDs = [f"{f}" for f in range(0, 401) if f not in exludeIDs]

    controlIDs: list[str] = []
    for id in possibleIDs:
        scanner = get_scanner_from_num(int(id))
        if scanner[3] == 0:
            controlIDs.append(id)

    return controlIDs


def createSliceMaskCSV(slices: list[str], masks: list[str], csv_output_path: str):
    """Populate a CSV file with everything necessary to create a dataset for the training of an inpainting model.

    The csv file will contain 2 columns: 'slice' and 'mask'. The 'slice' column will contain the path to the slice image, and the 'mask' column will contain the path to the mask image.

    Args:
        slices (list[str]): List of paths to the slices.
        masks (list[str]): List of paths to the masks.
    """
    assert len(slices) == len(
        masks
    ), "The number of slices and masks paths must be the same."

    rows = zip(slices, masks)
    with open(csv_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["slice", "mask"])
        writer.writerows(rows)
