import csv


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
