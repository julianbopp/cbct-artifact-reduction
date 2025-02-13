import csv
import os

import pandas as pd

from cbct_artifact_reduction import utils


def filter_data(csv_path):
    # Load the data
    df = pd.read_csv(csv_path)

    # Get unique values for filtering
    scanners = df["scanner"].unique().tolist()
    fovs = df["fov"].unique().tolist()
    materials = df["material"].dropna().unique().tolist()
    mandibles = df["mandible"].unique().tolist()
    implants = df["implants"].unique().tolist()
    radiations = df["radiation"].dropna().unique().tolist()

    # User input for scanner selection
    print("Available scanner types:", scanners)
    selected_scanners = input(
        "Enter scanner types to include (comma-separated, leave empty for all): "
    ).split(",")
    selected_scanners = [s.strip() for s in selected_scanners if s.strip()] or scanners

    # User input for FOV selection
    print("Available FOVs:", fovs)
    selected_fovs = input(
        "Enter FOVs to include (comma-separated, leave empty for all): "
    ).split(",")
    selected_fovs = [f.strip() for f in selected_fovs if f.strip()] or fovs

    # User input for material selection
    print("Available materials:", materials)
    selected_materials = input(
        "Enter materials to include (comma-separated, leave empty for all, type 'exclude_missing' to exclude missing values): "
    ).split(",")
    selected_materials = [
        m.strip() for m in selected_materials if m.strip() and m != "exclude_missing"
    ] or materials
    include_missing_materials = "exclude_missing" not in selected_materials

    # User input for mandible selection
    print("Available mandible values:", mandibles)
    selected_mandibles = input(
        "Enter mandible values to include (comma-separated, leave empty for all): "
    ).split(",")
    selected_mandibles = [
        int(m.strip()) for m in selected_mandibles if m.strip().isdigit()
    ] or mandibles

    # User input for implants selection
    print("Available implant counts:", implants)
    selected_implants = input(
        "Enter implant counts to include (comma-separated, leave empty for all): "
    ).split(",")
    selected_implants = [
        int(i.strip()) for i in selected_implants if i.strip().isdigit()
    ] or implants

    # User input for radiation selection
    print("Available radiation values:", radiations)
    selected_radiations = input(
        "Enter radiation values to include (comma-separated, leave empty for all, type 'exclude_missing' to exclude missing values): "
    ).split(",")
    selected_radiations = [
        r.strip() for r in selected_radiations if r.strip() and r != "exclude_missing"
    ] or radiations
    include_missing_radiations = "exclude_missing" not in selected_radiations

    # Apply filters
    filtered_df = df[
        df["scanner"].isin(selected_scanners)
        & df["fov"].isin(selected_fovs)
        & df["mandible"].isin(selected_mandibles)
        & df["implants"].isin(selected_implants)
    ]

    if include_missing_materials:
        filtered_df = filtered_df[
            filtered_df["material"].isin(selected_materials)
            | filtered_df["material"].isna()
        ]
    else:
        filtered_df = filtered_df[
            filtered_df["material"].isin(selected_materials)
            & filtered_df["material"].notna()
        ]

    if include_missing_radiations:
        filtered_df = filtered_df[
            filtered_df["radiation"].isin(selected_radiations)
            | filtered_df["radiation"].isna()
        ]
    else:
        filtered_df = filtered_df[
            filtered_df["radiation"].isin(selected_radiations)
            & filtered_df["radiation"].notna()
        ]

    # Save filtered IDs to a new CSV file
    return filtered_df["id"].tolist()


def get_slices_from_ids(csv_path: str, ids: list[str]):
    """Get the slices from the ids.

    Args:
        csv_path (str): Path to the data csv.
        ids (list[str]): List of ids.

    Returns:
        list[str]: List of slice paths.
    """

    data_csv = pd.read_csv(csv_path)
    amount_of_slices = data_csv[data_csv["id"].isin(ids)]["frames"]
    slice_paths = []

    for iter, id in enumerate(ids):
        num_slices = amount_of_slices.values[iter]
        for n in range(num_slices):
            slice_paths.append(f"{id}_{n}.nii.gz")
    return slice_paths


def createSliceCSV(slices: list[str], csv_output_path: str):
    """Populate a csv file paths to the slices.

    Args:
        slices (list[str]): List of paths to the slices.
        csv_output_path (str): Path to the csv file.
    """

    rows = zip(slices)
    with open(csv_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["slice"])
        writer.writerows(rows)


def createSliceMaskCSV(slices: list[str], masks: list[str], csv_output_path: str):
    """Populate a CSV file with everything necessary to create a dataset for the training of an inpainting model.

    The csv file will contain 2 columns: 'slice' and 'mask'. The 'slice' column will contain the path to the slice image, and the 'mask' column will contain the path to the mask image.

    Args:
        slices (list[str]): List of paths to the slices.
        masks (list[str]): List of paths to the masks.
    """
    assert len(slices) == len(masks), (
        "The number of slices and masks paths must be the same."
    )

    rows = zip(slices, masks)
    with open(csv_output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["slice", "mask"])
        writer.writerows(rows)


if __name__ == "__main__":
    csv_path = os.path.join(utils.ROOT_DIR, "data.csv")
    ids = filter_data(csv_path)
    slice_paths = get_slices_from_ids(csv_path, ids)
    args = input(
        "Type slice for csv with slice paths or mask for csv with slice+mask paths:"
    )

    if args == "slice":
        createSliceCSV(slice_paths, os.path.join(utils.ROOT_DIR, "training_data.csv"))
    elif args == "mask":
        print("Not implemented yet")
