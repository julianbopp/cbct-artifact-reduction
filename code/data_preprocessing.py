# Description: This script will preprocess the projection data. There is data in tif and raw format from 3 different scanners. Output will be in nii.gz format.

from brainglobe_utils.IO.image import load, save
import numpy as np
import tarfile
import os


# Process tif data
def tif_to_nifti(input_path: str, output_path: str):
    # Load the data
    data = load.load_img_stack(input_path, 1, 1, 1)
    # Save the data
    save.to_nii(data, output_path)


# Merge multiple raw files into one nifti file
def raws_to_nifti(filenames: list[str], output_path: str):
    pass
    # volume =
    # Load the data
    # data =
    # Save the data
    # save.to_nii(data, output_path)


def extract_tar_gz(tar_gz_file: str, output_path: str):
    file = tarfile.open(tar_gz_file)
    # print(file.getnames())
    file.extractall(output_path)
    file.close()


if __name__ == "__main__":

    # Define data and output folder paths
    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "../output")
    data_dir = os.path.join(script_dir, "../data")

    # get list of folders in data_dir
    folders = [
        f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))
    ]

    # filter folder for accuitomo data
    accuitomo_folders = [f for f in folders if "Accuitomo" in f]
    # filter folder for axeo data
    axeo_folders = [f for f in folders if "Axeo" in f]
    # filter folder for planmeca data
    planmeca_folders = [f for f in folders if "Planmeca" in f]

    # Define which data to process
    accuitomo = False
    axeo = True
    planmeca = False

    # Process Accuitomo data
    # First get all tif file paths
    for folder in accuitomo_folders:
        if not accuitomo:
            print("Skipping Accuitomo data")
            break
        else:
            print("Processing Accuitomo data")
        # Create folder in output directory
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        # Get list of files in folder
        files = os.listdir(os.path.join(data_dir, folder))
        # Join the data path with the file name from files and Capture.tif
        tif_paths = [os.path.join(data_dir, folder, f, "Capture.tif") for f in files]

        # Save all tif files as nifti
        for index, tif_path in enumerate(tif_paths):

            tif_file = load.load_img_stack(tif_path, 1, 1, 1)
            save.to_nii(
                tif_file,
                os.path.join(output_dir, os.path.basename(folder), f"{index}.nii.gz"),
            )

    # Process Axeos data
    # First get all tif file paths
    for folder in axeo_folders:
        if not axeo:
            print("Skipping Axeos data")
            break
        else:
            print("Processing Axeos data")
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        # Get list of files in folder
        files = os.listdir(os.path.join(data_dir, folder))
        # Join the data path with the file name from files and Capture.tif
        tif_paths = [
            os.path.join(data_dir, folder, f, "correctedRawData.tif") for f in files
        ]

        # Save all tif files as nifti
        for index, tif_path in enumerate(tif_paths):

            tif_file = load.load_img_stack(tif_path, 1, 1, 1)
            save.to_nii(
                tif_file,
                os.path.join(output_dir, os.path.basename(folder), f"{index}.nii.gz"),
            )

    # Process Planmeca data


