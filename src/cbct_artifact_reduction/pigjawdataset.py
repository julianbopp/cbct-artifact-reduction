import csv
import hashlib
import os
import re
from typing import Any

import numpy as np
from torch.utils.data.dataset import Dataset

import cbct_artifact_reduction.implantmaskcreator as imc
from cbct_artifact_reduction.dataprocessing import (
    min_max_normalize,
    remove_outliers,
    single_nifti_to_numpy,
)
from cbct_artifact_reduction.lakefs_own import CustomBoto3Client
from cbct_artifact_reduction.utils import lookup_num_in_datatable


def get_filenames_from_csv(csv_path):
    filenames = []
    with open(csv_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)  # Skip the header row
        for row in csv_reader:
            filenames.append(row[0].strip())
    return filenames


# TODO: Move this function to a separate file or module. Function is generated by ChatGPT.
def extract_number_before_underscore(input_string):
    # Match one or more digits (\d+) followed by an underscore (_)
    match = re.match(r"(\d+)_", input_string)
    if match:
        return int(match.group(1))  # Convert the matched number to an integer
    else:
        raise ValueError("Input string does not match the expected format.")


class InpaintingSliceDataset(Dataset):
    """A dataset containing slices of CBCT scans and binary masks to train a model for inpainting.

    The class uses a boto3client object to cache the data from LakeFS to the local filesystem when needed."""

    def __init__(
        self,
        lakefs_loader: CustomBoto3Client,
        filenames: str | list[str],
        slice_directory_path: str,
        random_masks: bool = True,
    ) -> None:
        """Initializes the dataset.

        Args:
            lakefs_loader (boto3client): The LakeFSLoader object used to load data from LakeFS.
            filenames list[str]: The list of filenames to load. Can be a path to a csv file or a list of filenames.
            relative_slice_directory_path (str): The relative path to the remote/local directory containing the slices.
            random_masks (bool): Whether to generate random masks or use the random generated masks with the hash of the file name.
        """

        super().__init__()
        self.lakefs_loader = lakefs_loader
        self.filenames = filenames
        self.relative_slice_directory_path = slice_directory_path
        self.random_masks = random_masks

        self.data_extension = ".nii.gz"
        self.dataset = self.prepare_dataset()
        # TODO: Don't hardcode the resolution
        self.mask_creator = imc.ImplantMaskCreator((256, 256))

    def prepare_dataset(self) -> list[dict[str, Any]]:
        """Create a list of data points, each containing the path to the slice and information about the data to which the slice belongs.

        Returns:
            list[dict]: A list of dictionaries containing slice_path and data_info.
        """
        dataset = []

        if isinstance(self.filenames, str):
            self.filenames = get_filenames_from_csv(self.filenames)

        for filename in self.filenames:
            slice_filename = filename.strip()

            slice_path = os.path.join(
                self.relative_slice_directory_path, slice_filename
            )
            # TODO: Don't hardcode the id length. Specify it somewhere or use regex or some function.
            id = extract_number_before_underscore(slice_filename)
            data_info = lookup_num_in_datatable(id)

            datapoint = {"slice_path": slice_path, "data_info": data_info}

            dataset.append(datapoint)

        return dataset

    def get_mask(self, local_slice_path: str) -> np.ndarray:
        """Create a random or deterministic mask for the given slice.
        Use self.random_masks to determine whether to create a random mask
        or a deterministic one based on the hash of the slice filename.

        Args:
            local_slice_path (str): The path to the slice on the local filesystem.

        Returns:
            np.ndarray: The mask for the given slice.
        """

        if self.random_masks:
            slice_hash = None
        else:
            slice_hash = int.from_bytes(
                hashlib.sha256(local_slice_path.encode("utf-8")).digest()[:4], "little"
            )

        mask_np_array = self.mask_creator.generate_mask_with_random_amount_of_implants(
            1, 4, random_state=slice_hash
        )

        return mask_np_array

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Downloads the idx-th slice and from LakeFS and creates a mask for it.
        Optionally, it can also return the data_info, if set in the class constructor.
        Uses cache if the files are already downloaded.

        Args:
            idx (int): The index of the slice to return.

        Returns:
            dict[str, Any]: A dictionary containing the following keys:
                - slice (np.ndarray): containing the slice.
                - mask (np.ndarray): containing the mask.
                - info (optional): dict containing the data_info.
        """

        assert 0 <= idx < self.__len__(), f"Index {idx} out of bounds"

        datapoint = self.dataset[idx]

        item_path = datapoint["slice_path"]
        item_info = datapoint["data_info"]

        # Download the slice from lakeFS. If it is not found this will be None.
        local_slice_path = self.lakefs_loader.get_file(item_path)

        if local_slice_path is None:
            print(f"File {item_path} not found on lakeFS. Deleting item from dataset.")
            self.dataset.pop(idx)
            return self.__getitem__(idx)

        slice_np_array = single_nifti_to_numpy(local_slice_path)
        mask_np_array = self.get_mask(local_slice_path)

        processed_slice_np_array = self.dataprocessing(slice_np_array)

        item = {
            "slice": processed_slice_np_array[np.newaxis, ...],
            "mask": mask_np_array[np.newaxis, ...],
            "info": item_info,
        }

        return item

    def dataprocessing(
        self,
        np_array: np.ndarray,
        scanner: str | None = None,
        fov: str | None = None,
        scanner_processing: bool = False,
    ) -> np.ndarray:
        """Preprocesses the numpy array by normalizing it and removing outliers. Addiotionally, if scanner and fov are provided, the array is preprocessed accordingly.

        Args:
            nparray (np.ndarray): The numpy array to preprocess.
            scanner (str, optional): The scanner used for the slice. Defaults to None.
            fov (str, optional): The field of view of the slice. Defaults to None.
            scanner_processing (bool, optional): Whether to apply scanner specific preprocessing. Defaults to False.
        Returns:
            np.ndarray: The preprocessed numpy array.
        """
        if scanner_processing:
            if scanner is not None and fov is not None:
                if scanner == "planmeca" and fov == "small":
                    np_array = -np.log(np_array / (3591 * 2.27))
                elif scanner == "planmeca" and fov == "large":
                    np_array = -np.log(np_array / (4326 * 2.27))
                elif scanner == "axeos":
                    np_array = -np.log(np_array / (2 * 10**16))
                else:
                    # TODO: Add more preprocessing for other scanners. Waiting for the details from Susanne.
                    print(f"No extra preprocessing for scanner {scanner} and fov {fov}")

        outliers_removed = remove_outliers(np_array)
        normalized = min_max_normalize(outliers_removed)

        return normalized

    def __len__(self) -> int:
        return len(self.dataset)
