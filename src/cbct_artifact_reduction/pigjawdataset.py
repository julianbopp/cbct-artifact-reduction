import hashlib
import os
import re

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


# TODO: Move this function to a separate file or module. Function is generated by ChatGPT.
def extract_number_before_underscore(input_string):
    # Match one or more digits (\d+) followed by an underscore (_)
    match = re.match(r"(\d+)_", input_string)
    if match:
        return int(match.group(1))  # Convert the matched number to an integer
    else:
        raise ValueError("Input string does not match the expected format.")


class SingleDataPoint:
    """Collection of path information of a single data point.

    Attributes:
        slice_path (str): The path to the slice
        mask_path (str): The path to the mask.
    """

    def __init__(
        self, relative_slice_path: str, relative_mask_path: str, data_info: dict | None
    ):
        """Initializes a SingleDataPoint object.

        Args:
            relative_slice_path (str): The path to the slice.
            relative_mask_path (str): The path to the mask.
            data_info (dict): Information about the data.
        """
        self.relative_slice_path = relative_slice_path
        self.relative_mask_path = relative_mask_path
        self.data_info = data_info


class InpaintingSliceDataset(Dataset):
    """A dataset containing slices of CBCT scans and binary masks to train a model for inpainting.

    The class uses a boto3client object to cache the data from LakeFS to the local filesystem when needed."""

    def __init__(
        self,
        lakefs_loader: CustomBoto3Client,
        data_specification_path: str,
        slice_directory_path: str,
        mask_directory_path: str,
        random_masks: bool = True,
    ) -> None:
        """Initializes the dataset.

        Args:
            lakefs_loader (boto3client): The LakeFSLoader object used to load data from LakeFS.
            data_specification_path (str): The path to the data specification file.
            relative_slice_directory_path (str): The relative path to the remote/local directory containing the slices.
            relative_mask_directory_path (str): The relative to the remote/local directory containing the masks.
        """

        super().__init__()
        self.lakefs_loader = lakefs_loader
        self.data_specification_path = data_specification_path
        self.relative_slice_directory_path = slice_directory_path
        self.relative_mask_directory_path = mask_directory_path
        self.random_masks = random_masks

        self.data_extension = ".nii.gz"
        self.dataset = self.prepare_dataset()
        # TODO: Don't hardcode the resolution
        self.mask_creator = imc.ImplantMaskCreator((256, 256))

    def prepare_dataset(self) -> list[SingleDataPoint]:
        """Create a list of SingleDataPoint objects containing the slices and masks that are specified in data_specification_path.

        Returns:
            list[SingleDataPoint]: A list of SingleDataPoint objects containing the filepaths of the slices and masks.
        """
        dataset = []
        with open(self.data_specification_path, "r") as f:
            next(f, None)  # Skip the header row
            for line in f:
                slice_filename, mask_filename = line.strip().split(",")

                relative_slice_path = os.path.join(
                    self.relative_slice_directory_path, slice_filename
                )
                relative_mask_path = os.path.join(
                    self.relative_mask_directory_path, mask_filename
                )
                # TODO: Don't hardcode the id length. Specify it somewhere or use regex or some function.
                id = extract_number_before_underscore(slice_filename)
                data_info = lookup_num_in_datatable(id)

                dataset.append(
                    SingleDataPoint(relative_slice_path, relative_mask_path, data_info)
                )

        return dataset

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Downloads the idx-th slice and mask from LakeFS and returns them as numpy arrays.

        Uses cache if the files are already downloaded.

        Args:
            idx (int): The index of the slice and mask to return.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the slice and mask at the given index.
        """

        assert 0 <= idx < self.__len__(), f"Index {idx} out of bounds"

        item = self.dataset[idx]
        item_info = item.data_info

        slice_path = self.lakefs_loader.get_file(item.relative_slice_path)

        assert slice_path is not None, (
            f"File {item.relative_slice_path} not found on lakeFS"
        )

        if self.random_masks:
            slice_hash = None
        else:
            slice_hash = int.from_bytes(
                hashlib.sha256(slice_path.encode("utf-8")).digest()[:4], "little"
            )

        slice_np_array = single_nifti_to_numpy(slice_path)
        # TODO: Don't hardcode the amount of implants. Specify it somewhere.
        mask_np_array = self.mask_creator.generate_mask_with_random_amount_of_implants(
            1, 4, random_state=slice_hash
        )

        if item_info is not None:
            try:
                scanner = item_info["scanner"][0]
                fov = item_info["fov"][0]
                processed_slice_np_array = self.dataprocessing(
                    slice_np_array, scanner, fov
                )
            except KeyError:
                print(f"No scanner information for item at slice_path: {slice_path}")
                processed_slice_np_array = self.dataprocessing(slice_np_array)
        else:
            processed_slice_np_array = self.dataprocessing(slice_np_array)

        return processed_slice_np_array[np.newaxis, ...], mask_np_array[np.newaxis, ...]

    def dataprocessing(
        self, np_array: np.ndarray, scanner: str | None = None, fov: str | None = None
    ) -> np.ndarray:
        """Preprocesses the numpy array by normalizing it and removing outliers. Addiotionally, if scanner and fov are provided, the array is preprocessed accordingly.

        Args:
            nparray (np.ndarray): The numpy array to preprocess.
            scanner (str, optional): The scanner used for the slice. Defaults to None.
            fov (str, optional): The field of view of the slice. Defaults to None.

        Returns:
            np.ndarray: The preprocessed numpy array.
        """
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
