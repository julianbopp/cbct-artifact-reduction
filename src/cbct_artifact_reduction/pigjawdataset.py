import os

import numpy as np
from torch.utils.data.dataset import Dataset

from cbct_artifact_reduction.dataprocessing import single_nifti_to_numpy
from cbct_artifact_reduction.lakefs_own import CustomBoto3Client


class SingleDataPoint:
    """Collection of path information of a single data point.

    Attributes:
        slice_path (str): The path to the slice
        mask_path (str): The path to the mask.
    """

    def __init__(self, relative_slice_path: str, relative_mask_path: str):
        """Initializes a SingleDataPoint object.

        Args:
            relative_slice_path (str): The path to the slice
            relative_mask_path (str): The path to the mask.
        """
        self.relative_slice_path = relative_slice_path
        self.relative_mask_path = relative_mask_path


class InpaintingSliceDataset(Dataset):
    """A dataset containing slices of CBCT scans and binary masks to train a model for inpainting.

    The class uses a boto3client object to cache the data from LakeFS to the local filesystem when needed."""

    def __init__(
        self,
        lakefs_loader: CustomBoto3Client,
        data_specification_path: str,
        slice_directory_path: str,
        mask_directory_path: str,
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

        self.data_extension = ".nii.gz"
        self.dataset = self.prepare_dataset()

    def prepare_dataset(self) -> list[SingleDataPoint]:
        """Create a list of SingleDataPoint objects containing the slices and masks that are specified in data_specification_path.

        Returns:
            list[SingleDataPoint]: A list of SingleDataPoint objects containing the filepaths of the slices and masks.
        """
        dataset = []
        with open(self.data_specification_path, "r") as f:
            for line in f:
                slice_filename, mask_filename = line.strip().split(",")

                relative_slice_path = os.path.join(
                    self.relative_slice_directory_path, slice_filename
                )
                relative_mask_path = os.path.join(
                    self.relative_mask_directory_path, mask_filename
                )

                dataset.append(SingleDataPoint(relative_slice_path, relative_mask_path))

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

        item_info = self.dataset[idx]

        self.lakefs_loader.get_file(item_info.relative_slice_path)
        self.lakefs_loader.get_file(item_info.relative_mask_path)

        base_path = self.lakefs_loader.cache_path
        slice_path = os.path.join(base_path, item_info.relative_slice_path)
        mask_path = os.path.join(base_path, item_info.relative_mask_path)

        slice_np_array = single_nifti_to_numpy(slice_path)
        mask_np_array = single_nifti_to_numpy(mask_path)

        return slice_np_array, mask_np_array

    def __len__(self) -> int:
        return len(self.dataset)
