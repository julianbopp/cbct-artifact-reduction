import os

import numpy as np
from torch.utils.data.dataset import Dataset

import cbct_artifact_reduction.dataprocessing as dp
from cbct_artifact_reduction.lakefs_other import boto3client


class InpaintingSliceDataset(Dataset):
    """A dataset containing slices of CBCT scans and binary masks to train a model for inpainting.

    Attributes:
        lakefs_loader (boto3client): The LakeFSLoader object used to load data from LakeFS.
        slice_directory_path (str): The path to the directory containing the CBCT slices.
        mask_directory_path (str): The path to the directory containing the binary masks.
    """

    def __init__(
        self,
        lakefs_loader: boto3client,
        slice_directory_path: str,
        mask_directory_path: str,
    ) -> None:
        """Initializes the dataset.

        Args:
            lakefs_loader (boto3client): The LakeFSLoader object used to load data from LakeFS.
            slice_directory_path (str): The path to the directory containing the CBCT slices.
            mask_directory_path (str): The path to the directory containing the binary masks.
        """
        super().__init__()
        self.data_extension = ".nii.gz"
        self.slice_directory_path = slice_directory_path
        self.mask_directory_path = mask_directory_path
        self.slice_filenames = [
            f
            for f in os.listdir(slice_directory_path)
            if f.endswith(f"{self.data_extension}")
        ]
        self.mask_filenames = [
            f
            for f in os.listdir(mask_directory_path)
            if f.endswith(f"{self.data_extension}")
        ]

        assert len(self.slice_filenames) == len(
            self.mask_filenames
        ), "Number of slices and masks must be equal."

    def __len__(self) -> int:
        return len(self.slice_filenames)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple containing the slice and mask at the given index.

        Args:
            idx (int): The index of the slice and mask to return.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the slice and mask at the given index.
        """
        slice_path = os.path.join(self.slice_directory_path, self.slice_filenames[idx])
        mask_path = os.path.join(self.mask_directory_path, self.mask_filenames[idx])

        slice_np_array = dp.single_nifti_to_numpy(slice_path)
        mask_np_array = dp.single_nifti_to_numpy(mask_path)

        return slice_np_array, mask_np_array
