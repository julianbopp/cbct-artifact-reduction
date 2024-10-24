from torch.utils.data.dataset import Dataset
from utils import get_scanner_from_num


class PigJawSlice:
    """Class that represents a pig jaw slice. Stores the relevant information about the projection stack that the slice comes from."""

    def __init__(self, file_path, id) -> None:
        self.file_path = file_path
        self.orig_num, self.scanner, self.material, self.implants, self.fov = get_scanner_from_num(id)

class PigJawDataset(Dataset):
    ""
    def __init__(self, lakefs_loader, cache_files=True, test_flag=False):

        self.jaw_slices = {}
        self.lakefs_loader = lakefs_loader
        self.cache_files = cache_files

        self.test_flag = test_flag
        if self.test_flag:
            self.seqtypes = ["voided", "mask"]
        else:
            self.seqtypes = ["voided", "mask", "healthy"]

        # parse all object file names to create a data set
        filenames = self.lakefs_loader.get_all_filenames()
        self.parse_files(filenames)

    def parse_files(self, filenames):
        # parse the filenames to create a dictionary
