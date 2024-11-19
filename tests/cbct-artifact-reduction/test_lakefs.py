import hashlib
import os

import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.lakefs_own as lakefs_own


def test_lakefs_connection():
    testLakeFSClient = lakefs_own.CustomLakeFSClient(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    testLakeFSClient.set_branch(f"{cfg.LAKEFS_COMMIT}")
    files = testLakeFSClient.list_all_files()
    print(files)


def testCustomBoto3ClientListFolder():
    client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    client.list_files_in_folder("test")


def testCustomBoto3ClientGetFileNoCache(tmp_path):
    client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    client.cache_path = tmp_path
    local_name = (
        hashlib.md5(
            "processed_data/frames/256x256/100_0.nii.gz".encode("utf-8")
        ).hexdigest()
        + ".nii.gz"
    )
    client.get_file("processed_data/frames/256x256/100_0.nii.gz")

    assert os.path.exists(tmp_path / local_name)
    os.remove(tmp_path / local_name)


def testCustomBoto3ClientGetFileFromCache(tmp_path):
    filename = (
        hashlib.md5(
            "processed_data/frames/256x256/100_0.nii.gz".encode("utf-8")
        ).hexdigest()
        + ".nii.gz"
    )
    with open(tmp_path / filename, "w") as f:
        f.write("test")
        f.close()

    client = lakefs_own.CustomBoto3Client(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    client.cache_path = tmp_path

    local_path = client.get_file("processed_data/frames/256x256/100_0.nii.gz")

    assert os.path.join(tmp_path, filename) == local_path


def test_lakefs_download():
    pass


def test_lakefs_caching():
    pass


if __name__ == "__main__":
    test_lakefs_connection()
    testCustomBoto3ClientListFolder()
