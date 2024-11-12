import cbct_artifact_reduction.config as cfg
import cbct_artifact_reduction.lakefs_own as lakefs_own


def test_lakefs_connection():
    testLakeFSClient = lakefs_own.CustomLakeFSClient(f"{cfg.LAKEFS_DATA_REPOSITORY}")
    testLakeFSClient.set_branch(f"{cfg.LAKEFS_COMMIT}")
    files = testLakeFSClient.list_all_files()
    print(files)


def test_lakefs_download():
    pass


def test_lakefs_caching():
    pass
