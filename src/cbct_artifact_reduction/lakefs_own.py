import lakefs
from lakefs.client import Client

import cbct_artifact_reduction.config as cfg

REPO = "cbct-pig-jaws"
BRANCH = "processed_data"


class CustomLakeFSClient:
    def __init__(self, repo: str) -> None:
        self.repo: str = repo
        self.client = Client(
            host=cfg.LAKEFS_HOST,
            verify_ssl=cfg.LAKEFS_VERIFY_SSL,
            ssl_ca_cert=cfg.LAKEFS_SSL_CA_CERT,
            username=cfg.LAKEFS_USERNAME,
            password=cfg.LAKEFS_PASSWORD,
        )
        self.branch: str

    def set_branch(self, branch: str) -> None:
        self.branch = branch

    def get_branches(self):
        return lakefs.repository(f"{self.repo}", client=self.client).branches()

    def list_all_files(self):
        return (
            lakefs.repository(f"{self.repo}", client=self.client)
            .branch(f"{self.branch}")
            .objects()
        )

    def list_files_in_folder(self, folder: str):
        return (
            lakefs.repository(f"{self.repo}", client=self.client)
            .branch(f"{self.branch}")
            .objects(prefix=f"{folder}")
        )


customLakeFSClient = CustomLakeFSClient(REPO)
customLakeFSClient.set_branch(BRANCH)
