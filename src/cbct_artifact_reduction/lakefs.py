import os
import platform
import hashlib
import warnings


import boto3

import lakefs_client
from lakefs_client.api import objects_api


class S3Client():
    def __init__(self, endpoint, access_key, secret_key, repo_name, commit_id, local_cache_path, ca_path):
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.repo_name = repo_name
        self.commit_id = commit_id
        self.local_cache_path = local_cache_path
        self.ca_path = ca_path

    def get_all_filenames(self):
        raise NotImplementedError("Function not implemented in child class")

    def get_file(self, object_name, file_obj=None):
        raise NotImplementedError("Function not implemented in child class")


class LakeFSClient(S3Client):
    def __init__(self, endpoint, access_key, secret_key, repo_name, commit_ID, local_cache_path, ca_path):
        super().__init__(endpoint, access_key, secret_key, repo_name, commit_ID, local_cache_path, ca_path)

        self.local_cache_path_commit = os.path.join(self.local_cache_path, self.commit_id)

        configuration = lakefs_client.Configuration(
            username=self.access_key,
            password=self.secret_key,
            host=self.endpoint + "/api/v1",
            ssl_ca_cert=ca_path
        )
        configuration.temp_folder_path = self.local_cache_path

        api_client = lakefs_client.ApiClient(configuration)

        self.lakefs = objects_api.ObjectsApi(api_client)

        # create the local path
        if not os.path.exists(self.local_cache_path):
            os.makedirs(self.local_cache_path)

    def __reduce__(self):
        return (LakeFSClient, (self.endpoint, self.access_key, self.secret_key, self.repo_name, self.commit_id,
                               self.local_cache_path, self.ca_path,))

    def get_all_filenames(self):

        api_response = self.lakefs.list_objects(self.repo_name, self.commit_id)

        filenames = []

        for element in api_response["results"]:
            filenames.append(element["path"])

        return filenames

    def get_file(self, object_name, file_obj=None):

        if file_obj is not None:
            raise ValueError("lakefs client does not support download to file obj")

        suffix = "." + ".".join(object_name.split(".")[-2:])
        local_filename = hashlib.md5(object_name.encode('utf-8')).hexdigest() + suffix
        local_path = os.path.join(self.local_cache_path_commit, local_filename)

        if not os.path.exists(local_path):
            try:
                file = self.lakefs.get_object(self.repo_name, self.commit_id, object_name)
                os.rename(file.name, local_path)
            except:
                local_path = None
                warnings.warn("An error occurred during downloading the file from lakefs")

        return local_path


class Boto3Client(S3Client):
    def __init__(self, endpoint, access_key, secret_key, repo_name, commit_id, local_cache_path, ca_path):
        super().__init__(endpoint, access_key, secret_key, repo_name, commit_id, local_cache_path, ca_path)

        self.local_cache_path_commit = os.path.join(self.local_cache_path, os.getlogin())

        self.lakefs = boto3.client('s3', endpoint_url=self.endpoint, aws_access_key_id=self.access_key,
                                   aws_secret_access_key=self.secret_key, verify=self.ca_path)

        # create the local path
        if not os.path.exists(self.local_cache_path_commit):
            os.makedirs(self.local_cache_path_commit)

    def __reduce__(self):
        """Used for the serialization of the opject if used with more proces i.e. in the data loader"""
        return (Boto3Client, (self.endpoint, self.access_key, self.secret_key, self.repo_name, self.commit_id,
                              self.local_cache_path, self.ca_path,))

    def get_all_filenames(self):
        """Get all the object filenames of repository at the given commit"""

        paginator = self.lakefs.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.repo_name, Prefix=self.commit_id + "/")

        filenames = []

        for page in pages:
            for obj in page['Contents']:
                key = obj['Key']
                filenames.append(key)

        return filenames

    def get_file(self, object_name, file_obj=None):
        """Load the file from the S3 storage to the local disk or directly into the ram. If caching is activated, a
        local cache is used"""
        local_path = None

        if file_obj is None:
            # download the file into the local cache if it is not already in the cache
            suffix = "." + ".".join(object_name.split(".")[-2:])
            local_filename = hashlib.md5(object_name.encode('utf-8')).hexdigest() + suffix
            local_path = os.path.join(self.local_cache_path_commit, local_filename)

            if not os.path.exists(local_path):
                try:
                    print(f"Cache miss. Download object {object_name}")
                    self.lakefs.download_file(self.repo_name, object_name, local_path)
                except:
                    local_path = None
                    warnings.warn("An error occurred during downloading the file from lakefs")
        else:
            # download the object to a file buffer
            self.lakefs.download_fileobj(self.repo_name, object_name, file_obj)

        return local_path
