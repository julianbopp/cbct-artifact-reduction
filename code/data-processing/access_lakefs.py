import boto3

s3 = boto3.client(
    "s3",
    endpoint_url="https://dbe-lakefs.dbe.unibas.ch:8000",
)

data_repository = "cbct-pig-jaws"
commit = "processed_data"
folder = "scaled"

list_repo = s3.list_objects_v2(Bucket=data_repository, Prefix=f"{commit}/{folder}/")
for obj in list_repo["Contents"]:
    print(obj["Key"])
