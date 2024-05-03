import os

from typing import Optional, Union, IO
from pathlib import Path

from google.cloud import storage

def get_blob_uri(blob: storage.Blob):
    blob.reload()
    return 'gs://' + blob.id[:-(len(str(blob.generation)) + 1)]

def upload_file_to_bucket(blob_name: str,
                          bucket_name: str="qna-staging", 
                          filepath: Optional[Union[str, Path]]=None,
                          data:Optional[Union[str, bytes]]=None,
                          file: Optional[IO]=None, 
                          client: storage.Client=None,
                          encoding: str="utf-8",
                          content_type: str="application/pdf"
                          ) -> None:
    """Upload a file through either 1 of three means: upload from local filepath, upload data (str or bytes), upload using file object
    A new blob with the name blob_name is created in Google Cloud Storage where this file is then uploaded.

    @params:
        blob_name (required, str): The name of the blob where the file be uploaded.
        bucket_name (optional, str): The name of the bucket where the blob will be created and the file will be uploaded. Default="qna-staging".
        filepath (optional, str|Path): The local filepath to upload the file from. Default=None.
        file (optional, file-like object): The file object of an opened file to upload to bucket. Default=None.
        client (optional, storage.Client): The client to use for managing Google Cloud Storage. Default=New Client Created
        encoding (optional, str): The encoding to use to decode the contents of the 'data' argument if it is a bytes object. Default="utf-8"
        content_type (optional): The content_type to be used to upload the file
        
    """
    if client is None:
        client = storage.Client()
    bucket = client.get_bucket(bucket_name) # Will raise Exception if bucket does not exist
    blob = bucket.blob(blob_name=blob_name) # Create new blob or get existing blob
    if blob.exists():
        print("File already exists")
        return get_blob_uri(blob)
    if data is not None: # data has the highest precedence
        # if isinstance(data, bytes):
        #     data = data.decode(encoding=encoding)
        blob.upload_from_string(data, content_type=content_type)
    elif file is not None: # ...before the file-like object
        blob.upload_from_file(file, rewind=True, content_type=content_type)
    elif filepath is not None and os.path.isfile(filepath): # ...and finally, the path to the file
        blob.upload_from_filename(filepath, content_type=content_type)
    else:
        raise ValueError("Need at least one of filename (a valid file path), data or file to upload to given bucket")
    
    return get_blob_uri(blob)

def delete_folder(folder_path: str,
                  client: Optional[storage.Client] = None,
                  bucket_name: str = "qna-staging",
                  ) -> None:
    if client is None:
        client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder_path))
    bucket.delete_blobs(blobs)
    print(f"Folder {folder_path} deleted.")
    
def download_folder(destination_directory: str,
                    prefix_path: str,
                    bucket_name: str = "qna-staging",
                    client: Optional[storage.Client] = None,
                    workers: int = 2 
                    ) -> None:
    '''
    Download all files/folders/blobs from the given GCP path: `gs://{bucket_name}/{prefix_path}/`
    
    Uses transfer manager to download multiple files in parallel processes to the `destination_directory`.
    '''
    from google.cloud.storage import transfer_manager
    
    if client is None:
        client = storage.Client()
        
    bucket = client.get_bucket(bucket_name)
    blob_names = [blob.name for blob in bucket.list_blobs(prefix=prefix_path) if blob.name.strip('/') != prefix_path.strip('/')]
    
    results = transfer_manager.download_many_to_path(
        bucket, blob_names, destination_directory=destination_directory, max_workers=workers
    )

    for name, result in zip(blob_names, results):
        # The results list is either `None` or an exception for each blob in
        # the input list, in order.

        if isinstance(result, Exception):
            print("Failed to download {} due to exception: {}".format(name, result))
        else:
            print("Downloaded {} to {}.".format(name, destination_directory + '/' + name))
    