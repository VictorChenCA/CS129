"""
GCS utility functions for Google Cloud Storage operations.
Shared utilities for downloading, uploading, and dataset loading.
"""

import os
from pathlib import Path
from typing import Optional, Union

try:
    from datasets import Dataset, DatasetDict
except ImportError:
    Dataset = None
    DatasetDict = None


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """
    Parse gs://bucket/path into (bucket, path).
    
    Args:
        uri: GCS URI (e.g., gs://bucket/path/to/file)
    
    Returns:
        Tuple of (bucket_name, blob_path)
    
    Raises:
        ValueError: If URI is invalid
    """
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    no_scheme = uri[len("gs://") :]
    bucket, _, blob = no_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def download_gcs_prefix_to_dir(
    gcs_prefix: str,
    local_dir: Path,
    *,
    project: Optional[str] = None
) -> None:
    """
    Download all objects under GCS prefix to local directory.
    
    Args:
        gcs_prefix: GCS URI prefix (e.g., gs://bucket/path/to/model/)
        local_dir: Local directory to download files to
        project: GCP project name (defaults to GOOGLE_CLOUD_PROJECT env var or gcloud config)
    
    Raises:
        ImportError: If google-cloud-storage is not installed
        ValueError: If gcs_prefix is not a valid GCS URI or project is not specified
    """
    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError("Install google-cloud-storage: `pip install google-cloud-storage`") from e
    
    bucket_name, prefix = parse_gcs_uri(gcs_prefix)
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"
    
    # Determine project if not provided
    if project is None:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if project is None:
            # Try to get from gcloud config
            import subprocess
            try:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True,
                    text=True,
                    check=False
                )
                if result.returncode == 0:
                    project = result.stdout.strip()
            except (FileNotFoundError, subprocess.SubprocessError):
                pass
    
    if project is None:
        raise ValueError(
            "GCP project not specified. Provide project parameter or set GOOGLE_CLOUD_PROJECT env var."
        )
    
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix)
    
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    for blob in blobs:
        # Skip "directory marker" blobs
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) :] if prefix else blob.name
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))
        downloaded_count += 1
    
    print(f"Downloaded {downloaded_count} files from {gcs_prefix} to {local_dir}")


def upload_dir_to_gcs(
    local_dir: Path,
    gcs_uri_prefix: str,
    *,
    project: str
) -> None:
    """
    Upload a local directory recursively to a GCS prefix.
    
    Args:
        local_dir: Local directory to upload
        gcs_uri_prefix: GCS URI prefix (e.g., gs://bucket/models/thinkprm_finetuned/)
        project: GCP project name
    
    Raises:
        ImportError: If google-cloud-storage is not installed
        ValueError: If gcs_uri_prefix is not a valid GCS URI
    """
    try:
        from google.cloud import storage
    except ImportError as e:
        raise ImportError("Install google-cloud-storage: `pip install google-cloud-storage`") from e
    
    if not str(gcs_uri_prefix).startswith("gs://"):
        raise ValueError(f"Expected gs://... URI for gcs_uri_prefix, got: {gcs_uri_prefix}")
    
    bucket_name, blob_prefix = parse_gcs_uri(
        gcs_uri_prefix if gcs_uri_prefix.endswith("/") else gcs_uri_prefix + "/"
    )
    
    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    
    local_dir = local_dir.resolve()
    for root, _, files in os.walk(local_dir):
        for fn in files:
            local_path = Path(root) / fn
            rel = local_path.relative_to(local_dir).as_posix()
            blob_name = f"{blob_prefix.rstrip('/')}/{rel}"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))


def load_dataset_with_fallback(dataset_path: str) -> Union[Dataset, DatasetDict]:
    """
    Load a dataset saved via HuggingFace `save_to_disk`.
    
    If feature metadata parsing fails (version mismatch), fallback to reading
    the first Arrow shard directly.
    
    Args:
        dataset_path: Path to dataset directory
    
    Returns:
        Dataset or DatasetDict
    
    Raises:
        ImportError: If datasets or pyarrow is not installed
        FileNotFoundError: If dataset path doesn't exist
    """
    if Dataset is None:
        raise ImportError("Install datasets: `pip install datasets`")
    
    try:
        from datasets import load_from_disk
    except ImportError as e:
        raise ImportError("Install datasets: `pip install datasets`") from e
    
    try:
        return load_from_disk(dataset_path)
    except (TypeError, Exception) as e:
        # If load_from_disk fails, try reading Arrow files directly
        try:
            import pyarrow.ipc as pa_ipc
        except ImportError as import_err:
            raise ImportError("Install pyarrow: `pip install pyarrow`") from import_err
        
        ds_dir = Path(dataset_path)
        arrow_files = sorted(ds_dir.glob("data-*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No Arrow files found in {dataset_path}") from e
        
        try:
            with pa_ipc.open_file(str(arrow_files[0])) as reader:
                table = reader.read_all()
        except Exception:
            with pa_ipc.open_stream(str(arrow_files[0])) as reader:
                table = reader.read_all()
        
        table = table.replace_schema_metadata(None)
        return Dataset.from_pandas(table.to_pandas(), preserve_index=False)


# Backward compatibility: also export with underscores for files that use that convention
_parse_gcs_uri = parse_gcs_uri
_download_gcs_prefix_to_dir = download_gcs_prefix_to_dir
_upload_dir_to_gcs = upload_dir_to_gcs
_load_dataset_with_fallback = load_dataset_with_fallback
