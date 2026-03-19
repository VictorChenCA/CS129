"""
Quick sanity-check script for a processed HF dataset produced by build_train_set.py.

It can load from:
  - local directory (datasets.save_to_disk output)
  - GCS prefix (gs://bucket/prefix/) containing the save_to_disk artifacts
"""


# SCRIPT FOR QUICK DATASET SANITY CHECK
import argparse
import tempfile
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    no_scheme = uri[len("gs://") :]
    bucket, _, blob = no_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def _download_gcs_prefix_to_dir(gcs_prefix: str, local_dir: Path, *, project: str) -> None:
    """Download all objects under a GCS prefix into local_dir, preserving relative paths."""
    try:
        from google.cloud import storage  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Install google-cloud-storage: `pip install google-cloud-storage`") from e

    bucket_name, prefix = _parse_gcs_uri(gcs_prefix)
    if prefix and not prefix.endswith("/"):
        prefix = prefix + "/"

    client = storage.Client(project=project)
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=prefix)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        rel = blob.name[len(prefix) :] if prefix else blob.name
        dst = local_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(dst))


def _resolve_dataset_path(dataset_dir: str, gcp_project: str) -> str:
    if dataset_dir.startswith("gs://"):
        temp_dir = Path(tempfile.mkdtemp(prefix="processed_ds_dl_"))
        _download_gcs_prefix_to_dir(dataset_dir, temp_dir, project=gcp_project)
        return str(temp_dir)
    return dataset_dir


def _get_train_split(ds_obj: Dataset | DatasetDict) -> Dataset:
    if isinstance(ds_obj, DatasetDict):
        if "train" in ds_obj:
            return ds_obj["train"]
        first_split = next(iter(ds_obj.keys()))
        return ds_obj[first_split]
    return ds_obj


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview a processed distillation training dataset.")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Local path or gs:// prefix from build_train_set.py")
    parser.add_argument("--gcp_project", type=str, default="cs224n-dapo-distill")
    parser.add_argument("--n", type=int, default=2, help="How many examples to print")
    args = parser.parse_args()

    dataset_path = _resolve_dataset_path(args.dataset_dir, args.gcp_project)
    ds_obj = load_from_disk(dataset_path)
    ds = _get_train_split(ds_obj)

    print(ds)
    print("num_rows:", len(ds))
    print("columns:", ds.column_names)

    n = max(0, min(args.n, len(ds)))
    for i in range(n):
        ex = ds[i]
        problem = ex.get("problem", "")
        rationale = ex.get("rationale", None)
        print(f"\n--- example {i} ---")
        print("problem:", (problem[:300] + ("..." if len(problem) > 300 else "")))
        print("rationale:", rationale if (rationale is not None and str(rationale).strip() != "") else "<missing>")


if __name__ == "__main__":
    main()

