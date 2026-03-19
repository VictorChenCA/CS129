"""
Upload stratified MATH dataset splits to GCS.

Uploads:
  dataset/stratified/          -> gs://cs224n-project-data/stratified_math/stratified/
  dataset/stratified_heldout/  -> gs://cs224n-project-data/stratified_math/stratified_heldout/

Usage:
  python3 upload_to_gcs.py
  python3 upload_to_gcs.py --dry_run
  python3 upload_to_gcs.py --gcp_project cs224n-dapo-distill --gcs_prefix gs://cs224n-project-data/stratified_math/
"""

import argparse
import os
from pathlib import Path


def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    no_scheme = uri[len("gs://"):]
    bucket, _, blob = no_scheme.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, blob


def _upload_files_to_gcs(
    local_dir: Path,
    gcs_prefix: str,
    *,
    project: str,
    glob: str = "*",
    dry_run: bool = False,
) -> list[str]:
    """
    Upload all files matching `glob` in `local_dir` (non-recursive) to `gcs_prefix`.
    Returns list of GCS URIs written.
    """
    if not dry_run:
        try:
            from google.cloud import storage  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install google-cloud-storage: `pip install google-cloud-storage`"
            ) from e

    prefix = gcs_prefix.rstrip("/") + "/"
    bucket_name, blob_prefix = _parse_gcs_uri(prefix)

    if not dry_run:
        client = storage.Client(project=project)
        bucket = client.bucket(bucket_name)

    uris = []
    for local_path in sorted(local_dir.glob(glob)):
        if not local_path.is_file():
            continue
        blob_name = blob_prefix + local_path.name
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        size_kb = local_path.stat().st_size / 1024
        if dry_run:
            print(f"  [dry-run] {local_path}  ->  {gcs_uri}  ({size_kb:.1f} KB)")
        else:
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(local_path))
            print(f"  uploaded  {local_path.name}  ->  {gcs_uri}  ({size_kb:.1f} KB)")
        uris.append(gcs_uri)

    return uris


def main():
    parser = argparse.ArgumentParser(description="Upload stratified MATH splits to GCS.")
    parser.add_argument("--gcp_project", type=str, default="cs224n-dapo-distill")
    parser.add_argument(
        "--gcs_prefix",
        type=str,
        default="gs://cs224n-project-data/stratified_math/",
        help="GCS prefix under which stratified/ and stratified_heldout/ subdirs are created.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print what would be uploaded without uploading.")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    uploads = [
        (script_dir / "stratified",         args.gcs_prefix.rstrip("/") + "/stratified/"),
        (script_dir / "stratified_heldout", args.gcs_prefix.rstrip("/") + "/stratified_heldout/"),
    ]

    total = 0
    for local_dir, gcs_dir in uploads:
        print(f"\n{local_dir.name}/  ->  {gcs_dir}")
        uris = _upload_files_to_gcs(
            local_dir,
            gcs_dir,
            project=args.gcp_project,
            dry_run=args.dry_run,
        )
        total += len(uris)

    print(f"\n{'[dry-run] ' if args.dry_run else ''}Done. {total} file(s) processed.")


if __name__ == "__main__":
    main()
