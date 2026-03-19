"""
This script loads QwQ-32B LLM reasoning model, extracts, and stores rationales for each train example.
"""

import re
import json
import argparse
import os
import sys
from pathlib import Path
import tempfile
import requests
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from tqdm.auto import tqdm

from tokenizer import load_math_dataset, load_math_datasets


def _get_vertex_bearer_token() -> str:
    """Get a Bearer token for Vertex OpenAPI via Application Default Credentials."""
    import google.auth
    from google.auth.transport.requests import Request
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(Request())
    return creds.token


# One-shot prompt
def _build_prompt(problem_text: str) -> str:
    return (
        "You are a math reasoning tutor.\n" # give model a persona
        "Your response must be EXACTLY one line and ONLY this XML block:\n" # set tags for model to put response between
        "<rationale>...</rationale>\n\n"
        "Inside <rationale>...</rationale>:\n"
        "- Write 3 to 6 complete sentences.\n" # set limit to reduce costs (still costs quite a bit)
        "- Explain key algebraic steps clearly and concisely.\n" # forgot to change and possibly not optimal for some subsets
        "- The final sentence must include the final answer in LaTeX boxed form: \\boxed{...}.\n"
        "- Do not include any text outside the tags.\n" # minimize costs
        "- Do not include bullet points.\n\n" # for formatting
        "Example 1:\n" # one shot example
        "Problem: For what values of a and b is this piecewise function continuous?\n"
        "Answer:\n"
        "<rationale>For the piecewise function to be continuous, the cases must meet at x=2 and x=-2. At x=2, ax+3 and x-5 must be equal, so 2a+3=-3 and therefore a=-3. At x=-2, x-5 and 2x-b must be equal, so -7=-4-b and therefore b=3. Adding the two values gives a+b=-3+3. Therefore the final value is \\boxed{0}.</rationale>\n\n"
        "Now solve this problem.\n"
        f"Problem: {problem_text}\n" # problem to be answered
        "Answer:" # final answer
    )


# extract rationale from between <rationale> tags
def _extract_rationale(text: str) -> str | None:
    # regular expression to get rationale between specified rationale tags
    rationale = re.search(r"<rationale>(.*?)</rationale>", text, flags=re.DOTALL | re.IGNORECASE)

    # error catching incase LLM does not return rationale in proper format
    if not rationale:
        return None

    return rationale.group(1).strip()


# don't alter args unless necessary
def main():
    parser = argparse.ArgumentParser(description="Extract pseudo-rationales from a teacher LLM and write JSONL.")
    parser.add_argument("--model_name", type=str, default="google/gemini-2.5-flash") # default teach model
    parser.add_argument("--dataset_name", type=str, default="EleutherAI/hendrycks_math") # default math dataset
    parser.add_argument("--subset", type=str, default="algebra") # default single subset for initial testing

    parser.add_argument("--batch_size", type=int, default=4, help="How many rationales to generate per batch") # increase batch size depending on VRAM
    parser.add_argument(
        "--max_parallel_requests",
        type=int,
        default=8,
        help="How many API requests to run concurrently inside each batch.", # speed up extraction process
    )
    parser.add_argument(
        "--request_retries",
        type=int,
        default=3,
        help="Retry attempts for transient API failures (429/5xx).", # keep low for latency but allow for more than one for connection difficulties
    )
    parser.add_argument(
        "--request_timeout_s",
        type=int,
        default=600, # quite a long timeout, feel free to change if not running overnight
        help="HTTP request timeout in seconds (read timeout). Increase if Vertex responses occasionally exceed 180s.",
    )
    parser.add_argument(
        "--subsets", # --subsets "algebra,geometry,intermediate_algebra,number_theory,precalculus" for train
        type=str,
        default=None,
        help='Optional comma-separated subset list (e.g. "algebra,geometry,prealgebra"). If set, overrides --subset.',
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="problem")
    parser.add_argument(
        "--start_idx", # for sharding
        type=int,
        default=0,
        help="Skip the first N dataset examples before processing (useful for sharding/resume).",
    )
    parser.add_argument( # no token limit for rationales if possible (hopefully prompt will do the job)
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum tokens to generate. Omit for backend default (no explicit cap from this script).",
    )
    parser.add_argument("--limit", type=int, default=50, help="How many examples to process (for sanity-checking).")
    parser.add_argument(
        "--out_jsonl",
        type=str,
        default="pseudo_rationales.jsonl",
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Path(s) to input JSONL (local or gs://..., one or more). "
            "When set, overrides --dataset_name / --subset / --subsets and loads "
            "problems directly from these files."
        ),
    )
    parser.add_argument(
        "--gcp_project",
        type=str,
        default="cs224n-dapo-distill",
        help="GCP project id used by the GCS client (only needed when writing to gs://...).",
    )
    parser.add_argument(
        "--vertex_location",
        type=str,
        default="us-central1",
        help="Vertex region used when OPENAI_BASE_URL is not set.",
    )
    args = parser.parse_args()


    # easy helper to parse gcs uri and grab bucket and blob names
    def _parse_gcs_uri(uri: str) -> tuple[str, str]:
        # gs://bucket/path/to/object
        no_scheme = uri[len("gs://") :]
        bucket, _, blob = no_scheme.partition("/")
        if not bucket or not blob:
            raise ValueError(f"Invalid GCS URI: {uri}")
        return bucket, blob

    # get bucket and upload rationales to blob
    def _upload_to_gcs(local_path: Path, gcs_uri: str) -> None:
        try:
            from google.cloud import storage  # type: ignore
        except Exception:
            # Fallback for environments without google-cloud-storage.
            # Requires `gsutil` and valid gcloud auth on the machine.
            try:
                subprocess.run(
                    ["gsutil", "-q", "cp", str(local_path), gcs_uri],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return
            except Exception as e:
                raise ImportError(
                    "To write to GCS, install google-cloud-storage (`pip install google-cloud-storage`) "
                    "or ensure `gsutil` is installed and authenticated."
                ) from e

        bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
        client = storage.Client(project=args.gcp_project)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(str(local_path))

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # # grab tokenizer for QwQ-32B
    # tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    #
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name,
    #     torch_dtype="auto",
    #     device_map="auto" if device == "cuda" else None,
    # )
    #
    # # make sure model is on cpu
    # if device != "cuda":
    #     model = model.to(device)
    #
    # Get GCP project if not provided
    if not args.gcp_project:
        gcp_project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not gcp_project:
            try:
                result = subprocess.run(
                    ["gcloud", "config", "get-value", "project"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                gcp_project = result.stdout.strip()
            except:
                gcp_project = "cs224n-dapo-distill"  # fallback
    else:
        gcp_project = args.gcp_project
    
    # Vertex AI OpenAPI endpoint (same as generate_thinkprm_training_data.py)
    api_base_url = (
        f"https://{args.vertex_location}-aiplatform.googleapis.com"
        f"/v1/projects/{gcp_project}/locations/{args.vertex_location}/endpoints/openapi"
    )
    api_url = f"{api_base_url}/chat/completions"
    
    # Format model name for Vertex (same as generate_thinkprm_training_data.py)
    model_for_api = args.model_name
    if "/" not in model_for_api:
        model_for_api = f"google/{model_for_api}"
    # Keep existing special case handling for qwen models if needed
    elif model_for_api.lower().startswith("qwen/"):
        pass  # Already formatted correctly

    # Load dataset: from explicit JSONL path(s) or HuggingFace
    if args.input_jsonl:
        from google.cloud import storage as _gcs

        def _download_gcs(uri: str) -> str:
            no_scheme = uri[len("gs://"):]
            bucket_name, _, blob_name = no_scheme.partition("/")
            client = _gcs.Client(project=gcp_project)
            blob = client.bucket(bucket_name).blob(blob_name)
            tmp_path = Path(tempfile.mkdtemp(prefix="input_jsonl_")) / Path(blob_name).name
            blob.download_to_filename(str(tmp_path))
            return str(tmp_path)

        uris = [u.strip() for u in args.input_jsonl if u.strip()]
        local_paths = [_download_gcs(u) if u.startswith("gs://") else u for u in uris]
        rows = []
        for path in local_paths:
            with open(path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if _line:
                        rows.append(json.loads(_line))
        from datasets import Dataset as _Dataset
        loaded_dataset = _Dataset.from_list(rows)
    else:
        # helper function from tokenizer.py
        subset_list = None
        if args.subsets:
            subset_list = [s.strip() for s in args.subsets.split(",") if s.strip()]

        if subset_list:
            loaded_dataset = load_math_datasets(
                dataset_name=args.dataset_name,
                subsets=subset_list,
                split=args.split,
            )
        else:
            loaded_dataset = load_math_dataset(
                dataset_name=args.dataset_name,
                subset=args.subset,
                split=args.split,
            )

    # Optional slicing for sharding/resume.
    # Note: we slice the HF Dataset (small: ~5.5k rows) to avoid O(N) manual skipping.
    try:
        ds_len = len(loaded_dataset)
    except Exception:
        ds_len = None

    start_idx = max(0, int(args.start_idx))
    if ds_len is not None and start_idx > ds_len:
        print(f"WARNING: --start_idx={start_idx} >= dataset size {ds_len}; nothing to do.", file=sys.stderr)
        return

    if ds_len is not None and (start_idx > 0 or args.limit is not None):
        end_idx = ds_len if args.limit is None else min(ds_len, start_idx + int(args.limit))
        if start_idx > 0 or end_idx < ds_len:
            idxs = list(range(start_idx, end_idx))
            loaded_dataset = loaded_dataset.select(idxs)
            # After slicing, process the entire sliced dataset.
            args.limit = len(loaded_dataset)

    def _request_rationale(prompt: str) -> str:
        """Request rationale from Vertex AI API (same pattern as generate_thinkprm_training_data.py)."""
        payload = {
            "model": model_for_api,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        # following should be None for most cases
        if args.max_new_tokens is not None:
            payload["max_tokens"] = args.max_new_tokens

        last_error: Exception | None = None
        for attempt in range(args.request_retries + 1):
            try:
                bearer_token = _get_vertex_bearer_token()
                headers = {
                    "Authorization": f"Bearer {bearer_token}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=payload,
                    timeout=args.request_timeout_s,
                )

            except (requests.exceptions.RequestException, TimeoutError) as e:
                # Network / timeout errors are retryable.
                last_error = e
                if attempt >= args.request_retries:
                    print(
                        f"ERROR: request failed after retries due to network/timeout: {e}",
                        file=sys.stderr,
                        flush=True,
                    )
                    return ""
                time.sleep(min(2 ** attempt, 8))
                continue
                
            if response.ok:
                response_json = response.json()
                choice = response_json.get("choices", [{}])[0]
                message = choice.get("message", {})
                return message.get("content", "")

            # Handle errors (same pattern as generate_thinkprm_training_data.py)
            status = response.status_code
            is_retryable = status in (401, 403, 429) or 500 <= status < 600
            last_error = RuntimeError(f"API request failed ({status}): {response.text}")
            
            if not is_retryable or attempt >= args.request_retries:
                if attempt >= args.request_retries:
                    print(
                        f"ERROR: Failed to generate rationale after {args.request_retries} retries: {last_error}",
                        file=sys.stderr,
                        flush=True,
                    )
                return ""
            
            time.sleep(min(2 ** attempt, 8))
        
        # Shouldn't happen, but keep extraction running.
        if last_error:
            print(f"ERROR: request failed: {last_error}", file=sys.stderr, flush=True)
        return ""

    # If output is gs://..., write locally first, then upload at the end.
    out_jsonl = args.out_jsonl
    
    # check if gs://...
    out_is_gcs = out_jsonl.startswith("gs://")

    # if output to google cloud storage then just make a temp directory
    # else make a permanent directory
    if out_is_gcs:
        tmp_dir = Path(tempfile.mkdtemp(prefix="pseudo_rationales_"))
        out_path = tmp_dir / "pseudo_rationales.jsonl"
    else:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # now grab problem batches
    n_rationales = 0
    iterations = iter(loaded_dataset)
    try:
        dataset_size = len(loaded_dataset)
    except Exception:
        dataset_size = None

    if args.limit is None:
        progress_total = dataset_size
    elif dataset_size is None:
        progress_total = args.limit
    else:
        progress_total = min(args.limit, dataset_size)

    # implement rationale extraction loop

    with out_path.open("w", encoding="utf-8") as f, tqdm(
        total=progress_total,
        desc="Generating rationales",
        unit="example",
        dynamic_ncols=True,
    ) as pbar:
        rationale_limit = args.limit

        while True:
            if rationale_limit is not None and n_rationales >= rationale_limit:
                break
            
            # manually build batches to gen rationales for
            batch_examples = []
            while len(batch_examples) < args.batch_size: 
                if rationale_limit is not None and (n_rationales + len(batch_examples)) >= rationale_limit:
                    break
                
                try:
                    example = next(iterations)
                except StopIteration:
                    break
                    
                # text field should just be "problem" default
                problem_text = example.get(args.text_field)

                if not problem_text:
                    continue

                batch_examples.append(example)

            # exit loop once batches are exhausted
            if not batch_examples:
                break

            # build prompts for QwQ model
            prompts = [_build_prompt(example[args.text_field]) for example in batch_examples]

            # texts = [
            #     tokenizer.apply_chat_template(
            #         [{"role" : "user", "content" : prompt}],
            #         tokenize=False,
            #         add_generation_prompt=True
            #     )
            #     for prompt in prompts
            # ]
            #
            # model_inputs = tokenizer(
            #     texts,
            #     return_tensors="pt",
            #     padding=True,
            #     truncation=True,
            #     max_length=args.max_input_length,
            # ).to(model.device)
            #
            # # per example prompt lengths so we can slice generated output correctly
            # input_lens = model_inputs["attention_mask"].sum(dim=1) # shape here is [batch]
            # generated_ids = model.generate(**model_inputs, max_new_tokens=args.max_new_tokens)
            #
            # API generation with true request concurrency.

            # where to store resposnes as they arrive, create list of empty strings equal to num prompts (length of prompt list)
            response_texts = [""] * len(prompts)

            max_workers = max(1, min(args.max_parallel_requests, len(prompts)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                fut_to_idx = {executor.submit(_request_rationale, prompt): i for i, prompt in enumerate(prompts)}

                # Use as_completed so one very slow request doesn't block the whole batch.
                # If some futures never complete (extreme stalls), we cancel them after a ceiling.
                try:
                    for fut in as_completed(fut_to_idx, timeout=args.request_timeout_s + 30):
                        i = fut_to_idx[fut]
                        try:
                            response_texts[i] = fut.result()
                        except Exception as e:
                            print(f"ERROR: worker failed: {e}", file=sys.stderr, flush=True)
                            response_texts[i] = ""
                except Exception:
                    # Timeout waiting for the full batch to complete; cancel remaining futures.
                    for fut, i in fut_to_idx.items():
                        if not fut.done():
                            fut.cancel()
                            response_texts[i] = ""

            # now decode and write each rationale to jsonl

            # for each example in batch
            for i, example in enumerate(batch_examples):
                response_text = response_texts[i]

                rationale = _extract_rationale(response_text)

                row = {
                    'dataset': args.dataset_name,
                    'subset': example.get("subset", args.subset),
                    'split': args.split,
                    'id': example.get("id", None),
                    'type': example.get("type", None),
                    'problem': example.get(args.text_field),
                    'rationale': rationale,
                    'raw_response': response_text
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                n_rationales += 1
                pbar.update(1)

    
    # finally upload the rationales to google cloud storage
    if out_is_gcs:
        _upload_to_gcs(out_path, args.out_jsonl)
        print(f"Wrote {n_rationales} examples to {args.out_jsonl} (uploaded from {out_path.resolve()})")
    else:
        print(f"Wrote {n_rationales} examples to {out_path.resolve()}")


if __name__ == "__main__":
    main()

