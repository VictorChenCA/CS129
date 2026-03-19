"""
This script contains methods to load the Hendrycks math dataset and tokenize each example.

"""

from datasets import load_dataset, DatasetDict, concatenate_datasets


# load hendrycks math dataset
def load_math_dataset(
    dataset_name: str = "EleutherAI/hendrycks_math",
    subset: str | None = "algebra",
    split: str = "train",
):
    """
    Load a math dataset from Hugging Face Datasets.

    Defaults to Hendrycks MATH algebra/train.
    """

    # hendrycks contains precalc, algebra, geometry, etc. (choose one)
    if subset is None:
        # loads all subsets
        return load_dataset(dataset_name, split=split)

    # return singular subset=subset
    return load_dataset(dataset_name, subset, split=split)


# loads multiple subsets instead of just one
def load_math_datasets(
    dataset_name: str,
    subsets: list[str],
    split: str,
):
    """
    Load multiple Hendrycks MATH subsets and concatenate them into one Dataset.

    Adds a `subset` column to each example so downstream joins are unambiguous.
    """

    # temp var to store the subsets being loaded via load_math_dataset (see above)
    parts = []
    for s in subsets:
        ds = load_math_dataset(dataset_name=dataset_name, subset=s, split=split)
        ds = ds.map(lambda ex, _s=s: {"subset": _s})
        parts.append(ds)
    return concatenate_datasets(parts)


# tokenize the dataset
def tokenize_math(
    model_name: str,
    dataset_name: str = "EleutherAI/hendrycks_math",
    subset: str | list[str] | None = "algebra",
    split: str = "train",
    text_field: str = "problem",
    max_length: int = 1024,
):
    """
    Returns a HF Dataset with tokenized `text_field`.

    Note: returns dataset you can iterate over with .map (see huggingface dataframes).
    """

    # if subset is a list of subsets: load and concatenate all of them
    if isinstance(subset, list):
        dataset = load_math_datasets(dataset_name=dataset_name, subsets=subset, split=split)
    else:
        dataset = load_math_dataset(dataset_name=dataset_name, subset=subset, split=split)


    # use hugging face tokenizer for gemini or qwen etc.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # can be condensed into lambda but written out as full function for error catching
    def _tok(ex):
        text = ex.get(text_field)
        if text is None:
            raise KeyError(
                f"Expected field `{text_field}` in dataset example. Available keys: {list(ex.keys())}"
            )
        return tokenizer(
            text,
            truncation=True,
            max_length=max_length,
        )

    # map is hugging face function which applies _tok to every example in dataset
    return dataset.map(_tok, remove_columns=[])

