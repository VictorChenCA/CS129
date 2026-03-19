"""
OUR DEFINITIONS OF COHERENCE AND RELEVANCE FROM THE CaSE Paper;
1) relevance: whether the current step is based on a correct understanding of the question, contributing meaningfully to solving it without redundancy.
2) coherence: whether the current step follows logically and naturally from the previous steps.

https://arxiv.org/pdf/2510.20603v1 

This script implements an operational version of CaSE reasoning evaluation.
Each reasoning step is evaluated for relevance and coherence using only the
question and previous steps, never future steps. This avoids future-step
leakage and follows the core principle of the CaSE protocol. The script uses
Vertex AI Gemini as an LLM judge and includes rate limiting and retry logic
for practical large-scale evaluation.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel

DEFAULT_PROJECT = "cs224n-dapo-distill"
DEFAULT_LOCATION = "us-central1"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_USER_EMAIL = "maxlrodwork@gmail.com"
JUDGE_BACKEND = "vertex_ai_gemini"

# don't treat failures the same
class JudgeResponseError(Exception):
    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


@dataclass
class RetryConfig:
    max_retries: int
    initial_retry_delay: float
    max_retry_delay: float


@dataclass
class Example:
    example_id: str
    question: str
    rationale: str
    answer: Optional[str] = None


@dataclass
class StepEvaluation:
    step_index: int
    step_text: str
    prior_context: list[str]
    relevance_score: int
    relevance_reason: str
    coherence_score: int
    coherence_reason: str
    judge_model: str
    judge_backend: str


@dataclass
class ExampleResult:
    id: str
    question: str
    answer: Optional[str]
    rationale: str
    steps: list[str]
    step_evaluations: list[StepEvaluation]
    example_metrics: Optional[dict[str, Any]]
    error: Optional[dict[str, Any]]


# ensures that we do not surpase rate limits
class RateLimiter:
    def __init__(self, requests_per_minute: int, verbose: bool = False):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self._min_interval = 60.0 / float(requests_per_minute)
        self._last_request = 0.0

        # get threading lock
        self._lock = threading.Lock()
        self.verbose = verbose

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_request > 0:
                sleep_s = self._last_request + self._min_interval - now
                if sleep_s > 0:
                    if self.verbose:
                        print(f"[rate_limit] sleeping {sleep_s:.2f}s", flush=True)
                    time.sleep(sleep_s)
            self._last_request = time.monotonic()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CaSE reasoning-quality evaluator with Vertex Gemini judge.")
    parser.add_argument("--input_jsonl", type=str, default=None)
    parser.add_argument("--use_default_example", action="store_true")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--step_split_mode", type=str, default="auto", choices=("auto", "newline", "sentence", "numbered"))
    parser.add_argument("--project", type=str, default=DEFAULT_PROJECT)
    parser.add_argument("--location", type=str, default=DEFAULT_LOCATION)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--requests_per_minute", type=int, default=60)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--initial_retry_delay", type=float, default=1.0)
    parser.add_argument("--max_retry_delay", type=float, default=30.0)
    parser.add_argument("--timeout_seconds", type=float, default=60.0)
    parser.add_argument("--example_timeout_seconds", type=float, default=120.0)
    parser.add_argument("--id_field", type=str, default="id")
    parser.add_argument("--question_field", type=str, default="question")
    parser.add_argument("--rationale_field", type=str, default="rationale")
    parser.add_argument("--answer_field", type=str, default="answer")
    parser.add_argument("--fail_on_judge_error", action="store_true")
    return parser.parse_args()


def _warn(message: str) -> None:
    print(f"[warning] {message}", file=sys.stderr, flush=True)


# default example with full coherency and relevancy
def build_default_example() -> Example:
    return Example(
        example_id="default_example",
        question="A school is buying notebooks. Each pack has 6 notebooks. If the school needs 54 notebooks, how many packs should it buy?",
        rationale="Each pack contains 6 notebooks. The school needs 54 notebooks total. To find the number of packs, divide 54 by 6. 54 / 6 = 9. Therefore, the school should buy 9 packs.",
        answer="9",
    )


# later will load in jsonl examples from model and model+verifier outputs.
def load_jsonl_examples(
    input_path: Path,
    id_field: str,
    question_field: str,
    rationale_field: str,
    answer_field: str,
    max_examples: Optional[int],
    verbose: bool,
) -> tuple[list[Example], int, int]:
    examples: list[Example] = []
    total_loaded = 0
    skipped = 0
    generated_id = 1

    # open with read priviledges 
    with input_path.open("r", encoding="utf-8") as f:
        for line_idx, raw in enumerate(f, start=1): # start=1 for index count start at 1 instead of 0!
            line = raw.strip() # strip white space

            if not line:
                continue

            total_loaded += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                skipped += 1
                _warn(f"line {line_idx}: malformed JSON ({exc}); skipping")
                continue

            # expects question per row along with rationale-based solution (along with question id and answer columns)
            question = str(row.get(question_field, "") or "").strip()
            rationale = str(row.get(rationale_field, "") or "").strip()
            if not question:
                skipped += 1
                _warn(f"line {line_idx}: missing/blank question; skipping")
                continue
            if not rationale:
                skipped += 1
                _warn(f"line {line_idx}: missing/blank rationale; skipping")
                continue

            raw_id = row.get(id_field)
            example_id = str(raw_id).strip() if raw_id not in (None, "") else f"example_{generated_id:04d}"
            if raw_id in (None, ""):
                generated_id += 1

            answer_val = row.get(answer_field)
            answer = None if answer_val is None else str(answer_val)

            # also get answer for correctness
            examples.append(Example(example_id=example_id, question=question, rationale=rationale, answer=answer))

            if max_examples is not None and len(examples) >= max_examples:
                if verbose:
                    print(f"[info] reached --max_examples={max_examples}", flush=True)
                break
    return examples, total_loaded, skipped


def split_numbered_steps(rationale: str) -> list[str]:
    text = rationale.replace("\r\n", "\n")
    lines = text.split("\n")

    # find punctuation and split lines based on that
    pattern = re.compile(r"^\s*(?:step\s*\d+\s*[:\-\)]|\d+\s*[\).\:]|\(\d+\))\s*", re.IGNORECASE)
    starts = [i for i, line in enumerate(lines) if pattern.match(line)]
    if len(starts) < 2:
        return []

    chunks: list[list[str]] = []
    current: list[str] = []
    started = False
    for line in lines:
        if pattern.match(line):
            if current:
                chunks.append(current)
            current = [line.strip()]
            started = True
        elif started and line.strip():
            current.append(line.strip())
    if current:
        chunks.append(current)

    # Return each "chunk" or solution step for rationales
    return [" ".join(chunk).strip() for chunk in chunks if chunk]


def split_newline_steps(rationale: str) -> list[str]:
    text = rationale.replace("\r\n", "\n").strip()
    if not text:
        return []
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    if len(blocks) >= 2:
        return blocks
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    return lines if len(lines) >= 2 else []


def split_sentence_steps(rationale: str) -> list[str]:
    text = rationale.strip()
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text) if p.strip()]
    if len(parts) <= 1:
        parts = [p.strip() for p in re.split(r";\s+", text) if p.strip()]
    return parts


def merge_tiny_fragments(steps: list[str]) -> list[str]:
    merged: list[str] = []
    for step in steps:
        s = step.strip()
        if not s:
            continue

        # ensure steps weren't fragmented, and merge fragments if necessary
        # less than or equal to two words and less than or equal to 22 characters
        is_tiny = (len(s.split()) <= 2 and len(s) <= 22) or len(s) <= 10
        if is_tiny and merged and not re.search(r"\d", s):
            merged[-1] = f"{merged[-1]} {s}".strip()
        else:
            merged.append(s)
    return merged


def split_rationale_into_steps(rationale: str, mode: str) -> list[str]:
    if mode == "numbered":
        return merge_tiny_fragments(split_numbered_steps(rationale))
    if mode == "newline":
        return merge_tiny_fragments(split_newline_steps(rationale))
    if mode == "sentence":
        return merge_tiny_fragments(split_sentence_steps(rationale))
    for splitter in (split_numbered_steps, split_newline_steps, split_sentence_steps):
        steps = merge_tiny_fragments(splitter(rationale))
        if len(steps) >= 2:
            return steps
    fallback = rationale.strip()
    return [fallback] if fallback else []


def build_case_prompt(question: str, prior_steps: list[str], current_step: str) -> str:
    prior = "\n".join(f"{i+1}. {s}" for i, s in enumerate(prior_steps)) if prior_steps else "EMPTY"
    return (
        "SYSTEM:\n"
        "You are a careful evaluator of reasoning quality.\n\n"
        "Evaluate ONLY the CURRENT reasoning step.\n\n"
        "You are given:\n- a question\n- previous reasoning steps\n- the current reasoning step\n\n"
        "Do not use any future reasoning steps.\n"
        "Do not infer information from steps that are not shown.\n"
        "Judge only the CURRENT_STEP using the QUESTION and PREVIOUS_REASONING_STEPS.\n\n"
        "Evaluate:\n" # KEY EVALUATION CRITERIA DEFINITIONS FROM PAPER RIGHT HERE
        "1) relevance: whether the current step is based on a correct understanding of the question, contributing meaningfully to solving it without redundancy.\n"
        "2) coherence: whether the current step follows logically and naturally from the previous steps.\n\n"
        "Return JSON ONLY.\n\n"
        "USER:\n"
        f"QUESTION:\n{question}\n\n"
        f"PREVIOUS_REASONING_STEPS:\n{prior}\n\n"
        f"CURRENT_STEP:\n{current_step}\n\n"
        "Return strict JSON with exactly these keys:\n"
        "{\n"
        '  "relevance_score": 0 or 1,\n'
        '  "relevance_reason": "brief concrete explanation",\n'
        '  "coherence_score": 0 or 1,\n'
        '  "coherence_reason": "brief concrete explanation"\n'
        "}\n\n"
        "Requirements:\n"
        "- scores must be integers 0 or 1\n"
        "- reasons must be concise\n"
        "- return valid JSON only\n"
        "- no markdown fences\n"
        "- no extra commentary\n"
    )

# easy jsonl extract from larger text blob
def extract_json_object(text: str) -> Optional[str]:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def cleanup_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.replace("\ufeff", "").replace("\u200b", "").replace("\u2060", "")
    cleaned = (
        cleaned.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00a0", " ")
    )
    cleaned = re.sub(r"^\s*json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bTrue\b", "true", cleaned)
    cleaned = re.sub(r"\bFalse\b", "false", cleaned)
    cleaned = re.sub(r"\bNone\b", "null", cleaned)
    # Gemini occasionally returns LaTeX-style backslashes inside JSON strings,
    # which makes the payload invalid JSON despite the structure being usable.
    cleaned = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
    open_braces = cleaned.count("{")
    close_braces = cleaned.count("}")
    if open_braces > close_braces and cleaned.lstrip().startswith("{"):
        cleaned = cleaned + ("}" * (open_braces - close_braces))
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)
    return cleaned


# additional fallback normalization (catch any binary form scoring output and re-map to 0/1)
def normalize_score(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    if isinstance(value, float) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"0", "1"}:
            return int(s)
        if s in {"yes", "true"}:
            return 1
        if s in {"no", "false"}:
            return 0
    raise ValueError(f"{field_name} must be binary 0/1, got {value!r}")


def _parse_with_json_or_literal_eval(text: str) -> Optional[dict[str, Any]]:
    candidate = cleanup_json_text(text)
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    try:
        payload = ast.literal_eval(candidate)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _reason_from_pattern(text: str, key: str, next_keys: list[str]) -> Optional[str]:
    if next_keys:
        next_key_pat = "|".join(re.escape(k) for k in next_keys)
        lookahead = rf'(?=,\s*["\']?(?:{next_key_pat})["\']?\s*[:=]|[}}\n\r]|$)'
    else:
        lookahead = r'(?=[}\n\r]|$)'
    pattern = re.compile(
        rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*(?P<quote>["\'])(?P<value>.*?)(?P=quote)\s*{lookahead}',
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    if match:
        return match.group("value").strip()
    return None


def _score_from_pattern(text: str, key: str) -> Optional[str]:
    pattern = re.compile(
        rf'["\']?{re.escape(key)}["\']?\s*[:=]\s*(?P<value>true|false|yes|no|0|1|"0"|"1"|"true"|"false"|"yes"|"no")',
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        return match.group("value").strip().strip('"').strip("'")
    return None


def _regex_extract_payload(text: str) -> Optional[dict[str, Any]]:
    candidate = cleanup_json_text(text)
    relevance_score = _score_from_pattern(candidate, "relevance_score")
    coherence_score = _score_from_pattern(candidate, "coherence_score")
    relevance_reason = _reason_from_pattern(candidate, "relevance_reason", ["coherence_score", "coherence_reason"])
    coherence_reason = _reason_from_pattern(candidate, "coherence_reason", [])
    if not (relevance_score and coherence_score and relevance_reason and coherence_reason):
        return None
    return {
        "relevance_score": relevance_score,
        "relevance_reason": relevance_reason,
        "coherence_score": coherence_score,
        "coherence_reason": coherence_reason,
    }


def _payload_candidates(raw_text: str) -> list[str]:
    text = raw_text.strip()
    candidates: list[str] = []
    if text:
        candidates.append(text)
        candidates.append(cleanup_json_text(text))

    snippet = extract_json_object(text)
    if snippet:
        candidates.append(snippet)
        candidates.append(cleanup_json_text(snippet))

    first = text.find("{")
    last = text.rfind("}")
    if first >= 0 and last > first:
        span = text[first : last + 1]
        candidates.append(span)
        candidates.append(cleanup_json_text(span))

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        norm = candidate.strip()
        if norm and norm not in seen:
            seen.add(norm)
            deduped.append(norm)
    return deduped



def parse_judge_response(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise JudgeResponseError("empty judge output", retryable=True)

    payload: Optional[dict[str, Any]] = None
    parse_errors: list[str] = []
    for candidate in _payload_candidates(text):
        payload = _parse_with_json_or_literal_eval(candidate)
        if payload is not None:
            break
        regex_payload = _regex_extract_payload(candidate)
        if regex_payload is not None:
            payload = regex_payload
            break
        try:
            json.loads(candidate)
        except Exception as exc:
            parse_errors.append(str(exc))

    if payload is None:
        if "{" not in text:
            raise JudgeResponseError("could not extract JSON object", retryable=True)
        detail = parse_errors[-1] if parse_errors else "unparseable judge output"
        raise JudgeResponseError(f"invalid extracted JSON: {detail}", retryable=True)
    for key in ("relevance_score", "relevance_reason", "coherence_score", "coherence_reason"):
        if key not in payload:
            raise JudgeResponseError(f"missing key: {key}", retryable=True)
    try:
        relevance_score = normalize_score(payload["relevance_score"], "relevance_score")
        coherence_score = normalize_score(payload["coherence_score"], "coherence_score")
    except ValueError as exc:
        raise JudgeResponseError(str(exc), retryable=True)
    relevance_reason = str(payload["relevance_reason"]).strip()
    coherence_reason = str(payload["coherence_reason"]).strip()
    if not relevance_reason or not coherence_reason:
        raise JudgeResponseError("empty reason field", retryable=True)
    return {
        "relevance_score": relevance_score,
        "relevance_reason": relevance_reason,
        "coherence_score": coherence_score,
        "coherence_reason": coherence_reason,
    }


def _retryable_google_types() -> tuple[type[BaseException], ...]:
    try:
        from google.api_core import exceptions as gex  # type: ignore

        return (
            gex.ResourceExhausted,
            gex.TooManyRequests,
            gex.ServiceUnavailable,
            gex.DeadlineExceeded,
            gex.InternalServerError,
            gex.Aborted,
            gex.GoogleAPICallError,
        )
    except Exception:
        return ()


RETRYABLE_GOOGLE_TYPES = _retryable_google_types()


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, JudgeResponseError):
        return exc.retryable
    if RETRYABLE_GOOGLE_TYPES and isinstance(exc, RETRYABLE_GOOGLE_TYPES):
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "429",
            "quota",
            "rate limit",
            "resource exhausted",
            "timeout",
            "timed out",
            "503",
            "502",
            "500",
            "unavailable",
            "temporar",
            "deadline exceeded",
        )
    )


def retry_with_backoff(
    operation: Callable[[], dict[str, Any]],
    retry_config: RetryConfig,
    verbose: bool,
    op_name: str,
) -> dict[str, Any]:
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as exc:
            if (not is_retryable_error(exc)) or attempt >= retry_config.max_retries:
                raise
            base = min(retry_config.max_retry_delay, retry_config.initial_retry_delay * (2**attempt))
            delay = base + random.uniform(0.0, 0.25 * base)
            if verbose:
                print(f"[retry] {op_name} attempt={attempt+1} sleep={delay:.2f}s err={exc}", flush=True)
            time.sleep(delay)
            attempt += 1


def init_vertex_model(project: str, location: str, model_name: str) -> GenerativeModel:
    vertexai.init(project=project, location=location)
    return GenerativeModel(model_name)


def _generate_with_timeout(
    model: GenerativeModel,
    prompt: str,
    generation_config: GenerationConfig,
    timeout_seconds: float,
) -> Any:
    kwargs: dict[str, Any] = {"generation_config": generation_config}
    if timeout_seconds > 0:
        kwargs["request_options"] = {"timeout": timeout_seconds}
    try:
        return model.generate_content(prompt, **kwargs)
    except TypeError:
        kwargs.pop("request_options", None)
        return model.generate_content(prompt, **kwargs)


def _extract_response_text(response: Any) -> str:
    direct = getattr(response, "text", None)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    candidates = getattr(response, "candidates", None) or []
    parts: list[str] = []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            txt = getattr(part, "text", None)
            if isinstance(txt, str) and txt.strip():
                parts.append(txt.strip())
    return "\n".join(parts).strip()


def _extract_finish_reason(response: Any) -> Optional[int]:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    finish_reason = getattr(candidates[0], "finish_reason", None)
    try:
        return int(finish_reason)
    except Exception:
        return None


def judge_step_with_gemini(
    model: GenerativeModel,
    question: str,
    prior_steps: list[str],
    current_step: str,
    timeout_seconds: float,
    rate_limiter: RateLimiter,
    retry_config: RetryConfig,
    verbose: bool,
) -> dict[str, Any]:
    prompt = build_case_prompt(question, prior_steps, current_step)
    generation_config = GenerationConfig(
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=8192,
        response_mime_type="application/json",
    )

    def _op() -> dict[str, Any]:
        rate_limiter.acquire()
        response = _generate_with_timeout(model, prompt, generation_config, timeout_seconds)
        finish_reason = _extract_finish_reason(response)
        text = _extract_response_text(response)
        if not text:
            raise JudgeResponseError("empty model output", retryable=True)
        if finish_reason == 2:
            raise JudgeResponseError("judge output hit max_output_tokens", retryable=True)
        parsed = parse_judge_response(text)
        parsed["raw_judge_output"] = text
        return parsed

    return retry_with_backoff(_op, retry_config, verbose, "judge_step_with_gemini")


def compute_example_metrics(step_evals: list[StepEvaluation]) -> dict[str, Any]:
    if not step_evals:
        return {
            "num_steps": 0,
            "step_relevance_mean": 0.0,
            "step_coherence_mean": 0.0,
            "all_steps_relevant": False,
            "all_steps_coherent": False,
        }
    rel = [s.relevance_score for s in step_evals]
    coh = [s.coherence_score for s in step_evals]
    n = len(step_evals)
    return {
        "num_steps": n,
        "step_relevance_mean": sum(rel) / n,
        "step_coherence_mean": sum(coh) / n,
        "all_steps_relevant": all(v == 1 for v in rel),
        "all_steps_coherent": all(v == 1 for v in coh),
    }


def evaluate_example_case(
    example: Example,
    model: GenerativeModel,
    model_name: str,
    step_split_mode: str,
    timeout_seconds: float,
    example_timeout_seconds: float,
    rate_limiter: RateLimiter,
    retry_config: RetryConfig,
    fail_on_judge_error: bool,
    verbose: bool,
) -> ExampleResult:
    steps = split_rationale_into_steps(example.rationale, step_split_mode)
    if not steps:
        err = {"type": "step_split_error", "message": "rationale could not be split into steps"}
        if fail_on_judge_error:
            raise RuntimeError(err["message"])
        return ExampleResult(example.example_id, example.question, example.answer, example.rationale, [], [], None, err)

    if verbose:
        print(f"[example] {example.example_id} steps={len(steps)}", flush=True)

    step_evals: list[StepEvaluation] = []
    example_start = time.monotonic()
    for k, step_text in enumerate(steps):
        if example_timeout_seconds > 0 and (time.monotonic() - example_start) >= example_timeout_seconds:
            err = {
                "type": "judge_timeout",
                "message": f"example exceeded example_timeout_seconds={example_timeout_seconds}",
                "step_index": k,
            }
            if fail_on_judge_error:
                raise RuntimeError(err["message"])
            _warn(f"skipping example {example.example_id}: {err['message']}")
            return ExampleResult(example.example_id, example.question, example.answer, example.rationale, steps, step_evals, None, err)
        prior = steps[:k]
        try:
            out = judge_step_with_gemini(
                model=model,
                question=example.question,
                prior_steps=prior,
                current_step=step_text,
                timeout_seconds=timeout_seconds,
                rate_limiter=rate_limiter,
                retry_config=retry_config,
                verbose=verbose,
            )
        except Exception as exc:
            err = {"type": "judge_error", "message": str(exc), "step_index": k}
            if fail_on_judge_error:
                raise RuntimeError(f"judge failed at example={example.example_id}, step={k}: {exc}") from exc
            _warn(f"skipping example {example.example_id}: judge failed at step {k}: {exc}")
            return ExampleResult(example.example_id, example.question, example.answer, example.rationale, steps, step_evals, None, err)

        step_eval = StepEvaluation(
            step_index=k,
            step_text=step_text,
            prior_context=prior,
            relevance_score=out["relevance_score"],
            relevance_reason=out["relevance_reason"],
            coherence_score=out["coherence_score"],
            coherence_reason=out["coherence_reason"],
            judge_model=model_name,
            judge_backend=JUDGE_BACKEND,
        )
        step_evals.append(step_eval)
        if verbose:
            print(
                f"  [step {k}] rel={step_eval.relevance_score} coh={step_eval.coherence_score} "
                f"| rel_reason={step_eval.relevance_reason} | coh_reason={step_eval.coherence_reason}",
                flush=True,
            )

    metrics = compute_example_metrics(step_evals)
    return ExampleResult(example.example_id, example.question, example.answer, example.rationale, steps, step_evals, metrics, None)


def aggregate_dataset_metrics(results: list[ExampleResult], total_loaded: int, skipped_load: int) -> dict[str, Any]:
    ok = [r for r in results if r.error is None and r.example_metrics is not None]
    failed_eval = [r for r in results if r.error is not None]

    step_rel: list[int] = []
    step_coh: list[int] = []
    ex_rel: list[float] = []
    ex_coh: list[float] = []
    ex_all_rel: list[bool] = []
    ex_all_coh: list[bool] = []

    for result in ok:
        assert result.example_metrics is not None
        ex_rel.append(float(result.example_metrics["step_relevance_mean"]))
        ex_coh.append(float(result.example_metrics["step_coherence_mean"]))
        ex_all_rel.append(bool(result.example_metrics["all_steps_relevant"]))
        ex_all_coh.append(bool(result.example_metrics["all_steps_coherent"]))
        for s in result.step_evaluations:
            step_rel.append(int(s.relevance_score))
            step_coh.append(int(s.coherence_score))

    def mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    def frac_true(vals: list[bool]) -> float:
        return sum(1 for v in vals if v) / len(vals) if vals else 0.0

    return {
        "total_examples_loaded": total_loaded,
        "total_examples_evaluated": len(ok),
        "total_examples_skipped": skipped_load + len(failed_eval),
        "total_steps_evaluated": len(step_rel),
        "mean_step_relevance": mean([float(v) for v in step_rel]),
        "mean_step_coherence": mean([float(v) for v in step_coh]),
        "mean_example_relevance": mean(ex_rel),
        "mean_example_coherence": mean(ex_coh),
        "fraction_examples_all_steps_relevant": frac_true(ex_all_rel),
        "fraction_examples_all_steps_coherent": frac_true(ex_all_coh),
        "fraction_examples_with_any_irrelevant_step": frac_true([not x for x in ex_all_rel]),
        "fraction_examples_with_any_incoherent_step": frac_true([not x for x in ex_all_coh]),
    }


def print_summary(summary: dict[str, Any]) -> None:
    print("\n===== CaSE Evaluation Summary =====", flush=True)
    print(f"Examples loaded: {summary['total_examples_loaded']}", flush=True)
    print(f"Examples evaluated: {summary['total_examples_evaluated']}", flush=True)
    print(f"Examples skipped: {summary['total_examples_skipped']}", flush=True)
    print(f"Steps evaluated: {summary['total_steps_evaluated']}", flush=True)
    print(f"Mean step relevance: {summary['mean_step_relevance']:.4f}", flush=True)
    print(f"Mean step coherence: {summary['mean_step_coherence']:.4f}", flush=True)
    print(f"Mean example relevance: {summary['mean_example_relevance']:.4f}", flush=True)
    print(f"Mean example coherence: {summary['mean_example_coherence']:.4f}", flush=True)
    print(f"Fraction examples all steps relevant: {summary['fraction_examples_all_steps_relevant']:.4f}", flush=True)
    print(f"Fraction examples all steps coherent: {summary['fraction_examples_all_steps_coherent']:.4f}", flush=True)
    print(f"Fraction examples with any irrelevant step: {summary['fraction_examples_with_any_irrelevant_step']:.4f}", flush=True)
    print(f"Fraction examples with any incoherent step: {summary['fraction_examples_with_any_incoherent_step']:.4f}", flush=True)


def log_example_progress(result: ExampleResult, example_index: int, total_examples: int, ok_count: int, err_count: int) -> None:
    parse_success = result.error is None and result.example_metrics is not None
    step_count = len(result.steps)
    judged_steps = len(result.step_evaluations)
    rel = "-" if result.example_metrics is None else f"{float(result.example_metrics['step_relevance_mean']):.3f}"
    coh = "-" if result.example_metrics is None else f"{float(result.example_metrics['step_coherence_mean']):.3f}"
    err = "none" if result.error is None else f"{result.error.get('type', 'error')}@step{result.error.get('step_index', '-')}"
    print(
        f"[progress] example {example_index}/{total_examples} id={result.id} "
        f"parse_success={str(parse_success).lower()} steps={step_count} judged_steps={judged_steps} "
        f"ok={ok_count} skipped={err_count} mean_rel={rel} mean_coh={coh} error={err}",
        flush=True,
    )


def save_results_json(path: Path, config: dict[str, Any], dataset_summary: dict[str, Any], examples: list[ExampleResult]) -> None:
    payload = {
        "config": config,
        "dataset_summary": dataset_summary,
        "examples": [asdict(ex) for ex in examples],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if not args.input_jsonl and not args.use_default_example:
        raise SystemExit("Provide --input_jsonl or --use_default_example")

    examples: list[Example] = []
    total_loaded = 0
    skipped_load = 0

    if args.input_jsonl:
        inp = Path(args.input_jsonl)
        if not inp.exists():
            raise FileNotFoundError(f"input JSONL not found: {inp}")
        loaded, file_total, file_skipped = load_jsonl_examples(
            input_path=inp,
            id_field=args.id_field,
            question_field=args.question_field,
            rationale_field=args.rationale_field,
            answer_field=args.answer_field,
            max_examples=args.max_examples if not args.use_default_example else None,
            verbose=args.verbose,
        )
        examples.extend(loaded)
        total_loaded += file_total
        skipped_load += file_skipped

    if args.use_default_example:
        examples.append(build_default_example())
        total_loaded += 1

    if args.max_examples is not None and len(examples) > args.max_examples:
        examples = examples[: args.max_examples]

    if args.verbose:
        print(f"[init] project={args.project} location={args.location} model={args.model_name}", flush=True)
        print(f"[auth_context] expected ADC account: {DEFAULT_USER_EMAIL}", flush=True)

    if not examples:
        summary = {
            "total_examples_loaded": total_loaded,
            "total_examples_evaluated": 0,
            "total_examples_skipped": total_loaded,
            "total_steps_evaluated": 0,
            "mean_step_relevance": 0.0,
            "mean_step_coherence": 0.0,
            "mean_example_relevance": 0.0,
            "mean_example_coherence": 0.0,
            "fraction_examples_all_steps_relevant": 0.0,
            "fraction_examples_all_steps_coherent": 0.0,
            "fraction_examples_with_any_irrelevant_step": 0.0,
            "fraction_examples_with_any_incoherent_step": 0.0,
        }
        print_summary(summary)
        if args.output_json:
            conf = vars(args).copy()
            conf["default_user_email"] = DEFAULT_USER_EMAIL
            save_results_json(Path(args.output_json), conf, summary, [])
        return

    model = init_vertex_model(args.project, args.location, args.model_name)
    rate_limiter = RateLimiter(args.requests_per_minute, verbose=args.verbose)
    retry_cfg = RetryConfig(args.max_retries, args.initial_retry_delay, args.max_retry_delay)

    results: list[ExampleResult] = []
    ok_count = 0
    err_count = 0
    for i, ex in enumerate(examples, start=1):
        if args.verbose:
            print(f"[run] example {i}/{len(examples)} id={ex.example_id}", flush=True)
        result = evaluate_example_case(
            example=ex,
            model=model,
                model_name=args.model_name,
                step_split_mode=args.step_split_mode,
                timeout_seconds=args.timeout_seconds,
                example_timeout_seconds=args.example_timeout_seconds,
                rate_limiter=rate_limiter,
                retry_config=retry_cfg,
                fail_on_judge_error=args.fail_on_judge_error,
            verbose=args.verbose,
        )
        results.append(result)
        if result.error is None and result.example_metrics is not None:
            ok_count += 1
        else:
            err_count += 1
        log_example_progress(result, i, len(examples), ok_count, err_count)

    summary = aggregate_dataset_metrics(results, total_loaded, skipped_load)
    print_summary(summary)

    if args.output_json:
        conf = vars(args).copy()
        conf["default_user_email"] = DEFAULT_USER_EMAIL
        save_results_json(Path(args.output_json), conf, summary, results)
        if args.verbose:
            print(f"[saved] {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
