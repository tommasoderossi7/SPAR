import os
import sys
import json
import math
import random
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import unicodedata, re, ftfy

import numpy as np
import httpx
from dotenv import load_dotenv
from tqdm import tqdm

from utils.utils import (
    extract_boxed_answers,
    check_answer,
    load_math_problems,
    append_to_json_array,
    generate_timestamp_id,
)

load_dotenv()

NOVITA_API_URL = "https://api.novita.ai/openai/v1/completions"
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
if not NOVITA_API_KEY:
    raise RuntimeError(
        "NOVITA_API_KEY is not set. Please define it in the environment or .env file."
    )
GENERATIONS_DIRNAME = "data/generations"
GENERATIONS_REGISTRY = "generations.json"
USAGE_FILENAME = "usage.json"

import argparse


parser = argparse.ArgumentParser(
    description="Collect forked rollouts with outcome distributions and drift analysis."
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="deepseek/deepseek-r1-distill-qwen-14b",  # "deepseek/deepseek-r1-distill-llama-8b",  # "deepseek/deepseek-r1-0528:free",
    help="Model identifier served via Novita.",
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="math_rollouts",
    help="Root directory to store rollout artefacts.",
)
parser.add_argument(
    "-np",
    "--num-problems",
    type=int,
    default=None,
    help="Maximum number of problems to process (after filtering).",
)
parser.add_argument(
    "-bt",
    "--base-temperature",
    type=float,
    default=0.0,
    help="Temperature for the greedy base path generation.",
)
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature used for fork continuations.",
)
parser.add_argument(
    "-tp",
    "--top_p",
    type=float,
    default=0.95,
    help="Top-p nucleus sampling parameter passed to the API.",
)
parser.add_argument(
    "-mt",
    "--max-tokens",
    type=int,
    default=4096,
    help="Maximum tokens requested for the base path completion.",
)
parser.add_argument(
    "-cmt",
    "--continuation-max-tokens",
    type=int,
    default=15000,
    help="Maximum tokens allowed for each fork continuation.",
)
parser.add_argument(
    "-sps",
    "--samples-per-fork",
    type=int,
    default=30,
    help="Number of independent continuations sampled for every alternate token.",
)
parser.add_argument(
    "-atk",
    "--alternate-top-k",
    type=int,
    default=10,
    help="Top-K alternate tokens to retain per position on the base path.",
)
parser.add_argument(
    "-amp",
    "--alternate-min-prob",
    type=float,
    default=0.05,
    help="Minimum probability threshold for alternate tokens to spawn forks.",
)
parser.add_argument(
    "-ctl",
    "--continuation-top-logprobs",
    type=int,
    default=1,
    help="Top logprobs to request for continuation samples (0 disables alternate capture).",
)
parser.add_argument(
    "-mr",
    "--max-retries",
    type=int,
    default=3,
    help="Maximum retries for Novita requests before failing a step.",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=44,
    help="Seed for python and numpy RNGs.",
)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Force regeneration even if rollout artefacts already exist.",
)
parser.add_argument(
    "-ty",
    "--type",
    type=str,
    default=None,
    help="Optional MATH dataset problem type filter.",
)
parser.add_argument(
    "-l",
    "--level",
    type=str,
    default=None,
    help="Optional MATH dataset level filter.",
)
parser.add_argument(
    "-sp",
    "--split",
    type=str,
    default="train",
    choices=["train", "test"],
    help="Dataset split to sample from.",
)
parser.add_argument(
    "-ip",
    "--include-problems",
    type=str,
    default=None,
    help="Comma-separated list of explicit problem indices to include.",
)
parser.add_argument(
    "-ep",
    "--exclude-problems",
    type=str,
    default=None,
    help="Comma-separated list of problem indices to exclude.",
)
parser.add_argument(
    "-cc",
    "--concurrency",
    type=int,
    default=4,
    help="Maximum concurrent Novita requests.",
)
parser.add_argument(
    "--problem-substring",
    type=str,
    default=None,
    help="Case-insensitive substring to match a single problem statement.",
)
parser.add_argument(
    "--frequency-penalty",
    type=float,
    default=None,
    help="Optional frequency penalty forwarded to Novita.",
)
parser.add_argument(
    "--presence-penalty",
    type=float,
    default=None,
    help="Optional presence penalty forwarded to Novita.",
)
parser.add_argument(
    "--repetition-penalty",
    type=float,
    default=None,
    help="Optional repetition penalty forwarded to Novita.",
)
parser.add_argument(
    "--min-p",
    type=float,
    default=None,
    help="Optional min-p sampling parameter forwarded to Novita.",
)
parser.add_argument(
    "--top-k",
    type=int,
    default=None,
    help="Optional top-k sampling parameter forwarded to Novita.",
)
args = parser.parse_args()

output_dir = (
    Path(args.output_dir)
    / args.model.replace("/", "_")
    / f"samples_{args.samples_per_fork}_topk_{args.alternate_top_k}_prob_{args.alternate_min_prob}"
)
output_dir.mkdir(parents=True, exist_ok=True)

random.seed(args.seed)
np.random.seed(args.seed)

REQUEST_SEMAPHORE: Optional[asyncio.Semaphore] = None


def record_generation(
    generation_id: str,
    problem_id: str,
    response_json: Dict[str, Any],
) -> None:
    base_dir = Path(__file__).resolve().parents[1]
    generations_dir = base_dir / GENERATIONS_DIRNAME
    generations_dir.mkdir(parents=True, exist_ok=True)

    generations_path = generations_dir / GENERATIONS_REGISTRY
    generations_entry = {
        "generation_id": generation_id,
        "model_name": args.model,
        "problem_id": problem_id,
    }
    append_to_json_array(generations_path, generations_entry)

    usage_path = generations_dir / USAGE_FILENAME
    usage_entry = {
        "generation_id": generation_id,
        "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens"),
        "completion_tokens": response_json.get("usage", {}).get("completion_tokens"),
        "total_tokens": response_json.get("usage", {}).get("total_tokens"),
        "api_key": "novita",  # "api_key": "personal" if OPENROUTER_API_KEY.endswith("d9a98e3f1c2") else "spar",
    }
    append_to_json_array(usage_path, usage_entry)


def build_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NOVITA_API_KEY}",
        "Content-Type": "application/json",
    }


async def make_api_request(
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_logprobs: int = 0,
    logprobs: bool = False,
    semaphore: Optional[asyncio.Semaphore] = None,
    retries: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    min_p: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "stream": False,
    }

    if logprobs and top_logprobs > 0:
        payload["logprobs"] = top_logprobs

    if frequency_penalty is not None:
        payload["frequency_penalty"] = frequency_penalty
    if presence_penalty is not None:
        payload["presence_penalty"] = presence_penalty
    if repetition_penalty is not None:
        payload["repetition_penalty"] = repetition_penalty
    if min_p is not None:
        payload["min_p"] = min_p
    if top_k is not None:
        payload["top_k"] = top_k

    headers = build_headers()
    max_retries = retries if retries is not None else args.max_retries
    base_delay = 2.0

    async def _send_request() -> httpx.Response:
        async with httpx.AsyncClient(timeout=240) as client:
            return await client.post(NOVITA_API_URL, headers=headers, json=payload)

    for attempt in range(max_retries):
        try:
            if semaphore:
                async with semaphore:
                    response = await _send_request()
            else:
                response = await _send_request()

            if response.status_code == 200:
                return response.json()

            if response.status_code in (429, 500) and attempt < max_retries - 1:
                jitter = random.uniform(0.5, 1.5)
                await asyncio.sleep(base_delay * (2**attempt) * jitter)
                continue

            return {
                "error": f"API error: {response.status_code}",
                "details": response.text,
            }
        except Exception as exc:
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {exc}"}
            await asyncio.sleep(base_delay * (2**attempt))

    return {"error": "All API request attempts failed"}


# --- text cleaner for model outputs ---
def _clean_token_display(s: object) -> str:
    """
    Make model outputs readable in logs:
      - Fix classic mojibake (UTF-8 decoded as Latin-1/Windows-1252, etc.)
      - Map common tokenizer artifacts to spaces/newlines
      - Normalize Unicode and strip stray control chars (except \\n/\\t)
      - Keep idempotent behavior on already-clean text
    """
    text = s if isinstance(s, str) else str(s)
    if not text:
        return ""

    # 1) Fix classic mojibake if ftfy is available.
    #    (e.g. 'Ã„' -> 'Ä', 'âœ”' -> '✔')
    #    See: ftfy.fix_encoding docs.
    if ftfy is not None:
        text = ftfy.fix_encoding(text)

    # 2) Map common tokenizer markers to human-readable whitespace.
    #    - SentencePiece metaspace: '▁' -> ' '
    #    - GPT-2/RoBERTa byte-level BPE: 'Ġ' (U+0120) ~ leading space,
    #      and 'Ċ/ċ' often surface around newlines in dumps.
    text = (
        text.replace("▁", " ")  # SentencePiece metaspace
        .replace("\u0120", " ")  # 'Ġ' : space-before-word
        .replace("Ġ", " ")  # visible 'Ġ'
        .replace("\u010a", "\n")  # 'Ċ'
        .replace("\u010b", "\n")  # 'ċ'
        .replace("Ċ", "\n")
        .replace("ċ", "\n")
    )

    # (Optionally keep your legacy mappings)
    text = text.replace("Ğ", " ").replace("ğ", " ")

    # 3) Normalize and clean controls.
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if (ch.isprintable() or ch in "\n\t"))

    # 4) Tidy whitespace a bit (without touching newlines).
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text


STRUCTURAL_MAP = {"Ġ": " ", "\u0120": " ", "Ċ": "\n", "\u010a": "\n", "\u010b": "\n"}
MOJIBAKE_PREFIXES = ("Â", "Ã", "Ä", "Å", "�")
MOJIBAKE_BAD = "�"


def is_safe_to_fork(tok: str) -> bool:
    if not tok:
        return False
    # allow structural markers (will detok when building the prompt)
    if tok in STRUCTURAL_MAP or "▁" in tok:
        return True
    # keep this guard!
    if any(unicodedata.category(c).startswith("C") and c not in "\n\t" for c in tok):
        return False
    # reject obvious mojibake
    if tok.startswith(MOJIBAKE_PREFIXES) or (MOJIBAKE_BAD in tok):
        return False
    return True


def extract_choice_text(choice: Dict[str, Any]) -> str:
    text_raw = choice.get("text")
    if isinstance(text_raw, str):
        text = _clean_token_display(text_raw)
        return text_raw, text
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def parse_top_logprob_block(block: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    def _push(tok, lp):
        candidates.append(
            {
                "token_raw": tok,
                "token": _clean_token_display(tok),
                "logprob": lp,
                "probability": (math.exp(lp) if isinstance(lp, (int, float)) else None),
            }
        )

    if isinstance(block, dict):
        for token_value, logprob_value in block.items():
            _push(token_value, logprob_value)
    elif isinstance(block, list):
        for item in block:
            if isinstance(item, dict):
                tok = item.get("token") or item.get("text")
                lp = item.get("logprob")
                if tok is not None:
                    _push(tok, lp)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                tok, lp = item
                _push(tok, lp)
    candidates.sort(key=lambda it: it.get("probability", 0.0) or 0.0, reverse=True)
    return candidates


def parse_logprob_entries(logprobs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not logprobs:
        return []
    entries: List[Dict[str, Any]] = []
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")

    if isinstance(tokens, list) and isinstance(token_logprobs, list):
        for idx, tok in enumerate(tokens):
            lp = token_logprobs[idx] if idx < len(token_logprobs) else None
            prob = math.exp(lp) if isinstance(lp, (int, float)) else None
            top_block = (
                top_logprobs[idx]
                if isinstance(top_logprobs, list) and idx < len(top_logprobs)
                else None
            )
            entries.append(
                {
                    "token_index": idx,
                    "token_raw": tok,
                    "token": _clean_token_display(tok),
                    "logprob": lp,
                    "probability": prob,
                    "top_candidates": parse_top_logprob_block(top_block),
                }
            )
        return entries

    content = logprobs.get("content")
    if isinstance(content, list):
        for idx, block in enumerate(content):
            tok_raw, lp, top_block = "", None, None
            if isinstance(block, dict):
                tok_raw = block.get("token") or block.get("text") or ""
                lp = block.get("logprob")
                top_block = block.get("top_logprobs")
            prob = math.exp(lp) if isinstance(lp, (int, float)) else None
            entries.append(
                {
                    "token_index": idx,
                    "token_raw": tok_raw,
                    "token": _clean_token_display(tok_raw),
                    "logprob": lp,
                    "probability": prob,
                    "top_candidates": parse_top_logprob_block(top_block),
                }
            )
        return entries

    return entries


def extract_tokens_and_logprobs(
    logprob_block: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str], List[float]]:
    if not logprob_block:
        return [], [], []

    if (
        isinstance(logprob_block, dict)
        and "tokens" in logprob_block
        and "token_logprobs" in logprob_block
    ):
        tokens = logprob_block.get("tokens")
        token_logprobs = logprob_block.get("token_logprobs")
        tokens_out: List[str] = []
        tokens_out_raw: List[str] = []
        logprobs_out: List[float] = []

        for token, logprob in zip(tokens or [], token_logprobs or []):
            tokens_out.append(_clean_token_display(token))
            tokens_out_raw.append(token)
            if isinstance(logprob, (int, float)):
                logprobs_out.append(logprob)
            else:
                logprobs_out.append(0.0)
        return tokens_out_raw, tokens_out, logprobs_out

    content = logprob_block.get("content")
    if isinstance(content, list):
        tokens_out: List[str] = []
        tokens_out_raw: List[str] = []
        logprobs_out: List[float] = []
        for item in content:
            if isinstance(item, dict):
                token_value = item.get("token") or item.get("text") or ""
                logprob_value = item.get("logprob")
                tokens_out_raw.append(token_value)
                tokens_out.append(_clean_token_display(token_value))
                if isinstance(logprob_value, (int, float)):
                    logprobs_out.append(logprob_value)
                else:
                    logprobs_out.append(0.0)
        return tokens_out_raw, tokens_out, logprobs_out

    return [], [], []


def default_outcome_extractor(text: str) -> str:
    answers = extract_boxed_answers(text)
    for answer in answers:
        cleaned = answer.strip()
        if cleaned:
            return cleaned
    stripped = text.strip()
    if stripped:
        return stripped
    return "__empty__"


def compute_tail_logprob_sums(token_entries: List[Dict[str, Any]]) -> List[float]:
    tail_sums: List[float] = []
    running = 0.0
    for entry in reversed(token_entries):
        tail_sums.append(running)
        logprob = entry.get("logprob")
        if isinstance(logprob, (int, float)):
            running += logprob
    tail_sums.reverse()
    return tail_sums


def build_outcome_distributions(
    base_outcome: str,
    token_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    tail_sums = compute_tail_logprob_sums(token_entries)
    distributions: List[Dict[str, Any]] = []

    for entry, tail_logprob in zip(token_entries, tail_sums):
        if entry.get("branches") == []:
            continue
        distribution_weights: defaultdict[str, float] = defaultdict(float)
        base_probability = entry.get("probability") or 0.0
        if base_probability > 0:
            continuation_probability = math.exp(tail_logprob)
            distribution_weights["base_greedy_completion"] += (
                base_probability * continuation_probability
            )

        accounted_probability = base_probability
        for branch in entry.get("branches", []):
            fork_probability = branch.get("probability") or 0.0
            if fork_probability <= 0:
                continue
            accounted_probability += fork_probability

            for sample in branch.get("samples", []):
                if sample.get("error"):
                    continue
                continuation_probability = sample.get("continuation_probability")
                if continuation_probability is None:
                    logprob_sum = sample.get("logprob_sum")
                    if isinstance(logprob_sum, (int, float)):
                        continuation_probability = math.exp(logprob_sum)
                    else:
                        continuation_probability = 0.0
                distribution_weights[
                    str(entry.get("token_index", -1))
                    + "__"
                    + str(sample.get("sample_idx", -1))
                ] += fork_probability * continuation_probability

        accounted_probability = min(max(accounted_probability, 0.0), 1.0)
        residual_probability = max(0.0, 1.0 - accounted_probability)
        if residual_probability > 0:
            distribution_weights["__residual__"] += residual_probability

        total_weight = sum(distribution_weights.values())
        normalized = (
            {key: value / total_weight for key, value in distribution_weights.items()}
            if total_weight > 0
            else {}
        )

        distributions.append(
            {
                "token_index": entry.get("token_index"),
                "token": entry.get("token"),
                "raw_weights": dict(distribution_weights),
                "normalized": normalized,
                "total_weight": total_weight,
                "residual_probability": residual_probability,
            }
        )

    base_distribution = {"base_greedy_completion": 1.0}
    return distributions, base_distribution


def l2_distance(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    keys = set(dist_a.keys()).union(dist_b.keys())
    return math.sqrt(
        sum((dist_a.get(key, 0.0) - dist_b.get(key, 0.0)) ** 2 for key in keys)
    )


def compute_drift_series(
    base_distribution: Dict[str, float],
    distributions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    drift_series: List[Dict[str, Any]] = [
        {
            "token_index": -1,
            "token": "__base__",
            "drift": 0.0,
            "outcomes_tracked": list(base_distribution.keys()),
        }
    ]

    for distribution in distributions:
        drift_series.append(
            {
                "token_index": distribution.get("token_index"),
                "token": distribution.get("token"),
                "drift": l2_distance(
                    base_distribution, distribution.get("normalized", {})
                ),
                "outcomes_tracked": list(distribution.get("normalized", {}).keys()),
            }
        )

    return drift_series


async def collect_base_path(
    problem: Dict[str, Any],
    problem_idx: int,
    prompt: str,
    *,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    generation_id = generate_timestamp_id()
    response = await make_api_request(
        prompt,
        temperature=args.base_temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        top_logprobs=args.alternate_top_k,
        logprobs=True,
        semaphore=semaphore,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        min_p=args.min_p,
        top_k=args.top_k,
    )

    if response.get("error"):
        return response

    choices = response.get("choices")
    if not choices:
        return {"error": "Novita response did not include any choices."}
    record_generation(generation_id, str(problem_idx), response)
    choice = choices[0]
    completion_text_raw, completion_text = extract_choice_text(choice)
    token_entries = parse_logprob_entries(choice.get("logprobs"))

    prefix_acc_raw = ""
    prefix_acc_disp = ""

    for entry in token_entries:
        entry["prefix_before_raw"] = prefix_acc_raw
        entry["prefix_before"] = prefix_acc_disp

        tok_raw = entry.get("token_raw") or ""
        tok_disp = entry.get("token") or _clean_token_display(tok_raw)

        prefix_acc_raw += tok_raw
        prefix_acc_disp += tok_disp

        entry["prefix_with_alt_token_raw"] = prefix_acc_raw
        entry["prefix_with_alt_token"] = prefix_acc_disp

    base_outcome = default_outcome_extractor(completion_text)
    is_correct = False
    if problem.get("gt_answer") and base_outcome and base_outcome != "__empty__":
        is_correct = check_answer(base_outcome, problem["gt_answer"])

    return {
        "completion_text": completion_text,
        "completion_text_raw": completion_text_raw,
        "token_entries": token_entries,
        "base_outcome": base_outcome,
        "base_is_correct": is_correct,
        "raw_response": response,
    }


def count_potential_forking_tokens(token_entries: List[Dict[str, Any]]) -> int:
    potential_forking_tokens = 0
    potential_forking_positions = 0
    for entry in token_entries:
        current_position_forking = False
        for candidate in entry.get("top_candidates", []):
            candidate_token = candidate.get("token")
            candidate_probability = candidate.get("probability")
            if (
                not candidate_token
                or candidate_token == entry.get("token")
                or not is_safe_to_fork(candidate_token)
            ):
                continue
            if not isinstance(candidate_probability, (int, float)):
                continue
            if candidate_probability < args.alternate_min_prob:
                continue
            potential_forking_tokens += 1
            current_position_forking = True
        if current_position_forking:
            potential_forking_positions += 1
    return potential_forking_tokens, potential_forking_positions


async def sample_fork_branches(
    problem_idx: int,
    prompt: str,
    token_entries: List[Dict[str, Any]],
    *,
    semaphore: asyncio.Semaphore,
    potential_forking_tokens: int,
) -> None:
    # show progress bar with samples drawn out of total samples to draw
    pbar = tqdm(
        total=potential_forking_tokens * args.samples_per_fork,
        desc="Sampling alternative branches",
    )
    for entry in token_entries:
        branches: List[Dict[str, Any]] = []
        entry["alt_candidates"] = []
        for candidate in entry.get("top_candidates", []):
            cand_raw = candidate.get("token_raw")
            cand_clean = candidate.get("token")
            cand_prob = candidate.get("probability")
            if (
                not cand_raw
                or cand_raw == entry.get("token_raw")
                or not is_safe_to_fork(cand_raw)
            ):
                # remove some fields from the entry
                entry.pop("prefix_before_raw", None)
                entry.pop("prefix_before", None)
                entry.pop("prefix_with_alt_token_raw", None)
                entry.pop("prefix_with_alt_token", None)
                # remove candidate from the list
                entry["top_candidates"].remove(candidate)
                if not is_safe_to_fork(cand_raw):
                    print(f"\nSkipping unsafe token for forking: {cand_raw}")
                continue
            if (
                not isinstance(cand_prob, (int, float))
                or cand_prob < args.alternate_min_prob
            ):
                # remove some fields from the entry
                entry.pop("prefix_before_raw", None)
                entry.pop("prefix_before", None)
                entry.pop("prefix_with_alt_token_raw", None)
                entry.pop("prefix_with_alt_token", None)
                # remove candidate from the list
                entry["top_candidates"].remove(candidate)
                continue

            # IMPORTANT: build fork prompt from RAW prefix + RAW candidate to match model state
            fork_prompt = f"{prompt}{entry.get('prefix_before_raw', '')}{cand_raw}"

            # clean up some fields from the entry to save space
            entry.pop("prefix_before_raw", None)
            entry.pop("prefix_before", None)
            entry.pop("prefix_with_alt_token_raw", None)
            # entry.pop("prefix_with_alt_token", None)
            # remove raw token from candidate and from original CoT to save space
            candidate.pop("token_raw", None)
            # remove logprob from candidate to save space
            candidate.pop("logprob", None)
            # add candidate to the list of actually used alternate candidates
            entry["alt_candidates"].append(candidate)

            tasks = [
                make_api_request(
                    fork_prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.continuation_max_tokens,
                    top_logprobs=args.continuation_top_logprobs,
                    logprobs=True,
                    semaphore=semaphore,
                    frequency_penalty=args.frequency_penalty,
                    presence_penalty=args.presence_penalty,
                    repetition_penalty=args.repetition_penalty,
                    min_p=args.min_p,
                    top_k=args.top_k,
                )
                for _ in range(args.samples_per_fork)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            pbar.update(args.samples_per_fork)

            # create a unique generation_id for every sample in this set of samples
            for i in range(args.samples_per_fork):
                generation_id = generate_timestamp_id()
                record_generation(generation_id, str(problem_idx), responses[i])

            samples: List[Dict[str, Any]] = []

            for sample_idx, result in enumerate(responses):
                if isinstance(result, Exception):
                    samples.append({"sample_index": sample_idx, "error": str(result)})
                    continue

                if not isinstance(result, dict):
                    samples.append(
                        {"sample_index": sample_idx, "error": "Invalid response object"}
                    )
                    continue

                if result.get("error"):
                    samples.append(
                        {
                            "sample_index": sample_idx,
                            "error": result.get("error"),
                            "details": result.get("details"),
                        }
                    )
                    continue

                sample_choices = result.get("choices") or []
                if not sample_choices:
                    samples.append(
                        {"sample_index": sample_idx, "error": "No choices returned"}
                    )
                    continue
                print(f"\nPrompt for forking token {cand_clean} to continue from:")
                # print(f"Raw: {fork_prompt}")
                print(
                    f"Clean: {_clean_token_display(fork_prompt)[0:50] + '...' + _clean_token_display(fork_prompt[-50:])}"
                )
                # print(f"\nSampled continuation: {sample_choices[0]}")
                sample_choice = sample_choices[0]
                sample_text_raw, sample_text = extract_choice_text(sample_choice)
                print(
                    f"\nRaw Response: {sample_choice['text'][0:100] + '...' + sample_choice['text'][-100:]}"
                )
                sample_tokens_raw, sample_tokens, sample_logprobs = (
                    extract_tokens_and_logprobs(sample_choice.get("logprobs"))
                )
                print(
                    f"Cleaned sampled text: {sample_text[0:100] + '...' + sample_text[-100:]}"
                )
                print(
                    f"Extracted tokens: {sample_tokens[0:10]} ... {sample_tokens[-10:]}"
                )

                logprob_sum = sum(
                    value
                    for value in sample_logprobs
                    if isinstance(value, (int, float))
                )
                continuation_probability = (
                    math.exp(logprob_sum) if sample_logprobs else "Unknown"
                )
                outcome = default_outcome_extractor(sample_text)

                samples.append(
                    {
                        "sample_index": sample_idx,
                        "final_answer_text": sample_text,
                        # "tokens": sample_tokens,
                        "logprob_sum": logprob_sum,
                        "continuation_probability": continuation_probability,
                        "outcome": outcome,
                    }
                )
                # sys.exit(0)

            branches.append(
                {
                    "token": cand_clean,
                    # "token_raw": cand_raw,
                    "probability": cand_prob,
                    # "logprob": candidate.get("logprob"),
                    "samples": samples,
                }
            )
            entry["branches"] = branches
        pbar.close()
        # remove unused fields from entry to save space
        entry.pop("top_candidates", None)
        entry.pop("token_raw", None)
        entry.pop("logprob", None)

        entry["branches"] = branches


async def generate_rollout(
    problem_idx: int,
    problem: Dict[str, Any],
    *,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    prompt = (
        "Solve this math problem step by step. You MUST put your final answer in \\boxed{}."
        " Problem: "
        f"{problem['problem']} Solution: \n<think>\n"
    )

    base_data = await collect_base_path(
        problem, problem_idx, prompt, semaphore=semaphore
    )
    if base_data.get("error"):
        return {
            "problem_index": problem_idx,
            "error": base_data.get("error"),
            "details": base_data.get("details"),
        }
    print(f"Base completion for problem {problem_idx} collected.")
    token_entries = base_data.get("token_entries", [])
    print(f"A total of {len(token_entries)} tokens in base path.")
    # print(f"Full token entries: {token_entries}")
    potential_forking_tokens, potential_forking_positions = (
        count_potential_forking_tokens(token_entries)
    )
    print(
        f"Identified {potential_forking_tokens} potential forking tokens across "
        f"{potential_forking_positions} potential forking positions."
    )
    print(
        f"Now for every of the {potential_forking_tokens} potential forking token, a set of {args.samples_per_fork} samples will be drawn.",
    )
    print(
        f"\nResulting in a total of {potential_forking_tokens * args.samples_per_fork} additional model calls.",
    )

    await sample_fork_branches(
        problem_idx,
        prompt,
        token_entries,
        semaphore=semaphore,
        potential_forking_tokens=potential_forking_tokens,
    )

    outcome_distributions, base_distribution = build_outcome_distributions(
        base_data.get("base_outcome", "__empty__"), token_entries
    )
    drift_series = compute_drift_series(base_distribution, outcome_distributions)

    return {
        "problem_index": problem_idx,
        "metadata": {
            "model": args.model,
            "base_temperature": args.base_temperature,
            "sample_temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens_base": args.max_tokens,
            "max_tokens_continuation": args.continuation_max_tokens,
            "samples_per_fork": args.samples_per_fork,
            "alternate_top_k": args.alternate_top_k,
            "alternate_min_prob": args.alternate_min_prob,
        },
        "prompt": prompt,
        "problem": {
            "problem_text": problem.get("problem"),
            "level": problem.get("level"),
            "type": problem.get("type"),
            "gt_answer": problem.get("gt_answer"),
        },
        "base": {
            "completion": base_data.get("completion_text"),
            "completion_raw": base_data.get("completion_text_raw"),
            "outcome": base_data.get("base_outcome"),
            "is_correct": base_data.get("base_is_correct"),
        },
        "token_steps": token_entries,
        "outcome_distributions": outcome_distributions,
        "drift_series": drift_series,
    }


async def process_problem(
    problem_idx: int,
    problem: Dict[str, Any],
    *,
    semaphore: asyncio.Semaphore,
) -> None:
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(parents=True, exist_ok=True)

    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(problem, f, indent=2, ensure_ascii=False)

    rollout_file = problem_dir / "rollout_analysis.json"
    error_file = problem_dir / "rollout_error.json"

    if rollout_file.exists() and not args.force:
        return

    result = await generate_rollout(problem_idx, problem, semaphore=semaphore)

    if result.get("error"):
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return

    if error_file.exists():
        error_file.unlink()

    with open(rollout_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


async def main() -> None:
    global REQUEST_SEMAPHORE
    REQUEST_SEMAPHORE = asyncio.Semaphore(args.concurrency)

    problems = load_math_problems(
        problem_type=args.type,
        level=args.level,
        num_problems=args.num_problems,
        split=args.split,
        include_problems=None,
    )

    if args.exclude_problems:
        exclude_ids = {
            int(pid.strip())
            for pid in args.exclude_problems.split(",")
            if pid.strip().isdigit()
        }
        problems = [(idx, prob) for idx, prob in problems if idx not in exclude_ids]

    if args.include_problems:
        include_ids = {
            int(pid.strip())
            for pid in args.include_problems.split(",")
            if pid.strip().isdigit()
        }
        problems = [(idx, prob) for idx, prob in problems if idx in include_ids]

    if args.problem_substring:
        needle = args.problem_substring.lower()
        matched = [
            (idx, prob)
            for idx, prob in problems
            if isinstance(prob.get("problem"), str)
            and needle in prob["problem"].lower()
        ]
        if not matched:
            print(f"No problems contain substring '{args.problem_substring}'. Exiting.")
            return
        if len(matched) > 1:
            matched_ids = ", ".join(str(idx) for idx, _ in matched)
            print(
                "Substring matched multiple problems (indices: "
                f"{matched_ids}). Refusing selection."
            )
            return
        problems = [matched[0]]

    if args.num_problems is not None:
        problems = problems[: args.num_problems]

    if not problems:
        print("No problems selected. Exiting.")
        return

    for problem_idx, problem in tqdm(problems, desc="Processing problems"):
        await process_problem(problem_idx, problem, semaphore=REQUEST_SEMAPHORE)


if __name__ == "__main__":
    asyncio.run(main())
