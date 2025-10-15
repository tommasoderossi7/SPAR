# generate_rollout.py — progressive saves + resume from checkpoint

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
from transformers import AutoTokenizer

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
    description="Collect forked rollouts with outcome distributions and drift analysis (progressive + resumable)."
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="deepseek/deepseek-r1-distill-qwen-14b",
    help="Model identifier served via Novita.",
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="math_rollouts",
    help="Root directory to store rollout artefacts.",
)
parser.add_argument("-np", "--num-problems", type=int, default=None)
parser.add_argument("-bt", "--base-temperature", type=float, default=0.0)
parser.add_argument("-t", "--temperature", type=float, default=0.7)
parser.add_argument("-tp", "--top_p", type=float, default=0.95)
parser.add_argument("-mt", "--max-tokens", type=int, default=4096)
parser.add_argument("-cmt", "--continuation-max-tokens", type=int, default=15000)
parser.add_argument("-sps", "--samples-per-fork", type=int, default=30)
parser.add_argument("-atk", "--alternate-top-k", type=int, default=10)
parser.add_argument("-amp", "--alternate-min-prob", type=float, default=0.05)
parser.add_argument("-ctl", "--continuation-top-logprobs", type=int, default=1)
parser.add_argument("-mr", "--max-retries", type=int, default=3)
parser.add_argument("-s", "--seed", type=int, default=44)
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument("-ty", "--type", type=str, default=None)
parser.add_argument("-l", "--level", type=str, default=None)
parser.add_argument(
    "-sp", "--split", type=str, default="train", choices=["train", "test"]
)
parser.add_argument("-ip", "--include-problems", type=str, default=None)
parser.add_argument("-ep", "--exclude-problems", type=str, default=None)
parser.add_argument("-cc", "--concurrency", type=int, default=10)
parser.add_argument("--problem-substring", type=str, default=None)
parser.add_argument("--frequency-penalty", type=float, default=None)
parser.add_argument("--presence-penalty", type=float, default=None)
parser.add_argument("--repetition-penalty", type=float, default=None)
parser.add_argument("--min-p", type=float, default=None)
parser.add_argument("--top-k", type=int, default=None)
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

# ---------------- NEW: tokenizer cache & helpers ----------------

_TOKENIZER_CACHE: Dict[str, Any] = {}


def _guess_hf_tokenizer_name(model_name: str) -> str:
    name = model_name.lower()
    # Map DeepSeek R1 variants to the tokenizer they advertise
    if "qwen" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    elif "llama" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    else:
        raise ValueError(f"Cannot guess tokenizer for model name: {model_name}")


def get_tokenizer(model_name: str):
    key = _guess_hf_tokenizer_name(model_name)
    tok = _TOKENIZER_CACHE.get(key)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(key, use_fast=True)
        _TOKENIZER_CACHE[key] = tok
    return tok


def detok_for_api(token_str: str) -> Optional[str]:
    """
    Convert a *token string* (e.g., 'ĠNow', '.ĊĊ', '▁The') into true text
    using the model's tokenizer. This avoids feeding display markers (Ġ/Ċ/▁) into the prompt.
    """
    if not isinstance(token_str, str) or token_str == "":
        return None
    tok = get_tokenizer(args.model)
    tid = tok.convert_tokens_to_ids(token_str)
    if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
        text = tok.decode([tid], clean_up_tokenization_spaces=False)
        return text
    else:
        return None


# ---------------- utility & API ----------------

PRICES_PER_MTOK = {
    "deepseek/deepseek-r1-distill-qwen-14b": {"input": 0.15, "output": 0.15},
}


def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    """
    Atomic-ish write: dump to a temp file and replace the target.
    On POSIX, replace/rename updates the path in one step so readers see old or new.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# count tokens with AutoTokenizer (no special tokens)
def count_tokens(text: str) -> int:
    tok = get_tokenizer(args.model)
    enc = tok(text, add_special_tokens=False)
    ids = enc.get("input_ids") or []
    return len(ids)


def load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def record_generation(
    generation_id: str, problem_id: str, response_json: Dict[str, Any]
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
        "api_key": "novita",
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


# --- display cleaner (unchanged) ---
def _clean_token_display(s: object) -> str:
    text = s if isinstance(s, str) else str(s)
    if not text:
        return ""
    if ftfy is not None:
        text = ftfy.fix_encoding(text)
    text = (
        text.replace("▁", " ")
        .replace("\u0120", " ")
        .replace("Ġ", " ")
        .replace("\u010a", "\n")
        .replace("\u010b", "\n")
        .replace("Ċ", "\n")
        .replace("ċ", "\n")
    )
    text = text.replace("Ğ", " ").replace("ğ", " ")
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if (ch.isprintable() or ch in "\n\t"))
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return text


STRUCTURAL_MAP = {"Ġ": " ", "\u0120": " ", "Ċ": "\n", "\u010a": "\n", "\u010b": "\n"}


def is_safe_to_fork(token_str: str) -> bool:
    decoded = detok_for_api(token_str)
    if decoded is None:
        return False
    if any(
        unicodedata.category(c).startswith("C") and c not in "\n\t" for c in decoded
    ):
        return False
    return True


def extract_choice_text(choice: Dict[str, Any]) -> Tuple[str, str]:
    text_raw = choice.get("text")
    if isinstance(text_raw, str):
        text = _clean_token_display(text_raw)
        return text_raw, text
    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content, _clean_token_display(content)
    return "", ""


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
    text_offsets = logprobs.get("text_offset") or logprobs.get("text_offsets")

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
                    "text_offset": (
                        text_offsets[idx]
                        if isinstance(text_offsets, list) and idx < len(text_offsets)
                        else None
                    ),
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
                    "text_offset": block.get("text_offset"),
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
        tokens_out_raw, tokens_out, logprobs_out = [], [], []
        for token, logprob in zip(tokens or [], token_logprobs or []):
            tokens_out.append(_clean_token_display(token))
            tokens_out_raw.append(token)
            logprobs_out.append(logprob if isinstance(logprob, (int, float)) else 0.0)
        return tokens_out_raw, tokens_out, logprobs_out
    content = logprob_block.get("content")
    if isinstance(content, list):
        tokens_out_raw, tokens_out, logprobs_out = [], [], []
        for item in content:
            if isinstance(item, dict):
                token_value = item.get("token") or item.get("text") or ""
                logprob_value = item.get("logprob")
                tokens_out_raw.append(token_value)
                tokens_out.append(_clean_token_display(token_value))
                logprobs_out.append(
                    logprob_value if isinstance(logprob_value, (int, float)) else 0.0
                )
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


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(v for v in dist.values() if isinstance(v, (int, float)))
    if total <= 0:
        return {}
    return {k: v / total for k, v in dist.items() if isinstance(v, (int, float))}


def _weighted_add(dst: Dict[str, float], src: Dict[str, float], weight: float) -> None:
    if weight <= 0:
        return
    for k, v in src.items():
        dst[k] = dst.get(k, 0.0) + weight * v


def _empirical_outcome_dist(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    counts: Dict[str, int] = defaultdict(int)
    total = 0
    for s in samples or []:
        if s.get("error"):
            continue
        o = s.get("outcome")
        if isinstance(o, str) and o:
            counts[o] += 1
            total += 1
    if total == 0:
        return {}
    return {k: v / float(total) for k, v in counts.items()}


def build_outcome_distributions_exact(
    base_outcome: str,
    token_entries: List[Dict[str, Any]],
    *,
    o0_dist: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Paper-conformant construction:
      • For each (t,w): o_{t,w} is the empirical outcome histogram over S samples (equal weights).
      • Mix across tokens: o_t ∝ Σ_w p(w|prefix) · o_{t,w}, renormalize over outcomes.
      • Include o0 (t = -1) if provided.
    """
    ot_list: List[Dict[str, Any]] = []
    otw_all: List[Dict[str, Any]] = []

    if o0_dist:
        ot_list.append({"t": -1, "token": "__o0__", "dist": dict(o0_dist)})

    for entry in token_entries:
        t = entry.get("token_index")
        token_at_t = entry.get("token") or ""
        p_w_star = entry.get("probability") or 0.0

        # Build a map of existing branches for this t (should include greedy after Patch 2)
        branch_map = {br.get("token"): br for br in (entry.get("branches") or [])}

        # --- Greedy branch o_{t,w*} from samples if present, else fallback to delta on base_outcome ---
        greedy_branch = branch_map.get(token_at_t)
        if greedy_branch and (greedy_branch.get("samples")):
            dist_w_star = _empirical_outcome_dist(greedy_branch.get("samples"))
        else:
            # Fallback (should be rare if Patch 2 runs): delta at base outcome
            dist_w_star = {base_outcome: 1.0} if base_outcome else {}
        otw_all.append({"t": t, "w": token_at_t, "p_w": p_w_star, "dist": dist_w_star})

        # --- Alternate branches: empirical outcome frequencies ---
        for br_tok, br in branch_map.items():
            if br_tok == token_at_t:
                continue
            p_w = br.get("probability") or 0.0
            dist_w = _empirical_outcome_dist(br.get("samples"))
            otw_all.append({"t": t, "w": br_tok, "p_w": p_w, "dist": dist_w})

        # --- Mix across tokens for this t, then renormalize over outcomes ---
        mix_raw: Dict[str, float] = defaultdict(float)
        for rec in otw_all:
            if rec["t"] != t:
                continue
            _weighted_add(mix_raw, rec["dist"], rec["p_w"])
        o_t = _normalize(mix_raw)
        ot_list.append({"t": t, "token": token_at_t, "dist": o_t})

    return ot_list, otw_all


def l2_distance(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    keys = set(dist_a.keys()).union(dist_b.keys())
    return math.sqrt(
        sum((dist_a.get(key, 0.0) - dist_b.get(key, 0.0)) ** 2 for key in keys)
    )


def compute_drift_series_from_ot(ot_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    drift: List[Dict[str, Any]] = []
    if not ot_list:
        return drift
    o0 = ot_list[0].get("dist", {}) or {}
    for rec in ot_list:
        t = rec.get("t", -1)
        ot = rec.get("dist", {}) or {}
        drift.append({"t": t, "drift": l2_distance(o0, ot)})
    return drift


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


def count_potential_forking_tokens(
    token_entries: List[Dict[str, Any]],
) -> Tuple[int, int]:
    # (unchanged)
    potential_forking_tokens = 0
    potential_forking_positions = 0
    for entry in token_entries:
        current_position_forking = False
        for candidate in entry.get("top_candidates", []):
            candidate_token = candidate.get("token_raw")
            candidate_probability = candidate.get("probability")
            if (not candidate_token) or (candidate_token == entry.get("token_raw")):
                continue
            if not is_safe_to_fork(candidate_token):
                continue
            if (
                not isinstance(candidate_probability, (int, float))
                or candidate_probability < args.alternate_min_prob
            ):
                continue
            potential_forking_tokens += 1
            current_position_forking = True
        if current_position_forking:
            potential_forking_positions += 1
    return potential_forking_tokens, potential_forking_positions


# NEW: exact per-entry alternative count (same filters as sampling)
def _count_valid_alternatives_for_entry(entry: Dict[str, Any]) -> int:
    m = 0
    for candidate in entry.get("top_candidates", []) or []:
        cand_raw = candidate.get("token_raw")
        cand_prob = candidate.get("probability")
        if (
            (not cand_raw)
            or (cand_raw == entry.get("token_raw"))
            or (not is_safe_to_fork(cand_raw))
        ):
            continue
        if (not isinstance(cand_prob, (int, float))) or (
            cand_prob < args.alternate_min_prob
        ):
            continue
        m += 1
    return m


# NEW: estimate fork-sampling usage & cost BEFORE we actually sample
def estimate_fork_sampling_usage(
    prompt: str, token_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    For each position t with at least one valid alternative:
      - We will sample each valid alternative m_t times: (m_t * S) samples
      - We will also sample the greedy token S times (to build o_{t,w*})
      - Input tokens per sample ≈ len(prompt_tokens) + t + 1
      - Output tokens per sample ≈ T - (t + 1)
    where S = args.samples_per_fork, T = len(token_entries).
    """
    T = len(token_entries)
    if T <= 0:
        return {
            "estimated_input_tokens": 0,
            "estimated_output_tokens": 0,
            "estimated_input_cost_usd": 0.0,
            "estimated_output_cost_usd": 0.0,
            "estimated_total_cost_usd": 0.0,
            "positions_with_alternatives": 0,
            "total_alt_tokens_considered": 0,
            "total_samples_planned": 0,
            "assumptions": "No tokens in greedy completion.",
        }

    prompt_tokens = count_tokens(prompt)

    total_in = 0
    total_out = 0
    positions_with_alts = 0
    total_alt_tokens = 0
    total_samples = 0

    for entry in token_entries:
        t = entry.get("token_index")
        if not isinstance(t, int):
            continue
        m_t = _count_valid_alternatives_for_entry(entry)
        if m_t <= 0:
            continue

        positions_with_alts += 1
        total_alt_tokens += m_t

        # per-sample token estimates
        in_per = prompt_tokens + t + 1
        out_per = max(0, T - (t + 1))

        # alternative tokens at this position
        alt_samples_here = m_t * args.samples_per_fork
        total_in += alt_samples_here * in_per
        total_out += alt_samples_here * out_per
        total_samples += alt_samples_here

        # greedy token samples at this position
        greedy_samples_here = args.samples_per_fork
        total_in += greedy_samples_here * in_per
        total_out += greedy_samples_here * out_per
        total_samples += greedy_samples_here

    input_cost = (
        total_in * PER_TOKEN_INPUT
    )  # PRICES_PER_MTOK.get(args.model, DEFAULT_PRICE)["input"] / 1_000_000.0
    output_cost = (
        total_out * PER_TOKEN_OUTPUT
    )  # PRICES_PER_MTOK.get(args.model, DEFAULT_PRICE)["output"] / 1_000_000.0

    return {
        "estimated_input_tokens": int(total_in),
        "estimated_output_tokens": int(total_out),
        "estimated_input_cost_usd": float(input_cost),
        "estimated_output_cost_usd": float(output_cost),
        "estimated_total_cost_usd": float(input_cost + output_cost),
        "positions_with_alternatives": int(positions_with_alts),
        "total_alt_tokens_considered": int(total_alt_tokens),
        "total_samples_planned": int(total_samples),
        "assumptions": (
            "Input per sample ≈ len(prompt)+t+1; output per sample ≈ T-(t+1); "
            "we sample each valid alt token m_t for S times and also the greedy token S times, "
            "with S=samples_per_fork and T=len(greedy_tokens)."
        ),
        "pricing": {
            "model": args.model,
            "usd_per_mtok_input": USD_PER_MTOK_INPUT,
            "usd_per_mtok_output": USD_PER_MTOK_OUTPUT,
        },
    }


def _existing_branch_map(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Map token -> branch dict for quick lookup while resuming."""
    out: Dict[str, Dict[str, Any]] = {}
    for br in entry.get("branches", []) or []:
        tok = br.get("token")
        if isinstance(tok, str):
            out[tok] = br
    return out


def _pending_samples_for_entry(entry: Dict[str, Any]) -> int:
    """How many samples still needed for this entry (across all eligible alt tokens)."""
    need = 0
    existing = _existing_branch_map(entry)
    for cand in entry.get("top_candidates", []) or []:
        cand_raw = cand.get("token_raw")
        cand_clean = cand.get("token")
        cand_prob = cand.get("probability")
        if (
            (not cand_raw)
            or (cand_raw == entry.get("token_raw"))
            or (not is_safe_to_fork(cand_raw))
        ):
            continue
        if (not isinstance(cand_prob, (int, float))) or (
            cand_prob < args.alternate_min_prob
        ):
            continue
        prev = existing.get(cand_clean, {})
        have = len(prev.get("samples", []) or [])
        if have < args.samples_per_fork:
            need += args.samples_per_fork - have
    return need


async def sample_fork_branches(
    problem_idx: int,
    prompt: str,
    token_entries: List[Dict[str, Any]],
    *,
    semaphore: asyncio.Semaphore,
    base_completion_text_raw: str,
    rollout_file: Path,
    result_sink: Dict[str, Any],
) -> None:
    """Resume-aware branch sampler. Saves progress to rollout_file after each branch update."""
    total_needed = sum(_pending_samples_for_entry(e) for e in token_entries)
    pbar = tqdm(total=total_needed, desc="Sampling alternative branches (resumable)")

    for entry in token_entries:
        # Prepare existing branches dict for this entry (if any)
        entry.setdefault("branches", [])
        existing_map = _existing_branch_map(entry)
        entry.setdefault("alt_candidates", [])

        # ---------- sample alternate token branches w != w* ----------
        alt_tokens_at_this_position = False
        for candidate in list(entry.get("top_candidates", []) or []):
            cand_raw = candidate.get("token_raw")
            cand_clean = candidate.get("token")
            cand_prob = candidate.get("probability")
            if (
                (not cand_raw)
                or (cand_raw == entry.get("token_raw"))
                or (not is_safe_to_fork(cand_raw))
            ):
                entry["top_candidates"].remove(candidate)
                continue
            if (not isinstance(cand_prob, (int, float))) or (
                cand_prob < args.alternate_min_prob
            ):
                entry["top_candidates"].remove(candidate)
                continue
            alt_tokens_at_this_position = True
            # Build fork prompt with char offsets + decoded token text
            offset = entry.get("text_offset")
            if not isinstance(offset, int):
                offset = 0
            alt_text = detok_for_api(cand_raw)
            if alt_text is None:
                # skip if can't decode safely
                continue
            fork_prompt = f"{prompt}{base_completion_text_raw[:offset]}{alt_text}"
            # Compact entry
            candidate.pop("token_raw", None)
            candidate.pop("logprob", None)
            entry["alt_candidates"].append(candidate)
            # Ensure branch container exists
            branch = existing_map.get(cand_clean)
            if branch is None:
                branch = {"token": cand_clean, "probability": cand_prob, "samples": []}
                entry["branches"].append(branch)
                existing_map[cand_clean] = branch

            have = len(branch.get("samples", []) or [])
            need = max(0, args.samples_per_fork - have)
            if need == 0:
                continue

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
                for _ in range(need)
            ]
            responses = await asyncio.gather(*tasks, return_exceptions=True)

            # Record usage for each response
            for res in responses:
                gen_id = generate_timestamp_id()
                if isinstance(res, dict):
                    record_generation(gen_id, str(problem_idx), res)

            for add_idx, result in enumerate(responses):
                sample_idx = have + add_idx  # next index
                if isinstance(result, Exception):
                    branch["samples"].append(
                        {"sample_index": sample_idx, "error": str(result)}
                    )
                    continue
                if not isinstance(result, dict):
                    branch["samples"].append(
                        {"sample_index": sample_idx, "error": "Invalid response object"}
                    )
                    continue
                if result.get("error"):
                    branch["samples"].append(
                        {
                            "sample_index": sample_idx,
                            "error": result.get("error"),
                            "details": result.get("details"),
                        }
                    )
                    continue
                sample_choices = result.get("choices") or []
                if not sample_choices:
                    branch["samples"].append(
                        {"sample_index": sample_idx, "error": "No choices returned"}
                    )
                    continue

                sample_choice = sample_choices[0]
                sample_text_raw, sample_text = extract_choice_text(sample_choice)
                logprob_sum = sum(
                    v
                    for v in (
                        sample_choice.get("logprobs", {}).get("token_logprobs") or []
                    )
                    if isinstance(v, (int, float))
                )
                continuation_probability = (
                    math.exp(logprob_sum)
                    if sample_choice.get("logprobs")
                    else "Unknown"
                )
                outcome = default_outcome_extractor(sample_text)
                branch["samples"].append(
                    {
                        "sample_index": sample_idx,
                        "final_answer_text": sample_text,
                        "logprob_sum": logprob_sum,
                        "continuation_probability": continuation_probability,
                        "outcome": outcome,
                    }
                )

            # After each branch update, persist progress
            result_sink["token_steps"] = token_entries
            result_sink["status"] = "fork_sampling_in_progress"
            safe_write_json(rollout_file, result_sink)

            pbar.update(need)
        # ---------- sample greedy token branch w* ----------
        if alt_tokens_at_this_position:
            w_star_raw = entry.get("token_raw")
            w_star_clean = entry.get("token")
            p_w_star = entry.get("probability") or 0.0
            offset = entry.get("text_offset")
            if not isinstance(offset, int):
                offset = 0
            w_star_text = (
                detok_for_api(w_star_raw) if isinstance(w_star_raw, str) else None
            )
            if w_star_text:
                fork_prompt_w_star = (
                    f"{prompt}{base_completion_text_raw[:offset]}{w_star_text}"
                )

                # ensure a branch object exists for the greedy token
                entry.setdefault("branches", [])
                existing_map = _existing_branch_map(entry)
                branch_star = existing_map.get(w_star_clean)
                if branch_star is None:
                    branch_star = {
                        "token": w_star_clean,
                        "probability": p_w_star,
                        "samples": [],
                    }
                    entry["branches"].append(branch_star)
                    existing_map[w_star_clean] = branch_star

                have = len(branch_star.get("samples", []) or [])
                need = max(0, args.samples_per_fork - have)
                if need > 0:
                    tasks = [
                        make_api_request(
                            fork_prompt_w_star,
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
                        for _ in range(need)
                    ]
                    responses = await asyncio.gather(*tasks, return_exceptions=True)

                    for res in responses:
                        gen_id = generate_timestamp_id()
                        if isinstance(res, dict):
                            record_generation(gen_id, str(problem_idx), res)

                    for add_idx, result in enumerate(responses):
                        sample_idx = have + add_idx
                        if isinstance(result, Exception):
                            branch_star["samples"].append(
                                {"sample_index": sample_idx, "error": str(result)}
                            )
                            continue
                        if not isinstance(result, dict):
                            branch_star["samples"].append(
                                {
                                    "sample_index": sample_idx,
                                    "error": "Invalid response object",
                                }
                            )
                            continue
                        if result.get("error"):
                            branch_star["samples"].append(
                                {
                                    "sample_index": sample_idx,
                                    "error": result.get("error"),
                                    "details": result.get("details"),
                                }
                            )
                            continue
                        sample_choices = result.get("choices") or []
                        if not sample_choices:
                            branch_star["samples"].append(
                                {
                                    "sample_index": sample_idx,
                                    "error": "No choices returned",
                                }
                            )
                            continue

                        sample_choice = sample_choices[0]
                        _, sample_text = extract_choice_text(sample_choice)
                        logprob_sum = sum(
                            v
                            for v in (
                                sample_choice.get("logprobs", {}).get("token_logprobs")
                                or []
                            )
                            if isinstance(v, (int, float))
                        )
                        continuation_probability = (
                            math.exp(logprob_sum)
                            if sample_choice.get("logprobs")
                            else "Unknown"
                        )
                        outcome = default_outcome_extractor(sample_text)
                        branch_star["samples"].append(
                            {
                                "sample_index": sample_idx,
                                "final_answer_text": sample_text,
                                "logprob_sum": logprob_sum,
                                "continuation_probability": continuation_probability,
                                "outcome": outcome,
                            }
                        )

                    # persist after greedy branch update
                    result_sink["token_steps"] = token_entries
                    result_sink["status"] = "fork_sampling_in_progress"
                    safe_write_json(rollout_file, result_sink)
                    pbar.update(need)

        if entry["branches"] == []:
            entry.pop("branches", None)
        # for entry in token_entries:
        entry.pop("token_raw", None)
        entry.pop("logprob", None)
        entry.pop("top_candidates", None)

    pbar.close()


async def sample_o0_distribution(
    problem_idx: int,
    prompt: str,
    *,
    semaphore: asyncio.Semaphore,
    samples: int,
) -> Dict[str, float]:
    tasks = [
        make_api_request(
            prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.continuation_max_tokens,
            logprobs=False,
            semaphore=semaphore,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            min_p=args.min_p,
            top_k=args.top_k,
        )
        for _ in range(samples)
    ]
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for r in responses:
        gen_id = generate_timestamp_id()
        if isinstance(r, dict):
            record_generation(gen_id, str(problem_idx), r)

    counts: Dict[str, int] = defaultdict(int)
    valid = 0
    for r in responses:
        if not isinstance(r, dict) or r.get("error"):
            continue
        ch = (r.get("choices") or [None])[0]
        if not ch:
            continue
        _, txt = extract_choice_text(ch)
        outcome = default_outcome_extractor(txt)
        if isinstance(outcome, str) and outcome:
            counts[outcome] += 1
            valid += 1

    if valid == 0:
        return {}

    return {k: v / float(valid) for k, v in counts.items()}


async def generate_rollout_resume(
    problem_idx: int,
    problem: Dict[str, Any],
    *,
    semaphore: asyncio.Semaphore,
    rollout_file: Path,
    existing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # (skeleton setup unchanged)
    result: Dict[str, Any] = existing or {}
    result.setdefault("problem_index", problem_idx)
    result.setdefault(
        "metadata",
        {
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
    )
    result.setdefault(
        "problem",
        {
            "problem_text": problem.get("problem"),
            "level": problem.get("level"),
            "type": problem.get("type"),
            "gt_answer": problem.get("gt_answer"),
        },
    )

    prompt = result.get("prompt")
    if not isinstance(prompt, str):
        prompt = (
            "Solve this math problem step by step. You MUST put your final answer in \\boxed{}."
            " Problem: "
            f"{problem['problem']} Solution: \n<think>\n"
        )
        result["prompt"] = prompt
        safe_write_json(rollout_file, result)

    # 1) Base path (if missing)
    if "base" not in result or "token_steps" not in result:
        base_data = await collect_base_path(
            problem, problem_idx, prompt, semaphore=semaphore
        )
        if base_data.get("error"):
            return {
                "problem_index": problem_idx,
                "error": base_data.get("error"),
                "details": base_data.get("details"),
            }
        result["base"] = {
            "completion": base_data.get("completion_text"),
            "completion_raw": base_data.get("completion_text_raw"),
            "outcome": base_data.get("base_outcome"),
            "is_correct": base_data.get("base_is_correct"),
        }
        result["token_steps"] = base_data.get("token_entries", [])
        result["status"] = "base_collected"
        safe_write_json(rollout_file, result)

    token_entries = result.get("token_steps", [])
    base_completion_text_raw: str = result.get("base", {}).get("completion_raw") or ""

    # 1.5) NEW — write the fork-sampling cost estimate BEFORE any sampling
    if "estimated_usage" not in result:
        try:
            estimate = estimate_fork_sampling_usage(result["prompt"], token_entries)
        except Exception as e:
            estimate = {
                "error": f"estimation_failed: {e}",
                "estimated_input_tokens": 0,
                "estimated_output_tokens": 0,
                "estimated_input_cost_usd": 0.0,
                "estimated_output_cost_usd": 0.0,
                "estimated_total_cost_usd": 0.0,
            }
        result["estimated_usage"] = estimate
        # Save immediately so it's recorded even if sampling is interrupted
        safe_write_json(rollout_file, result)

    # 2) Fork sampling (resume-aware)
    remaining = sum(_pending_samples_for_entry(e) for e in token_entries)
    if remaining > 0:
        await sample_fork_branches(
            problem_idx,
            prompt,
            token_entries,
            semaphore=semaphore,
            base_completion_text_raw=base_completion_text_raw,
            rollout_file=rollout_file,
            result_sink=result,
        )
        result["token_steps"] = token_entries
        result["status"] = "fork_sampling_done"
        safe_write_json(rollout_file, result)

    # 3) o0 omitted in this script (as in your provided version)

    # 4) Build ot / otw if missing
    outcome_distributions = result.setdefault("outcome_distributions", {})
    if ("ot" not in outcome_distributions) or ("otw" not in outcome_distributions):
        ot_list, otw_list = build_outcome_distributions_exact(
            result.get("base", {}).get("outcome", "__empty__"),
            token_entries,
            o0_dist=None,  # this script version omits o0 sampling
        )
        outcome_distributions["ot"] = ot_list
        outcome_distributions["otw"] = otw_list
        result["outcome_distributions"] = outcome_distributions
        result["status"] = "ot_built"
        safe_write_json(rollout_file, result)

    # 5) Drift series if missing
    if "drift_series" not in result:
        drift_series = compute_drift_series_from_ot(
            result["outcome_distributions"]["ot"]
        )
        result["drift_series"] = drift_series
        result["status"] = "done"
        safe_write_json(rollout_file, result)

    return result


async def process_problem(
    problem_idx: int, problem: Dict[str, Any], *, semaphore: asyncio.Semaphore
) -> None:
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(parents=True, exist_ok=True)
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(problem, f, indent=2, ensure_ascii=False)

    rollout_file = problem_dir / "rollout_analysis.json"
    error_file = problem_dir / "rollout_error.json"

    # NEW: resume from file if it exists (unless --force)
    existing = None
    if rollout_file.exists() and not args.force:
        existing = load_json_if_exists(rollout_file)
        if existing:
            print(
                f"[Resume] Found checkpoint for problem {problem_idx} with status={existing.get('status')}."
            )

    result = await generate_rollout_resume(
        problem_idx,
        problem,
        semaphore=REQUEST_SEMAPHORE,
        rollout_file=rollout_file,
        existing=existing,
    )

    if result.get("error"):
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return
    if error_file.exists():
        error_file.unlink()


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
