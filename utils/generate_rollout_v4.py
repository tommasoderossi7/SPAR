# generate_rollout.py  — fork-safe, no-mojibake version
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

# NEW: tokenizer for detok of single pieces
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
    description="Collect forked rollouts with outcome distributions and drift analysis."
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
parser.add_argument("-cc", "--concurrency", type=int, default=4)
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
    # fallback for Llama distill
    return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


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
    # fast path: try vocab lookup
    tid = tok.convert_tokens_to_ids(token_str)
    if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
        text = tok.decode([tid], clean_up_tokenization_spaces=False)
        return text
    else:
        return None

    # fallback: very small structural map (last resort)
    STRUCTURAL_FALLBACK = {
        "\u0120": " ",
        "Ġ": " ",
        "\u010a": "\n",
        "\u010b": "\n",
        "Ċ": "\n",
        "ċ": "\n",
        "▁": " ",  # SentencePiece metaspace
    }
    out = token_str
    for k, v in STRUCTURAL_FALLBACK.items():
        out = out.replace(k, v)
    out = ftfy.fix_encoding(out) if ftfy is not None else out
    return out


# ---------------- utility & API ----------------


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


# (kept for logs only)
STRUCTURAL_MAP = {"Ġ": " ", "\u0120": " ", "Ċ": "\n", "\u010a": "\n", "\u010b": "\n"}


def is_safe_to_fork(token_str: str) -> bool:
    """
    Accept if the token decodes to some text (incl. whitespace/newline).
    We only reject if decoding fails or yields non-printable controls (other than \n/\t).
    """
    decoded = detok_for_api(token_str)
    if decoded is None:
        return False
    # allow newline/tabs; otherwise drop control chars
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
    text_offsets = logprobs.get("text_offset") or logprobs.get(
        "text_offsets"
    )  # Novita uses text_offset

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


def build_outcome_distributions_exact(
    base_outcome: str,
    token_entries: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      ot_list  : list of { "t": int, "token": str, "dist": { outcome: prob } }  (normalized)
      otw_list : list of { "t": int, "w": str, "p_w": float, "dist": { outcome: prob } } (each branch normalized)
    Implements Eq. (1) and (2) in the paper:
      o_{t,w} = sum_s p(x_{>t}^{(s)} | prefix, w) * R(...) ; then normalize within branch
      o_t     = sum_w p(w | prefix) * o_{t,w} ; then normalize across outcomes
    """

    # Continuation logprob for the base greedy path tail at each t (for info only; not needed for delta R)
    tail_sums = compute_tail_logprob_sums(token_entries)

    ot_list: List[Dict[str, Any]] = []
    otw_all: List[Dict[str, Any]] = []

    # Iterate tokens of the base path
    for entry_idx, entry in enumerate(token_entries):
        t = entry.get("token_index")
        token_at_t = entry.get("token") or ""

        # p(w* | prefix) = probability of the greedy token at this t
        p_w_star = entry.get("probability") or 0.0

        # ---- Build o_{t,w*} (base/greedy branch) as a delta on the base outcome ----
        # Per Sec. 2.2 they form histograms weighted by sample probabilities;
        # for the greedy branch we only have the single observed outcome R for the base path,
        # so the branch distribution is a delta at base_outcome.
        ot_w_star = {}
        if isinstance(base_outcome, str) and base_outcome:
            ot_w_star[base_outcome] = 1.0  # already normalized (delta)

        otw_all.append(
            {
                "t": t,
                "w": token_at_t,
                "p_w": p_w_star,
                "dist": dict(ot_w_star),  # normalized
            }
        )

        # ---- Build o_{t,w} for each alternate branch using continuation probabilities ----
        accounted_prob_mass = p_w_star
        branches = entry.get("branches") or []
        for br in branches:
            w_txt = br.get("token") or ""
            p_w = br.get("probability") or 0.0
            accounted_prob_mass += p_w

            # Aggregate weighted histogram over outcomes using continuation probability
            # weight_s = exp(sum logprobs) = p(x_{>t}^{(s)} | prefix, w)
            raw_hist: Dict[str, float] = defaultdict(float)
            for sample in br.get("samples", []):
                if sample.get("error"):
                    continue
                outcome = sample.get("outcome")
                if not isinstance(outcome, str) or not outcome:
                    continue
                lp_sum = sample.get("logprob_sum")
                if not isinstance(lp_sum, (int, float)):
                    continue
                cont_prob = math.exp(lp_sum)
                raw_hist[outcome] += cont_prob

            # Normalize within branch to get o_{t,w}
            dist_w = _normalize(raw_hist)

            otw_all.append(
                {
                    "t": t,
                    "w": w_txt,
                    "p_w": p_w,
                    "dist": dist_w,  # normalized branch histogram
                }
            )

        # ---- Form o_t = sum_w p(w|prefix) * o_{t,w}  ----
        # Collect all branch dists for this t:
        mix_raw: Dict[str, float] = defaultdict(float)
        for rec in otw_all:
            if rec["t"] != t:
                continue
            _weighted_add(mix_raw, rec["dist"], rec["p_w"])

        # We only keep top-K tokens (>= args.alternate_min_prob) + the greedy token,
        # so renormalize across outcomes to sum to 1 for plotting/analysis
        o_t_norm = _normalize(mix_raw)

        ot_list.append({"t": t, "token": token_at_t, "dist": o_t_norm})

    return ot_list, otw_all


def l2_distance(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    keys = set(dist_a.keys()).union(dist_b.keys())
    return math.sqrt(
        sum((dist_a.get(key, 0.0) - dist_b.get(key, 0.0)) ** 2 for key in keys)
    )


def compute_drift_series_from_ot(ot_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    y_t = L2(o_0, o_t) as in Sec. 2.3 (semantic drift -> univariate time series).
    """
    drift: List[Dict[str, Any]] = []
    if not ot_list:
        return drift
    o0 = ot_list[0].get("dist", {}) or {}
    for rec in ot_list:
        t = rec.get("t", -1)
        ot = rec.get("dist", {}) or {}
        drift_val = l2_distance(o0, ot)
        drift.append({"t": t, "drift": drift_val})
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

    # NEW: store the completion text so we can use char offsets later
    for entry in token_entries:
        entry["completion_text_raw"] = completion_text_raw

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


async def sample_fork_branches(
    problem_idx: int,
    prompt: str,
    token_entries: List[Dict[str, Any]],
    *,
    semaphore: asyncio.Semaphore,
    potential_forking_tokens: int,
) -> None:
    pbar = tqdm(
        total=potential_forking_tokens * args.samples_per_fork,
        desc="Sampling alternative branches",
    )
    for entry in token_entries:
        branches: List[Dict[str, Any]] = []
        entry["alt_candidates"] = []
        for candidate in list(
            entry.get("top_candidates", [])
        ):  # iterate over a copy since we may remove
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

            # ===== KEY CHANGE: build fork prompt with char offsets + decoded token text =====
            base_completion_text_raw: str = entry.get("completion_text_raw") or ""
            offset = entry.get("text_offset")
            if not isinstance(offset, int):
                # Fallback: if offset missing, approximate with decoded prefix length of previous tokens
                # (rare on Novita since text_offset is present)
                offset = 0
            alt_text = detok_for_api(cand_raw)
            if alt_text is None:
                raise RuntimeError(
                    f"Failed to decode token {cand_raw!r} via tokenizer for model {args.model}"
                )
            fork_prompt = f"{prompt}{base_completion_text_raw[:offset]}{alt_text}"

            # Compact entry
            candidate.pop("token_raw", None)
            candidate.pop("logprob", None)
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

            for i in range(args.samples_per_fork):
                generation_id = generate_timestamp_id()
                if isinstance(responses[i], dict):
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

                print(
                    f"\nForking on token {cand_clean!r} at offset {offset} → prompt preview:"
                )
                print(
                    _clean_token_display(fork_prompt)[:50]
                    + " ... "
                    + _clean_token_display(fork_prompt)[-50:]
                )

                sample_choice = sample_choices[0]
                sample_text_raw, sample_text = extract_choice_text(sample_choice)
                print(
                    f"\nRaw Response (head/tail): {sample_text_raw[:100]} ... {sample_text_raw[-100:]}"
                )
                _, sample_tokens, sample_logprobs = extract_tokens_and_logprobs(
                    sample_choice.get("logprobs")
                )
                print(
                    f"Cleaned sampled text: {sample_text[:100]} ... {sample_text[-100:]}"
                )
                print(
                    f"Extracted tokens: {sample_tokens[:10]} ... {sample_tokens[-10:]}"
                )

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
                samples.append(
                    {
                        "sample_index": sample_idx,
                        "final_answer_text": sample_text,
                        "logprob_sum": logprob_sum,
                        "continuation_probability": continuation_probability,
                        "outcome": outcome,
                    }
                )

            branches.append(
                {"token": cand_clean, "probability": cand_prob, "samples": samples}
            )
            entry["branches"] = branches

        # tidy
        entry.pop("top_candidates", None)
        entry.pop("token_raw", None)
        entry.pop("logprob", None)
        entry.pop("completion_text_raw", None)
    pbar.close()


async def generate_rollout(
    problem_idx: int, problem: Dict[str, Any], *, semaphore: asyncio.Semaphore
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

    token_entries = base_data.get("token_entries", [])
    potential_forking_tokens, potential_forking_positions = (
        count_potential_forking_tokens(token_entries)
    )

    await sample_fork_branches(
        problem_idx,
        prompt,
        token_entries,
        semaphore=semaphore,
        potential_forking_tokens=potential_forking_tokens,
    )

    # --- NEW: compute exact ot and otw per the paper ---
    ot_list, otw_list = build_outcome_distributions_exact(
        base_data.get("base_outcome", "__empty__"), token_entries
    )
    drift_series = compute_drift_series_from_ot(ot_list)

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
        "token_steps": token_entries,  # unchanged; contains raw sampling details
        # --- MINIMAL saving here ---
        "outcome_distributions": {
            "ot": ot_list,  # [{ "t": int, "token": str, "dist": { outcome: prob } }, ...]
            "otw": otw_list,  # [{ "t": int, "w": str, "p_w": float, "dist": { outcome: prob } }, ...]
        },
        "drift_series": drift_series,  # [{ "t": int, "drift": float }, ...]
    }


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
