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
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY_SPAR")
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
parser.add_argument(
    "--intervention-mode",
    type=str,
    choices=["forced", "biased"],
    default="forced",
    help="forced = force each alt token and mix by alt probs; biased = ban base token with logit bias and sample the pool.",
)
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

parser.add_argument(
    "--external-base-root",
    type=str,
    required=True,
    help="Root of sentence-level cache, e.g. thought_anchors/.cache/math_rollouts_hf",
)
parser.add_argument(
    "--external-base-kind",
    type=str,
    choices=["correct_base_solution", "incorrect_base_solution"],
    default="correct_base_solution",
)
parser.add_argument("--external-temp", type=str, default="0.6")
parser.add_argument("--external-top-p", type=str, default="0.95")
parser.add_argument("--external-base-file", type=str, default="base_solution.json")
# IMPORTANT: we *require* using the external prompt (no fallback)
parser.add_argument(
    "--external-use-prompt",
    action="store_true",
    help="Must be set; we will fail if the external prompt is missing.",
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
    # USD per 1M tokens
    "deepseek/deepseek-r1-distill-qwen-14b": {"input": 0.15, "output": 0.15},
}
DEFAULT_PRICE = {"input": 0.15, "output": 0.15}


def _price_per_token(model: str) -> Tuple[float, float]:
    p = PRICES_PER_MTOK.get(model, DEFAULT_PRICE)
    return p["input"] / 1_000_000.0, p["output"] / 1_000_000.0


USD_PER_MTOK_INPUT = PRICES_PER_MTOK.get(args.model, DEFAULT_PRICE)["input"]
USD_PER_MTOK_OUTPUT = PRICES_PER_MTOK.get(args.model, DEFAULT_PRICE)["output"]
PER_TOKEN_INPUT, PER_TOKEN_OUTPUT = _price_per_token(args.model)

# Shared HTTPX client (connection pooling + HTTP/2)
HTTPX_CLIENT: Optional[httpx.AsyncClient] = None


async def _get_httpx_client() -> httpx.AsyncClient:
    """
    One client for the whole run: enables connection pooling & keep-alive.
    HTTP/2 allows multiplexing when the server supports it.
    """
    global HTTPX_CLIENT
    if HTTPX_CLIENT is None:
        limits = httpx.Limits(
            max_connections=args.concurrency * 2,
            max_keepalive_connections=args.concurrency,
        )
        HTTPX_CLIENT = httpx.AsyncClient(
            timeout=240.0,
            http2=True,  # enable HTTP/2 if server supports it
            limits=limits,
            headers=build_headers(),  # set auth once
        )
    return HTTPX_CLIENT


def _ensure_current_usage(result_sink: Dict[str, Any]) -> None:
    if "current_usage" not in result_sink:
        result_sink["current_usage"] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
        }


def _add_usage_numbers(
    result_sink: Dict[str, Any], prompt_tokens: int, completion_tokens: int
) -> None:
    _ensure_current_usage(result_sink)
    cu = result_sink["current_usage"]
    cu["input_tokens"] += int(prompt_tokens)
    cu["output_tokens"] += int(completion_tokens)
    cu["input_cost_usd"] += prompt_tokens * PER_TOKEN_INPUT
    cu["output_cost_usd"] += completion_tokens * PER_TOKEN_OUTPUT
    cu["total_cost_usd"] = cu["input_cost_usd"] + cu["output_cost_usd"]


def _accumulate_from_api_response(
    result_sink: Dict[str, Any],
    prompt_tokens_fallback: int,
    response_obj: Dict[str, Any],
    completion_text_fallback: Optional[str],
) -> None:
    usage = (response_obj or {}).get("usage") or {}
    pt = usage.get("prompt_tokens")
    ct = usage.get("completion_tokens")
    if isinstance(pt, int) and isinstance(ct, int):
        _add_usage_numbers(result_sink, pt, ct)
    else:
        # Fallback: count tokens locally
        ct_local = (
            count_tokens(completion_text_fallback or "")
            if completion_text_fallback
            else 0
        )
        _add_usage_numbers(result_sink, prompt_tokens_fallback, ct_local)


def _external_model_dir_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def _external_problem_dir(problem_idx: int) -> Path:
    root = Path(args.external_base_root)
    model_dir = _external_model_dir_name(args.model)
    seg = f"temperature_{args.external_temp}_top_p_{args.external_top_p}"
    p = root / model_dir / seg / args.external_base_kind / f"problem_{problem_idx}"
    if not p.exists():
        raise RuntimeError(f"External problem dir not found: {p}")
    return p


def _try_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_external_base_for_problem(problem_idx: int) -> Tuple[str, str]:
    """
    Reads the prompt and raw base completion produced by the sentence-level run.
    No fallbacks: both files must exist.
    """
    if not args.external_base_dir:
        raise RuntimeError(
            "external_base_dir is not set but load_external_base_for_problem was called."
        )

    ext_dir = _external_problem_dir(problem_idx)
    ext_path = ext_dir / args.external_base_file
    if not ext_path.exists():
        raise RuntimeError(f"External file not found: {ext_path}")
    ext_data = _try_read_json(ext_path)

    ext_prompt = ext_data.get("prompt")
    ext_completion_raw = ext_data.get("solution")
    return ext_prompt, ext_completion_raw


def token_raw_to_id(token_raw: str) -> Optional[int]:
    try:
        tok = get_tokenizer(args.model)
        tid = tok.convert_tokens_to_ids(token_raw)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            return tid
    except Exception:
        pass
    return None


def _tokenize_completion_for_offsets(text: str):
    tok = get_tokenizer(args.model)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]  # [(start,end), ...] in the given text
    return ids, offs


def _lcp_len_ids(a_ids: List[int], b_ids: List[int]) -> int:
    n = min(len(a_ids), len(b_ids))
    i = 0
    while i < n and a_ids[i] == b_ids[i]:
        i += 1
    return i


import hashlib
from datetime import datetime


def _ext_base_signature(
    prompt: str, completion_raw: str, temp: float, top_p: float
) -> str:
    s = f"{prompt}\n<SEP>\n{completion_raw}\n<CFG>temp={temp}|top_p={top_p}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _tokenize_completion_for_offsets(
    text: str,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    tok = get_tokenizer(args.model)
    enc = tok(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = enc["input_ids"]
    offs = enc["offset_mapping"]  # list[(start, end)]
    return ids, offs


def _save_alignment_progress(
    result_sink: Dict[str, Any],
    rollout_file: Path,
    *,
    ext_prompt: str,
    ext_completion_raw: str,
    total_tokens: int,
    matched_tokens: int,
    sig: str,
    token_steps: List[Dict[str, Any]],
    status: str,
) -> None:
    # ensure base metadata for later pipeline steps is present early
    result_sink.setdefault("base", {})
    result_sink["base"]["completion_raw"] = ext_completion_raw
    result_sink["base"]["completion"] = _clean_token_display(ext_completion_raw)
    result_sink["base"]["outcome"] = default_outcome_extractor(
        result_sink["base"]["completion"]
    )

    result_sink["token_steps"] = token_steps
    result_sink["alignment"] = {
        "status": status,  # "in_progress" | "done"
        "matched_tokens": int(matched_tokens),
        "total_tokens": int(total_tokens),
        "sig": sig,
        "external": {
            "prompt": ext_prompt,
            "temperature": float(args.external_temp),
            "top_p": float(args.external_top_p),
        },
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    # keep top-level status coherent
    result_sink["status"] = "alignment_" + status
    safe_write_json(rollout_file, result_sink)


def _entries_from_logprobs(logprobs_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    # Reuse your existing normalizer so shapes match the sampler’s expectations
    return parse_logprob_entries(logprobs_dict or {})


def _clip_entries_to_lcp_and_rebase_offsets(
    entries: List[Dict[str, Any]],
    base_cursor_raw_chars: int,
    matched_text: str,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Keep only those tokens whose concatenation equals the matched_text (LCP),
    and rebase their text_offset to *absolute* offsets into the external base raw.
    """
    out, acc = [], ""
    for e in entries:
        # prefer faithful detok; fallback to cleaned for length accounting
        piece = detok_for_api(e.get("token_raw")) or (e.get("token") or "")
        if not piece:
            continue
        if len(acc) + len(piece) > len(matched_text):
            break

        acc_next = acc + piece
        # Build a fresh record with absolute offset and probabilities preserved
        rec = {
            "token_index": None,  # set by caller
            "token_raw": e.get("token_raw"),
            "token": e.get("token"),
            "logprob": e.get("logprob"),
            "probability": (
                math.exp(e["logprob"])
                if isinstance(e.get("logprob"), (int, float))
                else None
            ),
            "top_candidates": e.get("top_candidates") or [],
            "text_offset": base_cursor_raw_chars + len(acc),
        }
        out.append(rec)
        acc = acc_next

    return out, len(acc)  # acc_len == len(matched_text) if everything lines up


async def build_token_steps_via_iterative_lcp(
    problem_idx: int,
    ext_prompt: str,
    ext_completion_raw: str,
    *,
    semaphore: asyncio.Semaphore,
    rollout_file: Path,
    result_sink: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Iteratively align next-token distributions to the external base completion.
    Saves progress (token_steps + alignment state) after every successful advance.
    Retries up to 20 times on the same prefix when k == 0. No other fallbacks.
    """

    # External sampling params (match sentence-level)
    align_temp = float(args.external_temp)
    align_top_p = float(args.external_top_p)

    # Signature to guarantee we're resuming the same base & params
    sig = _ext_base_signature(ext_prompt, ext_completion_raw, align_temp, align_top_p)

    # Tokenize external base once
    ext_ids, ext_offs = _tokenize_completion_for_offsets(ext_completion_raw)
    N = len(ext_ids)
    if N == 0:
        raise RuntimeError("External base completion is empty after tokenization.")

    # Resume if alignment state exists and signature matches
    token_steps: List[Dict[str, Any]] = result_sink.get("token_steps") or []
    matched = len(token_steps)
    prev_state = result_sink.get("alignment") or {}
    if prev_state:
        if prev_state.get("sig") != sig:
            raise RuntimeError(
                "Existing alignment state does not match the current external base/signature."
            )
        # sanity: clamp
        matched = min(matched, int(prev_state.get("matched_tokens", matched)))

    # Initial save (so restarts know we're aligning this base)
    _save_alignment_progress(
        result_sink,
        rollout_file,
        ext_prompt=ext_prompt,
        ext_completion_raw=ext_completion_raw,
        total_tokens=N,
        matched_tokens=matched,
        sig=sig,
        token_steps=token_steps,
        status=("done" if matched >= N else "in_progress"),
    )
    if matched >= N:
        # Nothing to do
        return {
            "token_steps": token_steps,
            "base_outcome": result_sink["base"]["outcome"],
            "base_is_correct": False,  # caller can fill
        }

    # Main loop
    while matched < N:
        # Current prompt = ext_prompt + already-aligned prefix (by char offsets)
        cur_char_start = 0 if matched == 0 else ext_offs[matched][0]
        current_prompt = f"{ext_prompt}{ext_completion_raw[:cur_char_start]}"

        # ---- retry loop on k == 0 ----
        success = False
        last_entries = None
        for attempt in range(1, 21):  # 1..20
            response = await make_api_request(
                current_prompt,
                temperature=align_temp,
                top_p=align_top_p,
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
                raise RuntimeError(
                    f"Alignment API error: {response.get('error')} | {response.get('details')}"
                )

            record_generation(generate_timestamp_id(), str(problem_idx), response)
            choice = (response.get("choices") or [None])[0]
            if not choice or not choice.get("logprobs"):
                print(
                    f"[Align p{problem_idx}] attempt {attempt}/20 @token {matched}: "
                    f"no logprobs (k=0); retrying..."
                )
                continue

            entries = parse_logprob_entries(choice.get("logprobs")) or []
            last_entries = entries

            # Convert generated tokens to ids
            gen_ids: List[int] = []
            for e in entries:
                tid = token_raw_to_id(e.get("token_raw"))
                gen_ids.append(tid if tid is not None else -1)

            # Longest common prefix vs remaining external ids
            rem = N - matched
            M = min(rem, len(gen_ids))
            k = 0
            while k < M and gen_ids[k] == ext_ids[matched + k]:
                k += 1

            if k > 0:
                success = True
                break
            else:
                print(
                    f"[Align p{problem_idx}] attempt {attempt}/20 @token {matched}: "
                    f"k=0 (no progress); retrying..."
                )

        if not success:
            # Persist the "stalled" state so you can see where it died; then raise
            _save_alignment_progress(
                result_sink,
                rollout_file,
                ext_prompt=ext_prompt,
                ext_completion_raw=ext_completion_raw,
                total_tokens=N,
                matched_tokens=matched,
                sig=sig,
                token_steps=token_steps,
                status="in_progress",
            )
            raise RuntimeError(
                f"Alignment stalled at token {matched} after 20 attempts (k=0 each time)."
            )

        # Commit only the first k aligned entries and checkpoint immediately
        for j in range(k):
            e = last_entries[j]
            abs_t = matched + j
            lp = e.get("logprob")
            prob = math.exp(lp) if isinstance(lp, (int, float)) else None

            token_steps.append(
                {
                    "token_index": abs_t,
                    "token_raw": e.get("token_raw"),
                    "token": e.get("token"),
                    "probability": prob,
                    "top_candidates": e.get("top_candidates"),
                    "text_offset": int(ext_offs[abs_t][0]),
                }
            )

        matched += k

        # Progressive checkpoint after every advance
        _save_alignment_progress(
            result_sink,
            rollout_file,
            ext_prompt=ext_prompt,
            ext_completion_raw=ext_completion_raw,
            total_tokens=N,
            matched_tokens=matched,
            sig=sig,
            token_steps=token_steps,
            status=("done" if matched >= N else "in_progress"),
        )

    # Completed
    return {
        "token_steps": token_steps,
        "base_outcome": result_sink["base"]["outcome"],
        "base_is_correct": False,  # caller can fill vs GT
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
    logit_bias: Optional[Dict[str, float]] = None,
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
    if logit_bias is not None:
        # OpenAI-compatible format: ids as strings
        payload["logit_bias"] = {str(k): float(v) for k, v in logit_bias.items()}

    client = await _get_httpx_client()
    max_retries = retries if retries is not None else args.max_retries
    base_delay = 2.0

    async def _post() -> httpx.Response:
        return await client.post(NOVITA_API_URL, json=payload)

    for attempt in range(max_retries):
        try:
            if semaphore:
                async with semaphore:
                    response = await _post()
            else:
                response = await _post()
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


def token_raw_to_id(token_raw: str) -> Optional[int]:
    try:
        tok = get_tokenizer(args.model)
        tid = tok.convert_tokens_to_ids(token_raw)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            return tid
    except Exception:
        pass
    return None


def prob_true_from_dist(dist: Dict[str, float], gt: Optional[str]) -> float:
    if not gt:
        return 0.0
    p = 0.0
    for ans, w in dist.items():
        try:
            if check_answer(ans, gt):
                p += float(w)
        except Exception:
            continue
    return float(p)


def mix_dists(weighted: List[Tuple[float, Dict[str, float]]]) -> Dict[str, float]:
    acc = defaultdict(float)
    for w, d in weighted:
        if w <= 0 or not d:
            continue
        for k, v in d.items():
            acc[k] += w * v
    s = sum(acc.values())
    return {k: v / s for k, v in acc.items()} if s > 0 else {}


def kl_full_dist(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    num = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), eps)
        qk = max(q.get(k, 0.0), eps)
        num += pk * math.log(pk / qk)
    return float(num)


def kl_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return float(p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))


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


def estimate_fork_sampling_usage(
    prompt: str, token_entries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    T = len(token_entries)
    prompt_tokens = count_tokens(prompt)

    total_in = total_out = positions_with_alts = total_alt_tokens = total_samples = 0

    for entry in token_entries:
        t = entry.get("token_index")
        if not isinstance(t, int):
            continue
        m_t = _count_valid_alternatives_for_entry(entry)
        if m_t <= 0:
            continue
        positions_with_alts += 1
        total_alt_tokens += m_t

        in_per = prompt_tokens + t + 1
        out_per = max(0, T - (t + 1))

        if args.intervention_mode == "forced":
            alt_samples_here = m_t * args.samples_per_fork
            greedy_samples_here = args.samples_per_fork
        else:  # biased
            alt_samples_here = args.samples_per_fork  # one ALT_POOL
            greedy_samples_here = args.samples_per_fork

        total_in += (alt_samples_here + greedy_samples_here) * in_per
        total_out += (alt_samples_here + greedy_samples_here) * out_per
        total_samples += alt_samples_here + greedy_samples_here

    input_cost = total_in * PER_TOKEN_INPUT
    output_cost = total_out * PER_TOKEN_OUTPUT

    return {
        "estimated_input_tokens": int(total_in),
        "estimated_output_tokens": int(total_out),
        "estimated_input_cost_usd": float(input_cost),
        "estimated_output_cost_usd": float(output_cost),
        "estimated_total_cost_usd": float(input_cost + output_cost),
        "positions_with_alternatives": int(positions_with_alts),
        "total_alt_tokens_considered": int(total_alt_tokens),
        "total_samples_planned": int(total_samples),
        "mode": args.intervention_mode,
        "assumptions": (
            "Input per sample ≈ len(prompt)+t+1; output per sample ≈ T-(t+1); "
            f"mode={args.intervention_mode}."
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
    need = 0
    existing = _existing_branch_map(entry)

    # Count greedy (baseline) branch if the position has any valid alts
    m_t = _count_valid_alternatives_for_entry(entry)
    if m_t > 0:
        w_star = entry.get("token")
        br_star = existing.get(w_star, {})
        have_star = len(br_star.get("samples", []) or [])
        if have_star < args.samples_per_fork:
            need += args.samples_per_fork - have_star

    if m_t <= 0:
        return need

    if args.intervention_mode == "forced":
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
    else:
        # one pooled alt branch
        pool = existing.get("__ALT_POOL__", {})
        have = len(pool.get("samples", []) or [])
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
    total_needed = sum(_pending_samples_for_entry(e) for e in token_entries)
    pbar = tqdm(
        total=total_needed, desc=f"Sampling branches ({args.intervention_mode})"
    )

    for entry in token_entries:
        entry.setdefault("branches", [])
        existing_map = _existing_branch_map(entry)
        entry.setdefault("alt_candidates", [])

        # Build/keep a list of valid alternates (filtered by safety & min prob)
        valid_alts = []
        for candidate in list(entry.get("top_candidates", []) or []):
            cand_raw = candidate.get("token_raw")
            cand_clean = candidate.get("token")
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
            # compact the stored candidate
            compact = {"token": cand_clean, "probability": cand_prob}
            valid_alts.append((cand_raw, compact))
        entry["alt_candidates"] = [c for _, c in valid_alts]

        if len(valid_alts) == 0:
            # nothing to do at this position
            continue

        # Convenience
        t_offset = entry.get("text_offset") or 0
        w_star_raw = entry.get("token_raw")
        w_star_clean = entry.get("token")
        p_w_star = float(entry.get("probability") or 0.0)

        # ---------- Baseline (greedy token) branch ----------
        greedy_text = detok_for_api(w_star_raw) if isinstance(w_star_raw, str) else None
        if greedy_text:
            fork_prompt_w_star = (
                f"{prompt}{base_completion_text_raw[:t_offset]}{greedy_text}"
            )
            prompt_tokens_branch = count_tokens(fork_prompt_w_star)
            br_star = existing_map.get(w_star_clean)
            if br_star is None:
                br_star = {
                    "token": w_star_clean,
                    "probability": p_w_star,
                    "samples": [],
                }
                entry["branches"].append(br_star)
                existing_map[w_star_clean] = br_star
            have = len(br_star.get("samples", []) or [])
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
                for add_idx, result_resp in enumerate(responses):
                    sample_idx = have + add_idx
                    if isinstance(result_resp, Exception):
                        br_star["samples"].append(
                            {"sample_index": sample_idx, "error": str(result_resp)}
                        )
                        continue
                    if not isinstance(result_resp, dict):
                        br_star["samples"].append(
                            {
                                "sample_index": sample_idx,
                                "error": "Invalid response object",
                            }
                        )
                        continue
                    if result_resp.get("error"):
                        br_star["samples"].append(
                            {
                                "sample_index": sample_idx,
                                "error": result_resp.get("error"),
                                "details": result_resp.get("details"),
                            }
                        )
                        continue
                    sample_choices = result_resp.get("choices") or []
                    if not sample_choices:
                        br_star["samples"].append(
                            {"sample_index": sample_idx, "error": "No choices returned"}
                        )
                        continue
                    sample_choice = sample_choices[0]
                    _, sample_text = extract_choice_text(sample_choice)
                    _accumulate_from_api_response(
                        result_sink, prompt_tokens_branch, result_resp, sample_text
                    )
                    logprob_sum = sum(
                        v
                        for v in (
                            sample_choice.get("logprobs", {}).get("token_logprobs")
                            or []
                        )
                        if isinstance(v, (int, float))
                    )
                    outcome = default_outcome_extractor(sample_text)
                    br_star["samples"].append(
                        {
                            "sample_index": sample_idx,
                            "final_answer_text": sample_text,
                            "logprob_sum": logprob_sum,
                            "outcome": outcome,
                        }
                    )
                result_sink["token_steps"] = token_entries
                result_sink["status"] = "fork_sampling_in_progress"
                safe_write_json(rollout_file, result_sink)
                pbar.update(need)

        # ---------- Counterfactual side ----------
        if args.intervention_mode == "forced":
            # one branch per alternate candidate
            for cand_raw, compact in valid_alts:
                alt_text = detok_for_api(cand_raw)
                if alt_text is None:
                    continue
                fork_prompt = f"{prompt}{base_completion_text_raw[:t_offset]}{alt_text}"
                prompt_tokens_branch = count_tokens(fork_prompt)
                tok = compact["token"]
                pw = float(compact["probability"] or 0.0)
                br = existing_map.get(tok)
                if br is None:
                    br = {"token": tok, "probability": pw, "samples": []}
                    entry["branches"].append(br)
                    existing_map[tok] = br
                have = len(br.get("samples", []) or [])
                need = max(0, args.samples_per_fork - have)
                if need <= 0:
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
                for res in responses:
                    gen_id = generate_timestamp_id()
                    if isinstance(res, dict):
                        record_generation(gen_id, str(problem_idx), res)
                for add_idx, result_resp in enumerate(responses):
                    sample_idx = have + add_idx
                    if isinstance(result_resp, Exception):
                        br["samples"].append(
                            {"sample_index": sample_idx, "error": str(result_resp)}
                        )
                        continue
                    if not isinstance(result_resp, dict):
                        br["samples"].append(
                            {
                                "sample_index": sample_idx,
                                "error": "Invalid response object",
                            }
                        )
                        continue
                    if result_resp.get("error"):
                        br["samples"].append(
                            {
                                "sample_index": sample_idx,
                                "error": result_resp.get("error"),
                                "details": result_resp.get("details"),
                            }
                        )
                        continue
                    sample_choices = result_resp.get("choices") or []
                    if not sample_choices:
                        br["samples"].append(
                            {"sample_index": sample_idx, "error": "No choices returned"}
                        )
                        continue
                    sample_choice = sample_choices[0]
                    _, sample_text = extract_choice_text(sample_choice)
                    _accumulate_from_api_response(
                        result_sink, prompt_tokens_branch, result_resp, sample_text
                    )
                    logprob_sum = sum(
                        v
                        for v in (
                            sample_choice.get("logprobs", {}).get("token_logprobs")
                            or []
                        )
                        if isinstance(v, (int, float))
                    )
                    outcome = default_outcome_extractor(sample_text)
                    br["samples"].append(
                        {
                            "sample_index": sample_idx,
                            "final_answer_text": sample_text,
                            "logprob_sum": logprob_sum,
                            "outcome": outcome,
                        }
                    )
                result_sink["token_steps"] = token_entries
                result_sink["status"] = "fork_sampling_in_progress"
                safe_write_json(rollout_file, result_sink)
                pbar.update(need)

        else:
            # biased: one pooled branch with logit bias that bans the greedy token
            fork_prompt = f"{prompt}{base_completion_text_raw[:t_offset]}"
            prompt_tokens_branch = count_tokens(fork_prompt)
            # sum of alt probs (for OT construction and for logging)
            alt_prob_sum = float(
                sum(float(c["probability"] or 0.0) for _, c in valid_alts)
            )
            br = existing_map.get("__ALT_POOL__")
            if br is None:
                br = {
                    "token": "__ALT_POOL__",
                    "probability": alt_prob_sum,
                    "samples": [],
                    "note": "logit_bias banning base token",
                }
                entry["branches"].append(br)
                existing_map["__ALT_POOL__"] = br

            have = len(br.get("samples", []) or [])
            need = max(0, args.samples_per_fork - have)
            if need > 0:
                # block the base token id
                tid = token_raw_to_id(w_star_raw)
                bias = {tid: -100.0} if isinstance(tid, int) else None
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
                        logit_bias=bias,
                    )
                    for _ in range(need)
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for res in responses:
                    gen_id = generate_timestamp_id()
                    if isinstance(res, dict):
                        record_generation(gen_id, str(problem_idx), res)
                for add_idx, result_resp in enumerate(responses):
                    sample_idx = have + add_idx
                    if isinstance(result_resp, Exception):
                        br["samples"].append(
                            {"sample_index": sample_idx, "error": str(result_resp)}
                        )
                        continue
                    if not isinstance(result_resp, dict):
                        br["samples"].append(
                            {
                                "sample_index": sample_idx,
                                "error": "Invalid response object",
                            }
                        )
                        continue
                    if result_resp.get("error"):
                        br["samples"].append(
                            {
                                "sample_index": sample_idx,
                                "error": result_resp.get("error"),
                                "details": result_resp.get("details"),
                            }
                        )
                        continue
                    sample_choices = result_resp.get("choices") or []
                    if not sample_choices:
                        br["samples"].append(
                            {"sample_index": sample_idx, "error": "No choices returned"}
                        )
                        continue
                    sample_choice = sample_choices[0]
                    _, sample_text = extract_choice_text(sample_choice)
                    _accumulate_from_api_response(
                        result_sink, prompt_tokens_branch, result_resp, sample_text
                    )
                    logprob_sum = sum(
                        v
                        for v in (
                            sample_choice.get("logprobs", {}).get("token_logprobs")
                            or []
                        )
                        if isinstance(v, (int, float))
                    )
                    outcome = default_outcome_extractor(sample_text)
                    br["samples"].append(
                        {
                            "sample_index": sample_idx,
                            "final_answer_text": sample_text,
                            "logprob_sum": logprob_sum,
                            "outcome": outcome,
                        }
                    )
                result_sink["token_steps"] = token_entries
                result_sink["status"] = "fork_sampling_in_progress"
                safe_write_json(rollout_file, result_sink)
                pbar.update(need)

        # cleanup of heavy fields that we no longer need here
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


def compute_token_cf_metrics_inplace(result_obj: Dict[str, Any]) -> None:
    steps = result_obj.get("token_steps") or []
    gt = ((result_obj.get("problem") or {}).get("gt_answer")) or None
    mode = (result_obj.get("metadata") or {}).get("intervention_mode", "forced")

    for entry in steps:
        t = entry.get("token_index")
        if t is None:
            continue
        br_map = {br.get("token"): br for br in (entry.get("branches") or [])}
        w_star = entry.get("token")
        br_star = br_map.get(w_star)

        if not br_star or not br_star.get("samples"):
            # can't form a baseline
            continue

        # Baseline = greedy token's empirical distribution
        base_dist = _empirical_outcome_dist(br_star.get("samples"))

        # Counterfactual
        if mode == "forced":
            # mix over alternates ONLY, renormalized on the alt set
            alts = []
            for tok, br in br_map.items():
                if tok == w_star or tok == "__ALT_POOL__":
                    continue
                pw = br.get("probability")
                if not isinstance(pw, (int, float)) or pw <= 0:
                    continue
                dist_w = _empirical_outcome_dist(br.get("samples"))
                alts.append((float(pw), dist_w))

            # renormalize weights over alt set
            s = sum(w for w, _ in alts)
            if s > 0:
                alts = [(w / s, d) for (w, d) in alts]
            cf_dist = mix_dists(alts) if alts else {}
        else:
            # biased: take the ALT_POOL empirical distribution
            pool = br_map.get("__ALT_POOL__")
            cf_dist = _empirical_outcome_dist(pool.get("samples")) if pool else {}

        # Metrics
        p_true_base = prob_true_from_dist(base_dist, gt)
        p_true_cf = prob_true_from_dist(cf_dist, gt)
        delta_acc = p_true_cf - p_true_base
        kl_true = kl_bernoulli(p_true_cf, p_true_base)
        kl_full = kl_full_dist(cf_dist, base_dist)

        entry["cf_metrics"] = {
            "mode": mode,
            "baseline_dist": base_dist,
            "cf_dist": cf_dist,
            "p_true_baseline": float(p_true_base),
            "p_true_cf": float(p_true_cf),
            "delta_acc": float(delta_acc),
            "kl_true": float(kl_true),
            "kl_full": float(kl_full),
        }


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
    result["metadata"]["intervention_mode"] = args.intervention_mode
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

    # 1) Base path / Alignment
    # If the user gave an external-base-dir, we align to that external base.
    # Otherwise we fall back to the original base collection logic.

    if args.external_base_dir:
        # Load the sentence-level prompt + base completion (raw)
        ext_prompt, ext_completion_raw = load_external_base_for_problem(problem_idx)

        # Ensure metadata records the external config for reproducibility
        result["metadata"]["external_base_dir"] = args.external_base_dir
        result["metadata"]["external_temp"] = args.external_temp
        result["metadata"]["external_top_p"] = args.external_top_p

        # Use the *external* prompt for the rest of the pipeline
        result["prompt"] = ext_prompt
        safe_write_json(rollout_file, result)

        # If we already aligned this base (status 'alignment_done'), we can skip.
        aligned = (result.get("alignment") or {}).get("status") == "done"
        if not aligned or ("token_steps" not in result):
            # Run the iterative LCP alignment (progressively saves into the same result)
            align_res = await build_token_steps_via_iterative_lcp(
                problem_idx,
                ext_prompt,
                ext_completion_raw,
                semaphore=semaphore,
                rollout_file=rollout_file,
                result_sink=result,
            )

        # At this point token_steps and base.completion(_raw) are persisted by _save_alignment_progress
        # Fill correctness now (vs ground truth)
        base_outcome = (result.get("base") or {}).get("outcome")
        is_correct = False
        if problem.get("gt_answer") and base_outcome and base_outcome != "__empty__":
            is_correct = check_answer(base_outcome, problem["gt_answer"])

        result["base"]["is_correct"] = bool(is_correct)
        result["status"] = "base_collected"
        safe_write_json(rollout_file, result)

    else:
        # Original path: collect base completion from the model directly
        if "base" not in result or "token_steps" not in result:
            base_data = await collect_base_path(
                problem, problem_idx, result["prompt"], semaphore=semaphore
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
        _ensure_current_usage(result)  # initialize progressive counters to zero
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

    # 2b) Compute per-token CF metrics (Δacc, KL)
    compute_token_cf_metrics_inplace(result)
    result["status"] = "cf_metrics_built"
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
    try:
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
                print(
                    f"No problems contain substring '{args.problem_substring}'. Exiting."
                )
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
    finally:
        client = HTTPX_CLIENT
        if client is not None:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
