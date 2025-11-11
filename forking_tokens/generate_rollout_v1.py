#!/usr/bin/env python3
# generate_rollout.py — external-base cached top-k + forced/biased rollouts
# Progressive/resumable:
#   • Base next-token top-k cache: saves after every token updated
#   • Intervention sampling: merges with existing rollout & saves after each batch

import os
import re
import json
import math
import time
import uuid
import random
import asyncio
import shutil
import unicodedata
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Set, Tuple, Union

import httpx
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Config & CLI
# =============================================================================

NOVITA_API_URL = "https://api.novita.ai/openai/v1/completions"
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY_SPAR")
if not NOVITA_API_KEY:
    raise RuntimeError("NOVITA_API_KEY_SPAR is not set in the environment.")

parser = ArgumentParser(
    description="Token-level rollouts from an external base completion (forced/biased) with cached base top-k (progressive/resumable)."
)

# Model & continuation sampling
parser.add_argument(
    "-m", "--model", type=str, default="deepseek/deepseek-r1-distill-qwen-14b"
)
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default=0.6,
    help="Continuation sampling temperature.",
)
parser.add_argument(
    "-tp", "--top_p", type=float, default=0.95, help="Continuation sampling top-p."
)
parser.add_argument(
    "-cmt",
    "--continuation-max-tokens",
    type=int,
    default=8192,
    help="Max tokens for rollouts.",
)
parser.add_argument(
    "-ctl",
    "--continuation-top-logprobs",
    type=int,
    default=1,
    help="Logprob detail on continuations.",
)

# Intervention mode
parser.add_argument(
    "--intervention-mode", choices=["forced", "biased"], default="forced"
)
parser.add_argument("-sps", "--samples-per-fork", type=int, default=30)

# Alternatives selection (at sampling time)
parser.add_argument(
    "-atk",
    "--alternate-top-k",
    type=int,
    default=10,
    help="Max alternatives considered per step.",
)
parser.add_argument(
    "-amp",
    "--alternate-min-prob",
    type=float,
    default=0.05,
    help="Min prob threshold for a valid alt.",
)

# Precompute & cache base next-token top-k
parser.add_argument(
    "--topk-base-completion",
    type=int,
    default=10,
    help="Top-k alternatives to cache (per token) for the base completion (max_tokens=1 lookups).",
)
parser.add_argument(
    "--external-base-root",
    type=str,
    required=True,
    help="Root of sentence-level cache (has prompt+solution per problem).",
)
parser.add_argument(
    "--external-kind",
    choices=["correct_base_solution", "incorrect_base_solution"],
    default="correct_base_solution",
)
parser.add_argument(
    "--external-temp",
    type=float,
    default=0.6,
    help="Temperature used when the base completion was generated.",
)
parser.add_argument(
    "--external-top-p",
    type=float,
    default=0.95,
    help="Top-p used when the base completion was generated.",
)
parser.add_argument(
    "--external-base-file",
    type=str,
    default="base_solution.json",
    help="File in each problem dir containing {'prompt','solution',...}.",
)

# Where to save the reusable per-token alternatives file
parser.add_argument(
    "--alt-cache-root",
    type=str,
    default="math_rollouts",
    help=r"Root dir for reusable cache.",
)

# Problem selection
parser.add_argument(
    "-ip",
    "--include-problems",
    type=str,
    default=None,
    help="Comma-separated problem ids to run.",
)
parser.add_argument(
    "-ep",
    "--exclude-problems",
    type=str,
    default=None,
    help="Comma-separated problem ids to skip.",
)

# Output & runtime
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="math_rollouts",
    help="Where to write rollouts.",
)
parser.add_argument("-cc", "--concurrency", type=int, default=6)
parser.add_argument(
    "-qps",
    "--requests-per-second",
    type=float,
    default=1.5,
    help="Global QPS limiter (approx).",
)
parser.add_argument(
    "-mr",
    "--max-retries",
    type=int,
    default=8,
    help="Max retries for 429/5xx/Cloudflare.",
)
parser.add_argument("-s", "--seed", type=int, default=44)
parser.add_argument(
    "-f",
    "--force",
    action="store_true",
    help="Overwrite checkpoints & recalc caches if set.",
)

args = parser.parse_args()
random.seed(args.seed)

# =============================================================================
# Tokenizer helpers
# =============================================================================

_TOKENIZER_CACHE: Dict[str, Any] = {}


def _guess_hf_tokenizer_name(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    elif "llama" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


def get_tokenizer(model_name: str):
    key = _guess_hf_tokenizer_name(model_name)
    tok = _TOKENIZER_CACHE.get(key)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(key, use_fast=True)
        _TOKENIZER_CACHE[key] = tok
    return tok


def token_raw_to_id(token_raw: str) -> Optional[int]:
    try:
        tok = get_tokenizer(args.model)
        tid = tok.convert_tokens_to_ids(token_raw)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            return tid
    except Exception:
        pass
    return None


def detok_for_api(token_str: str) -> Optional[str]:
    if not isinstance(token_str, str) or token_str == "":
        return None
    tok = get_tokenizer(args.model)
    tid = tok.convert_tokens_to_ids(token_str)
    if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
        return tok.decode([tid], clean_up_tokenization_spaces=False)
    return None


def _tokenize_with_offsets(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    tok = get_tokenizer(args.model)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


# =============================================================================
# Tiny text utils
# =============================================================================


def _clean_token_display(s: object) -> str:
    text = s if isinstance(s, str) else str(s)
    if not text:
        return ""
    text = (
        text.replace("▁", " ")
        .replace("\u0120", " ")
        .replace("Ġ", " ")
        .replace("\u010a", "\n")
        .replace("\u010b", "\n")
        .replace("Ċ", "\n")
        .replace("ċ", "\n")
    )
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[ \t]+", " ", text)


def extract_boxed_answers(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"\\boxed\{([^{}]+)\}", text) if m.strip()]


def default_outcome_extractor(text: str) -> str:
    ans = extract_boxed_answers(text)
    if ans:
        return ans[0]
    return "__empty__"


def _normalize_ans(x: str) -> str:
    return re.sub(r"\s+", "", x.strip()).lower()


def check_answer(pred: str, gt: Optional[str]) -> bool:
    if not gt:
        return False
    p = _normalize_ans(pred)
    g = _normalize_ans(gt)
    try:
        return abs(float(p) - float(g)) <= 1e-6
    except Exception:
        return p == g


def is_safe_to_fork(token_str: str) -> bool:
    decoded = detok_for_api(token_str)
    if decoded is None:
        return False
    if any(
        unicodedata.category(c).startswith("C") and c not in "\n\t" for c in decoded
    ):
        return False
    return True


# =============================================================================
# HTTP / Novita API with robust retries (429 / 524 / 525)
# =============================================================================

# ====== Metrics (RPM) ======
from collections import deque

_REQ_SENT_TIMES = deque()  # monotonic seconds when an HTTP request is SENT
_RESP_RCVD_TIMES = deque()  # monotonic seconds when an HTTP response ARRIVES
_METRICS_LOCK = asyncio.Lock()


async def _mark_request_sent():
    now = time.monotonic()
    async with _METRICS_LOCK:
        _REQ_SENT_TIMES.append(now)
        cutoff = now - 60.0
        while _REQ_SENT_TIMES and _REQ_SENT_TIMES[0] < cutoff:
            _REQ_SENT_TIMES.popleft()


async def _mark_response_arrived():
    now = time.monotonic()
    async with _METRICS_LOCK:
        _RESP_RCVD_TIMES.append(now)
        cutoff = now - 60.0
        while _RESP_RCVD_TIMES and _RESP_RCVD_TIMES[0] < cutoff:
            _RESP_RCVD_TIMES.popleft()


async def _rpm_snapshot() -> Tuple[int, int]:
    now = time.monotonic()
    async with _METRICS_LOCK:
        cutoff = now - 60.0
        # prune both deques
        while _REQ_SENT_TIMES and _REQ_SENT_TIMES[0] < cutoff:
            _REQ_SENT_TIMES.popleft()
        while _RESP_RCVD_TIMES and _RESP_RCVD_TIMES[0] < cutoff:
            _RESP_RCVD_TIMES.popleft()
        return len(_REQ_SENT_TIMES), len(_RESP_RCVD_TIMES)


async def _rpm_logger(stop_event: asyncio.Event, *, every_seconds: float = 10.0):
    global _TARGET_RPS
    last_sent = last_recv = -1
    while not stop_event.is_set():
        sent, recv = await _rpm_snapshot()
        err429 = await _429s_last_minute()

        if sent != last_sent or recv != last_recv:
            print(
                f"[RPM] last 60s — sent={sent:3d}, recv={recv:3d}, rps_target={_TARGET_RPS:.2f}, 429={err429}"
            )
            last_sent, last_recv = sent, recv

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=every_seconds)
        except asyncio.TimeoutError:
            pass


_ERR429_TIMES = deque()


async def _mark_429():
    now = time.monotonic()
    async with _METRICS_LOCK:
        _ERR429_TIMES.append(now)
        cutoff = now - 60.0
        while _ERR429_TIMES and _ERR429_TIMES[0] < cutoff:
            _ERR429_TIMES.popleft()


async def _429s_last_minute() -> int:
    now = time.monotonic()
    async with _METRICS_LOCK:
        cutoff = now - 60.0
        while _ERR429_TIMES and _ERR429_TIMES[0] < cutoff:
            _ERR429_TIMES.popleft()
        return len(_ERR429_TIMES)


HTTPX_CLIENT: Optional[httpx.AsyncClient] = None
_RATE_LOCK = asyncio.Lock()
_NEXT_ALLOWED = 0.0  # monotonic seconds


def build_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NOVITA_API_KEY}",
        "Content-Type": "application/json",
    }


async def _get_httpx_client() -> httpx.AsyncClient:
    global HTTPX_CLIENT
    if HTTPX_CLIENT is None:
        # ========== CHANGE: Increased connection pool ==========
        limits = httpx.Limits(
            max_connections=args.concurrency * 4,
            max_keepalive_connections=args.concurrency * 2,
        )
        # =======================================================
        HTTPX_CLIENT = httpx.AsyncClient(
            timeout=240.0, http2=True, limits=limits, headers=build_headers()
        )
    return HTTPX_CLIENT


# Global, near other limiter globals
_TARGET_RPS = args.requests_per_second  # mutable target (req/s), e.g. 0.9 ≈ 54 RPM


async def _respect_qps_limit():
    global _NEXT_ALLOWED
    # use mutable target
    if _TARGET_RPS <= 0:
        return
    async with _RATE_LOCK:
        now = time.monotonic()
        wait_for = _NEXT_ALLOWED - now
        if wait_for > 0:
            await asyncio.sleep(wait_for)
            now = time.monotonic()
        _NEXT_ALLOWED = max(now, _NEXT_ALLOWED) + (1.0 / _TARGET_RPS)


async def make_api_request(
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_logprobs: int = 0,
    logprobs: bool = False,
    semaphore: Optional[asyncio.Semaphore] = None,
    logit_bias: Optional[Dict[int, int]] = None,
    stream: bool = False,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": stream,
    }
    if logprobs and top_logprobs > 0:
        payload["logprobs"] = int(top_logprobs)
    if logit_bias:
        payload["logit_bias"] = {str(k): int(v) for k, v in logit_bias.items()}

    client = await _get_httpx_client()

    async def _post() -> httpx.Response:
        await _mark_request_sent()
        resp = await client.post(NOVITA_API_URL, json=payload)
        await _mark_response_arrived()
        return resp

    max_retries = max(0, int(args.max_retries))
    base_delay = 1.0

    for attempt in range(max_retries + 1):
        try:
            # 1) Wait for the global rate slot FIRST (do NOT hold the request semaphore here)
            await _respect_qps_limit()

            # 2) Now take one connection slot just for the POST round-trip
            if semaphore:
                async with semaphore:
                    resp = await _post()
            else:
                resp = await _post()
        except Exception as e:
            if attempt >= max_retries:
                return {"error": f"HTTP exception: {e}"}
            await asyncio.sleep(base_delay * (2**attempt) * random.uniform(0.9, 1.4))
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as e:
                return {"error": f"JSON parse error: {e}", "details": resp.text}

        if resp.status_code == 429:
            await _mark_429()
            retry_after = resp.headers.get("Retry-After")
            delay = None
            if retry_after:
                try:
                    delay = float(retry_after)
                except Exception:
                    delay = None
            if delay is None:
                delay = base_delay * (2**attempt) * random.uniform(0.9, 1.4)
            if attempt >= max_retries:
                return {"error": f"API 429", "details": resp.text}
            await asyncio.sleep(delay)
            continue

        if resp.status_code in (524, 525, 502, 503, 504):
            if attempt >= max_retries:
                return {"error": f"API {resp.status_code}", "details": resp.text}
            # ========== CHANGE: Cap retry delay at 10s for 5xx ==========
            delay = min(base_delay * (1.5**attempt), 10.0)  # Faster backoff, capped
            # ============================================================
            await asyncio.sleep(delay)
            continue

        if attempt >= max_retries:
            return {"error": f"API {resp.status_code}", "details": resp.text}
        await asyncio.sleep(base_delay * (2**attempt) * random.uniform(0.9, 1.4))

    return {"error": "All retries exhausted"}


# =============================================================================
# Logprobs parsing (for base top-k cache)
# =============================================================================


def parse_top_logprob_block(block: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def _push(tok, lp):
        prob = math.exp(lp) if isinstance(lp, (int, float)) else None
        out.append(
            {
                "token_raw": tok,
                "token": _clean_token_display(tok),
                "logprob": lp,
                "probability": prob,
            }
        )

    if isinstance(block, dict):
        for tok, lp in block.items():
            _push(tok, lp)
    elif isinstance(block, list):
        for item in block:
            if isinstance(item, dict):
                _push(item.get("token") or item.get("text"), item.get("logprob"))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                _push(item[0], item[1])
    out.sort(key=lambda d: d.get("probability") or 0.0, reverse=True)
    return out


def parse_logprob_entries_one_token(
    logprobs: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not logprobs:
        return []
    tokens = logprobs.get("tokens")
    token_logprobs = logprobs.get("token_logprobs")
    top_logprobs = logprobs.get("top_logprobs")
    if (
        isinstance(tokens, list)
        and isinstance(token_logprobs, list)
        and len(tokens) >= 1
    ):
        lp = token_logprobs[0] if len(token_logprobs) else None
        return [
            {
                "token_raw": tokens[0],
                "token": _clean_token_display(tokens[0]),
                "logprob": lp,
                "probability": (math.exp(lp) if isinstance(lp, (int, float)) else None),
                # top candidates should contain all tokens in top_logprobs not only the first one
                "top_candidates": parse_top_logprob_block(top_logprobs),
            }
        ]
    content = logprobs.get("content")
    if isinstance(content, list) and content:
        blk = content[0]
        tok_raw, lp, top_block = "", None, None
        if isinstance(blk, dict):
            tok_raw = blk.get("token") or blk.get("text") or ""
            lp = blk.get("logprob")
            top_block = blk.get("top_logprobs")
        return [
            {
                "token_raw": tok_raw,
                "token": _clean_token_display(tok_raw),
                "logprob": lp,
                "probability": (math.exp(lp) if isinstance(lp, (int, float)) else None),
                "top_candidates": parse_top_logprob_block(top_block),
            }
        ]
    return []


# =============================================================================
# Windows-safe atomic-ish JSON writer (with retries)
# =============================================================================


def safe_write_json(path: Path, data: Dict[str, Any], *, attempts: int = 6) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for i in range(attempts):
        tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            try:
                os.replace(str(tmp), str(path))
                return
            except PermissionError:
                try:
                    if path.exists():
                        os.remove(str(path))
                    shutil.move(str(tmp), str(path))
                    return
                except Exception as e2:
                    last_err = e2
            except Exception as e:
                last_err = e
        except Exception as e:
            last_err = e
        time.sleep(0.15 * (i + 1))
    raise OSError(f"Failed to save {path}: {last_err}")


# =============================================================================
# External base loader & alt cache
# =============================================================================


def _external_model_dir_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def _external_problem_dir(problem_idx: int) -> Path:
    root = Path(args.external_base_root)
    model_dir = _external_model_dir_name(args.model)
    seg = f"temperature_{args.external_temp}_top_p_{args.external_top_p}"
    p = root / model_dir / seg / args.external_kind / f"problem_{problem_idx}"
    if not p.exists():
        raise RuntimeError(f"External problem dir not found: {p}")
    return p


def load_external_base_for_problem(problem_idx: int) -> Dict[str, Any]:
    ext_dir = _external_problem_dir(problem_idx)
    data_path = ext_dir / args.external_base_file
    if not data_path.exists():
        raise RuntimeError(f"External base file not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "prompt" not in data or "solution" not in data:
        raise RuntimeError(f"File missing 'prompt'/'solution': {data_path}")
    return data  # may include 'problem', 'gt_answer'


def load_external_problem(problem_idx: int) -> Optional[str]:
    ext_dir = _external_problem_dir(problem_idx)
    data_path = ext_dir / "problem.json"
    if not data_path.exists():
        raise RuntimeError(f"External base file not found: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def _alt_cache_path(problem_id: int, *, correct: bool) -> Path:
    root = (
        Path(args.alt_cache_root)
        if args.alt_cache_root
        else (Path(args.output_dir) / args.model.replace("/", "_"))
    )
    sub = (
        "correct_base_completion_aligned.json"
        if correct
        else "incorrect_base_completion_aligned.json"
    )
    return root / f"{args.model.replace('/', '_')}" / f"problem_{problem_id}" / sub


# ----- Progressive cache helpers -----


def _make_cache_skeleton(
    problem_id: int,
    prompt: str,
    base_completion: str,
    N: int,
    align_temp: float,
    align_top_p: float,
    gt_answer: Optional[str],
    is_correct: bool,
    prior: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create or normalize a cache structure with a fixed-length 'tokens' list that can contain None entries."""
    tokens_list: List[Optional[Dict[str, Any]]] = [None] * N
    if prior:
        # Migrate any existing tokens (list) into fixed-length slots (by t index)
        existing = prior.get("tokens")
        if isinstance(existing, list):
            for rec in existing:
                if isinstance(rec, dict) and isinstance(rec.get("t"), int):
                    t = rec["t"]
                    if 0 <= t < N:
                        tokens_list[t] = rec
    return {
        "model": args.model,
        "problem_id": problem_id,
        "prompt": prompt,
        "base_completion": base_completion,
        "topk_base_cached": int(prior.get("topk_base_cached", 0) if prior else 0),
        "external_sampling": {"temperature": align_temp, "top_p": align_top_p},
        "gt_answer": gt_answer,
        "is_correct": bool(is_correct),
        "total_tokens": N,
        "tokens": tokens_list,  # may include None for not-yet-computed
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


async def precompute_and_cache_base_topk(
    problem_id: int,
    prompt: str,
    base_completion: str,
    *,
    topk: int,
    semaphore: asyncio.Semaphore,
    force: bool,
    gt_answer: Optional[str],
) -> Dict[str, Any]:
    """Progressive/resumable: per-token next-token top-k with PARALLEL fetching."""
    base_outcome = default_outcome_extractor(base_completion)
    is_correct = (
        check_answer(base_outcome, gt_answer)
        if gt_answer
        else (args.external_kind.startswith("correct"))
    )
    cache_path = _alt_cache_path(problem_id, correct=is_correct)

    ids, offs = _tokenize_with_offsets(base_completion)
    N = len(ids)
    if N == 0:
        raise RuntimeError("Base completion tokenizes to zero tokens.")

    align_temp = float(args.external_temp)
    align_top_p = float(args.external_top_p)

    existing: Optional[Dict[str, Any]] = None
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = None

    cached = _make_cache_skeleton(
        problem_id,
        prompt,
        base_completion,
        N,
        align_temp,
        align_top_p,
        gt_answer,
        is_correct,
        prior=existing or {},
    )

    need_indices: List[int] = []
    for t in range(N):
        rec = cached["tokens"][t]
        if force or (rec is None):
            need_indices.append(t)
        else:
            prev_k = len((rec.get("candidates") or [])) if isinstance(rec, dict) else 0
            if prev_k < topk:
                need_indices.append(t)

    if not need_indices and (cached.get("topk_base_cached", 0) >= topk):
        return cached

    # ========== CHANGE: Parallel token fetching ==========
    async def fetch_token_topk(t: int) -> Tuple[int, Dict[str, Any]]:
        char_start = 0 if t == 0 else offs[t][0]
        prefix = f"{prompt}{base_completion[:char_start]}"
        resp = await make_api_request(
            prefix,
            temperature=align_temp,
            top_p=align_top_p,
            max_tokens=1,
            top_logprobs=topk,
            logprobs=True,
            semaphore=semaphore,
        )
        return t, resp

    pbar = tqdm(total=len(need_indices), desc=f"[cache base top-k] p{problem_id}")

    # Create all tasks to be run in parallel
    tasks = [fetch_token_topk(t) for t in need_indices]
    results = await asyncio.gather(*tasks)

    # Process results, but only save periodically to reduce disk I/O
    SAVE_BATCH_SIZE = 250
    # =====================================================

    for idx, (t, resp) in enumerate(results):
        if resp.get("error"):
            # Always save on error before raising
            safe_write_json(cache_path, cached)
            raise RuntimeError(
                f"next_token_candidates API error on token {t}: {resp.get('error')} :: {resp.get('details')}"
            )

        ch = (resp.get("choices") or [None])[0]
        if not ch or not ch.get("logprobs"):
            safe_write_json(cache_path, cached)
            raise RuntimeError("next_token_candidates returned no logprobs.")

        entries = parse_logprobs_entries_one_or_fallback(ch.get("logprobs"))
        if not entries:
            safe_write_json(cache_path, cached)
            raise RuntimeError("Failed to parse logprobs entry.")

        entry = entries[0]
        record = {
            "t": t,
            "text_offset": int(offs[t][0]),
            "base": {
                "token_raw": entry["token_raw"],
                "token": entry["token"],
                "probability": float(entry["probability"] or 0.0),
            },
            "candidates": [
                {
                    "token_raw": c.get("token_raw"),
                    "token": c.get("token"),
                    "probability": float(c.get("probability") or 0.0),
                }
                for c in (entry.get("top_candidates") or [])[:topk]
            ],
        }
        cached["tokens"][t] = record
        pbar.update(1)

        # ========== CHANGE: Batched saves ==========
        if (idx + 1) % SAVE_BATCH_SIZE == 0:
            cached["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            safe_write_json(cache_path, cached)
        # ===========================================

    pbar.close()

    # Final save to capture any remaining changes
    cached["topk_base_cached"] = max(int(cached.get("topk_base_cached", 0)), int(topk))
    cached["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    safe_write_json(cache_path, cached)

    return cached


def parse_logprobs_entries_one_or_fallback(
    logprobs: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    # Helper to keep function list local & consistent
    return parse_logprob_entries_one_token(logprobs)


# =============================================================================
# Rollout sampling helpers (forced / biased)
# =============================================================================


def _empirical_outcome_dist(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    total = 0
    for s in samples or []:
        if s.get("error"):
            continue
        o = s.get("outcome")
        if isinstance(o, str) and o:
            counts[o] = counts.get(o, 0) + 1
            total += 1
    if total == 0:
        return {}
    return {k: v / float(total) for k, v in counts.items()}


def mix_dists(weighted: List[Tuple[float, Dict[str, float]]]) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    for w, d in weighted:
        if w <= 0 or not d:
            continue
        for k, v in d.items():
            acc[k] = acc.get(k, 0.0) + w * v
    s = sum(acc.values())
    return {k: v / s for k, v in acc.items()} if s > 0 else {}


def prob_true_from_dist(dist: Dict[str, float], gt: Optional[str]) -> float:
    if not gt:
        return 0.0
    p = 0.0
    for ans, w in dist.items():
        if check_answer(ans, gt):
            p += float(w)
    return float(p)


def kl_full_dist(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    out = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), eps)
        qk = max(q.get(k, 0.0), eps)
        out += pk * math.log(pk / qk)
    return float(out)


def kl_bernoulli(p: float, q: float, eps: float = 1e-12) -> float:
    p = min(max(p, eps), 1.0 - eps)
    q = min(max(q, eps), 1.0 - eps)
    return float(p * math.log(p / q) + (1 - p) * math.log((1 - q) / (1 - p)))


def _existing_branch_map(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for br in entry.get("branches", []) or []:
        tok = br.get("token")
        if isinstance(tok, str):
            out[tok] = br
    return out


def _build_token_entries_from_cache(
    cache: Dict[str, Any], *, alt_topk: int
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in cache.get("tokens") or []:
        if not isinstance(rec, dict):
            continue
        base = rec["base"]
        cands = rec.get("candidates") or []
        trimmed = []
        for c in cands[:alt_topk]:
            if c["token_raw"] == base["token_raw"]:
                continue
            trimmed.append(
                {
                    "token_raw": c["token_raw"],
                    "token": c["token"],
                    "probability": c["probability"],
                }
            )
        out.append(
            {
                "token_index": rec["t"],
                "token_raw": base["token_raw"],
                "token": base["token"],
                "probability": base["probability"],
                "text_offset": rec["text_offset"],
                "top_candidates": trimmed,
                "branches": [],
            }
        )
    return out


def _pending_samples_for_entry(entry: Dict[str, Any]) -> int:
    need = 0
    existing = _existing_branch_map(entry)
    valid_alts = [
        c
        for c in (entry.get("top_candidates") or [])
        if is_safe_to_fork(c.get("token_raw"))
        and float(c.get("probability") or 0.0) >= args.alternate_min_prob
        and c.get("token") != entry.get("token")
    ]
    if len(valid_alts) == 0:
        return 0
    w_star = entry.get("token")
    br_star = existing.get(w_star, {})
    have_star = len(br_star.get("samples", []) or [])
    if have_star < args.samples_per_fork:
        need += args.samples_per_fork - have_star
    if args.intervention_mode == "forced":
        for c in valid_alts:
            tok = c["token"]
            prev = existing.get(tok, {})
            have = len(prev.get("samples", []) or [])
            if have < args.samples_per_fork:
                need += args.samples_per_fork - have
    else:
        pool = existing.get("__ALT_POOL__", {})
        have = len(pool.get("samples", []) or [])
        if have < args.samples_per_fork:
            need += args.samples_per_fork - have
    return need


# ----- Merge existing rollout token_steps with fresh candidates -----


def _merge_token_steps(
    existing_steps: List[Dict[str, Any]], fresh_steps: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Keep existing branches/samples; refresh top_candidates from fresh (trim) for the same t."""
    by_t_existing = {
        e.get("token_index"): e
        for e in existing_steps or []
        if isinstance(e.get("token_index"), int)
    }
    merged: List[Dict[str, Any]] = []
    for e in fresh_steps:
        t = e.get("token_index")
        prior = by_t_existing.get(t)
        if prior:
            # Copy branches/samples forward, refresh top_candidates
            keep = dict(prior)
            keep["top_candidates"] = e.get("top_candidates", [])
            # Also update token/probability/text_offset if they'd changed in cache
            keep["token"] = e.get("token", keep.get("token"))
            keep["token_raw"] = e.get("token_raw", keep.get("token_raw"))
            keep["probability"] = e.get("probability", keep.get("probability"))
            keep["text_offset"] = e.get("text_offset", keep.get("text_offset"))
            merged.append(keep)
        else:
            merged.append(e)
    # Preserve any entries that may exist in prior but not in fresh (shouldn't happen)
    prior_only = [
        e
        for t, e in by_t_existing.items()
        if t not in {x.get("token_index") for x in fresh_steps}
    ]
    merged.extend(prior_only)
    merged.sort(key=lambda r: r.get("token_index", 10**9))
    return merged


# ==================== NEW: per-entry metrics helper (progressive) ====================


def compute_entry_cf_metrics(
    entry: Dict[str, Any], gt: Optional[str], mode: str
) -> Optional[Dict[str, Any]]:
    """Compute cf metrics for a single token entry based on its current branches/samples."""
    br_map = {br.get("token"): br for br in (entry.get("branches") or [])}
    w_star = entry.get("token")
    br_star = br_map.get(w_star)
    if not br_star or not br_star.get("samples"):
        return None

    base_dist = _empirical_outcome_dist(br_star.get("samples"))

    if mode == "forced":
        alts: List[Tuple[float, Dict[str, float]]] = []
        for tok, br in br_map.items():
            if tok == w_star or tok == "__ALT_POOL__":
                continue
            pw = float(br.get("probability") or 0.0)
            if pw <= 0:
                continue
            alts.append((pw, _empirical_outcome_dist(br.get("samples"))))
        s = sum(w for w, _ in alts)
        if s > 0:
            alts = [(w / s, d) for (w, d) in alts]
        cf_dist = mix_dists(alts) if alts else {}
    else:
        pool = br_map.get("__ALT_POOL__")
        cf_dist = _empirical_outcome_dist(pool.get("samples")) if pool else {}
    p_true_base = prob_true_from_dist(base_dist, gt)
    p_true_cf = prob_true_from_dist(cf_dist, gt)
    metrics = {
        "mode": mode,
        "baseline_dist": base_dist,
        "cf_dist": cf_dist,
        "p_true_baseline": float(p_true_base),
        "p_true_cf": float(p_true_cf),
        "delta_acc": float(p_true_cf - p_true_base),
        "kl_true": float(kl_bernoulli(p_true_cf, p_true_base)),
        "kl_full": float(kl_full_dist(cf_dist, base_dist)),
    }
    return metrics


# ----- Sampling core with progressive saves (unchanged behavior, plus metrics) -----


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
    # --------------------------
    # PHASE 1: PLAN THE REQUESTS
    # --------------------------
    from typing import Tuple, Dict, Any, List

    requests_to_make: List[Dict[str, Any]] = []

    # per-branch live state: (t_idx, branch_token) -> dict of counters
    branch_state: Dict[Tuple[int, str], Dict[str, int]] = {}

    for entry in token_entries:
        entry.setdefault("branches", [])
        existing_map = _existing_branch_map(entry)

        valid_alts = [
            c
            for c in (entry.get("top_candidates") or [])
            if is_safe_to_fork(c.get("token_raw"))
            and float(c.get("probability") or 0.0) >= args.alternate_min_prob
            and c.get("token") != entry.get("token")
        ]

        t_offset = int(entry.get("text_offset") or 0)
        w_star_raw = entry.get("token_raw")
        w_star_clean = entry.get("token")
        p_w_star = float(entry.get("probability") or 0.0)

        # Only sample baseline if there will be at least one alt (forced) or a pool (biased)
        should_sample_baseline = len(valid_alts) > 0

        branches_to_check: List[Dict[str, Any]] = []

        if should_sample_baseline and is_safe_to_fork(w_star_raw):
            greedy_text = detok_for_api(w_star_raw)
            if greedy_text:
                branches_to_check.append(
                    {
                        "token_clean": w_star_clean,
                        "prob": p_w_star,
                        "fork_prompt": f"{prompt}{base_completion_text_raw[:t_offset]}{greedy_text}",
                        "logit_bias": None,
                    }
                )
            entry["valid_alternatives"] = {
                c["token"]: c["probability"] for c in valid_alts
            }
        else:
            entry.pop("branches", None)

        if args.intervention_mode == "forced":
            for c in valid_alts:
                alt_text = detok_for_api(c["token_raw"])
                if alt_text:
                    branches_to_check.append(
                        {
                            "token_clean": c["token"],
                            "prob": float(c.get("probability") or 0.0),
                            "fork_prompt": f"{prompt}{base_completion_text_raw[:t_offset]}{alt_text}",
                            "logit_bias": None,
                        }
                    )
        elif args.intervention_mode == "biased" and len(valid_alts) > 0:
            tid = token_raw_to_id(w_star_raw)
            bias = {tid: -100} if isinstance(tid, int) else None
            branches_to_check.append(
                {
                    "token_clean": "__ALT_POOL__",
                    "prob": sum(float(c.get("probability") or 0.0) for c in valid_alts),
                    "fork_prompt": f"{prompt}{base_completion_text_raw[:t_offset]}",
                    "logit_bias": bias,
                }
            )

        # Plan per-branch based on SUCCESSFUL samples only
        for binfo in branches_to_check:
            tok_clean = binfo["token_clean"]
            br = existing_map.get(tok_clean)
            if br is None:
                br = {
                    "token": tok_clean,
                    "probability": binfo["prob"],
                    "samples": [],
                    # >>> Persist request shape so later passes can reuse it <<<
                    "fork_prompt": binfo["fork_prompt"],
                    "logit_bias": binfo["logit_bias"],
                }
                entry["branches"].append(br)
                existing_map[tok_clean] = br
            else:
                # Backfill for resumes from older checkpoints
                br.setdefault("fork_prompt", binfo["fork_prompt"])
                br.setdefault("logit_bias", binfo["logit_bias"])

            successes_so_far = sum(
                1
                for s in (br.get("samples") or [])
                if isinstance(s, dict) and not s.get("error")
            )
            errors_so_far = sum(
                1
                for s in (br.get("samples") or [])
                if isinstance(s, dict) and s.get("error")
            )
            attempts_completed = successes_so_far + errors_so_far
            need_success = max(0, args.samples_per_fork - successes_so_far)

            key = (entry["token_index"], tok_clean)
            branch_state[key] = {
                "successes": successes_so_far,
                "errors": errors_so_far,
                "attempts": attempts_completed,  # COMPLETED only (success+error)
                "inflight": 0,  # actively running
                "queued": 0,  # in pending queue not yet launched
                "issued": 0,  # total launched so far
                "target": args.samples_per_fork,
            }

            if need_success > 0:
                next_idx = attempts_completed
                for i in range(need_success):
                    req = {
                        "context": {
                            "token_index": entry["token_index"],
                            "branch_token": tok_clean,
                            "sample_index": next_idx + i,
                        },
                        "api_call": {
                            # NOTE: read from the persisted branch
                            "prompt": br["fork_prompt"],
                            "logit_bias": br.get("logit_bias"),
                        },
                    }
                    requests_to_make.append(req)
                    branch_state[key]["queued"] += 1  # queued, not inflight yet
        entry["token"] = w_star_clean
        entry["token_raw"] = w_star_raw
        entry["probability"] = p_w_star
    # Fast exit if nothing to do: scrub & compute metrics once and save.
    if not requests_to_make:

        def _scrub_all_and_update_metrics(entries: List[Dict[str, Any]]):
            gt = (result_sink.get("problem") or {}).get("gt_answer")
            mode = (result_sink.get("metadata") or {}).get(
                "intervention_mode", args.intervention_mode
            )
            for e in entries:
                # scrub transient fields everywhere
                e.pop("top_candidates", None)
                e.pop("token_raw", None)
                # compute cf only for complete-but-missing
                brs = e.get("branches") or []
                target = args.samples_per_fork
                is_complete = bool(brs) and all(
                    sum(1 for s in (b.get("samples") or []) if not s.get("error"))
                    >= target
                    for b in brs
                )
                if is_complete and "cf_metrics" not in e:
                    m = compute_entry_cf_metrics(e, gt, mode)
                    if m is not None:
                        e["cf_metrics"] = m

        _scrub_all_and_update_metrics(token_entries)
        result_sink["token_steps"] = token_entries
        result_sink["status"] = "fork_sampling_done"
        safe_write_json(rollout_file, result_sink)
        return

    # -------------------------------
    # PHASE 2: STREAM EXECUTION LOOP
    # -------------------------------
    SAVE_INTERVAL = 300  # save every N completions
    # Keep pipeline equal to semaphore capacity to avoid backlogged tasks
    IN_FLIGHT = max(1, int(getattr(args, "concurrency", 1)))
    MAX_ATTEMPTS_PAD = max(10, args.samples_per_fork // 2)

    from collections import Counter, deque

    error_stats = Counter()

    def _trunc(s: Optional[str], n: int = 200) -> str:
        if not s:
            return ""
        s = str(s).replace("\n", " ")
        return s if len(s) <= n else (s[: n - 3] + "...")

    async def _call(req: Dict[str, Any]):
        resp = await make_api_request(
            prompt=req["api_call"]["prompt"],
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.continuation_max_tokens,
            top_logprobs=args.continuation_top_logprobs,
            logprobs=True,
            semaphore=semaphore,
            logit_bias=req["api_call"]["logit_bias"],
        )
        return req["context"], resp, req["api_call"]

    # Fair launch ordering to avoid head-of-line blocking on identical prompts
    def _round_robin_reorder(reqs: List[Dict[str, Any]]) -> deque:
        from collections import defaultdict, deque as _dq

        buckets = defaultdict(list)
        for r in reqs:
            buckets[(r["context"]["token_index"], r["context"]["branch_token"])].append(
                r
            )
        order: List[Dict[str, Any]] = []
        while True:
            progressed = False
            for k in list(buckets.keys()):
                if buckets[k]:
                    order.append(buckets[k].pop(0))
                    progressed = True
            if not progressed:
                break
        return deque(order)

    pending = _round_robin_reorder(requests_to_make)
    token_map = {entry["token_index"]: entry for entry in token_entries}
    inflight_tasks: set = set()

    def _launch(n: int):
        for _ in range(n):
            if not pending:
                break
            req = pending.popleft()
            # update per-branch counters for the branch we are launching
            t_idx = req["context"]["token_index"]
            b_tok = req["context"]["branch_token"]
            key = (t_idx, b_tok)
            st = branch_state.get(key)
            if st:
                st["queued"] = max(0, st["queued"] - 1)
                st["inflight"] += 1
                st["issued"] += 1
            inflight_tasks.add(asyncio.create_task(_call(req)))

    _launch(min(IN_FLIGHT, len(pending)))
    pbar = tqdm(
        total=len(requests_to_make),
        desc=f"Sampling p{problem_idx} ({args.intervention_mode})",
    )

    processed = 0
    dirty_entries: set = set()

    def _entry_is_complete(e: Dict[str, Any]) -> bool:
        brs = e.get("branches") or []
        if not brs:
            return False
        target = args.samples_per_fork
        return all(
            sum(1 for s in (b.get("samples") or []) if not s.get("error")) >= target
            for b in brs
        )

    def _recompute_dirty_metrics_and_cleanup():
        """For changed entries: scrub transient fields; compute cf_metrics only
        for entries that just became complete and don't have metrics yet."""
        gt = (result_sink.get("problem") or {}).get("gt_answer")
        mode = (result_sink.get("metadata") or {}).get(
            "intervention_mode", args.intervention_mode
        )
        for t_idx in list(dirty_entries):
            entry = token_map.get(t_idx)
            if not entry:
                continue
            # scrub transient planning fields on touched entries
            entry.pop("top_candidates", None)
            entry.pop("token_raw", None)
            if "cf_metrics" not in entry and _entry_is_complete(entry):
                m = compute_entry_cf_metrics(entry, gt, mode)
                if m is not None:
                    entry["cf_metrics"] = m
        dirty_entries.clear()

    def _global_scrub_all_entries():
        """Remove transient fields from ALL entries so saved JSON is clean."""
        for e in token_entries:
            e.pop("top_candidates", None)
            e.pop("token_raw", None)

    def _maybe_requeue(t_idx: int, branch_token: str, api_call: Dict[str, Any]):
        """If a branch is short on SUCCESSES, enqueue another request (no double-count)."""
        key = (t_idx, branch_token)
        st = branch_state.get(key)
        if not st:
            return
        target = st["target"]
        attempts_cap = target + MAX_ATTEMPTS_PAD
        need_more = st["successes"] + st["inflight"] + st["queued"] < target
        under_cap = st["issued"] < attempts_cap
        if need_more and under_cap:
            pending.append(
                {
                    "context": {
                        "token_index": t_idx,
                        "branch_token": branch_token,
                        "sample_index": st["attempts"] + st["inflight"] + st["queued"],
                    },
                    "api_call": api_call,  # will still include the same prompt/bias
                }
            )
            st["queued"] += 1

    while inflight_tasks:
        done, inflight_tasks = await asyncio.wait(
            inflight_tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            try:
                context, resp, api_call = await task
            except Exception as e:
                context, resp, api_call = ({}, {"error": f"Task exception: {e}"}, {})

            t_idx = context.get("token_index")
            b_tok = context.get("branch_token")
            key = (t_idx, b_tok)
            st = branch_state.get(key)
            if st:
                st["inflight"] = max(0, st["inflight"] - 1)

            entry = token_map.get(t_idx)
            if not entry:
                print(f"[p{problem_idx}] WARN orphan response: context={context}")
                continue

            branch = next(
                (b for b in entry.get("branches") or [] if b.get("token") == b_tok),
                None,
            )
            if branch is None:
                # This should rarely happen, but keep request shape if possible
                branch = {
                    "token": b_tok,
                    "samples": [],
                    "fork_prompt": api_call.get("prompt"),
                    "logit_bias": api_call.get("logit_bias"),
                }
                entry["branches"].append(branch)
            else:
                # Ensure persisted (in case of resume through older states)
                branch.setdefault("fork_prompt", api_call.get("prompt"))
                branch.setdefault("logit_bias", api_call.get("logit_bias"))

            # --- classify/store sample
            def _log_err(kind: str, msg: str = "", details: str = ""):
                error_stats[kind] += 1
                print(
                    f"[p{problem_idx}] ERROR kind={kind} t={t_idx}:{b_tok} "
                    f"idx={context.get('sample_index')} msg={_trunc(msg)} det={_trunc(details)}"
                )

            if resp.get("error"):
                sample_data = {
                    "sample_index": context.get("sample_index"),
                    "error": str(resp),
                }
                if st:
                    st["errors"] += 1
                    st["attempts"] += 1
                det = str(resp.get("details", ""))
                err_s = str(resp.get("error", ""))
                low = (det + " " + err_s).lower()
                if "429" in low or "rate" in low:
                    _log_err("rate_limit", err_s, det)
                elif "524" in low or "timeout" in low or "timed out" in low:
                    _log_err("timeout", err_s, det)
                elif "json parse" in low:
                    _log_err("json_parse", err_s, det)
                elif "5" in low and "api " in low:
                    _log_err("server_5xx", err_s, det)
                else:
                    _log_err("api_error", err_s, det)
            else:
                ch = (resp.get("choices") or [None])[0]
                if not ch:
                    sample_data = {
                        "sample_index": context.get("sample_index"),
                        "error": "No choices",
                    }
                    if st:
                        st["errors"] += 1
                        st["attempts"] += 1
                    _log_err("no_choices", "No choices array from API", "")
                else:
                    text = _clean_token_display(
                        ch.get("text")
                        or (ch.get("message", {}) or {}).get("content")
                        or ""
                    )
                    outcome = default_outcome_extractor(text)
                    logprob_sum = sum(
                        v
                        for v in (ch.get("logprobs", {}).get("token_logprobs") or [])
                        if isinstance(v, (int, float))
                    )
                    sample_data = {
                        "sample_index": context.get("sample_index"),
                        "final_answer_text": text,
                        "logprob_sum": logprob_sum,
                        "outcome": outcome,
                    }
                    if st:
                        st["successes"] += 1
                        st["attempts"] += 1

            branch.setdefault("samples", []).append(sample_data)
            processed += 1
            pbar.update(1)
            dirty_entries.add(t_idx)

            # Requeue if we still need successes
            _maybe_requeue(t_idx, b_tok, api_call)

            # progressive save every SAVE_INTERVAL completions
            if processed % SAVE_INTERVAL == 0:
                _recompute_dirty_metrics_and_cleanup()  # compute for newly-complete
                _global_scrub_all_entries()  # scrub transients globally

                # Concise progress/errors
                short = []
                for (ti, tok), s in branch_state.items():
                    if s["successes"] < s["target"] and s["successes"] > 0:
                        short.append(
                            f"t={ti}:{tok} success {s['successes']}/{s['target']} "
                            f"(errors={s['errors']}, attempts_completed={s['attempts']}, "
                            f"inflight={s['inflight']}, queued={s['queued']}, issued={s['issued']})"
                        )
                if short:
                    print(f"[p{problem_idx}] [progress] " + "; ".join(short))

                sent, recv = await _rpm_snapshot()
                print(
                    f"[RPM] last 60s — sent={sent:2d}, recv={recv:2d}, "
                    f"rps_target={args.requests_per_second:.2f}, 429=0"
                )
                result_sink["token_steps"] = token_entries
                result_sink["status"] = "fork_sampling_in_progress"
                safe_write_json(rollout_file, result_sink)

        # keep the pipeline full (bounded to semaphore capacity)
        _launch(max(0, IN_FLIGHT - len(inflight_tasks)))

    pbar.close()

    # --------------------------
    # PHASE 3: FINALIZE & SAVE
    # --------------------------
    # Final recompute for any entries that became complete in the last batch
    _recompute_dirty_metrics_and_cleanup()

    # Global scrub + compute cf_metrics only for complete-but-missing entries
    gt = (result_sink.get("problem") or {}).get("gt_answer")
    mode = (result_sink.get("metadata") or {}).get(
        "intervention_mode", args.intervention_mode
    )
    for e in token_entries:
        e.pop("top_candidates", None)
        e.pop("token_raw", None)
        brs = e.get("branches") or []
        target = args.samples_per_fork
        is_complete = bool(brs) and all(
            sum(1 for s in (b.get("samples") or []) if not s.get("error")) >= target
            for b in brs
        )
        if is_complete and "cf_metrics" not in e:
            m = compute_entry_cf_metrics(e, gt, mode)
            if m is not None:
                e["cf_metrics"] = m

    # Final shortfall print (should be empty if all branches reached target)
    final_short = []
    for (ti, tok), s in branch_state.items():
        if s["successes"] < s["target"]:
            final_short.append(
                f"t={ti}:{tok} success {s['successes']}/{s['target']} "
                f"(errors={s['errors']}, attempts_completed={s['attempts']}, "
                f"inflight={s['inflight']}, queued={s['queued']}, issued={s['issued']})"
            )
    if final_short:
        print(f"[p{problem_idx}] [final shortfall] " + "; ".join(final_short))

    result_sink["token_steps"] = token_entries
    result_sink["status"] = "fork_sampling_done"
    safe_write_json(rollout_file, result_sink)


def _final_scrub_and_metrics(
    token_entries: List[Dict[str, Any]],
    result_sink: Dict[str, Any],
    target: int,
    resampled_token_indices: Optional[Set[int]] = None,
) -> None:
    """Delete transient fields everywhere; compute cf_metrics for complete-but-missing tokens."""
    gt = (result_sink.get("problem") or {}).get("gt_answer")
    mode = (result_sink.get("metadata") or {}).get(
        "intervention_mode", getattr(args, "intervention_mode", None)
    )
    for e in token_entries:
        # Always scrub these transient planning fields at the very end
        e.pop("top_candidates", None)
        e.pop("token_raw", None)
        e.pop("fork_prompt", None)
        e.pop("logit_bias", None)

        # Compute cf_metrics iff all tracked branches are complete and metrics missing
        brs = e.get("branches") or []
        is_complete = bool(brs) and all(
            sum(1 for s in (b.get("samples") or []) if not s.get("error")) >= target
            for b in brs
        )
        if is_complete and "cf_metrics" not in e:
            m = compute_entry_cf_metrics(e, gt, mode)
            if m is not None:
                e["cf_metrics"] = m
        if resampled_token_indices:
            if is_complete and e.get("token_index") in resampled_token_indices:
                m = compute_entry_cf_metrics(e, gt, mode)
                if m is not None:
                    e["cf_metrics"] = m


# --- SSE helpers --------------------------------------------------------------


def _iter_sse_payloads(raw: Union[str, bytes]) -> List[Dict[str, Any]]:
    """
    Parse SSE frames of the form:
      data: {"choices":[{"delta":{"content":"..."}}]}
      data: [DONE]
    Return a list of JSON dict payloads (skips [DONE] & unparsable lines).
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8", errors="ignore")
        except Exception:
            return []
    out: List[Dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            out.append(json.loads(data))
        except Exception:
            # ignore unparsable fragments
            pass
    return out


def _extract_piece_from_event(ev: Dict[str, Any]) -> str:
    """
    Support common shapes:
      - chat stream delta:  choices[0].delta.content
      - completion stream:  choices[0].text
      - message content:    choices[0].message.content
    """
    choices = ev.get("choices") or []
    if not choices:
        return ""
    ch0 = choices[0] or {}
    delta = ch0.get("delta") or {}
    if isinstance(delta, dict):
        piece = delta.get("content")
        if piece:
            return piece
    piece = ch0.get("text")
    if piece:
        return piece
    msg = ch0.get("message") or {}
    if isinstance(msg, dict):
        piece = msg.get("content")
        if piece:
            return piece
    return ""


async def _collect_stream_text(stream_obj: AsyncIterator[Any]) -> str:
    """
    Collect full text from a streaming response that may yield dicts OR raw SSE strings/bytes.
    """
    parts: List[str] = []
    try:
        async for chunk in stream_obj:
            # If the SDK already yields parsed dicts:
            if isinstance(chunk, dict):
                piece = _extract_piece_from_event(chunk)
                if piece:
                    parts.append(piece)
                continue
            # Otherwise it's likely raw SSE text/bytes:
            for ev in _iter_sse_payloads(chunk):
                piece = _extract_piece_from_event(ev)
                if piece:
                    parts.append(piece)
    except Exception:
        pass
    return _clean_token_display("".join(parts))


async def stream_resample_shortfalls(
    problem_idx: int,
    token_entries: List[Dict[str, Any]],
    *,
    semaphore: asyncio.Semaphore,
    rollout_file: Path,
    result_sink: Dict[str, Any],
    aligned_path: Path,
    resample_w_streaming: bool = False,
) -> None:
    """
    Phase 4: Re-sample shortfalls *via streaming*, rebuilding per-branch fork config
    from `correct_base_completion_aligned.json` (because token_raw was scrubbed).

    - Targets only tokens that have valid alternatives.
    - For each of {baseline, __ALT_POOL__}, if the branch has < target successes,
      resample the missing successes using streaming requests.
    - After completion: scrub transients globally and compute cf_metrics for
      entries that are complete but missing metrics.
    """
    import json
    from collections import deque

    target = getattr(args, "samples_per_fork", 30)

    # --- Load prompt + base text from result_sink (authoritative) ---
    prompt_str = result_sink.get("prompt") or ""
    base_text_raw = ((result_sink.get("base") or {}).get("completion_raw")) or ""

    # --- Load aligned tokens for reconstruction ---
    aligned = {}
    with open(aligned_path, "r", encoding="utf-8") as f:
        aligned = json.load(f)

    aligned_tokens = aligned.get("tokens") or []
    by_t = {int(tok.get("t", i)): tok for i, tok in enumerate(aligned_tokens)}

    # Helper: build fork config for baseline and ALT_POOL from aligned JSON
    def _build_fork_config_for_t(t_idx: int) -> Optional[Dict[str, Dict[str, Any]]]:
        node = by_t.get(int(t_idx))
        if not node:
            return None
        text_offset = int(node.get("text_offset", 0))
        base = node.get("base") or {}
        base_raw = base.get("token_raw")
        base_tok = base.get("token")
        if not base_raw or base_tok is None:
            return None

        # Determine if there are valid alts
        valid_alts = []
        for c in node.get("candidates") or []:
            if (
                is_safe_to_fork(c.get("token_raw"))
                and float(c.get("probability") or 0.0) >= args.alternate_min_prob
                and c.get("token") != base_tok
            ):
                valid_alts.append(c)

        if not valid_alts:
            return None  # nothing to do for this t

        # Baseline branch (greedy token appended)
        base_prompt = (
            f"{prompt_str}{base_text_raw[:text_offset]}{detok_for_api(base_raw)}"
        )
        # ALT_POOL branch (no token appended, but forbid the base token id)
        tid = token_raw_to_id(base_raw)
        bias = {tid: -100} if isinstance(tid, int) else None
        alt_prompt = f"{prompt_str}{base_text_raw[:text_offset]}"

        return {
            "BASE": {
                "fork_prompt": base_prompt,
                "logit_bias": None,
                "prob": float(base.get("probability") or 0.0),
            },
            "ALT": {
                "fork_prompt": alt_prompt,
                "logit_bias": bias,
                "prob": sum(float(c.get("probability") or 0.0) for c in valid_alts),
            },
        }

    # --- Plan streaming resamples for shortfalls (baseline + ALT_POOL) ---
    resample_reqs: List[Dict[str, Any]] = []
    token_map = {e["token_index"]: e for e in token_entries}

    for entry in token_entries:
        t_idx = int(entry.get("token_index"))
        cfg = _build_fork_config_for_t(t_idx)
        if not cfg:
            continue  # no valid alts → nothing to resample

        # Find/create baseline branch
        base_tok_clean = entry.get("token")
        base_br = next(
            (b for b in entry["branches"] if b.get("token") == base_tok_clean), None
        )
        # Find/create ALT_POOL branch
        alt_br = next(
            (b for b in entry["branches"] if b.get("token") == "__ALT_POOL__"), None
        )

        # Count successes & plan missing for each branch
        def _succ(b):
            return sum(
                1
                for s in (b.get("samples") or [])
                if isinstance(s, dict) and not s.get("error")
            )

        for br in (base_br, alt_br):
            succ = _succ(br)
            missing = max(0, target - succ)
            if missing <= 0:
                continue
            print(
                f"{missing} streaming resamples needed for p{problem_idx} t={t_idx} b={br['token']}"
            )
            next_idx = len(br.get("samples") or [])
            for i in range(missing):
                resample_reqs.append(
                    {
                        "context": {
                            "token_index": t_idx,
                            "branch_token": br["token"],
                            "sample_index": next_idx + i,
                        },
                        "api_call": {
                            "prompt": cfg["ALT"]["fork_prompt"]
                            if br == alt_br
                            else cfg["BASE"]["fork_prompt"],
                            "logit_bias": cfg["ALT"]["logit_bias"]
                            if br == alt_br
                            else cfg["BASE"]["logit_bias"],
                        },
                    }
                )

    # Nothing to do → just finalize scrub/metrics/save
    if not resample_reqs:
        print(f"[p{problem_idx}] no streaming resamples needed.")
        _final_scrub_and_metrics(token_entries, result_sink, target)
        safe_write_json(
            rollout_file,
            {
                **result_sink,
                "token_steps": token_entries,
                "status": "fork_sampling_done",
            },
        )
        return
    print(f"{len(resample_reqs)} streaming resamples planned for p{problem_idx}.")
    # Mark in-progress for crash safety
    result_sink["status"] = "fork_resample_in_progress"
    safe_write_json(rollout_file, {**result_sink, "token_steps": token_entries})

    # ---- Streaming caller (avoids 524s) ----
    async def _collect_stream_text(stream_obj) -> str:
        parts: List[str] = []
        try:
            async for chunk in stream_obj:
                if not isinstance(chunk, dict):
                    continue
                ch = (chunk.get("choices") or [None])[0]
                if not ch:
                    continue
                delta = ch.get("delta") or {}
                piece = delta.get("content") or ch.get("text") or ""
                if piece:
                    parts.append(piece)
        except Exception:
            pass
        return _clean_token_display("".join(parts))

    async def _call_stream(req: Dict[str, Any]):
        ctx = req.get("context", {}) or {}
        try:
            stream = await make_api_request(
                prompt=req["api_call"]["prompt"],
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.continuation_max_tokens,
                logprobs=False,
                semaphore=semaphore,
                logit_bias=req["api_call"]["logit_bias"],
                stream=True,
            )
        except Exception as e:
            # Ensure we always return context even on errors
            return ctx, {"error": f"{type(e).__name__}: {e}"}

        # CASE A: real async iterator -> collect normally
        if hasattr(stream, "__aiter__"):
            text = await _collect_stream_text(stream)
            resp = {"choices": [{"text": text, "logprobs": {"token_logprobs": []}}]}
            return ctx, resp

        # CASE B: dict result (e.g., JSON parse error with raw SSE in 'details')
        if isinstance(stream, dict):
            # If the SDK stashed the raw SSE in 'details', try to salvage it.
            if "details" in stream:
                details = stream.get("details") or ""
                text_parts: List[str] = []
                for ev in _iter_sse_payloads(details):
                    piece = _extract_piece_from_event(ev)
                    if piece:
                        text_parts.append(piece)
                text = _clean_token_display("".join(text_parts))
                if text:
                    resp = {
                        "choices": [{"text": text, "logprobs": {"token_logprobs": []}}]
                    }
                    return ctx, resp
            # If there are pre-parsed events, support that too.
            if "events" in stream and isinstance(stream["events"], list):
                text = _clean_token_display(
                    "".join(_extract_piece_from_event(ev) for ev in stream["events"])
                )
                resp = {"choices": [{"text": text, "logprobs": {"token_logprobs": []}}]}
                return ctx, resp

            # Otherwise, bubble up the error
            err = stream.get("error") or "stream_error"
            det = stream.get("details")
            return ctx, {"error": err, "details": det}

        # CASE C: unknown type -> fall back to non-stream call with retries
        return await _call_with_retries(req)

        # --- non-stream call with retries/backoff (heals 524/timeouts) ---

    async def _call_with_retries(req: Dict[str, Any]):
        max_tries = 4
        backoff = 1.0
        for attempt in range(1, max_tries + 1):
            try:
                resp = await make_api_request(
                    prompt=req["api_call"]["prompt"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.continuation_max_tokens,
                    logprobs=False,  # keep it simple for fallback
                    semaphore=semaphore,
                    logit_bias=req["api_call"]["logit_bias"],
                    stream=False,  # IMPORTANT: same path as Phase 2 (reliable)
                )
                return req["context"], resp
            except Exception as e:
                low = str(e).lower()
                transient = any(
                    tok in low
                    for tok in ("524", "timeout", "timed out", "gateway", "rate")
                )
                if attempt < max_tries and transient:
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue
                return req["context"], {"error": f"{type(e).__name__}: {e}"}

    # ---- Launch bounded streaming resamples ----
    # resample_reqs = resample_reqs[0:20]  # test on a subset
    pending = deque(resample_reqs)
    inflight: set = set()
    RESAMPLE_IN_FLIGHT = max(1, int(getattr(args, "concurrency", 8)))

    def _launch(n: int):
        for _ in range(n):
            if not pending:
                break
            req = pending.popleft()
            if resample_w_streaming:
                inflight.add(asyncio.create_task(_call_stream(req)))
            else:
                inflight.add(asyncio.create_task(_call_with_retries(req)))

    _launch(min(RESAMPLE_IN_FLIGHT, len(pending)))
    pbar = tqdm(total=len(resample_reqs), desc=f"Streaming resample p{problem_idx}")

    while inflight:
        done, inflight = await asyncio.wait(
            inflight, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done:
            context, resp = await task
            t_idx = context.get("token_index")
            b_tok = context.get("branch_token")
            entry = token_map.get(t_idx)
            if not entry:
                pbar.update(1)
                continue

            branch = next(
                (b for b in (entry.get("branches") or []) if b.get("token") == b_tok),
                None,
            )
            if branch is None:
                branch = {"token": b_tok, "samples": []}
                entry.setdefault("branches", []).append(branch)

            if resp.get("error"):
                branch.setdefault("samples", []).append(
                    {"sample_index": context.get("sample_index"), "error": str(resp)}
                )
            else:
                ch = (resp.get("choices") or [None])[0]
                text = _clean_token_display((ch or {}).get("text") or "")
                sample_data = {
                    "sample_index": context.get("sample_index"),
                    "final_answer_text": text,
                    "logprob_sum": None,
                    "outcome": default_outcome_extractor(text),
                }
                branch.setdefault("samples", []).append(sample_data)

            pbar.update(1)

        _launch(max(0, RESAMPLE_IN_FLIGHT - len(inflight)))

    pbar.close()

    # ---- Final scrub + metrics + save ----
    # pass to final scrub/metrics only the token indices that were modified (to compute cf_metrics only where resamples happened)
    resampled_token_indices = {req["context"]["token_index"] for req in resample_reqs}
    _final_scrub_and_metrics(
        token_entries, result_sink, target, resampled_token_indices
    )
    result_sink["status"] = "fork_sampling_done"
    safe_write_json(rollout_file, {**result_sink, "token_steps": token_entries})
    print(f"[p{problem_idx}] resample completed.")


# =============================================================================
# Orchestration (one problem)
# =============================================================================


async def run_problem(problem_idx: int, *, semaphore: asyncio.Semaphore) -> None:
    out_root = (
        Path(args.output_dir)
        / args.model.replace("/", "_")
        / f"problem_{problem_idx}"
        / f"samples_{args.samples_per_fork}_topk_{args.alternate_top_k}_prob_{args.alternate_min_prob}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    rollout_file = out_root / "rollout_analysis.json"
    # aligned path is outside samples specific folder
    aligned_path = out_root.parent / "correct_base_completion_aligned.json"

    # Load sentence-level base
    ext = load_external_base_for_problem(problem_idx)
    problem = load_external_problem(problem_idx)
    gt_answer = problem.get("gt_answer")
    problem_text = problem.get("problem")
    ext_prompt = ext["prompt"]
    ext_completion_raw = ext["solution"]

    # Progressive/resumable base top-k cache
    cache = await precompute_and_cache_base_topk(
        problem_idx,
        ext_prompt,
        ext_completion_raw,
        topk=args.topk_base_completion,
        semaphore=semaphore,
        force=args.force,
        gt_answer=gt_answer,
    )

    # Build token entries from cache (trim to runtime alternate_top_k)
    fresh_entries = _build_token_entries_from_cache(
        cache, alt_topk=args.alternate_top_k
    )

    # Initialize or resume rollout
    result: Dict[str, Any] = {}
    if rollout_file.exists() and not args.force:
        try:
            with open(rollout_file, "r", encoding="utf-8") as f:
                result = json.load(f)
            print(
                f"[Resume] Found checkpoint for problem {problem_idx} with status={result.get('status')}."
            )
        except Exception:
            result = {}

    existing_steps = result.get("token_steps") or []
    merged_steps = _merge_token_steps(existing_steps, fresh_entries)

    result.setdefault("metadata", {})
    result["metadata"].update(
        {
            "model": args.model,
            "sample_temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens_continuation": args.continuation_max_tokens,
            "samples_per_fork": args.samples_per_fork,
            "alternate_top_k": args.alternate_top_k,
            "alternate_min_prob": args.alternate_min_prob,
            "intervention_mode": args.intervention_mode,
            "external": {
                "root": args.external_base_root,
                "kind": args.external_kind,
                "temp": args.external_temp,
                "top_p": args.external_top_p,
                "file": args.external_base_file,
            },
            "topk_base_cached": cache.get("topk_base_cached"),
        }
    )
    result["problem"] = {"problem_text": problem_text, "gt_answer": gt_answer}
    result["prompt"] = ext_prompt
    result["base"] = {
        "completion_raw": ext_completion_raw,
        "completion": _clean_token_display(ext_completion_raw),
        "outcome": default_outcome_extractor(_clean_token_display(ext_completion_raw)),
        "is_correct": cache.get("is_correct", False),
    }
    result["token_steps"] = merged_steps
    result["status"] = "base_ready"
    safe_write_json(rollout_file, result)

    # 2) Branch sampling (progressive saves inside sampling)
    remaining = sum(_pending_samples_for_entry(e) for e in merged_steps)
    if remaining > 0 or args.force:
        await sample_fork_branches(
            problem_idx,
            ext_prompt,
            merged_steps,
            semaphore=semaphore,
            base_completion_text_raw=ext_completion_raw,
            rollout_file=rollout_file,
            result_sink=result,
        )
        result["token_steps"] = merged_steps
        result["status"] = "fork_sampling_done"
        safe_write_json(rollout_file, result)

    # 3) Streaming resample pass (rebuild fork config from aligned JSON)
    with open(rollout_file, "r", encoding="utf-8") as f:
        rollout = json.load(f)

    entries = rollout.get("token_steps", [])
    await stream_resample_shortfalls(
        problem_idx=problem_idx,
        token_entries=entries,
        semaphore=semaphore,
        rollout_file=rollout_file,
        result_sink=rollout,
        aligned_path=aligned_path,
        resample_w_streaming=True,
    )

    # 4) Final status
    rollout["status"] = "done"
    safe_write_json(rollout_file, rollout)
    print(f"[Done] Problem {problem_idx} → {rollout_file}")


# =============================================================================
# Discovery & main
# =============================================================================


def _external_model_dir_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def _discover_problem_ids() -> List[int]:
    root = Path(args.external_base_root)
    model_dir = _external_model_dir_name(args.model)
    seg = f"temperature_{args.external_temp}_top_p_{args.external_top_p}"
    base = root / model_dir / seg / args.external_kind
    if not base.exists():
        raise RuntimeError(f"External base path not found: {base}")
    ids: List[int] = []
    for p in base.glob("problem_*"):
        try:
            ids.append(int(p.name.split("_")[-1]))
        except Exception:
            continue
    ids.sort()
    return ids


async def main() -> None:
    if args.include_problems:
        problem_ids = [
            int(x.strip())
            for x in args.include_problems.split(",")
            if x.strip().isdigit()
        ]
    else:
        problem_ids = _discover_problem_ids()

    if args.exclude_problems:
        excluded = {
            int(x.strip())
            for x in args.exclude_problems.split(",")
            if x.strip().isdigit()
        }
        problem_ids = [i for i in problem_ids if i not in excluded]

    if not problem_ids:
        print("No problems to process (check external cache path / filters).")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    # start periodic RPM logger
    stop_rpm = asyncio.Event()
    rpm_task = asyncio.create_task(_rpm_logger(stop_rpm, every_seconds=30.0))
    try:
        for pid in tqdm(problem_ids, desc="Processing problems"):
            try:
                await run_problem(pid, semaphore=semaphore)
            except Exception as e:
                out_root = (
                    Path(args.output_dir)
                    / args.model.replace("/", "_")
                    / f"problem_{pid}"
                    / f"samples_{args.samples_per_fork}_topk_{args.alternate_top_k}_prob_{args.alternate_min_prob}"
                )
                out_root.mkdir(parents=True, exist_ok=True)
                with open(out_root / "rollout_error.json", "w", encoding="utf-8") as f:
                    json.dump({"error": str(e)}, f, indent=2, ensure_ascii=False)
                print(f"[Error] Problem {pid}: {e}")
    finally:
        stop_rpm.set()
        try:
            await rpm_task
        except Exception:
            pass
        client = HTTPX_CLIENT
        if client is not None:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
