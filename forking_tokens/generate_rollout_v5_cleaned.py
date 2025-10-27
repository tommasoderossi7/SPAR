#!/usr/bin/env python3
# generate_rollout.py — external-base alignment + forced/biased token interventions (progressive + resumable)

import os
import re
import json
import math
import time
import random
import asyncio
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from argparse import ArgumentParser
from tqdm import tqdm
from transformers import AutoTokenizer

# =============================================================================
# Configuration & CLI
# =============================================================================

NOVITA_API_URL = "https://api.novita.ai/openai/v1/completions"
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY_SPAR")
if not NOVITA_API_KEY:
    raise RuntimeError("NOVITA_API_KEY_SPAR is not set in the environment.")

parser = ArgumentParser(
    description="Token-level rollouts from an external base completion (forced/biased)."
)
# Model & sampling params
parser.add_argument(
    "-m", "--model", type=str, default="deepseek/deepseek-r1-distill-qwen-14b"
)
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature for continuations.",
)
parser.add_argument(
    "-tp", "--top_p", type=float, default=0.95, help="Top-p for continuations."
)
parser.add_argument(
    "-mt",
    "--max-tokens",
    type=int,
    default=4096,
    help="Max tokens for alignment calls.",
)
parser.add_argument(
    "-cmt",
    "--continuation-max-tokens",
    type=int,
    default=15000,
    help="Max tokens for sampled rollouts.",
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
    "--intervention-mode",
    type=str,
    choices=["forced", "biased"],
    default="forced",
    help="forced: force each alt token; biased: ban base token with logit_bias and sample pooled alts.",
)
parser.add_argument(
    "-sps", "--samples-per-fork", type=int, default=30, help="Samples per branch."
)

# Alt token discovery during alignment
parser.add_argument(
    "-atk",
    "--alternate-top-k",
    type=int,
    default=10,
    help="Top-k logprobs to capture per token.",
)
parser.add_argument(
    "-amp",
    "--alternate-min-prob",
    type=float,
    default=0.05,
    help="Min alt prob to consider.",
)

# External base (sentence-level cache)
parser.add_argument(
    "--external-base-root",
    type=str,
    required=True,
    help="Root of sentence-level cache, e.g. thought_anchors/.cache/math_rollouts_hf",
)
parser.add_argument(
    "--external-kind",
    type=str,
    choices=["correct_base_solution", "incorrect_base_solution"],
    default="correct_base_solution",
)
parser.add_argument(
    "--external-temp",
    type=float,
    default=0.6,
    help="Temperature used in sentence-level run.",
)
parser.add_argument(
    "--external-top-p",
    type=float,
    default=0.95,
    help="Top-p used in sentence-level run.",
)
parser.add_argument(
    "--external-base-file",
    type=str,
    default="base_solution.json",
    help="File containing prompt+solution.",
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

# Runtime & IO
parser.add_argument(
    "-o",
    "--output-dir",
    type=str,
    default="math_rollouts",
    help="Where to write results.",
)
parser.add_argument("-cc", "--concurrency", type=int, default=10)
parser.add_argument("-s", "--seed", type=int, default=44)
parser.add_argument(
    "-f", "--force", action="store_true", help="Overwrite/ignore resume checkpoints."
)

args = parser.parse_args()
random.seed(args.seed)

# =============================================================================
# Tokenizer (HF) helpers
# =============================================================================

_TOKENIZER_CACHE: Dict[str, Any] = {}


def _guess_hf_tokenizer_name(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    elif "llama" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # Fallback: use last component as HF id if it exists locally; otherwise require explicit mapping.
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
    """Turn a display token (e.g., '▁The', 'ĠNow') back into literal text for prompts."""
    if not isinstance(token_str, str) or token_str == "":
        return None
    tok = get_tokenizer(args.model)
    tid = tok.convert_tokens_to_ids(token_str)
    if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
        return tok.decode([tid], clean_up_tokenization_spaces=False)
    return None


def count_tokens(text: str) -> int:
    """Rough token count for budgeting (no special tokens)."""
    tok = get_tokenizer(args.model)
    enc = tok(text, add_special_tokens=False)
    return len(enc.get("input_ids") or [])


# =============================================================================
# Small text utilities
# =============================================================================


def _clean_token_display(s: object) -> str:
    """Human-ish rendering for logs/files (kept minimal, no ftfy dependency)."""
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
    text = "".join(ch for ch in text if (ch.isprintable() or ch in "\n\t"))
    return re.sub(r"[ \t]+", " ", text)


def extract_boxed_answers(text: str) -> List[str]:
    """Extract \\boxed{...} answers; return list of candidates."""
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return [m.strip() for m in matches if m.strip()]


def default_outcome_extractor(text: str) -> str:
    ans = extract_boxed_answers(text)
    if ans:
        return ans[0]
    stripped = text.strip()
    return stripped if stripped else "__empty__"


def _normalize_ans(x: str) -> str:
    return re.sub(r"\s+", "", x.strip()).lower()


def check_answer(pred: str, gt: Optional[str]) -> bool:
    if not gt:
        return False
    p = _normalize_ans(pred)
    g = _normalize_ans(gt)
    # Try numeric compare with tolerance
    try:
        return abs(float(p) - float(g)) <= 1e-6
    except Exception:
        return p == g


# =============================================================================
# Novita API plumbing
# =============================================================================

HTTPX_CLIENT: Optional[httpx.AsyncClient] = None


def build_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NOVITA_API_KEY}",
        "Content-Type": "application/json",
    }


async def _get_httpx_client() -> httpx.AsyncClient:
    global HTTPX_CLIENT
    if HTTPX_CLIENT is None:
        limits = httpx.Limits(
            max_connections=args.concurrency * 2,
            max_keepalive_connections=args.concurrency,
        )
        HTTPX_CLIENT = httpx.AsyncClient(
            timeout=240.0, http2=True, limits=limits, headers=build_headers()
        )
    return HTTPX_CLIENT


async def make_api_request(
    prompt: str,
    *,
    temperature: float,
    top_p: float,
    max_tokens: int,
    top_logprobs: int = 0,
    logprobs: bool = False,
    semaphore: Optional[asyncio.Semaphore] = None,
    min_p: Optional[float] = None,
    top_k: Optional[int] = None,
    logit_bias: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Single non-streaming completion call with optional logprobs."""
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
    if min_p is not None:
        payload["min_p"] = min_p
    if top_k is not None:
        payload["top_k"] = top_k
    if logit_bias is not None:
        payload["logit_bias"] = {str(k): float(v) for k, v in logit_bias.items()}

    client = await _get_httpx_client()

    async def _post() -> httpx.Response:
        return await client.post(NOVITA_API_URL, json=payload)

    if semaphore:
        async with semaphore:
            resp = await _post()
    else:
        resp = await _post()

    if resp.status_code == 200:
        return resp.json()
    return {"error": f"API {resp.status_code}", "details": resp.text}


# =============================================================================
# Logprobs parsing
# =============================================================================


def parse_top_logprob_block(block: Any) -> List[Dict[str, Any]]:
    """Normalize various top_logprobs formats into a sorted list."""
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


def parse_logprob_entries(logprobs: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return list of {token_raw, token, logprob, probability, top_candidates, text_offset} entries."""
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
                    "text_offset": text_offsets[idx]
                    if isinstance(text_offsets, list) and idx < len(text_offsets)
                    else None,
                }
            )
        return entries

    content = logprobs.get("content")
    if isinstance(content, list):
        for idx, block in enumerate(content):
            tok_raw = ""
            lp = None
            top_block = None
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


# =============================================================================
# External base loader & path helpers
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


def _try_read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_external_base_for_problem(problem_idx: int) -> Dict[str, Any]:
    """Read external prompt + base solution; returns a dict with 'prompt', 'solution', and optional fields."""
    ext_dir = _external_problem_dir(problem_idx)
    data_path = ext_dir / args.external_base_file
    if not data_path.exists():
        raise RuntimeError(f"External base file not found: {data_path}")
    data = _try_read_json(data_path)
    if "prompt" not in data or "solution" not in data:
        raise RuntimeError(
            f"External base file missing 'prompt'/'solution': {data_path}"
        )
    return data  # may contain 'problem' and 'gt_answer' optionally


# =============================================================================
# Alignment: iterative LCP vs external base (with k==0 retry up to 20)
# =============================================================================


def safe_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _tokenize_with_offsets(text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    tok = get_tokenizer(args.model)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["input_ids"], enc["offset_mapping"]


def _ext_signature(prompt: str, completion: str) -> str:
    import hashlib

    s = f"{prompt}\n<SEP>\n{completion}\n<CFG>temp={args.external_temp}|top_p={args.external_top_p}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def is_safe_to_fork(token_str: str) -> bool:
    decoded = detok_for_api(token_str)
    if decoded is None:
        return False
    if any(
        unicodedata.category(c).startswith("C") and c not in "\n\t" for c in decoded
    ):
        return False
    return True


def _save_alignment_progress(
    sink: Dict[str, Any],
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
    sink.setdefault("base", {})
    sink["base"]["completion_raw"] = ext_completion_raw
    sink["base"]["completion"] = _clean_token_display(ext_completion_raw)
    sink["base"]["outcome"] = default_outcome_extractor(sink["base"]["completion"])

    sink["token_steps"] = token_steps
    sink["alignment"] = {
        "status": status,
        "matched_tokens": int(matched_tokens),
        "total_tokens": int(total_tokens),
        "sig": sig,
        "external": {
            "prompt": ext_prompt,
            "temperature": float(args.external_temp),
            "top_p": float(args.external_top_p),
        },
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sink["status"] = "alignment_" + status
    safe_write_json(rollout_file, sink)


async def align_to_external_base(
    problem_idx: int,
    ext_prompt: str,
    ext_completion_raw: str,
    *,
    semaphore: asyncio.Semaphore,
    rollout_file: Path,
    result_sink: Dict[str, Any],
) -> None:
    """Iteratively align the next-token distributions to the external base completion. Progressive-save after each advance."""
    sig = _ext_signature(ext_prompt, ext_completion_raw)
    ext_ids, ext_offs = _tokenize_with_offsets(ext_completion_raw)
    N = len(ext_ids)
    if N == 0:
        raise RuntimeError("External base completion is empty after tokenization.")

    token_steps: List[Dict[str, Any]] = result_sink.get("token_steps") or []
    matched = len(token_steps)
    prev = result_sink.get("alignment") or {}
    if prev:
        if prev.get("sig") != sig and not args.force:
            raise RuntimeError(
                "Existing alignment state does not match current external base (use --force to overwrite)."
            )
        matched = min(matched, int(prev.get("matched_tokens", matched)))

    # Initial checkpoint
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
        return

    align_temp = float(args.external_temp)
    align_top_p = float(args.external_top_p)

    while matched < N:
        # Current prompt is ext_prompt + already-aligned external prefix (by char offsets)
        cur_char_start = 0 if matched == 0 else ext_offs[matched][0]
        current_prompt = f"{ext_prompt}{ext_completion_raw[:cur_char_start]}"

        success = False
        last_entries = None

        # Retry up to 20 attempts if k==0
        for attempt in range(1, 21):
            resp = await make_api_request(
                current_prompt,
                temperature=align_temp,
                top_p=align_top_p,
                max_tokens=args.max_tokens,
                top_logprobs=args.alternate_top_k,
                logprobs=True,
                semaphore=semaphore,
            )
            if resp.get("error"):
                raise RuntimeError(
                    f"[Align p{problem_idx}] API error: {resp.get('error')} :: {resp.get('details')}"
                )

            choice = (resp.get("choices") or [None])[0]
            if not choice or not choice.get("logprobs"):
                print(
                    f"[Align p{problem_idx}] attempt {attempt}/20 @token {matched}: no logprobs (k=0); retrying..."
                )
                continue

            entries = parse_logprob_entries(choice.get("logprobs")) or []
            last_entries = entries

            # Compare generated token ids to external base continuation
            gen_ids: List[int] = [
                token_raw_to_id(e.get("token_raw")) or -1 for e in entries
            ]
            rem = N - matched
            M = min(rem, len(gen_ids))
            k = 0
            while k < M and gen_ids[k] == ext_ids[matched + k]:
                k += 1

            if k > 0:
                success = True
                print(
                    f"[Align p{problem_idx}] advanced by k={k} tokens (attempt {attempt})."
                )
                break
            else:
                print(
                    f"[Align p{problem_idx}] attempt {attempt}/20 @token {matched}: k=0; retrying..."
                )

        if not success:
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
                f"Alignment stalled at token {matched} after 20 attempts."
            )

        # Commit the first k aligned entries & checkpoint
        for j in range(k):
            e = last_entries[j]
            abs_t = matched + j
            token_steps.append(
                {
                    "token_index": abs_t,
                    "token_raw": e.get("token_raw"),
                    "token": e.get("token"),
                    "logprob": e.get("logprob"),
                    "probability": math.exp(e["logprob"])
                    if isinstance(e.get("logprob"), (int, float))
                    else None,
                    "top_candidates": e.get("top_candidates"),
                    "text_offset": int(ext_offs[abs_t][0]),
                }
            )

        matched += k
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


# =============================================================================
# Sampling branches (forced/biased) from aligned base
# =============================================================================


def _existing_branch_map(entry: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for br in entry.get("branches", []) or []:
        tok = br.get("token")
        if isinstance(tok, str):
            out[tok] = br
    return out


def _count_valid_alternatives_for_entry(entry: Dict[str, Any]) -> int:
    cnt = 0
    for cand in entry.get("top_candidates", []) or []:
        cand_raw = cand.get("token_raw")
        cand_prob = cand.get("probability")
        if (
            (not cand_raw)
            or (cand_raw == entry.get("token_raw"))
            or (not is_safe_to_fork(cand_raw))
        ):
            continue
        if (
            not isinstance(cand_prob, (int, float))
            or cand_prob < args.alternate_min_prob
        ):
            continue
        cnt += 1
    return cnt


def _pending_samples_for_entry(entry: Dict[str, Any]) -> int:
    need = 0
    existing = _existing_branch_map(entry)
    m_t = _count_valid_alternatives_for_entry(entry)
    if m_t > 0:
        # Baseline (greedy) samples
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
            if (
                not isinstance(cand_prob, (int, float))
                or cand_prob < args.alternate_min_prob
            ):
                continue
            prev = existing.get(cand_clean, {})
            have = len(prev.get("samples", []) or [])
            if have < args.samples_per_fork:
                need += args.samples_per_fork - have
    else:
        pool = existing.get("__ALT_POOL__", {})
        have = len(pool.get("samples", []) or [])
        if have < args.samples_per_fork:
            need += args.samples_per_fork - have

    return need


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
    return float(p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q)))


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

        # Collect valid alternates for this position
        valid_alts: List[Tuple[str, Dict[str, float]]] = []
        for cand in entry.get("top_candidates", []) or []:
            cand_raw = cand.get("token_raw")
            cand_prob = cand.get("probability")
            if (
                (not cand_raw)
                or (cand_raw == entry.get("token_raw"))
                or (not is_safe_to_fork(cand_raw))
            ):
                continue
            if (
                not isinstance(cand_prob, (int, float))
                or cand_prob < args.alternate_min_prob
            ):
                continue
            valid_alts.append(
                (cand_raw, {"token": cand.get("token"), "probability": cand_prob})
            )

        if len(valid_alts) == 0:
            continue

        t_offset = entry.get("text_offset") or 0
        w_star_raw = entry.get("token_raw")
        w_star_clean = entry.get("token")
        p_w_star = float(entry.get("probability") or 0.0)

        # ---------- Baseline branch (greedy token) ----------
        greedy_text = detok_for_api(w_star_raw) if isinstance(w_star_raw, str) else None
        if greedy_text:
            fork_prompt_w_star = (
                f"{prompt}{base_completion_text_raw[:t_offset]}{greedy_text}"
            )
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
                    )
                    for _ in range(need)
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for add_idx, r in enumerate(responses):
                    sample_idx = have + add_idx
                    if (
                        isinstance(r, Exception)
                        or not isinstance(r, dict)
                        or r.get("error")
                    ):
                        br_star["samples"].append(
                            {"sample_index": sample_idx, "error": str(r)}
                        )
                        continue
                    ch = (r.get("choices") or [None])[0]
                    if not ch:
                        br_star["samples"].append(
                            {"sample_index": sample_idx, "error": "No choices"}
                        )
                        continue
                    _, text = (
                        _clean_token_display(ch.get("text") or "") or "",
                        _clean_token_display(ch.get("text") or ""),
                    )
                    if not text and isinstance(ch.get("message"), dict):
                        content = ch["message"].get("content") or ""
                        text = _clean_token_display(content)
                    outcome = default_outcome_extractor(text)
                    logprob_sum = sum(
                        v
                        for v in (ch.get("logprobs", {}).get("token_logprobs") or [])
                        if isinstance(v, (int, float))
                    )
                    br_star["samples"].append(
                        {
                            "sample_index": sample_idx,
                            "final_answer_text": text,
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
            # One branch per alternate, forced token
            for cand_raw, compact in valid_alts:
                alt_text = detok_for_api(cand_raw)
                if alt_text is None:
                    continue
                fork_prompt = f"{prompt}{base_completion_text_raw[:t_offset]}{alt_text}"
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
                    )
                    for _ in range(need)
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for add_idx, r in enumerate(responses):
                    sample_idx = have + add_idx
                    if (
                        isinstance(r, Exception)
                        or not isinstance(r, dict)
                        or r.get("error")
                    ):
                        br["samples"].append(
                            {"sample_index": sample_idx, "error": str(r)}
                        )
                        continue
                    ch = (r.get("choices") or [None])[0]
                    if not ch:
                        br["samples"].append(
                            {"sample_index": sample_idx, "error": "No choices"}
                        )
                        continue
                    _, text = (
                        _clean_token_display(ch.get("text") or "") or "",
                        _clean_token_display(ch.get("text") or ""),
                    )
                    if not text and isinstance(ch.get("message"), dict):
                        content = ch["message"].get("content") or ""
                        text = _clean_token_display(content)
                    outcome = default_outcome_extractor(text)
                    logprob_sum = sum(
                        v
                        for v in (ch.get("logprobs", {}).get("token_logprobs") or [])
                        if isinstance(v, (int, float))
                    )
                    br["samples"].append(
                        {
                            "sample_index": sample_idx,
                            "final_answer_text": text,
                            "logprob_sum": logprob_sum,
                            "outcome": outcome,
                        }
                    )
                result_sink["token_steps"] = token_entries
                result_sink["status"] = "fork_sampling_in_progress"
                safe_write_json(rollout_file, result_sink)
                pbar.update(need)

        else:
            # Biased: one pooled branch; ban the greedy token with logit_bias
            fork_prompt = f"{prompt}{base_completion_text_raw[:t_offset]}"
            alt_prob_sum = float(
                sum(float(c["probability"] or 0.0) for _, c in valid_alts)
            )
            pool = existing_map.get("__ALT_POOL__")
            if pool is None:
                pool = {
                    "token": "__ALT_POOL__",
                    "probability": alt_prob_sum,
                    "samples": [],
                    "note": "logit_bias ban base token",
                }
                entry["branches"].append(pool)
                existing_map["__ALT_POOL__"] = pool
            have = len(pool.get("samples", []) or [])
            need = max(0, args.samples_per_fork - have)
            if need > 0:
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
                        logit_bias=bias,
                    )
                    for _ in range(need)
                ]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                for add_idx, r in enumerate(responses):
                    sample_idx = have + add_idx
                    if (
                        isinstance(r, Exception)
                        or not isinstance(r, dict)
                        or r.get("error")
                    ):
                        pool["samples"].append(
                            {"sample_index": sample_idx, "error": str(r)}
                        )
                        continue
                    ch = (r.get("choices") or [None])[0]
                    if not ch:
                        pool["samples"].append(
                            {"sample_index": sample_idx, "error": "No choices"}
                        )
                        continue
                    _, text = (
                        _clean_token_display(ch.get("text") or "") or "",
                        _clean_token_display(ch.get("text") or ""),
                    )
                    if not text and isinstance(ch.get("message"), dict):
                        content = ch["message"].get("content") or ""
                        text = _clean_token_display(content)
                    outcome = default_outcome_extractor(text)
                    logprob_sum = sum(
                        v
                        for v in (ch.get("logprobs", {}).get("token_logprobs") or [])
                        if isinstance(v, (int, float))
                    )
                    pool["samples"].append(
                        {
                            "sample_index": sample_idx,
                            "final_answer_text": text,
                            "logprob_sum": logprob_sum,
                            "outcome": outcome,
                        }
                    )
                result_sink["token_steps"] = token_entries
                result_sink["status"] = "fork_sampling_in_progress"
                safe_write_json(rollout_file, result_sink)
                pbar.update(need)

        # Optional: free heavy fields *after* this entry is fully processed
        entry.pop("logprob", None)
        # Keep token_raw/top_candidates for debugging; comment next two lines if you prefer to retain them:
        # entry.pop("token_raw", None)
        # entry.pop("top_candidates", None)

    pbar.close()


def compute_token_cf_metrics_inplace(result_obj: Dict[str, Any]) -> None:
    steps = result_obj.get("token_steps") or []
    gt = ((result_obj.get("problem") or {}).get("gt_answer")) or None
    mode = (result_obj.get("metadata") or {}).get("intervention_mode", "forced")

    for entry in steps:
        br_map = {br.get("token"): br for br in (entry.get("branches") or [])}
        w_star = entry.get("token")
        br_star = br_map.get(w_star)
        if not br_star or not br_star.get("samples"):
            continue

        base_dist = _empirical_outcome_dist(br_star.get("samples"))

        if mode == "forced":
            # Mix across alternates only (renormalized on alt set)
            alts: List[Tuple[float, Dict[str, float]]] = []
            for tok, br in br_map.items():
                if tok == w_star or tok == "__ALT_POOL__":
                    continue
                pw = br.get("probability")
                if not isinstance(pw, (int, float)) or pw <= 0:
                    continue
                alts.append((float(pw), _empirical_outcome_dist(br.get("samples"))))
            s = sum(w for w, _ in alts)
            if s > 0:
                alts = [(w / s, d) for (w, d) in alts]
            cf_dist = mix_dists(alts) if alts else {}
        else:
            pool = br_map.get("__ALT_POOL__")
            cf_dist = _empirical_outcome_dist(pool.get("samples")) if pool else {}

        p_true_base = prob_true_from_dist(base_dist, gt)
        p_true_cf = prob_true_from_dist(cf_dist, gt)
        entry["cf_metrics"] = {
            "mode": mode,
            "baseline_dist": base_dist,
            "cf_dist": cf_dist,
            "p_true_baseline": float(p_true_base),
            "p_true_cf": float(p_true_cf),
            "delta_acc": float(p_true_cf - p_true_base),
            "kl_true": float(kl_bernoulli(p_true_cf, p_true_base)),
            "kl_full": float(kl_full_dist(cf_dist, base_dist)),
        }


# =============================================================================
# Orchestration (one problem)
# =============================================================================


async def run_problem(problem_idx: int, *, semaphore: asyncio.Semaphore) -> None:
    # Resolve IO paths
    out_root = (
        Path(args.output_dir)
        / args.model.replace("/", "_")
        / f"samples_{args.samples_per_fork}_topk_{args.alternate_top_k}_prob_{args.alternate_min_prob}"
        / f"problem_{problem_idx}"
    )
    out_root.mkdir(parents=True, exist_ok=True)
    rollout_file = out_root / "rollout_analysis.json"

    # Load external base
    ext = load_external_base_for_problem(problem_idx)
    ext_prompt = ext["prompt"]
    ext_completion_raw = ext["solution"]
    problem_text = ext.get("problem")
    gt_answer = ext.get("gt_answer")

    # Build or resume result
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

    # Minimal metadata/problem record
    result.setdefault("metadata", {})
    result["metadata"].update(
        {
            "model": args.model,
            "sample_temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens_alignment": args.max_tokens,
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
        }
    )
    result.setdefault("problem", {"problem_text": problem_text, "gt_answer": gt_answer})
    result["prompt"] = ext_prompt  # Pipeline uses the external prompt

    safe_write_json(rollout_file, result)

    # 1) Alignment (progressive + resumable)
    aligned_done = (result.get("alignment") or {}).get("status") == "done"
    if not aligned_done or ("token_steps" not in result) or args.force:
        await align_to_external_base(
            problem_idx,
            ext_prompt,
            ext_completion_raw,
            semaphore=semaphore,
            rollout_file=rollout_file,
            result_sink=result,
        )

    # 2) Branch sampling (progressive + resumable)
    token_entries = result.get("token_steps") or []
    base_completion_text_raw: str = (result.get("base") or {}).get(
        "completion_raw"
    ) or ext_completion_raw
    remaining = sum(_pending_samples_for_entry(e) for e in token_entries)
    if remaining > 0 or args.force:
        await sample_fork_branches(
            problem_idx,
            ext_prompt,
            token_entries,
            semaphore=semaphore,
            base_completion_text_raw=base_completion_text_raw,
            rollout_file=rollout_file,
            result_sink=result,
        )
        result["token_steps"] = token_entries
        result["status"] = "fork_sampling_done"
        safe_write_json(rollout_file, result)

    # 3) Metrics
    compute_token_cf_metrics_inplace(result)
    result["status"] = "done"
    safe_write_json(rollout_file, result)
    print(f"[Done] Problem {problem_idx} → {rollout_file}")


# =============================================================================
# Discovery & main
# =============================================================================


def _discover_problem_ids() -> List[int]:
    """Auto-discover problems from the external cache if --include-problems is not provided."""
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
    # Problem id selection
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
    try:
        for pid in tqdm(problem_ids, desc="Processing problems"):
            try:
                await run_problem(pid, semaphore=semaphore)
            except Exception as e:
                # Write a small error file next to the rollout to aid triage
                out_root = (
                    Path(args.output_dir)
                    / args.model.replace("/", "_")
                    / f"samples_{args.samples_per_fork}_topk_{args.alternate_top_k}_prob_{args.alternate_min_prob}"
                    / f"problem_{pid}"
                )
                out_root.mkdir(parents=True, exist_ok=True)
                with open(out_root / "rollout_error.json", "w", encoding="utf-8") as f:
                    json.dump({"error": str(e)}, f, indent=2, ensure_ascii=False)
                print(f"[Error] Problem {pid}: {e}")
    finally:
        client = HTTPX_CLIENT
        if client is not None:
            await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
