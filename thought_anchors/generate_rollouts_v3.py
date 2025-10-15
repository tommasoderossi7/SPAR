# generate_rollouts_novita.py — Novita-only, with pre-run estimates + progressive actuals (central + per-problem)

import os
import json
import random
import math
import asyncio
import httpx
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from tqdm import tqdm

# ---- project utils (keep your originals) ----
from utils.utils import (
    extract_boxed_answers,
    check_answer,
    split_solution_into_chunks,
    load_math_problems,
)

# ---- token counting via HF tokenizer (for estimates) ----
from transformers import AutoTokenizer

# --------------- ENV / CONSTANTS -----------------
load_dotenv()
NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
if not NOVITA_API_KEY:
    raise ValueError("NOVITA_API_KEY not found in environment variables")

NOVITA_API_URL = (
    "https://api.novita.ai/v3/openai/completions"  # OpenAI-compatible completions
)

# Pricing per your instruction
PRICES_PER_MTOK = {
    "deepseek/deepseek-r1-distill-qwen-14b": {"input": 0.15, "output": 0.15}
}
DEFAULT_PRICE = {"input": 0.15, "output": 0.15}  # fallback if model not in table

# --------- CLI ----------
import argparse

parser = argparse.ArgumentParser(
    description="Generate CoT rollouts (Novita-only) with cost estimates + progressive actuals"
)
parser.add_argument(
    "-m", "--model", type=str, default="deepseek/deepseek-r1-distill-qwen-14b"
)
parser.add_argument(
    "-b",
    "--base_solution_type",
    type=str,
    default="correct",
    choices=["correct", "incorrect"],
)
parser.add_argument(
    "-r",
    "--rollout_type",
    type=str,
    default="default",
    choices=["default", "forced_answer"],
)
parser.add_argument("-o", "--output_dir", type=str, default="math_rollouts")
parser.add_argument("-np", "--num_problems", type=int, default=100)
parser.add_argument("-nr", "--num_rollouts", type=int, default=100)
parser.add_argument("-t", "--temperature", type=float, default=0.6)
parser.add_argument("-tp", "--top_p", type=float, default=0.95)
parser.add_argument("-mt", "--max_tokens", type=int, default=16384)
parser.add_argument("-mc", "--max_chunks", type=int, default=275)
parser.add_argument("-s", "--seed", type=int, default=44)
parser.add_argument("-f", "--force", action="store_true")
parser.add_argument("-ep", "--exclude_problems", type=str, default=None)
parser.add_argument("-ip", "--include_problems", type=str, default=None)
parser.add_argument("-ic", "--include_chunks", type=str, default=None)
parser.add_argument("-ty", "--type", type=str, default=None)
parser.add_argument("-l", "--level", type=str, default="Level 5")
parser.add_argument(
    "-sp", "--split", type=str, default="train", choices=["train", "test"]
)
parser.add_argument(
    "--problem-substring",
    type=str,
    default=None,
    help="Pick the single problem whose text contains this substring",
)
parser.add_argument("-fp", "--frequency_penalty", type=float, default=None)
parser.add_argument("-pp", "--presence_penalty", type=float, default=None)
parser.add_argument("-rp", "--repetition_penalty", type=float, default=None)
parser.add_argument("-tk", "--top_k", type=int, default=None)
parser.add_argument("-mp", "--min_p", type=float, default=None)
parser.add_argument("-mr", "--max_retries", type=int, default=2)
parser.add_argument(
    "-cc", "--concurrency", type=int, default=8, help="Max in-flight rollout requests"
)
parser.add_argument("-os", "--output_suffix", type=str, default=None)
args = parser.parse_args()

# --------------- OUTPUT ROOT -----------------
base_output_dir = (
    Path(args.output_dir)
    / args.model.split("/")[-1]
    / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
)
if args.rollout_type == "forced_answer":
    output_dir = (
        base_output_dir
        / f"{args.base_solution_type}_base_solution_{args.rollout_type}_{args.output_suffix}"
        if args.output_suffix
        else base_output_dir
        / f"{args.base_solution_type}_base_solution_{args.rollout_type}"
    )
else:
    output_dir = (
        base_output_dir
        / f"{args.base_solution_type}_base_solution_{args.output_suffix}"
        if args.output_suffix
        else base_output_dir / f"{args.base_solution_type}_base_solution"
    )
output_dir.mkdir(exist_ok=True, parents=True)

CENTRAL_COSTS_FILE = (
    output_dir / "central_costs.json"
)  # single central file for estimates + progressive actuals

# --------------- SEED -----------------
random.seed(args.seed)
np.random.seed(args.seed)

# --------------- TOKENIZER (for estimates) ---------------
_TOKENIZER_CACHE: Dict[str, Any] = {}


def _guess_hf_tokenizer_name(model_name: str) -> str:
    name = model_name.lower()
    if "qwen" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    if "llama" in name:
        return "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # default to qwen variant
    return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


def get_tokenizer(model_name: str):
    key = _guess_hf_tokenizer_name(model_name)
    tok = _TOKENIZER_CACHE.get(key)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(key, use_fast=True)
        _TOKENIZER_CACHE[key] = tok
    return tok


def count_tokens(text: str) -> int:
    tok = get_tokenizer(args.model)
    enc = tok(text, add_special_tokens=False, truncation=False)
    return len(enc.get("input_ids") or [])


# --------------- PRICING HELPERS -----------------
def _prices_for(model: str) -> Dict[str, float]:
    return PRICES_PER_MTOK.get(model, DEFAULT_PRICE)


USD_PER_TOKEN_INPUT = _prices_for(args.model)["input"] / 1_000_000.0
USD_PER_TOKEN_OUTPUT = _prices_for(args.model)["output"] / 1_000_000.0


def tokens_to_cost(input_tokens: int, output_tokens: int) -> Dict[str, float]:
    ic = input_tokens * USD_PER_TOKEN_INPUT
    oc = output_tokens * USD_PER_TOKEN_OUTPUT
    return {
        "input_cost_usd": float(ic),
        "output_cost_usd": float(oc),
        "total_cost_usd": float(ic + oc),
    }


# --------------- FILE IO HELPERS -----------------
def safe_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_or_create(path: Path, updater) -> None:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    updater(data)
    safe_write_json(path, data)


# --------------- NOVITA REQUEST -----------------
async def make_api_request(
    prompt: str, temperature: float, top_p: float, max_tokens: int
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {NOVITA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.min_p is not None:
        payload["min_p"] = args.min_p

    max_retries = args.max_retries
    base_delay = 2.0

    async def _send(client):
        return await client.post(
            NOVITA_API_URL, headers=headers, json=payload, timeout=240
        )

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                resp = await _send(client)
            if resp.status_code == 200:
                data = resp.json()
                choice = data["choices"][0]
                result = {
                    "text": choice.get("text", ""),
                    "finish_reason": choice.get("finish_reason", ""),
                    "usage": data.get("usage", {}) or {},
                }
                return result
            if resp.status_code in (429, 500) and attempt < max_retries - 1:
                jitter = random.uniform(0.5, 1.5)
                await asyncio.sleep(base_delay * (2**attempt) * jitter)
                continue
            return {"error": f"API error: {resp.status_code}", "details": resp.text}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {e}"}
            await asyncio.sleep(base_delay * (2**attempt))
    return {"error": "All API request attempts failed"}


# --------------- ESTIMATION LOGIC -----------------
PROMPT_PREFIX = (
    "Solve this math problem step by step. You MUST put your final answer in \\boxed{}."
)


def build_base_prompt(problem_text: str) -> str:
    return f"{PROMPT_PREFIX} Problem: {problem_text} Solution: \n<think>\n"


def build_rollout_prompt(
    problem_text: str, prefix_without_chunk: str, rollout_type: str
) -> str:
    p = f"{PROMPT_PREFIX} Problem: {problem_text} Solution: \n<think>\n{prefix_without_chunk}"
    if rollout_type == "forced_answer":
        p += "\n</think>\n\nTherefore, the final answers is \\boxed{"
    return p


def estimate_problem_rollouts_tokens(
    problem: Dict[str, Any],
    base_solution_text: str,
    chunks: List[str],
    include_chunk_ids: Optional[List[int]],
) -> Tuple[int, int, int]:
    """
    Return (est_in_tokens, est_out_tokens, est_num_requests) for all rollouts on this problem.
    Output tokens per rollout ≈ tokens from current chunk to end of the solution (using base solution as proxy).
    Input tokens per rollout = tokens of rollout prompt (prefix without that chunk).
    """
    problem_text = problem["problem"]
    # solution body inside <think>...</think> (if any)
    if "<think>" in base_solution_text:
        sol = base_solution_text.split("<think>", 1)[1]
        sol = sol.split("</think>", 1)[0] if "</think>" in sol else sol
    else:
        sol = base_solution_text

    # cumulative prefixes (exactly how original script does)
    cumulative = []
    cur = ""
    for ch in chunks:
        cur = (cur + " " + ch).strip() if cur else ch
        cumulative.append(cur)

    total_in = 0
    total_out = 0
    reqs = 0

    for idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative)):
        if include_chunk_ids is not None and idx not in include_chunk_ids:
            continue
        # prefix_without_chunk = remove this chunk from the prefix
        prefix_wo = full_prefix.replace(chunk, "").strip()
        rollout_prompt = build_rollout_prompt(
            problem_text, prefix_wo, args.rollout_type
        )
        in_tokens = count_tokens(rollout_prompt)

        # estimate output tokens as tokens of suffix from current chunk to end, using base solution
        suffix_text = " ".join(chunks[idx:])  # chunk_i .. end
        out_tokens = count_tokens(suffix_text)

        total_in += in_tokens * args.num_rollouts
        total_out += out_tokens * args.num_rollouts
        reqs += args.num_rollouts

    return total_in, total_out, reqs


def write_or_update_central_estimates(estimates_block: Dict[str, Any]) -> None:
    def _updater(data):
        data.setdefault(
            "run",
            {
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "num_rollouts": args.num_rollouts,
                "rollout_type": args.rollout_type,
                "pricing_usd_per_mtok": _prices_for(args.model),
            },
        )
        data["estimates"] = estimates_block
        # ensure "actuals" exists
        data.setdefault(
            "actuals",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            },
        )

    append_or_create(CENTRAL_COSTS_FILE, _updater)


def bump_central_actuals(
    delta_input: int, delta_output: int, delta_requests: int = 1
) -> None:
    def _updater(data):
        act = data.setdefault(
            "actuals",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            },
        )
        act["input_tokens"] += int(delta_input)
        act["output_tokens"] += int(delta_output)
        act["requests"] += int(delta_requests)
        costs = tokens_to_cost(act["input_tokens"], act["output_tokens"])
        act.update(costs)

    append_or_create(CENTRAL_COSTS_FILE, _updater)


# --------------- BASE SOLUTION -----------------
async def generate_base_solution(
    problem: Dict[str, Any], temperature: float = 0.6
) -> Dict[str, Any]:
    prompt = build_base_prompt(problem["problem"])
    response = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
    if "error" in response:
        return {
            "prompt": prompt,
            "solution": f"Error: {response['error']}",
            "error": response["error"],
        }

    sol_text = response["text"]
    answers = extract_boxed_answers(sol_text)
    ans = answers[0] if answers else ""
    is_correct = False
    if problem.get("gt_answer") and ans:
        is_correct = check_answer(ans, problem["gt_answer"])

    # progressive actuals (central + per-problem later)
    u = response.get("usage") or {}
    bump_central_actuals(
        int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0), 1
    )

    return {
        "prompt": prompt,
        "solution": sol_text,
        "full_cot": prompt + sol_text,
        "answer": ans,
        "is_correct": is_correct,
        "usage": u,
    }


# --------------- ROLLOUT -----------------
async def generate_rollout(
    problem: Dict[str, Any],
    chunk_text: str,
    full_prefix: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    prefix_without_chunk = full_prefix.replace(chunk_text, "").strip()
    prompt = build_rollout_prompt(
        problem["problem"], prefix_without_chunk, args.rollout_type
    )

    async with semaphore:
        resp = await make_api_request(
            prompt, args.temperature, args.top_p, args.max_tokens
        )

    if "error" in resp:
        return {
            "chunk_removed": chunk_text,
            "prefix_without_chunk": prefix_without_chunk,
            "error": resp["error"],
        }

    rollout_text = resp["text"]
    chunk_resampled = (
        split_solution_into_chunks(rollout_text)[0] if rollout_text else ""
    )

    extracted_answers = extract_boxed_answers(
        (prompt + rollout_text)
        if args.rollout_type == "forced_answer"
        else rollout_text
    )
    answer = extracted_answers[0] if extracted_answers else ""
    is_correct = False
    if problem.get("gt_answer") and answer:
        is_correct = check_answer(answer, problem["gt_answer"])

    # progressive actuals
    u = resp.get("usage") or {}
    bump_central_actuals(
        int(u.get("prompt_tokens", 0) or 0), int(u.get("completion_tokens", 0) or 0), 1
    )

    return {
        "chunk_removed": chunk_text,
        "prefix_without_chunk": prefix_without_chunk,
        "chunk_resampled": chunk_resampled,
        "rollout": rollout_text,
        "full_cot": f"{prompt}{rollout_text}",
        "answer": answer,
        "is_correct": is_correct,
        "usage": u,
    }


# --------------- PROBLEM PIPELINE -----------------
async def process_problem(
    problem_idx: int, problem: Dict[str, Any], semaphore: asyncio.Semaphore
) -> None:
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)

    # Save the problem once
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(problem, f, indent=2)

    # Base solution (load or create)
    base_solution_file = problem_dir / "base_solution.json"
    base_solution = None
    if base_solution_file.exists() and not args.force:
        with open(base_solution_file, "r", encoding="utf-8") as f:
            base_solution = json.load(f)

            # Recompute correctness if requested (always compute here to keep simple)
            extracted = extract_boxed_answers(base_solution.get("solution", ""))
            ans = extracted[0] if extracted else ""
            is_corr = False
            if problem.get("gt_answer") and ans:
                is_corr = check_answer(ans, problem["gt_answer"])
            base_solution["answer"] = ans
            base_solution["is_correct"] = is_corr
            with open(base_solution_file, "w", encoding="utf-8") as f2:
                json.dump(base_solution, f2, indent=2)

    if base_solution is None:
        base_solution = await generate_base_solution(problem, args.temperature)
        # enforce base correctness/incorrectness per flag
        if args.base_solution_type == "correct" and not base_solution.get("is_correct"):
            # retry by tail recursion simple; in practice you might prefer a loop
            return await process_problem(problem_idx, problem, semaphore)
        if args.base_solution_type == "incorrect" and base_solution.get("is_correct"):
            return await process_problem(problem_idx, problem, semaphore)
        with open(base_solution_file, "w", encoding="utf-8") as f:
            json.dump(base_solution, f, indent=2)

    # Determine solution_text for chunking
    source_text = base_solution["full_cot"]
    if "<think>" in source_text:
        solution_text = source_text.split("<think>", 1)[1]
        solution_text = (
            solution_text.split("</think>", 1)[0]
            if "</think>" in solution_text
            else solution_text
        )
    else:
        solution_text = source_text

    # Chunking (load or compute)
    chunks_file = problem_dir / "chunks.json"
    if not chunks_file.exists() or args.force:
        chunks = split_solution_into_chunks(solution_text)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_text": source_text,
                    "solution_text": solution_text,
                    "chunks": chunks,
                },
                f,
                indent=2,
            )
    else:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)["chunks"]

    if len(chunks) > args.max_chunks:
        print(
            f"Problem {problem_idx}: Too many chunks ({len(chunks)} > {args.max_chunks}). Skipping rollouts."
        )
        return

    # ---- Per-problem estimate & write central estimates BEFORE any rollout ----
    include_chunk_ids = None
    if args.include_chunks:
        include_chunk_ids = [
            int(x.strip())
            for x in args.include_chunks.split(",")
            if x.strip().isdigit()
        ]
    est_in, est_out, est_reqs = estimate_problem_rollouts_tokens(
        problem, source_text, chunks, include_chunk_ids
    )

    # plus base solution actuals already done (we don't estimate that here)
    est_block = {
        "problems": {
            str(problem_idx): {
                "estimated_rollouts_input_tokens": int(est_in),
                "estimated_rollouts_output_tokens": int(est_out),
                "estimated_rollouts_requests": int(est_reqs),
                **tokens_to_cost(est_in, est_out),
            }
        },
    }

    # merge into central (accumulate across problems)
    def _merge_estimates(data):
        cur = data.setdefault(
            "estimates",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "per_problem": {},
            },
        )
        cur["input_tokens"] += est_block["problems"][str(problem_idx)][
            "estimated_rollouts_input_tokens"
        ]
        cur["output_tokens"] += est_block["problems"][str(problem_idx)][
            "estimated_rollouts_output_tokens"
        ]
        cur["requests"] += est_block["problems"][str(problem_idx)][
            "estimated_rollouts_requests"
        ]
        cur["per_problem"][str(problem_idx)] = est_block["problems"][str(problem_idx)]
        costs = tokens_to_cost(cur["input_tokens"], cur["output_tokens"])
        cur.update(costs)
        data.setdefault(
            "run",
            {
                "model": args.model,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "num_rollouts": args.num_rollouts,
                "rollout_type": args.rollout_type,
                "pricing_usd_per_mtok": _prices_for(args.model),
            },
        )
        data.setdefault(
            "actuals",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "requests": 0,
                "input_cost_usd": 0.0,
                "output_cost_usd": 0.0,
                "total_cost_usd": 0.0,
            },
        )

    append_or_create(CENTRAL_COSTS_FILE, _merge_estimates)

    # also write per-problem usage tracker (progressive actuals)
    per_problem_usage_file = problem_dir / "usage_costs.json"
    if not per_problem_usage_file.exists() or args.force:
        safe_write_json(
            per_problem_usage_file,
            {
                "progress_actuals": {
                    "input_tokens": int(
                        base_solution.get("usage", {}).get("prompt_tokens", 0) or 0
                    ),
                    "output_tokens": int(
                        base_solution.get("usage", {}).get("completion_tokens", 0) or 0
                    ),
                    "requests": 1,
                    **tokens_to_cost(
                        int(
                            base_solution.get("usage", {}).get("prompt_tokens", 0) or 0
                        ),
                        int(
                            base_solution.get("usage", {}).get("completion_tokens", 0)
                            or 0
                        ),
                    ),
                },
                "estimated_rollouts": est_block["problems"][str(problem_idx)],
            },
        )

    # Build cumulative prefixes (as in original)
    cumulative_chunks = []
    cur = ""
    for ch in chunks:
        cur = (cur + " " + ch).strip() if cur else ch
        cumulative_chunks.append(cur)

    # ---- Rollout generation (parallel with semaphore) ----
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        if include_chunk_ids is not None and chunk_idx not in include_chunk_ids:
            continue

        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)
        solutions_file = chunk_dir / "solutions.json"
        existing_solutions: List[Dict[str, Any]] = []
        valid_existing_solutions: List[Dict[str, Any]] = []

        if solutions_file.exists() and not args.force:
            with open(solutions_file, "r", encoding="utf-8") as f:
                existing_solutions = json.load(f)
            # sanity re-check correctness (optional)
            valid_existing_solutions = [
                s for s in existing_solutions if "error" not in s and "rollout" in s
            ]

        need = args.num_rollouts - len(valid_existing_solutions)
        if need <= 0:
            continue

        # fan out with concurrency control
        sem = semaphore
        tasks = [
            generate_rollout(problem, chunk, full_prefix, sem) for _ in range(need)
        ]
        new_solutions = await asyncio.gather(*tasks)

        all_solutions = existing_solutions + new_solutions
        with open(solutions_file, "w", encoding="utf-8") as f:
            json.dump(all_solutions, f, indent=2)

        # bump per-problem progressive actuals
        delta_in = sum(
            int(s.get("usage", {}).get("prompt_tokens", 0) or 0)
            for s in new_solutions
            if isinstance(s, dict)
        )
        delta_out = sum(
            int(s.get("usage", {}).get("completion_tokens", 0) or 0)
            for s in new_solutions
            if isinstance(s, dict)
        )

        def _bump_problem_usage(data):
            pa = data.setdefault(
                "progress_actuals",
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "requests": 0,
                    "input_cost_usd": 0.0,
                    "output_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                },
            )
            pa["input_tokens"] += int(delta_in)
            pa["output_tokens"] += int(delta_out)
            pa["requests"] += int(need)
            costs = tokens_to_cost(pa["input_tokens"], pa["output_tokens"])
            pa.update(costs)

        append_or_create(per_problem_usage_file, _bump_problem_usage)


# --------------- MAIN -----------------
async def main():
    # load problems
    problems = load_math_problems(
        problem_type=args.type,
        level=args.level,
        num_problems=args.num_problems,
        split=args.split,
        include_problems=args.include_problems,
    )

    # include/exclude lists
    if args.exclude_problems:
        exclude_ids = [
            int(x.strip())
            for x in args.exclude_problems.split(",")
            if x.strip().isdigit()
        ]
        problems = [(idx, p) for idx, p in problems if idx not in exclude_ids]
    if args.include_problems:
        include_ids = [
            int(x.strip())
            for x in args.include_problems.split(",")
            if x.strip().isdigit()
        ]
        problems = [(idx, p) for idx, p in problems if idx in include_ids]

    # optional substring filter (pick a single problem)
    if args.problem_substring:
        needle = args.problem_substring.lower()
        matched = [
            (idx, p)
            for idx, p in problems
            if isinstance(p.get("problem"), str) and needle in p["problem"].lower()
        ]
        if not matched:
            print(f"No problems contain substring '{args.problem_substring}'.")
            return
        if len(matched) > 1:
            # choose the first deterministically (you can make this stricter if you prefer)
            matched = [matched[0]]
        problems = matched

    if not problems:
        print("No problems loaded. Exiting.")
        return


# generate_rollouts_novita.py — Novita-only, usage+costs, fast HTTP/2, substring selection

import os
import json
import random
import math
import asyncio
import httpx
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import numpy as np

# utils: keep original structure, just import from your utils.utils
from utils.utils import (
    extract_boxed_answers,
    check_answer,
    split_solution_into_chunks,
    load_math_problems,
)

# ------- ENV & CONSTANTS -------
load_dotenv()

NOVITA_API_KEY = os.getenv("NOVITA_API_KEY")
if not NOVITA_API_KEY:
    raise ValueError("NOVITA_API_KEY not found in environment variables")

# Novita OpenAI-compatible completions base (completions endpoint)
NOVITA_COMPLETIONS_URL = "https://api.novita.ai/v3/openai/completions"

# Pricing (USD per 1M tokens) — as requested
PRICE_TABLE_USD_PER_MTOK = {
    "deepseek/deepseek-r1-distill-qwen-14b": {"input": 0.15, "output": 0.15}
}
DEFAULT_PRICE = {"input": 0.15, "output": 0.15}

# ------- ARGS -------
import argparse

parser = argparse.ArgumentParser(
    description="Generate chain-of-thought rollouts (Novita)"
)
parser.add_argument(
    "-m",
    "--model",
    type=str,
    default="deepseek/deepseek-r1-distill-qwen-14b",
    help="Model to use",
)
parser.add_argument(
    "-b",
    "--base_solution_type",
    type=str,
    default="correct",
    choices=["correct", "incorrect"],
    help="Type of base solution to keep",
)
parser.add_argument(
    "-r",
    "--rollout_type",
    type=str,
    default="default",
    choices=["default", "forced_answer"],
    help="Rollout type",
)
parser.add_argument(
    "-o", "--output_dir", type=str, default="math_rollouts", help="Output dir"
)
parser.add_argument(
    "-np",
    "--num_problems",
    type=int,
    default=100,
    help="How many problems to sample (after filters)",
)
parser.add_argument(
    "-nr", "--num_rollouts", type=int, default=100, help="Rollouts per chunk"
)
parser.add_argument("-t", "--temperature", type=float, default=0.6, help="Temperature")
parser.add_argument("-tp", "--top_p", type=float, default=0.95, help="Top-p")
parser.add_argument(
    "-mt", "--max_tokens", type=int, default=16384, help="Max new tokens"
)
parser.add_argument(
    "-mc", "--max_chunks", type=int, default=275, help="Max chunks allowed for rollout"
)
parser.add_argument("-s", "--seed", type=int, default=44, help="Seed")
parser.add_argument(
    "-f", "--force", action="store_true", help="Force regeneration if files exist"
)
parser.add_argument(
    "-ep",
    "--exclude_problems",
    type=str,
    default=None,
    help="Comma-separated problem IDs to exclude",
)
parser.add_argument(
    "-ip",
    "--include_problems",
    type=str,
    default=None,
    help="Comma-separated problem IDs to include",
)
parser.add_argument("-ty", "--type", type=str, default=None, help="Problem type filter")
parser.add_argument(
    "-l", "--level", type=str, default="Level 5", help="Problem level filter"
)
parser.add_argument(
    "-sp",
    "--split",
    type=str,
    default="train",
    choices=["train", "test"],
    help="Dataset split",
)
parser.add_argument(
    "--problem_substring",
    type=str,
    default=None,
    help="If set, pick exactly one problem whose text contains this substring (case-insensitive)",
)
parser.add_argument("-fp", "--frequency_penalty", type=float, default=None)
parser.add_argument("-pp", "--presence_penalty", type=float, default=None)
parser.add_argument("-rp", "--repetition_penalty", type=float, default=None)
parser.add_argument("-tk", "--top_k", type=int, default=None)
parser.add_argument("-mp", "--min_p", type=float, default=None)
parser.add_argument(
    "-mr", "--max_retries", type=int, default=3, help="Max retries per API call"
)
parser.add_argument(
    "-cc", "--concurrency", type=int, default=16, help="Concurrent API requests"
)
parser.add_argument(
    "-os", "--output_suffix", type=str, default=None, help="Suffix for output dir"
)
args = parser.parse_args()

# ------- OUTPUT DIR -------
base_output_dir = (
    Path(args.output_dir)
    / args.model.split("/")[-1]
    / f"temperature_{str(args.temperature)}_top_p_{str(args.top_p)}"
)
if args.rollout_type == "forced_answer":
    output_dir = (
        base_output_dir
        / f"{args.base_solution_type}_base_solution_{args.rollout_type}_{args.output_suffix}"
        if args.output_suffix
        else base_output_dir
        / f"{args.base_solution_type}_base_solution_{args.rollout_type}"
    )
else:
    output_dir = (
        base_output_dir
        / f"{args.base_solution_type}_base_solution_{args.output_suffix}"
        if args.output_suffix
        else base_output_dir / f"{args.base_solution_type}_base_solution"
    )
output_dir.mkdir(exist_ok=True, parents=True)

# ------- SEED -------
random.seed(args.seed)
np.random.seed(args.seed)


# ------- PRICING HELPERS -------
def _prices_for_model(model: str) -> Dict[str, float]:
    return PRICE_TABLE_USD_PER_MTOK.get(model, DEFAULT_PRICE)


def _cost_usd_from_tokens(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    p = _prices_for_model(model)
    return (prompt_tokens * (p["input"] / 1_000_000.0)) + (
        completion_tokens * (p["output"] / 1_000_000.0)
    )


# ------- SHARED HTTP CLIENT (HTTP/2, pooled) -------
HTTP_CLIENT: Optional[httpx.AsyncClient] = None
REQUEST_SEMAPHORE: Optional[asyncio.Semaphore] = None


def _novita_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {NOVITA_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


async def _init_http_client() -> None:
    global HTTP_CLIENT
    if HTTP_CLIENT is None:
        # HTTP/2 for multiplexing; tuned limits for concurrency
        limits = httpx.Limits(max_connections=256, max_keepalive_connections=64)
        HTTP_CLIENT = httpx.AsyncClient(http2=True, limits=limits, timeout=240)


async def _close_http_client() -> None:
    global HTTP_CLIENT
    if HTTP_CLIENT is not None:
        await HTTP_CLIENT.aclose()
        HTTP_CLIENT = None


# ------- NOVITA REQUEST -------
async def make_api_request(
    prompt: str, temperature: float, top_p: float, max_tokens: int
) -> Dict[str, Any]:
    """
    Novita /v3/openai/completions with OpenAI-compatible JSON.
    Returns { "text": str, "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int}, "finish_reason": str }
    """
    assert HTTP_CLIENT is not None
    payload: Dict[str, Any] = {
        "model": args.model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": 1,
        "stream": False,
    }
    # Optional params
    if args.frequency_penalty is not None:
        payload["frequency_penalty"] = args.frequency_penalty
    if args.presence_penalty is not None:
        payload["presence_penalty"] = args.presence_penalty
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.min_p is not None:
        payload["min_p"] = args.min_p

    max_retries = max(1, args.max_retries)
    base_delay = 1.5
    for attempt in range(max_retries):
        try:
            async with REQUEST_SEMAPHORE:
                resp = await HTTP_CLIENT.post(
                    NOVITA_COMPLETIONS_URL, headers=_novita_headers(), json=payload
                )
            if resp.status_code == 200:
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                text = choice.get("text", "")
                usage = data.get("usage", {}) or {}
                finish_reason = choice.get("finish_reason") or ""
                return {"text": text, "usage": usage, "finish_reason": finish_reason}
            # retry on 429/5xx
            if (
                resp.status_code in (429, 500, 502, 503, 504)
                and attempt < max_retries - 1
            ):
                jitter = random.uniform(0.6, 1.4)
                await asyncio.sleep(base_delay * (2**attempt) * jitter)
                continue
            return {"error": f"API error: {resp.status_code}", "details": resp.text}
        except Exception as e:
            if attempt == max_retries - 1:
                return {"error": f"Request exception: {e}"}
            await asyncio.sleep(base_delay * (2**attempt))

    return {"error": "All API request attempts failed"}


# ------- BASE SOLUTION -------
def _build_prompt(problem: Dict[str, Any], prefix: str = "") -> str:
    # Keep prompt shape; include prefix when resampling
    return f"Solve this math problem step by step. You MUST put your final answer in \\boxed{{}}. Problem: {problem['problem']} Solution: \n<think>\n{prefix}"


async def generate_base_solution(
    problem: Dict[str, Any], temperature: float = 0.6
) -> Dict[str, Any]:
    prompt = _build_prompt(problem)
    result = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
    if result.get("error"):
        return {
            "prompt": prompt,
            "solution": f"Error: {result['error']}",
            "error": result["error"],
        }

    solution_text = result["text"]
    extracted = extract_boxed_answers(solution_text)
    answer = extracted[0] if extracted else ""
    is_correct = False
    if problem.get("gt_answer") and answer:
        is_correct = check_answer(answer, problem["gt_answer"])

    usage = result.get("usage", {}) or {}
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    cost_usd = _cost_usd_from_tokens(args.model, pt, ct)

    return {
        "prompt": prompt,
        "solution": solution_text,
        "full_cot": prompt + solution_text,
        "answer": answer,
        "is_correct": is_correct,
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
        "cost_usd": cost_usd,
    }


# ------- ROLLOUT GENERATION -------
def _make_rollout_prompt(
    problem: Dict[str, Any], prefix_without_chunk: str, rollout_type: str
) -> str:
    base = _build_prompt(problem, prefix_without_chunk)
    if rollout_type == "forced_answer":
        return base + "\n</think>\n\nTherefore, the final answers is \\boxed{"
    return base


async def generate_rollout(
    problem: Dict[str, Any],
    chunk_text: str,
    full_cot_prefix: str,
    temperature: float = 0.7,
    rollout_type: str = "default",
) -> Dict[str, Any]:
    prefix_without_chunk = full_cot_prefix.replace(chunk_text, "").strip()
    prompt = _make_rollout_prompt(problem, prefix_without_chunk, rollout_type)

    result = await make_api_request(prompt, temperature, args.top_p, args.max_tokens)
    if result.get("error"):
        return {
            "chunk_removed": chunk_text,
            "prefix_without_chunk": prefix_without_chunk,
            "error": result["error"],
            "details": result.get("details"),
        }

    rollout_text = result["text"]
    # The "resampled chunk" is the first chunk of the new rollout continuation
    chunk_resampled = ""
    try:
        chunk_resampled = split_solution_into_chunks(rollout_text)[0]
    except Exception:
        chunk_resampled = ""

    full_for_answer = (
        f"{prompt}{rollout_text}" if rollout_type == "forced_answer" else rollout_text
    )
    extracted = extract_boxed_answers(full_for_answer)
    answer = extracted[0] if extracted else ""
    is_correct = False
    if problem.get("gt_answer") and answer:
        is_correct = check_answer(answer, problem["gt_answer"])

    usage = result.get("usage", {}) or {}
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)
    cost_usd = _cost_usd_from_tokens(args.model, pt, ct)

    return {
        "chunk_removed": chunk_text,
        "prefix_without_chunk": prefix_without_chunk,
        "chunk_resampled": chunk_resampled,
        "rollout": rollout_text,
        "full_cot": f"{prompt}{rollout_text}",
        "answer": answer,
        "is_correct": is_correct,
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
        "cost_usd": cost_usd,
    }


# ------- PROBLEM PROCESSING -------
async def process_problem(problem_idx: int, problem: Dict[str, Any]) -> None:
    problem_dir = output_dir / f"problem_{problem_idx}"
    problem_dir.mkdir(exist_ok=True, parents=True)

    # Save problem
    problem_file = problem_dir / "problem.json"
    if not problem_file.exists() or args.force:
        with open(problem_file, "w", encoding="utf-8") as f:
            json.dump(problem, f, indent=2)

    # Base solution
    base_solution_file = problem_dir / "base_solution.json"
    base_solution = None
    if base_solution_file.exists() and not args.force:
        with open(base_solution_file, "r", encoding="utf-8") as f:
            base_solution = json.load(f)
            # ensure accuracy up-to-date
            if "solution" in base_solution:
                extracted = extract_boxed_answers(base_solution["solution"])
                answer = extracted[0] if extracted else ""
                is_correct = False
                if problem.get("gt_answer") and answer:
                    is_correct = check_answer(answer, problem["gt_answer"])
                if (
                    base_solution.get("answer") != answer
                    or base_solution.get("is_correct") != is_correct
                ):
                    base_solution["answer"] = answer
                    base_solution["is_correct"] = is_correct
                    with open(base_solution_file, "w", encoding="utf-8") as f:
                        json.dump(base_solution, f, indent=2)

    if base_solution is None:
        base_solution = await generate_base_solution(problem, args.temperature)

        # Respect requested base type (correct/incorrect)
        if args.base_solution_type == "correct":
            if (not base_solution.get("is_correct")) or base_solution.get("error"):
                print(base_solution.get("solution", ""))
                print(
                    f"Problem {problem_idx}: Base solution INCORRECT or error. Retrying..."
                )
                return await process_problem(problem_idx, problem)
        else:  # "incorrect"
            if base_solution.get("is_correct") or base_solution.get("error"):
                print(base_solution.get("solution", ""))
                print(
                    f"Problem {problem_idx}: Base solution CORRECT or error. Retrying..."
                )
                return await process_problem(problem_idx, problem)

        with open(base_solution_file, "w", encoding="utf-8") as f:
            json.dump(base_solution, f, indent=2)

    # Chunks
    source_text = base_solution["full_cot"]
    if "<think>" in source_text:
        solution_text = source_text.split("<think>")[1].strip()
        if "</think>" in solution_text:
            solution_text = solution_text.split("</think>")[0].strip()
    else:
        solution_text = source_text

    chunks_file = problem_dir / "chunks.json"
    if not chunks_file.exists() or args.force:
        chunks = split_solution_into_chunks(solution_text)
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "source_text": source_text,
                    "solution_text": solution_text,
                    "chunks": chunks,
                },
                f,
                indent=2,
            )
    else:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)["chunks"]

    if len(chunks) > args.max_chunks:
        print(
            f"Problem {problem_idx}: Too many chunks ({len(chunks)} > {args.max_chunks}). Skipping rollouts."
        )
        return

    # Cumulative prefix text per chunk (to remove current chunk cleanly)
    cumulative_chunks: List[str] = []
    cur = ""
    for ch in chunks:
        cur += ch + " "
        cumulative_chunks.append(cur.strip())

    # For each chunk: generate missing rollouts in parallel (bounded)
    for chunk_idx, (chunk, full_prefix) in enumerate(zip(chunks, cumulative_chunks)):
        chunk_dir = problem_dir / f"chunk_{chunk_idx}"
        chunk_dir.mkdir(exist_ok=True, parents=True)

        solutions_file = chunk_dir / "solutions.json"
        existing_solutions: List[Dict[str, Any]] = []
        valid_existing: List[Dict[str, Any]] = []

        if solutions_file.exists() and not args.force:
            with open(solutions_file, "r", encoding="utf-8") as f:
                existing_solutions = json.load(f) or []
            # refresh accuracy if needed
            updated = 0
            for r in existing_solutions:
                if "rollout" in r and "error" not in r:
                    full_for_answer = (
                        r["full_cot"]
                        if args.rollout_type == "forced_answer"
                        else r["rollout"]
                    )
                    extracted = extract_boxed_answers(full_for_answer)
                    ans = extracted[0] if extracted else ""
                    ok = False
                    if problem.get("gt_answer") and ans:
                        ok = check_answer(ans, problem["gt_answer"])
                    if r.get("answer") != ans or r.get("is_correct") != ok:
                        r["answer"] = ans
                        r["is_correct"] = ok
                        updated += 1
            if updated > 0:
                with open(solutions_file, "w", encoding="utf-8") as f:
                    json.dump(existing_solutions, f, indent=2)

            valid_existing = [
                s for s in existing_solutions if "answer" in s and "error" not in s
            ]

        need = max(0, args.num_rollouts - len(valid_existing))
        if need <= 0:
            print(
                f"Problem {problem_idx}, Chunk {chunk_idx}: Already have {len(valid_existing)} valid solutions"
            )
            continue

        print(f"Problem {problem_idx}, Chunk {chunk_idx}: Generating {need} rollouts")
        tasks = [
            generate_rollout(
                problem, chunk, full_prefix, args.temperature, args.rollout_type
            )
            for _ in range(need)
        ]
        new_solutions = await asyncio.gather(*tasks)

        all_solutions = existing_solutions + new_solutions

        # Aggregate usage + costs at chunk level (just for convenience — structure still a list)
        agg_pt = sum(
            int(s.get("usage", {}).get("prompt_tokens", 0))
            for s in new_solutions
            if "usage" in s
        )
        agg_ct = sum(
            int(s.get("usage", {}).get("completion_tokens", 0))
            for s in new_solutions
            if "usage" in s
        )
        agg_cost = sum(float(s.get("cost_usd", 0.0)) for s in new_solutions)

        # Save list (structure preserved); also store a companion summary file
        with open(solutions_file, "w", encoding="utf-8") as f:
            json.dump(all_solutions, f, indent=2)

        summary = {
            "new_rollouts": need,
            "new_usage": {
                "prompt_tokens": agg_pt,
                "completion_tokens": agg_ct,
                "total_tokens": agg_pt + agg_ct,
                "cost_usd": agg_cost,
            },
        }
        with open(chunk_dir / "solutions_usage.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


# ------- MAIN -------
async def main():
    global REQUEST_SEMAPHORE
    REQUEST_SEMAPHORE = asyncio.Semaphore(args.concurrency)
    await _init_http_client()

    try:
        problems = load_math_problems(
            problem_type=args.type,
            level=args.level,
            num_problems=args.num_problems,
            split=args.split,
            include_problems=args.include_problems,
        )

        # Exclude/include by explicit IDs
        if args.exclude_problems:
            exclude_ids = {
                int(x.strip())
                for x in args.exclude_problems.split(",")
                if x.strip().isdigit()
            }
            problems = [(idx, p) for idx, p in problems if idx not in exclude_ids]
        if args.include_problems:
            include_ids = {
                int(x.strip())
                for x in args.include_problems.split(",")
                if x.strip().isdigit()
            }
            problems = [(idx, p) for idx, p in problems if idx in include_ids]

        # Optional: select a single problem by substring (case-insensitive, must be unique)
        if args.problem_substring:
            needle = args.problem_substring.lower()
            matched = [
                (idx, p)
                for idx, p in problems
                if isinstance(p.get("problem"), str) and needle in p["problem"].lower()
            ]
            if not matched:
                print(
                    f"No problems contain substring '{args.problem_substring}'. Exiting."
                )
                return
            if len(matched) > 1:
                ids = ", ".join(str(idx) for idx, _ in matched)
                print(
                    f"Substring matched multiple problems (indices: {ids}). Refusing selection."
                )
                return
            problems = [matched[0]]

        if not problems:
            print("No problems loaded after filters. Exiting.")
            return

        print(f"Loaded {len(problems)} problems.")
        for problem_idx, problem in tqdm(problems, desc="Processing problems"):
            await process_problem(problem_idx, problem)

    finally:
        await _close_http_client()


if __name__ == "__main__":
    asyncio.run(main())

    # initialize central file skeleton
    write_or_update_central_estimates(
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "requests": 0,
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "per_problem": {},
        }
    )

    semaphore = asyncio.Semaphore(args.concurrency)
    for problem_idx, problem in tqdm(problems, desc="Processing problems"):
        await process_problem(problem_idx, problem, semaphore)


if __name__ == "__main__":
    asyncio.run(main())
