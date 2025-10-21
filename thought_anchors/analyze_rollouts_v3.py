import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
import re
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from utils.utils import (
    normalize_answer,
    split_solution_into_chunks,
)
from utils.prompts import (
    DAG_PROMPT,
    NICKNAME_PROMPT,
    SUMMARY_PROMPT,
)
import math
import multiprocessing as mp
from functools import partial
import scipy.stats as stats
from matplotlib.lines import Line2D
import time


# ---------------------------
# CLI
# ---------------------------
IMPORTANCE_METRICS = [
    "resampling_importance_accuracy",
    "resampling_importance_kl",
    "counterfactual_importance_accuracy",
    "counterfactual_importance_kl",
    "forced_importance_accuracy",
    "forced_importance_kl",
]

parser = argparse.ArgumentParser(
    description="Analyze rollout data and label chunks (Novita + HF compatible)"
)
parser.add_argument(
    "-ic",
    "--correct_rollouts_dir",
    type=str,
    default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution",
    help="Directory or hf:// spec for CORRECT rollouts",
)
parser.add_argument(
    "-ii",
    "--incorrect_rollouts_dir",
    type=str,
    default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution",
    help="Directory or hf:// spec for INCORRECT rollouts",
)
parser.add_argument(
    "-icf",
    "--correct_forced_answer_rollouts_dir",
    type=str,
    default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution_forced_answer",
    help="Directory or hf:// spec for CORRECT forced-answer rollouts",
)
parser.add_argument(
    "-iif",
    "--incorrect_forced_answer_rollouts_dir",
    type=str,
    default="math_rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/incorrect_base_solution_forced_answer",
    help="Directory or hf:// spec for INCORRECT forced-answer rollouts",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    default="analysis/basic",
    help="Directory to save analysis",
)
parser.add_argument(
    "-p",
    "--problems",
    type=str,
    default=None,
    help="Comma-separated problem indices to analyze",
)
parser.add_argument(
    "-m",
    "--max_problems",
    type=int,
    default=None,
    help="Max number of problems to analyze",
)
parser.add_argument(
    "-a",
    "--absolute",
    default=False,
    action="store_true",
    help="Use absolute values for importance deltas",
)
parser.add_argument(
    "-f",
    "--force_relabel",
    default=False,
    action="store_true",
    help="Force re-label chunks (runs DAG labeling again)",
)
parser.add_argument(
    "-fm",
    "--force_metadata",
    default=False,
    action="store_true",
    help="Force regeneration of chunk summaries & problem nicknames",
)
parser.add_argument(
    "-d",
    "--dag_dir",
    type=str,
    default="archive/analysis/math",
    help="Dir with DAG-improved chunks (for token freq analysis)",
)
parser.add_argument(
    "-t",
    "--token_analysis_source",
    type=str,
    default="dag",
    choices=["dag", "rollouts"],
    help='Token frequency source: "dag" or "rollouts"',
)
parser.add_argument(
    "-tf",
    "--get_token_frequencies",
    default=False,
    action="store_true",
    help="Compute token frequencies",
)
parser.add_argument(
    "-mc",
    "--max_chunks_to_show",
    type=int,
    default=100,
    help="Max chunk index to show in certain plots",
)
parser.add_argument(
    "-tc",
    "--top_chunks",
    type=int,
    default=500,
    help="Top chunks to use for similar/dissimilar selection",
)
parser.add_argument(
    "-u",
    "--use_existing_metrics",
    default=False,
    action="store_true",
    help="Use existing metrics from chunks_labeled.json if available",
)
parser.add_argument(
    "-im",
    "--importance_metric",
    type=str,
    default="counterfactual_importance_accuracy",
    choices=IMPORTANCE_METRICS,
    help="Which importance metric to drive plots/analysis",
)
parser.add_argument(
    "-sm",
    "--sentence_model",
    type=str,
    default="all-MiniLM-L6-v2",
    help="SentenceTransformer model",
)
parser.add_argument(
    "-st",
    "--similarity_threshold",
    type=float,
    default=0.8,
    help="Similarity threshold for dissimilar resamples",
)
parser.add_argument(
    "-bs", "--batch_size", type=int, default=8192, help="Batch size for embeddings"
)
parser.add_argument(
    "-us",
    "--use_similar_chunks",
    default=True,
    action="store_true",
    help="Include similar chunks in CF KL computation",
)
parser.add_argument(
    "-np",
    "--num_processes",
    type=int,
    default=min(100, mp.cpu_count()),
    help="Parallel processes",
)
parser.add_argument(
    "-pt",
    "--use_prob_true",
    default=False,
    action="store_false",
    help="Use P(true) for KL instead of answer distribution",
)
# New: HF + Novita knobs
parser.add_argument(
    "--hf_cache_dir",
    type=str,
    default=".cache/math_rollouts_hf",
    help="Local cache dir for HF materialized files",
)
parser.add_argument(
    "--llm_provider",
    type=str,
    default="openrouter",
    choices=["openrouter", "none"],
    help="Provider for small metadata generations (nicknames, chunk summaries, DAG). 'none' skips.",
)
parser.add_argument(
    "--llm_model",
    type=str,
    default="openai/gpt-4o-mini",
    help="LLM model to use on OpenRouter for summaries/labels",
)
parser.add_argument(
    "--llm_temperature", type=float, default=0.0, help="Temperature for metadata calls"
)
parser.add_argument(
    "--llm_max_tokens", type=int, default=20, help="Max tokens for metadata calls"
)

args = parser.parse_args()

# ---------------------------
# Global config / style
# ---------------------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

FONT_SIZE = 20
plt.rcParams.update(
    {
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 4,
        "axes.labelsize": FONT_SIZE + 2,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.titlesize": FONT_SIZE + 6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelpad": 20,
        "axes.titlepad": 20,
    }
)
FIGSIZE = (20, 7)

# ---------------------------
# Env / tokenizer
# ---------------------------
load_dotenv()

if args.llm_provider == "openrouter":
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY_SPAR") or os.getenv(
        "OPENROUTER_API_KEY"
    )
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY_SPAR (or OPENROUTER_API_KEY) not found in .env"
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        # These are optional but recommended by OpenRouter
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://local"),
            "X-Title": os.getenv("OPENROUTER_X_TITLE", "Math Rollouts Analysis"),
        },
    )


# Tokenizer for token counts (same as in paper’s models list)
# (deepseek R1-distill qwen-14b)
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-r1-distill-qwen-14b")

# ---------------------------
# Stopwords for token freq
# ---------------------------
stopwords = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "at",
    "from",
    "for",
    "with",
    "about",
    "into",
    "through",
    "above",
    "ve",
    "below",
    "under",
    "again",
    "further",
    "here",
    "there",
    "all",
    "most",
    "other",
    "some",
    "such",
    "to",
    "on",
    "only",
    "own",
    "too",
    "very",
    "will",
    "wasn",
    "weren",
    "wouldn",
    "this",
    "that",
    "these",
    "those",
    "of",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "got",
    "does",
    "did",
    "doing",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "get",
    "in",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "so",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "as",
    "us",
}


def safe_json_parse(text: str) -> Optional[dict]:
    text = text.strip()
    # try to extract JSON substring if model added extra text
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        return json.loads(text)
    except Exception:
        return None


# ---------------------------
# Token counting
# ---------------------------
def count_tokens(text: str, approximate: bool = True) -> int:
    if approximate:
        return len(text) // 4
    return len(tokenizer.encode(text))


# ---------------------------
# HF materializer
# ---------------------------
def is_hf_spec(path_or_hf: Optional[str]) -> bool:
    return isinstance(path_or_hf, str) and path_or_hf.startswith("hf://")


def materialize_hf_prefix(hf_spec: str, cache_root: Path) -> Optional[Path]:
    """
    hf_spec format:
      hf://uzaymacar/math-rollouts/<prefix/path/from/dataset/root>
    Example:
      hf://uzaymacar/math-rollouts/deepseek-r1-distill-qwen-14b/temperature_0.6_top_p_0.95/correct_base_solution
    This will write all dataset rows whose 'path' startswith that prefix to cache_root/<prefix>/...
    """
    try:
        from datasets import load_dataset
        # DownloadConfig allows us to control retries and timeouts during HF downloads
        try:
            # Newer datasets versions
            from datasets.utils.download_manager import DownloadConfig
        except Exception:
            # Older datasets versions
            from datasets.utils.file_utils import DownloadConfig  # type: ignore
    except ImportError:
        raise RuntimeError("pip install datasets to use HF mode")

    assert hf_spec.startswith("hf://")
    parts = hf_spec[len("hf://") :].split("/", 2)
    if len(parts) < 2:
        raise ValueError(
            "hf spec must be hf://<org-or-user>/<dataset>/<optional-prefix>"
        )
    repo = f"{parts[0]}/{parts[1]}"
    prefix = parts[2] if len(parts) > 2 else ""

    # Be lenient with HF Hub timeouts on slow networks
    os.environ.setdefault("HF_HUB_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_CONNECT_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

    print(f"[HF] Loading dataset {repo} ...")
    # Use a generous timeout and retries to avoid ReadTimeouts
    # Note: DownloadConfig has no 'timeout' parameter; timeouts are controlled via HF_HUB_* env vars.
    dl_config = DownloadConfig(
        max_retries=5,
        resume_download=True,
        cache_dir=str(cache_root),
    )

    last_err = None
    for attempt in range(1, 6):
        try:
            ds = load_dataset(
                repo,
                split="train",
                download_config=dl_config,
                cache_dir=str(cache_root),
            )  # dataset is a single split with rows as files
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"[HF] load_dataset failed (attempt {attempt}/5): {e}")
            # brief backoff before retrying
            time.sleep(min(2 * attempt, 10))
    if last_err is not None:
        # Bubble up with a clearer message
        raise RuntimeError(
            f"Failed to load HF dataset '{repo}' after retries. Consider increasing network timeouts or re-running. Last error: {last_err}"
        )
    # Expected columns include 'path' and 'content' (auto-converted parquet listing)
    # See dataset card for structure. (We rely on 'path' and 'content'.)

    out_base = cache_root / prefix
    out_base.mkdir(parents=True, exist_ok=True)

    print(f"[HF] Materializing files with prefix '{prefix}' into {out_base} ...")
    # Iterate & write
    count = 0
    for row in tqdm(ds, total=len(ds)):
        p = row.get("path") or row.get("repo_path") or ""
        if not isinstance(p, str) or not p:
            continue
        if prefix and not p.startswith(prefix):
            continue
        # write content
        content = row.get("content", "")
        target_path = cache_root / p
        target_path.parent.mkdir(parents=True, exist_ok=True)
        # content is already text for json files (per dataset card preview)
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content if isinstance(content, str) else str(content))
        count += 1

    if count == 0:
        print(f"[HF] No files matched prefix '{prefix}'. Double-check the path.")
    else:
        print(f"[HF] Wrote {count} files.")
    return out_base


def resolve_dir(path_or_hf: Optional[str], cache_root: Path) -> Optional[Path]:
    if not path_or_hf:
        return None
    if is_hf_spec(path_or_hf):
        return materialize_hf_prefix(path_or_hf, cache_root)
    p = Path(path_or_hf)
    return p if p.exists() else None


# ---------------------------
# Importance helper args
# ---------------------------
class ImportanceArgs:
    def __init__(
        self,
        use_absolute=False,
        forced_answer_dir=None,
        similarity_threshold=0.8,
        use_similar_chunks=True,
        use_abs_importance=False,
        top_chunks=100,
        use_prob_true=True,
    ):
        self.use_absolute = use_absolute
        self.forced_answer_dir = forced_answer_dir
        self.similarity_threshold = similarity_threshold
        self.use_similar_chunks = use_similar_chunks
        self.use_abs_importance = use_abs_importance
        self.top_chunks = top_chunks
        self.use_prob_true = use_prob_true


# ---------------------------
# LLM metadata generation
# ---------------------------
def generate_chunk_summary(chunk_text: str) -> str:
    if args.llm_provider == "none":
        return "unknown action"
    prompt = SUMMARY_PROMPT.format(chunk=chunk_text.strip()[:1200])
    try:
        resp = client.chat.completions.create(
            model=args.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
        summary = resp.choices[0].message.content.strip()
        # (post-processing unchanged)
        return " ".join(summary.replace('"', "").split()[:5])
    except Exception as e:
        print(f"Error generating chunk summary: {e}")
        return "unknown action"


def generate_problem_nickname(problem_text: str) -> str:
    if args.llm_provider == "none":
        return "math problem"
    prompt = NICKNAME_PROMPT.format(problem=problem_text.strip()[:2000])
    try:
        resp = client.chat.completions.create(
            model=args.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.llm_temperature,
            max_tokens=args.llm_max_tokens,
        )
        nickname = resp.choices[0].message.content.strip()
        return " ".join(nickname.replace('"', "").split()[:5])
    except Exception as e:
        print(f"Error generating problem nickname: {e}")
        return "math problem"


def dag_label_chunks(problem_text: str, chunks: List[str]) -> Dict[str, Dict]:
    """Return {"chunks": {"0": {"function_tags":[...], "depends_on":[...]}, ...}}."""
    if args.llm_provider == "none":
        # minimal no-LLM fallback: tag everything as unknown with empty deps
        return {
            "chunks": {
                str(i): {"function_tags": ["unknown"], "depends_on": []}
                for i in range(len(chunks))
            }
        }
    full_chunked_text = ""
    for i, c in enumerate(chunks):
        full_chunked_text += f"Chunk {i}:\n{c}\n\n"
    prompt = DAG_PROMPT.format(
        problem_text=problem_text, full_chunked_text=full_chunked_text[:6000]
    )
    try:
        resp = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        out = resp.choices[0].message.content
    except Exception as e:
        print(f"Error generating DAG labels: {e}")
        out = ""
    data = safe_json_parse(out)
    if not data or "chunks" not in data:
        # fallback: unknowns
        print(
            "chunks not in DAG response or dag labeling response empty, defaulting to unknown"
        )
        return {
            "chunks": {
                str(i): {"function_tags": ["unknown"], "depends_on": []}
                for i in range(len(chunks))
            }
        }
    # sanitize shapes
    cleaned = {}
    for i in range(len(chunks)):
        key = str(i)
        obj = data["chunks"].get(key, {}) if isinstance(data["chunks"], dict) else {}
        tags = obj.get("function_tags", ["unknown"])
        deps = obj.get("depends_on", [])
        if not isinstance(tags, list):
            tags = ["unknown"]
        if not isinstance(deps, list):
            deps = []
        cleaned[key] = {
            "function_tags": tags,
            "depends_on": [str(d) for d in deps if isinstance(d, (int, str))],
        }
    return {"chunks": cleaned}


# ---------------------------
# Importance metrics (same logic as your original script)
# ---------------------------
def calculate_kl_divergence(
    chunk_sols1, chunk_sols2, laplace_smooth=False, use_prob_true=True
):
    if use_prob_true:
        correct1 = sum(1 for s in chunk_sols1 if s.get("is_correct", False) is True)
        total1 = sum(1 for s in chunk_sols1 if s.get("is_correct", None) is not None)
        correct2 = sum(1 for s in chunk_sols2 if s.get("is_correct", False) is True)
        total2 = sum(1 for s in chunk_sols2 if s.get("is_correct", None) is not None)
        if total1 == 0 or total2 == 0:
            return 0.0
        alpha = 1 if laplace_smooth else 1e-9
        p = (correct1 + alpha) / (total1 + 2 * alpha)
        q = (correct2 + alpha) / (total2 + 2 * alpha)
        kl = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
        return max(0.0, kl)
    # answer-distribution mode
    answer_counts1 = defaultdict(int)
    answer_counts2 = defaultdict(int)
    for s in chunk_sols1:
        ans = normalize_answer(s.get("answer", ""))
        if ans:
            answer_counts1[ans] += 1
    for s in chunk_sols2:
        ans = normalize_answer(s.get("answer", ""))
        if ans:
            answer_counts2[ans] += 1
    if not answer_counts1 or not answer_counts2:
        return 0.0
    all_ans = set(answer_counts1) | set(answer_counts2)
    V = len(all_ans)
    total1 = sum(answer_counts1.values())
    total2 = sum(answer_counts2.values())
    if total1 == 0 or total2 == 0:
        return 0.0
    alpha = 1 if laplace_smooth else 1e-9
    st1 = total1 + alpha * V
    st2 = total2 + alpha * V
    kl = 0.0
    for a in all_ans:
        p = (answer_counts1[a] + alpha) / st1
        q = (answer_counts2[a] + alpha) / st2
        kl += p * math.log(p / q)
    return max(0.0, kl)


def calculate_resampling_importance_accuracy(
    chunk_idx, chunk_accuracies, args_local=None
):
    if chunk_idx not in chunk_accuracies:
        return 0.0
    current = chunk_accuracies[chunk_idx]
    prev = [acc for idx, acc in chunk_accuracies.items() if idx <= chunk_idx]
    nexts = [acc for idx, acc in chunk_accuracies.items() if idx == chunk_idx + 1]
    if not prev or not nexts:
        return 0.0
    diff = (sum(nexts) / len(nexts)) - current
    if args_local and getattr(args_local, "use_abs_importance", False):
        return abs(diff)
    return diff


def calculate_resampling_importance_kl(chunk_idx, chunk_info, problem_dir):
    if chunk_idx not in chunk_info:
        return 0.0
    next_chunks = [i for i in chunk_info.keys() if i > chunk_idx]
    if not next_chunks:
        return 0.0
    curr_dir = problem_dir / f"chunk_{chunk_idx}"
    next_idx = min(next_chunks)
    next_dir = problem_dir / f"chunk_{next_idx}"
    sols1 = []
    sols2 = []
    f1 = curr_dir / "solutions.json"
    f2 = next_dir / "solutions.json"
    if f1.exists():
        try:
            sols1 = json.load(open(f1, "r", encoding="utf-8"))
        except Exception:
            pass
    if f2.exists():
        try:
            sols2 = json.load(open(f2, "r", encoding="utf-8"))
        except Exception:
            pass
    if not sols1 or not sols2:
        return 0.0
    return calculate_kl_divergence(sols1, sols2, use_prob_true=args.use_prob_true)


def calculate_forced_importance_accuracy(
    chunk_idx, forced_answer_accuracies, args_local=None
):
    if chunk_idx not in forced_answer_accuracies:
        return 0.0
    next_chunks = [i for i in forced_answer_accuracies if i > chunk_idx]
    if not next_chunks:
        return 0.0
    next_idx = min(next_chunks)
    diff = forced_answer_accuracies[next_idx] - forced_answer_accuracies[chunk_idx]
    if args_local and getattr(args_local, "use_abs_importance", False):
        return abs(diff)
    return diff


def calculate_forced_importance_kl(
    chunk_idx, forced_answer_accuracies, problem_dir, forced_answer_dir
):
    if chunk_idx not in forced_answer_accuracies:
        return 0.0
    forced_problem_dir = forced_answer_dir / problem_dir.name
    if not forced_problem_dir.exists():
        return 0.0
    curr_dir = forced_problem_dir / f"chunk_{chunk_idx}"
    curr_sols = []
    if curr_dir.exists():
        f = curr_dir / "solutions.json"
        if f.exists():
            try:
                curr_sols = json.load(open(f, "r", encoding="utf-8"))
            except Exception:
                pass
    next_chunks = [i for i in forced_answer_accuracies if i > chunk_idx]
    if not next_chunks:
        return 0.0
    next_idx = min(next_chunks)
    next_dir = forced_problem_dir / f"chunk_{next_idx}"
    next_sols = []
    if next_dir.exists():
        f = next_dir / "solutions.json"
        if f.exists():
            try:
                next_sols = json.load(open(f, "r", encoding="utf-8"))
            except Exception:
                pass
    if not curr_sols or not next_sols:
        return 0.0
    return calculate_kl_divergence(
        curr_sols, next_sols, use_prob_true=args.use_prob_true
    )


def calculate_counterfactual_importance_accuracy(
    chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, args_local
):
    if chunk_idx not in chunk_info:
        return (0.0, 0.0, 0.0)
    next_chunks = [i for i in chunk_info if i > chunk_idx]
    if not next_chunks:
        return (0.0, 0.0, 0.0)
    next_idx = min(next_chunks)
    current_sols = chunk_info[chunk_idx]
    next_sols = chunk_info.get(next_idx, [])
    dissimilar = []
    different_traj = 0
    pairs = []
    for sol in current_sols:
        removed = sol.get("chunk_removed", "")
        resampled = sol.get("chunk_resampled", "")
        if (
            isinstance(removed, str)
            and isinstance(resampled, str)
            and removed in chunk_embedding_cache
            and resampled in chunk_embedding_cache
        ):
            e1 = chunk_embedding_cache[removed]
            e2 = chunk_embedding_cache[resampled]
            sim = float(
                np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
            )
            if sim < args_local.similarity_threshold:
                dissimilar.append(sol)
                different_traj += 1
            pairs.append((removed, resampled, sol))
    diff_traj_frac = (different_traj / len(current_sols)) if current_sols else 0.0
    resampled_list = [p[1] for p in pairs]
    unique_resampled = set(resampled_list)
    overdet = (
        1.0 - (len(unique_resampled) / len(resampled_list)) if resampled_list else 0.0
    )
    if not dissimilar or not next_sols:
        return (0.0, diff_traj_frac, overdet)
    dis_corr = sum(
        1
        for s in dissimilar
        if s.get("is_correct", False) is True and s.get("answer", "") != ""
    )
    dis_tot = sum(
        1
        for s in dissimilar
        if s.get("is_correct", None) is not None and s.get("answer", "") != ""
    )
    dis_acc = (dis_corr / dis_tot) if dis_tot > 0 else 0.0
    next_corr = sum(
        1
        for s in next_sols
        if s.get("is_correct", True) is True and s.get("answer", "") != ""
    )
    next_tot = sum(
        1
        for s in next_sols
        if s.get("is_correct", None) is not None and s.get("answer", "") != ""
    )
    next_acc = (next_corr / next_tot) if next_tot > 0 else 0.0
    diff = dis_acc - next_acc
    if getattr(args_local, "use_abs_importance", False):
        diff = abs(diff)
    return (diff, diff_traj_frac, overdet)


def calculate_counterfactual_importance_kl(
    chunk_idx,
    chunk_info,
    chunk_embedding_cache,
    chunk_accuracies,
    chunk_answers,
    args_local,
):
    if chunk_idx not in chunk_info:
        return 0.0
    next_chunks = [i for i in chunk_info if i > chunk_idx]
    if not next_chunks:
        return 0.0
    next_idx = min(next_chunks)
    current_sols = chunk_info[chunk_idx]
    next_sols = chunk_info.get(next_idx, [])
    dissimilar = []
    similar = []
    for s in current_sols:
        removed = s.get("chunk_removed", "")
        resampled = s.get("chunk_resampled", "")
        if (
            isinstance(removed, str)
            and isinstance(resampled, str)
            and removed in chunk_embedding_cache
            and resampled in chunk_embedding_cache
        ):
            e1 = chunk_embedding_cache[removed]
            e2 = chunk_embedding_cache[resampled]
            sim = float(
                np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)
            )
            if sim < args_local.similarity_threshold:
                dissimilar.append(s)
            else:
                similar.append(s)
    if not dissimilar or not next_sols:
        return 0.0

    # build distributions
    def dist_from_sols(sols):
        d = defaultdict(int)
        for s in sols:
            a = normalize_answer(s.get("answer", ""))
            if a:
                d[a] += 1
        return d

    # convert to list-of-sols shape for kl calc (with is_correct)
    answer_correct = {}
    for s in current_sols + next_sols:
        a = normalize_answer(s.get("answer", ""))
        if a:
            answer_correct[a] = s.get("is_correct", False)

    def expand(d):
        out = []
        for a, c in d.items():
            for _ in range(c):
                out.append({"answer": a, "is_correct": answer_correct.get(a, False)})
        return out

    similar_d = dist_from_sols(similar)
    dissim_d = dist_from_sols(dissimilar)
    next_d = dist_from_sols(next_sols)
    return calculate_kl_divergence(
        expand(dissim_d),
        expand(next_d) + (expand(similar_d) if args_local.use_similar_chunks else []),
        use_prob_true=args_local.use_prob_true,
    )


# ---------------------------
# Problem analysis (adapted to Novita + HF)
# ---------------------------
embedding_model_cache = {}


def analyze_problem(
    problem_dir: Path,
    use_absolute: bool = False,
    force_relabel: bool = False,
    forced_answer_dir: Optional[Path] = None,
    use_existing_metrics: bool = False,
    sentence_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    force_metadata: bool = False,
) -> Optional[Dict]:
    base_solution_file = problem_dir / "base_solution.json"
    chunks_file = problem_dir / "chunks.json"
    problem_file = problem_dir / "problem.json"
    if not (
        base_solution_file.exists() and chunks_file.exists() and problem_file.exists()
    ):
        print(f"Problem {problem_dir.name}: Missing files")
        return None

    problem = json.load(open(problem_file, "r", encoding="utf-8"))
    # nickname
    if force_metadata or not problem.get("nickname"):
        try:
            problem["nickname"] = generate_problem_nickname(problem.get("problem", ""))
            json.dump(problem, open(problem_file, "w", encoding="utf-8"), indent=2)
        except Exception as e:
            print(f"Nickname error {problem_dir.name}: {e}")
            problem["nickname"] = "math problem"

    base_solution = json.load(open(base_solution_file, "r", encoding="utf-8"))
    chunks_data = json.load(open(chunks_file, "r", encoding="utf-8"))
    chunks = [c for c in chunks_data["chunks"] if len(c) >= 4]
    valid_idx = [i for i, c in enumerate(chunks_data["chunks"]) if len(c) >= 4]

    # check chunk folders
    chunk_folders = [problem_dir / f"chunk_{i}" for i in valid_idx]
    existing = [d for d in chunk_folders if d.exists()]
    if len(existing) < 0.1 * len(chunks):
        print(
            f"Problem {problem_dir.name}: only {len(existing)}/{len(chunks)} chunk dirs; skipping"
        )
        return None

    # precompute accuracies, answers, info
    chunk_accuracies = {}
    chunk_answers = {}
    chunk_info = {}
    token_counts = []

    # forced answers accuracies if present
    forced_answer_accuracies = {}
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            for chunk_idx in valid_idx:
                cd = forced_problem_dir / f"chunk_{chunk_idx}"
                sols_file = cd / "solutions.json"
                if not sols_file.exists():
                    forced_answer_accuracies[chunk_idx] = 0.0
                    continue
                try:
                    sols = json.load(open(sols_file, "r", encoding="utf-8"))
                    corr = sum(
                        1
                        for s in sols
                        if s.get("is_correct", False) is True
                        and s.get("answer", "") != ""
                    )
                    tot = sum(
                        1
                        for s in sols
                        if s.get("is_correct", None) is not None
                        and s.get("answer", "") != ""
                    )
                    forced_answer_accuracies[chunk_idx] = (
                        (corr / tot) if tot > 0 else 0.0
                    )
                except Exception:
                    forced_answer_accuracies[chunk_idx] = 0.0

    for chunk_idx in valid_idx:
        cd = problem_dir / f"chunk_{chunk_idx}"
        sols_file = cd / "solutions.json"
        if not sols_file.exists():
            continue
        sols = json.load(open(sols_file, "r", encoding="utf-8"))
        corr = sum(
            1
            for s in sols
            if s.get("is_correct", False) is True and s.get("answer", "") != ""
        )
        tot = sum(
            1
            for s in sols
            if s.get("is_correct", None) is not None and s.get("answer", "") != ""
        )
        chunk_accuracies[chunk_idx] = (corr / tot) if tot > 0 else 0.0

        if sols:
            avg_tokens = np.mean([count_tokens(s.get("full_cot", "")) for s in sols])
            token_counts.append((chunk_idx, float(avg_tokens)))

            chunk_answers[chunk_idx] = defaultdict(int)
            for s in sols:
                a = normalize_answer(s.get("answer", ""))
                if a:
                    chunk_answers[chunk_idx][a] += 1

            info_list = []
            for s in sols:
                if s.get("answer", "") and s.get("answer", "") != "None":
                    info_list.append(
                        {
                            "chunk_removed": s.get("chunk_removed", ""),
                            "chunk_resampled": s.get("chunk_resampled", ""),
                            "full_cot": s.get("full_cot", ""),
                            "is_correct": s.get("is_correct", False),
                            "answer": normalize_answer(s.get("answer", "")),
                        }
                    )
            if info_list:
                chunk_info[chunk_idx] = info_list

    # embeddings (GPU if available)
    global embedding_model_cache
    if sentence_model not in embedding_model_cache:
        try:
            model = SentenceTransformer(sentence_model)
            try:
                import torch

                if torch.cuda.is_available():
                    model = model.to("cuda:0")
            except Exception:
                pass
            embedding_model_cache[sentence_model] = model
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sentence-transformers model '{sentence_model}': {e}"
            )
    emb_model = embedding_model_cache[sentence_model]

    # build embedding cache
    chunk_embedding_cache = {}
    uniq_chunks = set()
    for sols in chunk_info.values():
        for s in sols:
            if isinstance(s.get("chunk_removed", ""), str):
                uniq_chunks.add(s["chunk_removed"])
            if isinstance(s.get("chunk_resampled", ""), str):
                uniq_chunks.add(s["chunk_resampled"])
    uniq_list = list(uniq_chunks)
    print(f"Embeddings for {len(uniq_list)} unique chunks ...")
    bs = max(8, min(args.batch_size, 8192))
    for i in tqdm(range(0, len(uniq_list), bs), desc="Embedding"):
        batch = uniq_list[i : i + bs]
        vecs = emb_model.encode(
            batch, batch_size=min(1024, bs), show_progress_bar=False
        )
        for txt, v in zip(batch, vecs):
            chunk_embedding_cache[txt] = v

    # labeled chunks path
    labeled_path = problem_dir / "chunks_labeled.json"
    labeled_chunks = [{"chunk_idx": i} for i in valid_idx]

    # Use existing (and optionally refresh metrics/metadata)
    if labeled_path.exists() and not args.force_relabel:
        labeled_chunks = json.load(open(labeled_path, "r", encoding="utf-8"))
        # keep only valid indices
        labeled_chunks = [c for c in labeled_chunks if c.get("chunk_idx") in valid_idx]

        # summaries
        if args.force_metadata:
            for c in labeled_chunks:
                txt = c.get("chunk", "")
                c["summary"] = generate_chunk_summary(txt) if txt else "unknown action"

        if not args.use_existing_metrics:
            print(
                f"{problem_dir.name}: recomputing importance for {len(labeled_chunks)} chunks"
            )
            chunk_indices = [c["chunk_idx"] for c in labeled_chunks]
            args_obj = ImportanceArgs(
                use_absolute=use_absolute,
                forced_answer_dir=forced_answer_dir,
                similarity_threshold=similarity_threshold,
                use_similar_chunks=args.use_similar_chunks,
                use_abs_importance=args.absolute,
                top_chunks=args.top_chunks,
                use_prob_true=args.use_prob_true,
            )
            with mp.Pool(processes=args.num_processes) as pool:
                process_func = partial(
                    process_chunk_importance,
                    chunk_info=chunk_info,
                    chunk_embedding_cache=chunk_embedding_cache,
                    chunk_accuracies=chunk_accuracies,
                    args=args_obj,
                    problem_dir=problem_dir,
                    forced_answer_accuracies=forced_answer_accuracies,
                    chunk_answers=chunk_answers,
                )
                results = list(
                    tqdm(
                        pool.imap(process_func, chunk_indices),
                        total=len(chunk_indices),
                        desc="Processing chunks",
                    )
                )
            # update
            for idx, metrics in results:
                for c in labeled_chunks:
                    if c["chunk_idx"] == idx:
                        c.pop("absolute_importance_accuracy", None)
                        c.pop("absolute_importance_kl", None)
                        c.update(metrics)
                        break
            for c in labeled_chunks:
                c.update({"accuracy": chunk_accuracies.get(c["chunk_idx"], 0.0)})

            json.dump(
                labeled_chunks, open(labeled_path, "w", encoding="utf-8"), indent=2
            )
        else:
            if args.force_metadata:
                json.dump(
                    labeled_chunks, open(labeled_path, "w", encoding="utf-8"), indent=2
                )
            print(f"{problem_dir.name}: using existing metrics")

    else:
        # Need to label from scratch
        print(f"{problem_dir.name}: labeling {len(chunks)} chunks")
        try:
            dag_result = dag_label_chunks(problem.get("problem", ""), chunks)
        except Exception as e:
            print(f"DAG labeling error {problem_dir.name}: {e}")
            dag_result = {"chunks": {}}

        # parallel importance
        chunk_indices = list(range(len(valid_idx)))
        args_obj = ImportanceArgs(
            use_absolute=use_absolute,
            forced_answer_dir=forced_answer_dir,
            similarity_threshold=similarity_threshold,
            use_similar_chunks=args.use_similar_chunks,
            use_abs_importance=args.absolute,
            top_chunks=args.top_chunks,
            use_prob_true=args.use_prob_true,
        )
        with mp.Pool(processes=args.num_processes) as pool:
            process_func = partial(
                process_chunk_importance,
                chunk_info=chunk_info,
                chunk_embedding_cache=chunk_embedding_cache,
                chunk_accuracies=chunk_accuracies,
                args=args_obj,
                problem_dir=problem_dir,
                forced_answer_accuracies=forced_answer_accuracies,
                chunk_answers=chunk_answers,
            )
            results = list(
                tqdm(
                    pool.imap(process_func, chunk_indices),
                    total=len(chunk_indices),
                    desc="Processing chunks",
                )
            )

        labeled_chunks = []
        for i, chunk_idx in enumerate(valid_idx):
            txt = chunks[i]
            data = {"chunk": txt, "chunk_idx": chunk_idx}
            # summary
            try:
                data["summary"] = generate_chunk_summary(txt)
            except Exception:
                data["summary"] = "unknown action"
            # function tags & deps from DAG
            key = str(i)
            mapping = dag_result.get("chunks", {}).get(key, {})
            data["function_tags"] = mapping.get("function_tags", ["unknown"])
            data["depends_on"] = mapping.get("depends_on", [])
            # attach metrics
            for got_idx, metrics in results:
                if got_idx == i:
                    data.update(metrics)
                    break
            data["accuracy"] = chunk_accuracies.get(chunk_idx, 0.0)
            labeled_chunks.append(data)

        json.dump(labeled_chunks, open(labeled_path, "w", encoding="utf-8"), indent=2)

    # forced answer accuracies list if present
    forced_list = None
    if forced_answer_dir:
        forced_problem_dir = forced_answer_dir / problem_dir.name
        if forced_problem_dir.exists():
            forced_list = []
            for chunk_idx in valid_idx:
                cd = forced_problem_dir / f"chunk_{chunk_idx}"
                sols_file = cd / "solutions.json"
                if not sols_file.exists():
                    forced_list.append(0.0)
                    continue
                try:
                    sols = json.load(open(sols_file, "r", encoding="utf-8"))
                    corr = sum(1 for s in sols if s.get("is_correct", False) is True)
                    tot = sum(
                        1
                        for s in sols
                        if s.get("is_correct", None) is not None
                        and s.get("answer", "") != ""
                    )
                    forced_list.append((corr / tot) if tot > 0 else 0.0)
                except Exception:
                    forced_list.append(0.0)

    return {
        "problem_idx": problem_dir.name.split("_")[1],
        "problem_type": problem.get("type"),
        "problem_level": problem.get("level"),
        "base_accuracy": base_solution.get("is_correct", False),
        "num_chunks": len(chunks),
        "labeled_chunks": labeled_chunks,
        "token_counts": token_counts,
        "forced_answer_accuracies": forced_list,
    }


# ---------------------------
# process_chunk_importance wrapper
# ---------------------------
def process_chunk_importance(
    chunk_idx,
    chunk_info,
    chunk_embedding_cache,
    chunk_accuracies,
    args,
    problem_dir=None,
    forced_answer_accuracies=None,
    chunk_answers=None,
):
    metrics = {}
    cf_acc, diff_frac, overdet = calculate_counterfactual_importance_accuracy(
        chunk_idx, chunk_info, chunk_embedding_cache, chunk_accuracies, args
    )
    cf_kl = calculate_counterfactual_importance_kl(
        chunk_idx,
        chunk_info,
        chunk_embedding_cache,
        chunk_accuracies,
        chunk_answers,
        args,
    )
    metrics.update(
        {
            "counterfactual_importance_accuracy": cf_acc,
            "counterfactual_importance_kl": cf_kl,
            "different_trajectories_fraction": diff_frac,
            "overdeterminedness": overdet,
        }
    )
    rs_acc = calculate_resampling_importance_accuracy(chunk_idx, chunk_accuracies, args)
    rs_kl = (
        calculate_resampling_importance_kl(chunk_idx, chunk_info, problem_dir)
        if problem_dir
        else 0.0
    )
    metrics.update(
        {"resampling_importance_accuracy": rs_acc, "resampling_importance_kl": rs_kl}
    )
    if forced_answer_accuracies and args.forced_answer_dir:
        f_acc = calculate_forced_importance_accuracy(
            chunk_idx, forced_answer_accuracies, args
        )
        f_kl = (
            calculate_forced_importance_kl(
                chunk_idx, forced_answer_accuracies, problem_dir, args.forced_answer_dir
            )
            if problem_dir
            else 0.0
        )
        metrics.update(
            {"forced_importance_accuracy": f_acc, "forced_importance_kl": f_kl}
        )
    return chunk_idx, metrics


# ---------------------------------------------------------------------------------------------------------------------------------------


def generate_plots(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> pd.DataFrame:
    """
    Generate plots from the analysis results.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
        importance_metric: Importance metric to use for plotting and analysis

    Returns:
        DataFrame with category importance rankings
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Prepare data for plots
    all_chunks = []
    all_chunks_with_forced = []  # New list for chunks with forced importance

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        problem_type = result.get("problem_type", "Unknown")
        problem_level = result.get("problem_level", "Unknown")

        for chunk in result.get("labeled_chunks", []):
            # Format function tags for better display
            raw_tags = chunk.get("function_tags", [])
            formatted_tags = []

            for tag in raw_tags:
                if tag.lower() == "unknown":
                    continue  # Skip unknown tags

                # Format tag for better display (e.g., "planning_step" -> "Planning Step")
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                formatted_tags.append(formatted_tag)

            # If no valid tags after filtering, skip this chunk
            if not formatted_tags:
                continue

            chunk_data = {
                "problem_idx": problem_idx,
                "problem_type": problem_type,
                "problem_level": problem_level,
                "chunk_idx": chunk.get("chunk_idx"),
                "function_tags": formatted_tags,
                "counterfactual_importance_accuracy": chunk.get(
                    "counterfactual_importance_accuracy", 0.0
                ),
                "counterfactual_importance_kl": chunk.get(
                    "counterfactual_importance_kl", 0.0
                ),
                "resampling_importance_accuracy": chunk.get(
                    "resampling_importance_accuracy", 0.0
                ),
                "resampling_importance_kl": chunk.get("resampling_importance_kl", 0.0),
                "forced_importance_accuracy": chunk.get(
                    "forced_importance_accuracy", 0.0
                ),
                "forced_importance_kl": chunk.get("forced_importance_kl", 0.0),
                "chunk_length": len(chunk.get("chunk", "")),
            }
            all_chunks.append(chunk_data)

            # If forced importance is available, add to the forced importance list
            if "forced_importance_accuracy" in chunk:
                forced_chunk_data = chunk_data.copy()
                forced_chunk_data["forced_importance_accuracy"] = chunk.get(
                    "forced_importance_accuracy", 0.0
                )
                all_chunks_with_forced.append(forced_chunk_data)

    # Convert to DataFrame
    df_chunks = pd.DataFrame(all_chunks)

    # Create a DataFrame for chunks with forced importance if available
    df_chunks_forced = (
        pd.DataFrame(all_chunks_with_forced) if all_chunks_with_forced else None
    )

    # Explode function_tags to have one row per tag
    df_exploded = df_chunks.explode("function_tags")

    # If we have forced importance data, create special plots for it
    if df_chunks_forced is not None and not df_chunks_forced.empty:
        # Explode function_tags for forced importance DataFrame
        df_forced_exploded = df_chunks_forced.explode("function_tags")

        # Plot importance by function tag (category) using violin plot with means for forced importance
        plt.figure(figsize=(12, 8))
        # Calculate mean importance for each category to sort by
        # Convert to percentage for display
        df_forced_exploded["importance_pct"] = (
            df_forced_exploded["forced_importance_accuracy"] * 100
        )
        category_means = (
            df_forced_exploded.groupby("function_tags", observed=True)["importance_pct"]
            .mean()
            .sort_values(ascending=False)
        )
        # Reorder the data based on sorted categories
        df_forced_exploded_sorted = df_forced_exploded.copy()
        df_forced_exploded_sorted["function_tags"] = pd.Categorical(
            df_forced_exploded_sorted["function_tags"],
            categories=category_means.index,
            ordered=True,
        )
        # Create the sorted violin plot
        ax = sns.violinplot(
            x="function_tags",
            y="importance_pct",
            data=df_forced_exploded_sorted,
            inner="quartile",
            cut=0,
        )

        # Add mean markers
        means = df_forced_exploded_sorted.groupby("function_tags", observed=True)[
            "importance_pct"
        ].mean()
        for i, mean_val in enumerate(means[means.index]):
            ax.plot([i], [mean_val], "o", color="red", markersize=8)

        # Add a legend for the mean marker
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="Mean",
            )
        ]
        ax.legend(handles=legend_elements, loc="upper right")

        plt.title("Forced Importance by Category")
        plt.ylabel("Forced Importance (Accuracy Difference %)")
        plt.xlabel(None)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "forced_importance_by_category.png")
        plt.close()

        # Create a comparison plot that shows both normal importance and forced importance
        # Calculate category means for both metrics
        both_metrics = []

        # Metrics to compare
        metrics = [importance_metric, "forced_importance_accuracy"]
        metric_labels = ["Normal Importance", "Forced Importance"]

        # Find categories present in both datasets
        normal_categories = df_exploded["function_tags"].unique()
        forced_categories = df_forced_exploded["function_tags"].unique()
        common_categories = sorted(set(normal_categories) & set(forced_categories))

        for category in common_categories:
            # Get mean for normal importance
            normal_mean = (
                df_exploded[df_exploded["function_tags"] == category][
                    importance_metric
                ].mean()
                * 100
            )

            # Get mean for forced importance
            forced_mean = (
                df_forced_exploded[df_forced_exploded["function_tags"] == category][
                    "forced_importance_accuracy"
                ].mean()
                * 100
            )

            # Add to list
            both_metrics.append(
                {
                    "Category": category,
                    "Normal Importance": normal_mean,
                    "Forced Importance": forced_mean,
                }
            )

        # Convert to DataFrame
        df_both = pd.DataFrame(both_metrics)

        # Sort by normal importance
        df_both = df_both.sort_values("Normal Importance", ascending=False)

        # Create bar plot
        plt.figure(figsize=(15, 10))

        # Set bar width
        bar_width = 0.35

        # Set positions for bars
        r1 = np.arange(len(df_both))
        r2 = [x + bar_width for x in r1]

        # Create grouped bar chart
        plt.bar(
            r1,
            df_both["Normal Importance"],
            width=bar_width,
            label="Normal Importance",
            color="skyblue",
        )
        plt.bar(
            r2,
            df_both["Forced Importance"],
            width=bar_width,
            label="Forced Importance",
            color="lightcoral",
        )

        # Add labels and title
        plt.xlabel("Category", fontsize=12)
        plt.ylabel("Importance (%)", fontsize=12)
        plt.title("Comparison of Normal vs Forced Importance by Category", fontsize=14)

        # Set x-tick positions and labels
        plt.xticks(
            [r + bar_width / 2 for r in range(len(df_both))],
            df_both["Category"],
            rotation=45,
            ha="right",
        )

        # Add legend
        plt.legend()

        # Add grid
        # plt.grid(axis='y', linestyle='--', alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the comparison plot
        plt.savefig(plots_dir / "importance_comparison_by_category.png")
        plt.close()

        print(f"Forced importance plots saved to {plots_dir}")

    # 1. Plot importance by function tag (category) using violin plot with means
    plt.figure(figsize=(12, 8))
    # Calculate mean importance for each category to sort by
    # Convert to percentage for display
    df_exploded["importance_pct"] = df_exploded[importance_metric] * 100
    category_means = (
        df_exploded.groupby("function_tags", observed=True)["importance_pct"]
        .mean()
        .sort_values(ascending=False)
    )
    # Reorder the data based on sorted categories
    df_exploded_sorted = df_exploded.copy()
    df_exploded_sorted["function_tags"] = pd.Categorical(
        df_exploded_sorted["function_tags"],
        categories=category_means.index,
        ordered=True,
    )
    # Create the sorted violin plot
    ax = sns.violinplot(
        x="function_tags",
        y="importance_pct",
        data=df_exploded_sorted,
        inner="quartile",
        cut=0,
    )

    # Add mean markers
    means = df_exploded_sorted.groupby("function_tags", observed=True)[
        "importance_pct"
    ].mean()
    for i, mean_val in enumerate(means[means.index]):
        ax.plot([i], [mean_val], "o", color="red", markersize=8)

    # Add a legend for the mean marker
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markersize=8,
            label="Mean",
        )
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.title("Chunk Importance by Category")
    plt.ylabel("Importance (Accuracy Difference %)")
    plt.xlabel(None)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_category.png")
    plt.close()

    # 2. Plot average importance by problem level
    plt.figure(figsize=(10, 6))
    level_importance = (
        df_chunks.groupby("problem_level")[importance_metric].mean().reset_index()
    )
    sns.barplot(x="problem_level", y=importance_metric, data=level_importance)
    plt.title("Average Chunk Importance by Problem Level")
    plt.xlabel(None)
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_level.png")
    plt.close()

    # 3. Plot average importance by problem type
    plt.figure(figsize=(12, 8))
    type_importance = (
        df_chunks.groupby("problem_type")[importance_metric].mean().reset_index()
    )
    type_importance = type_importance.sort_values(importance_metric, ascending=False)
    sns.barplot(x="problem_type", y=importance_metric, data=type_importance)
    plt.title("Average Chunk Importance by Problem Type")
    plt.xlabel(None)
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_type.png")
    plt.close()

    # 4. Plot token counts by chunk index
    token_data = []
    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        for chunk_idx, token_count in result.get("token_counts", []):
            token_data.append(
                {
                    "problem_idx": problem_idx,
                    "chunk_idx": chunk_idx,
                    "token_count": token_count,
                }
            )

    df_tokens = pd.DataFrame(token_data)

    # Get function tag for each chunk
    chunk_tags = {}
    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")
            raw_tags = chunk.get("function_tags", [])

            # Format and filter tags
            formatted_tags = []
            for tag in raw_tags:
                if tag.lower() == "unknown":
                    continue
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                formatted_tags.append(formatted_tag)

            if formatted_tags:
                chunk_tags[(problem_idx, chunk_idx)] = formatted_tags[0]

    # Add function tag to token data
    df_tokens["function_tag"] = df_tokens.apply(
        lambda row: chunk_tags.get((row["problem_idx"], row["chunk_idx"]), "Other"),
        axis=1,
    )

    # Remove rows with no valid function tag
    df_tokens = df_tokens[df_tokens["function_tag"] != "Other"]

    if not df_tokens.empty:
        # Plot token counts by function tag (category)
        plt.figure(figsize=(12, 8))
        sns.boxplot(x="function_tag", y="token_count", data=df_tokens)
        plt.title("Token Count by Category")
        plt.ylabel("Token Count")
        plt.xlabel(None)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "token_count_by_category.png")
        plt.close()

    # 5. Plot distribution of function tags (categories)
    plt.figure(figsize=(12, 8))
    tag_counts = df_exploded["function_tags"].value_counts()
    sns.barplot(x=tag_counts.index, y=tag_counts.values)
    plt.title("Distribution of Categories")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "category_distribution.png")
    plt.close()

    # 6. Plot importance vs. chunk position
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x="chunk_idx", y=importance_metric, data=df_chunks)
    plt.title("Chunk Importance by Position")
    plt.xlabel("Chunk Index")
    plt.ylabel("Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "importance_by_position.png")
    plt.close()

    # 7. Calculate and plot average importance by function tag (category) with error bars
    tag_importance = (
        df_exploded.groupby("function_tags")
        .agg({importance_metric: ["mean", "std", "count"]})
        .reset_index()
    )
    tag_importance.columns = ["categories", "mean", "std", "count"]
    tag_importance = tag_importance.sort_values("mean", ascending=False)

    # Convert to percentages for display
    tag_importance["mean_pct"] = tag_importance["mean"] * 100
    tag_importance["std_pct"] = tag_importance["std"] * 100

    # Calculate standard error (std/sqrt(n)) instead of using raw standard deviation
    tag_importance["se_pct"] = tag_importance["std_pct"] / np.sqrt(
        tag_importance["count"]
    )

    plt.figure(figsize=(12, 8))
    plt.errorbar(
        x=range(len(tag_importance)),
        y=tag_importance["mean_pct"],
        yerr=tag_importance["se_pct"],
        fmt="o",
        capsize=5,
    )
    plt.xticks(
        range(len(tag_importance)),
        tag_importance["categories"],
        rotation=45,
        ha="right",
    )
    plt.title("Average Importance by Category")
    plt.xlabel(None)
    plt.ylabel("Average Importance (Accuracy Difference %)")
    plt.tight_layout()
    plt.savefig(plots_dir / "avg_importance_by_category.png")
    plt.close()

    print(f"Generated plots in {plots_dir}")

    # Add the new analysis of top steps by category
    for top_n in [1, 3, 5, 10, 20, 30]:
        analyze_top_steps_by_category(results, output_dir, top_n=top_n, use_abs=False)

    # Add the new analysis of steps with high z-score by category
    for z_threshold in [1.5, 2, 2.5, 3]:
        analyze_high_zscore_steps_by_category(
            results, output_dir, z_threshold=z_threshold, use_abs=False
        )

    # Return the category importance ranking
    return tag_importance


def analyze_chunk_variance(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze variance in chunk importance scores to identify potential reasoning forks.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print(
        "Analyzing chunk variance within problems to identify potential reasoning forks..."
    )

    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)

    # Collect all chunks across problems
    all_chunks = []
    problem_chunks = {}

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        problem_chunks[problem_idx] = []

        for chunk in result.get("labeled_chunks", []):
            chunk_data = {
                "problem_idx": problem_idx,
                "chunk_idx": chunk.get("chunk_idx"),
                "chunk_text": chunk.get("chunk", ""),
                "function_tags": chunk.get("function_tags", []),
                "counterfactual_importance_accuracy": chunk.get(
                    "counterfactual_importance_accuracy", 0.0
                ),
                "counterfactual_importance_kl": chunk.get(
                    "counterfactual_importance_kl", 0.0
                ),
                "resampling_importance_accuracy": chunk.get(
                    "resampling_importance_accuracy", 0.0
                ),
                "resampling_importance_kl": chunk.get("resampling_importance_kl", 0.0),
                "forced_importance_accuracy": chunk.get(
                    "forced_importance_accuracy", 0.0
                ),
                "forced_importance_kl": chunk.get("forced_importance_kl", 0.0),
            }
            all_chunks.append(chunk_data)
            problem_chunks[problem_idx].append(chunk_data)

    # Calculate variance in importance within each problem
    problem_variances = {}
    high_variance_problems = []

    for problem_idx, chunks in problem_chunks.items():
        if len(chunks) < 3:  # Need at least 3 chunks for meaningful variance
            continue

        # Calculate importance variance
        importance_values = [chunk[importance_metric] for chunk in chunks]
        variance = np.var(importance_values)

        problem_variances[problem_idx] = {
            "variance": variance,
            "chunks": chunks,
            "importance_values": importance_values,
        }

        # Track problems with high variance
        high_variance_problems.append((problem_idx, variance))

    # Sort problems by variance
    high_variance_problems.sort(key=lambda x: x[1], reverse=True)

    # Save results
    with open(variance_dir / "chunk_variance.txt", "w", encoding="utf-8") as f:
        f.write(
            "Problems with highest variance in chunk importance (potential reasoning forks):\n\n"
        )

        for problem_idx, variance in high_variance_problems[:20]:  # Top 20 problems
            f.write(f"Problem {problem_idx}: Variance = {variance:.6f}\n")

            # Get chunks for this problem
            chunks = problem_chunks[problem_idx]

            # Sort chunks by importance
            sorted_chunks = sorted(
                chunks, key=lambda x: x[importance_metric], reverse=True
            )

            # Write chunk information
            f.write("  Chunks by importance:\n")
            for i, chunk in enumerate(sorted_chunks):
                chunk_idx = chunk["chunk_idx"]
                importance = chunk[importance_metric]
                tags = ", ".join(chunk["function_tags"])

                # Truncate chunk text for display
                chunk_text = chunk["chunk_text"]
                if len(chunk_text) > 50:
                    chunk_text = chunk_text[:47] + "..."

                f.write(
                    f"    {i + 1}. Chunk {chunk_idx}: {importance:.4f} - {tags} - '{chunk_text}'\n"
                )

            # Identify potential reasoning forks (clusters of important chunks)
            f.write("  Potential reasoning forks:\n")

            # Sort chunks by index to maintain sequence
            sequence_chunks = sorted(chunks, key=lambda x: x["chunk_idx"])

            # Find clusters of important chunks
            clusters = []
            current_cluster = []
            avg_importance = np.mean([c[importance_metric] for c in chunks])

            for chunk in sequence_chunks:
                if chunk[importance_metric] > avg_importance:
                    if (
                        not current_cluster
                        or chunk["chunk_idx"] - current_cluster[-1]["chunk_idx"] <= 2
                    ):
                        current_cluster.append(chunk)
                    else:
                        if len(current_cluster) >= 2:  # At least 2 chunks in a cluster
                            clusters.append(current_cluster)
                        current_cluster = [chunk]

            if len(current_cluster) >= 2:
                clusters.append(current_cluster)

            # Write clusters
            for i, cluster in enumerate(clusters):
                start_idx = cluster[0]["chunk_idx"]
                end_idx = cluster[-1]["chunk_idx"]
                f.write(f"    Fork {i + 1}: Chunks {start_idx}-{end_idx}\n")

                # Combine chunk text
                combined_text = " ".join([c["chunk_text"] for c in cluster])
                if len(combined_text) > 100:
                    combined_text = combined_text[:97] + "..."

                f.write(f"      Text: '{combined_text}'\n")

                # List tags
                all_tags = set()
                for chunk in cluster:
                    all_tags.update(chunk["function_tags"])
                f.write(f"      Tags: {', '.join(all_tags)}\n")

            f.write("\n")

    # Create visualization of variance distribution
    plt.figure(figsize=(12, 8))
    variances = [v for _, v in high_variance_problems]
    plt.hist(variances, bins=20)
    plt.xlabel("Variance in Chunk Importance")
    plt.ylabel("Number of Problems")
    plt.title("Distribution of Variance in Chunk Importance Across Problems")
    plt.tight_layout()
    plt.savefig(variance_dir / "chunk_variance_distribution.png")
    plt.close()

    # Create visualization of top high-variance problems
    plt.figure(figsize=(15, 10))
    top_problems = high_variance_problems[:20]
    problem_ids = [str(p[0]) for p in top_problems]
    problem_variances = [p[1] for p in top_problems]

    plt.bar(range(len(problem_ids)), problem_variances)
    plt.xticks(range(len(problem_ids)), problem_ids, rotation=45)
    plt.xlabel("Problem ID")
    plt.ylabel("Variance in Chunk Importance")
    plt.title("Top 20 Problems with Highest Variance in Chunk Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "top_variance_problems.png")
    plt.close()

    print(f"Chunk variance analysis saved to {variance_dir}")


def analyze_function_tag_variance(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze variance in importance scores grouped by function tags.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print("Analyzing variance in importance across function tags...")

    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)

    # Collect chunks by function tag with all metrics
    tag_chunks = {}

    for result in results:
        if not result:
            continue

        for chunk in result.get("labeled_chunks", []):
            for tag in chunk.get("function_tags", []):
                if tag.lower() == "unknown":
                    continue

                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))

                if formatted_tag not in tag_chunks:
                    tag_chunks[formatted_tag] = []

                tag_chunks[formatted_tag].append(
                    {
                        "problem_idx": result["problem_idx"],
                        "chunk_idx": chunk.get("chunk_idx"),
                        "counterfactual_importance_accuracy": chunk.get(
                            "counterfactual_importance_accuracy", 0.0
                        ),
                        "counterfactual_importance_kl": chunk.get(
                            "counterfactual_importance_kl", 0.0
                        ),
                        "resampling_importance_accuracy": chunk.get(
                            "resampling_importance_accuracy", 0.0
                        ),
                        "resampling_importance_kl": chunk.get(
                            "resampling_importance_kl", 0.0
                        ),
                        "forced_importance_accuracy": chunk.get(
                            "forced_importance_accuracy", 0.0
                        ),
                        "forced_importance_kl": chunk.get("forced_importance_kl", 0.0),
                    }
                )

    # Calculate variance for each tag using the BASE_IMPORTANCE_METRIC
    tag_variances = {}

    for tag, chunks in tag_chunks.items():
        if len(chunks) < 5:  # Need at least 5 chunks for meaningful variance
            continue

        importance_values = [chunk[importance_metric] for chunk in chunks]
        variance = np.var(importance_values)
        mean = np.mean(importance_values)
        count = len(chunks)

        tag_variances[tag] = {
            "variance": variance,
            "mean": mean,
            "count": count,
            "coefficient_of_variation": variance / mean if mean != 0 else 0,
        }

    # Sort tags by variance
    sorted_tags = sorted(
        tag_variances.items(), key=lambda x: x[1]["variance"], reverse=True
    )

    # Save results
    with open(variance_dir / "function_tag_variance.txt", "w", encoding="utf-8") as f:
        f.write("Function tags with highest variance in importance:\n\n")

        for tag, stats in sorted_tags:
            variance = stats["variance"]
            mean = stats["mean"]
            count = stats["count"]
            cv = stats["coefficient_of_variation"]

            f.write(f"{tag}:\n")
            f.write(f"  Variance: {variance:.6f}\n")
            f.write(f"  Mean: {mean:.6f}\n")
            f.write(f"  Count: {count}\n")
            f.write(f"  Coefficient of Variation: {cv:.6f}\n")
            f.write("\n")

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Plot top 15 tags by variance
    top_tags = sorted_tags[:15]
    tags = [t[0] for t in top_tags]
    variances = [t[1]["variance"] for t in top_tags]

    plt.bar(range(len(tags)), variances)
    plt.xticks(range(len(tags)), tags, rotation=45, ha="right")
    plt.xlabel("Function Tag")
    plt.ylabel("Variance in Importance")
    plt.title("Top 15 Function Tags by Variance in Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "function_tag_variance.png")
    plt.close()

    # Plot coefficient of variation (normalized variance)
    plt.figure(figsize=(15, 10))

    # Sort by coefficient of variation
    sorted_by_cv = sorted(
        tag_variances.items(),
        key=lambda x: x[1]["coefficient_of_variation"],
        reverse=True,
    )
    top_cv_tags = sorted_by_cv[:15]

    cv_tags = [t[0] for t in top_cv_tags]
    cvs = [t[1]["coefficient_of_variation"] for t in top_cv_tags]

    plt.bar(range(len(cv_tags)), cvs)
    plt.xticks(range(len(cv_tags)), cv_tags, rotation=45, ha="right")
    plt.xlabel("Function Tag")
    plt.ylabel("Coefficient of Variation (σ/μ)")
    plt.title("Top 15 Function Tags by Coefficient of Variation in Importance")
    plt.tight_layout()
    plt.savefig(variance_dir / "function_tag_cv.png")
    plt.close()

    print(f"Function tag variance analysis saved to {variance_dir}")


def analyze_within_problem_variance(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze variance in importance scores within problems to identify patterns across problem types and levels.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print("Analyzing within-problem variance to identify potential reasoning forks...")

    variance_dir = output_dir / "variance_analysis"
    variance_dir.mkdir(exist_ok=True, parents=True)

    # Analyze each problem for high-variance chunks (potential reasoning forks)
    high_variance_problems = []

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]
        chunks = result.get("labeled_chunks", [])

        if len(chunks) < 3:  # Need at least 3 chunks for meaningful analysis
            continue

        # Calculate importance values and their variance
        importance_values = [chunk.get(importance_metric, 0.0) for chunk in chunks]
        mean_importance = np.mean(importance_values)
        variance = np.var(importance_values)

        # Identify chunks with significantly higher or lower importance than average
        # These could represent "fork reasoning steps"
        potential_forks = []

        for i, chunk in enumerate(chunks):
            importance = chunk.get(importance_metric, 0.0)
            z_score = (importance - mean_importance) / (
                np.std(importance_values) if np.std(importance_values) > 0 else 1
            )

            # Consider chunks with importance significantly different from mean as potential forks
            if abs(z_score) > 1.5:  # Threshold can be adjusted
                potential_forks.append(
                    {
                        "chunk_idx": chunk.get("chunk_idx"),
                        "chunk_text": chunk.get("chunk", ""),
                        "counterfactual_importance_accuracy": chunk.get(
                            "counterfactual_importance_accuracy", 0.0
                        ),
                        "counterfactual_importance_kl": chunk.get(
                            "counterfactual_importance_kl", 0.0
                        ),
                        "resampling_importance_accuracy": chunk.get(
                            "resampling_importance_accuracy", 0.0
                        ),
                        "resampling_importance_kl": chunk.get(
                            "resampling_importance_kl", 0.0
                        ),
                        "forced_importance_accuracy": chunk.get(
                            "forced_importance_accuracy", 0.0
                        ),
                        "forced_importance_kl": chunk.get("forced_importance_kl", 0.0),
                        "z_score": z_score,
                        "function_tags": chunk.get("function_tags", []),
                    }
                )

        if potential_forks:
            high_variance_problems.append(
                {
                    "problem_idx": problem_idx,
                    "variance": variance,
                    "mean_importance": mean_importance,
                    "potential_forks": potential_forks,
                }
            )

    # Sort problems by variance
    high_variance_problems.sort(key=lambda x: x["variance"], reverse=True)

    # Save results
    with open(variance_dir / "within_problem_variance.txt", "w", encoding="utf-8") as f:
        f.write(
            "Problems with high variance in chunk importance (potential reasoning forks):\n\n"
        )

        for problem in high_variance_problems:
            problem_idx = problem["problem_idx"]
            variance = problem["variance"]
            mean_importance = problem["mean_importance"]

            f.write(f"Problem {problem_idx}:\n")
            f.write(f"  Overall variance: {variance:.6f}\n")
            f.write(f"  Mean importance: {mean_importance:.6f}\n")
            f.write("  Potential fork reasoning steps:\n")

            # Sort potential forks by absolute z-score
            sorted_forks = sorted(
                problem["potential_forks"],
                key=lambda x: abs(x["z_score"]),
                reverse=True,
            )

            for fork in sorted_forks:
                chunk_idx = fork["chunk_idx"]
                importance = fork[importance_metric]
                z_score = fork["z_score"]
                tags = (
                    ", ".join(fork["function_tags"])
                    if fork["function_tags"]
                    else "No tags"
                )

                f.write(f"    Chunk {chunk_idx}:\n")
                f.write(
                    f"      Importance: {importance:.6f} (z-score: {z_score:.2f})\n"
                )
                f.write(f"      Function tags: {tags}\n")
                f.write(
                    f"      Text: {fork['chunk_text'][:100]}{'...' if len(fork['chunk_text']) > 100 else ''}\n\n"
                )

    # Create visualization of fork distribution
    if high_variance_problems:
        plt.figure(figsize=(12, 8))

        # Collect data for visualization
        problem_indices = [
            p["problem_idx"] for p in high_variance_problems[:15]
        ]  # Top 15 problems
        variances = [p["variance"] for p in high_variance_problems[:15]]
        fork_counts = [len(p["potential_forks"]) for p in high_variance_problems[:15]]

        # Create bar chart
        plt.bar(range(len(problem_indices)), variances)

        # Add fork count as text on bars
        for i, count in enumerate(fork_counts):
            plt.text(
                i,
                variances[i] * 0.5,
                f"{count} forks",
                ha="center",
                color="white",
                fontweight="bold",
            )

        plt.xticks(
            range(len(problem_indices)),
            [f"Problem {idx}" for idx in problem_indices],
            rotation=45,
            ha="right",
        )
        plt.xlabel("Problem")
        plt.ylabel("Variance in Chunk Importance")
        plt.title(
            "Problems with Highest Variance in Chunk Importance (Potential Reasoning Forks)"
        )
        plt.tight_layout()
        plt.savefig(variance_dir / "within_problem_variance.png")
        plt.close()

    print(f"Within-problem variance analysis saved to {variance_dir}")


def plot_chunk_accuracy_by_position(
    results: List[Dict],
    output_dir: Path,
    rollout_type: str = "correct",
    max_chunks_to_show: Optional[int] = None,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Plot chunk accuracy by position to identify trends in where the model makes errors.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save plots
        rollout_type: Type of rollouts being analyzed
        max_chunks_to_show: Maximum number of chunks to include in the plots
        importance_metric: Importance metric to use for the analysis
    """
    print("Plotting chunk accuracy by position...")

    # Create explore directory
    explore_dir = output_dir / "explore"
    explore_dir.mkdir(exist_ok=True, parents=True)

    # Create problems directory for individual plots
    problems_dir = explore_dir / "problems"
    problems_dir.mkdir(exist_ok=True, parents=True)

    # Collect data for all chunks across problems
    chunk_data = []
    forced_chunk_data = []  # New list for forced answer data
    forced_importance_data = []  # For storing forced importance data

    for result in results:
        if not result:
            continue

        problem_idx = result["problem_idx"]

        # Get the solutions for each chunk
        for chunk in result.get("labeled_chunks", []):
            chunk_idx = chunk.get("chunk_idx")

            # Only include the first 100 chunks
            if max_chunks_to_show is not None and chunk_idx > max_chunks_to_show:
                continue

            # Get the solutions for this chunk
            accuracy = chunk.get("accuracy", 0.0)

            # Get forced importance if available
            forced_importance = chunk.get("forced_importance_accuracy", None)

            # Get the first function tag if available
            function_tags = chunk.get("function_tags", [])
            first_tag = ""
            if (
                function_tags
                and isinstance(function_tags, list)
                and len(function_tags) > 0
            ):
                # Get first tag and convert to initials
                tag = function_tags[0]
                if isinstance(tag, str):
                    # Convert tag like "planning_step" to "PS"
                    words = tag.split("_")
                    first_tag = "".join(word[0].upper() for word in words if word)

            chunk_data.append(
                {
                    "problem_idx": problem_idx,
                    "chunk_idx": chunk_idx,
                    "accuracy": accuracy,
                    "tag": first_tag,
                }
            )

            # Add forced importance if available
            if forced_importance is not None:
                forced_importance_data.append(
                    {
                        "problem_idx": problem_idx,
                        "chunk_idx": chunk_idx,
                        "importance": forced_importance,
                        "tag": first_tag,
                    }
                )

            # Add forced answer accuracy if available
            if (
                "forced_answer_accuracies" in result
                and result["forced_answer_accuracies"] is not None
                and len(result["forced_answer_accuracies"]) > chunk_idx
            ):
                forced_accuracy = result["forced_answer_accuracies"][chunk_idx]
                forced_chunk_data.append(
                    {
                        "problem_idx": problem_idx,
                        "chunk_idx": chunk_idx,
                        "accuracy": forced_accuracy,
                        "tag": first_tag,
                    }
                )

    if not chunk_data:
        print("No chunk data available for plotting.")
        return

    # Convert to DataFrame
    df_chunks = pd.DataFrame(chunk_data)
    df_forced = pd.DataFrame(forced_chunk_data) if forced_chunk_data else None
    df_forced_importance = (
        pd.DataFrame(forced_importance_data) if forced_importance_data else None
    )

    # Get unique problem indices
    problem_indices = df_chunks["problem_idx"].unique()

    # Create a colormap for the problems (other options: plasma, inferno, magma, cividis)
    import matplotlib.cm as cm

    colors = cm.viridis(np.linspace(0, 0.75, len(problem_indices)))
    color_map = dict(zip(sorted(problem_indices), colors))

    # Create a figure for forced importance data if available
    if df_forced_importance is not None and not df_forced_importance.empty:
        plt.figure(figsize=(15, 10))

        # Plot each problem with a unique color
        for problem_idx in problem_indices:
            problem_data = df_forced_importance[
                df_forced_importance["problem_idx"] == problem_idx
            ]

            # Skip if no data for this problem
            if problem_data.empty:
                continue

            # Sort by chunk index
            problem_data = problem_data.sort_values("chunk_idx")

            # Convert to numpy arrays for plotting to avoid pandas indexing issues
            chunk_indices = problem_data["chunk_idx"].to_numpy()
            importances = problem_data["importance"].to_numpy()

            # Plot with clear label
            plt.plot(
                chunk_indices,
                importances,
                marker="o",
                linestyle="-",
                color=color_map[problem_idx],
                alpha=0.7,
                label=f"Problem {problem_idx}",
            )

        # Calculate and plot the average across all problems
        avg_by_chunk = (
            df_forced_importance.groupby("chunk_idx")["importance"]
            .agg(["mean"])
            .reset_index()
        )

        plt.plot(
            avg_by_chunk["chunk_idx"],
            avg_by_chunk["mean"],
            marker=".",
            markersize=4,
            linestyle="-",
            linewidth=2,
            color="black",
            alpha=0.8,
            label="Average",
        )

        # Add labels and title
        plt.xlabel("Sentence Index")
        plt.ylabel("Forced Importance (Difference in Accuracy)")
        plt.title("Forced Importance by Sentence Index (First 100 Sentences)")

        # Set x-axis limits to focus on first 100 chunks
        plt.xlim(-3, 100 if max_chunks_to_show is None else max_chunks_to_show)

        # Add grid
        # plt.grid(True, alpha=0.3)

        # Add legend
        plt.legend(loc="upper right", ncol=2)

        # Save the forced importance plot
        plt.tight_layout()
        plt.savefig(explore_dir / "forced_importance_by_position.png")
        plt.close()
        print(
            f"Forced importance plot saved to {explore_dir / 'forced_importance_by_position.png'}"
        )

    # Create a single plot focusing on first 100 chunks
    plt.figure(figsize=(15, 10))

    # Plot each problem with a unique color
    for problem_idx in problem_indices:
        problem_data = df_chunks[df_chunks["problem_idx"] == problem_idx]

        # Sort by chunk index
        problem_data = problem_data.sort_values("chunk_idx")

        # Convert to numpy arrays for plotting to avoid pandas indexing issues
        chunk_indices = problem_data["chunk_idx"].to_numpy()
        accuracies = problem_data["accuracy"].to_numpy()
        tags = problem_data["tag"].to_numpy()

        # Plot with clear label
        line = plt.plot(
            chunk_indices,
            accuracies,
            marker="o",
            linestyle="-",
            color=color_map[problem_idx],
            alpha=0.7,
            label=f"Problem {problem_idx}",
        )[0]

        # Identify accuracy extrema (both minima and maxima)
        # Convert to numpy arrays for easier manipulation
        for i in range(1, len(chunk_indices) - 1):
            # For correct rollouts, annotate minima (lower than both neighbors)
            is_minimum = (
                accuracies[i] < accuracies[i - 1] and accuracies[i] < accuracies[i + 1]
            )
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = (
                accuracies[i] > accuracies[i - 1] and accuracies[i] > accuracies[i + 1]
            )

            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "correct" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = (
                        -15 if (rollout_type == "correct" and is_minimum) else 7.5
                    )

                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], accuracies[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha="center",
                        fontsize=8,
                        color=line.get_color(),
                        alpha=0.9,
                        weight="bold",
                    )

        # Plot forced answer accuracy if available
        if df_forced is not None:
            forced_problem_data = df_forced[df_forced["problem_idx"] == problem_idx]
            if not forced_problem_data.empty:
                forced_problem_data = forced_problem_data.sort_values("chunk_idx")
                plt.plot(
                    forced_problem_data["chunk_idx"],
                    forced_problem_data["accuracy"],
                    marker="x",
                    linestyle="--",
                    color=color_map[problem_idx],
                    alpha=0.5,
                    label=f"Problem {problem_idx} Forced Answer",
                )

    # Calculate and plot the average across all problems
    avg_by_chunk = (
        df_chunks.groupby("chunk_idx")["accuracy"].agg(["mean"]).reset_index()
    )

    avg_by_chunk_idx = avg_by_chunk["chunk_idx"].to_numpy()
    avg_by_chunk_mean = avg_by_chunk["mean"].to_numpy()

    # Plot average without error bars, in gray and thinner
    plt.plot(
        avg_by_chunk_idx,
        avg_by_chunk_mean,
        marker=".",
        markersize=4,
        linestyle="-",
        linewidth=1,
        color="gray",
        alpha=0.5,
        label="Average",
    )

    # Plot average for forced answer if available
    if df_forced is not None:
        forced_avg_by_chunk = (
            df_forced.groupby("chunk_idx")["accuracy"].agg(["mean"]).reset_index()
        )
        plt.plot(
            forced_avg_by_chunk["chunk_idx"],
            forced_avg_by_chunk["mean"],
            marker=".",
            markersize=4,
            linestyle="--",
            linewidth=1,
            color="black",
            alpha=0.5,
            label="Average Forced Answer",
        )

    # Add labels and title
    plt.xlabel("Sentence index")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by position (first 100 sentences)")

    # Set x-axis limits to focus on first 100 chunks
    plt.xlim(-3, 300 if max_chunks_to_show is None else max_chunks_to_show)

    # Set y-axis limits
    plt.ylim(-0.1, 1.1)

    # Add grid
    # plt.grid(True, alpha=0.3)

    # If not too many problems, include all in the main legend
    plt.legend(
        loc="lower right" if rollout_type == "correct" else "upper right", ncol=2
    )

    # Save the main plot
    plt.tight_layout()
    plt.savefig(explore_dir / "chunk_accuracy_by_position.png")
    plt.close()

    # Create individual plots for each problem
    print("Creating individual problem plots...")
    for problem_idx in problem_indices:
        problem_data = df_chunks[df_chunks["problem_idx"] == problem_idx]

        # Sort by chunk index
        problem_data = problem_data.sort_values("chunk_idx")

        if len(problem_data) == 0:
            continue

        # Create a new figure for this problem
        plt.figure(figsize=(10, 6))

        # Get the color for this problem
        color = color_map[problem_idx]

        problem_data_idx = problem_data["chunk_idx"].to_numpy()
        problem_data_accuracy = problem_data["accuracy"].to_numpy()

        # Plot the problem data
        line = plt.plot(
            problem_data_idx,
            problem_data_accuracy,
            marker="o",
            linestyle="-",
            color=color,
            label=f"Resampling",
        )[0]

        # Identify accuracy extrema (minima for correct, maxima for incorrect)
        # Convert to numpy arrays for easier manipulation
        chunk_indices = problem_data["chunk_idx"].values
        accuracies = problem_data["accuracy"].values
        tags = problem_data["tag"].values
        # print(f"[DEBUG] Accuracies: {accuracies}")

        # Add function tag labels for both minima and maxima
        for i in range(1, len(chunk_indices) - 1):
            # For correct rollouts, annotate minima (lower than both neighbors)
            is_minimum = (
                accuracies[i] < accuracies[i - 1] and accuracies[i] < accuracies[i + 1]
            )
            # For all rollouts, annotate maxima (higher than both neighbors)
            is_maximum = (
                accuracies[i] > accuracies[i - 1] and accuracies[i] > accuracies[i + 1]
            )

            if tags[i]:  # Only add if there's a tag
                if (rollout_type == "correct" and is_minimum) or is_maximum:
                    # Position below for minima, above for maxima
                    y_offset = (
                        -15 if (rollout_type == "correct" and is_minimum) else 7.5
                    )

                    plt.annotate(
                        tags[i],
                        (chunk_indices[i], accuracies[i]),
                        textcoords="offset points",
                        xytext=(0, y_offset),
                        ha="center",
                        fontsize=8,
                        color=color,
                        alpha=0.9,
                        weight="bold",
                    )

        # Plot forced answer accuracy if available
        if df_forced is not None:
            forced_problem_data = df_forced[df_forced["problem_idx"] == problem_idx]
            if not forced_problem_data.empty:
                forced_problem_data = forced_problem_data.sort_values("chunk_idx")
                forced_problem_data_idx = forced_problem_data["chunk_idx"].to_numpy()
                forced_problem_data_accuracy = forced_problem_data[
                    "accuracy"
                ].to_numpy()

                plt.plot(
                    forced_problem_data_idx,
                    forced_problem_data_accuracy,
                    marker=".",
                    linestyle="--",
                    color=color,
                    alpha=0.7,
                    label=f"Forced answer",
                )

        # Remove the forced importance plot with twin axis

        # Add labels and title
        plt.xlabel("Sentence index")
        plt.ylabel("Accuracy")
        suffix = (
            f"\n(R1-Distill-Llama-8B)"
            if "llama-8b" in args.correct_rollouts_dir
            else ""
        )
        plt.title(f"Problem {problem_idx}: Sentence accuracy by position{suffix}")

        # Set x-axis limits to focus on first 100 chunks
        plt.xlim(-3, 100 if max_chunks_to_show is None else max_chunks_to_show)

        # Set y-axis limits for accuracy
        plt.ylim(-0.1, 1.1)

        # Add grid
        # plt.grid(True, alpha=0.3)

        # Add legend in the correct position based on rollout type
        plt.legend(loc="lower right" if rollout_type == "correct" else "upper right")

        # Save the plot
        plt.tight_layout()
        plt.savefig(problems_dir / f"problem_{problem_idx}_accuracy.png")
        plt.close()

    print(f"Chunk accuracy plots saved to {explore_dir}")


def analyze_dag_token_frequencies(dag_dir: Path, output_dir: Path) -> None:
    """
    Analyze token frequencies from DAG-improved chunks.

    Args:
        dag_dir: Directory containing DAG-improved chunks
        output_dir: Directory to save analysis results
    """
    print("Analyzing token frequencies from DAG-improved chunks...")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Collect all chunks by category
    category_chunks = {}

    # Find all problem directories
    problem_dirs = sorted(
        [d for d in dag_dir.iterdir() if d.is_dir() and d.name.startswith("problem_")]
    )

    for problem_dir in problem_dirs:
        # Find seed directories
        seed_dirs = sorted(
            [
                d
                for d in problem_dir.iterdir()
                if d.is_dir() and d.name.startswith("seed_")
            ]
        )

        for seed_dir in seed_dirs:
            # Look for chunks_dag_improved.json
            chunks_file = seed_dir / "chunks_dag_improved.json"

            if not chunks_file.exists():
                continue

            # Load chunks
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks_data = json.load(f)

            # Process each chunk
            for chunk in chunks_data:
                # Get function tags (categories)
                function_tags = chunk.get("function_tags", [])

                # Skip chunks with no tags
                if not function_tags:
                    function_tags = chunk.get("categories", [])
                    if not function_tags:
                        continue

                # Get chunk text
                chunk_text = chunk.get("chunk", "")

                # Skip empty chunks
                if not chunk_text:
                    chunk_text = chunk.get("text", "")
                    if not chunk_text:
                        continue

                # Add chunk to each of its categories
                for tag in function_tags:
                    # Format tag for better display
                    if isinstance(tag, str):
                        formatted_tag = " ".join(
                            word.capitalize() for word in tag.split("_")
                        )
                    else:
                        continue  # Skip non-string tags

                    # Skip unknown category
                    if formatted_tag.lower() == "unknown":
                        continue

                    if formatted_tag not in category_chunks:
                        category_chunks[formatted_tag] = []

                    category_chunks[formatted_tag].append(chunk_text)

    # Skip if no categories found
    if not category_chunks:
        print("No categories found for token frequency analysis")
        return

    print(
        f"Found {len(category_chunks)} categories with {sum(len(chunks) for chunks in category_chunks.values())} total chunks"
    )

    # Generate plots for unigrams, bigrams, and trigrams
    for n in [1, 2, 3]:
        print(f"Analyzing {n}-gram frequencies...")

        # Tokenize chunks and count frequencies
        category_ngram_frequencies = {}

        for category, chunks in category_chunks.items():
            # Tokenize all chunks
            all_tokens = []
            for chunk in chunks:
                # Simple tokenization by splitting on whitespace and punctuation
                tokens = re.findall(r"\\b\\w+\\b", chunk.lower())

                # Filter out stopwords and numbers
                filtered_tokens = [
                    token
                    for token in tokens
                    if token not in stopwords and not token.isdigit() and len(token) > 1
                ]

                # Generate n-grams
                ngrams = []
                for i in range(len(filtered_tokens) - (n - 1)):
                    ngram = " ".join(filtered_tokens[i : i + n])
                    ngrams.append(ngram)

                all_tokens.extend(ngrams)

            # Count token frequencies
            token_counts = Counter(all_tokens)

            # Calculate percentages
            total_chunks = len(chunks)
            token_percentages = {}

            for token, count in token_counts.items():
                # Count in how many chunks this n-gram appears
                chunks_with_token = sum(
                    1
                    for chunk in chunks
                    if re.search(r"\\b" + re.escape(token) + r"\\b", chunk.lower())
                )
                percentage = (chunks_with_token / total_chunks) * 100
                token_percentages[token] = percentage

            # Store results
            category_ngram_frequencies[category] = token_percentages

        # Create a master plot with subplots for each category
        num_categories = len(category_ngram_frequencies)

        # Calculate grid dimensions
        cols = min(3, num_categories)
        rows = (num_categories + cols - 1) // cols  # Ceiling division

        # Create figure
        fig = plt.figure(figsize=(20, 5 * rows))

        # Create subplots
        for i, (category, token_percentages) in enumerate(
            category_ngram_frequencies.items()
        ):
            # Sort tokens by percentage (descending)
            sorted_tokens = sorted(
                token_percentages.items(), key=lambda x: x[1], reverse=True
            )

            # Take top 10 tokens
            top_tokens = sorted_tokens[:10]

            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)

            # Extract token names and percentages
            token_names = [token for token, _ in top_tokens]
            percentages = [percentage for _, percentage in top_tokens]

            # Create horizontal bar plot
            y_pos = range(len(token_names))
            ax.barh(y_pos, percentages, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(token_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel("Percentage of Chunks (%)")
            ax.set_title(f"Top 10 {n}-grams in {category} Chunks")

            # Add percentage labels
            for j, percentage in enumerate(percentages):
                ax.text(percentage + 1, j, f"{percentage:.1f}%", va="center")

        # Adjust layout
        plt.tight_layout()

        # Save plot
        ngram_name = "unigrams" if n == 1 else f"{n}-grams"
        plt.savefig(plots_dir / f"dag_token_{ngram_name}_by_category.png", dpi=300)
        plt.close()

        print(
            f"{n}-gram frequency analysis complete. Plot saved to {plots_dir / f'dag_token_{ngram_name}_by_category.png'}"
        )

        # Save token frequencies to JSON
        token_frequencies_file = output_dir / f"dag_token_{ngram_name}_by_category.json"
        with open(token_frequencies_file, "w", encoding="utf-8") as f:
            json.dump(category_ngram_frequencies, f, indent=2)


def analyze_token_frequencies(
    results: List[Dict],
    output_dir: Path,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze token frequencies in rollout chunks.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        importance_metric: Importance metric to use for the analysis
    """
    print("Analyzing token frequencies by category...")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Collect all chunks by category
    category_chunks = {}

    for result in results:
        if not result or "labeled_chunks" not in result:
            continue

        for chunk in result.get("labeled_chunks", []):
            # Get function tags (categories)
            function_tags = chunk.get("function_tags", [])

            # Skip chunks with no tags
            if not function_tags:
                continue

            # Get chunk text
            chunk_text = chunk.get("chunk", "")

            # Skip empty chunks
            if not chunk_text:
                continue

            # Add chunk to each of its categories
            for tag in function_tags:
                # Format tag for better display
                if isinstance(tag, str):
                    formatted_tag = " ".join(
                        word.capitalize() for word in tag.split("_")
                    )
                else:
                    continue  # Skip non-string tags

                # Skip unknown category
                if formatted_tag.lower() == "unknown":
                    continue

                if formatted_tag not in category_chunks:
                    category_chunks[formatted_tag] = []

                category_chunks[formatted_tag].append(chunk_text)

    # Skip if no categories found
    if not category_chunks:
        print("No categories found for token frequency analysis")
        return

    # Generate plots for unigrams, bigrams, and trigrams
    for n in [1, 2, 3]:
        print(f"Analyzing {n}-gram frequencies...")

        # Tokenize chunks and count frequencies
        category_ngram_frequencies = {}

        for category, chunks in category_chunks.items():
            # Tokenize all chunks
            all_tokens = []
            for chunk in chunks:
                # Simple tokenization by splitting on whitespace and punctuation
                tokens = re.findall(r"\\b\\w+\\b", chunk.lower())

                # Filter out stopwords and numbers
                filtered_tokens = [
                    token
                    for token in tokens
                    if token not in stopwords and not token.isdigit() and len(token) > 1
                ]

                # Generate n-grams
                ngrams = []
                for i in range(len(filtered_tokens) - (n - 1)):
                    ngram = " ".join(filtered_tokens[i : i + n])
                    ngrams.append(ngram)

                all_tokens.extend(ngrams)

            # Count token frequencies
            token_counts = Counter(all_tokens)

            # Calculate percentages
            total_chunks = len(chunks)
            token_percentages = {}

            for token, count in token_counts.items():
                # Count in how many chunks this n-gram appears
                chunks_with_token = sum(
                    1
                    for chunk in chunks
                    if re.search(r"\\b" + re.escape(token) + r"\\b", chunk.lower())
                )
                percentage = (chunks_with_token / total_chunks) * 100
                token_percentages[token] = percentage

            # Store results
            category_ngram_frequencies[category] = token_percentages

        # Create a master plot with subplots for each category
        num_categories = len(category_ngram_frequencies)

        # Calculate grid dimensions
        cols = min(3, num_categories)
        rows = (num_categories + cols - 1) // cols  # Ceiling division

        # Create figure
        fig = plt.figure(figsize=(20, 5 * rows))

        # Create subplots
        for i, (category, token_percentages) in enumerate(
            category_ngram_frequencies.items()
        ):
            # Sort tokens by percentage (descending)
            sorted_tokens = sorted(
                token_percentages.items(), key=lambda x: x[1], reverse=True
            )

            # Take top 10 tokens
            top_tokens = sorted_tokens[:10]

            # Create subplot
            ax = fig.add_subplot(rows, cols, i + 1)

            # Extract token names and percentages
            token_names = [token for token, _ in top_tokens]
            percentages = [percentage for _, percentage in top_tokens]

            # Create horizontal bar plot
            y_pos = range(len(token_names))
            ax.barh(y_pos, percentages, align="center")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(token_names)
            ax.invert_yaxis()  # Labels read top-to-bottom
            ax.set_xlabel("Percentage of Chunks (%)")
            ax.set_title(f"Top 10 {n}-grams in {category} Chunks")

            # Add percentage labels
            for j, percentage in enumerate(percentages):
                ax.text(percentage + 1, j, f"{percentage:.1f}%", va="center")

        # Adjust layout
        plt.tight_layout()

        # Save plot
        ngram_name = "unigrams" if n == 1 else f"{n}-grams"
        plt.savefig(plots_dir / f"token_{ngram_name}_by_category.png", dpi=300)
        plt.close()

        print(
            f"{n}-gram frequency analysis complete. Plot saved to {plots_dir / f'token_{ngram_name}_by_category.png'}"
        )

        # Save token frequencies to JSON
        token_frequencies_file = output_dir / f"token_{ngram_name}_by_category.json"
        with open(token_frequencies_file, "w", encoding="utf-8") as f:
            json.dump(category_ngram_frequencies, f, indent=2)


def analyze_top_steps_by_category(
    results: List[Dict],
    output_dir: Path,
    top_n: int = 20,
    use_abs: bool = True,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze the top N steps by category based on importance scores.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        top_n: Number of top steps to analyze
        use_abs: Whether to use absolute values for importance scores
        importance_metric: Importance metric to use for the analysis
    """
    print(f"Analyzing top {top_n} steps by category")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create a dictionary to store z-scores by category
    category_zscores = {}

    # Process each problem
    for result in results:
        if not result:
            continue

        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue

        # Extract importance scores and convert to z-scores
        importance_scores = [
            chunk.get(importance_metric, 0.0)
            if not use_abs
            else abs(chunk.get(importance_metric, 0.0))
            for chunk in labeled_chunks
        ]

        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue

        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)

        if std_importance == 0:
            continue

        z_scores = [
            (score - mean_importance) / std_importance for score in importance_scores
        ]

        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            # Use absolute or raw z-score based on parameter
            score_for_ranking = z_score
            chunk_data.append((i, z_score, score_for_ranking, function_tags))

        # Sort by z-score (absolute or raw) and get top N
        top_chunks = sorted(chunk_data, key=lambda x: x[2], reverse=True)[:top_n]

        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, _, function_tags in top_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score

            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                if formatted_tag.lower() == "unknown":
                    continue

                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)

    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}

    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)

    # Sort categories by average z-score
    sorted_categories = sorted(
        category_avg_zscores.keys(), key=lambda x: category_avg_zscores[x], reverse=True
    )

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]

    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [std / np.sqrt(count) for std, count in zip(std_zscores, counts)]

    # Create bar plot with standard error bars
    bars = plt.bar(
        range(len(sorted_categories)),
        avg_zscores,
        yerr=standard_errors,
        capsize=5,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Set labels and title
    plt.xlabel("Function Tag Category", fontsize=12)
    plt.ylabel(f"Average Z-Score (Top {top_n} Steps) ± SE", fontsize=12)
    plt.title(f"Average Z-Score of Top {top_n} Steps by Category", fontsize=14)

    # Set x-tick labels
    plt.xticks(
        range(len(sorted_categories)), sorted_categories, rotation=45, ha="right"
    )

    # Add grid
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend explaining the error bars
    plt.figtext(
        0.91,
        0.01,
        "Error bars: Standard Error (SE)",
        ha="right",
        fontsize=10,
        style="italic",
    )

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plot_path = plots_dir / f"top_{top_n}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {plot_path}")

    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append(
            {
                "category": category,
                "avg_zscore": category_avg_zscores[category],
                "std_zscore": category_std_zscores[category],
                "standard_error": category_std_zscores[category]
                / np.sqrt(category_counts[category]),
                "count": category_counts[category],
            }
        )

    csv_path = plots_dir / f"top_{top_n}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


def analyze_high_zscore_steps_by_category(
    results: List[Dict],
    output_dir: Path,
    z_threshold: float = 1.5,
    use_abs: bool = True,
    importance_metric: str = "counterfactual_importance_accuracy",
) -> None:
    """
    Analyze steps with high z-scores by category to identify outlier steps.

    Args:
        results: List of problem analysis results
        output_dir: Directory to save analysis results
        z_threshold: Threshold for z-scores to consider
        use_abs: Whether to use absolute values for z-scores
        importance_metric: Importance metric to use for the analysis
    """
    print(f"Analyzing steps with z-score > {z_threshold} by category...")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)

    # Create a dictionary to store z-scores by category
    category_zscores = {}
    total_high_zscore_steps = 0
    total_steps_analyzed = 0

    # Process each problem
    for result in results:
        if not result:
            continue

        labeled_chunks = result.get("labeled_chunks", [])
        if not labeled_chunks:
            continue

        # Extract importance scores and convert to z-scores
        importance_scores = [
            chunk.get(importance_metric, 0.0)
            if not use_abs
            else abs(chunk.get(importance_metric, 0.0))
            for chunk in labeled_chunks
        ]

        # Skip if all scores are the same or if there are too few chunks
        if len(set(importance_scores)) <= 1 or len(importance_scores) < 3:
            continue

        # Calculate z-scores
        mean_importance = np.mean(importance_scores)
        std_importance = np.std(importance_scores)

        if std_importance == 0:
            continue

        z_scores = [
            (score - mean_importance) / std_importance for score in importance_scores
        ]
        total_steps_analyzed += len(z_scores)

        # Create a list of (chunk_idx, z_score, function_tags) tuples
        chunk_data = []
        for i, (chunk, z_score) in enumerate(zip(labeled_chunks, z_scores)):
            function_tags = chunk.get("function_tags", ["unknown"])
            if not function_tags:
                function_tags = ["unknown"]
            chunk_data.append((i, z_score, function_tags))

        # Filter chunks by z-score threshold
        high_zscore_chunks = [
            chunk for chunk in chunk_data if abs(chunk[1]) > z_threshold
        ]
        total_high_zscore_steps += len(high_zscore_chunks)

        # Add to category dictionary - each chunk can have multiple tags
        for _, z_score, function_tags in high_zscore_chunks:
            # Use the actual z-score (not the ranking score)
            score_to_store = z_score

            for tag in function_tags:
                # Format tag for better display
                formatted_tag = " ".join(word.capitalize() for word in tag.split("_"))
                if formatted_tag.lower() == "unknown":
                    continue

                if formatted_tag not in category_zscores:
                    category_zscores[formatted_tag] = []
                category_zscores[formatted_tag].append(score_to_store)

    print(
        f"Found {total_high_zscore_steps} steps with z-score > {z_threshold} out of {total_steps_analyzed} total steps ({total_high_zscore_steps / total_steps_analyzed:.1%})"
    )

    # Skip if no categories found
    if not category_zscores:
        print(f"No steps with z-score > {z_threshold} found")
        return

    # Calculate average z-score for each category
    category_avg_zscores = {}
    category_std_zscores = {}
    category_counts = {}

    for category, zscores in category_zscores.items():
        if zscores:
            category_avg_zscores[category] = np.mean(zscores)
            category_std_zscores[category] = np.std(zscores)
            category_counts[category] = len(zscores)

    # Sort categories by average z-score
    sorted_categories = sorted(
        category_avg_zscores.keys(), key=lambda x: category_avg_zscores[x], reverse=True
    )

    # Create the plot
    plt.figure(figsize=(15, 10))

    # Plot average z-scores with standard error bars
    avg_zscores = [category_avg_zscores[cat] for cat in sorted_categories]
    std_zscores = [category_std_zscores[cat] for cat in sorted_categories]
    counts = [category_counts[cat] for cat in sorted_categories]

    # Calculate standard error (SE = standard deviation / sqrt(sample size))
    standard_errors = [std / np.sqrt(count) for std, count in zip(std_zscores, counts)]

    # Create bar plot with standard error bars
    bars = plt.bar(
        range(len(sorted_categories)),
        avg_zscores,
        yerr=standard_errors,
        capsize=5,
        alpha=0.7,
        color="skyblue",
        edgecolor="black",
    )

    # Add count labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Set labels and title
    plt.xlabel("Function Tag Category", fontsize=12)
    plt.ylabel(f"Average Z-Score (Steps with |Z| > {z_threshold}) ± SE", fontsize=12)
    plt.title(
        f"Average Z-Score of Steps with |Z| > {z_threshold} by Category", fontsize=14
    )

    # Set x-tick labels
    plt.xticks(
        range(len(sorted_categories)), sorted_categories, rotation=45, ha="right"
    )

    # Add grid
    # plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Add a legend explaining the error bars
    plt.figtext(
        0.91,
        0.01,
        "Error bars: Standard Error (SE)",
        ha="right",
        fontsize=10,
        style="italic",
    )

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plot_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"Plot saved to {plot_path}")

    # Also save the data as CSV
    csv_data = []
    for i, category in enumerate(sorted_categories):
        csv_data.append(
            {
                "category": category,
                "avg_zscore": category_avg_zscores[category],
                "std_zscore": category_std_zscores[category],
                "standard_error": category_std_zscores[category]
                / np.sqrt(category_counts[category]),
                "count": category_counts[category],
            }
        )

    csv_path = plots_dir / f"high_zscore_{z_threshold}_steps_by_category.csv"
    pd.DataFrame(csv_data).to_csv(csv_path, index=False)
    print(f"Data saved to {csv_path}")


def analyze_response_length_statistics(
    correct_rollouts_dir: Path = None,
    incorrect_rollouts_dir: Path = None,
    output_dir: Path = None,
) -> None:
    """
    Analyze response length statistics in sentences and tokens with 95% confidence intervals.
    Combines data from both correct and incorrect rollouts for aggregate statistics.

    Args:
        correct_rollouts_dir: Directory containing correct rollout data
        incorrect_rollouts_dir: Directory containing incorrect rollout data
        output_dir: Directory to save analysis results
    """
    print("Analyzing response length statistics across all rollouts...")

    # Create analysis directory
    analysis_dir = output_dir / "response_length_analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)

    # Collect response length data from both correct and incorrect rollouts
    sentence_lengths = []
    token_lengths = []

    # Process correct rollouts if provided
    if correct_rollouts_dir and correct_rollouts_dir.exists():
        problem_dirs = sorted(
            [
                d
                for d in correct_rollouts_dir.iterdir()
                if d.is_dir() and d.name.startswith("problem_")
            ]
        )

        for problem_dir in tqdm(
            problem_dirs, desc="Processing correct rollouts for length analysis"
        ):
            base_solution_file = problem_dir / "base_solution.json"
            if not base_solution_file.exists():
                continue

            try:
                with open(base_solution_file, "r", encoding="utf-8") as f:
                    base_solution = json.load(f)

                full_cot = base_solution.get("full_cot", "")
                if not full_cot:
                    continue

                # Count sentences
                sentences = split_solution_into_chunks(full_cot)
                sentence_lengths.append(len(sentences))

                # Count tokens
                num_tokens = count_tokens(full_cot, approximate=False)
                token_lengths.append(num_tokens)

            except Exception as e:
                print(f"Error processing correct rollout {problem_dir.name}: {e}")
                continue

    # Process incorrect rollouts if provided
    if incorrect_rollouts_dir and incorrect_rollouts_dir.exists():
        problem_dirs = sorted(
            [
                d
                for d in incorrect_rollouts_dir.iterdir()
                if d.is_dir() and d.name.startswith("problem_")
            ]
        )

        for problem_dir in tqdm(
            problem_dirs, desc="Processing incorrect rollouts for length analysis"
        ):
            base_solution_file = problem_dir / "base_solution.json"
            if not base_solution_file.exists():
                continue

            try:
                with open(base_solution_file, "r", encoding="utf-8") as f:
                    base_solution = json.load(f)

                full_cot = base_solution.get("full_cot", "")
                if not full_cot:
                    continue

                # Count sentences
                sentences = split_solution_into_chunks(full_cot)
                sentence_lengths.append(len(sentences))

                # Count tokens
                num_tokens = count_tokens(full_cot, approximate=False)
                token_lengths.append(num_tokens)

            except Exception as e:
                print(f"Error processing incorrect rollout {problem_dir.name}: {e}")
                continue

    # Skip if no data collected
    if not sentence_lengths or not token_lengths:
        print("No response length data collected")
        return

    # Calculate statistics
    sentence_lengths = np.array(sentence_lengths)
    token_lengths = np.array(token_lengths)

    # Calculate means
    mean_sentences = np.mean(sentence_lengths)
    mean_tokens = np.mean(token_lengths)

    # Calculate 95% confidence intervals using t-distribution

    # For sentences
    sentence_sem = stats.sem(sentence_lengths)
    sentence_ci = stats.t.interval(
        0.95, len(sentence_lengths) - 1, loc=mean_sentences, scale=sentence_sem
    )

    # For tokens
    token_sem = stats.sem(token_lengths)
    token_ci = stats.t.interval(
        0.95, len(token_lengths) - 1, loc=mean_tokens, scale=token_sem
    )

    # Create the summary string
    summary_text = (
        f"The average response is {mean_sentences:.1f} sentences long "
        f"(95% CI: [{sentence_ci[0]:.1f}, {sentence_ci[1]:.1f}]; "
        f"this corresponds to {mean_tokens:.0f} tokens "
        f"[95% CI: {token_ci[0]:.0f}, {token_ci[1]:.0f}])."
    )

    # Print the result
    print(f"\n{summary_text}")

    # Save detailed statistics
    stats_data = {
        "num_responses": len(sentence_lengths),
        "sentences": {
            "mean": float(mean_sentences),
            "std": float(np.std(sentence_lengths)),
            "median": float(np.median(sentence_lengths)),
            "min": int(np.min(sentence_lengths)),
            "max": int(np.max(sentence_lengths)),
            "ci_95_lower": float(sentence_ci[0]),
            "ci_95_upper": float(sentence_ci[1]),
            "sem": float(sentence_sem),
        },
        "tokens": {
            "mean": float(mean_tokens),
            "std": float(np.std(token_lengths)),
            "median": float(np.median(token_lengths)),
            "min": int(np.min(token_lengths)),
            "max": int(np.max(token_lengths)),
            "ci_95_lower": float(token_ci[0]),
            "ci_95_upper": float(token_ci[1]),
            "sem": float(token_sem),
        },
        "summary": summary_text,
    }

    # Save to JSON file
    stats_file = analysis_dir / "response_length_stats.json"
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    print(f"Response length analysis saved to {analysis_dir}")
    print(f"Statistics saved to {stats_file}")


# -----------------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------
# Response length stats (unchanged except imports)
# ---------------------------
def analyze_response_length_statistics(
    correct_rollouts_dir: Path = None,
    incorrect_rollouts_dir: Path = None,
    output_dir: Path = None,
) -> None:
    print("Analyzing response length statistics across all rollouts...")
    analysis_dir = output_dir / "response_length_analysis"
    analysis_dir.mkdir(exist_ok=True, parents=True)
    sentence_lengths = []
    token_lengths = []

    def collect(dirpath: Path):
        pdirs = sorted(
            [
                d
                for d in dirpath.iterdir()
                if d.is_dir() and d.name.startswith("problem_")
            ]
        )
        for pd in tqdm(pdirs, desc=f"Processing {dirpath.name}"):
            f = pd / "base_solution.json"
            if not f.exists():
                continue
            try:
                base = json.load(open(f, "r", encoding="utf-8"))
                full = base.get("full_cot", "")
                if not full:
                    continue
                sents = split_solution_into_chunks(full)
                sentence_lengths.append(len(sents))
                token_lengths.append(count_tokens(full, approximate=False))
            except Exception as e:
                print(f"Error {pd.name}: {e}")

    if correct_rollouts_dir and correct_rollouts_dir.exists():
        collect(correct_rollouts_dir)
    if incorrect_rollouts_dir and incorrect_rollouts_dir.exists():
        collect(incorrect_rollouts_dir)

    if not sentence_lengths or not token_lengths:
        print("No response length data collected")
        return

    sentence_lengths = np.array(sentence_lengths)
    token_lengths = np.array(token_lengths)
    mean_sent = float(np.mean(sentence_lengths))
    mean_tok = float(np.mean(token_lengths))
    sent_sem = stats.sem(sentence_lengths)
    tok_sem = stats.sem(token_lengths)
    sent_ci = stats.t.interval(
        0.95, len(sentence_lengths) - 1, loc=mean_sent, scale=sent_sem
    )
    tok_ci = stats.t.interval(0.95, len(token_lengths) - 1, loc=mean_tok, scale=tok_sem)
    summary = (
        f"The average response is {mean_sent:.1f} sentences long "
        f"(95% CI: [{sent_ci[0]:.1f}, {sent_ci[1]:.1f}]; "
        f"this corresponds to {mean_tok:.0f} tokens "
        f"[95% CI: {tok_ci[0]:.0f}, {tok_ci[1]:.0f}])."
    )
    print("\n" + summary)

    stats_data = {
        "num_responses": len(sentence_lengths),
        "sentences": {
            "mean": mean_sent,
            "std": float(np.std(sentence_lengths)),
            "median": float(np.median(sentence_lengths)),
            "min": int(np.min(sentence_lengths)),
            "max": int(np.max(sentence_lengths)),
            "ci_95_lower": float(sent_ci[0]),
            "ci_95_upper": float(sent_ci[1]),
            "sem": float(sent_sem),
        },
        "tokens": {
            "mean": mean_tok,
            "std": float(np.std(token_lengths)),
            "median": float(np.median(token_lengths)),
            "min": int(np.min(token_lengths)),
            "max": int(np.max(token_lengths)),
            "ci_95_lower": float(tok_ci[0]),
            "ci_95_upper": float(tok_ci[1]),
            "sem": float(tok_sem),
        },
        "summary": summary,
    }
    json.dump(
        stats_data,
        open(analysis_dir / "response_length_stats.json", "w", encoding="utf-8"),
        indent=2,
    )
    print(f"Response length analysis saved to {analysis_dir}")


# ---------------------------
# Rollouts processor (needs only source resolution tweaks)
# ---------------------------
def process_rollouts(
    rollouts_dir: Path,
    output_dir: Path,
    problems: str = None,
    max_problems: int = None,
    absolute: bool = False,
    force_relabel: bool = False,
    rollout_type: str = "correct",
    dag_dir: Optional[str] = None,
    forced_answer_dir: Optional[Path] = None,
    get_token_frequencies: bool = False,
    max_chunks_to_show: int = 100,
    use_existing_metrics: bool = False,
    importance_metric: str = "counterfactual_importance_accuracy",
    sentence_model: str = "all-MiniLM-L6-v2",
    similarity_threshold: float = 0.8,
    force_metadata: bool = False,
) -> None:
    # find problem dirs
    problem_dirs = sorted(
        [
            d
            for d in rollouts_dir.iterdir()
            if d.is_dir() and d.name.startswith("problem_")
        ]
    )

    if problems:
        idxs = [int(x) for x in problems.split(",")]
        problem_dirs = [d for d in problem_dirs if int(d.name.split("_")[1]) in idxs]
    if max_problems:
        problem_dirs = problem_dirs[:max_problems]

    total = len(problem_dirs)
    with_complete = 0
    with_partial = 0
    with_none = 0
    for pd in problem_dirs:
        cf = pd / "chunks.json"
        if not cf.exists():
            with_none += 1
            continue
        try:
            cdata = json.load(open(cf, "r", encoding="utf-8"))
            chunks = cdata.get("chunks", [])
            if not chunks:
                with_none += 1
                continue
            cdirs = [pd / f"chunk_{i}" for i in range(len(chunks))]
            exist = [p for p in cdirs if p.exists()]
            if len(exist) == len(chunks):
                with_complete += 1
            elif len(exist) > 0:
                with_partial += 1
            else:
                with_none += 1
        except Exception:
            with_none += 1
    print(f"\n=== {rollout_type.capitalize()} Rollouts Summary ===")
    print(f"Total problems found: {total}")
    if total > 0:
        print(
            f"Problems with complete chunk folders: {with_complete} ({with_complete / total * 100:.1f}%)"
        )
        print(
            f"Problems with partial chunk folders:  {with_partial} ({with_partial / total * 100:.1f}%)"
        )
        print(
            f"Problems with no chunk folders:       {with_none} ({with_none / total * 100:.1f}%)"
        )

    analyzable = []
    for pd in problem_dirs:
        cf = pd / "chunks.json"
        if not cf.exists():
            continue
        try:
            cdata = json.load(open(cf, "r", encoding="utf-8"))
            chunks = cdata.get("chunks", [])
            if not chunks:
                continue
            cdirs = [pd / f"chunk_{i}" for i in range(len(chunks))]
            if any(p.exists() for p in cdirs):
                analyzable.append(pd)
        except Exception:
            pass

    print(f"Analyzing {len(analyzable)} problems with at least some chunk folders")

    results = []
    for pd in tqdm(analyzable, desc=f"Analyzing {rollout_type} problems"):
        res = analyze_problem(
            pd,
            absolute,
            force_relabel,
            forced_answer_dir,
            use_existing_metrics,
            sentence_model,
            similarity_threshold,
            force_metadata,
        )
        if res:
            results.append(res)

    # downstream analyses (unchanged)
    category_importance = generate_plots(results, output_dir, importance_metric)
    plot_chunk_accuracy_by_position(
        results, output_dir, rollout_type, max_chunks_to_show, importance_metric
    )
    analyze_within_problem_variance(results, output_dir, importance_metric)
    analyze_chunk_variance(results, output_dir, importance_metric)
    analyze_function_tag_variance(results, output_dir, importance_metric)
    analyze_top_steps_by_category(
        results, output_dir, top_n=20, use_abs=True, importance_metric=importance_metric
    )
    analyze_high_zscore_steps_by_category(
        results,
        output_dir,
        z_threshold=1.5,
        use_abs=True,
        importance_metric=importance_metric,
    )

    if category_importance is not None and not category_importance.empty:
        print(f"\n{rollout_type.capitalize()} Category Importance Ranking:")
        for idx, row in category_importance.iterrows():
            print(
                f"{idx + 1}. {row['categories']}: {row['mean_pct']:.2f}% ± {row['se_pct']:.2f}% (n={int(row['count'])})"
            )

    if get_token_frequencies:
        if dag_dir and dag_dir != "None":
            print("\nAnalyzing token frequencies from DAG-improved chunks")
            analyze_dag_token_frequencies(Path(dag_dir), output_dir)
        else:
            print("\nAnalyzing token frequencies from rollout results")
            analyze_token_frequencies(results, output_dir, importance_metric)

    # save results
    results_file = output_dir / "analysis_results.json"
    serializable = []
    for r in results:
        obj = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                obj[k] = v.tolist()
            elif isinstance(v, np.integer):
                obj[k] = int(v)
            elif isinstance(v, np.floating):
                obj[k] = float(v)
            else:
                obj[k] = v
        serializable.append(obj)
    json.dump(serializable, open(results_file, "w", encoding="utf-8"), indent=2)
    print(
        f"{rollout_type.capitalize()} analysis complete. Results saved to {output_dir}"
    )


# ---------------------------
# Main
# ---------------------------
def main():
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.hf_cache_dir)

    # Resolve each input (local path or hf://)
    correct_dir = (
        resolve_dir(args.correct_rollouts_dir, cache_root)
        if args.correct_rollouts_dir
        else None
    )
    incorrect_dir = (
        resolve_dir(args.incorrect_rollouts_dir, cache_root)
        if args.incorrect_rollouts_dir
        else None
    )
    correct_forced_dir = (
        resolve_dir(args.correct_forced_answer_rollouts_dir, cache_root)
        if args.correct_forced_answer_rollouts_dir
        else None
    )
    incorrect_forced_dir = (
        resolve_dir(args.incorrect_forced_answer_rollouts_dir, cache_root)
        if args.incorrect_forced_answer_rollouts_dir
        else None
    )

    if not correct_dir and not incorrect_dir:
        print(
            "Error: Provide at least one of --correct_rollouts_dir / --incorrect_rollouts_dir (local path or hf://...)"
        )
        return

    if correct_dir and incorrect_dir:
        print("\n=== Analyzing Response Length Statistics ===\n")
        analyze_response_length_statistics(correct_dir, incorrect_dir, out_dir)

    if correct_dir:
        print(f"\n=== Processing CORRECT rollouts from {correct_dir} ===\n")
        c_out = out_dir / "correct_base_solution"
        c_out.mkdir(parents=True, exist_ok=True)
        process_rollouts(
            rollouts_dir=correct_dir,
            output_dir=c_out,
            problems=args.problems,
            max_problems=args.max_problems,
            absolute=args.absolute,
            force_relabel=args.force_relabel,
            rollout_type="correct",
            dag_dir=args.dag_dir if args.token_analysis_source == "dag" else None,
            forced_answer_dir=correct_forced_dir,
            get_token_frequencies=args.get_token_frequencies,
            max_chunks_to_show=args.max_chunks_to_show,
            use_existing_metrics=args.use_existing_metrics,
            importance_metric=args.importance_metric,
            sentence_model=args.sentence_model,
            similarity_threshold=args.similarity_threshold,
            force_metadata=args.force_metadata,
        )

        # Optional extra forced-importance analyses (same as your original script logic) — can be added here as needed.

    if incorrect_dir:
        print(f"\n=== Processing INCORRECT rollouts from {incorrect_dir} ===\n")
        i_out = out_dir / "incorrect_base_solution"
        i_out.mkdir(parents=True, exist_ok=True)
        process_rollouts(
            rollouts_dir=incorrect_dir,
            output_dir=i_out,
            problems=args.problems,
            max_problems=args.max_problems,
            absolute=args.absolute,
            force_relabel=args.force_relabel,
            rollout_type="incorrect",
            dag_dir=args.dag_dir if args.token_analysis_source == "dag" else None,
            forced_answer_dir=incorrect_forced_dir,
            get_token_frequencies=args.get_token_frequencies,
            max_chunks_to_show=args.max_chunks_to_show,
            use_existing_metrics=args.use_existing_metrics,
            importance_metric=args.importance_metric,
            sentence_model=args.sentence_model,
            similarity_threshold=args.similarity_threshold,
            force_metadata=args.force_metadata,
        )


if __name__ == "__main__":
    main()
