"""Analyze chain-of-thought token alternatives for a given generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "generations"
LOGITS_FILE = DATA_DIR / "logits.json"
COT_FILE = DATA_DIR / "cots.json"
OUTPUT_FILE = DATA_DIR / "cots_analysis.json"
PLOTS_DIR = DATA_DIR.parent / "plots" / "alt_tokens_distr"


def load_json_array(path: Path) -> List[Any]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Expected a list in {path}, found {type(data).__name__}")


def ensure_probability(entry: Dict[str, Any]) -> Optional[float]:
    probability = entry.get("probability")
    if isinstance(probability, (int, float)):
        return float(probability)
    logprob = entry.get("logprob")
    if isinstance(logprob, (int, float)):
        from math import exp

        try:
            return float(exp(logprob))
        except OverflowError:
            return None
    return None


def extract_primary_tokens(
    logits_entries: List[Dict[str, Any]],
) -> Tuple[List[str], List[Tuple[int, int]]]:
    tokens: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for entry in sorted(logits_entries, key=lambda x: x.get("token_index", 0)):
        token_candidates = entry.get("logits") or []
        token_text = ""
        if token_candidates:
            candidate = token_candidates[0]
            token_text = candidate.get("token") or ""
        tokens.append(token_text)
        start = cursor
        cursor += len(token_text)
        spans.append((start, cursor))
    return tokens, spans


def normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def find_cot_char_span(full_text: str, cot_text: str) -> Optional[Tuple[int, int]]:
    if not cot_text:
        return None
    full_norm = normalise_newlines(full_text)
    cot_norm = normalise_newlines(cot_text).strip()
    start = full_norm.find(cot_norm)
    if start == -1:
        import re

        compact_full = re.sub(r"\s+", " ", full_norm)
        compact_cot = re.sub(r"\s+", " ", cot_norm)
        start = compact_full.find(compact_cot)
        if start == -1:
            return None
        return None
    end = start + len(cot_norm)
    return (start, end)


def tokens_overlapping_span(
    spans: List[Tuple[int, int]], span: Tuple[int, int]
) -> List[int]:
    start, end = span
    indices: List[int] = []
    for idx, (token_start, token_end) in enumerate(spans):
        if token_end <= start:
            continue
        if token_start >= end:
            break
        indices.append(idx)
    return indices


def find_cot_indices(logits_entry: Dict[str, Any], cot_text: str) -> List[int]:
    tokens, spans = extract_primary_tokens(logits_entry.get("logits", []))
    if not tokens:
        return []
    full_text = "".join(tokens)
    span = find_cot_char_span(full_text, cot_text)
    if span is None:
        cot_len = len(normalise_newlines(cot_text).strip())
        indices: List[int] = []
        total = 0
        for idx, (_, token_end) in enumerate(spans):
            total = token_end
            indices.append(idx)
            if total >= cot_len:
                break
        return indices
    return tokens_overlapping_span(spans, span)


def analyse_generation(
    generation_id: str,
    top_logprobs: int,
    min_alternative_probability: float,
) -> Dict[str, Any]:
    logits_entries = load_json_array(LOGITS_FILE)
    target_logits = next(
        (
            entry
            for entry in logits_entries
            if entry.get("generation_id") == generation_id
        ),
        None,
    )
    if target_logits is None:
        raise RuntimeError(
            f"Generation ID '{generation_id}' not found in {LOGITS_FILE}"
        )

    cot_entries = load_json_array(COT_FILE)
    target_cot = next(
        (entry for entry in cot_entries if entry.get("generation_id") == generation_id),
        None,
    )
    if target_cot is None:
        raise RuntimeError(f"Generation ID '{generation_id}' not found in {COT_FILE}")

    logits = target_logits.get("logits") or []
    tokens, _ = extract_primary_tokens(logits)
    cot_text = target_cot.get("CoT_text", "")

    cot_indices = find_cot_indices(target_logits, cot_text)

    alternative_indices: List[int] = []
    index_to_entry = {entry.get("token_index"): entry for entry in logits}
    for idx in cot_indices:
        entry = index_to_entry.get(idx)
        if not entry:
            continue
        candidates = entry.get("logits") or []
        alternatives = candidates[1 : 1 + max(0, top_logprobs)]
        for candidate in alternatives:
            probability = ensure_probability(candidate) or 0.0
            if probability >= min_alternative_probability:
                alternative_indices.append(idx)
                break

    result = {
        "generation_id": generation_id,
        "cot_tokens_size": len(cot_indices),
        "full_answer_tokens_size": len(tokens),
        "cot_alternative_tokens_indices": alternative_indices,
        "cot_alternative_tokens_indices_size": len(alternative_indices),
    }
    return result


def append_analysis(result: Dict[str, Any]) -> None:
    entries = load_json_array(OUTPUT_FILE)
    entries = [
        entry
        for entry in entries
        if entry.get("generation_id") != result.get("generation_id")
    ]
    entries.append(result)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_analysis_entry(generation_id: str) -> Optional[Dict[str, Any]]:
    entries = load_json_array(OUTPUT_FILE)
    return next(
        (entry for entry in entries if entry.get("generation_id") == generation_id),
        None,
    )


def calculate_bucket_counts(
    cot_tokens_size: int, alternative_indices: List[int]
) -> Tuple[List[str], List[int]]:
    bucket_labels = [f"{start}-{start + 10}%" for start in range(0, 100, 10)]
    counts = [0] * len(bucket_labels)
    if cot_tokens_size <= 0:
        return bucket_labels, counts

    for index in alternative_indices:
        if index < 0:
            continue
        percentage = (index / cot_tokens_size) * 100
        bucket = min(int(percentage // 10), len(counts) - 1)
        counts[bucket] += 1

    return bucket_labels, counts


def create_alt_token_distribution_plot(
    analysis_entry: Dict[str, Any],
    top_logprobs: int,
    min_alternative_probability: float,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting requires matplotlib
        raise RuntimeError(
            "Matplotlib is required to generate plots. Please install it first."
        ) from exc

    cot_tokens_size = int(analysis_entry.get("cot_tokens_size", 0))
    alternative_indices = analysis_entry.get("cot_alternative_tokens_indices", [])
    alternative_indices = [
        int(index) for index in alternative_indices if isinstance(index, (int, float))
    ]

    bucket_labels, counts = calculate_bucket_counts(cot_tokens_size, alternative_indices)
    x_positions = range(len(bucket_labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_color = "#4c72b0"
    ax.bar(x_positions, counts, color=bar_color, alpha=0.85)
    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(bucket_labels, rotation=45, ha="right")
    ax.set_xlabel("CoT coverage range (% of tokens)")
    ax.set_ylabel("Alternative token count")
    ax.set_title(
        f"Alternative token distribution for {analysis_entry.get('generation_id', 'unknown')}"
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)

    alt_count = len(alternative_indices)
    fraction = (alt_count / cot_tokens_size) if cot_tokens_size else 0.0
    stats_lines = [
        f"CoT tokens: {cot_tokens_size}",
        f"Alt tokens: {alt_count}",
        f"Top-k: {top_logprobs}",
        f"Min prob: {min_alternative_probability:.2f}",
        f"Alt/CoT: {fraction:.2%}",
    ]
    stats_text = "\n".join(stats_lines)
    ax.text(
        0.98,
        0.98,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.8, "boxstyle": "round,pad=0.4"},
    )

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = PLOTS_DIR / f"{analysis_entry.get('generation_id', 'unknown')}.jpg"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze CoT token alternatives for a generation."
    )
    parser.add_argument(
        "generation_id", help="Identifier of the generation to analyse."
    )
    parser.add_argument(
        "--top-logprobs",
        type=int,
        default=10,
        help="Number of top alternative logits to inspect per token (excluding the sampled token).",
    )
    parser.add_argument(
        "--min-alternative-token-probability",
        type=float,
        default=0.05,
        help="Minimum probability threshold for an alternative token to be considered significant.",
    )
    parser.add_argument(
        "--create-plot",
        action="store_true",
        help=(
            "Generate a bar chart of alternative token positions using the stored "
            "analysis results."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = analyse_generation(
        generation_id=args.generation_id,
        top_logprobs=args.top_logprobs,
        min_alternative_probability=args.min_alternative_token_probability,
    )
    append_analysis(result)
    print(f"Saved CoT analysis for {args.generation_id} to {OUTPUT_FILE}")

    if args.create_plot:
        latest_entry = load_analysis_entry(args.generation_id)
        if latest_entry is None:
            raise RuntimeError(
                f"Analysis data for '{args.generation_id}' not found in {OUTPUT_FILE}"
            )
        plot_path = create_alt_token_distribution_plot(
            latest_entry,
            top_logprobs=args.top_logprobs,
            min_alternative_probability=args.min_alternative_token_probability,
        )
        print(f"Saved alternative token distribution plot to {plot_path}")


if __name__ == "__main__":
    main()
