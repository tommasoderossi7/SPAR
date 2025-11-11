#!/usr/bin/env python3
# compare_token_vs_sentence.py
# Compares token-level vs sentence-level counterfactual importance.
# - Overlap of top tokens inside top sentences
# - Segmentation similarity (boundaries from top tokens vs sentence boundaries)
# - Correlation between per-sentence importance and aggregated token importance

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------
# IO utilities
# ---------------------------


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# ---------------------------
# Text alignment utilities
# ---------------------------


def sequential_find_spans(full: str, parts: List[str]) -> List[Tuple[int, int]]:
    """
    Given a full string and a list of parts that should appear in order,
    return monotone (start, end) spans for each part using forward scanning.
    Falls back to whitespace-normalized matching if an exact substring match fails.
    """
    spans: List[Tuple[int, int]] = []
    i = 0
    for piece in parts:
        if not piece:
            spans.append((i, i))
            continue
        j = full.find(piece, i)
        if j < 0:
            # fallback to normalized scan
            norm_full = _norm_space(full[i:])
            norm_piece = _norm_space(piece)
            k = norm_full.find(norm_piece)
            if k < 0:
                return []  # give up (caller handles)
            j = i + k
        start = j
        end = j + len(piece)
        spans.append((start, end))
        i = end
    return spans


# ---------------------------
# KL + helpers
# ---------------------------


def kl_full(p: Dict[str, float], q: Dict[str, float], eps: float = 1e-12) -> float:
    keys = set(p.keys()) | set(q.keys())
    acc = 0.0
    for k in keys:
        pk = max(p.get(k, 0.0), eps)
        qk = max(q.get(k, 0.0), eps)
        acc += pk * math.log(pk / qk)
    return float(acc)


def dist_from_samples(samples: List[Dict[str, Any]]) -> Dict[str, float]:
    cnt: Dict[str, int] = {}
    tot = 0
    for s in samples or []:
        if s.get("error"):
            continue
        o = s.get("outcome")
        if isinstance(o, str) and o:
            cnt[o] = cnt.get(o, 0) + 1
            tot += 1
    if tot == 0:
        return {}
    return {k: v / float(tot) for k, v in cnt.items()}


# ---------------------------
# Loaders (sentence & token)
# ---------------------------


def load_sentence_analysis(sentence_json_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Input: the sentence-level analysis JSON (list of problem dicts).
    Output: dict keyed by problem_idx (string) with:
      {
        'chunks': [chunk_text, ...],
        'metric': [importance_value, ...]
      }
    """
    raw = _read_json(sentence_json_path)
    by_problem: Dict[str, Dict[str, Any]] = {}

    for rec in raw:
        pid = str(rec.get("problem_idx"))
        labeled = rec.get("labeled_chunks") or []
        chunks = [c.get("chunk", "") for c in labeled]
        # Default metric: counterfactual_importance_kl; caller can override later by rebuilding arrays
        metric_map = {
            i: c.get("counterfactual_importance_kl", 0.0) for i, c in enumerate(labeled)
        }
        metrics = [metric_map.get(i, 0.0) for i in range(len(chunks))]
        by_problem[pid] = {
            "chunks": chunks,
            "metrics": metrics,
            "raw_labeled": labeled,  # keep for alternate metric extraction
        }
    return by_problem


def _token_entries_from_rollout(
    rollout: Dict[str, Any],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Returns (base_completion_text, token_entries list with fields:
      - text_offset (char index)
      - importance (float, token-level KL)
      - any other fields passthrough
    This handles both modern 'token_steps' with 'cf_metrics.kl_full' and legacy 'positions' where
    we can compute KL from control/intervention dists (or samples).
    """
    # 1) base completion
    base = rollout.get("base") or {}
    base_text = (
        base.get("completion_raw")
        or base.get("completion")
        or rollout.get("completion")
        or ""
    )

    # 2) entries
    steps = rollout.get("token_steps")
    if isinstance(steps, list) and steps:
        out = []
        for e in steps:
            off = e.get("text_offset")
            cf = e.get("cf_metrics") or {}
            imp = cf.get("kl_full")
            if imp is None:
                # fallback to kl_true or delta_acc (not ideal)
                imp = cf.get("kl_true", 0.0)
            out.append(
                {
                    **e,
                    "text_offset": off,
                    "importance": float(imp) if isinstance(imp, (int, float)) else 0.0,
                }
            )
        return base_text, out

    # Fallback: legacy 'positions'
    positions = rollout.get("positions") or []
    out = []
    for p in positions:
        off = p.get("text_offset")
        # try to read dists directly
        ctrl_dist = ((p.get("control") or {}).get("dist")) or None
        intv_dist = ((p.get("intervention") or {}).get("dist")) or None

        if not ctrl_dist or not intv_dist:
            # compute from samples if possible
            ctrl_samples = (p.get("control") or {}).get("samples") or []
            intv_samples = (p.get("intervention") or {}).get("samples") or []
            ctrl_dist = dist_from_samples(ctrl_samples)
            intv_dist = dist_from_samples(intv_samples)

        imp = 0.0
        if ctrl_dist and intv_dist:
            imp = kl_full(
                intv_dist, ctrl_dist
            )  # intervention vs control (token CF importance)

        out.append(
            {
                **p,
                "text_offset": off,
                "importance": float(imp),
            }
        )

    return base_text, out


# ---------------------------
# Core comparisons
# ---------------------------


def top_k_indices(values: List[float], k: int, largest: bool = True) -> List[int]:
    if not values or k <= 0:
        return []
    idxs = list(range(len(values)))
    idxs.sort(key=lambda i: values[i], reverse=largest)
    return idxs[:k]


def which_sentence_of_offset(
    offset: Optional[int], sentence_spans: List[Tuple[int, int]]
) -> Optional[int]:
    if not isinstance(offset, int):
        return None
    # linear scan is fine; typical sentence counts are small
    for i, (s, e) in enumerate(sentence_spans):
        if s <= offset < e:
            return i
    return None


def comparison_overlap(
    *,
    token_entries: List[Dict[str, Any]],
    token_k: int,
    sentence_spans: List[Tuple[int, int]],
    sentence_importances: List[float],
    sentence_k: int,
) -> Dict[str, Any]:
    # Pick top tokens (by importance)
    token_vals = [float(t.get("importance", 0.0)) for t in token_entries]
    tok_idxs = top_k_indices(token_vals, token_k, largest=True)
    chosen_tokens = [
        token_entries[i]
        for i in tok_idxs
        if token_entries[i].get("text_offset") is not None
    ]

    # Pick top sentences (by importance)
    sent_idxs_top = set(top_k_indices(sentence_importances, sentence_k, largest=True))

    # Map chosen tokens -> sentences
    tok_sent_hits = 0
    sent_with_top_token: set = set()
    token_in_sent: List[Optional[int]] = []
    for rec in chosen_tokens:
        sidx = which_sentence_of_offset(rec.get("text_offset"), sentence_spans)
        token_in_sent.append(sidx)
        if sidx is not None:
            if sidx in sent_idxs_top:
                tok_sent_hits += 1
            if sidx in sent_idxs_top:
                sent_with_top_token.add(sidx)

    token_overlap_fraction = (
        (tok_sent_hits / max(len(chosen_tokens), 1)) if chosen_tokens else 0.0
    )
    sentence_recall_fraction = (
        (len(sent_with_top_token) / max(len(sent_idxs_top), 1))
        if sent_idxs_top
        else 0.0
    )

    return {
        "top_token_indices": tok_idxs,
        "top_sentence_indices": sorted(list(sent_idxs_top)),
        "token_in_which_sentence": token_in_sent,
        "token_overlap_fraction": float(token_overlap_fraction),
        "sentence_recall_fraction": float(sentence_recall_fraction),
        "token_hits": int(tok_sent_hits),
        "sent_hits": int(len(sent_with_top_token)),
    }


def comparison_segmentation(
    *,
    token_entries: List[Dict[str, Any]],
    base_text: str,
    sentence_spans: List[Tuple[int, int]],
    sentence_importances: List[float],
    tolerance: int,
) -> Dict[str, Any]:
    # Sentence boundaries = ends of each span except the very last end
    sent_bounds = [e for (_, e) in sentence_spans[:-1]]
    N = len(sentence_spans)
    # Select N-1 most important tokens (unique offsets, sorted by importance desc)
    token_sorted = sorted(
        [e for e in token_entries if isinstance(e.get("text_offset"), int)],
        key=lambda r: r.get("importance", 0.0),
        reverse=True,
    )
    tok_bounds_unique: List[int] = []
    seen = set()
    for rec in token_sorted:
        off = int(rec["text_offset"])
        if off in seen:
            continue
        seen.add(off)
        tok_bounds_unique.append(off)
        if len(tok_bounds_unique) >= max(N - 1, 0):
            break

    tok_bounds_unique.sort()
    # Compare boundaries with tolerance (in characters)
    matched_sent = set()
    matched_tok = set()
    # Greedy nearest match
    for ti, tb in enumerate(tok_bounds_unique):
        # find nearest sentence boundary within tolerance
        best_j = None
        best_d = None
        for j, sb in enumerate(sent_bounds):
            if j in matched_sent:
                continue
            d = abs(tb - sb)
            if d <= tolerance and (best_d is None or d < best_d):
                best_d = d
                best_j = j
        if best_j is not None:
            matched_tok.add(ti)
            matched_sent.add(best_j)

    precision = (
        len(matched_tok) / max(len(tok_bounds_unique), 1) if tok_bounds_unique else 0.0
    )
    recall = len(matched_sent) / max(len(sent_bounds), 1) if sent_bounds else 0.0
    f1 = (
        (2 * precision * recall / max(precision + recall, 1e-12))
        if (precision > 0 and recall > 0)
        else 0.0
    )

    # Boundary error stats (distance to nearest boundary, for all token boundaries)
    def nearest_dist(x: int, arr: List[int]) -> int:
        if not arr:
            return abs(x - 0)
        return min(abs(x - y) for y in arr)

    abs_errors = [nearest_dist(tb, sent_bounds) for tb in tok_bounds_unique]
    mean_abs_err = float(np.mean(abs_errors)) if abs_errors else 0.0
    median_abs_err = float(np.median(abs_errors)) if abs_errors else 0.0

    return {
        "n_sentences": N,
        "token_boundaries": tok_bounds_unique,
        "sentence_boundaries": sent_bounds,
        "tolerance": int(tolerance),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_abs_boundary_error": mean_abs_err,
        "median_abs_boundary_error": median_abs_err,
        "matched_token_boundaries": sorted(list(matched_tok)),
        "matched_sentence_boundaries": sorted(list(matched_sent)),
    }


def comparison_correlation(
    *,
    token_entries: List[Dict[str, Any]],
    sentence_spans: List[Tuple[int, int]],
    sentence_importances: List[float],
) -> Dict[str, Any]:
    """
    Aggregate token importances inside each sentence (sum/mean/max),
    and compute correlation vs. sentence importances.
    """
    # Map tokens to sentence index
    per_sentence_tokens: List[List[float]] = [[] for _ in range(len(sentence_spans))]
    for e in token_entries:
        off = e.get("text_offset")
        if not isinstance(off, int):
            continue
        sidx = which_sentence_of_offset(off, sentence_spans)
        if sidx is None:
            continue
        per_sentence_tokens[sidx].append(float(e.get("importance", 0.0)))

    # Build aggregated vectors
    agg_sum = [float(sum(v)) for v in per_sentence_tokens]
    agg_mean = [float(np.mean(v)) if v else 0.0 for v in per_sentence_tokens]
    agg_max = [float(max(v)) if v else 0.0 for v in per_sentence_tokens]
    y = np.array([float(v) for v in sentence_importances], dtype=float)

    def _safe_spearman(a: List[float], b: List[float]) -> float:
        try:
            from scipy.stats import spearmanr  # optional; fallback if unavailable

            return float(spearmanr(a, b, nan_policy="omit").correlation or 0.0)
        except Exception:
            # simple rank correlation (without ties handling) as fallback
            def _rank(vals):
                idx = list(range(len(vals)))
                idx.sort(key=lambda i: vals[i])
                r = [0] * len(vals)
                for rank, i in enumerate(idx):
                    r[i] = rank
                return r

            ra, rb = _rank(a), _rank(b)
            aa, bb = np.array(ra, float), np.array(rb, float)
            aa = (aa - aa.mean()) / (aa.std() + 1e-12)
            bb = (bb - bb.mean()) / (bb.std() + 1e-12)
            return float(np.clip((aa * bb).mean(), -1, 1))

    def _safe_kendall(a: List[float], b: List[float]) -> float:
        try:
            from scipy.stats import kendalltau

            return float(kendalltau(a, b, nan_policy="omit").correlation or 0.0)
        except Exception:
            # crude fallback: sign of Spearman
            return float(np.sign(_safe_spearman(a, b)))

    results = {
        "sum_spearman": _safe_spearman(agg_sum, sentence_importances),
        "sum_kendall": _safe_kendall(agg_sum, sentence_importances),
        "mean_spearman": _safe_spearman(agg_mean, sentence_importances),
        "mean_kendall": _safe_kendall(agg_mean, sentence_importances),
        "max_spearman": _safe_spearman(agg_max, sentence_importances),
        "max_kendall": _safe_kendall(agg_max, sentence_importances),
        "per_sentence_token_sum": agg_sum,
        "per_sentence_token_mean": agg_mean,
        "per_sentence_token_max": agg_max,
    }
    return results


# ---------------------------
# Orchestration for one problem
# ---------------------------


def analyze_problem(
    *,
    pid: str,
    rollout_path: Path,
    sentence_record: Dict[str, Any],
    token_top_k: int,
    sent_top_k: int,
    sentence_metric: str = "counterfactual_importance_kl",
    boundary_tolerance: int = 8,
) -> Optional[Dict[str, Any]]:
    # Load token rollout
    rollout = _read_json(rollout_path)
    base_text, token_entries_all = _token_entries_from_rollout(rollout)

    if not base_text or not token_entries_all:
        return None

    # Build sentence chunks + metric vector from the sentence record
    chunks = (
        [c.get("chunk", "") for c in sentence_record.get("raw_labeled", [])]
        or sentence_record.get("chunks")
        or []
    )
    if not chunks:
        return None

    labeled = sentence_record.get("raw_labeled") or []
    sent_metric_map = {
        i: float(c.get(sentence_metric, 0.0)) for i, c in enumerate(labeled)
    }
    sentence_importances = [sent_metric_map.get(i, 0.0) for i in range(len(chunks))]

    # Align sentence spans to base_text
    spans = sequential_find_spans(base_text, chunks)
    if not spans or len(spans) != len(chunks):
        # Could not align reliably
        return None

    # --- Comparison 1: overlap ---
    cmp1 = comparison_overlap(
        token_entries=token_entries_all,
        token_k=token_top_k,
        sentence_spans=spans,
        sentence_importances=sentence_importances,
        sentence_k=sent_top_k,
    )

    # --- Comparison 2: segmentation similarity ---
    cmp2 = comparison_segmentation(
        token_entries=token_entries_all,
        base_text=base_text,
        sentence_spans=spans,
        sentence_importances=sentence_importances,
        tolerance=boundary_tolerance,
    )

    # --- Comparison 3: correlation (aggregated token -> sentence) ---
    cmp3 = comparison_correlation(
        token_entries=token_entries_all,
        sentence_spans=spans,
        sentence_importances=sentence_importances,
    )

    return {
        "problem_idx": pid,
        "n_tokens_total": len(token_entries_all),
        "n_sentences": len(chunks),
        "overlap": cmp1,
        "segmentation": cmp2,
        "correlation": cmp3,
    }


# ---------------------------
# CLI
# ---------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Compare token-level and sentence-level counterfactual importance."
    )
    ap.add_argument(
        "--token-root",
        required=True,
        help="Root containing problem_*/rollout_analysis.json from token-level pipeline.",
    )
    ap.add_argument(
        "--sentence-json",
        required=True,
        help="Path to the sentence-level analysis JSON (list, one record per problem).",
    )
    ap.add_argument(
        "--top-tokens",
        type=int,
        default=20,
        help="K for top tokens in overlap (comparison 1).",
    )
    ap.add_argument(
        "--top-sentences",
        type=int,
        default=5,
        help="K for top sentences in overlap (comparison 1).",
    )
    ap.add_argument(
        "--sentence-metric",
        type=str,
        default="counterfactual_importance_kl",
        help="Field in labeled_chunks to use for sentence importance (default: counterfactual_importance_kl).",
    )
    ap.add_argument(
        "--boundary-tolerance",
        type=int,
        default=8,
        help="Tolerance in characters for boundary matching (comparison 2).",
    )
    ap.add_argument(
        "--out", type=str, default="comparison_out", help="Output directory."
    )
    args = ap.parse_args()

    token_root = Path(args.token_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sentence_map = load_sentence_analysis(Path(args.sentence_json))

    per_problem: List[Dict[str, Any]] = []
    for d in tqdm(sorted(token_root.glob("problem_*")), desc="Comparing"):
        pid = d.name.split("_")[-1]  # support both "problem_330" or nested paths
        rollout_path = d / "rollout_analysis.json"
        if not rollout_path.exists():
            continue
        srec = sentence_map.get(str(pid))
        if not srec:
            # try full directory name as key
            srec = sentence_map.get(d.name)
        if not srec:
            continue

        rec = analyze_problem(
            pid=str(pid),
            rollout_path=rollout_path,
            sentence_record=srec,
            token_top_k=args.top_tokens,
            sent_top_k=args.top_sentences,
            sentence_metric=args.sentence_metric,
            boundary_tolerance=args.boundary_tolerance,
        )
        if rec:
            per_problem.append(rec)

    # Save detailed results
    with open(out_dir / "token_sentence_comparison.json", "w", encoding="utf-8") as f:
        json.dump(per_problem, f, indent=2, ensure_ascii=False)

    # Aggregate summary
    if per_problem:
        df_overlap = pd.DataFrame(
            [
                {
                    "problem_idx": r["problem_idx"],
                    "token_overlap_fraction": r["overlap"]["token_overlap_fraction"],
                    "sentence_recall_fraction": r["overlap"][
                        "sentence_recall_fraction"
                    ],
                }
                for r in per_problem
            ]
        )

        df_seg = pd.DataFrame(
            [
                {
                    "problem_idx": r["problem_idx"],
                    "precision": r["segmentation"]["precision"],
                    "recall": r["segmentation"]["recall"],
                    "f1": r["segmentation"]["f1"],
                    "mean_abs_boundary_error": r["segmentation"][
                        "mean_abs_boundary_error"
                    ],
                    "median_abs_boundary_error": r["segmentation"][
                        "median_abs_boundary_error"
                    ],
                }
                for r in per_problem
            ]
        )

        df_corr = pd.DataFrame(
            [
                {
                    "problem_idx": r["problem_idx"],
                    "sum_spearman": r["correlation"]["sum_spearman"],
                    "sum_kendall": r["correlation"]["sum_kendall"],
                    "mean_spearman": r["correlation"]["mean_spearman"],
                    "mean_kendall": r["correlation"]["mean_kendall"],
                    "max_spearman": r["correlation"]["max_spearman"],
                    "max_kendall": r["correlation"]["max_kendall"],
                }
                for r in per_problem
            ]
        )

        summary = {
            "n_problems": int(len(per_problem)),
            # Overlap
            "mean_token_overlap": float(df_overlap["token_overlap_fraction"].mean()),
            "mean_sentence_recall": float(
                df_overlap["sentence_recall_fraction"].mean()
            ),
            "median_token_overlap": float(
                df_overlap["token_overlap_fraction"].median()
            ),
            "median_sentence_recall": float(
                df_overlap["sentence_recall_fraction"].median()
            ),
            # Segmentation
            "mean_seg_precision": float(df_seg["precision"].mean()),
            "mean_seg_recall": float(df_seg["recall"].mean()),
            "mean_seg_f1": float(df_seg["f1"].mean()),
            "mean_seg_mae_chars": float(df_seg["mean_abs_boundary_error"].mean()),
            "median_seg_mae_chars": float(df_seg["median_abs_boundary_error"].median()),
            # Correlations
            "mean_sum_spearman": float(df_corr["sum_spearman"].mean()),
            "mean_mean_spearman": float(df_corr["mean_spearman"].mean()),
            "mean_max_spearman": float(df_corr["max_spearman"].mean()),
            "mean_sum_kendall": float(df_corr["sum_kendall"].mean()),
            "mean_mean_kendall": float(df_corr["mean_kendall"].mean()),
            "mean_max_kendall": float(df_corr["max_kendall"].mean()),
        }

        # Write artifacts
        df_overlap.to_csv(out_dir / "overlap_metrics.csv", index=False)
        df_seg.to_csv(out_dir / "segmentation_metrics.csv", index=False)
        df_corr.to_csv(out_dir / "correlation_metrics.csv", index=False)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\nSummary:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("No comparable problems found (check paths / base alignment).")


if __name__ == "__main__":
    main()
