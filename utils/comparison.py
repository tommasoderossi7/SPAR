#!/usr/bin/env python3
# comparison.py
import json, math, argparse, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm


def _read_json(p: Path):
    return json.load(open(p, "r", encoding="utf-8"))


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _sequential_find_spans(full: str, parts: List[str]) -> List[Tuple[int, int]]:
    """
    Given the ground truth full string and a list of parts (chunks) that were
    extracted sequentially from it, return (start,end) spans for each part.
    Uses a forward scan with .find starting at the previous end, which matches
    how chunks.json is produced.
    """
    spans = []
    i = 0
    for piece in parts:
        if not piece:
            spans.append((i, i))
            continue
        j = full.find(piece, i)
        if j < 0:
            # fallback: try normalized whitespace match
            # (looser, but still monotone)
            norm_full = _norm_space(full[i:])
            norm_piece = _norm_space(piece)
            k = norm_full.find(norm_piece)
            if k < 0:
                return []  # give up, caller will handle
            # map back to approximate char index
            j = i + k
        start = j
        end = j + len(piece)
        spans.append((start, end))
        i = end
    return spans


def _load_chunk_problem(problem_dir: Path) -> Optional[Dict]:
    bs = problem_dir / "base_solution.json"
    cj = problem_dir / "chunks.json"
    cl = problem_dir / "chunks_labeled.json"
    if not (bs.exists() and cj.exists()):
        return None
    base = _read_json(bs)
    chunks = _read_json(cj).get("chunks", [])
    labeled = _read_json(cl) if cl.exists() else None
    return {
        "full_cot": base.get("full_cot", ""),
        "chunks": chunks,
        "labeled": labeled,
    }


def _top_k_indices(values: List[float], k: int, reverse: bool = False) -> List[int]:
    # reverse=False => take k smallest; reverse=True => take k largest
    if not values:
        return []
    idxs = list(range(len(values)))
    idxs.sort(key=lambda i: values[i], reverse=reverse)
    return idxs[:k]


def compare_one_problem(
    token_rollout: Dict,
    chunk_dir: Path,
    top_tokens: int,
    top_sents: int,
    require_exact_match: bool = True,
    use_chunk_metric: str = "counterfactual_importance_accuracy",
) -> Optional[Dict]:
    prob_idx = token_rollout.get("problem_index")
    tok_base = (token_rollout.get("base") or {}).get("completion_raw") or ""
    tok_imps = token_rollout.get("token_importance") or []
    if not tok_imps:
        return None

    # load chunk-side artifacts
    cprob = _load_chunk_problem(chunk_dir)
    if not cprob:
        return None
    full = cprob["full_cot"] or ""
    if require_exact_match and tok_base != full:
        return None  # different base completion → skip (best practice)

    # sentence spans
    spans = _sequential_find_spans(full, cprob["chunks"])
    if not spans or len(spans) != len(cprob["chunks"]):
        # if this fails, comparison would be unreliable
        return None

    # sentence importances (use per-problem labeled if present, else fallback to zeros)
    labeled = cprob["labeled"] or []
    sent_imp_map = {c.get("chunk_idx"): c.get(use_chunk_metric, 0.0) for c in labeled}
    # Only keep indices that actually exist in chunks.json
    sent_values = [sent_imp_map.get(i, 0.0) for i in range(len(cprob["chunks"]))]

    # choose top-K "most harmful" sentences (lowest metric)
    harmful_sent_idx = set(_top_k_indices(sent_values, top_sents, reverse=False))

    # token → sentence mapping
    def _sent_of_offset(off: Optional[int]) -> Optional[int]:
        if not isinstance(off, int):
            return None
        # binary search would be fine, len is small; do linear
        for si, (s, e) in enumerate(spans):
            if s <= off < e:
                return si
        return None

    # choose top-K "most harmful" tokens (lowest delta_acc)
    token_values = [ti.get("delta_acc", 0.0) for ti in tok_imps]
    tok_order = _top_k_indices(token_values, top_tokens, reverse=False)
    chosen_tokens = [tok_imps[i] for i in tok_order]

    # compute overlaps
    token_hits = 0
    token_in_which_sent = []
    for rec in chosen_tokens:
        sidx = _sent_of_offset(rec.get("text_offset"))
        token_in_which_sent.append(sidx)
        if sidx is not None and sidx in harmful_sent_idx:
            token_hits += 1

    # sentence coverage: how many harmful sentences contain ≥1 harmful token
    harmful_sents_with_token = (
        set([s for s in token_in_which_sent if s is not None]) & harmful_sent_idx
    )

    return {
        "problem_index": prob_idx,
        "n_tokens_considered": len(chosen_tokens),
        "n_sentences_considered": len(harmful_sent_idx),
        "token_overlap_fraction": (token_hits / len(chosen_tokens))
        if chosen_tokens
        else 0.0,
        "sentence_recall_fraction": (
            len(harmful_sents_with_token) / len(harmful_sent_idx)
        )
        if harmful_sent_idx
        else 0.0,
        "token_hits": int(token_hits),
        "sent_hits": int(len(harmful_sents_with_token)),
        "harmful_sentence_indices": sorted(list(harmful_sent_idx)),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Compare token- vs sentence-level importance overlap."
    )
    ap.add_argument(
        "--token_root",
        required=True,
        help="Root produced by token script (…/samples_*/)",
    )
    ap.add_argument(
        "--chunk_correct_root",
        required=False,
        help="Chunk pipeline root for correct_base_solution",
    )
    ap.add_argument(
        "--chunk_incorrect_root",
        required=False,
        help="Chunk pipeline root for incorrect_base_solution",
    )
    ap.add_argument("--top_tokens", type=int, default=20)
    ap.add_argument("--top_sentences", type=int, default=5)
    ap.add_argument(
        "--relax-match",
        action="store_true",
        help="Allow base completion mismatch (not recommended).",
    )
    ap.add_argument("--out", default="comparison_out")
    args = ap.parse_args()

    token_root = Path(args.token_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index chunk roots by problem id
    chunk_dirs: Dict[str, Path] = {}
    for root in filter(None, [args.chunk_correct_root, args.chunk_incorrect_root]):
        base = Path(root)
        for d in base.glob("problem_*"):
            if d.is_dir():
                chunk_dirs[d.name] = d

    results = []
    for d in tqdm(sorted(token_root.glob("problem_*")), desc="Comparing"):
        ra = d / "rollout_analysis.json"
        if not ra.exists():
            continue
        tok = _read_json(ra)
        prob_id = d.name
        chunk_dir = chunk_dirs.get(prob_id)
        if not chunk_dir:
            continue
        rec = compare_one_problem(
            tok,
            chunk_dir,
            top_tokens=args.top_tokens,
            top_sents=args.top_sentences,
            require_exact_match=(not args.relax_match),
        )
        if rec:
            results.append(rec)

    # Save
    json.dump(
        results, open(out_dir / "overlap_results.json", "w", encoding="utf-8"), indent=2
    )

    # Aggregate summary
    if results:
        df = pd.DataFrame(results)
        summary = {
            "n_problems": int(len(results)),
            "mean_token_overlap": float(df["token_overlap_fraction"].mean()),
            "mean_sentence_recall": float(df["sentence_recall_fraction"].mean()),
            "median_token_overlap": float(df["token_overlap_fraction"].median()),
            "median_sentence_recall": float(df["sentence_recall_fraction"].median()),
        }
        json.dump(
            summary, open(out_dir / "summary.json", "w", encoding="utf-8"), indent=2
        )
        df.to_csv(out_dir / "overlap_results.csv", index=False)

        print("\nSummary:")
        for k, v in summary.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.3f}")
            else:
                print(f"  {k}: {v}")
    else:
        print("No comparable problems found.")


if __name__ == "__main__":
    main()
