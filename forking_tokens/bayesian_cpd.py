#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bayesian_cpd.py
---------------
Post-process a rollout_analysis.json (from generate_rollout.py) to:
  1) Run Bayesian multi-change-point detection on the drift time series y_t
     using Rbeast/BEAST (R). We extract:
        - p_m_posterior: posterior over number of change points p(m|y)
        - cp_prob_by_t : change occurrence prob p(τ=t|y) for each token index t
        - bayes_factor : p(m>=1|y) / p(m=0|y)
        - forking_indices by BF>9 and p(τ=t|y) >= tau_threshold (default 0.7),
          optionally keeping only local maxima (--local-max-only).
  2) Token-value hazard analysis:
        - hazard h(t) = sum_{w: L2(otw, otw*) > ε} p(w | prefix)
        - survival S(t) = prod_{t'<=t} (1 - h(t'))
        - failure_cdf F(t) = 1 - S(t)

Outputs a compact JSON next to each input file:
   <rollout_analysis.json>.cpd.json

Requirements:
  - R (>= 3.6), package Rbeast installed  (install.packages("Rbeast"))
  - Optional: Python rpy2 (faster in-process). Otherwise we shell out to Rscript.

This script uses ONLY the minimal fields written by generate_rollout.py:
  outcome_distributions.ot   : [{ "t", "token", "dist": { outcome: prob } }]
  outcome_distributions.otw  : [{ "t", "w", "p_w", "dist": { outcome: prob } }]
  drift_series               : [{ "t", "drift" }]
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable, DefaultDict
from collections import defaultdict

# ----------------------------
# Utilities
# ----------------------------


def l2_distance(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    keys = set(dist_a.keys()).union(dist_b.keys())
    return math.sqrt(sum((dist_a.get(k, 0.0) - dist_b.get(k, 0.0)) ** 2 for k in keys))


def _normalize(d: Dict[str, float]) -> Dict[str, float]:
    tot = sum(v for v in d.values() if isinstance(v, (int, float)))
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in d.items() if isinstance(v, (int, float))}


def local_maxima(indices: List[int], values: List[float]) -> List[int]:
    keep = []
    n = len(values)
    for idx in indices:
        left = values[idx - 1] if idx - 1 >= 0 else -float("inf")
        mid = values[idx]
        right = values[idx + 1] if idx + 1 < n else -float("inf")
        if mid >= left and mid >= right:
            keep.append(idx)
    return keep


# ----------------------------
# Load minimal rollout fields
# ----------------------------


def load_rollout_minimal(
    path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns (ot, otw, drift_series)
      ot   : list of { "t": int, "token": str, "dist": { outcome: prob } }
      otw  : list of { "t": int, "w": str, "p_w": float, "dist": { outcome: prob } }
      drift_series : list of { "t": int, "drift": float }  (optional; may be empty)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    outd = data.get("outcome_distributions") or {}
    ot = outd.get("ot") or []
    otw = outd.get("otw") or []
    drift_series = data.get("drift_series") or []

    if not isinstance(ot, list) or not isinstance(otw, list):
        raise ValueError(f"{path}: outcome_distributions missing or malformed.")

    # Sort by t to be safe
    ot.sort(key=lambda r: r.get("t", 0))
    otw.sort(key=lambda r: (r.get("t", 0), str(r.get("w", ""))))
    if isinstance(drift_series, list):
        drift_series.sort(key=lambda r: r.get("t", 0))

    return ot, otw, drift_series


def get_y_series(drift_series: List[Dict[str, Any]]) -> Tuple[List[int], List[float]]:
    # Prefer drift_series if present (already computed against o0)
    ts = [int(r.get("t", i)) for i, r in enumerate(drift_series)]
    ys = [float(r.get("drift", 0.0)) for r in drift_series]
    return ts, ys


# ----------------------------
# Rbeast (R) – rpy2 (fast) or Rscript fallback
# ----------------------------


def run_beast_rpy2(y: List[float]) -> Tuple[List[float], List[float]]:
    """
    Returns:
      cpOccPr: p(τ=t|y) for each t (same length as y)
      ncpPr  : posterior over the number of trend CPs (index 0 => m=0)
    """
    import numpy as np
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr

    numpy2ri.activate()
    Rbeast = importr("Rbeast")  # package name is Rbeast; function is beast()
    r = Rbeast.beast(np.asarray(y, dtype=float), season="none")
    # Extract: trend$cpOccPr and trend$ncpPr
    trend = r.rx2("trend")
    cpOccPr = list(trend.rx2("cpOccPr"))
    ncpPr = list(trend.rx2("ncpPr"))
    return cpOccPr, ncpPr


def run_beast_rscript(y: List[float]) -> Tuple[List[float], List[float]]:
    """
    Use Rscript + Rbeast. Writes y to temp file, reads CSV outputs.
    """
    if not shutil.which("Rscript"):
        raise RuntimeError(
            "Rscript not found on PATH, and rpy2 unavailable. Please install R & Rbeast."
        )

    with tempfile.TemporaryDirectory() as td:
        y_txt = Path(td) / "y.txt"
        cp_csv = Path(td) / "cp.csv"
        ncp_csv = Path(td) / "ncp.csv"
        rfile = Path(td) / "run_beast.R"

        # Write y as one value per line
        with open(y_txt, "w", encoding="utf-8") as f:
            for v in y:
                f.write(f"{float(v)}\n")

        r_code = f"""
                    suppressPackageStartupMessages(library(Rbeast))
                    y <- scan("{y_txt.as_posix()}", what=double())
                    fit <- beast(y, season='none')
                    cp  <- fit$trend$cpOccPr
                    ncp <- fit$trend$ncpPr
                    write.csv(data.frame(cpOccPr = cp),  file="{cp_csv.as_posix()}",  row.names=FALSE)
                    write.csv(data.frame(ncpPr  = ncp), file="{ncp_csv.as_posix()}", row.names=FALSE)
        """
        rfile.write_text(r_code, encoding="utf-8")
        p = subprocess.run(
            ["Rscript", rfile.as_posix()], capture_output=True, text=True
        )
        if p.returncode != 0:
            raise RuntimeError(f"Rscript failed: {p.stderr.strip()}")

        import csv

        cpOccPr, ncpPr = [], []
        with open(cp_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cpOccPr.append(float(row["cpOccPr"]))
        with open(ncp_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ncpPr.append(float(row["ncpPr"]))
        return cpOccPr, ncpPr


def run_beast(y: List[float]) -> Tuple[List[float], List[float]]:
    """
    Try rpy2 first; fallback to Rscript.
    """
    # Filter NaNs/Infs
    clean_y = [
        0.0
        if (not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v))
        else float(v)
        for v in y
    ]
    try:
        return run_beast_rpy2(clean_y)
    except Exception as e_rpy2:
        try:
            return run_beast_rscript(clean_y)
        except Exception as e_r:
            raise RuntimeError(
                "Failed to run Rbeast via rpy2 or Rscript. "
                "Install rpy2 or ensure R (with Rbeast) is available.\n"
                f"rpy2 error: {e_rpy2}\nRscript error: {e_r}"
            )


# ----------------------------
# CPD + Forking index selection
# ----------------------------


def compute_cpd(
    ts: List[int],
    ys: List[float],
    tau_prob_threshold: float,
    bf_threshold: float,
    local_max_only: bool,
) -> Dict[str, Any]:
    """
    Run Rbeast on y(t). Build outputs exactly like the paper:
      - p(m|y) via trend$ncpPr
      - p(τ=t|y) via trend$cpOccPr
      - BF = p(m>=1|y) / p(m=0|y)
      - Forking indices: BF>bf_threshold and p(τ=t|y) >= tau_prob_threshold (optionally local maxima)
    """
    if not ts or not ys or len(ts) != len(ys):
        raise ValueError("y-series empty or misaligned")

    cpOccPr, ncpPr = run_beast(ys)  # same length as ys; ncpPr length is (max+1), 0..m
    if len(cpOccPr) != len(ys):
        # BEAST can pad ends; trim or extend
        L = min(len(cpOccPr), len(ys))
        cpOccPr = cpOccPr[:L]
        ts = ts[:L]
        ys = ys[:L]

    # Posterior over number of changes: by convention ncpPr[0] = P(m=0)
    p_m0 = float(ncpPr[0]) if ncpPr else 1.0
    p_m_ge1 = max(0.0, 1.0 - p_m0)
    bayes_factor = (p_m_ge1 / p_m0) if p_m0 > 0 else float("inf")

    cp_prob_by_t = [{"t": int(t), "p_change": float(p)} for t, p in zip(ts, cpOccPr)]

    # Indices where p(τ=t|y) >= tau_prob_threshold
    idxs = [i for i, p in enumerate(cpOccPr) if p >= tau_prob_threshold]
    if local_max_only and idxs:
        idxs = local_maxima(idxs, cpOccPr)

    # Apply BF decision rule
    forking_indices = [int(ts[i]) for i in idxs] if bayes_factor > bf_threshold else []

    return {
        "p_m_posterior": [{"m": i, "prob": float(p)} for i, p in enumerate(ncpPr)],
        "bayes_factor": float(bayes_factor),
        "cp_prob_by_t": cp_prob_by_t,
        "forking_indices": forking_indices,
        "thresholds": {
            "bf_threshold": float(bf_threshold),
            "tau_prob_threshold": float(tau_prob_threshold),
            "local_max_only": bool(local_max_only),
        },
    }


# ----------------------------
# Hazards & survival
# ----------------------------


def hazards_and_survival(
    ot: List[Dict[str, Any]], otw: List[Dict[str, Any]], epsilon: float
) -> Dict[str, Any]:
    """
    Build hazard h(t), survival S(t)=Π(1-h), failure CDF F(t)=1-S(t).
    otw has, for each t, one record with the base greedy token w* (w==ot[t]["token"])
    and other records for alternate tokens (with p_w).
    """
    # Group otw by t; also map base token per t from ot
    token_by_t = {
        int(rec.get("t", i)): str(rec.get("token", "")) for i, rec in enumerate(ot)
    }
    otw_by_t: DefaultDict[int, List[Dict[str, Any]]] = defaultdict(list)
    for rec in otw:
        otw_by_t[int(rec.get("t", 0))].append(rec)

    ts_sorted = sorted(otw_by_t.keys())
    hazards = []
    survival = []
    failure_cdf = []
    S = 1.0

    for t in ts_sorted:
        w_star = token_by_t.get(t, None)

        # Find base branch distribution
        base_rec = None
        for rec in otw_by_t[t]:
            if w_star is not None and str(rec.get("w", "")) == w_star:
                base_rec = rec
                break
        if base_rec is None:
            # fallback: choose the branch with maximum p_w as "base"
            base_rec = max(
                otw_by_t[t], key=lambda r: float(r.get("p_w", 0.0)), default=None
            )
        base_dist = (base_rec or {}).get("dist", {}) or {}

        # hazard: sum p_w over alternates whose L2 > epsilon
        h = 0.0
        for rec in otw_by_t[t]:
            if rec is base_rec:
                continue
            dist_w = rec.get("dist", {}) or {}
            d = l2_distance(dist_w, base_dist)
            if d > float(epsilon):
                h += float(rec.get("p_w", 0.0))
        # Clamp [0,1]
        h = min(max(h, 0.0), 1.0)
        hazards.append({"t": t, "hazard": h})

        S *= 1.0 - h
        S = max(min(S, 1.0), 0.0)
        survival.append({"t": t, "S": S})
        failure_cdf.append({"t": t, "F": 1.0 - S})

    return {
        "epsilon": float(epsilon),
        "hazard_by_t": hazards,
        "survival_by_t": survival,
        "failure_cdf_by_t": failure_cdf,
    }


# ----------------------------
# CLI
# ----------------------------


def process_one_file(path: Path, args: argparse.Namespace) -> Path:
    ot, otw, drift = load_rollout_minimal(path)
    ts, ys = get_y_series(ot, drift)  # y_t (semantic drift)

    cpd = compute_cpd(
        ts=ts,
        ys=ys,
        tau_prob_threshold=args.tau_threshold,
        bf_threshold=args.bf_threshold,
        local_max_only=args.local_max_only,
    )
    haz = hazards_and_survival(ot=ot, otw=otw, epsilon=args.epsilon)

    out = {
        "input_file": str(path),
        "cpd": cpd,
        "hazards": haz,
    }
    out_path = path.with_suffix(path.suffix + ".cpd.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    ap = argparse.ArgumentParser(
        description="Bayesian CPD and token-value hazard analysis for Forking Paths."
    )
    ap.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to a rollout_analysis.json or a directory containing such files.",
    )
    ap.add_argument(
        "--epsilon",
        "-e",
        type=float,
        default=0.6,
        help="L2 threshold ε for token-value hazard.",
    )
    ap.add_argument(
        "--tau-threshold",
        type=float,
        default=0.7,
        help="Threshold on p(τ=t|y) to flag candidate forking tokens.",
    )
    ap.add_argument(
        "--bf-threshold",
        type=float,
        default=9.0,
        help="Bayes factor threshold; BF>threshold => '≥1 change' over 'no change'.",
    )
    ap.add_argument(
        "--local-max-only",
        action="store_true",
        help="Keep only local maxima among τ candidates.",
    )
    args = ap.parse_args()

    p = Path(args.input)
    to_process: List[Path] = []
    if p.is_dir():
        for fp in p.rglob("rollout_analysis.json"):
            to_process.append(fp)
        if not to_process:
            print(f"No rollout_analysis.json found under: {p}")
            sys.exit(1)
    elif p.is_file():
        to_process = [p]
    else:
        print(f"Input path not found: {p}")
        sys.exit(1)

    print(f"Found {len(to_process)} file(s). Running CPD + hazards...")
    written: List[Path] = []
    for fp in to_process:
        try:
            outp = process_one_file(fp, args)
            written.append(outp)
            print(f"Wrote: {outp}")
        except Exception as ex:
            print(f"[ERROR] {fp}: {ex}")

    if not written:
        sys.exit(2)


if __name__ == "__main__":
    main()
