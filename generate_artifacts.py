#!/usr/bin/env python3
"""
Reproducible synthetic benchmark + figures/tables used in the IJCA draft:
- Table: QRNG variations -> observable symptoms -> affected bound -> Hmin intuition
- Table: Synthetic case definitions (artifact parameters)
- Table: Aggregated proxy validation results (N=200k bits, multiple seeds)

This is NOT a full SP 800-90B implementation. It is a *proxy* benchmark that mirrors the paper's
framework: (IID vs non-IID screening) + conservative min-entropy bounds (MCV + first-order Markov)
and basic health-test indicators (APT/RCT proxies).
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


# -------------------------
# Utilities
# -------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def hmin_from_maxprob(x: float) -> float:
    x = float(x)
    x = min(max(x, 1e-12), 1.0)
    return -math.log(x, 2)

def lag1_corr(bits: np.ndarray) -> float:
    """Pearson correlation of x[t] and x[t-1] for bits in {0,1}."""
    x = bits.astype(np.float64)
    if len(x) < 2:
        return 0.0
    a = x[1:]
    b = x[:-1]
    sa = a.std()
    sb = b.std()
    if sa == 0.0 or sb == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])

def markov_counts(bits: np.ndarray) -> Tuple[int,int,int,int,int,int]:
    """Return N, N0, N1, N01, N10, N00, N11 counts."""
    x = bits.astype(np.uint8)
    N = len(x)
    N0 = int((x == 0).sum())
    N1 = N - N0
    if N < 2:
        return N, N0, N1, 0, 0, 0, 0
    prev = x[:-1]
    curr = x[1:]
    N01 = int(((prev == 0) & (curr == 1)).sum())
    N10 = int(((prev == 1) & (curr == 0)).sum())
    N00 = int(((prev == 0) & (curr == 0)).sum())
    N11 = int(((prev == 1) & (curr == 1)).sum())
    return N, N0, N1, N01, N10, N00, N11

def hmin_mcv(bits: np.ndarray) -> Tuple[float,float]:
    N, N0, N1, *_ = markov_counts(bits)
    p_max = max(N0 / max(N,1), N1 / max(N,1))
    return hmin_from_maxprob(p_max), p_max

def hmin_markov(bits: np.ndarray) -> Tuple[float, Dict[str,float]]:
    N, N0, N1, N01, N10, N00, N11 = markov_counts(bits)
    # avoid division by zero
    a_hat = N01 / N0 if N0 > 0 else 0.0  # P(0->1)
    b_hat = N10 / N1 if N1 > 0 else 0.0  # P(1->0)
    # conservative max conditional probability term (q_max)
    q_max = max(1-a_hat, a_hat, b_hat, 1-b_hat)
    return hmin_from_maxprob(q_max), {"a_hat": a_hat, "b_hat": b_hat, "q_max": q_max}

def iid_proxy(bits: np.ndarray, bias_thr: float, lag1_thr: float) -> bool:
    """Simple IID screening proxy: |p1-0.5| <= bias_thr and |lag1| <= lag1_thr."""
    p1 = float(bits.mean())
    l1 = abs(lag1_corr(bits))
    return (abs(p1 - 0.5) <= bias_thr) and (l1 <= lag1_thr)

def apt_proxy_alarm(bits: np.ndarray, W: int, z: float, p_nom: float=0.5) -> float:
    """
    APT proxy: sliding windows; alarm if count(1) outside [L,U] where
    L=Wp - z*sqrt(W p(1-p)), U=Wp + z*sqrt(W p(1-p))
    Returns alarm rate (fraction of windows triggering).
    """
    x = bits.astype(np.uint8)
    if len(x) < W:
        return 0.0
    counts = np.convolve(x, np.ones(W, dtype=np.uint32), mode="valid")
    sigma = math.sqrt(W * p_nom * (1-p_nom))
    L = W * p_nom - z * sigma
    U = W * p_nom + z * sigma
    alarms = (counts < L) | (counts > U)
    return float(alarms.mean())

def rct_proxy_threshold(p_max: float, alpha: float) -> int:
    """Geometric tail proxy: C = ceil(log(alpha) / log(p_max))."""
    p_max = min(max(p_max, 1e-12), 1.0-1e-12)
    return int(math.ceil(math.log(alpha) / math.log(p_max)))

def rct_proxy_alarm(bits: np.ndarray, C: int) -> bool:
    """Alarm if any run of identical values length >= C."""
    x = bits.astype(np.uint8)
    if len(x) == 0:
        return False
    run = 1
    for i in range(1, len(x)):
        if x[i] == x[i-1]:
            run += 1
            if run >= C:
                return True
        else:
            run = 1
    return False


# -------------------------
# Synthetic generators
# -------------------------
def gen_iid_bias(N: int, p1: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(N) < p1).astype(np.uint8)

def gen_markov(N: int, a: float, b: float, rng: np.random.Generator, x0: int|None=None) -> np.ndarray:
    """
    Two-state Markov chain with P(0->1)=a, P(1->0)=b.
    """
    if x0 is None:
        x = 1 if rng.random() < 0.5 else 0
    else:
        x = int(x0) & 1
    out = np.empty(N, dtype=np.uint8)
    for i in range(N):
        out[i] = x
        u = rng.random()
        if x == 0:
            x = 1 if u < a else 0
        else:
            x = 0 if u < b else 1
    return out

def gen_drift(N: int, p_start: float, p_end: float, rng: np.random.Generator) -> np.ndarray:
    """
    Slow drift in Bernoulli parameter over time: p(t) linear.
    """
    t = np.linspace(0.0, 1.0, N, dtype=np.float64)
    p = p_start + (p_end - p_start) * t
    return (rng.random(N) < p).astype(np.uint8)


# -------------------------
# Figures
# -------------------------


# -------------------------
# Tables
# -------------------------
def table_qrng_variations() -> pd.DataFrame:
    # NOTE: Use plain ASCII so the table renders correctly in Excel on systems/fonts that
    # don't include certain Unicode glyphs (e.g., arrows, minus sign, Greek letters).
    return pd.DataFrame([
        ["Efficiency mismatch / offset", "Bias (P(1) != 0.5)", "IID min-entropy (MCV/IID)",
         "p_max increases -> H_min decreases"],
        ["Dead time (refractory)", "Low transition rate, run anomalies", "Markov/non-IID bound",
         "(1-a) increases -> max(1-a, a, b, 1-b) increases -> H_min decreases"],
        ["Afterpulsing", "Lag-1 positive correlation", "Markov/non-IID bound",
         "a increases -> max(1-a, a, b, 1-b) increases -> H_min decreases"],
        ["Optical power drift", "Non-stationary bias over time", "APT + conservative bound",
         "Windowed proportion deviates from 0.5 -> alarms increase -> conservative bound tightens"],
    ], columns=["Variation (QRNG)", "Observable symptom", "Primary bound affected", "Effect on H_min (intuition)"])


@dataclass
class CaseDef:
    ID: str
    family: str   # IID_BIAS / MARKOV / DRIFT
    params: Dict[str, float]

def default_cases() -> List[CaseDef]:
    # These cases are intentionally simple and match the paper-style narrative.
    return [
        CaseDef("C1", "IID_BIAS", {"p1": 0.50}),
        CaseDef("C2", "IID_BIAS", {"p1": 0.55}),
        CaseDef("Q1", "MARKOV", {"a": 0.05, "b": 0.80}),  # strong dependence
        CaseDef("Q2", "MARKOV", {"a": 0.20, "b": 0.60}),  # moderate dependence
        CaseDef("D1", "DRIFT", {"p_start": 0.50, "p_end": 0.58}),
    ]

def table_case_definitions(cases: List[CaseDef]) -> pd.DataFrame:
    rows=[]
    for c in cases:
        if c.family == "IID_BIAS":
            rows.append([c.ID, c.family, f"p1={c.params['p1']:.3f}", "Bernoulli IID with static bias (imbalance/offset proxy)"])
        elif c.family == "MARKOV":
            rows.append([c.ID, c.family, f"a={c.params['a']:.3f}, b={c.params['b']:.3f}", "First-order Markov dependence (dead-time / afterpulsing proxy)"])
        elif c.family == "DRIFT":
            rows.append([c.ID, c.family, f"p_start={c.params['p_start']:.3f}, p_end={c.params['p_end']:.3f}", "Windowed drift / non-stationarity proxy"])
        else:
            rows.append([c.ID, c.family, str(c.params), ""])
    return pd.DataFrame(rows, columns=["ID", "Family", "Parameters", "Description"])

def run_benchmark(
    cases: List[CaseDef],
    N: int,
    seeds: int,
    bias_thr: float,
    lag1_thr: float,
    apt_W: int,
    apt_z: float,
    rct_alpha: float,
) -> pd.DataFrame:
    rows=[]
    for c in cases:
        iid_passes=[]
        lag1s=[]
        p1s=[]
        hmins=[]
        apt_rates=[]
        rct_alarms=[]
        for s in range(seeds):
            rng = np.random.default_rng(12345 + s)
            if c.family == "IID_BIAS":
                bits = gen_iid_bias(N, c.params["p1"], rng)
            elif c.family == "MARKOV":
                bits = gen_markov(N, c.params["a"], c.params["b"], rng)
            elif c.family == "DRIFT":
                bits = gen_drift(N, c.params["p_start"], c.params["p_end"], rng)
            else:
                raise ValueError(f"Unknown family: {c.family}")

            p1 = float(bits.mean())
            l1 = float(lag1_corr(bits))
            iid = iid_proxy(bits, bias_thr=bias_thr, lag1_thr=lag1_thr)

            h_mcv, p_max = hmin_mcv(bits)
            h_mkv, mkv = hmin_markov(bits)

            # conservative selection: if IID proxy pass -> MCV else Markov (simple rule)
            h_bound = h_mcv if iid else h_mkv

            apt_rate = apt_proxy_alarm(bits, W=apt_W, z=apt_z, p_nom=0.5)
            C = rct_proxy_threshold(p_max=max(p_max, mkv["q_max"]), alpha=rct_alpha)
            rct_alarm = 1.0 if rct_proxy_alarm(bits, C=C) else 0.0

            iid_passes.append(1.0 if iid else 0.0)
            lag1s.append(l1)
            p1s.append(p1)
            hmins.append(h_bound)
            apt_rates.append(apt_rate)
            rct_alarms.append(rct_alarm)

        rows.append({
            "ID": c.ID,
            "IID_pass_rate": float(np.mean(iid_passes)),
            "Lag1_mean": float(np.mean(lag1s)),
            "p1_mean": float(np.mean(p1s)),
            "Hmin_bound_mean": float(np.mean(hmins)),
            "APT_alarm_rate": float(np.mean(apt_rates)),
            "RCT_alarm_rate": float(np.mean(rct_alarms)),
            "N_bits": N,
            "Seeds": seeds,
        })
    return pd.DataFrame(rows)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="out", help="Output directory")
    ap.add_argument("--N", type=int, default=200_000, help="Bits per case")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    ap.add_argument("--bias_thr", type=float, default=0.02, help="IID proxy bias threshold |p1-0.5|")
    ap.add_argument("--lag1_thr", type=float, default=0.05, help="IID proxy lag-1 correlation threshold")
    ap.add_argument("--apt_W", type=int, default=1024, help="APT window length (proxy)")
    ap.add_argument("--apt_z", type=float, default=4.0, help="APT z-score (proxy)")
    ap.add_argument("--rct_alpha", type=float, default=1e-6, help="RCT false-alarm target (proxy)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ensure_dir(outdir)
    ensure_dir(outdir / "figures")
    ensure_dir(outdir / "tables")

    # Figures

    # Tables
    t1 = table_qrng_variations()
    t1.to_csv(outdir / "tables" / "table_qrng_variations.csv", index=False)

    cases = default_cases()
    t2 = table_case_definitions(cases)
    t2.to_csv(outdir / "tables" / "table_case_definitions.csv", index=False)

    t3 = run_benchmark(
        cases=cases,
        N=args.N,
        seeds=args.seeds,
        bias_thr=args.bias_thr,
        lag1_thr=args.lag1_thr,
        apt_W=args.apt_W,
        apt_z=args.apt_z,
        rct_alpha=args.rct_alpha,
    )
    t3.to_csv(outdir / "tables" / "table_aggregated_results.csv", index=False)

    # Also emit a Markdown snippet for easy pasting into paper
    md = []
    md.append("# Repro Outputs\n")
    md.append("## Figures\n")
    md.append("- fig_validation_flow_black.png\n- fig_hmin_vs_maxprob.png\n")
    md.append("## Tables\n")
    md.append("- table_qrng_variations.csv\n- table_case_definitions.csv\n- table_aggregated_results.csv\n")
    (outdir / "REPRO_OUTPUTS.md").write_text("\n".join(md), encoding="utf-8")

    print("Done. Outputs written to:", outdir.resolve())

if __name__ == "__main__":
    main()
