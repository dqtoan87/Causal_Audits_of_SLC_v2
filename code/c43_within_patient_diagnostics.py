#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C43 — Chẩn đoán cho thiết kế trong-bệnh-nhân của C41.

Vì sao. C41 tuyên bố định danh bằng thiết kế, nhưng bản thân thiết kế chưa có phép kiểm nào.
Phân tích hiệu chỉnh có placebo refuter; thiết kế phân tầng thì chưa. Reviewer sẽ hỏi ba câu.
Kết quả có bị vài tầng lớn chi phối không. Có nhiễu cấp TỔN THƯƠNG biến thiên trong cùng bệnh
nhân mà thiết kế không hấp thụ được không. Và đường ống có tự chế ra hiệu ứng từ nhiễu không.

Ba phần.
  1. Chẩn đoán tầng: số tầng đóng góp, phân bố kích thước tầng và số ca mỗi tầng.
  2. Ảnh hưởng leave-one-stratum-out, xấp xỉ một bước Δβ ≈ H⁻¹ g_s. Chính xác đến bậc nhất và
     không cần fit lại 258 lần cho mỗi concept.
  3. Placebo refuter trong tầng: hoán vị BIẾN XỬ LÝ bên trong từng tầng, giữ nguyên nhãn và mọi
     hiệp biến, nên hiệu ứng thật bằng không theo cấu tạo. Nếu ước lượng vẫn dương thì đường ống
     đang chế ra hiệu ứng. Cho p-value hoán vị hai phía.

Out -> results/c43_within_patient_diagnostics.json
Chạy:  python code/c43_within_patient_diagnostics.py [--perm 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c1_tier1_ate as c1        # noqa: E402
import c41_within_patient as c41  # noqa: E402

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c43_within_patient_diagnostics.json")
GROUP, OUTCOME = c41.GROUP, c41.OUTCOME
SEED = 42


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def hessian(beta, X, y, strata, eps=1e-5):
    p = len(beta); H = np.zeros((p, p))
    for j in range(p):
        bp = beta.copy(); bp[j] += eps
        bm = beta.copy(); bm[j] -= eps
        H[:, j] = (c41.clogit_nll(bp, X, y, strata)[1] -
                   c41.clogit_nll(bm, X, y, strata)[1]) / (2 * eps)
    return 0.5 * (H + H.T)


def strata_diag(X, y, strata):
    sizes = np.array([len(s) for s in strata])
    cases = np.array([int(y[s].sum()) for s in strata])
    q = lambda a, p: float(np.percentile(a, p))
    return {"n_strata": int(len(strata)), "n_rows": int(len(y)), "n_cases": int(y.sum()),
            "stratum_size": {"min": int(sizes.min()), "q25": q(sizes, 25), "median": q(sizes, 50),
                             "q75": q(sizes, 75), "max": int(sizes.max())},
            "cases_per_stratum": {"min": int(cases.min()), "median": q(cases, 50),
                                  "max": int(cases.max()),
                                  "hist": {str(k): int((cases == k).sum())
                                           for k in range(1, int(cases.max()) + 1)}},
            "share_rows_top10_strata": float(np.sort(sizes)[::-1][:10].sum() / sizes.sum())}


def loso_influence(beta, X, y, strata):
    """Δβ ≈ H⁻¹ g_s khi bỏ tầng s. g_s là score của tầng tại beta đã fit."""
    _, _, S = c41.clogit_nll(beta, X, y, strata)
    Hinv = np.linalg.pinv(hessian(beta, X, y, strata))
    d = (S @ Hinv.T)[:, 0]          # thay đổi của hệ số điều trị khi bỏ từng tầng
    k = int(np.argmax(np.abs(d)))
    return {"max_abs_delta_log_or": float(np.abs(d).max()),
            "max_abs_delta_as_pct_of_beta": float(100 * np.abs(d).max() / abs(beta[0]))
            if abs(beta[0]) > 1e-9 else None,
            "worst_stratum_index": k, "worst_stratum_size": int(len(strata[k])),
            "worst_stratum_cases": int(y[strata[k]].sum()),
            "n_strata_moving_beta_over_5pct": int((np.abs(d) > 0.05 * abs(beta[0])).sum())
            if abs(beta[0]) > 1e-9 else None}


def perm_refuter(beta, X, y, strata, n_perm, rng):
    """Hoán vị cột điều trị BÊN TRONG từng tầng. Nhãn và hiệp biến giữ nguyên."""
    obs = float(beta[0]); null = []
    Xp = X.copy()
    for t in range(n_perm):
        for s in strata:
            Xp[s, 0] = X[s, 0][rng.permutation(len(s))]
        try:
            b, _ = c41.fit_clogit(Xp, y, strata, beta0=np.zeros(X.shape[1]), maxiter=60)
            null.append(float(b[0]))
        except Exception:
            pass
        if (t + 1) % 50 == 0:
            log(f"      perm {t+1}/{n_perm}")
    null = np.array(null)
    if len(null) < 10:
        return {"n_perm": int(len(null)), "error": "too few successful permutations"}
    return {"n_perm": int(len(null)), "observed_log_or": obs,
            "null_mean_log_or": float(null.mean()), "null_sd_log_or": float(null.std(ddof=1)),
            "null_or_ci": [float(np.exp(np.percentile(null, 2.5))),
                           float(np.exp(np.percentile(null, 97.5)))],
            "p_two_sided": float((np.abs(null) >= abs(obs)).mean()),
            "null_centred_on_zero": bool(abs(null.mean()) < 2 * null.std(ddof=1) / np.sqrt(len(null)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--max-patients", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    data = json.load(open(os.path.join(RES, "tier1_ate.json")))["data"]
    log(f"Load {data}")
    df = pd.read_csv(data, low_memory=False)
    df["ITA_stratum"] = c1.compute_ita(df)
    if args.max_patients:
        g = df.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
        keep = g[(g["sum"] > 0) & (g["sum"] < g["count"])].index[: args.max_patients]
        df = df[df[GROUP].isin(keep)].reset_index(drop=True)
        log(f"  smoke test: {args.max_patients} bệnh nhân, {len(df):,} tile")

    rng = np.random.RandomState(SEED)
    out = []
    for c in c1.CONCEPTS:
        prep = c41.prepare(df, c["col"], adjust=True)
        if prep is None:
            continue
        X, y, strata, _ = prep
        beta, _ = c41.fit_clogit(X, y, strata)
        log(f"  {c['name']:15s} OR={np.exp(beta[0]):.3f}  strata={len(strata)}  cases={int(y.sum())}")
        rec = {"concept": c["name"], "column": c["col"], "or": float(np.exp(beta[0])),
               "strata": strata_diag(X, y, strata),
               "influence": loso_influence(beta, X, y, strata),
               "permutation": perm_refuter(beta, X, y, strata, args.perm, rng)}
        out.append(rec)
        p = rec["permutation"]
        log(f"      influence max Δlog OR = {rec['influence']['max_abs_delta_log_or']:.4f}"
            f" | perm p = {p.get('p_two_sided')} | null OR CI "
            f"{[round(v,3) for v in p.get('null_or_ci',[float('nan')]*2)]}")
        json.dump({"status": "partial", "concepts": out}, open(OUT + ".partial", "w"), indent=1)

    res = {"phase": "C43_within_patient_diagnostics", "data": data, "seed": SEED,
           "n_perm": args.perm,
           "note": ("Diagnostics for the within-patient conditional design of C41. Stratum "
                    "composition, one-step leave-one-stratum-out influence on the treatment "
                    "coefficient, and a within-stratum permutation refuter in which the treatment "
                    "column is permuted inside each stratum so that the true effect is null by "
                    "construction."),
           "concepts": out, "runtime_sec": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    if os.path.exists(OUT + ".partial"):
        os.remove(OUT + ".partial")
    log(f"saved -> {OUT}  ({res['runtime_sec']}s)")

    print()
    print(f"{'concept':16s}{'OR':>8s}{'strata':>8s}{'max Δlog OR':>13s}{'perm p':>9s}{'null OR CI':>22s}")
    for r in out:
        p = r["permutation"]; ci = p.get("null_or_ci", [float("nan")] * 2)
        print(f"{r['concept']:16s}{r['or']:8.3f}{r['strata']['n_strata']:8d}"
              f"{r['influence']['max_abs_delta_log_or']:13.4f}"
              f"{p.get('p_two_sided', float('nan')):9.3f}"
              f"{f'[{ci[0]:.3f}, {ci[1]:.3f}]':>22s}")


if __name__ == "__main__":
    main()
