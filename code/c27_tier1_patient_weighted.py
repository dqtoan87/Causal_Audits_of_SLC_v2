#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CausalXAI v14 — C27: Tier-1 với TRỌNG SỐ BẰNG NHAU THEO BỆNH NHÂN.

Vì sao cần. Estimand của Tier 1 là ATE mức TILE: trung bình lấy trên phân phối thực nghiệm của
các tile trong cohort. Bootstrap theo cụm sửa phần SUY DIỄN (các tile của một bệnh nhân không độc
lập) nhưng KHÔNG đổi estimand. Bệnh nhân có nhiều tile vẫn có trọng số lớn hơn. Một bệnh nhân đóng
góp hàng trăm tile có thể chi phối ước lượng.

Script này giữ nguyên mọi thứ của C1 (dichotomy tertile, tập confounder, GBM, cross-fit theo bệnh
nhân, trimming) và chỉ đổi TRỌNG SỐ khi lấy trung bình: mỗi tile nhận trọng số 1/(số tile của bệnh
nhân đó), nên mọi bệnh nhân có tổng trọng số bằng nhau. So sánh dấu, độ lớn và khoảng tin cậy với
bản mức tile để xem kết luận có phụ thuộc cách đánh trọng số không.

Chỉ chạy ba proxy khẳng định (color, size, contrast).

Out -> results/c27_tier1_patient_weighted.json
Chạy:  .venv/bin/python Causal_XAI_v3/code/c27_tier1_patient_weighted.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c1_tier1_ate as c1   # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c27_tier1_patient_weighted.json")
CONFIRMATORY = ["C_color", "D_size", "contrast_LB"]
N_BOOT = 1000


def weighted_cluster_ci(v, w, groups, n_boot, seed):
    """Trung bình có trọng số + bootstrap cụm theo bệnh nhân.

    Gộp trước theo bệnh nhân thành (tổng w·v, tổng w) rồi tái chọn bệnh nhân, nên bootstrap rẻ và
    tôn trọng đúng cấu trúc phụ thuộc.
    """
    d = pd.DataFrame({"v": v, "w": w, "g": groups}).dropna()
    d["wv"] = d["w"] * d["v"]
    agg = d.groupby("g")[["wv", "w"]].sum()
    S, W = agg["wv"].to_numpy(), agg["w"].to_numpy()
    point = float(S.sum() / W.sum())
    rng = np.random.default_rng(seed); G = len(S)
    b = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, G, G)
        b[k] = S[i].sum() / W[i].sum()
    lo, hi = np.percentile(b, [2.5, 97.5])
    return point, float(lo), float(hi)


def weighted_ratio_ci(ey1, ey0, w, groups, n_boot, seed):
    """RR = trung bình có trọng số của ey1 chia trung bình có trọng số của ey0, cùng một lần tái chọn."""
    d = pd.DataFrame({"a": ey1, "b": ey0, "w": w, "g": groups}).dropna()
    agg = d.assign(wa=d.w * d.a, wb=d.w * d.b).groupby("g")[["wa", "wb", "w"]].sum()
    A, B, W = agg["wa"].to_numpy(), agg["wb"].to_numpy(), agg["w"].to_numpy()
    point = float(A.sum() / B.sum())
    rng = np.random.default_rng(seed); G = len(A)
    b = np.empty(n_boot)
    for k in range(n_boot):
        i = rng.integers(0, G, G)
        b[k] = A[i].sum() / max(B[i].sum(), 1e-12)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    t0 = time.time()
    print(f"##### C27 Tier-1 patient-weighted {time.strftime('%H:%M:%S')} #####", flush=True)
    df = pd.read_csv(c1.DATA_CSV, low_memory=False)
    df["ITA_stratum"] = c1.compute_ita(df)
    Zfull = c1.build_Z(df)
    print(f"{len(df):,} tiles / {df[c1.GROUP].nunique():,} patients  ({time.time()-t0:.0f}s)", flush=True)

    res = {"n_rows": int(len(df)), "n_patients": int(df[c1.GROUP].nunique()), "n_boot": N_BOOT,
           "note": ("Tier-1 AIPW re-averaged with every patient given equal total weight, so that "
                    "patients contributing many tiles do not dominate. Nuisance models, cross-fitting, "
                    "dichotomisation and trimming are identical to the tile-weighted analysis; only "
                    "the averaging weights differ."),
           "concepts": {}}

    W = 104
    print("=" * W)
    print(f"{'concept':14s}{'ATE tile':>13s}{'ATE patient':>13s}{'CI patient':>26s}"
          f"{'RR tile':>10s}{'RR patient':>12s}{'sig':>8s}")
    for c in c1.CONCEPTS:
        if c["name"] not in CONFIRMATORY:
            continue
        name, col = c["name"], c["col"]
        x = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = x.quantile(1 / 3), x.quantile(2 / 3)
        mask = (x.notna() & ((x <= q1) | (x >= q3))).to_numpy()
        T = (x >= q3).astype(int).to_numpy()[mask]
        Y = df[c1.OUTCOME].to_numpy()[mask].astype(int)
        g = df[c1.GROUP].to_numpy()[mask]
        Z = Zfull.to_numpy()[mask]
        psi, ey1, ey0, pos = c1.aipw_crossfit(Z, T, Y, g, c1.RANDOM_SEED)

        # trọng số: mỗi bệnh nhân tổng trọng số 1
        cnt = pd.Series(g).value_counts()
        w = (1.0 / pd.Series(g).map(cnt)).to_numpy()

        ate_t, lo_t, hi_t, _ = c1.cluster_stats(psi, g, N_BOOT, c1.RANDOM_SEED)
        ate_p, lo_p, hi_p = weighted_cluster_ci(psi, w, g, N_BOOT, c1.RANDOM_SEED)
        _p1, _p0, rr_t, rlo_t, rhi_t = c1.cluster_ratio_ci(ey1, ey0, g, N_BOOT, c1.RANDOM_SEED)
        rr_p, rlo_p, rhi_p = weighted_ratio_ci(ey1, ey0, w, g, N_BOOT, c1.RANDOM_SEED)
        ev_p = c1.e_value(rr_p if rr_p > 1 else 1 / max(rr_p, 1e-9))
        sig_p = bool(lo_p > 0 or hi_p < 0)

        res["concepts"][name] = {
            "column": col, "n": int(mask.sum()),
            "tile": {"ate": ate_t, "ci": [lo_t, hi_t], "rr": rr_t, "rr_ci": [rlo_t, rhi_t]},
            "patient": {"ate": ate_p, "ci": [lo_p, hi_p], "rr": rr_p, "rr_ci": [rlo_p, rhi_p],
                        "e_value": ev_p, "significant": sig_p},
            "ratio_patient_to_tile": float(ate_p / ate_t) if ate_t else None}
        print(f"{name:14s}{ate_t*1e3:>12.2f}‰{ate_p*1e3:>12.2f}‰"
              f"{('[%.2f, %.2f]‰' % (lo_p*1e3, hi_p*1e3)):>26s}"
              f"{rr_t:>10.2f}{rr_p:>12.2f}{('yes' if sig_p else 'no'):>8s}", flush=True)

    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("-" * W)
    print("ATE in units of 10⁻³ (‰). Tile column reproduces Table I; patient column gives every "
          "patient equal weight.")
    print(f"saved -> {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
