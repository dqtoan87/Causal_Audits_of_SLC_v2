#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalXAI v14 — PHASE C5: EXTERNAL REPLICATION (PAD-UFES-20, modality lâm sàng độc lập).

Replicate Tầng-1 (disease ATE) trên DATASET KHÁC MODALITY (ảnh clinical, không phải TBP-3D tiles)
để chứng minh cơ chế nhân quả của C1 KHÔNG phải artifact của ISIC/TBP. Đây là external validity ở
cấp SCM LÂM SÀNG — KHÔNG qua transfer ảnh (transfer ảnh đã thất bại AUC<0.5 ở v13). Xem DESIGN §7 C5.

★ Lợi thế PAD-UFES-20:
  - FITZPATRICK THẬT (1-6) -> adjust + CATE skin-tone đáng tin (vá lỗi "ITA-TBP tuyệt đối" của v13).
  - Risk-features CÓ CẤU TRÚC (grew/bleed/elevation/changed/itch/hurt) = concept nhân quả lâm sàng.
  - diameter -> map TRỰC TIẾP vào D_size của C1 => replicate cơ chế SIZE cross-modality.

Estimand & method = HỆT C1 (tái dùng AIPW doubly-robust + E-value + cluster-bootstrap của
c1_tier1_ate.py). Concept liên tục (size) dichotomize Q3-vs-Q1; concept lâm sàng nhị phân TRUE-vs-FALSE
(bỏ UNK). Backdoor adjust {age, gender, region, fitzpatrick, skin/cancer history}.

Target ác tính = diagnostic ∈ {BCC, SCC, MEL} (chuẩn PAD-UFES); lành = {ACK, NEV, SEK}.

Out -> results/tier1_padufes.json. Đọc-only metadata, nhanh, CPU.

Chạy:
  .venv/bin/python Causal_XAI_v3/code/c5_padufes_ate.py
  .venv/bin/python Causal_XAI_v3/code/c5_padufes_ate.py --refute
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c1_tier1_ate as c1   # tái dùng aipw_crossfit/cluster_stats/cluster_ratio_ci/e_value(_ci)

DATA = "/mnt/data2/Toan/Causal_ML_XAI/data/PAD-UFES-20/metadata.csv"
OUT_DIR = os.path.join(os.path.dirname(HERE), "results")
OUT_JSON = os.path.join(OUT_DIR, "tier1_padufes.json")

GROUP = "patient_id"
MALIGNANT = {"BCC", "SCC", "MEL"}     # ác tính (chuẩn PAD-UFES)
SEED = 42

# Concept: kind='cont' (dichotomize Q3/Q1) | 'bin' (TRUE/FALSE). dir_hint chỉ để diễn giải.
CONCEPTS = [
    {"name": "size_diam", "kind": "cont", "dir_hint": "pos (↔ D_size C1)"},
    {"name": "elevation", "kind": "bin", "dir_hint": "pos (ABCDE Elevation)"},
    {"name": "grew",      "kind": "bin", "dir_hint": "pos (Evolving)"},
    {"name": "bleed",     "kind": "bin", "dir_hint": "pos"},
    {"name": "changed",   "kind": "bin", "dir_hint": "pos"},
    {"name": "itch",      "kind": "bin", "dir_hint": "weak"},
    {"name": "hurt",      "kind": "bin", "dir_hint": "weak"},
    {"name": "NEGCTRL_smoke", "kind": "bin", "dir_hint": "null (negative control lối sống)"},
]

NUM_CONF = ["age"]
CAT_CONF = ["gender", "region", "skin_cancer_history", "cancer_history", "fitz_stratum"]


def fitz_stratum(s):
    """Fitzpatrick THẬT -> 3 nhóm: light(1-2)/med(3-4)/dark(5-6); NA -> missing."""
    f = pd.to_numeric(s, errors="coerce")
    out = pd.Series("fitz_missing", index=s.index, dtype="object")
    out[f.isin([1, 2])] = "fitz_I_II_light"
    out[f.isin([3, 4])] = "fitz_III_IV_med"
    out[f.isin([5, 6])] = "fitz_V_VI_dark"
    return out


def build_Z(df):
    parts = []
    for cc in NUM_CONF:
        x = pd.to_numeric(df[cc], errors="coerce")
        parts.append(x.fillna(x.median()).rename(cc))
        parts.append(x.isna().astype(int).rename(f"{cc}__isna"))
    cat = df[CAT_CONF].astype("object").fillna("missing")
    dummies = pd.get_dummies(cat, prefix=CAT_CONF, dummy_na=False).astype(float)
    return pd.concat(parts + [dummies], axis=1)


def get_treatment(df, concept):
    """Trả (T[int array], mask[bool]) cho concept. cont->Q3/Q1 (drop giữa); bin->TRUE/FALSE (drop UNK)."""
    if concept["kind"] == "cont":   # size = diameter_1 * diameter_2
        x = pd.to_numeric(df["diameter_1"], errors="coerce") * pd.to_numeric(df["diameter_2"], errors="coerce")
        q1, q3 = x.quantile(1 / 3), x.quantile(2 / 3)
        mask = x.notna() & ((x <= q1) | (x >= q3))
        T = (x >= q3).astype(int)
        return T.to_numpy(), mask.to_numpy(), (float(q1), float(q3))
    col = "smoke" if concept["name"] == "NEGCTRL_smoke" else concept["name"]
    s = df[col].astype(str).str.upper()
    mask = s.isin(["TRUE", "FALSE"])
    T = (s == "TRUE").astype(int)
    return T.to_numpy(), mask.to_numpy(), None


def cate_by_fitz(psi, gv, fitz, n_boot, seed):
    out = {}
    for k in pd.unique(fitz):
        m = fitz == k
        if m.sum() < 80:
            continue
        p, lo, hi, _ = c1.cluster_stats(psi[m], gv[m], max(300, n_boot // 3), seed)
        out[str(k)] = {"n": int(m.sum()), "ate": p, "ci": [lo, hi]}
    return out


def run_concept(df, Zfull, concept, args):
    name = concept["name"]
    T, mask, thr = get_treatment(df, concept)
    Z = Zfull.to_numpy()[mask]
    Tv = T[mask]
    Yv = df["target"].to_numpy()[mask].astype(int)
    gv = df[GROUP].to_numpy()[mask]
    fitz = df["fitz_stratum"].to_numpy()[mask]

    if Tv.sum() < 20 or (1 - Tv).sum() < 20:
        print(f"  [{name}] BỎ (treatment quá lệch: T1={int(Tv.sum())} T0={int((1-Tv).sum())})")
        return None
    print(f"  [{name}] n={mask.sum():,} (T1={int(Tv.sum())}/T0={int((1-Tv).sum())}) pos={int(Yv.sum())}"
          + (f" | Q1={thr[0]:.1f} Q3={thr[1]:.1f}" if thr else ""), flush=True)

    psi, ey1, ey0, pos_frac = c1.aipw_crossfit(Z, Tv, Yv, gv, SEED)
    ate, lo, hi, se = c1.cluster_stats(psi, gv, args.n_boot, SEED)
    p1, p0, rr, rr_lo, rr_hi = c1.cluster_ratio_ci(ey1, ey0, gv, args.n_boot, SEED)
    ev, ev_ci = c1.e_value(rr), c1.e_value_ci(rr_lo, rr_hi)
    res = dict(concept=name, kind=concept["kind"], dir_hint=concept["dir_hint"],
               n=int(mask.sum()), n_pos=int(Yv.sum()), n_treat=int(Tv.sum()), n_ctrl=int((1 - Tv).sum()),
               thresholds=thr, ate=ate, ate_ci95=[lo, hi], ate_if_se=se,
               significant=bool(not (lo <= 0 <= hi)),
               rr=rr, rr_ci95=[rr_lo, rr_hi], p1_treated=p1, p0_control=p0,
               e_value=ev, e_value_ci=ev_ci, positivity_extreme_frac=pos_frac,
               cate_by_fitz=cate_by_fitz(psi, gv, fitz, args.n_boot, SEED))

    if args.refute:
        rng = np.random.default_rng(SEED + 7)
        Tp = rng.permutation(Tv)
        psi_p, _, _, _ = c1.aipw_crossfit(Z, Tp, Yv, gv, SEED)
        atep, lop, hip, _ = c1.cluster_stats(psi_p, gv, args.n_boot, SEED)
        res["refute"] = dict(placebo_ate=atep, placebo_ci95=[lop, hip],
                             placebo_pass=bool(lop <= 0 <= hip))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--refute", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    t0 = time.time()
    df = pd.read_csv(args.data)
    df["target"] = df["diagnostic"].isin(MALIGNANT).astype(int)
    df["fitz_stratum"] = fitz_stratum(df["fitspatrick"])
    print(f"PAD-UFES-20: {df.shape}  patients={df[GROUP].nunique()}  "
          f"malignant_rate={df['target'].mean():.3f} (n_pos={int(df['target'].sum())})")
    print("  diagnostic:", df["diagnostic"].value_counts().to_dict())
    print("  fitz strata:", df["fitz_stratum"].value_counts().to_dict())

    Zfull = build_Z(df)
    print(f"  Z dims={Zfull.shape[1]}")
    results = [r for r in (run_concept(df, Zfull, c, args) for c in CONCEPTS) if r]

    out = dict(phase="C5_external_padufes", data=args.data,
               n_rows=int(len(df)), n_patients=int(df[GROUP].nunique()),
               n_pos=int(df["target"].sum()), malignant_rate=float(df["target"].mean()),
               target_def="malignant = diagnostic in {BCC,SCC,MEL}",
               method="AIPW DR cross-fit GroupKFold(patient), E-value, cluster bootstrap; "
                      "backdoor age/gender/region/fitzpatrick(REAL)/histories; reuse C1 estimator.",
               config=dict(n_boot=args.n_boot, refute=args.refute),
               concepts=results, runtime_sec=round(time.time() - t0, 1))
    json.dump(out, open(args.out, "w"), indent=2, default=float)
    print(f"\nSaved -> {args.out} ({out['runtime_sec']}s)")

    print("\n" + "=" * 88)
    print(f"{'concept':16s}{'ATE':>9s}{'CI95':>20s}{'RR':>6s}{'E-val':>6s}{'E-CI':>6s}{'sig':>5s}")
    print("-" * 88)
    for r in results:
        ci = f"[{r['ate_ci95'][0]:+.3f},{r['ate_ci95'][1]:+.3f}]"
        print(f"{r['concept']:16s}{r['ate']:+.3f}{ci:>20s}{r['rr']:6.2f}{r['e_value']:6.2f}"
              f"{r['e_value_ci']:6.2f}{'YES' if r['significant'] else 'no':>5s}")
    print("=" * 88)
    print("★ size_diam dương & sig => replicate D_size của C1 cross-modality (modality lâm sàng, "
          "adjust Fitzpatrick THẬT). NEGCTRL_smoke kỳ vọng n.s.")
    if args.refute:
        print("Refuters: " + " | ".join(
            f"{r['concept']}={'PASS' if r['refute']['placebo_pass'] else 'FAIL'}" for r in results))


if __name__ == "__main__":
    main()
