#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalXAI v14 — PHASE C1: Tier-1 Disease Causal Model (ATE concept -> bệnh).

Ước lượng tác động NHÂN QUẢ của từng concept lâm sàng (ABCD + contrast) lên xác suất ác tính
(target), ĐIỀU CHỈNH backdoor theo confounder ĐO ĐƯỢC (age, sex, anatomical site, site thu nhận =
attribution, skin-tone ITA). Đây là "clinical causal ground-truth" — trục cho mọi phase sau (CaCE,
alignment). Xem `Causal_XAI_v3/CausalXAI_v14_DESIGN.md` §4.

Phương pháp (không phụ thuộc econml/dowhy — tự cài để minh bạch & tránh thiếu lib):
  - Estimand: ATE của concept dichotomize Q3-vs-Q1 (drop tertile giữa) lên target. (DESIGN §2.1)
  - Identification: backdoor adjust Z (đo được). Giả định no-unmeasured-confounder CHO TỪNG CẶP,
    kiểm bằng E-value (VanderWeele 2017). Positivity kiểm bằng overlap propensity + trim.
  - Estimator: AIPW (doubly-robust) với CROSS-FITTING GroupKFold theo patient_id (chống leakage
    bệnh nhân — 401k lesion lồng trong 1042 bệnh nhân). Nuisance = HistGradientBoosting.
  - Thống kê: CLUSTER bootstrap theo bệnh nhân (CI chính) + IF cluster-robust SE (phụ).
  - Sensitivity: E-value điểm & cận-CI. Refuters: placebo (hoán vị treatment) + subset.
  - Heterogeneity: CATE theo strata ITA (Q1 sẫm -> Q4 sáng) và theo site.
  - Negative control: tbp_lv_nevi_confidence (proxy lành) -> kỳ vọng ATE âm.

KỶ LUẬT (kế thừa v13 THESIS A2/A4): mọi số có CI; báo % mẫu trim (no silent cap); không dùng
"effect" khi CI chứa 0. Direction mỗi concept ghi rõ ở metadata kết quả.

Chạy:
  .venv/bin/python Causal_XAI_v3/code/c1_tier1_ate.py            # full
  .venv/bin/python Causal_XAI_v3/code/c1_tier1_ate.py --refute   # + refutation (chậm hơn)
  .venv/bin/python Causal_XAI_v3/code/c1_tier1_ate.py --max-rows 40000   # smoke test nhanh
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Cấu hình
# --------------------------------------------------------------------------------------
DATA_CSV = "/mnt/data2/Toan/Causal_ML/data/data_ISIC2024/train-metadata.csv"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_JSON = os.path.join(OUT_DIR, "tier1_ate.json")

OUTCOME = "target"
GROUP = "patient_id"

# Concept lâm sàng (ABCD + contrast). 'dir_hint' = kỳ vọng dấu lâm sàng (chỉ để diễn giải, KHÔNG ép).
#   asymmetry: tbp_lv_symm_2axis là điểm ĐỐI XỨNG (cao = đối xứng hơn) -> kỳ vọng ÂM với ác tính.
#   border/color/size/contrast cao -> nghi ngờ ác -> kỳ vọng DƯƠNG.
CONCEPTS = [
    {"name": "A_asymmetry", "col": "tbp_lv_symm_2axis", "dir_hint": "neg (đối xứng cao -> ít ác)"},
    {"name": "B_border",    "col": "tbp_lv_norm_border", "dir_hint": "pos"},
    {"name": "C_color",     "col": "tbp_lv_norm_color",  "dir_hint": "pos"},
    {"name": "C_colorstd",  "col": "tbp_lv_color_std_mean", "dir_hint": "pos"},
    {"name": "D_size",      "col": "tbp_lv_areaMM2",     "dir_hint": "pos"},
    {"name": "contrast_LB", "col": "tbp_lv_deltaLBnorm", "dir_hint": "pos"},
    # Negative control — KHÔNG phải concept, để kiểm protocol:
    {"name": "NEGCTRL_nevi", "col": "tbp_lv_nevi_confidence", "dir_hint": "neg (NEGATIVE CONTROL)"},
]

# Confounder ĐO ĐƯỢC (backdoor). LƯU Ý (v13): age/sex/location là risk-factor HỢP LỆ -> đưa vào
# như confounder, KHÔNG debias. Không đưa concept khác vào Z (DAG: concept là sibling, không phải
# confounder của nhau) — xem DESIGN §3; có sensitivity tùy chọn.
NUM_CONF = ["age_approx"]
CAT_CONF = ["sex", "anatom_site_general", "attribution"]  # ITA-stratum thêm vào sau khi tính

N_FOLDS = 5
N_BOOT = 1000
TRIM = 0.02           # trim propensity về [TRIM, 1-TRIM]
POS_FLAG = 0.05       # ngưỡng cảnh báo positivity
RANDOM_SEED = 42
GBM_KW = dict(max_depth=4, max_iter=200, learning_rate=0.07, l2_regularization=1.0,
              early_stopping=False, random_state=RANDOM_SEED)


# --------------------------------------------------------------------------------------
# Tiện ích
# --------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_ita(df: pd.DataFrame) -> pd.Series:
    """ITA = arctan((L*-50)/b*)·180/π trên da NỀN (Lext, Bext). Chỉ dùng STRATA TƯƠNG ĐỐI (Q1..Q4),
    KHÔNG dùng thang Fitzpatrick tuyệt đối (cảnh báo v13: TBP-ITA tuyệt đối không tin được)."""
    L, b = df["tbp_lv_Lext"].astype(float), df["tbp_lv_Bext"].astype(float)
    b = b.replace(0, np.nan)
    ita = np.degrees(np.arctan2(L - 50.0, b))
    q = pd.qcut(ita, 4, labels=["ITA_Q1_dark", "ITA_Q2", "ITA_Q3", "ITA_Q4_light"], duplicates="drop")
    return q.astype("object").fillna("ITA_missing")


def build_Z(df: pd.DataFrame) -> pd.DataFrame:
    """Ma trận confounder (one-hot cho categorical, raw cho numeric — GBM không cần scale).
    NA: age -> median + cờ thiếu; categorical -> hạng 'missing'."""
    parts = []
    for c in NUM_CONF:
        x = pd.to_numeric(df[c], errors="coerce")
        parts.append(x.fillna(x.median()).rename(c))
        parts.append(x.isna().astype(int).rename(f"{c}__isna"))
    cat_cols = CAT_CONF + ["ITA_stratum"]
    cat = df[cat_cols].astype("object").fillna("missing")
    dummies = pd.get_dummies(cat, prefix=cat_cols, dummy_na=False).astype(float)
    Z = pd.concat(parts + [dummies], axis=1)
    return Z


def aipw_crossfit(Z: np.ndarray, T: np.ndarray, Y: np.ndarray, groups: np.ndarray, seed: int):
    """AIPW doubly-robust với cross-fitting GroupKFold(patient). Trả về psi, ey1, ey0 (per-row, đã
    cross-fit) + chẩn đoán positivity. ATE = mean(psi); RR = mean(ey1)/mean(ey0)."""
    n = len(Y)
    psi = np.full(n, np.nan)
    ey1 = np.full(n, np.nan)
    ey0 = np.full(n, np.nan)
    e_all = np.full(n, np.nan)
    gkf = GroupKFold(n_splits=N_FOLDS)
    for tr, te in gkf.split(Z, T, groups):
        # propensity e(z) = P(T=1|Z)
        ps = HistGradientBoostingClassifier(**{**GBM_KW, "random_state": seed}).fit(Z[tr], T[tr])
        e = ps.predict_proba(Z[te])[:, 1]
        e = np.clip(e, TRIM, 1 - TRIM)
        # outcome mu1, mu0 (T-learner: 2 model riêng từng arm)
        tr1, tr0 = tr[T[tr] == 1], tr[T[tr] == 0]
        m1 = HistGradientBoostingClassifier(**{**GBM_KW, "random_state": seed}).fit(Z[tr1], Y[tr1])
        m0 = HistGradientBoostingClassifier(**{**GBM_KW, "random_state": seed}).fit(Z[tr0], Y[tr0])
        mu1 = m1.predict_proba(Z[te])[:, 1]
        mu0 = m0.predict_proba(Z[te])[:, 1]
        Tt, Yt = T[te], Y[te]
        ey1[te] = mu1 + Tt * (Yt - mu1) / e
        ey0[te] = mu0 + (1 - Tt) * (Yt - mu0) / (1 - e)
        psi[te] = ey1[te] - ey0[te]
        e_all[te] = e
    pos_frac = float(np.mean((e_all < POS_FLAG) | (e_all > 1 - POS_FLAG)))
    return psi, ey1, ey0, pos_frac


def cluster_stats(values: np.ndarray, groups: np.ndarray, n_boot: int, seed: int):
    """Mean + cluster bootstrap CI (resample bệnh nhân). Gộp theo nhóm để bootstrap rẻ & đúng."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"v": values, "g": groups}).dropna()
    agg = df.groupby("g")["v"].agg(["sum", "count"])
    s, c = agg["sum"].to_numpy(), agg["count"].to_numpy()
    point = float(s.sum() / c.sum())
    G = len(s)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        boots[b] = s[idx].sum() / c[idx].sum()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # cluster-robust IF SE (phụ)
    gmean = df.groupby("g")["v"].apply(lambda x: (x - point).sum())
    se = float(np.sqrt((gmean.to_numpy() ** 2).sum()) / c.sum())
    return point, float(lo), float(hi), se


def cluster_ratio_ci(ey1, ey0, groups, n_boot, seed):
    """RR = E[Y1]/E[Y0] + cluster bootstrap CI (cho E-value)."""
    rng = np.random.default_rng(seed + 1)
    df = pd.DataFrame({"a": ey1, "b": ey0, "g": groups}).dropna()
    agg = df.groupby("g").agg(a_s=("a", "sum"), b_s=("b", "sum"), n=("a", "count"))
    a_s, b_s, n = agg["a_s"].to_numpy(), agg["b_s"].to_numpy(), agg["n"].to_numpy()
    ntot = n.sum()
    p1, p0 = a_s.sum() / ntot, b_s.sum() / ntot
    rr = p1 / p0 if p0 > 0 else np.nan
    G = len(n)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, G, G)
        nt = n[idx].sum()
        boots[b] = (a_s[idx].sum() / nt) / (b_s[idx].sum() / nt)
    lo, hi = np.percentile(boots[np.isfinite(boots)], [2.5, 97.5])
    return float(p1), float(p0), float(rr), float(lo), float(hi)


def e_value(rr: float) -> float:
    """E-value VanderWeele 2017. RR<1 -> nghịch đảo trước."""
    if not np.isfinite(rr) or rr <= 0:
        return float("nan")
    r = rr if rr >= 1 else 1.0 / rr
    return float(r + math.sqrt(r * (r - 1)))


def e_value_ci(rr_lo: float, rr_hi: float) -> float:
    """E-value cho cận-CI gần null nhất (1.0). Nếu CI bắc cầu 1 -> 1.0 (không loại được null)."""
    if rr_lo <= 1 <= rr_hi:
        return 1.0
    bound = rr_lo if rr_lo > 1 else rr_hi  # cận gần 1 nhất
    return e_value(bound)


# --------------------------------------------------------------------------------------
# Một concept
# --------------------------------------------------------------------------------------
def run_concept(df, Zfull, concept, args, seed=RANDOM_SEED):
    name, col = concept["name"], concept["col"]
    x = pd.to_numeric(df[col], errors="coerce")
    if getattr(args, "split", "tertile") == "median":   # sensitivity #10: median split (dùng cả mẫu)
        med = x.median()
        q1 = q3 = med
        mask = x.notna()
        T = (x >= med).astype(int)
    else:                                                # mặc định: tertile Q3-vs-Q1, drop giữa
        q1, q3 = x.quantile(1 / 3), x.quantile(2 / 3)
        mask = x.notna() & ((x <= q1) | (x >= q3))
        T = (x >= q3).astype(int)
    sub = mask.to_numpy()

    Z = Zfull.to_numpy()[sub]
    Tv = T.to_numpy()[sub]
    Yv = df[OUTCOME].to_numpy()[sub].astype(int)
    gv = df[GROUP].to_numpy()[sub]
    ita = df["ITA_stratum"].to_numpy()[sub]
    site = df["attribution"].to_numpy()[sub]

    log(f"  [{name}] n={sub.sum():,} (T1={int(Tv.sum()):,}/T0={int((1-Tv).sum()):,}) "
        f"pos={int(Yv.sum())} | thresh Q1={q1:.3g} Q3={q3:.3g}")

    psi, ey1, ey0, pos_frac = aipw_crossfit(Z, Tv, Yv, gv, seed)
    ate, lo, hi, se = cluster_stats(psi, gv, args.n_boot, seed)
    p1, p0, rr, rr_lo, rr_hi = cluster_ratio_ci(ey1, ey0, gv, args.n_boot, seed)
    ev, ev_ci = e_value(rr), e_value_ci(rr_lo, rr_hi)
    sig = not (lo <= 0 <= hi)

    # CATE theo strata ITA & site (heterogeneity) — dùng psi đã cross-fit
    def cate_by(key_arr):
        out = {}
        for k in pd.unique(key_arr):
            m = key_arr == k
            if m.sum() < 200:   # tránh strata quá nhỏ
                continue
            p, l, h, _ = cluster_stats(psi[m], gv[m], max(300, args.n_boot // 3), seed)
            out[str(k)] = {"n": int(m.sum()), "ate": p, "ci": [l, h]}
        return out

    res = {
        "concept": name, "column": col, "dir_hint": concept["dir_hint"],
        "n": int(sub.sum()), "n_pos": int(Yv.sum()),
        "n_treat": int(Tv.sum()), "n_ctrl": int((1 - Tv).sum()),
        "thresholds": {"q1_3": float(q1), "q2_3": float(q3)},
        "ate": ate, "ate_ci95": [lo, hi], "ate_if_se": se, "significant": bool(sig),
        "rr": rr, "rr_ci95": [rr_lo, rr_hi], "p1_treated": p1, "p0_control": p0,
        "e_value": ev, "e_value_ci": ev_ci,
        "positivity_extreme_frac": pos_frac,
        "cate_by_ita": cate_by(ita),
        "cate_by_site": cate_by(site),
    }

    # Refuters (tùy chọn, chậm) -----------------------------------------------------
    if args.refute:
        rng = np.random.default_rng(seed + 7)
        # placebo: hoán vị treatment -> ATE kỳ vọng ~0
        Tp = rng.permutation(Tv)
        psi_p, _, _, _ = aipw_crossfit(Z, Tp, Yv, gv, seed)
        atep, lop, hip, _ = cluster_stats(psi_p, gv, args.n_boot, seed)
        # subset 60% bệnh nhân -> ATE kỳ vọng ổn định
        gids = pd.unique(gv)
        keep = set(rng.choice(gids, int(len(gids) * 0.6), replace=False))
        ms = np.array([g in keep for g in gv])
        psi_s, _, _, _ = aipw_crossfit(Z[ms], Tv[ms], Yv[ms], gv[ms], seed)
        ates, los, his, _ = cluster_stats(psi_s, gv[ms], args.n_boot, seed)
        res["refute"] = {
            "placebo_ate": atep, "placebo_ci95": [lop, hip],
            "placebo_pass": bool(lop <= 0 <= hip),     # placebo PASS = CI chứa 0
            "subset60_ate": ates, "subset60_ci95": [los, his],
            "subset_stable": bool(min(his, hi) >= max(los, lo)),  # CI chồng lấp với full
        }
    return res


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_CSV)
    ap.add_argument("--out", default=None,
                    help="mặc định: results/tier1_ate.json cho tertile, "
                         "results/tier1_ate_median.json cho --split median")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--max-rows", type=int, default=0, help="0 = dùng full; >0 = smoke test")
    ap.add_argument("--refute", action="store_true", help="chạy placebo + subset (chậm)")
    ap.add_argument("--split", choices=["tertile", "median"], default="tertile",
                    help="cutpoint cho concept liên tục (sensitivity #10)")
    args = ap.parse_args()
    if args.out is None:
        args.out = OUT_JSON if args.split == "tertile" else os.path.join(
            OUT_DIR, "tier1_ate_median.json")

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    log(f"Load {args.data}")
    df = pd.read_csv(args.data, low_memory=False)
    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=RANDOM_SEED).reset_index(drop=True)
        log(f"  smoke test: lấy {len(df):,} dòng")
    log(f"  shape={df.shape} patients={df[GROUP].nunique()} pos_rate={df[OUTCOME].mean():.5f} "
        f"n_pos={int(df[OUTCOME].sum())}")

    df["ITA_stratum"] = compute_ita(df)
    Zfull = build_Z(df)
    log(f"  Z dims={Zfull.shape[1]} (confounders one-hot)")

    results = []
    for c in CONCEPTS:
        results.append(run_concept(df, Zfull, c, args))

    out = {
        "phase": "C1_tier1_disease_ate",
        "data": args.data,
        "n_rows": int(len(df)), "n_patients": int(df[GROUP].nunique()),
        "n_pos": int(df[OUTCOME].sum()), "pos_rate": float(df[OUTCOME].mean()),
        "method": "AIPW doubly-robust, cross-fit GroupKFold(patient), cluster bootstrap CI, "
                  "E-value sensitivity; concept dichotomized Q3-vs-Q1; backdoor adjust "
                  "age/sex/anatom_site/attribution/ITA-stratum.",
        "config": {"n_folds": N_FOLDS, "n_boot": args.n_boot, "trim": TRIM,
                   "gbm": GBM_KW, "refute": args.refute},
        "concepts": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"Saved -> {args.out}  ({out['runtime_sec']}s)")

    # Bảng tóm tắt ----------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"{'concept':14s} {'ATE':>9s} {'CI95':>20s} {'RR':>6s} {'E-val':>6s} "
          f"{'E-CI':>6s} {'pos%':>5s} {'sig':>4s}")
    print("-" * 92)
    for r in results:
        ci = f"[{r['ate_ci95'][0]:+.2e},{r['ate_ci95'][1]:+.2e}]"
        print(f"{r['concept']:14s} {r['ate']:+.2e} {ci:>20s} {r['rr']:6.2f} "
              f"{r['e_value']:6.2f} {r['e_value_ci']:6.2f} "
              f"{100*r['positivity_extreme_frac']:5.1f} {'YES' if r['significant'] else 'no':>4s}")
    print("=" * 92)
    print("Đọc: ATE = thay đổi P(ác) khi concept ở tertile cao vs thấp (đã adjust confounder). "
          "sig=CI không chứa 0. E-val=confounder ẩn cần mạnh cỡ nào mới lật (cao=robust). "
          "NEGCTRL kỳ vọng dấu âm & |ATE| nhỏ.")
    if args.refute:
        print("\nRefuters: placebo_pass (CI chứa 0 = tốt) | subset_stable (CI chồng full = tốt)")
        for r in results:
            rf = r["refute"]
            print(f"  {r['concept']:14s} placebo={'PASS' if rf['placebo_pass'] else 'FAIL'} "
                  f"({rf['placebo_ate']:+.1e}) | subset={'OK' if rf['subset_stable'] else 'DRIFT'} "
                  f"({rf['subset60_ate']:+.1e})")


if __name__ == "__main__":
    sys.exit(main())
