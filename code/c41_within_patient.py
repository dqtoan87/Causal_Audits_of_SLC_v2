#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C41 — Tier 1 với thiết kế TRONG-BỆNH-NHÂN (conditional logistic, phân tầng theo patient_id).

Vì sao. Backdoor adjustment của C1 chỉ khống chế được nhiễu ĐO ĐƯỢC; E-value chỉ định lượng
mức phơi nhiễm với nhiễu KHÔNG đo được chứ không loại bỏ nó. Trong ISIC-2024, mỗi bệnh nhân
đóng góp hàng trăm tile, và 258 bệnh nhân có cả tile lành lẫn ác — mang 392/393 ca ác tính.
Phân tầng theo bệnh nhân do đó khử TOÀN BỘ nhiễu ở cấp bệnh nhân, đo được hay không, chỉ mất
đúng 1 ca ác tính. Kiểm tra trên metadata: acquisition site và giới tính hằng số trong mỗi
bệnh nhân (1042/1042), tuổi hằng số ở 942/1042 — cả ba bị hấp thụ bởi thiết kế. Vị trí giải
phẫu và ITA thì THAY ĐỔI trong cùng bệnh nhân nên vẫn phải hiệu chỉnh, và script làm điều đó.

Ước lượng. Conditional logistic likelihood chính xác (đa thức đối xứng sơ cấp), tối ưu bằng
BFGS với gradient giải tích. Sai số chuẩn là sandwich cụm với BỆNH NHÂN là cụm — ở đây mỗi
tầng đúng bằng một cụm, nên phương sai cụm là hệ quả trực tiếp của thiết kế chứ không phải
một hiệu chỉnh thêm. Kèm bootstrap cụm (resample bệnh nhân) làm kiểm chứng.

Đọc estimand cho đúng. Đây là ODDS RATIO CÓ ĐIỀU KIỆN TRONG BỆNH NHÂN, KHÁC estimand của
Bảng I (risk difference chuẩn hoá ở cấp tile trên toàn cohort). Với outcome hiếm thì OR xấp
xỉ RR, nên so sánh được về HƯỚNG và ĐỘ LỚN TƯƠNG ĐỐI, không phải một ước lượng lại của cùng
một con số. Giá phải trả: tổ hợp trong bệnh nhân rất mất cân bằng (trung vị 435 tile, 1–2 ca
ác), nên khoảng tin cậy rộng hơn Bảng I.

Out -> results/c41_within_patient.json
Chạy:  python Causal_XAI_v3/code/c41_within_patient.py [--boot 500] [--data CSV]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c1_tier1_ate as c1   # noqa: E402  (CONCEPTS, compute_ita, đường dẫn dữ liệu)

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c41_within_patient.json")

GROUP, OUTCOME = c1.GROUP, c1.OUTCOME
# Confounder BIẾN THIÊN trong bệnh nhân. Những cái hằng số (age, sex, attribution) bị tầng
# hấp thụ và sẽ tự triệt tiêu khỏi conditional likelihood, nên không đưa vào.
WITHIN_CONF = ["anatom_site_general", "ITA_stratum"]
SEED = 42


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------
# Conditional logistic likelihood
# ----------------------------------------------------------------------------------
def _esp_and_incl(w, k):
    """Trả về (e_k, p) với e_k = đa thức đối xứng sơ cấp bậc k của w, và
    p_i = w_i·e_{k-1}(w bỏ i)/e_k = xác suất unit i nằm trong k ca được chọn.

    Dùng DP prefix/suffix nên chi phí O(n·k²), tránh phép chia không ổn định."""
    n = len(w)
    # prefix[j, i] = e_j(w[:i]);  suffix[j, i] = e_j(w[i:])
    pre = np.zeros((k + 1, n + 1)); pre[0, :] = 1.0
    for i in range(1, n + 1):
        wi = w[i - 1]
        for j in range(min(k, i), 0, -1):
            pre[j, i] = pre[j, i - 1] + wi * pre[j - 1, i - 1]
    suf = np.zeros((k + 1, n + 2)); suf[0, :] = 1.0
    for i in range(n, 0, -1):
        wi = w[i - 1]
        for j in range(min(k, n - i + 1), 0, -1):
            suf[j, i] = suf[j, i + 1] + wi * suf[j - 1, i + 1]
    ek = pre[k, n]
    if not np.isfinite(ek) or ek <= 0:
        return ek, np.zeros(n)
    # e_{k-1}(w bỏ i) = Σ_j pre[j, i-1]·suf[k-1-j, i+1]
    incl = np.empty(n)
    for i in range(1, n + 1):
        s = 0.0
        for j in range(0, k):
            s += pre[j, i - 1] * suf[k - 1 - j, i + 1]
        incl[i - 1] = w[i - 1] * s / ek
    return ek, incl


def clogit_nll(beta, X, y, strata):
    """-log conditional likelihood và gradient, cộng score theo từng tầng (cho sandwich)."""
    eta = X @ beta
    nll = 0.0
    grad = np.zeros_like(beta)
    scores = []
    for idx in strata:
        e = eta[idx]
        e = e - e.max()                      # bất biến: tỉ số thuần nhất bậc k theo w
        w = np.exp(e)
        yi = y[idx]
        k = int(yi.sum())
        ek, incl = _esp_and_incl(w, k)
        if ek <= 0 or not np.isfinite(ek):
            continue
        nll -= float(e[yi == 1].sum() - np.log(ek))
        g = (X[idx] * (yi - incl)[:, None]).sum(0)   # score của tầng
        grad -= g
        scores.append(g)
    return nll, grad, np.array(scores)


def fit_clogit(X, y, strata, beta0=None, maxiter=200):
    p = X.shape[1]
    b0 = np.zeros(p) if beta0 is None else beta0
    f = lambda b: clogit_nll(b, X, y, strata)[:2]
    r = minimize(f, b0, jac=True, method="BFGS",
                 options={"maxiter": maxiter, "gtol": 1e-6})
    return r.x, r


def sandwich_se(beta, X, y, strata):
    """Sai số chuẩn cụm. Mỗi tầng = một bệnh nhân = một cụm độc lập, nên
    V = H⁻¹ (Σ_s g_s g_sᵀ) H⁻¹ là phương sai cụm theo đúng thiết kế."""
    _, _, S = clogit_nll(beta, X, y, strata)
    eps = 1e-5
    p = len(beta)
    H = np.zeros((p, p))
    for j in range(p):
        bp = beta.copy(); bp[j] += eps
        bm = beta.copy(); bm[j] -= eps
        H[:, j] = (clogit_nll(bp, X, y, strata)[1] - clogit_nll(bm, X, y, strata)[1]) / (2 * eps)
    H = 0.5 * (H + H.T)
    try:
        Hinv = np.linalg.pinv(H)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    meat = S.T @ S
    V = Hinv @ meat @ Hinv
    d = np.diag(V)
    return np.sqrt(np.where(d > 0, d, np.nan))


# ----------------------------------------------------------------------------------
# Chuẩn bị một concept
# ----------------------------------------------------------------------------------
def prepare(df, col, adjust=True):
    """Cắt tertile trên TOÀN cohort (giống C1), giữ Q3 vs Q1, rồi chỉ giữ những tầng còn
    cả hai lớp — tầng đồng nhất không đóng góp gì cho conditional likelihood."""
    x = pd.to_numeric(df[col], errors="coerce")
    ok = x.notna()
    q1, q3 = x.quantile(1 / 3), x.quantile(2 / 3)
    T = pd.Series(np.nan, index=df.index)
    T[ok & (x <= q1)] = 0.0
    T[ok & (x >= q3)] = 1.0
    sub = T.notna()
    d = df.loc[sub].copy(); d["_T"] = T[sub].to_numpy()

    g = d.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
    keep = g[(g["sum"] > 0) & (g["sum"] < g["count"])].index
    d = d[d[GROUP].isin(keep)]
    if len(d) == 0:
        return None

    cols = [d["_T"].rename("T")]
    if adjust:
        cat = d[WITHIN_CONF].astype("object").fillna("missing")
        du = pd.get_dummies(cat, prefix=WITHIN_CONF, drop_first=True).astype(float)
        cols.append(du)
    X = pd.concat(cols, axis=1)

    y = d[OUTCOME].to_numpy().astype(float)
    codes, _ = pd.factorize(d[GROUP])
    order = np.argsort(codes, kind="stable")
    X = X.to_numpy()[order]; y = y[order]; codes = codes[order]
    bounds = np.searchsorted(codes, np.arange(codes.max() + 2))
    strata = [np.arange(bounds[i], bounds[i + 1]) for i in range(codes.max() + 1)]
    strata = [s for s in strata if len(s) > 1]

    # bỏ cột không có biến thiên TRONG tầng nào (không định danh được)
    if X.shape[1] > 1:
        var = np.zeros(X.shape[1])
        for s in strata:
            var += X[s].var(0) * len(s)
        drop = np.where(var < 1e-10)[0]
        drop = drop[drop != 0]
        if len(drop):
            X = np.delete(X, drop, axis=1)
    return X, y, strata, d


def run_concept(df, cname, col, boot, rng):
    prep = prepare(df, col, adjust=True)
    if prep is None:
        return {"concept": cname, "column": col, "error": "no contributing stratum"}
    X, y, strata, d = prep
    n_case = int(y.sum())
    log(f"  {cname:15s} rows={len(y):>7,} strata={len(strata):>4} cases={n_case:>4} cols={X.shape[1]}")

    beta, r = fit_clogit(X, y, strata)
    se = sandwich_se(beta, X, y, strata)
    b, sb = float(beta[0]), float(se[0])
    z = b / sb if np.isfinite(sb) and sb > 0 else np.nan
    from math import erfc, sqrt
    pval = float(erfc(abs(z) / sqrt(2))) if np.isfinite(z) else float("nan")

    # phiên bản chỉ phân tầng, không hiệu chỉnh site/ITA
    Xu, yu, su, _ = prepare(df, col, adjust=False)
    bu, _ = fit_clogit(Xu, yu, su)

    # bootstrap cụm: resample BỆNH NHÂN (tức là tầng), giữ nguyên mọi tile của họ
    bs = []
    if boot:
        for t in range(boot):
            pick = rng.randint(0, len(strata), len(strata))
            Xb = np.concatenate([X[strata[i]] for i in pick])
            yb = np.concatenate([y[strata[i]] for i in pick])
            off, sb2 = 0, []
            for i in pick:
                n = len(strata[i]); sb2.append(np.arange(off, off + n)); off += n
            try:
                bb, _ = fit_clogit(Xb, yb, sb2, beta0=beta, maxiter=80)
                bs.append(float(bb[0]))
            except Exception:
                pass
            if (t + 1) % 100 == 0:
                log(f"      boot {t+1}/{boot}")
    bs = np.array(bs)

    return {
        "concept": cname, "column": col,
        "n_rows": int(len(y)), "n_strata": int(len(strata)), "n_cases": n_case,
        "n_covariate_cols": int(X.shape[1]),
        "log_or": b, "or": float(np.exp(b)),
        "se_cluster": sb,
        "ci_cluster": [float(np.exp(b - 1.96 * sb)), float(np.exp(b + 1.96 * sb))]
                      if np.isfinite(sb) else None,
        "z": float(z) if np.isfinite(z) else None, "p": pval,
        "or_stratum_only": float(np.exp(bu[0])),
        "boot_n": int(len(bs)),
        "boot_ci": [float(np.exp(np.percentile(bs, 2.5))),
                    float(np.exp(np.percentile(bs, 97.5)))] if len(bs) > 50 else None,
        "converged": bool(r.success),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=c1.ISIC_META_CSV if hasattr(c1, "ISIC_META_CSV") else None)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--max-patients", type=int, default=0,
                    help="chỉ giữ N bệnh nhân discordant đầu tiên (smoke test)")
    args = ap.parse_args()
    data = args.data or json.load(open(os.path.join(RES, "tier1_ate.json")))["data"]

    t0 = time.time()
    log(f"Load {data}")
    df = pd.read_csv(data, low_memory=False)
    if args.max_rows:
        keep = df.groupby(GROUP, group_keys=False).head(args.max_rows // df[GROUP].nunique() + 1)
        df = pd.concat([keep, df[df[OUTCOME] == 1]]).drop_duplicates("isic_id").reset_index(drop=True)
        log(f"  smoke test: {len(df):,} dòng")
    df["ITA_stratum"] = c1.compute_ita(df)
    if args.max_patients:
        gg = df.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
        keep = gg[(gg["sum"] > 0) & (gg["sum"] < gg["count"])].index[: args.max_patients]
        df = df[df[GROUP].isin(keep)].reset_index(drop=True)
        log(f"  smoke test: {args.max_patients} bệnh nhân discordant, {len(df):,} tile")
    g = df.groupby(GROUP)[OUTCOME].agg(["sum", "count"])
    disc = g[(g["sum"] > 0) & (g["sum"] < g["count"])]
    log(f"  tiles={len(df):,} patients={df[GROUP].nunique()} cases={int(df[OUTCOME].sum())}")
    log(f"  discordant patients={len(disc)} carrying {int(disc['sum'].sum())} cases "
        f"({100*disc['sum'].sum()/max(df[OUTCOME].sum(),1):.1f}% of all)")

    rng = np.random.RandomState(SEED)

    def snapshot(concepts, status):
        json.dump({"phase": "C41_tier1_within_patient", "status": status,
                   "note": "Ghi tăng dần trong lúc chạy; file cuối cùng là c41_within_patient.json.",
                   "concepts": concepts}, open(OUT + ".partial", "w"), indent=1)

    # Lượt 1 — điểm ước lượng và sai số chuẩn cụm (sandwich). Đây là suy diễn CHÍNH:
    # mỗi tầng đúng bằng một bệnh nhân, nên phương sai cụm có sẵn từ chính lần fit này.
    # Chạy nhanh, nên kết quả cốt lõi có mặt sớm dù bootstrap còn dài.
    log("PASS 1/2 — điểm ước lượng + sandwich cụm (không bootstrap)")
    out = []
    for c in c1.CONCEPTS:
        out.append(run_concept(df, c["name"], c["col"], 0, rng))
        snapshot(out, "pass1_point_estimates")
    log("PASS 1/2 xong — điểm ước lượng và CI cụm đã nằm trong file .partial")

    # Lượt 2 — bootstrap cụm làm kiểm chứng cho sandwich. Chậm, và không kết luận nào phụ thuộc nó.
    if args.boot:
        log(f"PASS 2/2 — bootstrap cụm {args.boot} lần mỗi concept (kiểm chứng)")
        for i, c in enumerate(c1.CONCEPTS):
            out[i] = run_concept(df, c["name"], c["col"], args.boot, rng)
            snapshot(out, f"pass2_bootstrap_{i+1}_of_{len(c1.CONCEPTS)}")

    # đối chiếu hướng với Bảng I (estimand khác nhau, chỉ so hướng và độ lớn tương đối)
    try:
        t1 = {x["concept"]: x for x in json.load(open(os.path.join(RES, "tier1_ate.json")))["concepts"]}
        for o in out:
            ref = t1.get(o["concept"], {})
            for k in ("rr", "risk_ratio", "RR"):
                if k in ref:
                    o["marginal_rr_table1"] = ref[k]; break
    except Exception as e:
        log(f"  (không đối chiếu được tier1_ate.json: {e})")

    res = {"phase": "C41_tier1_within_patient",
           "data": data,
           "design": ("Conditional logistic regression stratified by patient_id. Strata absorb every "
                      "patient-level confounder, measured or not, including acquisition site, sex and "
                      "(for 942 of 1042 patients) age, all of which are constant within patient. "
                      "Anatomical site and ITA stratum vary within patient and are adjusted for. "
                      "Standard errors are the cluster sandwich with the patient as the cluster, which "
                      "here coincides with the stratum."),
           "estimand": ("Within-patient conditional odds ratio for the top-versus-bottom tertile of the "
                        "proxy. This is NOT the estimand of Table I, which is a tile-level standardized "
                        "risk difference over the whole cohort. With a rare outcome the odds ratio "
                        "approximates a risk ratio, so the two are comparable in direction and rough "
                        "magnitude only."),
           "n_rows_total": int(len(df)), "n_patients_total": int(df[GROUP].nunique()),
           "n_cases_total": int(df[OUTCOME].sum()),
           "n_discordant_patients": int(len(disc)),
           "n_cases_in_discordant": int(disc["sum"].sum()),
           "within_patient_confounders": WITHIN_CONF,
           "boot": args.boot, "seed": SEED,
           "concepts": out, "runtime_sec": round(time.time() - t0, 1)}
    json.dump(res, open(OUT, "w"), indent=1)
    if os.path.exists(OUT + ".partial"):
        os.remove(OUT + ".partial")
    log(f"saved -> {OUT}  ({res['runtime_sec']}s)")
    print()
    print(f"{'concept':16s} {'OR':>7s} {'95% CI (cluster)':>22s} {'p':>10s} {'OR strat-only':>14s}")
    for o in out:
        if "or" not in o:
            continue
        ci = o["ci_cluster"] or [float("nan")] * 2
        print(f"{o['concept']:16s} {o['or']:7.3f} [{ci[0]:8.3f},{ci[1]:8.3f}] {o['p']:10.2e} "
              f"{o['or_stratum_only']:14.3f}")


if __name__ == "__main__":
    main()
