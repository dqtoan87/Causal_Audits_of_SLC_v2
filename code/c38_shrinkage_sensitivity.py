#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CausalXAI v14 — C38: hồi quy co, khoảng tin cậy và độ nhạy với biên cắt.

Vì sao cần. Quan hệ Δℓ = −α·L₀ + β là thành phần thực nghiệm trung tâm giải thích vì sao trừ
placebo thất bại, nhưng Bảng IX chỉ báo α, β và R². Hai câu hỏi còn để ngỏ: α có khác 0 không khi
kể tới phụ thuộc giữa các ảnh của một bệnh nhân, và quan hệ này có phải chỉ là hệ quả của việc
đầu ra bị chặn hay không.

Script này bổ sung cả hai, không cần GPU: đọc lại dự đoán đã lưu ở c19_preds.npz, bootstrap α theo
BỆNH NHÂN, rồi khớp lại sau khi loại mọi quan sát mà logit gốc hoặc logit sau chỉnh sửa chạm biên
cắt ±13.8155.

Out -> results/c38_shrinkage_sensitivity.json
Chạy:  .venv/bin/python Causal_XAI_v3/code/c38_shrinkage_sensitivity.py
"""
from __future__ import annotations

import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c21_patient_cluster as c21
import audit_cohort               # noqa: E402

RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c38_shrinkage_sensitivity.json")
EPS, BOUND, BOOT, SEED = 1e-6, 13.815509557963773, 20000, 23
PB = ["pb:flip", "pb:jitter", "pb:hue", "pb:noise"]
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}


def clog(p):
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


def _design(x, kind, knots=None):
    """Ma trận thiết kế: hằng số, tuyến tính, hoặc spline bậc ba tự nhiên."""
    x = np.asarray(x, float)
    if kind == "const":
        return np.ones((len(x), 1))
    if kind == "linear":
        return np.column_stack([np.ones(len(x)), x])
    # natural cubic spline, cơ sở truncated power với ràng buộc tuyến tính ở hai đuôi
    k = np.asarray(knots, float)
    d = lambda j: (np.maximum(x - k[j], 0) ** 3 - np.maximum(x - k[-1], 0) ** 3) / (k[-1] - k[j])
    cols = [np.ones(len(x)), x] + [d(j) - d(len(k) - 2) for j in range(len(k) - 2)]
    return np.column_stack(cols)


def cv_r2_by_patient(y, x, groups, kind, n_knots=5, seed=SEED):
    """R² ngoài mẫu, chia fold theo BỆNH NHÂN nên không rò rỉ giữa các ảnh cùng người."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    g = np.asarray(groups); uq = np.unique(g)
    rng = np.random.RandomState(seed); order = rng.permutation(len(uq))
    fold = {uq[i]: order[j] % 5 for j, i in enumerate(range(len(uq)))}
    fid = np.array([fold[v] for v in g])
    knots = np.quantile(x, np.linspace(0.05, 0.95, n_knots)) if kind == "spline" else None
    pred = np.empty(len(y)); pred[:] = np.nan
    for f in range(5):
        tr, te = fid != f, fid == f
        if te.sum() == 0 or tr.sum() < 10:
            continue
        Xtr, Xte = _design(x[tr], kind, knots), _design(x[te], kind, knots)
        beta, *_ = np.linalg.lstsq(Xtr, y[tr], rcond=None)
        pred[te] = Xte @ beta
    ok = ~np.isnan(pred)
    ss_res = ((y[ok] - pred[ok]) ** 2).sum()
    ss_tot = ((y[ok] - y[ok].mean()) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-12))


def ols(y, x):
    """Trả (slope, intercept, R²) của y ~ x."""
    if len(x) < 5 or x.std() < 1e-12:
        return float("nan"), float("nan"), float("nan")
    b, a = np.polyfit(x, y, 1)
    r = y - (b * x + a)
    ss = ((y - y.mean()) ** 2).sum()
    return float(b), float(a), float(1 - (r ** 2).sum() / ss) if ss > 0 else float("nan")


def boot_slope(y, x, groups, n_boot=BOOT, seed=SEED):
    gi = np.unique(groups, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_boot):
        sel = np.concatenate([idx[k] for k in rng.randint(0, len(idx), len(idx))])
        b, _, _ = ols(y[sel], x[sel])
        if np.isfinite(b):
            out.append(b)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def main():
    t0 = time.time()
    print(f"##### C38 hồi quy co: CI và độ nhạy biên cắt {time.strftime('%H:%M:%S')} #####", flush=True)
    _imgs, _labels, _pid, _ids = audit_cohort.load()   # tập đánh giá cố định
    d = np.load(os.path.join(RES, "c19_preds.npz"), allow_pickle=True)
    labels = d["labels"]; names = [str(x) for x in d["names"]]
    pid = _pid
    print("  %d ảnh / %d bệnh nhân" % (len(pid), len(np.unique(pid))), flush=True)

    res = {"n_images": int(len(pid)), "n_patients": int(len(np.unique(pid))), "boot": BOOT,
           "clip_bound": BOUND,
           "note": ("Shrinkage regression of the neutral-edit logit change on the baseline logit, "
                    "with 95% intervals for the slope bootstrapped over patients, and a refit that "
                    "drops every observation whose baseline or edited logit sits at the clipping "
                    "bound. The second answers whether the relation is an artefact of the bounded "
                    "output."), "models": {}}
    print("\n%-5s %-10s %8s %8s %8s | %8s %8s %6s %s"
          % ("model", "edit", "α", "CI lo", "CI hi", "α no-clip", "R² nc", "n bỏ", "đổi"), flush=True)
    for m, nm in NAME.items():
        L0 = clog(d["p0_" + m])
        for e in PB:
            j = names.index(e)
            Le = clog(d["pe_" + m][:, j])
            dl = Le - L0
            b, a, r2 = ols(dl, L0)
            lo, hi = boot_slope(dl, L0, pid)
            keep = (np.abs(L0) < BOUND - 1e-9) & (np.abs(Le) < BOUND - 1e-9)
            b2, a2, r22 = ols(dl[keep], L0[keep])
            res["models"].setdefault(nm, {})[e] = {
                "slope": b, "intercept": a, "r2": r2, "slope_ci_patient": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "n_dropped": int((~keep).sum()),
                "slope_noclip": b2, "intercept_noclip": a2, "r2_noclip": r22,
                "rel_change": abs(b2 - b) / max(abs(b), 1e-9)}
            print("%-5s %-10s %+8.3f %+8.3f %+8.3f | %+8.3f %8.2f %6d %5.1f%%"
                  % (nm, e[3:], b, lo, hi, b2, r22, int((~keep).sum()), 100 * abs(b2 - b) / max(abs(b), 1e-9)),
                  flush=True)
    # Hai kiểm chứng cho hai khẳng định ở IV-C: quan hệ có giữ trên thang xác suất (nơi biên cắt
    # không đóng vai trò gì) và phép sửa khái niệm có tuân theo cùng dạng hay không.
    CPT = [n for n in names if n.startswith("cpt:")]
    res["probability_scale"] = {}
    res["concept_edits_logit"] = {}
    for m, nm in NAME.items():
        p0 = d["p0_" + m]; L0 = clog(p0)
        for e in PB:
            j = names.index(e)
            b, a, r2 = ols(d["pe_" + m][:, j] - p0, p0)
            res["probability_scale"].setdefault(nm, {})[e] = {"slope": b, "intercept": a, "r2": r2}
        for e in CPT:
            j = names.index(e)
            dl = clog(d["pe_" + m][:, j]) - L0
            b, a, r2 = ols(dl, L0)
            lo, hi = boot_slope(dl, L0, pid)
            res["concept_edits_logit"].setdefault(nm, {})[e] = {
                "slope": b, "intercept": a, "r2": r2, "slope_ci_patient": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0)}
    # Độ dốc có phải chỉ là đặc tính của chính 183 ảnh này không? Chia BỆNH NHÂN làm hai nửa,
    # khớp trên một nửa và báo cáo độ dốc của nửa kia. Hai nửa không dùng chung bệnh nhân nào.
    res["holdout_slopes"] = {}
    uq = np.unique(pid); rr = np.random.RandomState(SEED)
    perm = rr.permutation(len(uq)); A = set(uq[perm[:len(uq)//2]])
    mA = np.array([p in A for p in pid]); mB = ~mA
    for m, nm in NAME.items():
        L0 = clog(d["p0_" + m])
        for e in PB:
            j = names.index(e); dl = clog(d["pe_" + m][:, j]) - L0
            bA = ols(dl[mA], L0[mA]); bB = ols(dl[mB], L0[mB])
            res["holdout_slopes"].setdefault(nm, {})[e] = {
                "slope_half_a": bA[0], "slope_half_b": bB[0],
                "r2_half_a": bA[2], "r2_half_b": bB[2],
                "same_sign": bool(bA[0] * bB[0] > 0)}
    # Kiểm chứng ĐẦY ĐỦ phương trình Δℓ = −α·L₀ + b: khớp α và b trên nửa A, dự báo trên nửa B,
    # rồi hỏi mô hình có dự báo đúng sai số của phép trừ placebo hay không. Sai số đó là
    # (Δℓ_edit − Δℓ_control) − (b_e − b_c), tức phần mà công thức thiên lệch quy cho chênh lệch α.
    res["holdout_equation"] = {}
    for m, nm in NAME.items():
        L0 = clog(d["p0_" + m])
        for e in PB:
            j = names.index(e); dl = clog(d["pe_" + m][:, j]) - L0
            bA, aA, _ = ols(dl[mA], L0[mA])
            predB = bA * L0[mB] + aA
            ssr = ((dl[mB] - predB) ** 2).sum(); sst = ((dl[mB] - dl[mB].mean()) ** 2).sum()
            cal = ols(dl[mB], predB)          # hồi quy thực tế lên dự báo: hệ số 1 là hiệu chuẩn hoàn hảo
            res["holdout_equation"].setdefault(nm, {})[e] = {
                "r2_heldout": float(1 - ssr / max(sst, 1e-12)),
                "calibration_slope": cal[0], "calibration_intercept": cal[1],
                "rmse_heldout": float(np.sqrt(ssr / len(predB))),
                "sd_heldout": float(dl[mB].std())}
    # Công thức thiên lệch có dự báo đúng sai số trừ placebo không? Với mỗi cặp edit/control trung
    # tính, so sai số quan sát được với (α_c − α_e)·E[L₀] mà mô hình khớp trên nửa A tiên đoán.
    res["bias_prediction"] = {}
    for m, nm in NAME.items():
        L0 = clog(d["p0_" + m]); fit = {}
        for e in PB + [x for x in names if x.startswith("cpt:") and x.endswith("@2")]:
            j = names.index(e); dl = clog(d["pe_" + m][:, j]) - L0
            fit[e] = ols(dl[mA], L0[mA])[:2]          # (slope, intercept) trên nửa A
        for ce in [x for x in names if x.startswith("cpt:") and x.endswith("@2")]:
            for pc in PB:
                # fit trả (slope, intercept) với slope = −α, nên slope_e − slope_c = α_c − α_e.
                slope_e, b_e = fit[ce]; slope_c, b_c = fit[pc]
                pred = (b_e - b_c) + (slope_e - slope_c) * float(L0[mB].mean())
                je, jp = names.index(ce), names.index(pc)
                obs = float(((clog(d["pe_" + m][:, je]) - L0) - (clog(d["pe_" + m][:, jp]) - L0))[mB].mean())
                res["bias_prediction"].setdefault(nm, {})["%s|%s" % (ce[4:], pc[3:])] = {
                    "predicted_corrected": float(pred), "observed_corrected": obs,
                    "abs_error": float(abs(pred - obs)), "same_sign": bool(pred * obs > 0)}
    he = [v for mm in res["holdout_equation"].values() for v in mm.values()]
    bp = [v for mm in res["bias_prediction"].values() for v in mm.values()]
    print("\nphương trình trên nửa giữ lại: R² %.2f tới %.2f | hệ số hiệu chuẩn %.2f tới %.2f"
          % (min(v["r2_heldout"] for v in he), max(v["r2_heldout"] for v in he),
             min(v["calibration_slope"] for v in he), max(v["calibration_slope"] for v in he)), flush=True)
    print("dự báo sai số trừ placebo: cùng dấu %d/%d | |sai số| trung vị %.2f logit"
          % (sum(v["same_sign"] for v in bp), len(bp),
             float(np.median([v["abs_error"] for v in bp]))), flush=True)

    hs = [v for mm in res["holdout_slopes"].values() for v in mm.values()]
    print("\nchia bệnh nhân hai nửa (%d/%d bệnh nhân): cùng dấu ở %d/%d ô, |chênh| lớn nhất %.2f"
          % (len(A), len(uq), sum(v["same_sign"] for v in hs), len(hs),
             max(abs(v["slope_half_a"] - v["slope_half_b"]) for v in hs)), flush=True)

    # Dạng tuyến tính có bị áp đặt không? So sánh hằng số / tuyến tính / spline bậc ba tự nhiên
    # bằng R² ngoài mẫu, chia fold theo bệnh nhân. Hằng số = "edit cộng một lượng cố định".
    res["functional_form"] = {}
    for m, nm in NAME.items():
        L0 = clog(d["p0_" + m])
        for e in PB:
            j = names.index(e)
            dl = clog(d["pe_" + m][:, j]) - L0
            r = {k: cv_r2_by_patient(dl, L0, pid, k) for k in ["const", "linear", "spline"]}
            res["functional_form"].setdefault(nm, {})[e] = r
    ff = [v for mm in res["functional_form"].values() for v in mm.values()]
    print("\ndạng hàm (R² ngoài mẫu, fold theo bệnh nhân):", flush=True)
    print("  tuyến tính > hằng số ở %d/%d ô; spline > tuyến tính ở %d/%d ô"
          % (sum(v["linear"] > v["const"] for v in ff), len(ff),
             sum(v["spline"] > v["linear"] for v in ff), len(ff)), flush=True)
    print("  cải thiện trung vị spline so với tuyến tính: %+.4f"
          % float(np.median([v["spline"] - v["linear"] for v in ff])), flush=True)

    ps = [v["slope"] for mm in res["probability_scale"].values() for v in mm.values()]
    ce = [v for mm in res["concept_edits_logit"].values() for v in mm.values()]
    print("\nthang xác suất: độ dốc %+.2f tới %+.2f, tất cả âm=%s"
          % (min(ps), max(ps), all(x < 0 for x in ps)), flush=True)
    print("phép sửa khái niệm (logit): độ dốc %+.2f tới %+.2f, %d/%d khoảng loại trừ 0"
          % (min(c["slope"] for c in ce), max(c["slope"] for c in ce),
             sum(c["excludes_zero"] for c in ce), len(ce)), flush=True)

    cells = [v for mm in res["models"].values() for v in mm.values()]
    res["summary"] = {"n_cells": len(cells),
                      "slope_ci_excludes_zero": sum(c["excludes_zero"] for c in cells),
                      "max_dropped": max(c["n_dropped"] for c in cells),
                      "max_rel_change": max(c["rel_change"] for c in cells)}
    print("\nsummary:", json.dumps(res["summary"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved -> %s  (%.0fs)" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
