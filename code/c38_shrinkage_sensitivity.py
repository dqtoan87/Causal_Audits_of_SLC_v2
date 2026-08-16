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
