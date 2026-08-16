#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C26 — Đáp ứng liều của can thiệp "hair" trên THANG XÁC SUẤT.

Thang logit không hiển thị được đáp ứng liều vì 93–99% dự đoán của hai liều cao đã
chạm biên clipping (Section V-B). Script này tính dịch chuyển xác suất CÓ DẤU của ba
mật độ hair (1.0%, 2.4%, 4.1% pixel bị đổi) trên tập ác tính đã lọc ngưỡng của từng
model, với khoảng tin cậy bootstrap theo BỆNH NHÂN.

Đọc dự đoán đã cache (results/c19_preds.npz) nên KHÔNG cần GPU và KHÔNG cần ảnh gốc;
chỉ cần metadata CSV để dựng lại patient_id của 178 ảnh đánh giá, đúng cách c32/c33 làm.

Out -> results/c26_hair_dose_prob.json   (nguồn của Fig 3d và mọi số liều trong §V-B,
       gồm cả tương phản giữa các bước liều và khoảng tin cậy của chúng)
Chạy:  python Causal_XAI_v3/code/c26_hair_dose_prob.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c21_patient_cluster as c21   # noqa: E402

BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c26_hair_dose_prob.json")

DOSES = ["art:hair_light", "art:hair", "art:hair_heavy"]
X_PCT = [1.0, 2.4, 4.1]
THRESH, BOOT, SEED = 0.10, 4000, 7
NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}


def boot_ci(v, groups, n_boot=BOOT, seed=SEED):
    """Bootstrap cụm: resample BỆNH NHÂN, giữ nguyên mọi ảnh của bệnh nhân được chọn."""
    gi = np.unique(groups, return_inverse=True)[1]
    idx = [np.where(gi == k)[0] for k in range(gi.max() + 1)]
    rng = np.random.RandomState(seed)
    pick = rng.randint(0, len(idx), (n_boot, len(idx)))
    b = np.array([v[np.concatenate([idx[k] for k in row])].mean() for row in pick])
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main():
    _, df_va = c21.p2.build_split({"seed": c21.SEED, "n_folds": 5})
    df_va = df_va.reset_index(drop=True)
    d = np.load(os.path.join(RES, "c19_preds.npz"), allow_pickle=True)
    labels = d["labels"]; names = [str(x) for x in d["names"]]
    pos = df_va[df_va["target"] == 1].sample(n=int((labels == 1).sum()), random_state=c21.SEED)
    neg = df_va[df_va["target"] == 0].sample(n=int((labels == 0).sum()), random_state=c21.SEED)
    pid = df_va.loc[np.r_[pos.index.values, neg.index.values], "patient_id"].values
    assert len(pid) == len(labels)

    cols = [names.index(x) for x in DOSES]
    res = {"x_pct": X_PCT,
           "note": ("Signed mean probability shift of the hair edit at three densities, on each "
                    "model's thresholded malignant set, with 95% bootstrap intervals over patients.")}
    for m in ["erm", "slim", "causal"]:
        p0, pe = d[f"p0_{m}"], d[f"pe_{m}"]
        sel = (labels == 1) & (p0 >= THRESH)
        rows = []
        for c in cols:
            v = (pe[:, c] - p0)[sel]
            lo, hi = boot_ci(v, pid[sel])
            rows.append({"mean": float(v.mean()), "ci": [lo, hi]})
        # tương phản giữa các bước liều: đây là đại lượng §V-B trích dẫn, nên phải
        # bootstrap trên HIỆU theo từng ảnh chứ không trừ hai trung bình đã bootstrap riêng.
        steps = []
        for lab, (i, j) in [("1.0->2.4", (0, 1)), ("2.4->4.1", (1, 2))]:
            v = (pe[:, cols[j]] - pe[:, cols[i]])[sel]
            lo, hi = boot_ci(v, pid[sel])
            steps.append({"step": lab, "mean": float(v.mean()), "ci": [lo, hi],
                          "excludes_zero": bool((lo > 0) == (hi > 0))})
        res[NAME[m]] = {"n": int(sel.sum()), "dose": rows, "steps": steps}
        print(f"  {NAME[m]:5s} n={int(sel.sum()):3d}  " +
              "  ".join(f"{r['mean']:+.3f}" for r in rows) +
              "   steps " + "  ".join(f"{s['mean']:+.3f}[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]"
                                      for s in steps), flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("saved ->", OUT)


if __name__ == "__main__":
    main()
