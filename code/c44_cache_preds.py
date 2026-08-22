#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C44 — Sinh cache dự đoán cho tập đánh giá cố định.

Vì sao. `results/c19_preds.npz` được bốn phân tích đọc (c26, c32, c38 và script vẽ hình co rút)
nhưng trước đây không script nào sinh ra nó. Một file cache không có nguồn gốc là chỗ mà bài báo
không tái lập được, y như hai file kết quả mồ côi đã xử lý trước đó. Script này sinh nó, từ cùng
tập ảnh cố định và cùng bộ edit mà c13 dùng, nên mọi phân tích đọc nó đều nhất quán.

Nội dung file: labels, names (18 edit), và p0_*, pe_* cho ba model EfficientNet-B3.

Out -> results/c19_preds.npz
Chạy:  python code/c44_cache_preds.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c13_fullpop as c13          # noqa: E402  (build_all_edits, eval_model, CKPTS, ORDER)
import audit_cohort                # noqa: E402

p1, DEVICE = c13.p1, c13.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results")
OUT = os.path.join(RES, "c19_preds.npz")


def main():
    t0 = time.time()
    print(f"##### C44 cache dự đoán {time.strftime('%H:%M:%S')} #####", flush=True)
    imgs01, labels, pid, ids = audit_cohort.load()
    labels = np.array(labels)
    print(f"  {len(imgs01)} ảnh / {len(set(pid))} bệnh nhân / {int(labels.sum())} ác tính", flush=True)

    edits, names = c13.build_all_edits(imgs01)
    print(f"  {len(names)} edit mỗi ảnh ({time.time()-t0:.0f}s)", flush=True)

    out = {"labels": labels, "names": np.array(names, dtype=object),
           "patient_id": np.array(pid, dtype=object), "isic_id": np.array(ids, dtype=object)}
    for m in c13.ORDER:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(c13.CKPTS[m], map_location=DEVICE))
        model.eval()
        p0, pe = c13.eval_model(model, imgs01, edits, names)
        out[f"p0_{m}"] = p0
        out[f"pe_{m}"] = pe
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        print(f"  [{m}] p0 trung bình {p0.mean():.3f}  ({time.time()-t0:.0f}s)", flush=True)

    np.savez_compressed(OUT, **out)
    print(f"saved -> {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
