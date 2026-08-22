#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C42 — Positive control: chứng minh tiêu chí artifact CÓ vùng chấp nhận.

Vấn đề. Trong toàn bộ bài, không ô nào đạt `equivalence`. Một tiêu chí chỉ biết bác bỏ thì
chưa chứng minh được là nó PHÂN BIỆT được. Phép kiểm identity edit trả về đúng 0 chỉ xác nhận
đường ống, không xác nhận lực phân biệt trên can thiệp thật.

Thiết kế. Hạ liều hair xuống dưới mức đang dùng trong bài và chạy đúng tiêu chí ba nhánh trên
đúng cohort patient-disjoint của Bảng 2. Nếu ở liều đủ nhỏ có ô đạt `equivalence`, và verdict
chuyển dần exceedance khi liều tăng, thì tiêu chí vừa có vùng chấp nhận vừa bám cường độ can
thiệp. Điều đó biến dose-response từ một mô tả thành một HIỆU CHUẨN của chính tiêu chí.

Thang liều. Hàm _hair(x, rng, n, alpha) nhận số sợi n và độ mờ alpha. Ba liều của bài là
(6, 0.40), (14, 0.60), (24, 0.75), tương ứng 1.0%, 2.4% và 4.1% pixel bị đổi. Ta thêm ba liều
thấp hơn hẳn ở phía dưới.

Out -> results/c42_positive_control.json
Chạy:  python code/c42_positive_control.py [--boot 4000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c2_cace_artifact as c2m      # noqa: E402
import c29_disjoint_train as c29    # noqa: E402
import c30_disjoint_audit as c30    # noqa: E402
import c3_cace_concept as c3m       # noqa: E402
import audit_cohort               # noqa: E402  (danh sách ảnh audit cố định)

p2, DEVICE = c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c42_positive_control.json")

SEED, N_EVAL = c30.SEED, c30.N_EVAL
MARGIN_P = c30.MARGIN_P

# thang liều: ba mức mới ở dưới, rồi ba mức của bài
LADDER = [("hair_x1", 1, 0.15), ("hair_x2", 2, 0.25), ("hair_x3", 3, 0.35),
          ("hair_light", 6, 0.40), ("hair", 14, 0.60), ("hair_heavy", 24, 0.75)]


def register_doses():
    """Thêm ba liều thấp vào bảng ARTIFACTS. Ba liều còn lại đã có sẵn và giữ nguyên định nghĩa."""
    for name, n, a in LADDER:
        if name in c2m.ARTIFACTS:
            continue
        c2m.ARTIFACTS[name] = (lambda n_, a_: (lambda x, rng: c2m._hair(x, rng, n=n_, alpha=a_)))(n, a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=c30.BOOT)
    ap.add_argument("--models", default="all", choices=["all", "b3"])
    args = ap.parse_args()

    t0 = time.time()
    print(f"##### C42 positive control {time.strftime('%H:%M:%S')} #####", flush=True)
    register_doses()
    names = [n for n, _, _ in LADDER] + ["ruler", "identity"]
    HELD = ["hair_light", "hair", "hair_heavy", "ruler"]   # đúng tập của Bảng 2

    imgs, labels, pid, _ = audit_cohort.load()   # danh sách cố định, không dựng lại phân hoạch
    print(f"audit cohort: {len(imgs)} ảnh / {len(np.unique(pid))} bệnh nhân "
          f"({int((labels==1).sum())} ác tính)", flush=True)

    edits, change_frac = c2m.precompute_edits(imgs, names, SEED)
    print("pixel bị đổi theo liều: " +
          "  ".join(f"{n}={100*change_frac[n]:.2f}%" for n in names), flush=True)

    trio = dict(c30.TRIO) if args.models == "all" else {"b3": c30.TRIO["b3"]}
    res = {"margin_p": MARGIN_P, "boot": args.boot, "seed": SEED,
           "cohort": {"n_images": int(len(imgs)), "n_patients": int(len(np.unique(pid))),
                      "n_malignant": int((labels == 1).sum())},
           "ladder": [{"name": n, "n_strands": s, "alpha": a, "change_frac": change_frac[n]}
                      for n, s, a in LADDER],
           "identity_change_frac": change_frac["identity"],
           "note": ("Calibration of the artifact criterion. The same three-way interval rule and "
                    "the same patient-disjoint audit cohort as the main artifact table, evaluated "
                    "over a dose ladder that extends below the doses reported there. A cell that "
                    "demonstrates equivalence shows the criterion has an acceptance region."),
           "models": {}}

    W = 104
    print("=" * W)
    print(f"{'backbone':9s}{'model':6s}{'dose':11s}{'%px':>7s}{'subgroup':11s}"
          f"{'E|Δp|':>9s}{'95% CI':>22s}{'verdict':>16s}")
    for bb, models in trio.items():
        for ck, kind, nm in models:
            f = os.path.join(CKPT, ck + ".pt")
            if not os.path.exists(f):
                print(f"  [{ck}] thiếu checkpoint, bỏ qua", flush=True); continue
            m = c30.make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = c30.probs(m, imgs)
            cells = {}
            for a in names:
                pe = c30.probs(m, [edits[i][a] for i in range(len(imgs))])
                absdp = np.abs(pe - p0)
                cells[a] = {}
                for sub, msk in [("all", np.ones(len(labels), bool)),
                                 ("malignant", labels == 1), ("benign", labels == 0)]:
                    _, lo, hi = c30.boot_ci(absdp[msk], pid[msk], n_boot=args.boot)
                    v = c30.verdict(lo, hi, MARGIN_P)
                    cells[a][sub] = {"mean": float(absdp[msk].mean()), "ci": [lo, hi],
                                     "verdict": v}
                    print(f"{bb:9s}{nm:6s}{a:11s}{100*change_frac[a]:7.2f}{sub:>11s}"
                          f"{absdp[msk].mean():9.4f}{f'[{lo:.4f}, {hi:.4f}]':>22s}{v:>16s}",
                          flush=True)
            # đại lượng của Bảng 2: trung bình |Δp| trên BỐN edit held-out
            agg = np.mean([np.abs(c30.probs(m, [edits[i][a] for i in range(len(imgs))]) - p0)
                           for a in HELD], 0)
            cells["_heldout_aggregate"] = {}
            for sub, msk in [("all", np.ones(len(labels), bool)),
                             ("malignant", labels == 1), ("benign", labels == 0)]:
                _, lo, hi = c30.boot_ci(agg[msk], pid[msk], n_boot=args.boot)
                cells["_heldout_aggregate"][sub] = {"mean": float(agg[msk].mean()),
                                                    "ci": [lo, hi],
                                                    "verdict": c30.verdict(lo, hi, MARGIN_P)}
            print(f"{bb:9s}{nm:6s}{'AGG(4 held-out)':21s}"
                  f"{agg.mean():9.4f}   <- đại lượng của Bảng 2", flush=True)
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            res["models"].setdefault(bb, {})[nm] = cells
            json.dump(res, open(OUT + ".partial", "w"), indent=1)

    # tổng kết: tiêu chí có vùng chấp nhận không, và chuyển pha ở đâu
    summ = {"equivalence_cells": [], "indeterminate_cells": [], "n_cells": 0}
    for bb, mm in res["models"].items():
        for nm, cells in mm.items():
            for a in names:
                for sub, c in cells[a].items():
                    summ["n_cells"] += 1
                    if c["verdict"] == "equivalence":
                        summ["equivalence_cells"].append(f"{bb}/{nm}/{a}/{sub}")
                    elif c["verdict"] == "indeterminate":
                        summ["indeterminate_cells"].append(f"{bb}/{nm}/{a}/{sub}")
    summ["criterion_has_acceptance_region"] = bool(summ["equivalence_cells"])
    res["summary"] = summ
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    if os.path.exists(OUT + ".partial"):
        os.remove(OUT + ".partial")

    print("=" * W)
    print(f"ô đạt equivalence : {len(summ['equivalence_cells'])} / {summ['n_cells']}")
    print(f"ô indeterminate   : {len(summ['indeterminate_cells'])}")
    print("tiêu chí CÓ vùng chấp nhận" if summ["criterion_has_acceptance_region"]
          else "tiêu chí KHÔNG đạt equivalence ở bất kỳ liều nào đã thử")
    print(f"saved -> {OUT}  ({res['runtime_sec']}s)")


if __name__ == "__main__":
    main()
