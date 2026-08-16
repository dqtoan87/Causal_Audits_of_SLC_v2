#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CausalXAI v14 — C30: audit artifact Tier-2 trên fold AUDIT tách bệnh nhân.

Sáu model của C29 được chọn checkpoint chỉ dựa trên fold SELECT. Fold AUDIT chưa hề được dùng cho
huấn luyện, chọn checkpoint hay hiệu chỉnh. Vì vậy con số ở đây KHÔNG phải chẩn đoán hậu-chọn, khác
với audit chính của bài.

Đo đúng tiêu chí chính của bài: E|Δp| của các artifact held-out (hair ba liều, ruler), theo từng
model và từng nhóm con (tất cả, ác tính, lành), với khoảng tin cậy bootstrap theo bệnh nhân, rồi
áp quy tắc khoảng ba chiều so với dung sai 0.05.

Cũng khớp calibration (temperature và Platt) trên SELECT rồi báo cáo trên AUDIT, nên các chỉ số
calibration ở đây là ước lượng ngoài mẫu thật sự.

Out -> results/c30_disjoint_audit.json
Chạy:  .venv/bin/python Causal_XAI_v3/code/c30_disjoint_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c2_cace_artifact as c2m       # noqa: E402  (ARTIFACTS, precompute_edits)
import c8_backbone2_train as c8      # noqa: E402
import c29_disjoint_train as c29     # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import audit_cohort               # noqa: E402  (danh sách ảnh audit cố định)

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c30_disjoint_audit.json")
SEED, N_EVAL, BOOT = 42, 200, 4000
EPS = 1e-6
MARGIN_P = 0.05
ART = ["hair_light", "hair", "hair_heavy", "ruler", "patch", "vignette", "jitter", "identity"]
HELD = ["hair_light", "hair", "hair_heavy", "ruler"]
TRIO = {"b3": [("dj_erm_b3", "erm", "ERM"), ("dj_slim_b3", "slim", "Slim"),
               ("dj_causal_b3", "slim", "Inv")],
        "r50": [("dj_erm_r50", "erm_r50", "ERM"), ("dj_slim_r50", "slim_r50", "Slim"),
                ("dj_causal_r50", "slim_r50", "Inv")]}


def make_model(kind):
    if kind == "erm":
        return p1.build_model("erm", p1.CFG["z_dim"])
    if kind == "slim":
        return p1.build_model("slim", p1.CFG["z_dim"])
    if kind == "erm_r50":
        return c8.ERM_R50()
    return c8.Slim_R50(p1.CFG["z_dim"])


def clipped_logit(p):
    q = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(q / (1 - q))


@torch.no_grad()
def probs(model, batch):
    out = []
    for i in range(0, len(batch), 32):
        x = torch.stack(batch[i:i + 32]).to(DEVICE)
        x = (x - c3m.MEAN.to(DEVICE)) / c3m.STD.to(DEVICE) if hasattr(c3m, "MEAN") else x
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            out.append(torch.softmax(model(x).float(), 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def boot_ci(v, groups, n_boot=BOOT, seed=3):
    g = [np.where(groups == q)[0] for q in np.unique(groups)]
    r = np.random.RandomState(seed)
    b = np.array([v[np.concatenate([g[i] for i in r.randint(0, len(g), len(g))])].mean()
                  for _ in range(n_boot)])
    return float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def verdict(lo, hi, margin):
    """Quy tắc khoảng ba chiều của Section III-C, trên đại lượng không âm E|Δp|."""
    if lo > margin:
        return "exceedance"
    if hi < margin:
        return "equivalence"
    return "indeterminate"


def main():
    t0 = time.time()
    print(f"##### C30 audit trên fold tách bệnh nhân {time.strftime('%H:%M:%S')} #####", flush=True)
    imgs, labels, pid, _ = audit_cohort.load()   # danh sách cố định, không dựng lại phân hoạch
    print(f"AUDIT eval: {len(imgs)} ảnh / {len(np.unique(pid))} bệnh nhân "
          f"({int((labels==1).sum())} ác tính)", flush=True)

    edits, change_frac = c2m.precompute_edits(imgs, ART, SEED)
    res = {"n_images": int(len(imgs)), "n_patients": int(len(np.unique(pid))),
           "n_malignant": int((labels == 1).sum()), "margin_p": MARGIN_P, "boot": BOOT,
           "change_frac": change_frac,
           "note": ("Tier-2 artifact criterion measured on a patient-disjoint audit fold that was "
                    "never used for training, checkpoint selection or calibration. Unlike the "
                    "primary audit, these numbers are not post-selection."),
           "models": {}}

    W = 96
    print("=" * W)
    print(f"{'backbone':9s}{'model':6s}{'subgroup':11s}{'E|Δp| held-out':>16s}{'95% CI':>22s}"
          f"{'vs 0.05':>14s}")
    n_exc = n_cell = 0
    for bb, trio in TRIO.items():
        for ck, kind, nm in trio:
            f = os.path.join(CKPT, ck + ".pt")
            if not os.path.exists(f):
                print(f"  [{ck}] chưa có checkpoint, bỏ qua", flush=True); continue
            m = make_model(kind).to(DEVICE)
            m.load_state_dict(torch.load(f, map_location=DEVICE)); m.eval()
            p0 = probs(m, imgs)
            per = {}
            for a in ART:
                pe = probs(m, [edits[i][a] for i in range(len(imgs))])
                per[a] = pe
            del m
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            absdp = np.mean([np.abs(per[a] - p0) for a in HELD], 0)
            cell = {}
            for sub, msk in [("all", np.ones(len(labels), bool)), ("malignant", labels == 1),
                             ("benign", labels == 0)]:
                mean, lo, hi = boot_ci(absdp[msk], pid[msk])
                v = verdict(lo, hi, MARGIN_P)
                n_exc += (v == "exceedance"); n_cell += 1
                cell[sub] = {"mean": mean, "ci": [lo, hi], "verdict": v}
                print(f"{bb:9s}{nm:6s}{sub:11s}{mean:16.3f}"
                      f"{('[%.3f, %.3f]' % (lo, hi)):>22s}{v:>14s}")
            cell["identity_max_abs_dlogit"] = float(
                np.abs(clipped_logit(per["identity"]) - clipped_logit(p0)).max())
            cell["hair_dose_dp"] = [float((per[a] - p0)[labels == 1].mean())
                                    for a in ["hair_light", "hair", "hair_heavy"]]
            cell["p0_mean_malignant"] = float(p0[labels == 1].mean())
            res["models"].setdefault(bb, {})[nm] = cell
        print("-" * W)
    res["summary"] = {"cells": n_cell, "exceedance": n_exc}
    print(f"  vượt dung sai {MARGIN_P}: {n_exc}/{n_cell} ô")
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print("=" * W)
    print(f"saved -> {OUT}   ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
