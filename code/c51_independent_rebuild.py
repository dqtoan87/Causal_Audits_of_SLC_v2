#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C51 — Đối chiếu với một bản CÀI ĐẶT LẠI ĐỘC LẬP của cùng đường ống.

Vì sao. Mọi kiểm tra khác trong kho này chạy trên cùng một mã nguồn, nên chúng bắt được lỗi số
học và lỗi lấy mẫu nhưng KHÔNG bắt được lỗi nằm trong chính cách hiểu đặc tả. Cách duy nhất để
dò lớp lỗi đó là cài lại từ đặc tả bằng mã khác, rồi so.

Bản đối chiếu tự dựng phân hoạch riêng, tự huấn luyện checkpoint riêng, và tự cài ước lượng tử.
Hệ quả bắt buộc: bên MÔ HÌNH không so được theo số, vì hai bên audit hai bộ mô hình khác nhau
trên hai bộ ảnh khác nhau. Chỉ so được theo PHÁN QUYẾT. Bên BỆNH thì so được theo số, vì nó chạy
trên toàn bộ metadata với ngưỡng tam phân vị tất định, không có phân hoạch ngẫu nhiên nào.

Khi hai bên bất đồng ở một ô, script làm hai việc để định vị nguyên nhân:
  (1) THÍ NGHIỆM CHÉO 2x2 — chạy CẢ HAI checkpoint trên CẢ HAI nhóm ảnh. Nếu số bám theo
      checkpoint thì nguyên nhân là mô hình; nếu bám theo nhóm ảnh thì nguyên nhân là mẫu.
  (2) QUÉT SEED — chạy checkpoint của bên kia trên chính nhóm ảnh của bên kia qua nhiều hiện
      thực chồng lớp, để biết ô đó có ổn định hay chỉ là một lần rút may.

Out -> results/c51_independent_rebuild.json
Chạy:  python Causal_XAI_v3/code/c51_independent_rebuild.py --other /đường/dẫn/bản/đối/chiếu
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c1_tier1_ate as c1            # noqa: E402
import c2_cace_artifact as c2m       # noqa: E402
import c3_cace_concept as c3m        # noqa: E402
import c30_disjoint_audit as c30     # noqa: E402
import audit_cohort                  # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c51_independent_rebuild.json")
HELD, MARGIN = c30.HELD, c30.MARGIN_P
SEEDS = [42, 1, 2, 3, 7, 11, 99]
# Đầu mô hình của bản đối chiếu gói thành MỘT Sequential; kích thước lớp giống hệt, chỉ khác tên.
REMAP = {"head.0": "proj_c.0", "head.3": "proj_c.3",
         "head.5": "disease_head.0", "head.8": "disease_head.3"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_ckpt(model, path):
    sd = torch.load(path, map_location=DEVICE)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
        sd = {(REMAP[k.rsplit(".", 1)[0]] + "." + k.rsplit(".", 1)[1]
               if k.rsplit(".", 1)[0] in REMAP else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.eval()


def read_images(ids):
    import torchvision.transforms as T
    from PIL import Image
    to_t = T.ToTensor(); size = (p1.CFG["img_size"],) * 2
    return [to_t(p1.center_crop_resize(
        Image.open(os.path.join(p2.ISIC_IMAGE_DIR, f"{i}.jpg")).convert("RGB"), size)) for i in ids]


def cell(p0, pe, msk, pid):
    absdp = np.mean([np.abs(pe[a] - p0) for a in HELD], 0)
    sgndp = np.mean([pe[a] - p0 for a in HELD], 0)
    e, lo, hi = c30.boot_ci(absdp[msk], pid[msk])
    se, slo, shi = c30.boot_ci(sgndp[msk], pid[msk])
    v, sv = c30.verdict(lo, hi, MARGIN), c30.verdict_signed(slo, shi, MARGIN)
    return {"magnitude": e, "ci": [lo, hi], "magnitude_verdict": v,
            "signed": se, "signed_ci": [slo, shi], "signed_verdict": sv,
            "joint_verdict": c30.combine(sv, v)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--other", required=True, help="gốc cây của bản cài đặt lại")
    args = ap.parse_args()
    O = args.other
    t0 = time.time()
    assert os.path.isdir(os.path.join(O, "results")), f"không thấy {O}/results"

    res = {"note": ("Comparison against an independent reimplementation of the same pipeline, "
                    "written from the specification with its own split, its own trained "
                    "checkpoints and its own estimator code. The disease side is comparable "
                    "numerically; the model side is comparable only by verdict."),
           "other_root": O, "seeds": SEEDS, "margin_p": MARGIN}

    # ---- 1. bên bệnh: so theo SỐ ------------------------------------------------------------
    a = {x["concept"]: x for x in json.load(open(RES + "/c41_within_patient.json"))["concepts"]}
    b = json.load(open(O + "/results/c41_within_patient.json"))["proxies"]
    dis = {}
    for k in a:
        if k in b:
            dis[k] = {"or_here": a[k]["or"], "or_other": b[k]["or"],
                      "abs_diff": abs(a[k]["or"] - b[k]["or"]),
                      "n_strata_here": a[k]["n_strata"], "n_strata_other": b[k]["n_strata"]}
    res["disease_side"] = {"proxies": dis,
                           "max_abs_diff": max(v["abs_diff"] for v in dis.values()),
                           "strata_all_equal": all(v["n_strata_here"] == v["n_strata_other"]
                                                   for v in dis.values())}
    log(f"bên bệnh: chênh lệch OR tối đa {res['disease_side']['max_abs_diff']:.3e}, "
        f"số tầng khớp {res['disease_side']['strata_all_equal']}")

    # ---- 2. bên mô hình: so theo PHÁN QUYẾT --------------------------------------------------
    here = json.load(open(RES + "/c30_disjoint_audit.json"))["models"]
    oth = json.load(open(O + "/results/c30_disjoint_audit.json"))["cells"]
    okey = {("b3", "ERM"): "b3_erm_s42", ("b3", "Slim"): "b3_slim_s42", ("b3", "Inv"): "b3_inv_s42",
            ("r50", "ERM"): "r50_erm_s42", ("r50", "Slim"): "r50_slim_s42", ("r50", "Inv"): "r50_inv_s42"}
    VM = {"equivalence": "within_tolerance"}
    cmp, disagree = {}, []
    for bb in here:
        for m in here[bb]:
            for s in ("all", "malignant", "benign"):
                vh = here[bb][m][s]["joint_verdict"]
                oc = oth[f"{okey[(bb, m)]}|{s}"]
                vo = "exceedance" if "exceedance" in (oc["magnitude"]["verdict"], oc["signed"]["verdict"]) else (
                    "within_tolerance" if VM.get(oc["magnitude"]["verdict"], oc["magnitude"]["verdict"]) ==
                    VM.get(oc["signed"]["verdict"], oc["signed"]["verdict"]) == "within_tolerance" else "indeterminate")
                cmp[f"{bb}|{m}|{s}"] = {"verdict_here": vh, "verdict_other": vo,
                                        "magnitude_here": here[bb][m][s]["mean"],
                                        "magnitude_other": oc["magnitude"]["estimate"],
                                        "agree": vh == vo}
                if vh != vo:
                    disagree.append((bb, m, s))
    res["model_side"] = {"n_cells": len(cmp), "n_agree": sum(c["agree"] for c in cmp.values()),
                         "cells": cmp, "disagreeing": ["|".join(d) for d in disagree]}
    log(f"bên mô hình: {res['model_side']['n_agree']}/{len(cmp)} ô cùng phán quyết; "
        f"bất đồng {res['model_side']['disagreeing']}")
    if not disagree:
        res["runtime_sec"] = round(time.time() - t0, 1)
        json.dump(res, open(OUT, "w"), indent=1); log(f"saved -> {OUT}"); return

    # ---- 3. thí nghiệm chéo 2x2 trên ô bất đồng ---------------------------------------------
    meta = pd.read_csv(json.load(open(RES + "/tier1_ate.json"))["data"], low_memory=False,
                       usecols=["isic_id", "target", "patient_id"]).set_index("isic_id")
    ih, lh, ph, _ = audit_cohort.load()
    oid = json.load(open(O + "/results/audit_cohort.json"))["isic_ids"]
    mo = meta.loc[oid]
    io_, lo_, po_ = read_images(oid), mo["target"].to_numpy(), mo["patient_id"].astype(str).to_numpy()
    res["cohorts"] = {"n_here": len(ih), "n_other": len(oid),
                      "shared_images": len(set(json.load(open(RES + "/audit_cohort.json"))["isic_id"]) & set(oid))}
    log(f"nhóm ảnh: {len(ih)} và {len(oid)}, chung {res['cohorts']['shared_images']} ảnh")

    ed = {"here": {}, "other": {}}
    for tag, imgs in [("here", ih), ("other", io_)]:
        for s in SEEDS:
            per, _ = c2m.precompute_edits(imgs, c30.ART, s)
            ed[tag][s] = {x: [d[x] for d in per] for x in HELD}
        log(f"  chồng lớp {tag}: xong {len(SEEDS)} hiện thực")

    bb, mname, sub = disagree[0]
    fn_here = {v[2]: v[0] for v in c30.TRIO[bb]}[mname]
    ckpts = [("here", os.path.join(BASE, "checkpoints", fn_here + ".pt")),
             ("other", os.path.join(O, "checkpoints", okey[(bb, mname)] + ".pt"))]
    kind = {v[2]: v[1] for v in c30.TRIO[bb]}[mname]
    grid, sweep = {}, {}
    for ctag, path in ckpts:
        m = load_ckpt(c30.make_model(kind).to(DEVICE), path)
        for dtag, imgs, lab, pid in [("here", ih, lh, ph), ("other", io_, lo_, po_)]:
            p0 = c30.probs(m, imgs)
            msk = {"all": np.ones(len(lab), bool), "malignant": lab == 1, "benign": lab == 0}[sub]
            for s in SEEDS:
                pe = {x: c30.probs(m, ed[dtag][s][x]) for x in HELD}
                r = cell(p0, pe, msk, pid)
                if s == SEEDS[0]:
                    r["auroc"] = float(roc_auc_score(lab, p0))
                    grid[f"ckpt={ctag}|cohort={dtag}"] = r
                if ctag == "other" and dtag == "other":
                    sweep[str(s)] = r
            log(f"  ckpt {ctag:5s} × cohort {dtag:5s}: {grid[f'ckpt={ctag}|cohort={dtag}']['magnitude']:.4f} "
                f"-> {grid[f'ckpt={ctag}|cohort={dtag}']['joint_verdict']}")
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    mags = [v["magnitude"] for v in sweep.values()]
    res["disagreement_diagnosis"] = {
        "cell": f"{bb}|{mname}|{sub}", "cross_grid": grid,
        "checkpoint_effect": abs(grid["ckpt=here|cohort=here"]["magnitude"]
                                 - grid["ckpt=other|cohort=here"]["magnitude"]),
        "cohort_effect": abs(grid["ckpt=here|cohort=here"]["magnitude"]
                             - grid["ckpt=here|cohort=other"]["magnitude"]),
        "other_checkpoint_seed_sweep": sweep,
        "sweep_magnitude_range": [min(mags), max(mags)],
        "sweep_verdicts": sorted({v["joint_verdict"] for v in sweep.values()}),
        "sweep_resolves_cell": len({v["joint_verdict"] for v in sweep.values()}) == 1
                               and list({v["joint_verdict"] for v in sweep.values()})[0] != "indeterminate"}
    d = res["disagreement_diagnosis"]
    log(f"  hiệu ứng checkpoint {d['checkpoint_effect']:.4f} so với hiệu ứng nhóm ảnh {d['cohort_effect']:.4f}")
    log(f"  quét seed trên checkpoint bên kia: {min(mags):.4f}–{max(mags):.4f}, phán quyết {d['sweep_verdicts']}")
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved -> {OUT}")


if __name__ == "__main__":
    main()
