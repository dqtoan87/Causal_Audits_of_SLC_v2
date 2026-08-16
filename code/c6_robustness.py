#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalXAI v14 — C6: ROBUSTNESS của central finding (đáp phản biện #1, #2b, #3).

Central claim cần test: "ERM concept-aligned, Inv anti-aligned" có sống sót khi:
  #1 COMMON SET   — tính CaCE trên GIAO các eval-set (ảnh cả 3 model đều p_full≥0.1) → loại bias
                    do Inv giữ 62/78 còn ERM/Slim 78/78.
  #3 CONFIDENT     — tính trên ảnh mà CẢ ERM lẫn Inv đều confident (p_full≥τ) → tách calibration
                    artifact (Inv p≈0.5 gần boundary dễ lật dấu).
  #2b PER-DOSE     — alignment ở từng mức liều (leakage thấp ở dose thấp) → không chỉ top dose.

Edit dùng CHUNG cho mọi model (tái dùng c3_cace_concept). Bootstrap CI theo ảnh. Đọc-only model.
Out -> results/c6_robustness.json.
Chạy: .venv/bin/python Causal_XAI_v3/code/c6_robustness.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import c3_cace_concept as c3m   # soft_mask, make_edit, CONCEPTS, DOSES, model_prob, logit, precompute, p1, p2

p1, p2 = c3m.p1, c3m.p2
DEVICE = c3m.DEVICE
OUT = os.path.join(os.path.dirname(HERE), "results", "c6_robustness.json")
import audited_models                # noqa: E402  (phân giải checkpoint ở một chỗ)
CKPTS = audited_models.CKPTS         # đường dẫn TUYỆT ĐỐI tới bộ đã cố định
SEED, N_EVAL, BOOT = 42, 200, 2000
CONF_TAU = 0.7         # ngưỡng confident cho #3
CONCEPTS = ["contrast", "color_var"]   # đã qua manipulation-check ở C3


def boot_ci(vals, seed=SEED, n=BOOT):
    return p2.boot_mean_ci(np.asarray(vals, float), n, seed)


def main():
    print(f"##### C6 robustness {time.strftime('%H:%M:%S')} #####", flush=True)
    df_tr, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    imgs01, labels = p2.load_eval_images(df_va, N_EVAL, SEED)
    labels = np.array(labels)
    mal_idx = np.where(labels == 1)[0]
    print(f"eval {len(imgs01)} ảnh, malignant {len(mal_idx)}", flush=True)

    edits, names, _ = c3m.precompute(imgs01)   # names: contrast@0..2, color_var@0..2

    # p0 + dl[name] per model, CHỈ trên ảnh malignant (giữ TẤT CẢ, không lọc p_floor)
    P0 = {}                      # model -> array over mal images
    DL = {}                      # model -> {name -> array over mal images}
    for mname, fn in CKPTS.items():
        path = fn
        model = p1.build_model("erm" if mname == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE)); model.eval()
        p0s = np.empty(len(mal_idx)); dls = {n: np.empty(len(mal_idx)) for n in names}
        for j, i in enumerate(mal_idx):
            batch = torch.stack([imgs01[i]] + [edits[i][n] for n in names])
            ps = c3m.model_prob(model, batch)
            p0s[j] = ps[0]
            for k, n in enumerate(names):
                dls[n][j] = c3m.logit(ps[k + 1]) - c3m.logit(ps[0])
        P0[mname] = p0s; DL[mname] = dls
        del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
        print(f"[{mname}] p_full mean={p0s.mean():.3f}  kept(≥0.1)={int((p0s>=0.1).sum())}", flush=True)

    order = ["erm", "slim", "causal"]

    def alignment_on(mask, dose_tag="@2"):
        """A(f) và per-concept CaCE (top hoặc dose chỉ định) trên subset mask, mỗi model."""
        out = {}
        for m in order:
            per = {}
            contribs = []
            for c in CONCEPTS:
                v = DL[m][f"{c}{dose_tag}"][mask]
                mn, lo, hi = boot_ci(v)
                per[c] = dict(cace=mn, ci=[lo, hi], n=int(mask.sum()),
                              aligned=bool(lo > 0), anti=bool(hi < 0))
                contribs.append(mn)                 # sign(ATE)=+ cho cả hai
            out[m] = dict(A=float(np.mean(contribs)), per_concept=per)
        return out

    # ---- subsets ----
    keep = {m: (P0[m] >= 0.1) for m in order}
    common = keep["erm"] & keep["slim"] & keep["causal"]          # #1
    confident = (P0["erm"] >= CONF_TAU) & (P0["causal"] >= CONF_TAU)  # #3
    own = {m: keep[m] for m in order}                              # baseline (như C3)

    res = {
        "n_malignant": int(len(mal_idx)),
        "n_common": int(common.sum()), "n_confident": int(confident.sum()),
        "n_own": {m: int(keep[m].sum()) for m in order},
        "conf_tau": CONF_TAU,
        # baseline reproduces C3: mỗi model lọc theo p_full riêng của nó
        "per_model_own_filtered": {m: alignment_on(keep[m])[m] for m in order},
        "common_topdose": alignment_on(common),
        # #3
        "confident_topdose": alignment_on(confident),
        # #2b: per-dose trên common set
        "common_perdose": {f"dose{k}": alignment_on(common, f"@{k}") for k in range(len(c3m.DOSES))},
    }
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    def show(title, block):
        print(f"\n--- {title} ---")
        for m in order:
            d = block[m]; pc = d["per_concept"]
            flags = " ".join(f"{c}={pc[c]['cace']:+.2f}[{pc[c]['ci'][0]:+.2f},{pc[c]['ci'][1]:+.2f}]"
                             f"{'✓' if pc[c]['aligned'] else ('✗' if pc[c]['anti'] else '·')}"
                             for c in CONCEPTS)
            print(f"  {m:7s} A={d['A']:+.3f}  {flags}")

    print("\n" + "=" * 80)
    print(f"n_malignant={len(mal_idx)} | own keep: {res['n_own']} | COMMON={res['n_common']} | "
          f"CONFIDENT(both≥{CONF_TAU})={res['n_confident']}")
    show("baseline: own-filtered per model (reproduces C3, top dose)", res["per_model_own_filtered"])
    show("#1 COMMON set (same images all models, top dose)", res["common_topdose"])
    show(f"#3 CONFIDENT (ERM&Inv p_full≥{CONF_TAU}, top dose)", res["confident_topdose"])
    for k in range(len(c3m.DOSES)):
        show(f"#2b PER-DOSE on COMMON — dose{k} (gain={c3m.DOSES[k]})", res["common_perdose"][f"dose{k}"])
    print("=" * 80)
    print("Đọc: ✓=aligned(CI>0) ✗=anti(CI<0) ·=n.s. Central claim SỐNG nếu ERM A>0 & Inv A<0 "
          "vẫn giữ trên COMMON + CONFIDENT.")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
