#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalXAI v14 — C11: đáp #1 (sign-robustness principle), #2 (magnitude-vs-response), #3 (placebo
table), #5 (MDE). Dữ liệu đã có; bổ sung phép đo perturbation-magnitude.

Ý chính (placebo-free): nếu response chỉ do perturbation MAGNITUDE (smooth-function property) thì
concept-edit và neutral-edit phải CÙNG DẤU (đều là perturbation -> cùng hướng response, scale theo
magnitude). Quan sát: với ERM concept→DƯƠNG nhưng MỌI neutral→ÂM (dấu NGƯỢC) => không thể là magnitude
=> concept-specific. Với Inv concept & neutral đều ÂM => không phân biệt => indifferent. Sign-dissociation
là invariant KHÔNG cần chọn "control đúng".

Đo: per edit {flip,jitter,hue,noise,contrast@0,contrast@2,color@0,color@2}: (a) magnitude = mean
|Δpixel| (L2 màu) + %pixel; (b) signed CaCE per model (common set) + MDE. Out -> c11_magnitude.json
+ fig_magnitude.png.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m
import c7_extra_controls as c7m
import c10_multiplacebo as c10
p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
RES = os.path.join(os.path.dirname(HERE), "results"); OUT = os.path.join(RES, "c11_magnitude.json")
FIG = os.path.join(os.path.dirname(HERE), "paper", "figures")
import audited_models                # noqa: E402  (phân giải checkpoint ở một chỗ)
CKPTS = audited_models.CKPTS         # đường dẫn TUYỆT ĐỐI tới bộ đã cố định
SEED, N_EVAL, BOOT = 42, 200, 2000
COL = {"erm": "#1f77b4", "slim": "#7f7f7f", "causal": "#d62728"}; NAME = {"erm": "ERM", "slim": "Slim", "causal": "Inv"}
# kind: neutral vs concept
EDITS = [("flip", "neutral"), ("jitter", "neutral"), ("hue", "neutral"), ("noise", "neutral"),
         ("contrast@0", "concept"), ("contrast@2", "concept"), ("color_var@0", "concept"), ("color_var@2", "concept")]


def build(x, name, rng):
    w, skin, lm = c3m.soft_mask(x)
    if name in c10.PLACEBOS:
        return c10.PLACEBOS[name](x, w, skin, np.random.RandomState(SEED + 1))
    c, k = name.split("@"); return c3m.make_edit(x, c, c3m.DOSES[int(k)], w, skin, lm)


def boot(v): return p2.boot_mean_ci(np.asarray(v, float), BOOT, SEED)


def main():
    print(f"##### C11 magnitude {time.strftime('%H:%M:%S')} #####", flush=True)
    _, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    imgs01, labels = p2.load_eval_images(df_va, N_EVAL, SEED)
    labels = np.array(labels); mal = np.where(labels == 1)[0]
    names = [e for e, _ in EDITS]

    edits, mag, pct = [], {n: [] for n in names}, {n: [] for n in names}
    for ii in mal:
        x = imgs01[ii]; rng = np.random.RandomState(SEED + int(ii)); d = {}
        for n in names:
            e = build(x, n, rng); d[n] = e
            dpix = (e - x).pow(2).sum(0).sqrt()         # L2 màu mỗi pixel
            mag[n].append(float(dpix.mean())); pct[n].append(float((dpix > 0.02).float().mean()))
        edits.append(d)
    mag = {n: float(np.mean(mag[n])) for n in names}; pct = {n: float(np.mean(pct[n])) for n in names}

    order = ["erm", "slim", "causal"]; P0, DL = {}, {}
    for m in order:
        model = p1.build_model("erm" if m == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(CKPTS[m], map_location=DEVICE)); model.eval()
        p0 = np.empty(len(mal)); dl = {n: np.empty(len(mal)) for n in names}
        for j, ii in enumerate(mal):
            ps = c3m.model_prob(model, torch.stack([imgs01[ii]] + [edits[j][n] for n in names])); p0[j] = ps[0]
            for k, n in enumerate(names): dl[n][j] = c3m.logit(ps[k + 1]) - c3m.logit(ps[0])
        P0[m] = p0; DL[m] = dl; del model
        if DEVICE == "cuda": torch.cuda.empty_cache()
    common = (P0["erm"] >= 0.1) & (P0["slim"] >= 0.1) & (P0["causal"] >= 0.1)

    res = {"n_common": int(common.sum()), "magnitude": mag, "pct_pixels": pct, "models": {}}
    for m in order:
        mm = {"signed_cace": {}, "mde": {}}
        for n in names:
            v = DL[m][n][common]; mn, lo, hi = boot(v)
            mm["signed_cace"][n] = dict(mean=mn, ci=[lo, hi], sig=bool(not (lo <= 0 <= hi)))
            mm["mde"][n] = float(2.8 * np.std(v) / np.sqrt(len(v)))   # ~MDE 80% power, alpha .05
        # sign dissociation: concept vs neutral
        conc = [mm["signed_cace"][n]["mean"] for n, k in EDITS if k == "concept"]
        neut = [mm["signed_cace"][n]["mean"] for n, k in EDITS if k == "neutral"]
        mm["concept_mean"] = float(np.mean(conc)); mm["neutral_mean"] = float(np.mean(neut))
        mm["sign_dissociation"] = bool(np.mean(conc) > 0 and np.mean(neut) < 0)  # concept+ vs neutral−
        res["models"][m] = mm
    json.dump(res, open(OUT, "w"), indent=2, default=float)

    # ---- figure: signed CaCE vs perturbation magnitude ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, m in zip(axes, order):
        for n, kind in EDITS:
            c = ("#2ca02c" if kind == "concept" else "#888")
            mk = ("o" if kind == "concept" else "x")
            ax.scatter(mag[n], res["models"][m]["signed_cace"][n]["mean"], c=c, marker=mk, s=70, zorder=3)
            ax.annotate(n.replace("_var", ""), (mag[n], res["models"][m]["signed_cace"][n]["mean"]),
                        (4, 3), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(NAME[m]); ax.set_xlabel("perturbation magnitude (mean |Δ|)")
    axes[0].set_ylabel("signed CaCE (logit)")
    fig.suptitle("Signed response vs perturbation magnitude — concept (green ●) vs neutral (grey ✕).\n"
                 "ERM: concept POSITIVE, neutral NEGATIVE (sign dissociation ⇒ not a magnitude effect). "
                 "Inv: all negative.", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig_magnitude.png"), bbox_inches="tight", dpi=140); plt.close(fig)

    print("magnitude (mean|Δ|) / %pix:")
    for n in names: print(f"  {n:13s} mag={mag[n]:.4f}  %pix={100*pct[n]:.1f}")
    print("\nsigned CaCE (common):")
    print(f"{'edit':13s}{'kind':9s}" + "".join(f"{NAME[m]:>10s}" for m in order))
    for n, kind in EDITS:
        print(f"{n:13s}{kind:9s}" + "".join(f"{res['models'][m]['signed_cace'][n]['mean']:>+10.2f}" for m in order))
    print("-" * 60)
    for m in order:
        mm = res["models"][m]
        print(f"{NAME[m]:6s} concept_mean={mm['concept_mean']:+.2f} neutral_mean={mm['neutral_mean']:+.2f} "
              f"sign_dissociation={'YES' if mm['sign_dissociation'] else 'no'}  "
              f"MDE(contrast@2)={mm['mde']['contrast@2']:.2f}")
    print(f"Saved -> {OUT} + fig_magnitude.png")


if __name__ == "__main__":
    main()
