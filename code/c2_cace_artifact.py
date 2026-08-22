#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalXAI v14 — PHASE C2: CaCE ARTIFACT (negative control, editing-only counterfactual).

Đo tác động NHÂN QUẢ của việc CHÈN một artifact thu nhận (hair/ruler/patch/vignette/jitter) lên
xác suất ác tính của classifier — bằng can thiệp editing-only (do(U=present) trên CHÍNH ảnh đó,
giữ nguyên mọi thứ khác). Artifact KHÔNG phải nguyên nhân của bệnh ⇒ model nhân-quả-trung-thực
phải cho **CaCE_U ≈ 0**. CaCE_U lớn = model bám shortcut. Xem `CausalXAI_v14_DESIGN.md` §5.2(A).

ESTIMAND (DESIGN §5.3):
    CaCE_U(f) = E_x[ f(add_U(x)) − f(x) ]                       # prob-scale, CÓ DẤU
    CaCE_logit(f) = E_x[ logit f(add_U(x)) − logit f(x) ]       # logit-scale (CHỐNG calibration,
                                                                 #  bài học v13 B4)

★ TÍNH HONEST (chống metric-coupling đã giết v13):
  - artifact-CF lúc TRAIN CAUSAL dùng {vignette, jitter, patch} ⇒ 3 cái này COUPLED (in-distribution,
    KHÔNG dùng làm bằng chứng sạch cho CAUSAL).
  - {hair, ruler} = HELD-OUT (CAUSAL chưa từng train để bất biến) ⇒ bằng chứng SẠCH.
  - Với ERM/SLIM: mọi artifact đều non-tautological (chưa từng có artifact-CF).
  Mỗi artifact gắn nhãn coupled/held-out; cross-model claim CHỈ dựa held-out.

THIẾT KẾ:
  - Edit MỘT LẦN per ảnh (rng deterministic theo index) → DÙNG CHUNG cho cả 3 model ⇒ can thiệp
    giống hệt, loại nhiễu rng khỏi so sánh model.
  - Đo trên ảnh malignant thật có p_full ≥ floor (quyết định lâm sàng quan trọng) + báo cáo ALL.
  - Falsification: identity-edit phải cho CaCE = 0 (sanity pipeline). placebo.
  - Bootstrap CI; "faithful" = CI(CaCE_logit) chứa 0 (không đổi prediction = tốt cho negative control).

Nạp 3 ckpt cache ở code/lib/checkpoints (p2_{erm,slim,causal}_s42.pt). Không train lại.
Out -> Causal_XAI_v3/results/tier2_cace_artifact.json. Chạy nhanh (~1-3 phút), cần GPU.

Chạy:
  .venv/bin/python Causal_XAI_v3/code/c2_cace_artifact.py
  .venv/bin/python Causal_XAI_v3/code/c2_cace_artifact.py --n-eval 240
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# --- nạp hạ tầng huấn luyện (vendored: code/lib) -----------------------------
# v3: the upstream training modules are vendored under code/lib together with the three
# primary EfficientNet-B3 checkpoints, so this tree is self-contained apart from the raw
# image data. Override with the CAUSALXAI_LIB environment variable if they are moved.
XAI_OLD = os.environ.get(
    "CAUSALXAI_LIB", os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
sys.path.insert(0, XAI_OLD)
import train_XAI_new_v01 as p1       # noqa: E402  (DEVICE, build_model, CKPT_DIR, CFG, ...)
import phase2_causal_xai_v01 as p2   # noqa: E402  (ARTIFACTS, _to_norm, build_split, ...)

DEVICE = p1.DEVICE
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
OUT_JSON = os.path.join(OUT_DIR, "tier2_cace_artifact.json")

P_FLOOR = 0.10
SEED = 42
import audited_models                # noqa: E402  (phân giải checkpoint ở một chỗ)
CKPTS = audited_models.CKPTS         # đường dẫn TUYỆT ĐỐI tới bộ đã cố định

# Nhãn coupling (artifact-CF train CAUSAL = vignette+jitter+patch; xem phase3 gpu_artifact)
COUPLING = {"hair_light": "held-out", "hair": "held-out", "hair_heavy": "held-out",
            "ruler": "held-out",
            "patch": "train-coupled", "vignette": "train-coupled", "jitter": "train-coupled"}


# --- HELD-OUT artifacts: REALISTIC (mảnh + alpha-blend bán trong suốt), tránh phá hủy ảnh -------
# (Bản thô của phase2 vẽ nét đen ĐẶC đè lên tổn thương -> off-manifold, CaCE phản ánh "phá ảnh"
#  chứ không phải shortcut. Bản này mô phỏng lông/thước thật: mảnh 1px, bán trong suốt.)
def _hair(x, rng, n, alpha, dark=0.10):
    """n sợi lông cong mảnh 1px, alpha-blend (bán trong suốt) — như dermoscopy thật."""
    x = x.clone(); _, H, W = x.shape
    for _ in range(n):
        x0, y0 = rng.randint(0, W), rng.randint(0, H)
        ang = rng.uniform(0, 2 * np.pi)
        length = rng.randint(W // 3, W)
        curv = rng.uniform(-0.0025, 0.0025)
        cs, sn = np.cos(ang), np.sin(ang)
        pc, ps = np.cos(ang + np.pi / 2), np.sin(ang + np.pi / 2)
        for t in range(length):
            px = int(x0 + t * cs + curv * t * t * pc)
            py = int(y0 + t * sn + curv * t * t * ps)
            if 0 <= px < W and 0 <= py < H:
                x[:, py, px] = alpha * dark + (1 - alpha) * x[:, py, px]
    return x


# Dose-response: 3 mức lông -> kiểm |CaCE| đơn điệu (dấu hiệu nhân quả kinh điển)
def art_hair_light(x, rng): return _hair(x, rng, n=6, alpha=0.4)
def art_hair(x, rng):       return _hair(x, rng, n=14, alpha=0.6)
def art_hair_heavy(x, rng): return _hair(x, rng, n=24, alpha=0.75)


def art_ruler(x, rng, alpha=0.7, dark=0.0):
    """Thước mảnh bán trong suốt ở RÌA DƯỚI (vùng da nền, không đè trung tâm tổn thương)."""
    x = x.clone(); _, H, W = x.shape
    y = rng.randint(int(0.75 * H), max(int(0.75 * H) + 1, H - 3))
    x[:, y:y + 2, :] = alpha * dark + (1 - alpha) * x[:, y:y + 2, :]
    for tx in range(0, W, 18):                      # vạch chia mảnh
        yt0, yt1 = max(0, y - 4), y + 4
        x[:, yt0:yt1, tx:tx + 1] = alpha * dark + (1 - alpha) * x[:, yt0:yt1, tx:tx + 1]
    return x


def art_jitter(x, rng):
    """Brightness/contrast jitter — KHỚP gpu_artifact lúc train CAUSAL (coupled artifact)."""
    bright = (rng.rand() - 0.5) * 0.3
    contrast = 0.7 + 0.6 * rng.rand()
    return torch.clamp((x - 0.5) * contrast + 0.5 + bright, 0, 1)


def art_identity(x, rng):
    """Falsification: không đổi gì -> CaCE phải = 0."""
    return x.clone()


# Tập can thiệp: held-out REALISTIC (sạch, có dose-response lông) + coupled KHỚP-TRAIN + identity
ARTIFACTS = {
    "hair_light": art_hair_light, "hair": art_hair, "hair_heavy": art_hair_heavy,  # HELD-OUT dose
    "ruler": art_ruler,                                   # HELD-OUT
    "patch": p2.art_patch, "vignette": p2.art_vignette,   # coupled (khớp train CAUSAL)
    "jitter": art_jitter,                                  # coupled
    "identity": art_identity,                              # sanity (CaCE≈0)
}
HAIR_DOSE = ["hair_light", "hair", "hair_heavy"]          # để kiểm đơn điệu


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


@torch.no_grad()
def model_prob(model, x01_batch):
    """x01_batch: [B,3,H,W] in [0,1]. Trả prob malignant [B]."""
    xn = p2._to_norm(x01_batch).to(DEVICE)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
        return torch.softmax(model(xn), 1)[:, 1].float().cpu().numpy()


def precompute_edits(imgs01, art_names, seed, thr=0.02):
    """Edit MỘT LẦN per ảnh (rng deterministic theo index) -> dùng chung mọi model.
    Trả (edits, change_frac): edits=list[dict art->tensor]; change_frac[a]=% pixel đổi >thr (minh
    bạch độ phá hủy/off-manifold, DESIGN §5.4)."""
    edits = []
    cf = {a: [] for a in art_names}
    for i, x01 in enumerate(imgs01):
        rng = np.random.RandomState(seed + i)   # cố định per ảnh
        d = {}
        for a in art_names:
            e = ARTIFACTS[a](x01, rng)
            d[a] = e
            cf[a].append(float((e - x01).abs().amax(0).gt(thr).float().mean()))
        edits.append(d)
    change_frac = {a: float(np.mean(cf[a])) for a in art_names}
    return edits, change_frac


def eval_model(model, imgs01, labels, edits, art_names, cfg):
    model.eval()
    # batch [orig, edit_a1, edit_a2, ...] per ảnh -> 1 forward/ảnh
    p0_all = np.empty(len(imgs01))
    dp = {a: [] for a in art_names}      # Δp có dấu, theo ảnh giữ lại
    dl = {a: [] for a in art_names}      # Δlogit có dấu
    keep_mask = []
    for i, (x01, lab) in enumerate(zip(imgs01, labels)):
        batch = torch.stack([x01] + [edits[i][a] for a in art_names])
        ps = model_prob(model, batch)
        p0 = ps[0]
        p0_all[i] = p0
        keep = (lab == 1) and (p0 >= cfg["p_floor"])
        keep_mask.append(keep)
        if not keep:
            continue
        for j, a in enumerate(art_names):
            pa = ps[j + 1]
            dp[a].append(pa - p0)
            dl[a].append(logit(pa) - logit(p0))
    keep_mask = np.array(keep_mask)
    return p0_all, dp, dl, keep_mask


def summarize(dp, dl, art_names, bn, seed):
    """Per-artifact CaCE + CI + faithful flag."""
    out = {}
    for a in art_names:
        v_p, v_l = np.array(dp[a]), np.array(dl[a])
        p_mean, p_lo, p_hi = p2.boot_mean_ci(v_p, bn, seed)
        l_mean, l_lo, l_hi = p2.boot_mean_ci(v_l, bn, seed)
        ap_mean, ap_lo, ap_hi = p2.boot_mean_ci(np.abs(v_p), bn, seed)
        faithful = bool(l_lo <= 0 <= l_hi)        # CI logit chứa 0 = không đổi prediction = tốt
        out[a] = dict(
            coupling=COUPLING.get(a, "n/a"),
            cace_dp=p_mean, cace_dp_ci=[p_lo, p_hi],
            cace_logit=l_mean, cace_logit_ci=[l_lo, l_hi],
            abs_dp=ap_mean, abs_dp_ci=[ap_lo, ap_hi],
            faithful=faithful, n=int(len(v_p)),
        )
    return out


def aggregate_group(dp, dl, names, bn, seed):
    """Gộp |Δlogit| & |Δp| trên một nhóm artifact (vd held-out) -> chỉ số tổng hợp."""
    allp = np.concatenate([np.abs(dp[a]) for a in names]) if names else np.array([])
    alll = np.concatenate([np.abs(dl[a]) for a in names]) if names else np.array([])
    mp = p2.boot_mean_ci(allp, bn, seed)
    ml = p2.boot_mean_ci(alll, bn, seed)
    return dict(abs_dp=mp[0], abs_dp_ci=[mp[1], mp[2]],
                abs_logit=ml[0], abs_logit_ci=[ml[1], ml[2]], n=int(len(allp)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--boot-n", type=int, default=2000)
    ap.add_argument("--p-floor", type=float, default=P_FLOOR)
    ap.add_argument("--out", default=OUT_JSON)
    args = ap.parse_args()
    cfg = dict(seed=SEED, p_floor=args.p_floor, boot_n=args.boot_n)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"##### PHASE C2 — CaCE artifact {time.strftime('%H:%M:%S')} (dev={DEVICE}) #####", flush=True)
    df_tr, df_va = p2.build_split({"seed": SEED, "n_folds": 5})
    imgs01, labels = p2.load_eval_images(df_va, args.n_eval, SEED)
    n_mal = int(sum(labels))
    print(f"eval: {len(imgs01)} ảnh (malignant thật={n_mal})", flush=True)

    art_names = list(ARTIFACTS.keys())
    held = [a for a in art_names if COUPLING.get(a) == "held-out"]
    coup = [a for a in art_names if COUPLING.get(a) == "train-coupled"]
    edits, change_frac = precompute_edits(imgs01, art_names, SEED)
    print(f"can thiệp: held-out={held} | coupled={coup} | sanity=['identity']", flush=True)
    print("  %pixel đổi: " + " ".join(f"{a}={100*change_frac[a]:.1f}%" for a in art_names), flush=True)

    results = {}
    for name, path in CKPTS.items():
        if not os.path.exists(path):
            print(f"[skip] thiếu {path}"); continue
        model = p1.build_model("erm" if name == "erm" else "slim", p1.CFG["z_dim"]).to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        t0 = time.time()
        p0_all, dp, dl, keep = eval_model(model, imgs01, labels, edits, art_names, cfg)
        per_art = summarize(dp, dl, art_names, args.boot_n, SEED)
        results[name] = dict(
            n_kept=int(keep.sum()),
            pred_mal_prob=float(np.mean(p0_all[[l == 1 for l in labels]])),  # calibration exposure
            per_artifact=per_art,
            heldout=aggregate_group(dp, dl, held, args.boot_n, SEED),
            coupled=aggregate_group(dp, dl, coup, args.boot_n, SEED),
            identity_cace_logit=per_art["identity"]["cace_logit"],
            hair_dose_monotonic=bool(
                abs(per_art["hair_light"]["cace_logit"]) <= abs(per_art["hair"]["cace_logit"])
                <= abs(per_art["hair_heavy"]["cace_logit"])),
        )
        print(f"[{name}] xong ({time.time()-t0:.0f}s, n_kept={results[name]['n_kept']}, "
              f"p_mal={results[name]['pred_mal_prob']:.3f})", flush=True)
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    out = dict(
        phase="C2_cace_artifact",
        estimand="CaCE_U(f)=E_x[f(add_U(x))-f(x)]; signed prob & logit scale; editing-only CF.",
        note_coupling="artifact-CF train CAUSAL={vignette,jitter,patch} (coupled); "
                      "{hair,ruler}=held-out (clean). ERM/SLIM: all non-tautological.",
        eval=dict(n_images=len(imgs01), n_malignant=n_mal, p_floor=args.p_floor,
                  boot_n=args.boot_n, heldout=held, coupled=coup, change_frac=change_frac),
        ckpt_set=__import__('audited_models').SET_NAME, models=results,
    )
    json.dump(out, open(args.out, "w"), indent=2, default=float)

    # ---- bảng tóm tắt ----
    order = [m for m in ["erm", "slim", "causal"] if m in results]
    print("\n" + "=" * 96)
    print("CaCE ARTIFACT (negative control): faithful = |CaCE|≈0 (CI logit chứa 0). "
          "★ bằng chứng SẠCH = HELD-OUT (hair/ruler)")
    print("=" * 96)
    print(f"{'artifact':12s}{'coupling':14s}" + "".join(f"{m:>22s}" for m in order))
    print(f"{'':26s}" + "".join(f"{'CaCE_logit [CI] faith':>22s}" for m in order))
    print("-" * 96)
    for a in art_names:
        row = f"{a:12s}{COUPLING.get(a,'n/a'):14s}"
        for m in order:
            r = results[m]["per_artifact"][a]
            tag = "✓" if r["faithful"] else "✗"
            row += f"{r['cace_logit']:+.2f}[{r['cace_logit_ci'][0]:+.1f},{r['cace_logit_ci'][1]:+.1f}]{tag:>2s}".rjust(22)
        print(row)
    print("-" * 96)
    print(f"{'HELD-OUT |Δlogit|':26s}" + "".join(
        f"{results[m]['heldout']['abs_logit']:.3f}[{results[m]['heldout']['abs_logit_ci'][0]:.2f},{results[m]['heldout']['abs_logit_ci'][1]:.2f}]".rjust(22)
        for m in order))
    print(f"{'coupled |Δlogit|':26s}" + "".join(
        f"{results[m]['coupled']['abs_logit']:.3f}[{results[m]['coupled']['abs_logit_ci'][0]:.2f},{results[m]['coupled']['abs_logit_ci'][1]:.2f}]".rjust(22)
        for m in order))
    print(f"{'pred_mal_prob (calib)':26s}" + "".join(f"{results[m]['pred_mal_prob']:.3f}".rjust(22) for m in order))
    print(f"{'identity (sanity≈0)':26s}" + "".join(f"{results[m]['identity_cace_logit']:+.4f}".rjust(22) for m in order))
    print(f"{'hair dose monotonic':26s}" + "".join(
        f"{'YES' if results[m]['hair_dose_monotonic'] else 'no':>22s}" for m in order))
    print("=" * 96)
    print("Đọc: CaCE_logit > 0 = artifact ĐẨY prediction về ác (shortcut). faith ✓ = CI chứa 0 "
          "(model bỏ qua artifact = nhân-quả-trung-thực). So sánh model CHỈ dùng dòng HELD-OUT; "
          "dòng coupled chỉ in-distribution (CAUSAL đã train né). identity phải ≈0.")
    print(f"\nĐã lưu: {args.out}")


if __name__ == "__main__":
    main()
