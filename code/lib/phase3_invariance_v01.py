#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CausalXAI v13 — PHASE 3: Lõi Invariance (V-REx + artifact-CF)
=============================================================
Đây là phần biến "causal" thành nội dung kiểm chứng được (design §1,§4).

Mô hình CAUSAL = CausalSlimNet (KIẾN TRÚC y hệt SLIM của Phase 1/2 — V-REx và
artifact-CF là LOSS lúc train, không phải kiến trúc), huấn luyện với:

    L = focal  +  λ_rex · Var_e[ risk_e ]  +  λ_cf · ||p(x) − p(artifact(x))||²

  - V-REx (Krueger 2021): phạt PHƯƠNG SAI của risk giữa 7 site thu nhận
    (attribution của ISIC 2024) ⇒ ép predictor bất biến qua môi trường ⇒ loại
    đặc trưng site-specific (spurious thu nhận). ĐÂY là tín hiệu nhân quả thật.
  - artifact-CF: ép prob bất biến khi thêm artifact thu nhận (vignette/patch/
    jitter lúc train). Eval ở Phase 2 dùng cả artifact KHÁC (hair/ruler) → không
    vòng tròn.

Đánh giá (so trực tiếp ERM & SLIM, KHÔNG tự chấm một mình):
  1. AUC: multi-seed × StratifiedGroupKFold GIỐNG HỆT Phase 1 ⇒ tái dùng preds
     ERM/SLIM đã lưu để chạy DeLong paired (cùng mẫu OOF).
  2. Causal XAI: nạp checkpoint CAUSAL (seed42 fold0, cùng split Phase 2) chạy
     lại Module B của Phase 2 ⇒ bảng 3 chiều ERM / SLIM / CAUSAL.

Vận hành: chạy nền; resume qua phase3_progress.json + preds_causal_*.npz +
checkpoints; mọi output trong XAI_NEW/.
"""
import os, sys, json, time, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_XAI_new_v01 as p1
import phase2_causal_xai_v01 as p2

CKPT_DIR, RES_DIR, LOG_DIR, DEVICE = p1.CKPT_DIR, p1.RES_DIR, p1.LOG_DIR, p1.DEVICE
ISIC_IMAGE_DIR = p1.ISIC_IMAGE_DIR
ISIC_META_CSV  = p1.ISIC_META_CSV
PROGRESS3 = os.path.join(RES_DIR, "phase3_progress.json")
SUMMARY3  = os.path.join(RES_DIR, "phase3_summary.json")
MEAN, STD = p2.MEAN, p2.STD

CFG3 = dict(
    seeds       = [42, 123, 456],
    n_folds     = 5,
    epochs      = 15,
    warmup      = 3,        # epoch đầu chỉ focal, sau đó bật V-REx + artifact-CF
    patience    = 4,
    lambda_rex  = 2.0,
    lambda_cf   = 0.5,
    grad_clip   = 1.0,
)


# ============================================================
# Data với nhãn môi trường (env = attribution / site)
# ============================================================
def load_isic_env():
    df = pd.read_csv(ISIC_META_CSV, low_memory=False,
                     usecols=["isic_id", "patient_id", "target", "attribution"])
    df["patient_id"] = df["patient_id"].fillna("unknown")
    df["target"] = df["target"].astype(int)
    df["attribution"] = df["attribution"].fillna("unknown")
    envs = sorted(df["attribution"].unique())
    emap = {e: i for i, e in enumerate(envs)}
    df["env"] = df["attribution"].map(emap)
    return df, emap


class ISICEnvDataset(p1.ISICImgDataset):
    def __init__(self, df, *a, **k):
        super().__init__(df, *a, **k)
        self.envs = torch.tensor(df.reset_index(drop=True)["env"].values, dtype=torch.long)
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.envs[idx]


# ============================================================
# Losses
# ============================================================
def focal_per_sample(logits, y, weight, gamma=2.0):
    ce = F.cross_entropy(logits, y, weight=weight, reduction="none")
    pt = torch.exp(-ce)
    return (1 - pt) ** gamma * ce


def vrex(per_sample_loss, env):
    """Trả (mean_risk, var_risk) — risk = loss trung bình mỗi môi trường."""
    risks = []
    for e in torch.unique(env):
        m = (env == e)
        if m.sum() > 0:
            risks.append(per_sample_loss[m].mean())
    risks = torch.stack(risks)
    var = risks.var(unbiased=False) if risks.numel() > 1 else risks.new_zeros(())
    return risks.mean(), var


def gpu_artifact(x_norm):
    """Thêm artifact thu nhận (vignette+jitter+patch) ngay trên batch GPU."""
    dev = x_norm.device
    mean = MEAN.to(dev); std = STD.to(dev)
    img = x_norm * std + mean
    B, C, H, W = img.shape
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev),
                            torch.linspace(-1, 1, W, device=dev), indexing="ij")
    r = torch.sqrt(xx**2 + yy**2).view(1, 1, H, W)
    strength = 0.3 + 0.5 * torch.rand(B, 1, 1, 1, device=dev)
    img = img * torch.clamp(1.0 - strength * torch.relu(r - 0.5), 0, 1)
    bright = (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.3
    contrast = 0.7 + 0.6 * torch.rand(B, 1, 1, 1, device=dev)
    img = (img - 0.5) * contrast + 0.5 + bright
    s = H // 6
    pm = torch.zeros(1, 1, H, W, device=dev); pm[:, :, :s, :s] = 1.0
    sel = (torch.rand(B, 1, 1, 1, device=dev) < 0.5).float()
    patch = torch.zeros_like(img); patch[:, 0, :s, :s] = 0.9
    patch[:, 1, :s, :s] = 0.2; patch[:, 2, :s, :s] = 0.2
    img = img * (1 - pm * sel) + patch * (pm * sel)
    return (img.clamp(0, 1) - mean) / std


# ============================================================
# Train / eval
# ============================================================
@torch.no_grad()
def evaluate_env(model, loader):
    model.eval()
    ys, yt = [], []
    for img, y, _ in loader:
        img = img.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            p = torch.softmax(model(img), 1)[:, 1]
        ys.append(p.float().cpu().numpy()); yt.append(y.numpy())
    return np.concatenate(yt), np.concatenate(ys)


def train_causal(model, tr_loader, va_loader, cfg, weight, tag):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=p1.CFG["lr"],
                            weight_decay=p1.CFG["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    weight = weight.to(DEVICE)
    use_amp = (DEVICE == "cuda")
    best_auc, best_state, no_imp = -1.0, None, 0

    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time()
        use_pen = ep >= cfg["warmup"]
        agg = dict(focal=0.0, vrex=0.0, cf=0.0, nb=0)
        for img, y, env in tr_loader:
            img = img.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True); env = env.to(DEVICE, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(img)
                fps = focal_per_sample(logits, y, weight)
                mean_r, var_r = vrex(fps, env)
                loss = mean_r
                cf = torch.zeros((), device=DEVICE)
                if use_pen:
                    x2 = gpu_artifact(img)
                    logits2 = model(x2)
                    cf = F.mse_loss(torch.softmax(logits2, 1), torch.softmax(logits, 1))
                    loss = mean_r + cfg["lambda_rex"] * var_r + cfg["lambda_cf"] * cf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            agg["focal"] += float(mean_r); agg["vrex"] += float(var_r)
            agg["cf"] += float(cf); agg["nb"] += 1
        sched.step()

        yt, ys = evaluate_env(model, va_loader)
        if not np.all(np.isfinite(ys)):
            print(f"      [{tag}] ep{ep+1}: NaN -> bỏ", flush=True); no_imp += 1
            if no_imp >= cfg["patience"]: break
            continue
        auc = roc_auc_score(yt, ys) if len(np.unique(yt)) > 1 else 0.5
        nb = max(agg["nb"], 1)
        print(f"      [{tag}] ep{ep+1:02d}/{cfg['epochs']} auc={auc:.4f} "
              f"focal={agg['focal']/nb:.3f} vrex={agg['vrex']/nb:.4f} "
              f"cf={agg['cf']/nb:.4f} pen={'on' if use_pen else 'off'} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if auc > best_auc:
            best_auc = auc; no_imp = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                print(f"      [{tag}] early stop @ ep{ep+1}", flush=True); break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


# ============================================================
# Run multi-seed × fold (mirror Phase 1 split để DeLong paired hợp lệ)
# ============================================================
def run(cfg):
    df, emap = load_isic_env()
    print(f"ISIC2024: {len(df)} ảnh, {df['target'].sum()} malignant, "
          f"{len(emap)} môi trường (site): {list(emap.values())}", flush=True)

    prog = json.load(open(PROGRESS3)) if os.path.exists(PROGRESS3) else {"units": {}}
    y = df["target"].values; groups = df["patient_id"].values

    for seed in cfg["seeds"]:
        sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
        for fold, (tr_idx, va_idx) in enumerate(sgkf.split(df, y, groups)):
            us = seed * 100 + fold
            df_tr = p1.undersample(df.iloc[tr_idx], p1.CFG["train_neg_ratio"], us)
            df_va = p1.undersample(df.iloc[va_idx], p1.CFG["val_neg_ratio"], us + 1)
            unit = f"causal_s{seed}_f{fold}"
            if unit in prog["units"] and os.path.exists(p1.preds_path(unit)):
                print(f"[skip] {unit}", flush=True); continue

            print(f"\n[run ] {unit}  train={len(df_tr)} (pos={int(df_tr['target'].sum())}) "
                  f"val={len(df_va)} (pos={int(df_va['target'].sum())})", flush=True)
            try:
                p1.set_seed(seed)
                tr_ds = ISICEnvDataset(df_tr, ISIC_IMAGE_DIR, p1.CFG["img_size"], train=True)
                va_ds = ISICEnvDataset(df_va, ISIC_IMAGE_DIR, p1.CFG["img_size"], train=False)
                tr_loader = DataLoader(tr_ds, p1.CFG["batch_size"], shuffle=True,
                                       num_workers=p1.CFG["num_workers"], pin_memory=True,
                                       drop_last=True)
                va_loader = DataLoader(va_ds, p1.CFG["batch_size"], shuffle=False,
                                       num_workers=p1.CFG["num_workers"], pin_memory=True)
                n_pos = int(df_tr["target"].sum()); n_neg = len(df_tr) - n_pos
                w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32)

                model = p1.build_model("slim", p1.CFG["z_dim"])
                t0 = time.time()
                model, _ = train_causal(model, tr_loader, va_loader, cfg, w, unit)
                yt, ys = evaluate_env(model, va_loader)
                if not np.all(np.isfinite(ys)):
                    print(f"[FAIL] {unit}: NaN", flush=True); continue
                mins = (time.time() - t0) / 60
                m = p1.metrics_from_scores(yt, ys)
                np.savez(p1.preds_path(unit), y_true=yt, y_score=ys,
                         seed=seed, fold=fold, model="causal")
                # lưu ckpt fold0 (cho XAI); seed42 fold0 = ckpt dùng cho bảng Phase 2
                if fold == 0:
                    torch.save(model.state_dict(),
                               os.path.join(CKPT_DIR, f"p3_causal_s{seed}_f0.pt"))
                    if seed == 42:
                        torch.save(model.state_dict(),
                                   os.path.join(CKPT_DIR, "p2_causal_s42.pt"))
                prog["units"][unit] = dict(model="causal", seed=seed, fold=fold,
                                           minutes=round(mins, 1), **m)
                json.dump(prog, open(PROGRESS3, "w"), indent=2)
                print(f"[done] {unit} AUC={m['auc']:.4f} sens@.5={m['sens05']:.3f} "
                      f"sens@Y={m['sens_y']:.3f} ({mins:.1f}min)", flush=True)
                del model; torch.cuda.empty_cache()
            except Exception as e:
                import traceback
                print(f"[ERROR] {unit}: {e}\n{traceback.format_exc()}", flush=True)
                continue

    aggregate(cfg)
    run_xai(cfg)


# ============================================================
# Aggregation: AUC mean±std + DeLong CAUSAL vs ERM/SLIM (preds Phase 1)
# ============================================================
def _pooled(model_tag, seed, n_folds):
    yts, yss = [], []
    for f in range(n_folds):
        pth = p1.preds_path(f"{model_tag}_s{seed}_f{f}")
        if not os.path.exists(pth):
            return None
        d = np.load(pth); yts.append(d["y_true"]); yss.append(d["y_score"])
    return np.concatenate(yts), np.concatenate(yss)


def aggregate(cfg):
    prog = json.load(open(PROGRESS3))
    rows = [v for v in prog["units"].values()]
    if not rows:
        print("Chưa có kết quả causal."); return
    sub = pd.DataFrame(rows)
    print("\n" + "=" * 64)
    print("PHASE 3 — AUC mô hình CAUSAL (mean±std qua fold×seed)")
    print("=" * 64)
    for col in ["auc", "sens05", "spec05", "sens_y", "spec_y"]:
        v = sub[col].astype(float).values
        print(f"   {col:8s} = {np.nanmean(v):.4f} ± {np.nanstd(v):.4f}")

    print("\nDeLong (pooled OOF mỗi seed) — CAUSAL vs ERM, CAUSAL vs SLIM:")
    out = {"per_seed": []}
    for seed in cfg["seeds"]:
        pc = _pooled("causal", seed, cfg["n_folds"])
        pe = _pooled("erm", seed, cfg["n_folds"])
        ps = _pooled("slim", seed, cfg["n_folds"])
        rec = {"seed": seed}
        if pc is None:
            continue
        rec["auc_causal"] = float(roc_auc_score(pc[0], pc[1]))
        lo, hi = p1.bootstrap_auc_ci(pc[0], pc[1], 2000, seed=seed)
        rec["ci_causal"] = [lo, hi]
        line = f"  seed {seed}: AUC_causal={rec['auc_causal']:.4f} CI=[{lo:.4f},{hi:.4f}]"
        if pe is not None and np.array_equal(pc[0], pe[0]):
            a_c, a_e, pval = p1.delong_test(pe[0], pe[1], pc[1])
            rec["vs_erm"] = dict(auc_erm=a_e, delta=a_c - a_e, p=pval)
            line += f" | vs ERM Δ={a_c-a_e:+.4f} p={pval:.3f}"
        if ps is not None and np.array_equal(pc[0], ps[0]):
            a_c, a_s, pval = p1.delong_test(ps[0], ps[1], pc[1])
            rec["vs_slim"] = dict(auc_slim=a_s, delta=a_c - a_s, p=pval)
            line += f" | vs SLIM Δ={a_c-a_s:+.4f} p={pval:.3f}"
        print(line, flush=True)
        out["per_seed"].append(rec)
    json.dump(out, open(SUMMARY3, "w"), indent=2)
    print(f"\nĐã lưu: {SUMMARY3}", flush=True)


# ============================================================
# Causal XAI: nạp ckpt CAUSAL, chạy Module B (Phase 2) -> bảng 3 chiều
# ============================================================
def run_xai(cfg):
    ckpt = os.path.join(CKPT_DIR, "p2_causal_s42.pt")
    if not os.path.exists(ckpt):
        print("\n[XAI] chưa có ckpt p2_causal_s42.pt, bỏ qua XAI.", flush=True); return
    print("\n" + "=" * 64)
    print("PHASE 3 — Causal XAI (Module B): ERM vs SLIM vs CAUSAL")
    print("=" * 64, flush=True)

    df_tr, df_va = p2.build_split(p2.CFG2)
    imgs01, labels = p2.load_eval_images(df_va, p2.CFG2["n_eval"], p2.CFG2["seed"])
    model = p1.build_model("slim", p1.CFG["z_dim"]).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    res = p2.evaluate_model(model, imgs01, labels, p2.CFG2)

    results = json.load(open(p2.RESULTS_FILE)) if os.path.exists(p2.RESULTS_FILE) else {}
    results["causal"] = res
    json.dump(results, open(p2.RESULTS_FILE, "w"), indent=2)

    order = [m for m in ["erm", "slim", "causal"] if m in results]
    print(f"{'metric':<22}" + "".join(f"{m:>20}" for m in order))
    for key, note in [("deletion_auc", "THẤP tốt (necessity)"),
                      ("insertion_auc", "CAO tốt (sufficiency)"),
                      ("artifact_delta_overall", "THẤP tốt (robust)")]:
        row = f"{key:<22}"
        for m in order:
            v, lo, hi = results[m][key]
            row += f"{v:.3f}[{lo:.3f},{hi:.3f}]".rjust(20)
        print(row + f"   <- {note}")
    print("\nArtifact-Δ theo loại:")
    for m in order:
        print(f"  {m:7s}: " + "  ".join(
            f"{k}={results[m]['artifact_delta_by_type'][k]:.4f}" for k in p2.ARTIFACTS))
    print(f"\nĐã cập nhật: {p2.RESULTS_FILE}", flush=True)


def main():
    sys.stdout = p1.Tee(os.path.join(LOG_DIR, "phase3_v01.log"))
    print(f"\n##### PHASE 3 START {time.strftime('%Y-%m-%d %H:%M:%S')} #####", flush=True)
    print(f"Device={DEVICE} seeds={CFG3['seeds']} λ_rex={CFG3['lambda_rex']} "
          f"λ_cf={CFG3['lambda_cf']} warmup={CFG3['warmup']}", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "--aggregate":
        aggregate(CFG3); run_xai(CFG3)
    else:
        run(CFG3)
    print(f"##### PHASE 3 END {time.strftime('%Y-%m-%d %H:%M:%S')} #####", flush=True)


if __name__ == "__main__":
    main()
