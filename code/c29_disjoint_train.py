#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CausalXAI v14 — C29: huấn luyện lại sáu model trên SPLIT BA PHẦN TÁCH BỆNH NHÂN.

Vì sao cần. Trong thiết kế hiện tại, một fold validation duy nhất đóng ba vai: chọn checkpoint,
khớp calibration, và cung cấp ảnh cho Tier 2. Bootstrap theo bệnh nhân sửa được phần phụ thuộc
giữa các quan sát nhưng KHÔNG sửa được rò rỉ do chọn model. Mọi kết quả Tier-2 vì thế là chẩn đoán
hậu-chọn, không phải kiểm định độc lập.

Script này dựng ba phần TÁCH BỆNH NHÂN từ StratifiedGroupKFold(5, seed 42):

    fold 0            -> AUDIT   (không bao giờ chạm tới khi huấn luyện hay hiệu chỉnh)
    fold 1            -> SELECT  (early stopping, chọn checkpoint, khớp calibration)
    fold 2, 3, 4      -> TRAIN

rồi huấn luyện lại cả sáu model (ba EfficientNet-B3 và ba ResNet-50) với việc chọn checkpoint chỉ
dựa trên SELECT. Tier 2 sau đó chạy trên AUDIT (script C30).

Chi phí: tập huấn luyện giảm từ 315 xuống 237 ca dương, vì một phần năm bệnh nhân được giữ lại
cho AUDIT.

Out -> checkpoints/dj_*.pt và results/c29_disjoint_split.json
Chạy:  .venv/bin/python Causal_XAI_v3/code/c29_disjoint_train.py
"""
from __future__ import annotations

import json
import os
import sys
import time


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c8_backbone2_train as c8      # noqa: E402  (ERM_R50, Slim_R50)
import c3_cace_concept as c3m        # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
sys.path.insert(0, os.path.dirname(p1.__file__))
import phase3_invariance_v01 as p3   # noqa: E402  (train_causal, ISICEnvDataset)

BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints"); os.makedirs(CKPT, exist_ok=True)
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c29_disjoint_split.json")
META = "/mnt/data2/Toan/Causal_ML/data/data_ISIC2024/train-metadata.csv"
SEED, N_FOLDS, EPOCHS, NUM_WORKERS = 42, 5, 15, 0
FORCE = os.environ.get("C29_FORCE", "") == "1"   # huấn luyện lại dù đã có checkpoint
CAUSAL_BATCH = 24          # job causal cần gấp đôi bộ nhớ kích hoạt
# /dev/shm chỉ 64 MB trên máy này, không đủ cho worker của DataLoader -> nạp đơn tiến trình


def build_split3(seed=SEED, n_folds=N_FOLDS):
    """Ba phần tách bệnh nhân, ĐỌC TỪ MANIFEST CỐ ĐỊNH.

    Trước đây hàm này dựng lại phân hoạch mỗi lần gọi. StratifiedGroupKFold không ổn định giữa
    các phiên bản sklearn, nên cùng seed vẫn ra fold khác, và checkpoint huấn luyện dưới phân
    hoạch cũ không còn ghép được với cohort audit một cách kiểm chứng được. Nay phân hoạch được
    ghi ra results/frozen_split.json và mọi thứ đọc từ đó."""
    import frozen_split
    return frozen_split.load()


def _build_split3_legacy(seed=SEED, n_folds=N_FOLDS):
    """Bản dựng lại cũ, giữ để tham chiếu. Không dùng."""
    df = p1.load_isic_df()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = [va for _, va in sgkf.split(df, df["target"].values, df["patient_id"].values)]
    audit_i, select_i = folds[0], folds[1]
    train_i = np.setdiff1d(np.arange(len(df)), np.concatenate([audit_i, select_i]))
    df_tr = p1.undersample(df.iloc[train_i], p1.CFG["train_neg_ratio"], seed * 100)
    df_se = p1.undersample(df.iloc[select_i], p1.CFG["val_neg_ratio"], seed * 100 + 1)
    df_au = p1.undersample(df.iloc[audit_i], p1.CFG["val_neg_ratio"], seed * 100 + 2)
    # bất biến phải giữ: không bệnh nhân nào xuất hiện ở hai phần
    S = [set(d["patient_id"]) for d in (df_tr, df_se, df_au)]
    assert not (S[0] & S[1]) and not (S[0] & S[2]) and not (S[1] & S[2]), "rò rỉ bệnh nhân"
    return df_tr, df_se, df_au


def add_env(dfs):
    att = pd.read_csv(META, usecols=["isic_id", "attribution"]).fillna({"attribution": "unknown"})
    envs = sorted(att["attribution"].unique()); emap = {e: i for i, e in enumerate(envs)}
    amap = dict(zip(att["isic_id"], att["attribution"]))
    for d in dfs:
        d["attribution"] = d["isic_id"].map(amap).fillna("unknown")
        d["env"] = d["attribution"].map(emap)
    return len(envs)


def loaders(ds_cls, df_tr, df_se, batch):
    IMG, IMDIR = p1.CFG["img_size"], p1.ISIC_IMAGE_DIR
    tr = DataLoader(ds_cls(df_tr, IMDIR, IMG, train=True), batch, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    se = DataLoader(ds_cls(df_se, IMDIR, IMG, train=False), batch, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    return tr, se


def main():
    t0 = time.time()
    print(f"##### C29 disjoint-split training {time.strftime('%H:%M:%S')} dev={DEVICE} #####",
          flush=True)
    df_tr, df_se, df_au = build_split3()
    n_env = add_env([df_tr, df_se, df_au])
    for nm, d in [("TRAIN", df_tr), ("SELECT", df_se), ("AUDIT", df_au)]:
        print("  %-7s %6d ảnh  %4d bệnh nhân  %4d dương (%.2f%%)"
              % (nm, len(d), d["patient_id"].nunique(), int(d["target"].sum()),
                 100 * d["target"].mean()), flush=True)
    print(f"  môi trường (site): {n_env}   epochs={EPOCHS}", flush=True)

    json.dump({"seed": SEED, "n_folds": N_FOLDS, "epochs": EPOCHS,
               "note": ("Patient-disjoint three-way split. Checkpoint selection and calibration "
                        "use SELECT only; AUDIT is untouched until Tier 2."),
               "splits": {nm: {"n": int(len(d)), "n_patients": int(d["patient_id"].nunique()),
                               "n_pos": int(d["target"].sum())}
                          for nm, d in [("train", df_tr), ("select", df_se), ("audit", df_au)]}},
              open(OUT, "w"), indent=1)

    cfg = dict(p1.CFG); cfg["epochs"] = EPOCHS
    w = c8.class_weight(df_tr)
    cfg3 = dict(p3.CFG3); cfg3["epochs"] = EPOCHS

    jobs = [
        ("dj_erm_b3",    lambda: p1.build_model("erm", p1.CFG["z_dim"]),  "plain"),
        ("dj_slim_b3",   lambda: p1.build_model("slim", p1.CFG["z_dim"]), "plain"),
        ("dj_causal_b3", lambda: p1.build_model("slim", p1.CFG["z_dim"]), "causal"),
        ("dj_erm_r50",   lambda: c8.ERM_R50(),                            "plain"),
        ("dj_slim_r50",  lambda: c8.Slim_R50(p1.CFG["z_dim"]),            "plain"),
        ("dj_causal_r50", lambda: c8.Slim_R50(p1.CFG["z_dim"]),           "causal"),
    ]
    for name, make, kind in jobs:
        f = os.path.join(CKPT, name + ".pt")
        if os.path.exists(f) and not FORCE:
            print(f"[{name}] đã có, bỏ qua (dùng --force để huấn luyện lại)", flush=True); continue
        print(f"\n[{name}] huấn luyện ({kind}) ...", flush=True)
        p1.set_seed(SEED)
        if kind == "plain":
            tr, se = loaders(p1.ISICImgDataset, df_tr, df_se, cfg["batch_size"])
            m, auc = p1.train_one(make(), tr, se, cfg, w, name)
        else:
            # train_causal chạy hai lượt forward (ảnh gốc và ảnh có artifact), nên cần gấp đôi
            # bộ nhớ kích hoạt. GPU này đang bị tiến trình khác chiếm phần lớn, nên hạ batch.
            tr, se = loaders(p3.ISICEnvDataset, df_tr, df_se, CAUSAL_BATCH)
            m, auc = p3.train_causal(make(), tr, se, cfg3, w, name)
        torch.save(m.state_dict(), f)
        print(f"[{name}] lưu, select-AUC={auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        del m, tr, se
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nXONG sáu checkpoint -> {CKPT}   ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
