#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phân hoạch ba phần CỐ ĐỊNH, lưu theo isic_id.

Vì sao. `StratifiedGroupKFold` cho fold khác nhau giữa các phiên bản sklearn dù cùng seed. Trên
máy này, dựng lại phân hoạch cho train 553 bệnh nhân / 232 dương và audit 199 / 83, trong khi lần
huấn luyện gốc lưu 574 / 237 và 202 / 78. Hệ quả nặng hơn một chút lệch số: checkpoint được huấn
luyện dưới phân hoạch cũ, mà thành viên fold cũ KHÔNG được lưu ở đâu cả, nên không còn cách nào
kiểm chứng rằng cohort audit dựng lại thật sự tách bệnh nhân khỏi tập huấn luyện. Tuyên bố trung
tâm của audit phụ thuộc đúng vào tính tách đó.

Cách chữa. Dựng phân hoạch một lần, ghi ba danh sách isic_id vào `results/frozen_split.json`, rồi
mọi thứ về sau đọc từ file. Huấn luyện phải chạy lại dưới phân hoạch này thì checkpoint mới ghép
được với cohort audit một cách kiểm chứng được.

Dùng:
    import frozen_split
    df_tr, df_se, df_au = frozen_split.load()

Dựng lại (chỉ khi cố ý đổi phân hoạch, và khi đó PHẢI huấn luyện lại):
    python code/frozen_split.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m       # noqa: E402

p1 = c3m.p1
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
MANIFEST = os.path.join(RES, "frozen_split.json")
SEED, N_FOLDS = 42, 5


def _build():
    """Đúng logic gốc: fold0 = audit, fold1 = select, phần còn lại = train, rồi undersample."""
    df = p1.load_isic_df()
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = [va for _, va in sgkf.split(df, df["target"].values, df["patient_id"].values)]
    audit_i, select_i = folds[0], folds[1]
    train_i = np.setdiff1d(np.arange(len(df)), np.concatenate([audit_i, select_i]))
    df_tr = p1.undersample(df.iloc[train_i], p1.CFG["train_neg_ratio"], SEED * 100)
    df_se = p1.undersample(df.iloc[select_i], p1.CFG["val_neg_ratio"], SEED * 100 + 1)
    df_au = p1.undersample(df.iloc[audit_i], p1.CFG["val_neg_ratio"], SEED * 100 + 2)
    return df_tr, df_se, df_au


def _check(df_tr, df_se, df_au):
    S = [set(d["patient_id"].astype(str)) for d in (df_tr, df_se, df_au)]
    assert not (S[0] & S[1]), "rò rỉ bệnh nhân train/select"
    assert not (S[0] & S[2]), "rò rỉ bệnh nhân train/audit"
    assert not (S[1] & S[2]), "rò rỉ bệnh nhân select/audit"


def load(rebuild=False):
    """Trả (df_tr, df_se, df_au) dựng từ manifest. Bất biến tách bệnh nhân được khẳng định lại
    mỗi lần nạp, nên nếu manifest hỏng thì hỏng ồn ào chứ không âm thầm."""
    if rebuild or not os.path.exists(MANIFEST):
        df_tr, df_se, df_au = _build()
        _check(df_tr, df_se, df_au)
        os.makedirs(RES, exist_ok=True)
        json.dump({"seed": SEED, "n_folds": N_FOLDS,
                   "note": ("Frozen three-way patient-disjoint split, stored by isic_id. Fold "
                            "assignment from StratifiedGroupKFold is not stable across library "
                            "versions, so it is built once and read from this file thereafter. "
                            "Any checkpoint audited against it must have been trained under it."),
                   "counts": {k: {"n": int(len(d)), "n_patients": int(d["patient_id"].nunique()),
                                  "n_pos": int(d["target"].sum())}
                              for k, d in [("train", df_tr), ("select", df_se), ("audit", df_au)]},
                   "train": list(df_tr["isic_id"]), "select": list(df_se["isic_id"]),
                   "audit": list(df_au["isic_id"])},
                  open(MANIFEST, "w"), indent=1)
        print(f"[frozen_split] dựng mới và ghi {MANIFEST}", flush=True)

    m = json.load(open(MANIFEST))
    df = p1.load_isic_df()
    by_id = df.set_index("isic_id")
    out = []
    for part in ("train", "select", "audit"):
        ids = m[part]
        missing = [i for i in ids if i not in by_id.index]
        assert not missing, f"{len(missing)} isic_id trong manifest không có trong metadata"
        out.append(by_id.loc[ids].reset_index())
    _check(*out)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    df_tr, df_se, df_au = load(rebuild=a.rebuild)
    for nm, d in [("train", df_tr), ("select", df_se), ("audit", df_au)]:
        print("  %-7s n=%5d  patients=%4d  pos=%3d"
              % (nm, len(d), d["patient_id"].nunique(), int(d["target"].sum())))
    print("bất biến tách bệnh nhân: OK")
    print("manifest:", MANIFEST)


if __name__ == "__main__":
    main()
