#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Danh sách ảnh audit CỐ ĐỊNH.

Vì sao. Trước đây mỗi script tự dựng lại phân hoạch bằng StratifiedGroupKFold rồi lấy mẫu ảnh.
Cách đó phụ thuộc phiên bản thư viện: cùng seed nhưng khác phiên bản sklearn hoặc pandas thì ra
fold khác. Trên máy này, dựng lại cho 183 ảnh / 100 bệnh nhân / 83 ca ác tính, trong khi lần chạy
gốc lưu 178 / 90 / 78. Kết luận không đổi nhưng con số thì lệch tới 13%, và đó là thứ không chấp
nhận được với một bài báo.

Cách chữa. Dựng một lần, ghi danh sách `isic_id` theo đúng thứ tự cùng nhãn và mã bệnh nhân vào
`results/audit_cohort.json`, rồi từ đó về sau mọi script nạp thẳng từ file. Ảnh được đọc theo id
nên không còn phụ thuộc bất kỳ bước lấy mẫu nào.

Dùng:
    import audit_cohort
    imgs, labels, pid, ids = audit_cohort.load()

Dựng lại từ đầu (chỉ làm khi cố ý đổi cohort):
    python code/audit_cohort.py --rebuild
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torchvision.transforms as T
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m       # noqa: E402
import frozen_split                 # noqa: E402

p1, p2 = c3m.p1, c3m.p2
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
MANIFEST = os.path.join(RES, "audit_cohort.json")

SEED, N_EVAL = 42, 200


def _read(ids):
    """Đọc ảnh theo id, đúng thứ tự, đúng phép biến đổi mà phần còn lại của đường ống dùng."""
    to_t = T.ToTensor()
    size = (p1.CFG["img_size"], p1.CFG["img_size"])
    out = []
    for i in ids:
        img = Image.open(os.path.join(p2.ISIC_IMAGE_DIR, f"{i}.jpg")).convert("RGB")
        out.append(to_t(p1.center_crop_resize(img, size)))
    return out


def build(seed=SEED, n_eval=N_EVAL):
    """Dựng cohort từ phân hoạch ba phần, sao lại đúng logic của p2.load_eval_images để lấy được
    thứ tự id thật sự, rồi trả về mọi thứ cần cho manifest."""
    _, _, df_au = frozen_split.load()
    df_au = df_au.reset_index(drop=True)
    k = n_eval // 2
    pos = df_au[df_au["target"] == 1].sample(n=min(k, int((df_au["target"] == 1).sum())),
                                             random_state=seed)
    neg = df_au[df_au["target"] == 0].sample(n=min(k, int((df_au["target"] == 0).sum())),
                                             random_state=seed)
    sel = np.r_[pos.index.values, neg.index.values]
    sub = df_au.loc[sel].reset_index(drop=True)
    return (list(sub["isic_id"]), [int(t) for t in sub["target"]],
            list(sub["patient_id"].astype(str)))


def load(rebuild=False):
    """Trả (imgs, labels, pid, ids). Đọc manifest nếu có, nếu chưa thì dựng rồi ghi."""
    if rebuild or not os.path.exists(MANIFEST):
        ids, labels, pid = build()
        os.makedirs(RES, exist_ok=True)
        json.dump({"n_images": len(ids), "n_patients": len(set(pid)),
                   "n_malignant": int(sum(labels)), "seed": SEED, "n_eval": N_EVAL,
                   "note": ("Frozen audit cohort. Built once from the three-way patient-disjoint "
                            "split and stored by isic_id so that no later analysis depends on "
                            "reconstructing the fold assignment, which is not stable across "
                            "library versions."),
                   "isic_id": ids, "target": labels, "patient_id": pid},
                  open(MANIFEST, "w"), indent=1)
        print(f"[audit_cohort] dựng mới và ghi {MANIFEST}", flush=True)
    m = json.load(open(MANIFEST))
    ids = m["isic_id"]; labels = np.array(m["target"]); pid = np.array(m["patient_id"])
    assert len(ids) == len(labels) == len(pid)
    return _read(ids), labels, pid, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    imgs, labels, pid, ids = load(rebuild=a.rebuild)
    print(f"cohort: {len(ids)} ảnh / {len(set(pid))} bệnh nhân / {int(labels.sum())} ác tính")
    print(f"manifest: {MANIFEST}")
    print("5 id đầu:", ids[:5])


if __name__ == "__main__":
    main()
