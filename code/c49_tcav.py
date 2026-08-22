#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C49 — TCAV trên CÙNG sáu checkpoint, CÙNG hai khái niệm, CÙNG nhóm ảnh kiểm định.

Vì sao. Đóng góp thứ ba của bài nói ước lượng tử placebo KHÔNG định danh được thành phần khái
niệm trong lớp can thiệp đã dựng. Một người phản biện có quyền hỏi: vậy phương pháp quy gán
khái niệm đang dùng phổ biến trả về gì trên đúng dữ liệu này? Nếu chúng trả về câu trả lời tự
tin ở chỗ khung của bài báo động là không định danh được, thì đó chính là điều khung này thêm
vào — và phải đo, không được lập luận suông.

TCAV (Kim et al. 2018) đo đạo hàm có hướng của logit theo một hướng khái niệm trong không gian
kích hoạt. Nó KHÔNG can thiệp lên khái niệm; nó đọc độ nhạy kết hợp dọc một hướng tuyến tính.

Giao thức, giữ đúng bản gốc:
  1. CAV: hồi quy logistic tuyến tính tách kích hoạt của ảnh ĐÃ SỬA khái niệm khỏi ảnh GỐC,
     ở lớp đặc trưng gộp. Vector pháp tuyến v_C là hướng khái niệm. Như bản gốc, huấn luyện
     N_CAV lần trên các nửa mẫu ngẫu nhiên khác nhau để có PHÂN BỐ điểm TCAV chứ không phải
     một điểm duy nhất.
  2. Đạo hàm có hướng: S_i = ∇_a logit_malignant(a_i) · v_C tại kích hoạt của ảnh gốc i.
  3. Điểm TCAV = tỉ lệ ảnh có S_i > 0.
  4. Kiểm định ý nghĩa: N_RAND CAV NGẪU NHIÊN từ nhãn hoán vị, cùng cách huấn luyện. Khái niệm
     có ý nghĩa khi t-test HAI MẪU giữa N_CAV điểm khái niệm và N_RAND điểm ngẫu nhiên bác bỏ
     ở mức 0,05 — đúng phép kiểm bản gốc dùng.

     Một lưu ý về null này, phải báo cáo chứ không giấu: gradient ở lớp đặc trưng gần như cùng
     hướng trên mọi ảnh, nên một hướng ngẫu nhiên cho tích vô hướng cùng dấu ở gần hết ảnh, và
     điểm TCAV ngẫu nhiên dồn về 0 hoặc 1 chứ không tụ quanh 0,5. Trung bình null vẫn quanh
     0,5 nhưng phương sai rất lớn, nên phép kiểm này có LỰC THẤP ở đây.

Điểm cần nhấn khi đọc kết quả: dùng CHÍNH các phép sửa khái niệm của bài để dựng CAV là cách
đối xử THUẬN LỢI NHẤT có thể cho TCAV — hướng khái niệm được lấy đúng từ can thiệp mà bài thực
hiện, nên không thể trách CAV là không đại diện cho khái niệm.

Out -> results/c49_tcav.json
Chạy:  python Causal_XAI_v3/code/c49_tcav.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c3_cace_concept as c3m        # noqa: E402  (make_edit, soft_mask, DOSES)
import c30_disjoint_audit as c30     # noqa: E402  (make_model, probs)
import audit_cohort                  # noqa: E402

p1, p2, DEVICE = c3m.p1, c3m.p2, c3m.DEVICE
BASE = os.path.dirname(HERE)
CKPT = os.path.join(BASE, "checkpoints")
RES = os.path.join(BASE, "results")
OUT = os.path.join(RES, "c49_tcav.json")

SEED, N_CAV, N_RAND = 42, 20, 20
CONCEPTS = ["contrast", "color_var"]
DOSE = c3m.DOSES[-1]          # liều đỉnh, đúng liều tầng khái niệm của bài dùng
MODELS = [("erm", "dj_erm_b3.pt", "EfficientNet-B3 ERM"),
          ("slim", "dj_slim_b3.pt", "EfficientNet-B3 Slim"),
          ("slim", "dj_causal_b3.pt", "EfficientNet-B3 Inv"),
          ("erm_r50", "dj_erm_r50.pt", "ResNet-50 ERM"),
          ("slim_r50", "dj_slim_r50.pt", "ResNet-50 Slim"),
          ("slim_r50", "dj_causal_r50.pt", "ResNet-50 Inv")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def backbone_of(model, kind):
    """Lớp đặc trưng gộp: nơi TCAV chuẩn đặt CAV."""
    if kind in ("erm", "slim"):
        return model.backbone
    return model.backbone


def head_of(model, kind):
    """Phần từ đặc trưng gộp tới logit."""
    if kind == "erm":
        return model.head
    if kind == "slim":
        return lambda a: model.disease_head(model.proj_c(a))
    if kind == "erm_r50":
        return model.head
    return lambda a: model.disease_head(model.proj_c(a))


@torch.no_grad()
def activations(model, kind, batch):
    bb = backbone_of(model, kind)
    out = []
    for i in range(0, len(batch), 32):
        x = p2._to_norm(torch.stack(batch[i:i + 32])).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(DEVICE == "cuda")):
            out.append(bb(x).float().cpu().numpy())
    return np.concatenate(out)


def grads(model, kind, batch):
    """∇_a logit_malignant tại kích hoạt của từng ảnh gốc."""
    bb, hd = backbone_of(model, kind), head_of(model, kind)
    out = []
    for i in range(0, len(batch), 16):
        x = p2._to_norm(torch.stack(batch[i:i + 16])).to(DEVICE)
        with torch.no_grad():
            a = bb(x).float()
        a = a.detach().requires_grad_(True)
        z = hd(a).float()
        s = (z[:, 1] - z[:, 0]).sum()          # logit lớp ác tính
        g, = torch.autograd.grad(s, a)
        out.append(g.cpu().numpy())
    return np.concatenate(out)


def main():
    t0 = time.time()
    rng = np.random.RandomState(SEED)
    imgs01, labels, pid, ids = audit_cohort.load()
    log(f"nhóm kiểm định: {len(imgs01)} ảnh, {len(set(pid))} bệnh nhân")

    # ảnh đã sửa khái niệm ở liều đỉnh — chính các phép sửa của Section III-D
    edited = {}
    for c in CONCEPTS:
        eb = []
        for x in imgs01:
            w, skin, lm = c3m.soft_mask(x)
            eb.append(c3m.make_edit(x, c, DOSE, w, skin, lm))
        edited[c] = eb
        log(f"  dựng {len(eb)} ảnh đã sửa cho khái niệm {c}")

    res = {"n_images": len(imgs01), "n_patients": int(len(set(pid))), "dose": float(DOSE),
           "n_random_cavs": N_RAND, "seed": SEED,
           "note": ("TCAV on the same checkpoints, concepts and audit images as the concept tier. "
                    "The CAV is trained to separate activations of concept-edited images from "
                    "activations of the originals, so the concept direction comes from the very "
                    "intervention the paper performs. This is the most favourable construction "
                    "available to TCAV. Significance is against N_RAND random CAVs from permuted "
                    "labels, the standard TCAV control."),
           "models": {}}

    for kind, fn, label in MODELS:
        path = os.path.join(CKPT, fn)
        if not os.path.exists(path):
            log(f"BỎ QUA {label}: thiếu {fn}"); continue
        m = c30.make_model(kind).to(DEVICE)
        m.load_state_dict(torch.load(path, map_location=DEVICE)); m.eval()
        A0 = activations(m, kind, imgs01)
        G0 = grads(m, kind, imgs01)
        entry = {"label": label, "feature_dim": int(A0.shape[1]), "concepts": {}}
        for c in CONCEPTS:
            A1 = activations(m, kind, edited[c])
            X = np.concatenate([A1, A0]); y = np.concatenate([np.ones(len(A1)), np.zeros(len(A0))])
            n = len(A0)
            def one(labels_, half):
                cl = LogisticRegression(max_iter=2000, C=1.0).fit(X[half], labels_[half])
                vv = cl.coef_[0]; vv = vv / (np.linalg.norm(vv) + 1e-12)
                return float(((G0 @ vv) > 0).mean()), float(cl.score(X, labels_))
            scores, accs = [], []
            for r in range(N_CAV):
                h = rng.permutation(len(X))[: len(X) // 2]
                sc, ac = one(y, h); scores.append(sc); accs.append(ac)
            null = []
            for r in range(N_RAND):
                h = rng.permutation(len(X))[: len(X) // 2]
                null.append(one(rng.permutation(y), h)[0])
            scores, null, accs = np.array(scores), np.array(null), np.array(accs)
            t, p = stats.ttest_ind(scores, null, equal_var=False)
            entry["concepts"][c] = {
                "cav_accuracy_mean": float(accs.mean()),
                "tcav_mean": float(scores.mean()),
                "tcav_sd": float(scores.std(ddof=1)),
                "tcav_range": [float(scores.min()), float(scores.max())],
                "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
                "null_range": [float(null.min()), float(null.max())],
                "p_vs_random": float(p),
                "significant": bool(p < 0.05),
                "n_cav": N_CAV, "n_random": N_RAND}
            e = entry["concepts"][c]
            log(f"  {label:22s} {c:10s} CAV acc={accs.mean():.3f}  "
                f"TCAV={scores.mean():.3f}±{scores.std(ddof=1):.3f} "
                f"[{scores.min():.2f},{scores.max():.2f}]  "
                f"null={null.mean():.3f}±{null.std(ddof=1):.3f}  p={p:.2e}  "
                f"có ý nghĩa={e['significant']}")
        res["models"][fn.replace(".pt", "")] = entry
        del m
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    allc = [e for v in res["models"].values() for e in v["concepts"].values()]
    res["summary"] = {"n_cells": len(allc),
                      "n_significant": sum(e["significant"] for e in allc),
                      "tcav_mean_range": [min(e["tcav_mean"] for e in allc),
                                          max(e["tcav_mean"] for e in allc)],
                      "cav_accuracy_range": [min(e["cav_accuracy_mean"] for e in allc),
                                             max(e["cav_accuracy_mean"] for e in allc)],
                      "null_sd_range": [min(e["null_sd"] for e in allc),
                                        max(e["null_sd"] for e in allc)]}
    res["runtime_sec"] = round(time.time() - t0, 1)
    json.dump(res, open(OUT, "w"), indent=1)
    log(f"saved -> {OUT}  ({res['summary']['n_significant']}/{res['summary']['n_cells']} ô có ý nghĩa)")


if __name__ == "__main__":
    main()
