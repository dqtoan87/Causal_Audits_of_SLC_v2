#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ba model EfficientNet-B3 được audit, phân giải ở MỘT chỗ duy nhất.

Vì sao. Trước đây mười script tự khai báo đường dẫn checkpoint. Chúng trỏ tới bộ huấn luyện dưới
một phân hoạch không còn dựng lại được, nên không kiểm chứng được rằng cohort đánh giá tách bệnh
nhân khỏi tập huấn luyện. Với tầng artifact điều đó là chí mạng vì tuyên bố ở đó là tuyên bố về
tổng quát hoá. Với tầng concept thì nhẹ hơn, vì không claim nào ở đó nói về tổng quát hoá, nhưng
vẫn còn một hệ quả bậc hai: nếu một ảnh đánh giá từng nằm trong tập huấn luyện thì logit nền của
nó bị thổi phồng do ghi nhớ, mà toàn bộ hồi quy co rút lấy logit nền làm biến độc lập.

Nay mọi phân tích dùng bộ huấn luyện dưới phân hoạch cố định (results/frozen_split.json), nên
tính tách bệnh nhân kiểm chứng được cho cả hai tầng.

Ghi đè bằng biến môi trường CAUSALXAI_CKPT_SET=legacy nếu cần đối chiếu với bộ cũ.
"""
from __future__ import annotations

import os

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)

_FROZEN = {"erm": os.path.join(BASE, "checkpoints", "dj_erm_b3.pt"),
           "slim": os.path.join(BASE, "checkpoints", "dj_slim_b3.pt"),
           "causal": os.path.join(BASE, "checkpoints", "dj_causal_b3.pt")}

_LEGACY = {"erm": os.path.join(HERE, "lib", "checkpoints", "p2_erm_s42.pt"),
           "slim": os.path.join(HERE, "lib", "checkpoints", "p2_slim_s42.pt"),
           "causal": os.path.join(HERE, "lib", "checkpoints", "p2_causal_s42.pt")}

CKPTS = _LEGACY if os.environ.get("CAUSALXAI_CKPT_SET") == "legacy" else _FROZEN
SET_NAME = "legacy" if CKPTS is _LEGACY else "frozen"


def check():
    missing = [k for k, v in CKPTS.items() if not os.path.exists(v)]
    assert not missing, f"thiếu checkpoint: {missing}"
    return CKPTS


if __name__ == "__main__":
    print(f"bộ checkpoint: {SET_NAME}")
    for k, v in check().items():
        print(f"  {k:7s} {v}  ({os.path.getsize(v)/1e6:.0f} MB)")
