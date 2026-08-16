#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CausalXAI v14 — C37: Tier-1 bỏ acquisition site khỏi tập điều chỉnh.

Vì sao cần. Trong tập Z, acquisition site là biến duy nhất nằm cả trên đường đo lường: nó vào qua
Z → X và Z → U, không chỉ qua Z → C và Z → Y. Nếu một trung tâm ảnh hưởng nhãn được ghi thông qua
quá trình xác định bệnh, điều kiện hoá theo nó sẽ chặn mất một phần đường mà ta muốn giữ. Câu hỏi
này không giải quyết được bằng lập luận, nên chạy cả hai phiên bản và báo cáo cả hai.

Chạy lại AIPW cross-fit theo bệnh nhân cho ba proxy confirmatory, một lần với Z đầy đủ và một lần
bỏ `attribution`, mọi thứ khác giữ nguyên.

Out -> results/c37_no_site.json
Chạy:  .venv/bin/python Causal_XAI_v3/code/c37_no_site_sensitivity.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import c1_tier1_ate as c1

OUT = os.path.join(os.path.dirname(HERE), "results", "c37_no_site.json")
CONF = ["C_color", "D_size", "contrast_LB"]   # ba proxy confirmatory


def main():
    t0 = time.time()
    print(f"##### C37 Tier-1 bỏ acquisition site {time.strftime('%H:%M:%S')} #####", flush=True)
    args = argparse.Namespace(data=c1.DATA_CSV, out=OUT, n_boot=c1.N_BOOT, max_rows=0,
                              refute=False, split="tertile")
    df = pd.read_csv(c1.DATA_CSV, low_memory=False)
    df["ITA_stratum"] = c1.compute_ita(df)
    base_cat = list(c1.CAT_CONF)
    res = {"note": ("Tier-1 adjusted proxy risk difference for the three confirmatory proxies, with "
                    "and without acquisition site in the adjustment set. Everything else identical: "
                    "same AIPW estimator, same patient-grouped cross-fitting, same trimming, same "
                    "patient cluster bootstrap."), "concepts": {}}
    Z = {}
    for lab, cats in [("with_site", base_cat),
                      ("without_site", [c for c in base_cat if c != "attribution"])]:
        c1.CAT_CONF = cats
        Z[lab] = c1.build_Z(df)
        print("  %-13s Z dims = %d" % (lab, Z[lab].shape[1]), flush=True)
    c1.CAT_CONF = base_cat

    print("\n%-13s %12s %12s %7s %6s | %12s %12s %7s %6s"
          % ("proxy", "rd đủ Z", "RR", "E", "sig", "rd bỏ site", "RR", "E", "sig"), flush=True)
    CMAP = {c["name"]: c for c in c1.CONCEPTS}
    for name in CONF:
        row = {}
        for lab in ("with_site", "without_site"):
            r = c1.run_concept(df, Z[lab], CMAP[name], args)
            row[lab] = {"ate": r["ate"], "ci": r["ate_ci95"], "rr": r["rr"],
                        "e_value": r["e_value"], "significant": r["significant"]}
        res["concepts"][name] = row
        a, b = row["with_site"], row["without_site"]
        print("%-13s %+12.3e %12.2f %7.2f %6s | %+12.3e %12.2f %7.2f %6s"
              % (name, a["ate"], a["rr"], a["e_value"], a["significant"],
                 b["ate"], b["rr"], b["e_value"], b["significant"]), flush=True)
    res["summary"] = {
        "sign_unchanged": all((v["with_site"]["ate"] > 0) == (v["without_site"]["ate"] > 0)
                              for v in res["concepts"].values()),
        "significance_unchanged": all(v["with_site"]["significant"] == v["without_site"]["significant"]
                                      for v in res["concepts"].values()),
        "max_abs_rel_change": max(abs(v["without_site"]["ate"] - v["with_site"]["ate"])
                                  / abs(v["with_site"]["ate"]) for v in res["concepts"].values())}
    print("\nsummary:", json.dumps(res["summary"]), flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
    print("saved -> %s  (%.0fs)" % (OUT, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
