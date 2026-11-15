#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Check the numbers paper3.md and supplementary3.md print against the results this tree produced.

Every check does two things. It recomputes the quantity from this tree's results/ and compares it with
the value the manuscript prints, at the precision printed. And, where a check names the text it
backs, it confirms that this exact text is present in the manuscript or the supplement, so a
number edited in the paper without a matching result (or the reverse) is caught.

Tables 4 (main text) and S29 (supplement) are parsed from the markdown and compared cell by cell.

Reads only this tree's results/, the two manuscripts, and two read-only external sources when they
are present: the training log of the audited checkpoints and the ISIC-2024 metadata file.
Exit status is non-zero if any check fails.

Usage:  python code/verify_paper3.py            (prints one line per check and a summary)
"""
from __future__ import annotations

import json
import numpy as np
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
RES = os.path.join(BASE, "results")
PAPER = os.path.join(BASE, "paper", "paper3.md")
SUPP = os.path.join(BASE, "paper", "supplementary3.md")
TRAIN_LOG = os.path.join(os.path.dirname(BASE), "logs", "c29_frozen.log")
ISIC_META = "/mnt/data2/Toan/Causal_ML/data/data_ISIC2024/train-metadata.csv"
B = 13.815509557963773

TEXT = {"paper": open(PAPER, encoding="utf-8").read(), "supp": open(SUPP, encoding="utf-8").read()}
rows = []


def load(name):
    p = os.path.join(RES, name + ".json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def check(claim, got, want, tol=0.0, floor=False, ceil=False, text=None, where="paper"):
    """Compare a recomputed value with the printed one and, optionally, confirm the printed text.

    want: scalar, (lo, hi) pair, or string. floor/ceil: got >= want / got <= want.
    text: a literal (or list of literals) that must occur in the named document(s).
    """
    if got is None:
        rows.append(("MISSING", claim, "-", str(want)))
        return
    if isinstance(want, tuple):
        ok = abs(got[0] - want[0]) <= tol and abs(got[1] - want[1]) <= tol
        g, w = "%.4g to %.4g" % tuple(got), "%.4g to %.4g" % want
    elif isinstance(want, str):
        ok, g, w = str(got) == want, str(got), want
    elif floor:
        ok, g, w = got >= want, "%.6g" % got, ">= %.6g" % want
    elif ceil:
        ok, g, w = got <= want, "%.6g" % got, "<= %.6g" % want
    else:
        ok, g, w = abs(got - want) <= tol, "%.6g" % got, "%.6g" % want
    if ok and text is not None:
        docs = [where] if isinstance(where, str) else where
        for t in ([text] if isinstance(text, str) else text):
            if not any(t in TEXT[d] for d in docs):
                ok = False
                w += "  [text not found: %r]" % t
    rows.append(("ok" if ok else "MISMATCH", claim, g, w))


def num(s):
    return float(s.replace("−", "-").replace("+", ""))


def rng(vals):
    return (min(vals), max(vals))


# ================================================================ cohort and data
d = load("audit_cohort")
if d:
    check("audit cohort: 183 images", d["n_images"], 183, text="183 images from 100 patients")
    check("audit cohort: 100 patients", d["n_patients"], 100)
    check("audit cohort: 83 malignant", d["n_malignant"], 83, text="83 malignant and 100 benign")

d = load("padufes_split")
if d:
    c = d["counts"]
    check("PAD-UFES-20 split 1,367 / 475 / 456",
          "%d/%d/%d" % (c["train"]["n"], c["select"]["n"], c["audit"]["n"]), "1367/475/456",
          text="1,367 training, 475 selection and 456 audit images")
    check("PAD-UFES-20 audit fold 278 patients", c["audit"]["n_patients"], 278,
          text="456 audit images from 278 patients")
    check("PAD-UFES-20 audit-fold malignant rate 47.6%",
          100 * c["audit"]["n_pos"] / c["audit"]["n"], 47.6, 0.05, text="malignant rate of 47.6%")

d = load("frozen_split")
if d:
    check("training fold holds 232 malignant cases", d["counts"]["train"]["n_pos"], 232,
          text="315 to 232 malignant cases", where="supp")

if os.path.exists(ISIC_META):
    try:
        import pandas as pd
        m = pd.read_csv(ISIC_META, usecols=["target", "iddx_1", "iddx_2", "patient_id"], low_memory=False)
        check("ISIC-2024: 401,059 tiles", len(m), 401059, text="401,059")
        check("ISIC-2024: 1,042 patients", m["patient_id"].nunique(), 1042, text="1,042 patients")
        check("ISIC-2024: 393 malignant", int(m["target"].sum()), 393, text="393 malignant labels")
        check("ISIC-2024: prevalence 0.098%", 100 * m["target"].mean(), 0.098, 5e-4, text="0.098%")
        check("ISIC-2024: 1,068 histopathology-level diagnoses", int(m["iddx_2"].notna().sum()), 1068,
              text="1,068")
        check("ISIC-2024: 114 indeterminate", int((m["iddx_1"] == "Indeterminate").sum()), 114,
              text="114 indeterminate", where="supp")
    except Exception as exc:  # pragma: no cover
        rows.append(("MISSING", "ISIC-2024 metadata could not be read: %s" % exc, "-", "-"))

if os.path.exists(TRAIN_LOG):
    lg = open(TRAIN_LOG, encoding="utf-8").read()
    auc = {k: float(v) for k, v in re.findall(r"\[(dj_\w+)\] (?:saved|lưu), select-AUC=([\d.]+)", lg)}
    if auc:
        check("selection AUROC, EfficientNet-B3 trio",
              "%.2f/%.2f/%.2f" % (auc["dj_erm_b3"], auc["dj_slim_b3"], auc["dj_causal_b3"]),
              "0.90/0.89/0.86", text="0.90, 0.89 and 0.86")
        check("selection AUROC, ResNet-50 trio",
              "%.2f/%.2f/%.2f" % (auc["dj_erm_r50"], auc["dj_slim_r50"], auc["dj_causal_r50"]),
              "0.87/0.88/0.89", text="0.87, 0.88 and 0.89", where="supp")
    secs = [int(x) for x in re.findall(r"\((\d+)s\)", lg)]
    if secs:
        check("training the six checkpoints takes 1.9 hours", max(secs) / 3600, 1.9, 0.05,
              text="1.9 hours")

# ================================================================ artifact tier
d = load("c30_disjoint_audit")
cells = {}
if d:
    for bk in d["models"]:
        for mk in d["models"][bk]:
            for sg in ("all", "malignant", "benign"):
                cells[(bk, mk, sg)] = d["models"][bk][mk][sg]
    cl = list(cells.values())
    check("eighteen artifact cells", len(cl), 18)
    check("every cell exceeds jointly", sum(c["joint_verdict"] == "exceedance" for c in cl), 18)
    check("seven cells do not exceed on the signed component",
          sum(c["signed_verdict"] != "exceedance" for c in cl), 7,
          text="Seven of the eighteen cells do not exceed on the signed component")
    check("two cells within the signed band", sum(c["signed_verdict"] == "equivalence" for c in cl), 2)
    check("primary magnitudes 0.085 to 0.769", rng([c["mean"] for c in cl]), (0.085, 0.769), 5e-4)
    au = [d["models"][b][m]["audit_auroc"] for b in d["models"] for m in d["models"][b]]
    check("audit-cohort AUROC 0.864 to 0.893", rng(au), (0.864, 0.893), 5e-4, text="0.864 to 0.893")
    check("hair alters 2.4% of pixels", 100 * d["change_frac"]["hair"], 2.4, 0.05, text="2.4% of pixels")
    check("identity edit alters no pixels", d["change_frac"]["identity"], 0.0)
    b3 = d["models"]["b3"]
    for mk, p0s, p1s in (("ERM", "0.733", "0.070"), ("Slim", "0.859", "0.170"), ("Inv", "0.610", "0.520")):
        p0 = b3[mk]["p0_mean_malignant"]
        p1 = p0 + b3[mk]["hair_dose_dp"][1]
        check("%s malignant score under hair %s to %s" % (mk, p0s, p1s), (p0, p1),
              (float(p0s), float(p1s)), 5e-4, text="from %s to %s" % (p0s, p1s))
    # frontier at tolerance 0.20
    def exceeds_at(c, t):
        mag = c["ci"][0] > t
        sg = c["signed_ci"][0] > t or c["signed_ci"][1] < -t
        return mag or sg
    check("six cells still exceed at tolerance 0.20", sum(exceeds_at(c, 0.20) for c in cl), 6,
          text="at 0.20, six of the eighteen cells still exceed")
    # Table 4, cell by cell
    tbl = TEXT["paper"][TEXT["paper"].index("**TABLE 4."):]
    tbl = tbl[:tbl.index("\n\n", tbl.index("| --- "))]
    bk = mk = None
    nrow = nbad = 0
    vmap = {"exceeds": "exceedance", "within": "equivalence", "indet.": "indeterminate"}
    for line in tbl.splitlines():
        if not line.startswith("| ") or line.startswith("| Backbone") or line.startswith("| ---"):
            continue
        f = [x.strip() for x in line.strip("|").split("|")]
        bk = {"EfficientNet-B3": "b3", "ResNet-50": "r50"}.get(f[0], bk)
        mk = f[1] or mk
        c = cells[(bk, mk, f[2])]
        m1 = re.match(r"([−+\d.]+) \[([−+\d.]+), ([−+\d.]+)\]", f[3])
        m2 = re.match(r"([−+\d.]+) \[([−+\d.]+), ([−+\d.]+)\]", f[4])
        printed = [num(x) for x in m1.groups() + m2.groups()]
        actual = [c["mean"]] + c["ci"] + [c["signed_mean"]] + c["signed_ci"]
        ok = all(abs(a - p) <= 5e-4 + 1e-9 for a, p in zip(actual, printed))
        ok &= (vmap[f[5]], vmap[f[6]], vmap[f[7]]) == (c["verdict"], c["signed_verdict"], c["joint_verdict"])
        nrow += 1
        nbad += not ok
        if not ok:
            rows.append(("MISMATCH", "Table 4 row %s %s %s" % (bk, mk, f[2]), str(actual), line))
    check("Table 4: all 18 rows match c30 (values, intervals, verdicts)", nrow - nbad, 18)

d = load("c40_simultaneous")
if d:
    lo = [v["ci_simultaneous"][0] for v in d["disjoint_18"]["cells"].values()]
    check("secondary 18-magnitude band: critical value 2.94", d["disjoint_18"]["crit_z"], 2.94, 5e-3, text="critical value of 2.94")
    check("secondary 18-magnitude band: frontier 0.067", min(lo), 0.067, floor=True,
          text="frontier of 0.067")
    check("simultaneous replicates 20,000", d["boot"], 20000)
    p9 = d["primary_9"]
    check("nine cached-prediction cells: critical value 2.66", p9["crit_z"], 2.66, 5e-3,
          text="critical value is 2.66", where="supp")
    check("nine cached-prediction cells: bound 0.122",
          min(v["ci_simultaneous"][0] for v in p9["cells"].values()), 0.122, 5e-4,
          text="0.122", where="supp")
d = load("c57_stratified_band")
if d:
    check("stratified band: critical value 3.10", d["crit_36"], 3.10, 5e-3, text="critical value of 3.10")
    check("stratified band: frontier 0.065", d["frontier_36"], 0.065, 5e-4, text="frontier of 0.065")
    check("stratified band: frontier 0.0651 (supp)", d["frontier_36"], 0.0651, 5e-5, text="0.0651", where="supp")
    check("stratified band: strata 44 and 56", tuple(d["n_patients_strata"]), (44, 56), 0, text="the 44 patients", where="supp")
    check("stratified band: all 18 still exceed", d["magnitude_exceed_36"], 18)
    dr = d.get("design_ratio")
    if dr:
        check("observed malignant share 45%", 100 * dr["malignant_share_observed"], 45, 0.5, text="observed share is 45%", where="supp")
        check("plain bootstrap share 34 to 57%", tuple(100 * x for x in dr["malignant_share_plain_2.5_97.5"]), (34, 57), 0.5, text="34 to 57%", where="supp")
        check("stratified share 36 to 55%", tuple(100 * x for x in dr["malignant_share_stratified_2.5_97.5"]), (36, 55), 0.5, text="36 to 55%", where="supp")
        check("19 patients hold both labels, 57 images", (dr["n_patients_with_both_labels"], dr["n_images_of_patients_with_both_labels"]), (19, 57), 0, text="Nineteen patients hold both labels, with 57 images", where="supp")
d = load("c58_rebuild_trace")
if d:
    mix = lambda k: d[k]["mixture"]["magnitude"]
    check("trace: bright ruler moves the cell by 0.0003",
          round(mix("L2_L3_this_pipeline") - mix("L4_this_pipeline_bright_ruler"), 4), 0.0003, 5e-5, text="0.0003", where="supp")
    check("trace: single blend moves the cell by 0.0007",
          round(mix("L2_L3_this_pipeline") - mix("L5_this_pipeline_single_blend_hair"), 4), 0.0007, 5e-5, text="0.0007", where="supp")
    check("trace: reimplementation inference on same edits gives 0.047",
          mix("L6_reimplementation_inference_on_this_pipeline_edits"), 0.047, 5e-4, text="give 0.047", where="supp")
    check("trace: inference codes agree to 0.003", d["L6_unedited"]["reimpl_inference_vs_this_pipeline_mean_abs"],
          0.003, 5e-4, text="to 0.003 on average", where="supp")
    r7 = d["L7_reimplementation_rerun_vs_its_cache"]
    check("trace: rerun vs cache mean 0.016", r7["mean_abs"], 0.016, 5e-4, text="by 0.016 on average", where="supp")
    check("trace: rerun vs cache max 0.72", r7["max_abs"], 0.72, 5e-3, text="up to 0.72", where="supp")
    check("trace: 4.9% of images above 0.05", 100 * r7["share_over_0.05"], 4.9, 0.05, text="4.9% of images", where="supp")
    check("trace: 1.04 blends per touched pixel", d["blends_per_touched_pixel_heavy_hair"], 1.04, 5e-3, text="1.04 times", where="supp")
d = load("c59_rebuild_generator")
if d:
    m = d["models"]["r50_inv_s42"]
    check("rerun: overlays reproduce the cached pixel fractions", all(d["pixel_fraction_matches_cache"].values()), True)
    check("rerun: cell 0.041", m["rerun"]["benign"]["magnitude"], 0.041, 5e-4, text="0.041", where="supp")
    check("rerun: interval [0.030, 0.052]", tuple(m["rerun"]["benign"]["ci"]), (0.030, 0.052), 5e-4, text="[0.030, 0.052]", where="supp")
    check("rerun: cell indeterminate", m["rerun"]["benign"]["joint_verdict"], "indeterminate")
    check("cache: cell within tolerance", m["cache"]["benign"]["joint_verdict"], "equivalence")
    same = sum(v["rerun"][g]["joint_verdict"] == v["cache"][g]["joint_verdict"]
               for v in d["models"].values() for g in ("all", "malignant", "benign"))
    check("rerun keeps 17 of 18 cached decisions", same, 17, text="keeps the other 17 decisions", where="supp")
    mm = [v["unedited_rerun_vs_cache"]["mean_abs"] for v in d["models"].values()]
    check("rerun vs cache, unedited, 0.016 to 0.033 on average", (min(mm), max(mm)), (0.016, 0.033), 5e-4,
          text="0.016 to 0.033 on average", where="supp")
    check("rerun is deterministic", max(v["unedited_rerun_vs_rerun_max_abs"] for v in d["models"].values()), 0.0, 0.0)
d = load("c60_design_coverage")
if d:
    check("benign redraw: joint coverage 97.0%", 100 * d["joint_coverage"], 97.0, 0.05, text="in 97.0% of draws")
    check("benign redraw: MC se 0.8 points", 100 * d["joint_coverage_mc_se"], 0.8, 0.05, text="0.8 percentage points", where="supp")
    check("benign redraw: component coverage min 99.0%", 100 * d["component_coverage_min"], 99.0, 0.05, text="fewer than 99.0% of draws", where="supp")
    check("benign redraw: crit 3.01 to 3.26", tuple(d["crit_range"]), (3.01, 3.26), 5e-3, text="from 3.01 to 3.26", where="supp")
    check("benign redraw: crit median 3.10", d["crit_median"], 3.10, 5e-3, text="median of 3.10", where="supp")
    check("benign redraw: frontier median 0.0697", d["frontier_median"], 0.0697, 5e-5, text="median of 0.0697", where="supp")
    check("benign redraw: frontier range 0.068 to 0.072", tuple(d["frontier_2.5_97.5"]), (0.068, 0.072), 5e-4, text="0.068 to 0.072", where="supp")
    check("benign redraw: frontier min 0.0668", d["frontier_min"], 0.0668, 5e-5, text="minimum of 0.0668", where="supp")
    check("benign redraw: every tolerance below 0.066", d["frontier_min"], 0.066, floor=True, text="every tolerance below 0.066 in every draw")
    check("benign redraw: all 18 exceed in every draw", d["share_draws_all_18_exceed"], 1.0, 0.0)
    check("benign redraw: frame 4,150 benign from 198 patients", (d["n_benign_frame"], d["n_benign_frame_patients"]), (4150, 198), 0, text="4,150 benign images from 198 patients", where="supp")
    check("benign redraw: 500 draws", d["draws"], 500, text="redraws the 100 benign images 500 times")
import math as _m
_Phi = lambda z: 0.5 * (1 + _m.erf(z / _m.sqrt(2)))
check("signed exceedance size at eps/se = 0.25 is 0.032", _Phi(-1.959964) + _Phi(-1.959964 - 0.5), 0.032, 5e-4,
      text="0.032 at *ε*/*se* = 0.25", where="supp")
d = load("c62_evaluation")
if d:
    reg, con = load("c62_preregistration"), load("c62_construction")
    pr, comp = d["primary"], d["primary"]["components"]
    check("route test: 78 locked images from 54 patients", (reg["locked_set"]["n_images"], reg["locked_set"]["n_patients"]), (78, 54), 0,
          text="78 malignant images of the selection fold, from 54 patients")
    a4 = [con["summary"][t]["n_all_three_g1"] for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2")]
    check("route test: all four controls pass G1 on 74 to 78 images", (min(a4), max(a4)), (74, 78), 0, text="74 to 78 of the 78 images")
    ex = sorted(v["est"] for v in comp.values() if v.get("decision_sim") == "exceeds")
    check("route test: disagreements 1.99 to 5.27 logits", (ex[0], ex[-1]), (1.99, 5.27), 5e-3, text="1.99 to 5.27 logits")
    check("route test: abstract 2.0 to 5.3 logits", (ex[0], ex[-1]), (2.0, 5.3), 0.05, text="2.0 to 5.3 logits")
    check("route test: 3 disagree / 2 agree / 7 indeterminate",
          "%d/%d/%d" % (pr["summary"]["routes disagree"], pr["summary"]["routes agree"], pr["summary"]["indeterminate"]), "3/2/7",
          text="In three of the twelve cells")
    check("route test: no direction pair exceeds", pr["summary"]["direction_pair_exceeds"], 0,
          text="None did under the band, although 14 of the 24 direction differences")
    dm = sum(1 for c, v in comp.items() if ("D_rad4-hue4" in c or "D_rad16-hue16" in c) and (v["ci"][0] > 0 or v["ci"][1] < 0))
    check("route test: 14 of 24 direction differences exclude zero marginally", dm, 14, text="14 of the 24 direction differences")
    sa = [comp["%s|S_%s" % (c, k)]["est"] for c, v in pr["cells"].items() if v["class"] == "routes agree" for k in ("rad4", "rad16", "hue4", "hue16")]
    check("route test: agreeing route sums 0.00 to +0.20", (round(min(sa), 2) + 0.0, round(max(sa), 2)), (0.0, 0.20), 1e-9,
          text="route sums from 0.00 to +0.20 logits")
    check("route test: simultaneous critical value 3.30", pr["crit_simultaneous"], 3.30, 5e-3, text="critical value of the band was 3.30", where="supp")
    for cell, txt in (("slim|contrast@0", "+1.99 [+0.51, +3.47]"), ("erm|contrast@2", "+4.63 [+1.92, +7.34]"),
                      ("slim|color_var@2", "+5.27 [+1.84, +8.70]")):
        q = comp[cell + "|D_rad4-rad16"]
        nums = [float(x) for x in txt.replace("[", "").replace("]", "").replace(",", "").split()]
        check("route test: Table S31 %s estimate" % cell, q["est"], nums[0], 5e-3, text=txt, where="supp")
        check("route test: Table 5 %s interval" % cell, tuple(q["sim"]), (nums[1], nums[2]), 5e-3)
    ps = []
    rs = []
    for m in ("erm", "slim"):
        for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2"):
            S = [comp["%s|%s|S_%s" % (m, t, k)]["est"] for k in ("rad4", "rad16", "hue4", "hue16")]
            PC = [comp["%s|%s|PC_%s" % (m, t, k)]["est"] for k in ("hue", "flip", "jitter", "noise")]
            rs.append(max(S) - min(S)); ps.append(max(PC) - min(PC))
    check("route test: route sums spread 0.9 to 5.3", (min(rs), max(rs)), (0.9, 5.3), 0.05, text="spread by 0.9 to 5.3 logits")
    check("route test: placebos spread 2.0 to 4.5", (min(ps), max(ps)), (2.0, 4.5), 0.05, text="spread by 2.0 to 4.5 logits", where="supp")
    sall = d["secondary_all_g1"]["summary"]
    check("route test: without the floor 2 disagree, 0 agree, 10 indeterminate",
          "%d/%d/%d" % (sall["routes disagree"], sall["routes agree"], sall["indeterminate"]), "2/0/10",
          text="two cells disagree, none agrees and ten are indeterminate", where="supp")
    ni = [d["secondary_independent"]["targets"][t]["n_images"] for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2")]
    check("route test: independent subsets 16, 1, 0, 0", "/".join(map(str, ni)), "16/1/0/0", text="16, 1, 0 and 0 images", where="supp")
    cs = [comp["causal|%s|D_%s" % (t, q)]["est"] for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
          for q in ("rad4-hue4", "rad16-hue16", "rad4-rad16", "hue4-hue16")]
    check("route test: Inv differences at most 0.42 logits", max(abs(x) for x in cs), 0.42, 5e-3,
          text="no difference exceeds 0.42 logits")
    isum = [comp["causal|%s|S_%s" % (t, k)]["est"] for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
            for k in ("rad4", "rad16", "hue4", "hue16")]
    check("route test: Inv route sums -0.15 to +0.54", (min(isum), max(isum)), (-0.15, 0.54), 5e-3,
          text="from −0.15 to +0.54 logits")
    ipc = [comp["causal|%s|PC_%s" % (t, k)]["est"] for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
           for k in ("hue", "flip", "jitter", "noise")]
    check("route test: Inv placebo corrections reach +0.65", max(ipc), 0.65, 5e-3, text="placebo corrections reach +0.65")
    nf = [d["secondary_all_g1"]["cells"]["causal|%s" % t]["class"] for t in ("contrast@0", "color_var@0")]
    check("route test: without the floor both Inv cells indeterminate", "/".join(nf), "indeterminate/indeterminate",
          text="two cells disagree, none agrees and ten are indeterminate")
t34 = load("c34_ctrl_vs_ctrl")
if t34:
    med = {}
    for q in t34["pairs"]:
        med.setdefault(q["model"], []).append(abs(q["diff"]))
    got = "%.1f/%.1f/%.1f" % tuple(float(np.median(med[m])) for m in ("ERM", "Slim", "Inv"))
    check("diagnostic (T): per-model median disagreement 8.7, 9.9, 1.2", got, "8.7/9.9/1.2",
          text="median disagreements of 8.7, 9.9 and 1.2 logits for ERM, Slim and Inv")
    got2 = "%.2f/%.2f/%.2f" % tuple(float(np.median(med[m])) for m in ("ERM", "Slim", "Inv"))
    check("S6: per-model medians 8.66, 9.92, 1.18", got2, "8.66/9.92/1.18",
          text="8.66 logits for ERM, 9.92 for Slim and 1.18 for Inv", where="supp")
rb = load("c65_route_robustness")
if rb:
    check("robustness: stored responses reproduce the registered band", rb["reproduction_max_abs_difference"], 0.0, 1e-9)
    s = rb["decisive"]["slim|contrast@0|D_rad4-rad16"]
    check("robustness: borderline lower bound 0.510", s["sim_lower_unrounded"], 0.510, 5e-4, text="lower bound of 0.510")
    check("robustness: borderline under 10 of 20 seeds", s["seed_exceeds"], 10, text="under 10 of 20 other seeds")
    check("robustness: borderline in 32 of 54 leave-one-out", (s["lopo_exceeds"], s["lopo_n"]), (32, 54), 0,
          text="in 32 of 54 leave-one-patient-out subsets")
    rob = [rb["decisive"][k] for k in ("erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16")]
    check("robustness: two rejections hold for 20 of 20 seeds and all 54 subsets",
          "/".join("%d-%d" % (q["seed_exceeds"], q["lopo_exceeds"]) for q in rob), "20-54/20-54",
          text="hold under 20 of 20 other bootstrap seeds and in every leave-one-patient-out subset")
    for cell, txt in (("slim|contrast@0", "disagree (0.51; 5.62)"), ("erm|contrast@2", "disagree (1.92; 7.34)"),
                      ("slim|color_var@2", "disagree (1.84; 8.70)"), ("causal|contrast@0", "agree (0.00; 0.34)"),
                      ("causal|color_var@0", "agree (0.00; 0.27)")):
        c = rb["cells"][cell]
        check("Table 5 frontiers %s" % cell, (c["disagrees_below_margin"], c["agrees_above_margin"]),
              tuple(float(x) for x in txt.split("(")[1].rstrip(")").split("; ")), 5e-3, text=txt)
sc = load("c64_route_screens")
if sc:
    fr = sc["frontier"]
    nval = sum(s["valid"] for part in ("registered", "frontier") for t in sc[part] for s in sc[part][t].values())
    check("screens: no constructed control passes all three", nval, 0, text="No constructed control passes all three screens")
    check("screens: optimised radial bottlenecks 0.56 and 0.06", (fr["contrast@2"]["rad"]["rho"], fr["color_var@2"]["rad"]["rho"]),
          (0.56, 0.06), 5e-3, text="bottlenecks of 0.56 and 0.06")
    check("screens: proxy shift 20 and 29%", (100 * fr["contrast@2"]["rad"]["proxy_shift"], 100 * fr["color_var@2"]["rad"]["proxy_shift"]),
          (20, 29), 0.5, text="mean proxy shift is 20 and 29%")
    check("screens: 30 and 71% of the edit's shift", (100 * fr["contrast@2"]["rad"]["proxy_ratio"], 100 * fr["color_var@2"]["rad"]["proxy_ratio"]),
          (30, 71), 0.5, text="30 and 71% of the concept edit's own shift")
    check("screens: hue misses balance by 9.2 and 10.9", (fr["contrast@0"]["hue"]["rho"], fr["color_var@0"]["hue"]["rho"]),
          (9.2, 10.9), 0.05, text="factors of 9.2 and 10.9")
    rg = sc["registered"]
    check("screens: off-target 4.9 and 5.3% at the lowest contrast dose",
          (100 * rg["contrast@0"]["rad4"]["offtarget_median_rel_control"], 100 * rg["contrast@0"]["rad16"]["offtarget_median_rel_control"]),
          (4.9, 5.3), 0.05, text="4.9 and 5.3% at the lowest contrast dose", where="supp")
d = load("c63_route_exploratory")
if d:
    q = d["components"]["slim|contrast@0|D_rad4-rad16"]
    check("exploratory: Slim persists at 2.32 on 49 images", (q["est"], q["n_images"]), (2.32, 49), 5e-3, text="2.32 logits on 49 images")
    check("exploratory: decision exceeds", q["decision_sim"], "exceeds")
    check("exploratory: critical value 2.80", d["crit_simultaneous"], 2.80, 5e-3, text="critical value 2.80", where="supp")
d = load("c62_frontier")
if d:
    tks = ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
    fine = [sum(r["targets"][t]["rad"]["fine_feasible"] for r in d["images"]) for t in tks]
    check("frontier: radial within 0.1% on 13 and 15 images at the high doses, none at the low doses", "/".join(map(str, fine)), "0/13/0/15",
          text="for 13 and 15 of the 78 images at the two high doses")
    h10 = sum(1 for t in tks for r in d["images"] if r["targets"][t]["hue"]["neutral_converged"]
              and max(abs(r["targets"][t]["hue"]["rel_dE"]), abs(r["targets"][t]["hue"]["rel_hf"])) <= 0.10)
    check("frontier: hue never within 10%", h10, 0, text="The hue controls never came within 10%")
d = load("c56_simultaneous36")
if d:
    check("36-component critical value 3.11 (headline band)", d["crit_36"], 3.11, 5e-3, text="1.96 to 3.11")
    check("every cell exceeds every tolerance below 0.064 (36 components)", d["frontier_36"], 0.064, floor=True,
          text="every tolerance below 0.064 probability units")
    check("36-component frontier 0.0647 (supp)", d["frontier_36"], 0.0647, 5e-5, text="0.0647", where="supp")
    check("36 band: magnitude still exceeds in all 18", d["magnitude_exceed_36"], 18)
    check("36 band signed: 11 exceed / 1 within / 6 indet",
          "%d/%d/%d" % (d["signed_counts_36"]["exceeds"], d["signed_counts_36"]["within"], d["signed_counts_36"]["indet."]),
          "11/1/6", text="exceeds in 11 cells, is within tolerance in 1 and is indeterminate in 6", where="supp")
    ri = d["components"]["r50-Inv/mal/sgn"]["sim36"]
    check("R50 Inv malignant signed sim36 [-0.040, +0.029]", tuple(ri), (-0.040, 0.029), 5e-4,
          text="[−0.040, +0.029]", where="supp")
    check("36 band critical value agrees with c40 (2.94 for 18)", d["crit_18_magnitude_only"], 2.94, 5e-3)
if cells:
    mlb = min(c["ci"][0] for c in cells.values())
    check("marginal bound 0.077", mlb, 0.077, 5e-4, text="put the bound at 0.077")
    check("0.077 is 1.54 times the tolerance", mlb / 0.05, 1.54, 5e-3, text="1.54 times")

d = load("c32_absdp_patient_ci")
if d:
    c = d["cells"]
    check("cached run: Inv benign 0.140 [0.127, 0.153]",
          (c["Inv"]["benign"]["mean"], c["Inv"]["benign"]["ci"][0]), (0.140, 0.127), 5e-4,
          text="0.140 with interval [0.127, 0.153]", where="supp")
    check("cached run: Slim malignant 0.762 [0.707, 0.808]",
          (c["Slim"]["malignant"]["mean"], c["Slim"]["malignant"]["ci"][1]), (0.762, 0.808), 5e-4,
          text="0.762 with interval [0.707, 0.808]", where="supp")

d = load("c50_overlay_seed")
if d:
    sp = [v["spread_pct_of_seed42"] for v in d["cells"].values()]
    check("seven realizations move magnitudes 3.2 to 12.3%", rng(sp), (3.2, 12.3), 0.05,
          text="3.2 to 12.3%")
    check("no verdict changes across realizations", all(v["verdict_stable"] for v in d["cells"].values()), True)
    check("smallest lower bound across realizations 0.077",
          min(v["lower_bound_min"] for v in d["cells"].values()), 0.077, floor=True)
    check("seven realizations", len(d["seeds"]), 7)

d = load("c42_positive_control")
if d:
    order = [l["name"] for l in d["ladder"]]
    frac = {l["name"]: 100 * l["change_frac"] for l in d["ladder"]}
    verd = []
    first = []
    for bk in d["models"]:
        for mk in d["models"][bk]:
            ed = d["models"][bk][mk]
            for e, subs in ed.items():
                if e.startswith("_"):
                    continue
                verd += [v["verdict"] for v in subs.values()]
            fe = next(e for e in order if ed[e]["all"]["verdict"] == "exceedance")
            first.append(frac[fe])
    check("dose grid: 144 cells", len(verd), 144, text="gives 144 cells")
    check("dose grid: 56 within tolerance", verd.count("equivalence"), 56, text="56 of 144 cells", where="supp")
    check("dose grid: 17 indeterminate", verd.count("indeterminate"), 17)
    check("first exceedance 0.34 to 2.37%", rng(first), (0.34, 2.37), 5e-3, text="0.34 to 2.37%")
    idv = [v["verdict"] for bk in d["models"] for mk in d["models"][bk]
           for v in d["models"][bk][mk]["identity"].values()]
    check("identity edit within tolerance in all eighteen cells", idv.count("equivalence"), 18)
    check("non-identity: 38 of 126 within tolerance",
          (verd.count("equivalence") - idv.count("equivalence"), len(verd) - len(idv)), (38, 126),
          text="Of the 126 other cells, 38 fall within tolerance")
    check("dose grid spans 0.16 to 4.06%", rng(list(frac.values())), (0.16, 4.06), 5e-3,
          text="0.16 to 4.06%")

d = load("c46_padufes_artifact")
if d:
    agg = [d["models"][m]["_heldout_aggregate"][sg]["mean"]
           for m in d["models"] for sg in ("all", "malignant", "benign")]
    check("second cohort: six cells exceed", d["summary"]["exceedance"], 6)
    check("second cohort: 0.069 to 0.096", rng(agg), (0.069, 0.096), 5e-4, text="0.069 to 0.096")
    first = {m: next(e for e in ("hair_x1", "hair_x2", "hair_x3", "hair_light")
                     if d["models"][m][e]["all"]["verdict"] == "exceedance") for m in d["models"]}
    check("second cohort: both first exceed at 1.02%", str(sorted(set(first.values()))), "['hair_light']",
          text="first exceed at 1.02% of altered pixels")

d = load("c13_fullpop")
if d:
    ph = [abs(d["models"][m]["edits"][e]["malignant"]["dp"]["mean"])
          for m in d["models"] for e in ("art:vignette", "art:jitter")]
    check("photometric edits move the malignant score by at most 0.09", max(ph), 0.095, ceil=True,
          text="by at most 0.09")
    lg = [abs(d["models"][m]["edits"][e]["malignant"]["dlogit"]["mean"])
          for m in d["models"] for e in ("art:vignette", "art:jitter")]
    check("photometric edits move the malignant logit by at most 1.5", max(lg), 1.5, ceil=True,
          text="at most 1.5", where="supp")
    resp = [d["models"][m]["edits"][t]["all"]["dlogit"]["mean"] for m in d["models"]
            for t in ("cpt:contrast@0", "cpt:contrast@2", "cpt:color_var@0", "cpt:color_var@2")]
    mx = max(abs(x) for x in resp)
    check("largest concept response 0.44 logits", mx, 0.44, 5e-3,
          text="At the largest, 0.44 logits", where="supp")
    check("component-range width 27.19 logits", 2 * B - mx, 27.19, 5e-3, text="27.19 logits", where="supp")
    check("width is 98.4% of the 27.63-logit range", 100 * (2 * B - mx) / (2 * B), 98.4, 0.05,
          text=["98.4%", "27.63-logit"])
    check("concept responses run −0.06 to +0.44", rng(resp), (-0.06, 0.44), 5e-3,
          text="−0.06 to +0.44", where="supp")

d = load("c19_saturation_mde")
if d:
    check("clipped logit bound B = 13.8155", d["clip_bound"], 13.8155, 5e-5, text="13.8155")
    check("61.2% of hair-edit predictions at the bound",
          100 * d["saturation_by_edit"]["art:hair"]["fraction"], 61.2, 0.05, text="61.2%")
    check("13.3% of edited predictions at the bound",
          100 * d["saturation"]["ALL models, edited only"]["fraction"], 13.3, 0.05, text="13.3%",
          where="supp")
    mde = []

    def wm(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "mde" and isinstance(v, (int, float)):
                    mde.append(v)
                else:
                    wm(v)
        elif isinstance(o, list):
            for v in o:
                wm(v)
    wm(d["mde_paired_contrast"])
    check("minimum detectable effect 0.13 to 1.05 logits", rng(mde), (0.13, 1.05), 5e-3,
          text="0.13 to 1.05")

d = load("c20_numerical_repro")
if d:
    cf = d["configs"]
    diffs = [abs(cf[c][m] - cf["bf16_per_image"][m]) for c in ("fp32_per_image", "bf16_chunk32")
             for m in cf["bf16_per_image"]]
    check("numerical check on the 183-image cohort", d["n_images"], 183)
    check("pipeline noise floor 0.303 logits", max(diffs), 0.303, 5e-4, text="0.303 logits", where=["paper", "supp"])
    pc = d["pct_change_vs_reference"]
    check("fp32 changes the aggregate by −1.8 to +2.8%", rng(list(pc["fp32_per_image"].values())),
          (-1.8, 2.8), 0.05, text="−1.8 to +2.8%", where="supp")
    check("batching changes it by −0.3 to +0.3%", rng(list(pc["bf16_chunk32"].values())),
          (-0.3, 0.3), 0.05, text="−0.3 to +0.3%", where="supp")

d = load("c21_patient_cluster")
if d:
    rat, agree, tot = [], 0, 0
    for mm in d["quantities"].values():
        for qq in mm.values():
            for v in qq.values():
                wi = v["ci_image"][1] - v["ci_image"][0]
                wp = v["ci_patient"][1] - v["ci_patient"][0]
                rat.append(wp / wi)
                agree += v["sig_image"] == v["sig_patient"]
                tot += 1
    check("patient-clustering comparison on the 183-image cohort", d["n_images"], 183)
    check("42 quantities, 39 agree", "%d/%d" % (agree, tot), "39/42", text="agrees in 39 of the 42",
          where="supp")
    check("width ratio 0.86 to 1.41", rng(rat), (0.86, 1.41), 5e-3, text="0.86 and 1.41", where="supp")

d = load("c26_hair_dose_prob")
if d:
    st = d["ERM"]["steps"][0]
    check("ERM first dose step −0.395", st["mean"], -0.395, 5e-4, text="−0.395 with [−0.482, −0.292]",
          where="supp")

# ================================================================ concept tier
d = load("c25_screen_full")
if d:
    sp = list(d["spearman"].values())
    check("Spearman −0.51 to −0.82", (max(sp), min(sp)), (-0.51, -0.82), 5e-3, text="−0.51 to −0.82")
    weak = d["spearman_ci"][max(d["spearman"], key=d["spearman"].get)]
    check("weakest Spearman interval [−0.79, −0.04]", tuple(weak), (-0.79, -0.04), 5e-3,
          text="[−0.79, −0.04]")
    check("four threshold schemes", len(d["schemes"]), 4, text="four threshold schemes")
    check("no valid control under any scheme",
          sum(s["valid_total"] for s in d["schemes"].values()), 0)
    bal = [s["tol"][0] for s in d["schemes"].values()]
    check("balance tolerances span 0.05 to 0.30", rng(bal), (0.05, 0.30), 1e-9,
          text="balance tolerances from 0.05 to 0.30")
    n_ctrl = {t: len([k for k in v if not k.startswith("_")]) for t, v in d["design"].items()}
    check("19 candidate controls per target", str(sorted(set(n_ctrl.values()))), "[19]",
          text="19 candidate controls")
    rotd = d["design"]["contrast@2"]["rotd"]
    check("best-balanced control: SMD 0.05, proxy 99%", (rotd["max_abs_smd"], 100 * rotd["proxy_ratio"]),
          (0.05, 99), 0.6, text="0.05 but moves the concept proxy by 99%")
    check("most neutral control: SMD 1.40",
          d["design"]["contrast@2"]["pixperm_m"]["max_abs_smd"], 1.40, 5e-3, text="1.40")
    neutral_smd = [x["max_abs_smd"] for v in d["design"].values() for k, x in v.items()
                   if not k.startswith("_") and x["proxy_shift"] <= 0.10 and abs(x["proxy_ratio"]) <= 0.15]
    check("neutral controls miss balance by at least 11 times", min(neutral_smd) / 0.10, 11, floor=True,
          text="at least 11 times")
    disjoint = all(not (x["max_abs_smd"] <= 0.10 and x["proxy_shift"] <= 0.10 and abs(x["proxy_ratio"]) <= 0.15)
                   for v in d["design"].values() for k, x in v.items() if not k.startswith("_"))
    check("balanced and neutral sets are disjoint", disjoint, True)

d = load("c55_identification_graph")
if d:
    pt = d["per_target"]
    check("88 performed edits", d["n_edits"], 88, text="88 performed edits")
    g0 = d["exact_graph_G0"]
    check("four exactly matched pairs", g0["n_exactly_matched_pairs"], 4, text="exactly four pairs")
    check("eight distinct edits in those pairs", len({e for p in g0["pairs"] for e in p}), 8,
          text="eight distinct edits")
    check("length-two route rho 9.7 to 13.3", rng([v["bottleneck_length_two"] for v in pt.values()]),
          (9.7, 13.3), 0.05, text="9.7 to 13.3")
    check("best route rho 1.17 to 2.13", rng([v["bottleneck_over_all_paths"] for v in pt.values()]),
          (1.17, 2.13), 5e-3, text="1.17 to 2.13")
    bf = sorted(v["boot_frac_at_or_below_one"] for v in pt.values())
    check("three targets: bottleneck above one in 98.5 to 100% of resamples", (100 - 100 * bf[2], 100 - 100 * bf[0]), (98.5, 100.0), 0.05,
          text="98.5 to 100% of resamples")
    check("color variegation at top dose: above one in 89.9%", 100 - 100 * pt["color_var@2"]["boot_frac_at_or_below_one"],
          89.9, 0.05, text="89.9%")
    check("its bottleneck interval [0.90, 1.95]", tuple(pt["color_var@2"]["bottleneck_ci95"]),
          (0.90, 1.95), 5e-3, text="[0.90, 1.95]")
    check("route bootstrap 5,000 resamples", d["boot"], 5000, text="5,000 resamples")
    check("route reselected in every resample", d.get("route_reselected_each_resample"), True,
          text="re-runs the route search")
    ci = {t: pt[t]["bottleneck_ci95"] for t in pt}
    check("Table S26 intervals", "; ".join("%.2f, %.2f" % tuple(ci[t]) for t in ("contrast@2", "color_var@2", "contrast@0", "color_var@0")),
          "1.10, 2.34; 0.90, 1.95; 1.09, 1.95; 1.24, 2.10", text="[1.10, 2.34] | 1.5%", where="supp")
    fr = {t: 100 * pt[t]["boot_frac_at_or_below_one"] for t in pt}
    check("Table S26 shares", "%.1f, %.1f, %.1f, %.1f" % (fr["contrast@2"], fr["color_var@2"], fr["contrast@0"], fr["color_var@0"]),
          "1.5, 10.1, 0.4, 0.0", text="[1.24, 2.10] | 0.0%", where="supp")
    check("footprint parts 0.47 to 3.41", rng([v["total_footprint_part"] for v in pt.values()]),
          (0.47, 3.41), 5e-3, text="0.47, 0.71, 0.72 and 3.41", where="supp")
    check("concept parts 2.66 to 6.52", rng([v["total_concept_part"] for v in pt.values()]),
          (2.66, 6.52), 5e-3, text="6.52, 5.69, 4.72 and 2.66", where="supp")
    L = d["lipschitz"]
    wp = L["worst_pair"]
    check("binding pair 9.64 logits", wp["abs_diff_logits"], 9.64, 5e-3, text="9.64 logits")
    check("at 7.24 scored units", wp["distance_tolerance_units"], 7.24, 5e-3, text="7.24 scored units")
    check("L^g >= 1.33", L["implied_L_binding"], 1.33, 5e-3, text="= 1.33$")
    check("dropping the binding pair gives 1.16", L["implied_L_second_largest"], 1.16, 5e-3,
          text="leaves *L*ᵍ ≥ 1.16", where="supp")
    check("conservative floor 0.84", L["implied_L_conservative_interval_ends"], 0.84, 5e-3, text="gives *L*ᵍ ≥ 0.84", where="supp")
    check("median implied constant 0.26", L["implied_L_median"], 0.26, 5e-3, text="over the twenty-one pairs is 0.26", where="supp")
    check("quartiles 0.12 and 0.71", tuple(L["implied_L_quartiles"]), (0.12, 0.71), 5e-3,
          text="quartiles 0.12 and 0.71", where="supp")
    check("twenty-one Lipschitz comparisons", L["n_pairs"], 21)
    hw = L["footprint_only_half_width_at_binding_L"]
    check("half-widths 0.62, 0.94, 0.95, 4.54",
          "%.2f, %.2f, %.2f, %.2f" % (hw["color_var@0"], hw["contrast@2"], hw["contrast@0"], hw["color_var@2"]),
          "0.62, 0.94, 0.95, 4.54", text="0.62, 0.94, 0.95 and 4.54", where="supp")
    be = L["breakeven_kappa_Lh"]
    check("full-width break-even 0.017, 0.017, 0.064 and none at color_var@2",
          "%.3f, %.3f, %.3f, %.3f" % (be["contrast@2"], be["contrast@0"], be["color_var@0"], be["color_var@2"]),
          "0.017, 0.017, 0.064, 0.000", text="0.017, 0.017 and 0.064")
    check("break-even uses full width (hw <= range/2)",
          be["contrast@2"], (L["score_range_logits"] / 2 - hw["contrast@2"]) / pt["contrast@2"]["total_concept_part"], 1e-9)
    check("break-even is 21 to 80 times below the floor",
          rng([L["implied_L_binding"] / v for v in be.values() if v > 0]), (21, 80), 0.6, text="21 to 80 times", where="supp")
    check("full width 9.09 at color_var@2", 2 * hw["color_var@2"], 9.09, 5e-3, text="full width is 9.09 logits")
    check("corrected scores span 2.07 logits", L["score_range_logits"], 2.07, 5e-3, text="2.07 logits")
    check("interaction price eps_q >= 4.82", wp["abs_diff_logits"] / 2, 4.82, 5e-3, text="4.82$ logits")

d = load("c34_ctrl_vs_ctrl")
if d:
    su = d["summary"]
    diffs = [p["diff"] for p in d["pairs"]]
    check("twenty-one control comparisons", su["n_comparisons"], 21, text="twenty-one model-by-pair")
    check("all twenty-one exclude zero (patient)", su["excluding_zero_patient"], 21,
          text="all twenty-one patient-clustered intervals exclude zero")
    check("all twenty-one exclude zero (image)", su["excluding_zero_image"], 21)
    check("median disagreement 3.26 logits", statistics.median(abs(x) for x in diffs), 3.26, 5e-3,
          text="median of 3.26 logits")
    check("disagreement 0.55 to 12.91", rng([abs(x) for x in diffs]), (0.55, 12.91), 5e-3,
          text="0.55 to 12.91")
    check("fifteen positive, six negative", "%d/%d" % (sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)),
          "15/6", text="fifteen positive and six negative")
    check("pairs per target 0/1/3/3", str(su["n_pairs_per_model"]),
          "{'contrast@2': 0, 'color_var@2': 1, 'contrast@0': 3, 'color_var@0': 3}",
          text="3, 3, 1 and 0 pairs", where="supp")

d = load("c12_matched_control")
if d:
    r2 = {m: {f: v["r2_within"] for f, v in d["models"][m]["g_fit"]["lofo"].items()} for m in d["models"]}
    allr = [x for m in r2.values() for x in m.values()]
    check("leave-one-family-out R^2 negative everywhere", max(allr), 0.0, ceil=True)
    check("below −100 for one family in two of three models",
          sum(r2[m]["globrot"] < -100 for m in r2), 2, text="below −100 for one family in two of the three models")
    check("globrot R^2 −751.6 / −55.1 / −334.7",
          "%.1f/%.1f/%.1f" % (r2["erm"]["globrot"], r2["slim"]["globrot"], r2["causal"]["globrot"]),
          "-751.6/-55.1/-334.7", text="−751.6 for ERM, −55.1 for Slim and −334.7", where="supp")
    check("LOFO fit on 82 common images", d["models"]["erm"]["g_fit"]["n_img"], 82, text="82 images",
          where="supp")

d = load("c45_bracketing")
if d:
    su = d["summary"]
    check("balance+overlap controls bracket no cell", su["A1+A3"]["cells_bracketed"], 0)
    check("whole library brackets 11 of 12", su["all"]["cells_bracketed"], 11, text="11 of the 12")
    lo, hi = su["all"]["edge_control_smd_range"]
    check("bracketing controls 4.4 to 24 times outside tolerance", (lo / 0.1, hi / 0.1), (4.4, 24.3), 0.06,
          text="4.4 to 24 times")

d = load("c38_shrinkage_sensitivity")
if d:
    w = [v["slope_ci_patient"][1] - v["slope_ci_patient"][0] for m in d["models"].values() for v in m.values()]
    check("slope interval widths 0.04 to 0.40", rng(w), (0.04, 0.40), 5e-3, text="0.04 to 0.40", where="supp")
    ff = d["functional_form"]
    check("linear beats constant in ten of twelve",
          sum(v["linear"] > v["const"] for m in ff.values() for v in m.values()), 10,
          text="ten of twelve", where="supp")
    hs = d["holdout_slopes"]
    check("largest half-split disagreement 0.32",
          max(abs(v["slope_half_a"] - v["slope_half_b"]) for m in hs.values() for v in m.values()), 0.32, 5e-3,
          text="disagreement between halves is 0.32", where="supp")

d = load("c54_synthetic_scm")
if d:
    s = json.dumps(d)
    tgt = d["target_h"]
    rg = d["regimes"]
    check("simulated target 1.117", tgt, 1.117, 5e-4, text="1.117")
    check("valid regime within 0.8%", 100 * rg["R1_valid"]["abs_error_frac_of_target"], 0.8, 0.05,
          text="within 0.8%")
    check("model-state regime biased 6.7%", 100 * rg["R4_model_state"]["abs_error_frac_of_target"], 6.7, 0.05,
          text="6.7")
    check("proxy-blind regime biased 42.5%", 100 * rg["R5_proxy_blind"]["abs_error_frac_of_target"], 42.5, 0.05,
          text="42.5%")
    check("screens admit R1, R4, R5 and reject R2, R3",
          "".join("A" if v["screen"]["admissible"] else "R" for v in rg.values()), "ARRAA")
    ch = float(re.search(r'"chained_route_estimate": ([\d.]+)', s).group(1))
    ci = re.search(r'"bootstrap_ci95": \[([\d.]+), ([\d.]+)\]', s)
    near = float(re.search(r'"nearest_neutral_control_footprint_distance_max_smd": ([\d.]+)', s).group(1))
    check("designed length-four route +1.111", ch, 1.111, 5e-4, text="+1.111")
    check("its interval [1.085, 1.151]", (float(ci.group(1)), float(ci.group(2))), (1.085, 1.151), 5e-4,
          text="[1.085, 1.151]")
    check("recovery within 0.6%", 100 * abs(ch - tgt) / tgt, 0.6, ceil=True, text="within 0.6")
    check("nearest neutral control 1.65 SMD away", near, 1.65, 5e-3, text="1.65 SMD")
    check("no length-two route there", re.search(r'"placebo_route_exists": (\w+)', s).group(1), "false")

d = load("c49_tcav")
if d:
    accs, flags = [], []

    def walk(o):
        if isinstance(o, dict):
            if "cav_accuracy_mean" in o:
                accs.append(o["cav_accuracy_mean"])
            if "significant" in o:
                flags.append(bool(o["significant"]))
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    check("CAV accuracy 0.83 to 0.94", rng(accs), (0.83, 0.94), 6e-3, text="0.83 to 0.94")
    check("significant in 9 of 12 cells", "%d of %d" % (sum(flags), len(flags)), "9 of 12", text="9 of 12")

d = load("c10_multiplacebo")
if d:
    check("multi-placebo sample: 82 malignant audit images", d["n_common"], 82,
          text="82 malignant audit images", where="supp")
    check("multi-placebo intervals are patient-clustered", d.get("resampling_unit"), "patient",
          text="with the patient as the resampling unit", where="supp")
    check("multi-placebo: 44 patients", d.get("n_patients_common"), 44)
    a = {m: {k: v["A"] for k, v in mv["A_corr_by_placebo"].items()} for m, mv in d["models"].items()}
    check("Slim −0.42 under flip", a["slim"]["flip"], -0.42, 5e-3, text="−0.42")
    check("Slim +4.23 under noise", a["slim"]["noise"], 4.23, 5e-3, text="+4.23")
    flips = sum(1 for m in a if min(a[m].values()) < 0 < max(a[m].values()))
    check("sign changes for two of three models", flips, 2, text="two of the three")
    # Table S29, cell by cell
    tx = TEXT["supp"]
    t = tx[tx.index("| Model | Placebo | proxy change of placebo"):]
    t = t[:t.index("\n\n")]
    name = {"ERM": "erm", "Slim": "slim", "Inv": "causal"}
    mk = None
    bad = n = 0
    for line in t.splitlines()[2:]:
        f = [x.strip() for x in line.strip("|").split("|")]
        mk = name.get(f[0], mk)
        p = f[1]
        mv = d["models"][mk]
        got = ([100 * d["placebo_proxychg"][p]] + mv["placebo_cace"][p] + mv["A_corr_by_placebo"][p]["contrast"]
               + mv["A_corr_by_placebo"][p]["color"] + [mv["A_corr_by_placebo"][p]["A"]])
        nums = [num(x) for x in re.findall(r"[−+]?\d+\.\d+", line)]
        ok = len(nums) == 11 and abs(nums[0] - got[0]) <= 0.05 and all(abs(x - y) <= 5e-3 + 1e-9 for x, y in zip(nums[1:], got[1:]))
        n += 1
        bad += not ok
        if not ok:
            rows.append(("MISMATCH", "Table S29 row %s %s" % (mk, p), str([round(x, 3) for x in got]), line))
    check("Table S29: all 12 rows match c10", n - bad, 12)

# ================================================================ report
w = max(len(r[1]) for r in rows)
print("verification of paper3.md and supplementary3.md against %s" % RES)
print("=" * (w + 50))
for st, cl, g, wv in rows:
    print("%-9s %-*s  got %-22s paper %s" % (st, w, cl, g, wv))
print("=" * (w + 50))
nbad = sum(r[0] != "ok" for r in rows)
print("%d checks, %d ok, %d not ok" % (len(rows), len(rows) - nbad, nbad))
sys.exit(1 if nbad else 0)
