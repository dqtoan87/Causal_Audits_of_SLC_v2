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
import math
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
TRAIN_LOG_B3INV = os.path.join(BASE, "log", "c29_b3inv_retrain.log")
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
    _mp = len({p for p, t in zip(d["patient_id"], d["target"]) if t == 1})
    check("audit cohort: malignant images come from 44 patients", _mp, 44,
          text="The audit cohort holds 100 patients, and its malignant subgroup 44.")

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
    # the EfficientNet-B3 Inv model was trained again with selection restricted to epochs with the penalties on
    lg2 = open(TRAIN_LOG_B3INV, encoding="utf-8").read() if os.path.exists(TRAIN_LOG_B3INV) else ""
    auc = {k: float(v) for k, v in re.findall(r"\[(dj_\w+)\] (?:saved|lưu), select-AUC=([\d.]+)", lg)}
    auc2 = {k: float(v) for k, v in re.findall(r"\[(dj_\w+)\] (?:saved|lưu), select-AUC=([\d.]+)", lg2)}
    auc.update(auc2)
    if auc:
        check("selection AUROC, EfficientNet-B3 trio",
              "%.2f/%.2f/%.2f" % (auc["dj_erm_b3"], auc["dj_slim_b3"], auc["dj_causal_b3"]),
              "0.90/0.89/0.87", text="0.90, 0.89 and 0.87")
        check("selection AUROC, ResNet-50 trio",
              "%.2f/%.2f/%.2f" % (auc["dj_erm_r50"], auc["dj_slim_r50"], auc["dj_causal_r50"]),
              "0.87/0.88/0.89", text="0.87, 0.88 and 0.89", where="supp")
    _ep = {}
    for _tag, _lg in (("dj_causal_b3", lg2), ("dj_causal_r50", lg)):
        _eps = re.findall(r"\[%s\] ep(\d+)/\d+ auc=([\d.]+) .*? pen=(on|off)" % _tag, _lg)
        _best = [e for e in _eps if _tag in auc and abs(float(e[1]) - auc[_tag]) < 5e-5 and e[2] == "on"]
        _ep[_tag] = _best[0] if _best else None
    if all(_ep.values()):
        check("Inv checkpoints saved at epochs with the penalties on: B3 epoch 7, R50 epoch 6",
              "%d %s / %d %s" % (int(_ep["dj_causal_b3"][0]), _ep["dj_causal_b3"][2],
                                 int(_ep["dj_causal_r50"][0]), _ep["dj_causal_r50"][2]), "7 on / 6 on",
              text=["the saved epochs are 7 for EfficientNet-B3 and 6 for ResNet-50",
                    "The EfficientNet-B3 Inv checkpoint was saved at epoch 7 and the ResNet-50 Inv checkpoint at epoch 6"],
              where=["paper", "supp"])
    secs = [int(x) for x in re.findall(r"\((\d+)s\)", lg)]
    if secs and lg2:
        # cumulative seconds per job in the first log; the B3 Inv job is replaced by the second log's total
        cum = {k: int(v) for k, v in re.findall(r"\[(dj_\w+)\] (?:saved|lưu), select-AUC=[\d.]+  \((\d+)s\)", lg)}
        old_b3inv = cum["dj_causal_b3"] - cum["dj_slim_b3"]
        new_b3inv = int(re.findall(r"\[dj_causal_b3\] saved, select-AUC=[\d.]+  \((\d+)s\)", lg2)[0])
        check("training the six checkpoints takes 2.1 hours", (max(secs) - old_b3inv + new_b3inv) / 3600, 2.1, 0.05,
              text="2.1 hours")

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
    check("six cells do not exceed on the signed component",
          sum(c["signed_verdict"] != "exceedance" for c in cl), 6,
          text="Six of the eighteen cells do not exceed on the signed component")
    check("two cells within the signed band", sum(c["signed_verdict"] == "equivalence" for c in cl), 2)
    check("primary magnitudes 0.085 to 0.769", rng([c["mean"] for c in cl]), (0.085, 0.769), 5e-4)
    au = [d["models"][b][m]["audit_auroc"] for b in d["models"] for m in d["models"][b]]
    check("audit-cohort AUROC 0.856 to 0.893", rng(au), (0.856, 0.893), 5e-4, text="0.856 to 0.893")
    check("hair alters 2.4% of pixels", 100 * d["change_frac"]["hair"], 2.4, 0.05, text="2.4% of pixels")
    check("identity edit alters no pixels", d["change_frac"]["identity"], 0.0)
    b3 = d["models"]["b3"]
    for mk, p0s, p1s in (("ERM", "0.733", "0.070"), ("Slim", "0.859", "0.170"), ("Inv", "0.501", "0.046")):
        p0 = b3[mk]["p0_mean_malignant"]
        p1 = p0 + b3[mk]["hair_dose_dp"][1]
        check("%s malignant score under hair %s to %s" % (mk, p0s, p1s), (p0, p1),
              (float(p0s), float(p1s)), 5e-4, text="from %s to %s" % (p0s, p1s))
    # frontier at tolerance 0.20
    def exceeds_at(c, t):
        mag = c["ci"][0] > t
        sg = c["signed_ci"][0] > t or c["signed_ci"][1] < -t
        return mag or sg
    check("seven cells still exceed at tolerance 0.20", sum(exceeds_at(c, 0.20) for c in cl), 7,
          text="at 0.20, seven of the eighteen cells still exceed")
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
    check("secondary 18-magnitude band: critical value 2.96", d["disjoint_18"]["crit_z"], 2.96, 5e-3, text="critical value of 2.96")
    check("secondary 18-magnitude band: frontier 0.044", min(lo), 0.044, floor=True,
          text="frontier of 0.044")
    check("simultaneous replicates 20,000", d["boot"], 20000)
    p9 = d["primary_9"]
    check("nine cached-prediction cells: critical value 2.68", p9["crit_z"], 2.68, 5e-3,
          text="critical value is 2.68", where="supp")
    check("nine cached-prediction cells: Inv benign bound 0.049",
          min(v["ci_simultaneous"][0] for v in p9["cells"].values()), 0.049, 5e-4,
          text="whose simultaneous lower bound is 0.049", where="supp")
d = load("c57_stratified_band")
if d:
    check("stratified band: critical value 3.12", d["crit_36"], 3.12, 5e-3, text="critical value of 3.12")
    check("stratified band: frontier 0.041", d["frontier_36"], 0.041, floor=True, text="frontier of 0.041")
    check("stratified band: frontier 0.0417 (supp)", d["frontier_36"], 0.0417, 5e-5, text="0.0417", where="supp")
    check("stratified band: strata 44 and 56", tuple(d["n_patients_strata"]), (44, 56), 0, text="the 44 patients", where="supp")
    check("stratified band: 17 of 18 still exceed", d["magnitude_exceed_36"], 17, text="Seventeen of the eighteen cells still exceed", where="supp")
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
    check("benign redraw: joint coverage 95.4%", 100 * d["joint_coverage"], 95.4, 0.05, text="in 95.4% of draws")
    check("benign redraw: MC se 0.9 points", 100 * d["joint_coverage_mc_se"], 0.9, 0.05, text="0.9 percentage points", where="supp")
    check("benign redraw: component coverage min 98.8%", 100 * d["component_coverage_min"], 98.8, 0.05, text="fewer than 98.8% of draws", where="supp")
    check("benign redraw: crit 2.99 to 3.23", tuple(d["crit_range"]), (2.99, 3.23), 5e-3, text="from 2.99 to 3.23", where="supp")
    check("benign redraw: crit median 3.08", d["crit_median"], 3.08, 5e-3, text="median of 3.08", where="supp")
    check("benign redraw: frontier median 0.0470", d["frontier_median"], 0.0470, 5e-5, text="median of 0.0470", where="supp")
    check("benign redraw: frontier min 0.0280", d["frontier_min"], 0.0280, 5e-5, text="minimum of 0.0280", where="supp")
    check("benign redraw: every tolerance below 0.027 in every draw", d["frontier_min"], 0.027, floor=True, text="every tolerance below 0.027 in every draw", where="supp")
    check("benign redraw: frontier central range 0.033 to 0.066", tuple(d["frontier_2.5_97.5"]), (0.033, 0.066), 5e-4, text="0.033 to 0.066", where=["paper", "supp"])
    check("benign redraw: all 18 exceed 0.05 jointly in 38% of draws", 100 * d["share_draws_all_18_exceed"], 38.2, 0.05, text="jointly in 38% of draws")
    check("benign redraw: frame 4,150 benign from 198 patients", (d["n_benign_frame"], d["n_benign_frame_patients"]), (4150, 198), 0, text="4,150 benign images from 198 patients", where="supp")
    check("benign redraw: 500 draws", d["draws"], 500, text="redraws the 100 benign images 500 times")
import math as _m
_Phi = lambda z: 0.5 * (1 + _m.erf(z / _m.sqrt(2)))
check("signed exceedance size at eps/se = 0.25 is 0.032", _Phi(-1.959964) + _Phi(-1.959964 - 0.5), 0.032, 5e-4,
      text="0.032 at *ε*/*se* = 0.25", where="supp")
def _sg2(x):
    r = round(x, 2)
    return ("+%.2f" % r) if r > 0 else (("\u2212%.2f" % abs(r)) if r < 0 else "0.00")


d = load("c62_evaluation_inv_retrained")
if d:
    # second run of the unchanged plan, with the Inv checkpoint of Section III-D: reported in the main text
    reg, con = load("c62_preregistration"), load("c62_construction")
    pr, comp = d["primary"], d["primary"]["components"]
    TT = ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
    check("route test: 78 locked images from 54 patients", (reg["locked_set"]["n_images"], reg["locked_set"]["n_patients"]), (78, 54), 0,
          text="78 malignant images of the selection fold, from 54 patients")
    a4 = [con["summary"][t]["n_all_three_g1"] for t in TT]
    check("route test: all four controls pass G1 on 74 to 78 images", (min(a4), max(a4)), (74, 78), 0, text="74 to 78 of the 78 images")
    sx = sorted(v["est"] for k, v in comp.items() if v.get("decision_sim") == "exceeds" and ("D_rad4-rad16" in k or "D_hue4-hue16" in k))
    check("route test: scale-pair disagreements 5.02 to 6.62 logits", (sx[0], sx[-1]), (5.02, 6.62), 5e-3, text="5.02 to 6.62 logits")
    check("route test: abstract 5.0 to 6.6 logits", (sx[0], sx[-1]), (5.0, 6.6), 0.05, text="5.0 to 6.6 logits")
    dis = sorted(c for c, v in pr["cells"].items() if v["class"] == "routes disagree")
    check("route test: one disagreeing cell per classifier", ",".join(dis), ",".join(sorted(["erm|contrast@2", "causal|contrast@2", "slim|color_var@2"])),
          text="In three of the twelve cells, one for each classifier")
    check("route test: 3 disagree / 0 agree / 9 indeterminate",
          "%d/%d/%d" % (pr["summary"]["routes disagree"], pr["summary"]["routes agree"], pr["summary"]["indeterminate"]), "3/0/9",
          text="Nine cells are indeterminate, and none agrees")
    check("route test: one direction pair exceeds, Inv at the highest contrast dose",
          (pr["summary"]["direction_pair_exceeds"], round(comp["causal|contrast@2|D_rad4-hue4"]["est"], 2)), (1, 4.90), 5e-3,
          text="One did under the band, for Inv at the highest contrast dose, by 4.90 logits")
    dm = sum(1 for c, v in comp.items() if ("D_rad4-hue4" in c or "D_rad16-hue16" in c) and (v["ci"][0] > 0 or v["ci"][1] < 0))
    check("route test: 12 of 24 direction differences exclude zero marginally", dm, 12, text="12 of the 24 direction differences")
    dme = sum(1 for c, v in comp.items() if ("D_rad4-hue4" in c or "D_rad16-hue16" in c) and v.get("decision_marginal") == "exceeds")
    check("route test: 7 direction differences exceed on marginal intervals", dme, 7, text="and 7 exceed the margin on their marginal intervals", where="supp")
    check("route test: simultaneous critical value 3.37", pr["crit_simultaneous"], 3.37, 5e-3, text="critical value of the band was 3.37", where="supp")
    nT = [(pr["targets"][t]["n_images"], pr["targets"][t]["n_patients"]) for t in TT]
    check("route test: primary sets 44 to 47 images from 32 or 33 patients",
          "%d-%d/%d-%d" % (min(x[0] for x in nT), max(x[0] for x in nT), min(x[1] for x in nT), max(x[1] for x in nT)), "44-47/32-33",
          text="44 to 47 images from 32 or 33 patients", where="supp")
    for t, txt in zip(TT, ("| 45 (32) |", "| 44 (32) |", "| 47 (33) |", "| 46 (32) |")):
        check("Table 5 images (patients) %s" % t, "| %d (%d) |" % nT[TT.index(t)], txt, text=txt)
    add = [d["secondary_all_g1"]["targets"][t]["n_images"] - pr["targets"][t]["n_images"] for t in TT]
    check("route test: dropping the floor adds 30 or 31 images per target", (min(add), max(add)), (30, 31), 0,
          text="adds 30 or 31 images per target", where=["paper", "supp"])
    for cell, q, txt in (("erm|contrast@2", "rad4-rad16", "+5.02 [+1.20, +8.83]"), ("slim|color_var@2", "rad4-rad16", "+6.16 [+0.62, +11.69]"),
                         ("causal|contrast@2", "rad4-rad16", "+6.62 [+1.99, +11.24]"), ("causal|contrast@2", "rad4-hue4", "+4.90 [+1.02, +8.79]")):
        v = comp["%s|D_%s" % (cell, q)]
        check("route test: Table S31 %s %s" % (cell, q), "%s [%s, %s]" % (_sg2(v["est"]), _sg2(v["sim"][0]), _sg2(v["sim"][1])), txt,
              text=txt, where="supp")
    rs, ps = [], []
    for m in ("erm", "slim", "causal"):
        for t in TT:
            S = [comp["%s|%s|S_%s" % (m, t, k)]["est"] for k in ("rad4", "rad16", "hue4", "hue16")]
            PC = [comp["%s|%s|PC_%s" % (m, t, k)]["est"] for k in ("hue", "flip", "jitter", "noise")]
            rs.append(max(S) - min(S)); ps.append(max(PC) - min(PC))
    check("route test: route sums spread 0.5 to 6.7", (min(rs), max(rs)), (0.5, 6.7), 0.05, text="spread by 0.5 to 6.7 logits", where=["paper", "supp"])
    check("route test: placebos spread 3.1 to 6.7", (min(ps), max(ps)), (3.1, 6.7), 0.05, text="3.1 to 6.7", where=["paper", "supp"])
    sall = d["secondary_all_g1"]
    check("route test: without the floor the same three cells disagree, nine indeterminate",
          "%d/%d/%d %s" % (sall["summary"]["routes disagree"], sall["summary"]["routes agree"], sall["summary"]["indeterminate"],
                           sorted(c for c, v in sall["cells"].items() if v["class"] == "routes disagree") == dis), "3/0/9 True",
          text="the same three cells disagree and the other nine are indeterminate", where=["paper", "supp"])
    ni = [d["secondary_independent"]["targets"][t]["n_images"] for t in TT]
    check("route test: independent subsets 11, 0, 0, 0", "/".join(map(str, ni)), "11/0/0/0", text="11, 0, 0 and 0 images", where="supp")
d1 = load("c62_evaluation")
if d1:
    # first run, registered outcome, reported in Supplementary Section S19
    p1r, c1 = d1["primary"], d1["primary"]["components"]
    check("first run: 3 disagree / 2 agree / 7 indeterminate",
          "%d/%d/%d" % (p1r["summary"]["routes disagree"], p1r["summary"]["routes agree"], p1r["summary"]["indeterminate"]), "3/2/7",
          text="The two lowest-dose Inv cells agreed, and seven cells were indeterminate", where="supp")
    check("first run: critical value 3.30", p1r["crit_simultaneous"], 3.30, 5e-3, text="the critical value was 3.30", where="supp")
    for cell, txt in (("slim|contrast@0", "+1.99 [+0.51, +3.47]"), ("erm|contrast@2", "+4.63 [+1.92, +7.34]"), ("slim|color_var@2", "+5.27 [+1.84, +8.70]")):
        q = c1[cell + "|D_rad4-rad16"]
        check("first run: %s" % cell, "%s [%s, %s]" % (_sg2(q["est"]), _sg2(q["sim"][0]), _sg2(q["sim"][1])), txt, text=txt, where="supp")
    s1 = d1["secondary_all_g1"]["summary"]
    check("first run: without the floor 2 disagree, 0 agree, 10 indeterminate",
          "%d/%d/%d" % (s1["routes disagree"], s1["routes agree"], s1["indeterminate"]), "2/0/10",
          text="two cells disagreed, none agreed and ten were indeterminate", where="supp")
    n1 = [(p1r["targets"][t]["n_images"], p1r["targets"][t]["n_patients"]) for t in ("contrast@0", "contrast@2", "color_var@0", "color_var@2")]
    check("first run: 73 to 77 images from 52 to 54 patients",
          "%d-%d/%d-%d" % (min(x[0] for x in n1), max(x[0] for x in n1), min(x[1] for x in n1), max(x[1] for x in n1)), "73-77/52-54",
          text="73 to 77 images from 52 to 54 patients", where="supp")
t34 = load("c34_ctrl_vs_ctrl")
if t34:
    med = {}
    for q in t34["pairs"]:
        med.setdefault(q["model"], []).append(abs(q["diff"]))
    got = "%.1f/%.1f/%.1f" % tuple(float(np.median(med[m])) for m in ("ERM", "Slim", "Inv"))
    check("diagnostic (T): per-model median disagreement 7.8, 9.5, 11.4", got, "7.8/9.5/11.4",
          text="median disagreements of 7.8, 9.5 and 11.4 logits for ERM, Slim and Inv")
    got2 = "%.2f/%.2f/%.2f" % tuple(float(np.median(med[m])) for m in ("ERM", "Slim", "Inv"))
    check("S6: per-model medians 7.80, 9.47, 11.40", got2, "7.80/9.47/11.40",
          text="7.80 logits for ERM, 9.47 for Slim and 11.40 for Inv", where="supp")
rb = load("c65_route_robustness_inv_retrained")
if rb:
    check("robustness (second run): stored responses reproduce the band", rb["reproduction_max_abs_difference"], 0.0, 1e-9)
    rob = [rb["decisive"][k] for k in ("erm|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-rad16", "causal|contrast@2|D_rad4-hue4")]
    check("robustness: ERM and Inv at the highest contrast dose hold for 20 of 20 seeds and all 32 subsets",
          "/".join("%d-%d/%d" % (q["seed_exceeds"], q["lopo_in_cell_exceeds"], q["lopo_in_cell_n"]) for q in rob), "20-32/32/20-32/32/20-32/32",
          text="in all 32 subsets that leave out one of the cell's patients", where="supp")
    check("robustness: smallest lower bounds 0.79, 1.61 and 0.74", "%.2f/%.2f/%.2f" % tuple(q["lopo_in_cell_lower_min"] for q in rob), "0.79/1.61/0.74",
          text=["smallest lower bounds of 0.79 and 1.61"], where="supp")
    check("robustness: Inv direction pair smallest bound 0.74", round(rob[2]["lopo_in_cell_lower_min"], 2), 0.74, 1e-9,
          text="with a smallest lower bound of 0.74", where="supp")
    s = rb["decisive"]["slim|color_var@2|D_rad4-rad16"]
    check("robustness: Slim highest variegation lower bound 0.62", s["sim_lower_unrounded"], 0.62, 5e-3, text="simultaneous lower bound is 0.62", where="supp")
    check("robustness: Slim highest variegation in 17 of 32 subsets and 20 of 20 seeds",
          "%d/%d %d" % (s["lopo_in_cell_exceeds"], s["lopo_in_cell_n"], s["seed_exceeds"]), "17/32 20",
          text="17 of the 32 subsets that leave out one of its patients", where=["paper", "supp"])
    check("robustness: its seed lower bounds 0.55 to 0.70", tuple(s["seed_lower_range"]), (0.55, 0.70), 5e-3,
          text="with lower bounds from 0.55 to 0.70", where="supp")
    check("robustness: its smallest subset bound 0.06", round(s["lopo_in_cell_lower_min"], 2), 0.06, 1e-9,
          text="with a smallest lower bound of 0.06", where="supp")
    check("robustness: critical value across seeds 3.35 to 3.39", tuple(rb["seed_crit_range"]), (3.35, 3.39), 5e-3,
          text="runs from 3.35 to 3.39", where="supp")
    # Table 5 and Table S35: the disagree margin is rounded down and the agree margin up
    _ev2 = load("c62_evaluation_inv_retrained")
    _cls = {"routes disagree": "disagree", "routes agree": "agree", "indeterminate": "indeterminate"}
    for _c, _v in rb["cells"].items():
        _txt = "%s (%.2f; %.2f)" % (_cls[_ev2["primary"]["cells"][_c]["class"]],
                                   math.floor(_v["disagrees_below_margin"] * 100 + 1e-9) / 100,
                                   math.ceil(_v["agrees_above_margin"] * 100 - 1e-9) / 100)
        check("Table 5 cell %s" % _c, _txt in TEXT["paper"], True, text=_txt)
    cp = rb["cross_pairs"]
    out = ["%d" % sum(1 for k, v in cp.items() if k.startswith(m + "|") and (v["ci"][0] > 0.5 or v["ci"][1] < -0.5)) for m in ("erm", "slim", "causal")]
    ins = sum(1 for v in cp.values() if v["ci"][0] > -0.5 and v["ci"][1] < 0.5)
    check("cross pairs: 4, 5 and 2 outside, none inside", "/".join(out) + " %d" % ins, "4/5/2 0",
          text="11 lie wholly outside the margin, 4 for ERM, 5 for Slim and 2 for Inv", where="supp")
rb1 = load("c65_route_robustness")
if rb1:
    s = rb1["decisive"]["slim|contrast@0|D_rad4-rad16"]
    check("first run: borderline lower bound 0.510", s["sim_lower_unrounded"], 0.510, 5e-4, text="lower bound of 0.510", where="supp")
    check("first run: borderline under 10 of 20 seeds", s["seed_exceeds"], 10, text="under 10 of 20 other seeds", where="supp")
    check("first run: borderline in 31 of the 53 subsets", (s["lopo_in_cell_exceeds"], s["lopo_in_cell_n"]), (31, 53), 0,
          text="in 31 of the 53 subsets that leave out one of its patients", where="supp")
    rob1 = [rb1["decisive"][k] for k in ("erm|contrast@2|D_rad4-rad16", "slim|color_var@2|D_rad4-rad16")]
    check("first run: the two highest-dose disagreements hold for 20 of 20 seeds and every in-cell subset",
          "/".join("%d-%d/%d" % (q["seed_exceeds"], q["lopo_in_cell_exceeds"], q["lopo_in_cell_n"]) for q in rob1), "20-52/52/20-54/54",
          text="The two highest-dose disagreements held under 20 of 20 other bootstrap seeds", where="supp")
sc = load("c64_route_screens")
if sc:
    fr = sc["frontier"]
    nval = sum(s["valid"] for part in ("registered", "frontier") for t in sc[part] for s in sc[part][t].values())
    check("screens: no constructed control passes all three", nval, 0, text="no constructed control passes all three screens")
    check("screens: optimised radial footprint-link scores 0.56 and 0.06",
          (fr["contrast@2"]["rad"]["footprint_link_score"], fr["color_var@2"]["rad"]["footprint_link_score"]),
          (0.56, 0.06), 5e-3, text="footprint-link scores $d^M/\\varepsilon^M$ of 0.56 and 0.06")
    tonly = sorted("%s|%s|%s" % (part, t, c) for part in ("registered", "frontier") for t in sc[part] for c, s in sc[part][t].items() if s["valid_target_only"])
    check("screens: two controls pass under a target-only (A2)", "/".join(tonly),
          "frontier|color_var@2|rad/registered|color_var@2|rad16",
          text="Counting the target proxy alone, two controls at the highest variegation dose would pass all three screens")
    check("screens: they move the contrast proxy by 47 and 58%",
          (100 * sc["registered"]["color_var@2"]["rad16"]["offtarget_proxy_shift"], 100 * fr["color_var@2"]["rad"]["offtarget_proxy_shift"]),
          (47, 58), 0.5, text="move the contrast proxy by 47 and 58% on average")
    check("screens: target proxy of every constructed control moves < 0.15%",
          max(s["target_proxy_shift"] for part in ("registered", "frontier") for t in sc[part] for s in sc[part][t].values()) < 0.0015, True,
          text="less than 0.15% of its unedited value", where="supp")
    eo = sc["edit_offtarget"]
    check("C3b proxies: edit off-target CIELAB 25 and 78% (contrast)",
          (100 * eo["contrast@0"]["median_rel_cielab"], 100 * eo["contrast@2"]["median_rel_cielab"]), (25, 78), 0.5,
          text="by a median of 25 and 78% at the two contrast doses")
    check("C3b proxies: edit off-target CIELAB 16 and 51% (variegation)",
          (100 * eo["color_var@0"]["median_rel_cielab"], 100 * eo["color_var@2"]["median_rel_cielab"]), (16, 51), 0.5,
          text="16 and 51% at the two variegation doses")
    rg2 = sc["registered"]
    check("off-target: CIELAB 20.0 and 26.4% at the highest contrast dose",
          (100 * rg2["contrast@2"]["rad4"]["offtarget_median_rel_control"], 100 * rg2["contrast@2"]["rad16"]["offtarget_median_rel_control"]),
          (20.0, 26.4), 0.05, text="by 20.0 and 26.4%")
    check("off-target: contrast proxy 18 and 47% at the highest variegation dose",
          (100 * rg2["color_var@2"]["rad4"]["offtarget_proxy_shift"], 100 * rg2["color_var@2"]["rad16"]["offtarget_proxy_shift"]),
          (18, 47), 0.5, text="change the contrast proxy by 18 and 47%")
    check("screens: proxy shift 20 and 29%", (100 * fr["contrast@2"]["rad"]["proxy_shift"], 100 * fr["color_var@2"]["rad"]["proxy_shift"]),
          (20, 29), 0.5, text="mean proxy shift is 20 and 29%")
    check("screens: 30 and 71% of the edit's shift", (100 * fr["contrast@2"]["rad"]["proxy_ratio"], 100 * fr["color_var@2"]["rad"]["proxy_ratio"]),
          (30, 71), 0.5, text="30 and 71% of the concept edit's own shift")
    _hs = (("registered", ("hue4", "hue16")), ("frontier", ("hue",)))
    check("screens: every low-dose hue control passes (A2)",
          all(sc[pp][t][c]["A2"] for pp, cc in _hs for t in ("contrast@0", "color_var@0") for c in cc), True)
    check("screens: hue misses balance by 9.2 to 9.4 at the lowest contrast dose",
          rng([sc[pp]["contrast@0"][c]["footprint_link_score"] for pp, cc in _hs for c in cc]), (9.2, 9.4), 0.05,
          text="factors of 9.2 to 9.4 and 10.8 to 10.9")
    check("screens: and by 10.8 to 10.9 at the lowest variegation dose",
          rng([sc[pp]["color_var@0"][c]["footprint_link_score"] for pp, cc in _hs for c in cc]), (10.8, 10.9), 0.05,
          text="are 9.2 to 9.4 and 10.8 to 10.9", where="supp")
    check("screens: concept edit moves the contrast proxy by 16% on the images of the two candidates",
          (100 * sc["registered"]["color_var@2"]["rad16"]["edit_offtarget_proxy_shift"],
           100 * sc["frontier"]["color_var@2"]["rad"]["edit_offtarget_proxy_shift"]), (16, 16), 0.5,
          text="against 16% for the concept edit on the same images", where="supp")
    rg = sc["registered"]
    check("screens: off-target 4.9 and 5.3% at the lowest contrast dose",
          (100 * rg["contrast@0"]["rad4"]["offtarget_median_rel_control"], 100 * rg["contrast@0"]["rad16"]["offtarget_median_rel_control"]),
          (4.9, 5.3), 0.05, text="4.9 and 5.3% in CIELAB", where="supp")
import hashlib as _hl
_man = os.path.join(os.path.dirname(RES), "results", "c62_manifest.sha256")
if os.path.exists(_man):
    _bad = [ln.split()[1] for ln in open(_man).read().splitlines() if ln.strip() and
            _hl.sha256(open(os.path.join(os.path.dirname(RES), ln.split()[1]), "rb").read()).hexdigest() != ln.split()[0]]
    check("route-test manifest: every listed file matches its SHA-256", len(_bad), 0, text="c62_manifest.sha256", where="supp")
d = load("c63_route_exploratory_inv_retrained")
if d:
    check("exploratory (second run): nine differences analyzed, none decided", "%d %d" % (d["n_analysed"], d["summary"]["indeterminate"]), "9 9",
          text="None of the nine differences is decided on these images", where="supp")
    check("exploratory (second run): main text says no pair is decided", d["summary"]["exceeds"] + d["summary"]["within"], 0,
          text="decides no pair at the low doses")
    check("exploratory (second run): critical value 2.86", d["crit_simultaneous"], 2.86, 5e-3, text="(critical value 2.86)", where="supp")
d = load("c63_route_exploratory")
if d:
    q = d["components"]["slim|contrast@0|D_rad4-rad16"]
    check("first run exploratory: Slim persists at 2.32 on 49 images", (q["est"], q["n_images"]), (2.32, 49), 5e-3, text="2.32 logits on 49 images", where="supp")
    check("first run exploratory: decision exceeds", q["decision_sim"], "exceeds")
d = load("c62_frontier")
if d:
    tks = ("contrast@0", "contrast@2", "color_var@0", "color_var@2")
    fine = [sum(r["targets"][t]["rad"]["fine_feasible"] for r in d["images"]) for t in tks]
    check("frontier: radial within 0.1% on 13 and 15 images at the high doses, none at the low doses", "/".join(map(str, fine)), "0/13/0/15",
          text="for 13 and 15 of the 78 images at the two high doses")
    h10 = sum(1 for t in tks for r in d["images"] if r["targets"][t]["hue"]["neutral_converged"]
              and max(abs(r["targets"][t]["hue"]["rel_dE"]), abs(r["targets"][t]["hue"]["rel_hf"])) <= 0.10)
    check("frontier: hue never within 10%", h10, 0, text="no hue control came within 10%")
    _inex = sorted(r["targets"][t][m]["max_rel_mag"] for r in d["images"] for t in tks for m in ("rad", "hue")
                   if (r["targets"][t][m].get("max_rel_mag") or 0) > 1e-5)
    check("frontier: two inexact hue solutions, 0.007 and 0.22%",
          "%d: %.3f/%.2f" % (len(_inex), 100 * _inex[0], 100 * _inex[-1]), "2: 0.007/0.22",
          text="miss the magnitude map by 0.007 and 0.22%", where="supp")
    _con = load("c62_construction")
    if _con:
        _pre = sum(1 for t in tks for r in _con["images"] for c in ("rad4", "rad16")
                   if max(abs(r["targets"][t]["controls"][c]["rel_dE"]), abs(r["targets"][t]["controls"][c]["rel_hf"])) <= 0.001)
        check("frontier: pre-specified radial controls never within 0.1%", _pre, 0,
              text="The pre-specified radial controls never did")
        _g2 = [sum(r["targets"][t]["controls"][c]["g2"] for r in _con["images"]) / len(_con["images"])
               for t in ("contrast@2", "color_var@2") for c in ("rad4", "rad16", "hue4", "hue16")]
        check("high doses: seven of eight controls fail G2 on most images", sum(x < 0.5 for x in _g2), 7,
              text="seven of the eight controls fail the independent check of neutrality on most images")
d = load("c56_simultaneous36")
if d:
    check("36-component critical value 3.12 (headline band)", d["crit_36"], 3.12, 5e-3, text="1.96 to 3.12")
    check("every cell exceeds every tolerance below 0.042 (36 components)", d["frontier_36"], 0.042, floor=True,
          text="every tolerance below 0.042 probability units")
    check("36-component frontier 0.0422 (supp)", d["frontier_36"], 0.0422, 5e-5, text="0.0422", where="supp")
    check("36 band: 17 of 18 cells exceed jointly at 0.05", d["magnitude_exceed_36"], 17, text="17 of the 18 cells exceed jointly")
    check("36 band signed: 11 exceed / 1 within / 6 indet",
          "%d/%d/%d" % (d["signed_counts_36"]["exceeds"], d["signed_counts_36"]["within"], d["signed_counts_36"]["indet."]),
          "11/1/6", text="exceeds in 11 cells, is within tolerance in 1 and is indeterminate in 6", where="supp")
    ri = d["components"]["r50-Inv/mal/sgn"]["sim36"]
    check("R50 Inv malignant signed sim36 [-0.040, +0.029]", tuple(ri), (-0.040, 0.029), 5e-4,
          text="[−0.040, +0.029]", where="supp")
    check("36 band critical value agrees with c40 (2.96 for 18)", d["crit_18_magnitude_only"], 2.96, 5e-3)
if cells:
    mlb = min(c["ci"][0] for c in cells.values())
    check("marginal bound 0.063", mlb, 0.063, 5e-4, text="put the bound at 0.063")
    check("0.063 is 1.25 times the tolerance", mlb / 0.05, 1.25, 5e-3, text="1.25 times")

d = load("c32_absdp_patient_ci")
if d:
    c = d["cells"]
    check("cached run: Inv benign 0.091 [0.063, 0.124]",
          (c["Inv"]["benign"]["mean"], c["Inv"]["benign"]["ci"][0]), (0.091, 0.063), 5e-4,
          text="0.091 with interval [0.063, 0.124]", where="supp")
    check("cached run: Slim malignant 0.762 [0.707, 0.808]",
          (c["Slim"]["malignant"]["mean"], c["Slim"]["malignant"]["ci"][1]), (0.762, 0.808), 5e-4,
          text="0.762 with interval [0.707, 0.808]", where="supp")

d = load("c50_overlay_seed")
if d:
    sp = [v["spread_pct_of_seed42"] for v in d["cells"].values()]
    check("seven realizations move magnitudes 3.2 to 24.9%", rng(sp), (3.2, 24.9), 0.05,
          text="3.2 to 24.9%")
    check("no verdict changes across realizations", all(v["verdict_stable"] for v in d["cells"].values()), True)
    check("smallest lower bound across realizations 0.058",
          min(v["lower_bound_min"] for v in d["cells"].values()), 0.058, floor=True, text="smallest marginal lower bound is 0.058")
    check("seven realizations", len(d["seeds"]), 7)
    _pc = d["pixel_change_frac"]
    _spr = max(max(_pc[q][e] for q in _pc) - min(_pc[q][e] for q in _pc) for e in ("hair_light", "hair", "hair_heavy", "ruler"))
    check("altered-pixel fractions vary by at most 0.06 percentage points", 100 * _spr, 0.065, ceil=True,
          text="at most 0.06 percentage points", where="supp")

d = load("c42_positive_control")
if d:
    _ri = d["models"]["r50"]["Inv"]["ruler"]
    check("ResNet-50 Inv within tolerance under the ruler alone",
          "/".join(_ri[g]["verdict"] for g in ("all", "malignant", "benign")), "equivalence/equivalence/equivalence",
          text="the ResNet-50 checkpoint is within tolerance under the ruler")
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
    check("dose grid: 55 within tolerance", verd.count("equivalence"), 55, text="55 of 144 cells", where="supp")
    check("dose grid: 19 indeterminate", verd.count("indeterminate"), 19, text=["19 are indeterminate"], where=["paper", "supp"])
    check("first exceedance 0.34 to 2.37%", rng(first), (0.34, 2.37), 5e-3, text="0.34 to 2.37%")
    idv = [v["verdict"] for bk in d["models"] for mk in d["models"][bk]
           for v in d["models"][bk][mk]["identity"].values()]
    check("identity edit within tolerance in all eighteen cells", idv.count("equivalence"), 18)
    check("non-identity: 37 of 126 within tolerance",
          (verd.count("equivalence") - idv.count("equivalence"), len(verd) - len(idv)), (37, 126),
          text="Of the 126 other cells, 37 fall within tolerance")
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
    lg = [d["models"][m]["edits"][e]["all"]["abs_dlogit"]["mean"]
          for m in d["models"] for e in ("art:vignette", "art:jitter")]
    check("photometric edits move the malignant logit by at most 1.7 (mean |change|, 183 images)", max(lg), 1.7, ceil=True,
          text="by at most 1.7", where="supp")
    hl = [d["models"][m]["edits"]["art:hair"]["all"]["abs_dlogit"]["mean"] for m in d["models"]]
    check("main hair edit moves it by up to 13.8 (mean |change|, 183 images)", max(hl), 13.8, 0.05,
          text="by up to 13.8 in mean absolute value over the 183 images", where="supp")
    resp = [d["models"][m]["edits"][t]["all"]["dlogit"]["mean"] for m in d["models"]
            for t in ("cpt:contrast@0", "cpt:contrast@2", "cpt:color_var@0", "cpt:color_var@2")]
    mx = max(abs(x) for x in resp)
    check("largest concept response 1.06 logits", mx, 1.06, 5e-3,
          text="At the largest, 1.06 logits", where="supp")
    check("component-range width 26.57 logits", 2 * B - mx, 26.57, 5e-3, text="26.57 logits", where="supp")
    check("width is 96.1% of the 27.63-logit range", 100 * (2 * B - mx) / (2 * B), 96.1, 0.05,
          text=["96.1%", "27.63-logit"])
    check("concept responses run −0.06 to +1.06", rng(resp), (-0.06, 1.06), 5e-3,
          text="−0.06 to +1.06", where="supp")

d = load("c19_saturation_mde")
if d:
    check("clipped logit bound B = 13.8155", d["clip_bound"], 13.8155, 5e-5, text="13.8155")
    check("87.1% of hair-edit predictions at the bound",
          100 * d["saturation_by_edit"]["art:hair"]["fraction"], 87.1, 0.05, text="87.1%")
    check("16.3% of edited predictions at the bound",
          100 * d["saturation"]["ALL models, edited only"]["fraction"], 16.3, 0.05, text="16.3%",
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
    check("minimum detectable effect 0.48 to 1.79 logits", rng(mde), (0.48, 1.79), 5e-3,
          text="0.48 to 1.79")

d = load("c20_numerical_repro")
if d:
    cf = d["configs"]
    diffs = [abs(cf[c][m] - cf["bf16_per_image"][m]) for c in ("fp32_per_image", "bf16_chunk32")
             for m in cf["bf16_per_image"]]
    check("numerical check on the 183-image cohort", d["n_images"], 183)
    check("pipeline noise floor 0.303 logits", max(diffs), 0.303, 5e-4, text="0.303 logits", where=["paper", "supp"])
    pc = d["pct_change_vs_reference"]
    check("fp32 changes the aggregate by −2.8 to +2.8%", rng(list(pc["fp32_per_image"].values())),
          (-2.8, 2.8), 0.05, text="−2.8 to +2.8%", where="supp")
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
          (0.05, 99), 0.6, text="0.05 but moves the concept proxies by 99%")
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
    check("binding pair 10.24 logits", wp["abs_diff_logits"], 10.24, 5e-3, text="10.24 logits")
    check("at 7.24 scored units", wp["distance_tolerance_units"], 7.24, 5e-3, text="7.24 scored units")
    check("L^g >= 1.41", L["implied_L_binding"], 1.41, 5e-3, text="= 1.41$")
    check("dropping the binding pair gives 1.33", L["implied_L_second_largest"], 1.33, 5e-3,
          text="leaves *L*ᵍ ≥ 1.33", where="supp")
    check("conservative floor 0.96", L["implied_L_conservative_interval_ends"], 0.96, 5e-3, text="gives *L*ᵍ ≥ 0.96", where="supp")
    check("median implied constant 0.71", L["implied_L_median"], 0.71, 5e-3, text="over the twenty-one pairs is 0.71", where="supp")
    check("quartiles 0.34 and 0.91", tuple(L["implied_L_quartiles"]), (0.34, 0.91), 5e-3,
          text="quartiles 0.34 and 0.91", where="supp")
    check("twenty-one Lipschitz comparisons", L["n_pairs"], 21)
    hw = L["footprint_only_half_width_at_binding_L"]
    check("half-widths 0.66, 1.00, 1.01, 4.82",
          "%.2f, %.2f, %.2f, %.2f" % (hw["color_var@0"], hw["contrast@2"], hw["contrast@0"], hw["color_var@2"]),
          "0.66, 1.00, 1.01, 4.82", text="0.66, 1.00, 1.01 and 4.82", where="supp")
    be = L["breakeven_kappa_Lh"]
    check("full-width break-even 0.117 to 0.154 at three targets, color_var@2 vacuous",
          "%.3f, %.3f, %.3f, %.3f" % (be["contrast@2"], be["contrast@0"], be["color_var@0"], be["color_var@2"]),
          "0.117, 0.138, 0.154, 0.000", text=["at most 0.12 to 0.15 logits per scored unit", "below 0.117 to 0.154 logits per scored unit"], where=["paper", "supp"])
    check("break-even uses full width (hw <= range/2)",
          be["color_var@0"], (L["score_range_logits"] / 2 - hw["color_var@0"]) / pt["color_var@0"]["total_concept_part"], 1e-9)
    check("break-even is 9.2 to 12.1 times below the floor",
          (L["implied_L_binding"] / max(be[t] for t in ("contrast@2", "contrast@0", "color_var@0")),
           L["implied_L_binding"] / min(be[t] for t in ("contrast@2", "contrast@0", "color_var@0"))), (9.2, 12.1), 0.05,
          text="9.2 to 12.1 times smaller", where="supp")
    check("only the color_var@2 full width, 9.65, exceeds the 3.33 span",
          "%.2f, %.2f, %.2f %s" % (2 * hw["contrast@2"], 2 * hw["contrast@0"], 2 * hw["color_var@2"],
                                   [t for t in hw if 2 * hw[t] > L["score_range_logits"]]), "2.00, 2.03, 9.65 ['color_var@2']",
          text=["9.65 logits for color variegation at the highest dose", "full widths of 2.00 and 2.03 logits, below the range"],
          where=["paper", "supp"])
    _c12 = load("c12_matched_control")
    if _c12:
        _sc = [_c12["models"][m]["dcace"][t]["u:hue"]["mean"] for m in _c12["models"]
               for t in ("contrast@2", "color_var@2", "contrast@0", "color_var@0")]
        check("span recomputed from c12 equals c55's", L["score_range_logits"], max(_sc) - min(_sc), 1e-12)
        check("corrected scores span 3.33 logits, 0.96 to 4.29",
              "%.2f/%.2f/%.2f" % (max(_sc) - min(_sc), min(_sc), max(_sc)), "3.33/0.96/4.29",
              text="span 3.33 logits, from 0.96 to 4.29")
    check("interaction price eps_q >= 5.12", wp["abs_diff_logits"] / 2, 5.12, 5e-3, text="5.12$ logits")

d = load("c34_ctrl_vs_ctrl")
if d:
    su = d["summary"]
    diffs = [p["diff"] for p in d["pairs"]]
    check("twenty-one control comparisons", su["n_comparisons"], 21, text="twenty-one model-by-pair")
    check("all twenty-one exclude zero (patient)", su["excluding_zero_patient"], 21,
          text="all twenty-one patient-clustered intervals exclude zero")
    check("all twenty-one exclude zero (image)", su["excluding_zero_image"], 21)
    check("median disagreement 9.47 logits", statistics.median(abs(x) for x in diffs), 9.47, 5e-3,
          text="median of 9.47 logits")
    check("disagreement 2.26 to 14.43", rng([abs(x) for x in diffs]), (2.26, 14.43), 5e-3,
          text="2.26 to 14.43")
    _rank = {"globrot_m": 0, "noise_m": 1, "pixperm_m": 2}
    _ord = all((p["diff"] > 0) == (_rank[p["a"]] < _rank[p["b"]]) for p in d["pairs"])
    check("every comparison follows globrot_m > noise_m > pixperm_m", _ord, True,
          text="Every comparison follows one order of the three controls")
    check("the order as printed in S6", _ord, True, text="`globrot_m` above `noise_m` above `pixperm_m`", where="supp")
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
    check("globrot R^2 −1,631.9 / −66.1 / −204.4",
          "%.1f/%.1f/%.1f" % (r2["erm"]["globrot"], r2["slim"]["globrot"], r2["causal"]["globrot"]),
          "-1631.9/-66.1/-204.4", text="−1,631.9 for ERM, −66.1 for Slim and −204.4", where="supp")
    check("LOFO fit on 58 common images", d["models"]["erm"]["g_fit"]["n_img"], 58, text="58 images",
          where="supp")

d = load("c45_bracketing")
if d:
    su = d["summary"]
    check("balance+overlap controls bracket no cell", su["A1+A3"]["cells_bracketed"], 0)
    check("whole library brackets 7 of 12", su["all"]["cells_bracketed"], 7, text="7 of the 12")
    check("the whole library is the 15 constructed controls", str(su["all"]["set_size_range"]), "[15, 15]",
          text="All 15 constructed controls together bracket both in 7 of the 12")
    _bal = [v for t in d["targets"].values() for v in t["sets"]["A1+A3"]["models"].values()]
    check("balanced controls bracket the slope in 2 cells, the intercept in none",
          "%d/%d" % (sum(v["alpha_bracketed"] for v in _bal), sum(v["beta_bracketed"] for v in _bal)), "2/0",
          text="bracket the slope in two cells but the intercept in none", where="supp")
    lo, hi = su["all"]["edge_control_smd_range"]
    check("bracketing controls 4.4 to 21 times outside tolerance", (lo / 0.1, hi / 0.1), (4.4, 21.0), 0.06,
          text="4.4 to 21 times")

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
    check("CAV accuracy 0.84 to 0.94", rng(accs), (0.84, 0.94), 6e-3, text="0.84 to 0.94")
    check("significant in 7 of 12 cells", "%d of %d" % (sum(flags), len(flags)), "7 of 12", text="7 of 12")

d = load("c10_multiplacebo")
if d:
    check("multi-placebo sample: 57 of 83 malignant audit images", d["n_common"], 57,
          text="57 of 83 malignant audit images")
    check("multi-placebo intervals are patient-clustered", d.get("resampling_unit"), "patient",
          text="with the patient as the resampling unit", where="supp")
    check("multi-placebo: 29 patients", d.get("n_patients_common"), 29, text="from 29 patients")
    a = {m: {k: v["A"] for k, v in mv["A_corr_by_placebo"].items()} for m, mv in d["models"].items()}
    check("Slim −0.34 under flip", a["slim"]["flip"], -0.34, 5e-3, text="−0.34")
    check("Slim +5.14 under noise", a["slim"]["noise"], 5.14, 5e-3, text="+5.14")
    flips = sum(1 for m in a if min(a[m].values()) < 0 < max(a[m].values()))
    check("sign changes for one of three models", flips, 1, text="changes sign for one of the three")
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

# ---- checks added with the number audit of the supplement ----
_a, _b = load("c40_simultaneous"), load("c30_disjoint_audit")
if _a and _b:
    _SG = {"all": "all", "mal": "malignant", "ben": "benign"}
    _mx = 0.0
    for _k, _v in _a["disjoint_18"]["cells"].items():
        _mdl, _sg = _k.split("/"); _bb, _mm = _mdl.split("-")
        for _i in (0, 1):
            _mx = max(_mx, abs(_v["ci_marginal"][_i] - _b["models"][_bb][_mm][_SG[_sg]]["ci"][_i]))
    check("marginal endpoints of the two runs differ by at most 0.0043", _mx, 0.00435, ceil=True,
          text="differ by at most 0.0043", where="supp")
_c51 = load("c51_independent_rebuild")
if _c51:
    _cg = _c51["disagreement_diagnosis"]["cross_grid"]
    check("reported checkpoint gives 0.123 on both cohorts",
          "%.3f/%.3f" % (_cg["ckpt=here|cohort=here"]["magnitude"], _cg["ckpt=here|cohort=other"]["magnitude"]), "0.123/0.123",
          text="gives 0.123 on both cohorts", where="supp")
_c52 = load("c52_overlay_perceptual")
if _c52:
    check("between-lesion scale keeps 1,991 of 2,000 draws", "%d/%d" % (_c52["n_pairs"], _c52["between_lesion"]["n_pairs"]),
          "2000/1991", text="keeps the 1,991 that join two different images", where="supp")
_c25m = load("c25_screen_mal")
if _c25m:
    _D = _c25m["design"]
    check("S6: noise_m fails neutrality at both top doses, 20 and 18%",
          "%.0f/%.0f" % (100 * _D["contrast@2"]["noise_m"]["proxy_shift"], 100 * _D["color_var@2"]["noise_m"]["proxy_shift"]),
          "20/18", text="with proxy shifts of 20 and 18%", where="supp")
    check("S6: pixperm_m overlap 92% at the contrast top dose", round(100 * _D["contrast@2"]["pixperm_m"]["positivity"]), 92,
          text="by overlap, at 92%", where="supp")
    _hp = [t for t in _D if _D[t]["hue_m"]["proxy_shift"] <= 0.10 and _D[t]["hue_m"]["proxy_ratio"] <= 0.15]
    check("S6: hue_m passes neutrality only at the contrast top dose", str(_hp), "['contrast@2']",
          text="`hue_m` passes the neutrality screen only at the contrast top dose", where="supp")
_c38 = load("c38_shrinkage_sensitivity")
if _c38:
    _nd = max(v.get("n_dropped", 0) for m in _c38["models"].values() for v in m.values() if isinstance(v, dict))
    check("at most 8 of 183 bounded observations in any one shrinkage fit", _nd, 8,
          text="at most 8 of 183 in any one fit", where="supp")

_c42b = load("c42_positive_control")
if _c42b:
    _ord = [l["name"] for l in _c42b["ladder"]]
    _fr = {l["name"]: 100 * l["change_frac"] for l in _c42b["ladder"]}
    _fe = {}
    for _bk in _c42b["models"]:
        for _mk in _c42b["models"][_bk]:
            _ed = _c42b["models"][_bk][_mk]
            _fe["%s-%s" % (_bk, _mk)] = _fr[next(e for e in _ord if _ed[e]["all"]["verdict"] == "exceedance")]
    _top = max(_fe.values())
    check("ResNet-50 Inv alone has the highest first-exceeding hair dose, 2.37%",
          "%s %.2f" % (",".join(k for k, v in _fe.items() if v == _top), _top), "r50-Inv 2.37",
          text="it first exceeds at 2.37% of altered pixels, a higher dose than for any other model")
_c62p = load("c62_preregistration")
if _c62p:
    check("cycle test uses 20,000 resamples", _c62p["criteria"]["boot"], 20000,
          text="the cycle test of Section IV-B use 20,000")
_c64b = load("c64_route_screens")
if _c64b:
    _tp = max(_c64b["frontier"][t]["rad"]["target_proxy_shift"] for t in ("contrast@2", "color_var@2"))
    _sh = min(_c64b["frontier"][t]["rad"]["offtarget_proxy_shift"] / (2 * _c64b["frontier"][t]["rad"]["proxy_shift"])
              for t in ("contrast@2", "color_var@2"))
    check("optimized radial controls: target proxy moves < 0.1%, off-target share of the mean shift > 99%",
          "%s %s" % (_tp < 1e-3, _sh > 0.99), "True True",
          text="almost all of it comes from the other concept's proxy")

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
