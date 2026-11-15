# Causal_Audits_of_SLC_v2

Analysis code and cached results for a causal audit of skin-lesion classifiers. The audit has two tiers. The artifact tier asks whether a classifier's malignant score moves under performed acquisition edits (hair and ruler overlays), judged against a stated tolerance with a three-way interval rule. The concept tier asks whether a library of performed edits identifies the classifier's response to a clinical concept, decided by the connectivity of an audit graph. A pre-specified route test on held-out images, described at the end of this file, tests the working form behind that decision.

## Layout

```
code/          analysis scripts, shared modules, the runner and the verifier
  lib/         vendored training modules
results/       cached JSON and NPZ outputs, one per script, plus the frozen manifests
log/           one log per script; run_retrain_b3inv.log for the last re-run of the analyses;
               e2e_fresh_copy_run.txt for a run in a fresh directory
requirements.txt  exact package versions
checkpoints/   audited checkpoints, downloaded from the release below (not tracked)
paper/         the manuscript, read by the verifier (not included)
```

Model checkpoints are published as assets of the release [`checkpoints-v1.1`](https://github.com/dqtoan87/Causal_Audits_of_SLC_v2/releases/tag/checkpoints-v1.1), with a `SHA256SUMS` file. Download them into `checkpoints/` at the repository root:

```bash
mkdir -p checkpoints && cd checkpoints
gh release download checkpoints-v1.1 --repo dqtoan87/Causal_Audits_of_SLC_v2
sha256sum -c SHA256SUMS
```

The release holds the six audited classifiers (`dj_{erm,slim,causal}_{b3,r50}.pt`), the two PAD-UFES-20 classifiers (`pad_{erm,slim}.pt`) and the checkpoints of the selection analysis (`sel_{blind,leaky}_{erm,slim,causal}.pt`). Analyses that read only cached predictions run without them (see below).

In `checkpoints-v1.1` the EfficientNet-B3 Inv classifier (`dj_causal_b3.pt`) is selected only among epochs in which both invariance penalties are on (epoch 7). The release [`checkpoints-v1.0`](https://github.com/dqtoan87/Causal_Audits_of_SLC_v2/releases/tag/checkpoints-v1.0) keeps the checkpoint that the original selection rule saved at epoch 1, inside the penalty warm-up. It is needed only to reproduce the first run of the route test below; its other thirteen files are identical to those of `checkpoints-v1.1`.

## Setup

Python 3.12 with PyTorch, torchvision, scikit-learn, pandas, numpy, scipy, matplotlib and Pillow. `requirements.txt` pins the exact versions the reported numbers were produced with (PyTorch 2.10.0, CUDA 12.8 build). A CUDA device is needed only by the scripts that run a model over images.

The two datasets are public and are not redistributed here.

| Dataset | Expected at | Set the path in |
| --- | --- | --- |
| ISIC-2024 (SLICE-3D) | `<DATA_ROOT>/data_ISIC2024/train-metadata.csv`, and the images unpacked as `<DATA_ROOT>/data_ISIC2024/train-image/image/<isic_id>.jpg` | `DATA_ROOT` in `code/lib/train_XAI_new_v01.py`; `DATA_CSV` in `code/c1_tier1_ate.py`; `META` in `code/c8_backbone2_train.py`, `code/c29_disjoint_train.py` and `code/c35_selection_isolation.py`; `ISIC_META` in `code/verify_paper3.py` |
| PAD-UFES-20 | `<ROOT>/metadata.csv` and `<ROOT>/imgs_part_1..3/` | `ROOT` in `code/c46_padufes_artifact.py` |

These variables hold the absolute paths of the machine on which the results were produced. Replace them with your own paths before running. Every other path is resolved relative to the repository root, and the route test (`c62` to `c65`) reads the ISIC-2024 images through `DATA_ROOT`.

## Frozen inputs

`results/frozen_split.json`, `results/audit_cohort.json` and `results/padufes_split.json` record the patient-disjoint partition and the audit cohort by image identifier. They are read, never regenerated: fold assignment from a grouped stratified splitter is not stable across library versions, so rebuilding a split without retraining would silently break the patient-disjoint property. `frozen_split.py` and `audit_cohort.py` rebuild them only under an explicit `--rebuild` flag, and `audited_models.py` is the single place that resolves which checkpoint every analysis loads.

## Running

All commands are run from the repository root. Every script resolves its paths relative to its own location and writes into `results/`.

```bash
bash code/run_all.sh                                   # all analyses in dependency order, logs to log/
PY=/path/to/python setsid nohup bash code/run_all.sh > log/nohup.out 2>&1 < /dev/null &   # detached
```

`run_all.sh` audits the released PAD-UFES-20 checkpoints (`c46 --use-saved`), which is how the reported numbers were produced, and runs `c51`, `c58` and `c59` only when `OTHER` points at the reimplementation. It runs the analysis scripts in dependency order (`c25` before `c45`, `c34` before `c55`), regenerates the figures, and then calls the verifier. A failing script is logged with its traceback and the sequence continues. Training is not part of the run: the audited checkpoints are frozen and every reported number is about those weights. To retrain, run `c29_disjoint_train.py` and `c8_backbone2_train.py` first; retrained weights need not reproduce the reported magnitudes. `c29_disjoint_train.py` selects each Inv checkpoint among the epochs with both penalties on, and `C29_ONLY=dj_causal_b3` trains one job only. After any checkpoint changes, run `c44_cache_preds.py` before `run_all.sh`: it writes `results/c19_preds.npz`, which `c26`, `c32`, `c38` and the shrinkage figure read, and `run_all.sh` does not call it. `code/run_retrain_b3inv.sh` records how the EfficientNet-B3 Inv checkpoint was retrained and every analysis that reads it was run again, including the second run of the route test. `log/e2e_fresh_copy_run.txt` records a run of the released code and checkpoints in a fresh directory, compared file by file with `results/`.

Scripts that read only cached predictions and need neither a GPU nor raw images include `c26_hair_dose_prob`, `c32_absdp_patient_ci`, `c38_shrinkage_sensitivity`, `c41_within_patient`, `c45_partial_id`, `c47_heterogeneity`, `c54_synthetic_scm`, `c55_identification_graph` and `make_fig_shrinkage`.

## What each script produces

| Script | Output in `results/` | Computes |
| --- | --- | --- |
| `c1_tier1_ate.py` | `tier1_ate.json` | disease side: concept-to-malignancy effects by doubly robust AIPW |
| `c41_within_patient.py` | `c41_within_patient.json` | disease side: conditional logistic regression stratified by patient |
| `c47_heterogeneity.py` | `c47_heterogeneity.json` | common-odds-ratio check for the within-patient model |
| `c29_disjoint_train.py` | `c29_disjoint_split.json` | trains the EfficientNet-B3 trio under the frozen partition; Inv checkpoints are selected among epochs with both penalties on |
| `c8_backbone2_train.py` | checkpoints | trains the ResNet-50 trio |
| `c2_cace_artifact.py` | `tier2_cace_artifact.json` | artifact overlays and edit precomputation |
| `c30_disjoint_audit.py` | `c30_disjoint_audit.json` | artifact criterion on the audit cohort |
| `c40_simultaneous.py` | `c40_simultaneous.json` | max-statistic band over the 18 magnitude statistics (secondary) |
| `c56_simultaneous36.py` | `c56_simultaneous36.json` | max-statistic band over all 36 components, the headline band |
| `c57_stratified_band.py` | `c57_stratified_band.json` | the 36-component band with patients resampled within the design strata |
| `c60_design_coverage.py` | `c60_design_coverage.json` | coverage of the 36-component band over 500 redraws of the 100 benign audit images, with the fold's patients fixed |
| `c32_absdp_patient_ci.py` | `c32_absdp_patient_ci.json` | patient-clustered intervals for the artifact effect |
| `c21_patient_cluster.py` | `c21_patient_cluster.json` | patient-clustered recomputation of the model-side quantities |
| `c26_hair_dose_prob.py` | `c26_hair_dose_prob.json` | dose response on the probability scale |
| `c42_positive_control.py` | `c42_positive_control.json` | dose calibration of the decision rule |
| `c50_overlay_seed.py` | `c50_overlay_seed.json` | stability across independent overlay realizations |
| `c52_overlay_perceptual.py` | `c52_overlay_perceptual.json` | size of each overlay as a change to the image |
| `c19_saturation_mde.py` | `c19_saturation_mde.json` | clipping saturation and minimum detectable effects |
| `c20_numerical_repro.py` | `c20_numerical_repro.json` | fp32 and batch-composition reproducibility |
| `c35_selection_isolation.py` | `c35_selection_isolation.json` | checkpoint-selection effect separated from training-set size |
| `c46_padufes_artifact.py` | `c46_padufes_artifact.json`, `padufes_split.json` | the artifact audit on a second cohort |
| `c3_cace_concept.py` | `tier2_cace_concept.json` | concept edits and the soft lesion mask |
| `c7_extra_controls.py` | `c7_controls.json` | additional placebo controls |
| `c10_multiplacebo.py` | `c10_multiplacebo.json` | placebo families with patient-resampled intervals |
| `c12_matched_control.py` | `c12_matched_control.json`, `c12_preds.npz` | the 19-control library and per-image responses |
| `c13_fullpop.py` | `c13_fullpop.json` | edit responses on the full evaluation cohort |
| `c25_screen_full_sample.py` | `c25_screen_full.json`, `c25_screen_mal.json` | balance, neutrality and overlap screens for every control |
| `c34_ctrl_vs_ctrl.py` | `c34_ctrl_vs_ctrl.json` | control-versus-control diagnostic |
| `c45_partial_id.py` | `c45_bracketing.json` | whether the controls span the concept edit's response |
| `c55_identification_graph.py` | `c55_identification_graph.json` | the audit graph, its connectivity and the approximate-route bounds |
| `c54_synthetic_scm.py` | `c54_synthetic_scm.json` | synthetic response model with a known concept component |
| `c38_shrinkage_sensitivity.py` | `c38_shrinkage_sensitivity.json` | shrinkage slopes with clustered intervals |
| `c44_cache_preds.py` | `c19_preds.npz` | cached baseline and edited predictions |
| `c49_tcav.py` | `c49_tcav.json` | concept activation vectors on the same checkpoints and concepts |
| `c51_independent_rebuild.py` | `c51_independent_rebuild.json` | comparison against a separate reimplementation (see the note below) |
| `c58_rebuild_trace.py` | `c58_rebuild_trace.json` | layer-by-layer trace of the one cell on which the reimplementation disagrees (see the note below) |
| `c59_rebuild_generator.py` | `c59_rebuild_generator.json` | reruns the reimplementation's own loader, overlay generator, inference and checkpoints, and compares them with its cached predictions (see the note below) |
| `make_figures_split.py`, `make_fig_shrinkage.py` | `paper/figures/` | figures |

`c15_symmetry.py` and `audit_cohort.py`, `frozen_split.py`, `audited_models.py` are shared modules. `p2_source_guard.py` stops the data-reading figure scripts when the tree lacks its frozen manifest.

`c51_independent_rebuild.py`, `c58_rebuild_trace.py` and `c59_rebuild_generator.py` compare this pipeline with a separate clean-room reimplementation of the same audit. They read that reimplementation's files from a sibling directory, passed with `--other`, which is not part of this repository, so they do not run on their own here. Their cached outputs are included.

`verify_paper3.py` checks the load-bearing numbers of the manuscript against the files in `results/`. It reads the manuscript text from `paper/`, which is not part of this repository.

## Pre-specified route test on held-out images (c62 to c65)

`code/c62_route_test.py` asks whether exact routes can be built on held-out images under a footprint made of the four magnitude descriptors (altered-pixel fraction, mean absolute change, spatial support, lesion coverage), and whether the route sums of different controls agree on the audited EfficientNet-B3 classifiers. Every control keeps the concept edit's per-pixel change magnitude, so it matches that footprint exactly. The controls form a two-by-two design: change direction (radial, hue) by patch scale (4, 16 px).

The plan is in `results/c62_preregistration.json`. It fixes the locked image set (every malignant image of the selection fold: 78 images from 54 patients), the gates, the analysis sets, the score floor, the 0.5-logit margin and the decision rule. It was committed with the script in commit 40b6367, before any response to an edited image of the locked set was computed; GitHub's activity log records the push at 02:21:30 UTC on 2026-11-15. It is a locked plan in this repository, not an entry in an external registry. The evaluation phase stops if `code/c62_route_test.py` differs from the SHA-256 recorded in the plan.

```bash
PY=/path/to/python bash code/run_c62.sh   # construct (no model), then evaluate and frontier; logs in log/c62_*.log
```

| Script | Output in `results/` | Computes |
| --- | --- | --- |
| `c62_route_test.py` | `c62_preregistration.json`, `c62_construction.json`, `c62_evaluation.json`, `c62_frontier.json` | the pre-specified test: construction without a model, cycle test on the audited classifiers, six-descriptor frontier |
| `c63_route_exploratory.py` | `c63_route_exploratory.json`, `c63_route_scores.npz` | exploratory: each pair restricted to images on which both controls pass the independent check; stores every per-image response |
| `c64_route_screens.py` | `c64_route_screens.json` | exploratory and model-free: the observable screens under the six-descriptor footprint, and off-target concept changes |
| `c65_route_robustness.py` | `c65_route_robustness.json` | exploratory, from the stored responses: the margins at which verdicts change, other bootstrap seeds, leave one patient out, the two cross pairs |
| `run_retrain_b3inv.sh` | `c62_evaluation_inv_retrained.json`, `c63_route_exploratory_inv_retrained.json`, `c63_route_scores_inv_retrained.npz`, `c65_route_robustness_inv_retrained.json` | the second run: the unchanged plan, script and construction applied to the trio with the EfficientNet-B3 Inv checkpoint of `checkpoints-v1.1` |

`c63` scores the locked images again; `c64` and `c65` read no new model output.

**Results.** The plan was run twice, each time without deviation. Every control matched the magnitude footprint exactly, and all four controls passed the exactness and neutrality gate on 74 to 78 of the 78 images per target. The concept edit's response cancels from every difference of two route sums, so the cycle test compares the controls with each other. The decision uses a simultaneous 95% band over 48 differences and a 0.5-logit margin.

- First run (registered outcome; EfficientNet-B3 Inv checkpoint of `checkpoints-v1.0`; files without suffix): on the primary set (73 to 77 images, 52 to 54 patients) routes disagree in 3 of 12 model-by-target cells (ERM and Slim, all through the radial scale pair), agree in 2 (the Inv model at both low doses) and are indeterminate in 7. The two highest-dose disagreements are robust to the bootstrap seed and to leaving out any patient; the third (Slim, lowest contrast dose) is borderline.
- Second run (the unchanged plan applied after the Inv checkpoint was retrained; files with the suffix `_inv_retrained`): the score floor reads all three models, so the primary set changes to 44 to 47 images from 32 or 33 patients. Routes disagree in 3 of 12 cells, one per classifier (ERM and Inv at the highest contrast dose, Slim at the highest variegation dose), none agree and 9 are indeterminate. The ERM and Inv disagreements are robust to the bootstrap seed and to leaving out any patient; the Slim one depends on single patients.

A disagreement rejects the magnitude-only working form jointly with control neutrality; it does not test the six-descriptor footprint. Agreement would certify no route sum, because a bias shared by all routes cancels. Under the six-descriptor footprint and the library's screens, no constructed control passes all three screens.

## License

MIT, see `LICENSE`.
