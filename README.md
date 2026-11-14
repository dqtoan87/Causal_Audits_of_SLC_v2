# Causal_Audits_of_SLC

Analysis code and cached results for a causal audit of skin-lesion classifiers. The audit has two tiers. The artifact tier asks whether a classifier's malignant score moves under performed acquisition edits (hair and ruler overlays), judged against a stated tolerance with a three-way interval rule. The concept tier asks whether a library of performed edits identifies the classifier's response to a clinical concept, decided by the connectivity of an audit graph.

## Layout

```
code/          analysis scripts, shared modules, the runner and the verifier
  lib/         vendored training modules
results/       cached JSON and NPZ outputs, one per script, plus the frozen manifests
log/           one log per script and run_all.log for the last full run
requirements.txt  exact package versions
```

Model checkpoints are published as assets of the release [`checkpoints-v1.0`](https://github.com/dqtoan87/Causal_Audits_of_SLC/releases/tag/checkpoints-v1.0), with a `SHA256SUMS` file. Download them into `checkpoints/` at the repository root:

```bash
mkdir -p checkpoints && cd checkpoints
gh release download checkpoints-v1.0 --repo dqtoan87/Causal_Audits_of_SLC
sha256sum -c SHA256SUMS
```

The release holds the six audited classifiers (`dj_{erm,slim,causal}_{b3,r50}.pt`), the two PAD-UFES-20 classifiers (`pad_{erm,slim}.pt`) and the checkpoints of the selection analysis (`sel_{blind,leaky}_{erm,slim,causal}.pt`). Analyses that read only cached predictions run without them (see below).

## Setup

Python 3.12 with PyTorch, torchvision, scikit-learn, pandas, numpy, scipy, matplotlib and Pillow. `requirements.txt` pins the exact versions the reported numbers were produced with (PyTorch 2.10.0, CUDA 12.8 build). A CUDA device is needed only by the scripts that run a model over images.

The two datasets are public and are not redistributed here.

| Dataset | Expected at | Set the path in |
| --- | --- | --- |
| ISIC-2024 (SLICE-3D) | `<DATA_ROOT>/data_ISIC2024/` with `train-metadata.csv` and the image archive | `DATA_ROOT` in `code/lib/train_XAI_new_v01.py`, and `META` / `DATA_CSV` at the top of `c1`, `c8`, `c29`, `c35` |
| PAD-UFES-20 | `PAD-UFES-20/` with `metadata.csv` and `imgs_part_1..3/` | `ROOT` at the top of `code/c46_padufes_artifact.py` |

## Frozen inputs

`results/frozen_split.json`, `results/audit_cohort.json` and `results/padufes_split.json` record the patient-disjoint partition and the audit cohort by image identifier. They are read, never regenerated: fold assignment from a grouped stratified splitter is not stable across library versions, so rebuilding a split without retraining would silently break the patient-disjoint property. `frozen_split.py` and `audit_cohort.py` rebuild them only under an explicit `--rebuild` flag, and `audited_models.py` is the single place that resolves which checkpoint every analysis loads.

## Running

All commands are run from the repository root. Every script resolves its paths relative to its own location and writes into `results/`.

```bash
bash code/run_all.sh                                   # all analyses in dependency order, logs to log/
PY=/path/to/python setsid nohup bash code/run_all.sh > log/nohup.out 2>&1 < /dev/null &   # detached
```

`run_all.sh` runs the analysis scripts in dependency order (`c25` before `c45`, `c34` before `c55`), regenerates the figures, and then calls the verifier. A failing script is logged with its traceback and the sequence continues. Training is not part of the run: the audited checkpoints are frozen and every reported number is about those weights. To retrain, run `c29_disjoint_train.py` and `c8_backbone2_train.py` first; retrained weights need not reproduce the reported magnitudes.

Scripts that read only cached predictions and need neither a GPU nor raw images include `c26_hair_dose_prob`, `c32_absdp_patient_ci`, `c38_shrinkage_sensitivity`, `c41_within_patient`, `c45_partial_id`, `c47_heterogeneity`, `c54_synthetic_scm`, `c55_identification_graph` and `make_fig_shrinkage`.

## What each script produces

| Script | Output in `results/` | Computes |
| --- | --- | --- |
| `c1_tier1_ate.py` | `tier1_ate.json` | disease side: concept-to-malignancy effects by doubly robust AIPW |
| `c41_within_patient.py` | `c41_within_patient.json` | disease side: conditional logistic regression stratified by patient |
| `c47_heterogeneity.py` | `c47_heterogeneity.json` | common-odds-ratio check for the within-patient model |
| `c29_disjoint_train.py` | `c29_disjoint_split.json` | trains the EfficientNet-B3 trio under the frozen partition |
| `c8_backbone2_train.py` | checkpoints | trains the ResNet-50 trio |
| `c2_cace_artifact.py` | `tier2_cace_artifact.json` | artifact overlays and edit precomputation |
| `c30_disjoint_audit.py` | `c30_disjoint_audit.json` | artifact criterion on the audit cohort |
| `c40_simultaneous.py` | `c40_simultaneous.json` | max-statistic simultaneous bands over the 18 magnitude statistics |
| `c56_simultaneous36.py` | `c56_simultaneous36.json` | the same bands over all 36 components |
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
| `make_figures_split.py`, `make_fig_shrinkage.py` | `paper/figures/` | figures |

`c15_symmetry.py` and `audit_cohort.py`, `frozen_split.py`, `audited_models.py` are shared modules. `p2_source_guard.py` stops the data-reading figure scripts when the tree lacks its frozen manifest.

`c51_independent_rebuild.py` compares this pipeline with a separate clean-room reimplementation of the same audit. It reads that reimplementation's result files from a sibling directory, which is not part of this repository, so it does not run on its own here. Its cached output, `results/c51_independent_rebuild.json`, is included.

`verify_paper3.py` checks the load-bearing numbers of the manuscript against the files in `results/`. It reads the manuscript text from `paper/`, which is not part of this repository.

## License

MIT, see `LICENSE`.
