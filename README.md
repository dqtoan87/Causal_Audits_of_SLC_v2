# Causal_XAI_v3

Analysis code for a causal audit of skin-lesion classifiers. Two tiers are computed: the effect of measured lesion proxies on malignancy, estimated both by confounder adjustment and by a within-patient design, and the effect of performed image edits on a classifier, judged against a tolerance with a three-way interval rule.

## Layout

```
code/                37 analysis scripts and 4 driver scripts
  lib/               vendored training modules and the legacy checkpoints
results/             37 cached JSON and NPZ outputs
checkpoints/         six patient-disjoint models, the leaky and blind pair, two PAD-UFES models
logs/                run logs
```

## Setup

Python 3.13 with PyTorch, scikit-learn, pandas, numpy, scipy, matplotlib and Pillow.

A CUDA device is needed only by the scripts that run a model over images. Eleven of the analysis scripts read cached predictions instead and run on CPU.

The two datasets are public and are not redistributed here.

| Dataset | Expected at | Set the path in |
| --- | --- | --- |
| ISIC-2024 (SLICE-3D) | `$DATA_ROOT/data_ISIC2024/` with `train-metadata.csv` and `train-image/image/` | `DATA_ROOT` at the top of `code/lib/train_XAI_new_v01.py` |
| PAD-UFES-20 | `data/PAD-UFES-20/` with `metadata.csv` and `imgs_part_1..3/` | `ROOT` at the top of `code/c46_padufes_artifact.py` |

Check the setup before running anything else. Each of these loads a frozen manifest and asserts its invariants.

```bash
python3 code/frozen_split.py     # train / select / audit partition, stored by isic_id
python3 code/audit_cohort.py     # the 183 audit images, in order
python3 code/audited_models.py   # which checkpoints every analysis loads
```

## Running

All commands are run from this directory. Scripts resolve paths relative to it and write into `results/`.

### Cache only, no GPU and no raw images

```bash
python3 code/c41_within_patient.py --boot 0              # within-patient disease-side estimate
python3 code/c43_within_patient_diagnostics.py --perm 200  # stratum, influence, permutation refuter
python3 code/c45_partial_id.py                           # does the control library span the response
python3 code/c26_hair_dose_prob.py                       # dose contrasts on the probability scale
python3 code/c32_absdp_patient_ci.py                     # patient-clustered artifact intervals
python3 code/c38_shrinkage_sensitivity.py                # shrinkage slopes with clustered intervals
python3 code/c25_screen_full_sample.py                   # control validity screen, model free
python3 code/make_figures_v3.py                          # causal graph figure
python3 code/make_fig_shrinkage.py                       # shrinkage figure
```

### Needs a GPU and the raw images

The driver scripts run their steps in order and stop on the first failure.

```bash
bash code/run_frozen_pipeline.sh    # retrain six models, artifact criterion, dose calibration
bash code/run_concept_frozen.sh     # the concept tier on the frozen cohort
python3 code/c46_padufes_artifact.py --epochs 15 --boot 4000   # second cohort, trains two models
```

Retraining takes a few hours, so detach it:

```bash
setsid nohup bash code/run_frozen_pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &
```

Every long-running script writes a `.partial` file after each unit of work, so an interrupted run loses at most the unit in flight.

## What each script produces

| Script | Output | Computes |
| --- | --- | --- |
| `c1_tier1_ate.py` | `tier1_ate.json` | disease side by doubly-robust AIPW with cross-fitting |
| `c41_within_patient.py` | `c41_within_patient.json` | disease side by conditional logistic regression stratified by patient |
| `c43_within_patient_diagnostics.py` | `c43_within_patient_diagnostics.json` | stratum composition, leave-one-stratum-out influence, within-stratum permutation refuter |
| `c27_tier1_patient_weighted.py` | `c27_tier1_patient_weighted.json` | patient-weighted re-average of the same scores |
| `c37_no_site_sensitivity.py` | `c37_no_site.json` | refit with acquisition site removed |
| `c6_robustness.py` | `c6_robustness.json` | placebo and subset refuters |
| `c5_padufes_ate.py` | `tier1_padufes.json` | external replication of the size association |
| `c29_disjoint_train.py` | `c29_disjoint_split.json` | trains six models under the frozen partition |
| `c30_disjoint_audit.py` | `c30_disjoint_audit.json` | artifact criterion on the audit cohort |
| `c42_positive_control.py` | `c42_positive_control.json` | dose calibration of the criterion, six doses |
| `c40_simultaneous.py` | `c40_simultaneous.json` | max-statistic simultaneous bands |
| `c32_absdp_patient_ci.py` | `c32_absdp_patient_ci.json` | patient-clustered intervals for the artifact effect |
| `c26_hair_dose_prob.py` | `c26_hair_dose_prob.json` | dose response and the contrast between doses |
| `c46_padufes_artifact.py` | `c46_padufes_artifact.json`, `padufes_split.json` | the whole artifact audit on a second cohort |
| `c35_selection_isolation.py` | `c35_selection_isolation.json` | how much the measured magnitude moves across fits |
| `c12_matched_control.py` | `c12_matched_control.json`, `c12_preds.npz` | builds the 19-control library and caches per-image responses |
| `c25_screen_full_sample.py` | `c25_screen_full.json` | balance, neutrality and overlap for every control |
| `c34_ctrl_vs_ctrl.py` | `c34_ctrl_vs_ctrl.json` | control-versus-control falsification test |
| `c45_partial_id.py` | `c45_bracketing.json` | whether the controls span the concept edit's response function |
| `c13_fullpop.py` | `c13_fullpop.json` | full-population edit responses on the evaluation cohort |
| `c38_shrinkage_sensitivity.py` | `c38_shrinkage_sensitivity.json` | shrinkage slopes, clustered intervals, refit without clipped points |
| `c44_cache_preds.py` | `c19_preds.npz` | caches baseline and edited predictions for the cache-only analyses |
| `c19_saturation_mde.py` | `c19_saturation_mde.json` | clipping saturation and minimum detectable effects |
| `c20_numerical_repro.py` | `c20_numerical_repro.json` | fp32 and batch-composition reproducibility |
| `c21_patient_cluster.py` | `c21_patient_cluster.json` | patient-clustered recomputation of the model-side quantities |
| `c11_magnitude.py` | `c11_magnitude.json` | per-edit artifact magnitudes on each model's own thresholded set |


The remaining files under `code/` are shared modules rather than entry points. `c2_cace_artifact.py` defines the artifact overlays and the edit precomputation, `c3_cace_concept.py` the concept edits and the soft lesion mask, `c7_extra_controls.py` and `c10_multiplacebo.py` the placebo families, `c8_backbone2_train.py` the ResNet-50 variants, and `c15_symmetry.py` helper routines that the clustering scripts import. `audit_cohort.py`, `frozen_split.py` and `audited_models.py` resolve the frozen manifests and are described under Setup.

## Three invariants to preserve

**Splits are frozen, not rebuilt.** Fold assignment from a grouped stratified splitter is not stable across library versions. On this machine, rebuilding gives 199 audit patients where the original training run recorded 202, and the original fold membership was never stored. Every audited checkpoint is therefore trained under `results/frozen_split.json`, and `results/audit_cohort.json` fixes the evaluation images by identifier. Rebuilding a split without retraining silently breaks the patient-disjoint property.

**Checkpoints resolve in one place.** `code/audited_models.py` is the single source for which models every analysis loads, so training and evaluation images are verifiably separate rather than assumed to be.

**Every cached file has a generator.** Nothing in `results/` is an orphan. Do not add one.
