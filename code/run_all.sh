#!/bin/bash
# Independent re-run of every analysis the manuscript rests on.
#
# Everything resolves relative to this tree: the scripts derive their output directory as
# <parent of code/>/results, so nothing here can touch the results of any other tree.
#   results  -> ./results
#   logs     -> ./log        (one file per script, plus run_all.log)
#   models   -> ./checkpoints (own copies, not shared)
#
# Training is NOT part of this run. The six audited checkpoints are frozen and the paper's
# claims are about them, so re-running the analysis against those weights is what verifies
# the reported numbers. To retrain instead, run c29_disjoint_train.py and c8_backbone2_train.py
# first; they will overwrite ./checkpoints only, never the originals.

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
LOG="$BASE/log"
RES="$BASE/results"
PY="${PY:-python3}"
MASTER="$LOG/run_all.log"

mkdir -p "$LOG" "$RES" "$BASE/paper/figures"

# analysis scripts, in dependency order (c25 before c45, c34 before c55)
SCRIPTS=(
  c54_synthetic_scm
  c2_cace_artifact
  c3_cace_concept
  c7_extra_controls
  c10_multiplacebo
  c12_matched_control
  c13_fullpop
  c19_saturation_mde
  c20_numerical_repro
  c21_patient_cluster
  c25_screen_full_sample
  c26_hair_dose_prob
  c30_disjoint_audit
  c32_absdp_patient_ci
  c34_ctrl_vs_ctrl
  c35_selection_isolation
  c38_shrinkage_sensitivity
  c40_simultaneous
  c56_simultaneous36
  c57_stratified_band
  c42_positive_control
  c45_partial_id
  c46_padufes_artifact
  c49_tcav
  c50_overlay_seed
  c51_independent_rebuild
  c58_rebuild_trace
  c52_overlay_perceptual
  c55_identification_graph
  make_figures_split
  make_fig_shrinkage
)

{
  echo "=== run started $(date -Is) ==="
  echo "tree      : $BASE"
  echo "python    : $PY"
  echo "scripts   : ${#SCRIPTS[@]}"
  "$PY" -c "import torch;print('torch     :',torch.__version__,'| cuda',torch.cuda.is_available())" 2>&1
  echo
} > "$MASTER"

# Per-script arguments. c46 audits the released PAD-UFES-20 checkpoints rather than retraining them,
# which is how the reported numbers were produced. c51 and c58 compare against a separate
# reimplementation; they run only when OTHER points at its root directory.
args_for() {
    case "$1" in
        c46_padufes_artifact) echo "--use-saved" ;;
        c51_independent_rebuild|c58_rebuild_trace) [ -n "${OTHER:-}" ] && echo "--other $OTHER" ;;
    esac
}

ok=0; fail=0; failed=(); skipped=()
for s in "${SCRIPTS[@]}"; do
    if { [ "$s" = c51_independent_rebuild ] || [ "$s" = c58_rebuild_trace ]; } && [ -z "${OTHER:-}" ]; then
        skipped+=("$s"); printf '%-26s skipped (set OTHER to the reimplementation root)\n' "$s" >> "$MASTER"
        continue
    fi
    t0=$(date +%s)
    printf '%-26s start %s\n' "$s" "$(date +%H:%M:%S)" >> "$MASTER"
    # shellcheck disable=SC2046
    "$PY" "$HERE/$s.py" $(args_for "$s") > "$LOG/$s.log" 2>&1
    rc=$?
    t1=$(date +%s)
    if [ $rc -eq 0 ]; then
        ok=$((ok+1));   printf '%-26s   ok  %5ds\n' "$s" "$((t1-t0))" >> "$MASTER"
    else
        fail=$((fail+1)); failed+=("$s")
        printf '%-26s FAIL rc=%d %5ds  (see log/%s.log)\n' "$s" "$rc" "$((t1-t0))" "$s" >> "$MASTER"
        tail -n 15 "$LOG/$s.log" | sed 's/^/      | /' >> "$MASTER"
    fi
done

{
  echo
  echo "=== run finished $(date -Is) ==="
  echo "ok=$ok fail=$fail"
  [ $fail -gt 0 ] && echo "failed: ${failed[*]}"
  [ ${#skipped[@]} -gt 0 ] && echo "skipped: ${skipped[*]}"
  echo "results written: $(ls -1 "$RES" | wc -l) files"
} >> "$MASTER"

# compare what was produced against the numbers the manuscript states
"$PY" "$HERE/verify_paper3.py" > "$LOG/verify_paper3.log" 2>&1
echo "verification (paper3): see log/verify_paper3.log" >> "$MASTER"
echo "=== all done $(date -Is) ===" >> "$MASTER"
