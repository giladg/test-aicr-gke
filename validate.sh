#!/bin/bash
# Full AICR empirical validation of the deployed GKE stack.
#
# Runs: GPU pre-check -> recipe -> snapshot -> deployment+conformance
# validation -> performance validation (bounded). Artifacts land in
# ./validation-results/, timestamped per run.
#
# Usage:            ./validate.sh
# Skip perf phase:  SKIP_PERF=1 ./validate.sh
set -u

ZONE=us-central1-a
PROJECT=aicr-test-504619
OUT=./validation-results
STAMP=$(date +%Y%m%d-%H%M)
mkdir -p "$OUT"

CLUSTER=$(pulumi stack output cluster_name 2>/dev/null)
if [ -z "${CLUSTER}" ]; then
    echo "ERROR: could not read cluster_name from the pulumi stack." >&2
    echo "Run from ~/dev/test-aicr-gke with the dev stack selected." >&2
    exit 1
fi

echo "== cluster: $CLUSTER"
gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT" >/dev/null

# --- Stage 1: GPU pre-check -------------------------------------------------
# The snapshot agent requests nvidia.com/gpu and would otherwise sit Pending
# for its whole timeout; fail fast with a useful message instead.
GPUS=$(kubectl get nodes -o jsonpath='{range .items[*]}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}' | grep -c '^[1-9]' || true)
if [ "$GPUS" -lt 1 ]; then
    echo "ERROR: no node advertises nvidia.com/gpu yet." >&2
    echo "Either the GPU node has not materialized (quota/capacity/pulumi up)" >&2
    echo "or the GKE driver install is still running (~5 min after node join)." >&2
    echo "Watch with:" >&2
    echo "  kubectl get nodes -o custom-columns='NODE:.metadata.name,GPUS:.status.allocatable.nvidia\\.com/gpu' -w" >&2
    exit 1
fi
echo "== GPU pre-check: $GPUS GPU node(s) ready"

# --- Stage 2: the full-fidelity recipe (matches the deployed ClusterStack) --
echo "== generating recipe (gke/h100/training/cos/kubeflow)"
aicr recipe --service gke --accelerator h100 --intent training \
    --os cos --platform kubeflow --output "$OUT/recipe-$STAMP.yaml" \
    > "$OUT/recipe-log-$STAMP.txt" 2>&1 || { echo "ERROR: recipe generation failed; see $OUT/recipe-log-$STAMP.txt" >&2; exit 1; }

# --- Stage 3: snapshot (deploys a short-lived agent Job on the GPU node) ----
echo "== capturing cluster snapshot (~2-5 min)"
aicr snapshot --require-gpu --os cos --output "$OUT/snapshot-$STAMP.yaml" \
    > "$OUT/snapshot-log-$STAMP.txt" 2>&1 || { echo "ERROR: snapshot failed; see $OUT/snapshot-log-$STAMP.txt" >&2; exit 1; }

# --- Stage 4: deployment + conformance phases (~10-15 min) ------------------
echo "== running deployment + conformance validation"
aicr validate --recipe "$OUT/recipe-$STAMP.yaml" --snapshot "$OUT/snapshot-$STAMP.yaml" \
    --phase deployment --phase conformance --fail-on-error=false \
    --output "$OUT/validation-report-$STAMP.yaml" \
    > "$OUT/validation-log-$STAMP.txt" 2>&1
echo "   deployment+conformance: $(grep -c 'status=passed' "$OUT/validation-log-$STAMP.txt" || true) passed, $(grep -c 'status=failed' "$OUT/validation-log-$STAMP.txt" || true) failed, $(grep -c 'status=skipped' "$OUT/validation-log-$STAMP.txt" || true) skipped"

# --- Stage 5: performance phase (H100 benchmarks; bounded) ------------------
if [ "${SKIP_PERF:-0}" = "1" ]; then
    echo "== performance phase skipped (SKIP_PERF=1)"
else
    echo "== running performance validation (bounded at 15m per job)"
    aicr validate --recipe "$OUT/recipe-$STAMP.yaml" --snapshot "$OUT/snapshot-$STAMP.yaml" \
        --phase performance --fail-on-error=false --timeout 15m \
        --output "$OUT/performance-report-$STAMP.yaml" \
        > "$OUT/performance-log-$STAMP.txt" 2>&1
    echo "   performance: $(grep -c 'status=passed' "$OUT/performance-log-$STAMP.txt" || true) passed, $(grep -c 'status=failed' "$OUT/performance-log-$STAMP.txt" || true) failed, $(grep -c 'status=skipped' "$OUT/performance-log-$STAMP.txt" || true) skipped"
fi

echo
echo "== done. Reports:"
ls -1 "$OUT"/*report-$STAMP.yaml 2>/dev/null
echo "Full logs alongside them in $OUT/."
