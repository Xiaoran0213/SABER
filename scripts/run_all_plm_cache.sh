#!/usr/bin/env bash
# Build per-protein PLM caches for the benchmark datasets (x the sequence PLMs).
# Skips caches that already exist (pass --overwrite via EXTRA_ARGS to force).
#
# Requires: SABER_DATA_ROOT / SABER_PLM_ROOT configured (env or .saber_env),
# HF weights for the requested models (SABER_HF_CACHE), and a Python env with
# torch + transformers (override the interpreter with SABER_PLM_PYTHON).
set -u

PY="${SABER_PLM_PYTHON:-python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/scripts/_smoke_logs"
mkdir -p "$LOGDIR"

export HF_HUB_OFFLINE=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Manuscript datasets (see configs/benchmarks/saber_benchmark.yaml).
DATASETS="${SABER_PLM_DATASETS:-Specht_2019 Derks_2022 Huffman_2023_melanoma_HEK Woo_2021}"
PLMS="${SABER_PLMS:-prot_t5_xl esm2_650m esm2_150m prot_bert ankh_large}"

for PLM in $PLMS; do
  for DS in $DATASETS; do
    echo "### $PLM / $DS  START $(date +%T)"
    "$PY" -u "$ROOT/scripts/extract_plm_cache.py" --plm "$PLM" --dataset "$DS" \
        --device "${SABER_PLM_DEVICE:-cuda}" --batch-tokens 8192 $SABER_PLM_EXTRA_ARGS \
        > "$LOGDIR/${PLM}__${DS}.log" 2>&1
    ec=$?
    echo "### $PLM / $DS  EXIT=$ec  | $(tail -1 "$LOGDIR/${PLM}__${DS}.log" 2>/dev/null)"
  done
done

echo "===== ALL DONE ====="
echo "caches under \${SABER_PLM_ROOT:-$(cd "$ROOT" && python -c 'from saber.paths import plm_root; print(plm_root())')}:" >/dev/null
ls -la "$(cd "$ROOT" && "$PY" -c 'from saber.paths import plm_root; print(plm_root())')" 2>/dev/null | head -20 || true
echo "logs: $LOGDIR"
