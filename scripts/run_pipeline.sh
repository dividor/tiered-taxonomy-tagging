#!/usr/bin/env bash
# Run everything, in order, stopping at the first failure: download the data, run the notebooks, launch and
# wait for the GPU jobs between them, and build the report.
#
#   scripts/run_pipeline.sh            # everything
#   scripts/run_pipeline.sh 08 09      # only the notebooks whose names start with these numbers
#
# Every model reply, embedding and finished training step is saved in the data folder, so a rerun skips what is
# done and pays for nothing twice. Each notebook is executed in place, so its outputs are kept, then trimmed.
set -euo pipefail
cd "$(dirname "$0")/.."

BEFORE_GPU=(02_baselines 03_bakeoff 04_funnel 05_tree_walk 06_evidence 07_rollup)
AFTER_GPU=(08_trained_models 09_paper_recipes 10_combined_method 11_second_attempt 12_published_benchmarks
  13_humbert_reproduction 14_like_for_like 15_humset_v2 16_paper_measure 17_humset_full_test)

wanted() {
  local notebook="$1"
  shift
  [ "$#" -eq 0 ] && return 0
  for prefix in "$@"; do
    [[ "$notebook" == "$prefix"* ]] && return 0
  done
  return 1
}

run_notebook() {
  echo "running $1"
  # No time limit per cell: notebook 17 makes about a day of LLM calls in one cell.
  uv run jupyter nbconvert --to notebook --execute --inplace "notebooks/$1.ipynb" \
    --ExecutePreprocessor.timeout=-1
  uv run python scripts/trim_notebook.py "notebooks/$1.ipynb"
}

if [ "$#" -eq 0 ]; then
  uv run python scripts/check_setup.py
  uv run python scripts/download_data.py
fi
wanted 01_test_sets "$@" && run_notebook 01_test_sets
# MultiEURLEX is readable once notebook 01 has converted it.
[ "$#" -eq 0 ] && uv run python scripts/embed_corpus.py
for notebook in "${BEFORE_GPU[@]}"; do
  if wanted "$notebook" "$@"; then run_notebook "$notebook"; fi
done
if [ "$#" -eq 0 ]; then
  uv run python scripts/train_local_models.py
  uv run python scripts/gpu_jobs.py
fi
for notebook in "${AFTER_GPU[@]}"; do
  if wanted "$notebook" "$@"; then run_notebook "$notebook"; fi
done
if [ "$#" -eq 0 ]; then
  uv run python scripts/export_label_browser.py
  uv run python scripts/make_report.py
fi
