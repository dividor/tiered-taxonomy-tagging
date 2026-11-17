# Tagging documents with a tiered taxonomy

Code, notebooks and report behind an astrobagel Medium post that tests ways of applying a tiered taxonomy, on
MultiEURLEX (government) and HumSet (humanitarian). See `README.md` for running it and `CONTRIBUTING.md` for the
checks.

## Commands

```bash
uv sync                               # environment
uv run pre-commit run --all-files     # everything CI's code-quality job runs
uv run pytest                         # unit tests; no data, network or credentials needed
uv run python scripts/check_setup.py  # data folder, disk, Hugging Face login, jobs dataset
scripts/run_pipeline.sh               # everything, in order; `scripts/run_pipeline.sh 10` for one notebook
uv run python scripts/make_report.py  # outputs/report/index.html from outputs/results
```

## Conventions

- **Data stays out of the repository.** Test sets, predictions, saved model replies and embeddings live in
  `TAXONOMY_DATA_ROOT`. The repository holds code, summaries and charts. The one exception: the report's example
  records in `outputs/results/label_browser.json`.
- **Models go through LangChain** (`src/tiered_taxonomy/models.py`), chosen in `configs/experiment.yaml`.
- **GPU runs are listed in `configs/experiment.yaml` under `gpu_jobs`** and launched by `scripts/gpu_jobs.py`;
  notebooks read their outputs at the commit it records. Nothing about a run is edited by hand.
- **Every reply is saved** in the data folder and reused. A malformed reply counts against its method; only
  transport failures are retried.
- **Pass marks go in a notebook's first cell before it is run.** A failed check raises; nothing falls back silently.
- **Claims come from a number a notebook printed.** Anything else is labelled a hypothesis.
- **Say when a method is reduced** from its paper, and which model and provider a result came from.
- `blog/` is git-ignored and never committed.
