## What this changes

A short description, and the issue it closes if there is one.

## Type of change

- [ ] Bug fix
- [ ] New method or experiment
- [ ] Results rerun
- [ ] Documentation
- [ ] Refactoring, with no change to results
- [ ] Build or CI

## Checks

- [ ] `uv run pre-commit run --all-files` passes, with no suppressions or exclusions added
- [ ] `uv run pytest` passes, and new functions have unit tests
- [ ] A changed notebook was rerun with `scripts/run_pipeline.sh <number>`, and its pass marks were set before it ran
- [ ] `uv run python scripts/make_report.py` builds, and `outputs/report/index.html` is committed with the results it reads
- [ ] Notebook outputs hold no document text

## Results

If results changed: which numbers, by how much, and which model and provider produced them.
