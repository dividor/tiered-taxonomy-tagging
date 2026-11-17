# Contributing

## Requirements for every change

1. **Pre-commit hooks pass** (`uv run pre-commit run --all-files`). Do not use `git commit --no-verify`.
2. **Tests pass** (`uv run pytest`). New functions come with unit tests, which exercise the real code in
   `src/tiered_taxonomy` on small invented inputs and need no data, network or credentials.
3. **No suppressions.** Do not add `# noqa`, `# type: ignore`, `# nosec` or exclusions to pass a check: fix the code.
4. **No silent fallbacks.** A missing input or a failed check raises; it never quietly substitutes a default.
5. **No secrets or credentials.** `detect-secrets` and `gitleaks` run on every commit. Never commit `.env`.
6. **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/).

## Set-up

```bash
uv sync
cp .env.example .env        # set TAXONOMY_DATA_ROOT and TAXONOMY_JOBS_DATASET, and log in with `hf auth login`
uv run pre-commit install
uv run pytest
```

## What the hooks do

- **Formatting and linting:** `ruff format` and `ruff check` (line length 110)
- **Types:** `mypy` on `src/`
- **Security:** `bandit` on `src/` and `scripts/` (medium and high severity fail)
- **Secrets:** `detect-secrets` against `.secrets.baseline`, and `gitleaks`
- **File hygiene:** whitespace, line endings, YAML, JSON and TOML syntax, merge markers, files over 2 MB
- **Notebooks:** no long console output, such as progress bars, left in a committed notebook
- **Code metrics:** `scripts/quality/code_metrics.py` fails any file rated bad

| Metric | Good | Okay | Bad (fails) |
|---|---|---|---|
| Cyclomatic (per function, file maximum) | 1–10 | 11–20 | 21 and over |
| Cognitive (per function, file maximum) | 0–10 | 11–20 | 21 and over |
| Maintainability index (per file) | ≥ 20 | 10–19 | under 10 |

When a function is rated bad, split it into smaller named steps rather than adding an exclusion. If
`detect-secrets` flags a value that is not a secret, such as a pinned commit or a checksum, update the baseline with
`uv run detect-secrets scan --baseline .secrets.baseline` and check what it added.

## CI

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`, and can be started by hand:

- **code-quality:** `uv sync --locked`, then all the pre-commit hooks on every file
- **tests:** `uv run pytest`, then a check that the report rebuilt from the committed results is unchanged
- **security-scan:** `pip-audit` on the locked dependencies, `bandit`, and `gitleaks` over the history
- **sbom:** a CycloneDX bill of materials of the locked dependencies, kept with the run

Dependabot opens monthly update pull requests for Python dependencies and GitHub Actions; patch and minor updates
merge on their own once CI passes (`.github/workflows/dependabot-auto-merge.yml`).

## Notebooks

- Numbered in the order they run, and committed with their outputs; `scripts/run_pipeline.sh` runs them.
- Heavy work belongs in `src/tiered_taxonomy`, with unit tests; a notebook calls it and shows the result.
- Outputs are counts, shares and short labels, never document text.
- Pass marks go in a notebook's first cell before it is run.
- A notebook never falls back silently: if an input is missing or a check fails, it raises.
- Every model reply is saved in the data folder and reused; a malformed reply counts against its method, and only
  transport failures are retried.

## Results and the report

`scripts/make_report.py` builds `outputs/report/index.html` from `outputs/results` alone. Its findings are written
for particular results, so it raises if a rerun changes any fact they state: update the finding, with the numbers
that support it, rather than the check. Say when a method is reduced from its paper, and which model and provider
a result came from.
