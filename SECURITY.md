# Security Policy

## Reporting a vulnerability

If you find a security problem in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for it.
2. Open a [private security advisory](https://github.com/dividor/tiered-taxonomy-tagging/security/advisories/new)
   on this repository, with steps to reproduce.
3. Allow a reasonable time for a fix before public disclosure.

## What is scanned

Every commit runs, through pre-commit hooks (`.pre-commit-config.yaml`) and again in CI
(`.github/workflows/ci.yml`):

| Tool | Purpose |
|---|---|
| **Bandit** | Python static analysis; MEDIUM and HIGH severity findings fail the check |
| **detect-secrets** | Blocks credentials being committed (`.secrets.baseline` lists the audited false positives) |
| **Gitleaks** | Pattern-based secret scanning, over the whole history in CI |
| **pip-audit** (CI) | Known vulnerabilities in the locked Python dependencies |
| **CycloneDX** (CI) | A software bill of materials for the locked dependencies, kept with each CI run |
| **Dependabot** | Monthly dependency and GitHub Actions updates; minor and patch updates merge once CI passes |

## Credentials this project uses

- **Hugging Face:** the token from `hf auth login` (`~/.cache/huggingface/token`) or `HF_TOKEN`. It is used for
  Inference Providers, for Hugging Face Jobs and for the jobs dataset named by `TAXONOMY_JOBS_DATASET`. It is
  passed to each GPU job as a secret, never as an argument, and is never written to the repository, the data
  folder or a log.
- **Other LLM providers:** if you swap a model in `configs/experiment.yaml` for another LangChain provider, its key
  is read from the environment variable that provider's LangChain package uses. Keep it in `.env`, which is
  git-ignored.

## What the data folder holds

The data folder (`TAXONOMY_DATA_ROOT`) holds the test sets and every saved model reply, which include the text of
EU laws and humanitarian excerpts. It stays outside the repository. The GPU jobs' outputs go to the jobs dataset,
which `scripts/gpu_jobs.py` creates private.
