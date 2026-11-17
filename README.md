# Tagging documents with a tiered taxonomy

**Read the blog post** (link to follow) · **[Explore the results report](https://dividor.github.io/tiered-taxonomy-tagging/)**

Which ways of tagging documents with labels from a tiered taxonomy actually work? This repository tests 23 of
them on two public test sets with labels assigned by people: EU laws tagged with EuroVoc, and humanitarian
excerpts tagged with HumSet's sectors, pillars and subpillars. It scores every method the same way, at the exact
label and tier by tier.

The methods range from prompting an LLM, through searching the taxonomy and walking down its tree, to small
models trained on the labelled documents. The most accurate was a combined method: several trained models each
suggest labels, and a scorer trained on documents they never saw decides which to keep.

| | Combined method | Published benchmark |
|---|---|---|
| EU laws, all 5,000 test laws, mean R-Precision | **0.640** | 0.484 (BERT, Chalkidis et al., 2021) |
| HumSet 2.0.0, all 12,896 test excerpts, mean macro F1 | **0.663** | 0.648 (HumBERT, Tamagnone et al., 2023) |

Methods that prompt an LLM came well behind the trained models, though on the EU laws they found more of the
labels that training data rarely or never used. The report has every result, a card for each method and the
papers behind it. Most methods are reduced forms of their papers; each module's docstring says how.

## The test sets

| | [MultiEURLEX](https://huggingface.co/datasets/coastalcph/multi_eurlex) (government) | [HumSet](https://huggingface.co/datasets/nlp-thedeep/humset) (humanitarian) |
|---|---|---|
| **Content** | 65,000 EU laws | 148,621 excerpts from crisis reports, in English, French and Spanish |
| **Taxonomy** | [EuroVoc](https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http://publications.europa.eu/resource/dataset/eurovoc): 7,372 labels in 8 tiers | sectors and pillars over subpillars, in two tiers |
| **Labels from** | the EU Publications Office | humanitarian analysts |
| **Licence** | CC BY 4.0 | Apache 2.0 |

`scripts/download_data.py` fetches both, pinned to one version and checked byte for byte.

## Running it yourself

Everything runs from one command, and every step it can skip, it does: each model reply, embedding and trained
model is saved in your data folder, so a rerun asks and pays for nothing twice.

### What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- A [Hugging Face](https://huggingface.co) account with a token that may call
  [Inference Providers](https://huggingface.co/docs/inference-providers), run
  [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) and write datasets. Jobs need a paid plan or
  credits.
- About 15 GB of free disk for the data folder, plus the Hugging Face cache for one embedding model.
- An Apple-silicon Mac or a machine with a GPU makes the local training faster; a plain CPU works, more slowly.

### What it costs

At November 2026 prices, as our run recorded them:

| Step | Cost | Time |
|---|---|---|
| LLM calls (open models through Hugging Face Inference Providers) | about $100, two thirds of it notebook 17 | about a day, mostly notebook 17 |
| 19 GPU fine-tuning runs on Hugging Face Jobs, run side by side | about $37 | about four hours |
| Local models and embeddings | free | a few hours |

`uv run python scripts/gpu_jobs.py --dry-run` lists the GPU runs still to do and the most each could cost.
To spend less, run the first notebooks only (`scripts/run_pipeline.sh 01 02 03 04 05 06 07`, about $13 of LLM
calls and no GPU), or swap in cheaper models in [configs/experiment.yaml](configs/experiment.yaml).

### Steps

```bash
uv sync
cp .env.example .env                  # set TAXONOMY_DATA_ROOT and TAXONOMY_JOBS_DATASET
hf auth login                         # or set HF_TOKEN; the token is never written to this repository
uv run python scripts/check_setup.py  # the data folder, disk space, the login and the jobs dataset
scripts/run_pipeline.sh               # everything, in order
```

`scripts/run_pipeline.sh` stops at the first failure, and runs again from where it stopped. In order, it:

1. downloads the test sets (`scripts/download_data.py`);
2. runs notebook 01, which reads them and downloads EuroVoc's tree;
3. embeds every document on your machine with BGE-M3 (`scripts/embed_corpus.py`);
4. runs notebooks 02 to 07, the LLM methods on samples of both test sets;
5. trains the local models (`scripts/train_local_models.py`);
6. launches the 19 GPU runs on Hugging Face Jobs, waits for them and records which commit of your jobs dataset
   holds their outputs (`scripts/gpu_jobs.py`);
7. runs notebooks 08 to 17: the trained models, the combined method, the published benchmarks and HumSet 2.0.0;
8. builds the report (`scripts/make_report.py`).

To rerun one notebook, name it: `scripts/run_pipeline.sh 10`.

### Will you get the same numbers?

Close, not identical. The test sets and GPU recipes are pinned, but three things vary:

- **Omikuji cannot be seeded.** It starts its label clustering from random points, so each training differs a
  little. It is also one of the models the combined method may pick, so a retrain can change which models the
  combined method uses. In two full reruns of ours with Omikuji retrained, the headline figures moved by up to
  about 0.01: EU laws R-Precision 0.630 and 0.632 (ours 0.640), and HumSet 2.0.0 mean macro F1 0.670 and 0.672
  (ours 0.663). Every finding in the report held in both. The report's 95% intervals cover the choice of test
  documents, not this.
- **LLM providers** may answer differently from one month to the next.
- **EuroVoc's tree** is downloaded live from the EU, so it may have changed.

Given the same trained models and saved replies, everything else is repeatable: replayed from our saved replies,
the pipeline rebuilds our report exactly. The report's findings are written for our results, so
`scripts/make_report.py` stops and names any finding your run no longer supports. `git diff outputs/results`
shows how your numbers differ from ours.

## Swapping models

Models are reached through [LangChain](https://www.langchain.com), so changing one means editing
[configs/experiment.yaml](configs/experiment.yaml) and no code. The defaults are open models on Hugging Face's
Inference Providers. For another provider, give LangChain's provider name, install its package, and put its key in
`.env`:

```yaml
models:
  main:
    provider: anthropic       # or openai, google_genai, and so on
    model: <model name>
```

Results depend on the models, so note which model and provider produced any result you report.

## Repository layout

```
configs/experiment.yaml       models, samples, settings, and the GPU runs
notebooks/                    the experiments, in run order, each with its pass marks in its first cell
outputs/results/              what the notebooks measured
outputs/figures/              the notebooks' charts
outputs/report/index.html     the results report, built from outputs/results alone
scripts/                      the pipeline and its steps, and the setup check
scripts/hf_jobs/              what runs on Hugging Face Jobs: fine-tuning, XR-Transformer, HumBERT's recipe
src/tiered_taxonomy/          the taxonomy, scoring, model layer, data loading and the report
src/tiered_taxonomy/methods/  the methods
tests/                        unit tests on small invented data; no data, network or credentials needed
docs/research/                the literature review behind the choice of methods
```

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for the checks every change passes, and [SECURITY.md](SECURITY.md) for
how credentials are handled.

## Licences

The code is MIT-licensed. The test sets and models keep their own terms, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The test sets stay outside this repository; their documents
remain the property of their authors. This work is not affiliated with, or endorsed by, the European Union, the
United Nations, or any of the agencies whose documents appear in the test sets.
