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
| EU laws, all 5,000 test laws, mean R-Precision, trained on the training laws only as BERT was (0.640 trained on newer laws as well) | **0.610** | 0.484 (BERT, Chalkidis et al., 2021) |
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

## Results

<!-- results:start -->

Every number below comes from `outputs/results`, through the same code as the [interactive report](https://dividor.github.io/tiered-taxonomy-tagging/), which also has a card for each method. Scores run from 0 to 1, and higher is better. *Exact labels* gives credit only for the very label a person assigned; *Tier N* gives credit for landing in the right branch at that level. Recall columns are the share found of the correct labels that training used least, and that it never used. This section is written by `scripts/make_report.py`; do not edit it by hand.

### Against the published benchmarks

**EU laws: mean R-Precision on all 5,000 test laws.** For each law, take as many of the top-ranked labels as it really has, and count the share that are correct.

| Method | Version | Mean R-Precision |
| --- | --- | ---: |
| Combined method, second attempt | With newer laws | 0.640 |
| Combined method, first attempt | With newer laws | 0.624 |
| Combined method, second attempt | Training laws only | 0.610 |
| Combined method, first attempt | Training laws only | 0.600 |
| Fine-tuned Legal-BERT | With newer laws | 0.529 |
| Fine-tuned Legal-BERT | Training laws only | 0.500 |
| BERT, published | The benchmark to beat (Chalkidis et al., 2021) | 0.484 |
| XR-Transformer | Training laws only | 0.474 |
| Omikuji | Training laws only | 0.471 |
| Embedding classifier | Training laws only | 0.459 |
| Similar documents | Training laws only | 0.424 |
| Label search | No training | 0.139 |

**HumSet 2.0.0: macro F1 by label group on all 12,896 test excerpts**, the measure the HumBERT paper reports. Its published figures have two decimals; their mean is worked out from them.

| Label group | HumBERT, published | HumBERT recipe, rerun | Combined method | 95% interval |
| --- | ---: | ---: | ---: | ---: |
| Sectors | 0.78 | 0.778 | 0.790 | 0.780 to 0.799 |
| Pillars 1D | 0.70 | 0.685 | 0.705 | 0.696 to 0.718 |
| Subpillars 1D | 0.54 | 0.548 | 0.574 | 0.562 to 0.588 |
| Pillars 2D | 0.71 | 0.705 | 0.721 | 0.707 to 0.735 |
| Subpillars 2D | 0.51 | 0.501 | 0.527 | 0.512 to 0.541 |
| Five-group mean | 0.648 | 0.643 | 0.663 | 0.657 to 0.670 |

### EU laws (MultiEURLEX): 600 test laws, EuroVoc's top three tiers

| Method | Kind | Exact labels | Tier 1 | Tier 2 | Tier 3 | Rarest quarter recall | Never-seen recall | Labels per law | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Combined method, second attempt, with newer laws | Combined | 0.57 | 0.75 | 0.71 | 0.71 | 0.17 | 0.01 | 3.9 | $10.92 GPU, once |
| Combined method, second attempt, training laws only | Combined | 0.55 | 0.77 | 0.72 | 0.72 | 0.09 | 0.00 | 5.1 | $3.85 GPU, once |
| Combined method, first attempt, with newer laws | Combined | 0.55 | 0.73 | 0.70 | 0.70 | 0.12 | 0.02 | 3.8 | $8.46 GPU, once |
| Combined method, first attempt, training laws only | Combined | 0.53 | 0.73 | 0.69 | 0.68 | 0.05 | 0.00 | 3.9 | $3.85 GPU, once |
| Fine-tuned Legal-BERT, with newer laws | Fine-tuned | 0.50 | 0.75 | 0.70 | 0.69 | 0.13 | 0.01 | 5.2 | $1.09 GPU, once |
| Fine-tuned Legal-BERT, training laws only | Fine-tuned | 0.47 | 0.73 | 0.68 | 0.67 | 0.06 | 0.00 | 4.6 | $1.03 GPU, once |
| GLiNER2.5, with newer laws | Fine-tuned | 0.47 | 0.74 | 0.68 | 0.67 | 0.19 | 0.06 | 4.6 | $2.46 GPU, once |
| Omikuji, with newer laws | Trained | 0.46 | 0.69 | 0.65 | 0.63 | 0.01 | 0.00 | 4.8 | laptop |
| Fine-tuned small LLM, with newer laws | Fine-tuned | 0.46 | 0.70 | 0.66 | 0.66 | 0.02 | 0.00 | 4.1 | $3.37 GPU, once |
| Embedding classifier, with newer laws | Trained | 0.44 | 0.71 | 0.65 | 0.64 | 0.01 | 0.00 | 5.5 | laptop |
| XR-Transformer, training laws only | Trained | 0.43 | 0.70 | 0.65 | 0.64 | 0.00 | 0.00 | 4.0 | $2.82 GPU, once |
| Omikuji, training laws only | Trained | 0.43 | 0.67 | 0.62 | 0.61 | 0.00 | 0.00 | 4.2 | laptop |
| Similar documents, with newer laws | Trained | 0.43 | 0.71 | 0.64 | 0.60 | 0.11 | 0.00 | 5.0 | laptop |
| Embedding classifier, training laws only | Trained | 0.42 | 0.71 | 0.64 | 0.62 | 0.00 | 0.00 | 5.5 | laptop |
| Fine-tuned reranker, with newer laws | Fine-tuned | 0.40 | 0.68 | 0.62 | 0.61 | 0.12 | 0.07 | 4.2 | $4.00 GPU, once |
| Similar documents, training laws only | Trained | 0.39 | 0.63 | 0.57 | 0.53 | 0.02 | 0.00 | 3.4 | laptop |
| KI-FSPrompt | LLM + examples | 0.34 | 0.68 | 0.59 | 0.54 | 0.25 | 0.25 | 9.8 | $0.40 per 1,000 |
| Trained model and search, then LLM chooses | LLM + trained model | 0.24 | 0.54 | 0.42 | 0.36 | 0.15 | 0.20 | 4.4 | $0.61 per 1,000 |
| LLM keyword pipeline | LLM | 0.21 | 0.61 | 0.46 | 0.38 | 0.35 | 0.49 | 15.3 | $0.87 per 1,000 |
| Search label names, then LLM chooses | LLM | 0.19 | 0.43 | 0.32 | 0.27 | 0.15 | 0.21 | 3.9 | $0.46 per 1,000 |
| LLM-Ensemble | LLM | 0.17 | 0.50 | 0.37 | 0.30 | 0.18 | 0.26 | 10.0 | $6.69 per 1,000 |
| Search label definitions, then LLM chooses | LLM | 0.17 | 0.41 | 0.31 | 0.26 | 0.18 | 0.23 | 3.8 | $0.59 per 1,000 |
| Search definitions, LLM chooses, keep quoted labels | LLM | 0.15 | 0.37 | 0.27 | 0.23 | 0.14 | 0.18 | 2.7 | $0.59 per 1,000 |
| Search definitions, LLM chooses and double-checks | LLM | 0.15 | 0.36 | 0.26 | 0.22 | 0.15 | 0.20 | 3.0 | $0.80 per 1,000 |
| Tree walk from search results | LLM | 0.14 | 0.39 | 0.25 | 0.17 | 0.12 | 0.17 | 2.9 | $0.88 per 1,000 |
| Label search | No training | 0.12 | 0.50 | 0.33 | 0.25 | 0.10 | 0.19 | 9.8 | laptop |
| Tree walk, three branches | LLM | 0.05 | 0.32 | 0.14 | 0.08 | 0.08 | 0.10 | 4.2 | $1.35 per 1,000 |
| Tree walk, one branch | LLM | 0.02 | 0.18 | 0.06 | 0.03 | 0.01 | 0.03 | 1.4 | $0.59 per 1,000 |

### Humanitarian excerpts (HumSet 2.0.0): all 12,896 test excerpts

| Method | Kind | Macro F1, sectors and pillars (paper's Avg) | Exact labels | Tier 1 | Tier 2 | Macro F1 | Rarest quarter recall | Never-seen recall | Labels per excerpt | Cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| XR-Transformer | Trained | 0.49 | 0.72 | 0.76 | 0.69 | 0.51 | 0.44 | none | 2.1 | $1.38 GPU, once |
| Combined method | Combined | 0.74 | 0.72 | 0.78 | 0.71 | 0.66 | 0.58 | none | 2.4 | $0.67 GPU, once |
| Authors' HumBERT recipe | Fine-tuned | 0.72 | 0.69 | 0.77 | 0.68 | 0.63 | 0.33 | none | 2.3 | $1.26 GPU, once |
| Similar documents | Trained | 0.65 | 0.69 | 0.73 | 0.65 | 0.59 | 0.41 | none | 2.2 | laptop |
| Fine-tuned HumBERT | Fine-tuned | 0.72 | 0.68 | 0.76 | 0.68 | 0.64 | 0.45 | none | 2.2 | $0.67 GPU, once |
| Omikuji | Trained | 0.63 | 0.67 | 0.71 | 0.64 | 0.57 | 0.44 | none | 2.3 | laptop |
| Embedding classifier | Trained | 0.49 | 0.63 | 0.68 | 0.59 | 0.47 | 0.47 | none | 2.2 | laptop |
| KI-FSPrompt | LLM + examples | 0.58 | 0.58 | 0.65 | 0.57 | 0.53 | 0.59 | none | 2.8 | $0.15 per 1,000 |
| Fine-tuned reranker | Fine-tuned | 0.61 | 0.54 | 0.63 | 0.55 | 0.51 | 0.24 | none | 3.0 | $3.00 GPU, once |
| GLiNER2.5 | Fine-tuned | 0.58 | 0.50 | 0.62 | 0.45 | 0.46 | 0.11 | none | 2.6 | $2.80 GPU, once |
| Fine-tuned small LLM | Fine-tuned | 0.57 | 0.49 | 0.62 | 0.44 | 0.45 | 0.12 | none | 2.4 | $1.67 GPU, once |
| Tree walk, three branches | LLM | 0.40 | 0.41 | 0.47 | 0.26 | 0.30 | 0.42 | none | 1.8 | $0.16 per 1,000 |
| LLM chooses from the full list | LLM | 0.43 | 0.39 | 0.48 | 0.32 | 0.35 | 0.44 | none | 1.9 | $0.15 per 1,000 |
| Search label definitions, then LLM chooses | LLM | 0.42 | 0.37 | 0.45 | 0.33 | 0.35 | 0.46 | none | 1.7 | $0.29 per 1,000 |
| Search label names, then LLM chooses | LLM | 0.43 | 0.36 | 0.46 | 0.30 | 0.35 | 0.54 | none | 2.4 | $0.18 per 1,000 |
| Tree walk, one branch | LLM | 0.36 | 0.36 | 0.42 | 0.22 | 0.26 | 0.24 | none | 1.0 | $0.13 per 1,000 |
| LLM keyword pipeline | LLM | 0.38 | 0.34 | 0.44 | 0.23 | 0.29 | 0.28 | none | 1.4 | $0.33 per 1,000 |
| Search definitions, LLM chooses and double-checks | LLM | 0.40 | 0.34 | 0.42 | 0.30 | 0.33 | 0.39 | none | 1.5 | $0.39 per 1,000 |
| Search definitions, LLM chooses, keep quoted labels | LLM | 0.37 | 0.32 | 0.40 | 0.28 | 0.31 | 0.37 | none | 1.4 | $0.29 per 1,000 |
| LLM-Ensemble | LLM | 0.37 | 0.32 | 0.43 | 0.27 | 0.29 | 0.27 | none | 2.7 | $2.84 per 1,000 |
| Tree walk from search results | LLM | 0.20 | 0.22 | 0.33 | 0.30 | 0.20 | 0.40 | none | 1.5 | $0.22 per 1,000 |
| Label search | No training | 0.33 | 0.11 | 0.25 | 0.07 | 0.25 | 0.52 | none | 7.5 | laptop |

<!-- results:end -->

## Running it yourself

Everything runs from one command, and every step it can skip, it does: each model reply, embedding and trained
model is saved in your data folder, so a rerun asks and pays for nothing twice.

### What you need

- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- A [Hugging Face](https://huggingface.co) account with a token that may call
  [Inference Providers](https://huggingface.co/docs/inference-providers), run
  [Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs), and create and write datasets. The LLM calls
  cost more than the free monthly credit, and Jobs need a paid plan or credits, so set up billing first.
- Access to three gated models that two of the LLM methods use: request it on the pages of
  [gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it),
  [gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) and
  [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). Their owners approve
  requests by hand.
- About 15 GB of free disk. A full run leaves about 13 GB in the data folder, and the Hugging Face cache keeps a
  second copy of the 2.9 GB download and the BGE-M3 embedding model.
- An Apple-silicon Mac or a machine with a GPU makes the local training faster; a plain CPU works, more slowly.
- A machine that can stay awake for a day or more: the GPU jobs take about four hours and notebook 17 about a
  day.

### What it costs

At November 2026 prices, as our run recorded them:

| Step | Cost | Time |
|---|---|---|
| LLM calls (open models through Hugging Face Inference Providers) | about $100, two thirds of it notebook 17 | about a day, mostly notebook 17 |
| 19 GPU fine-tuning runs on Hugging Face Jobs, run side by side | about $37 | about four hours |
| Local models, and embedding every document | free | a few hours |

Some LLM methods also embed their queries through Hugging Face's hosted service, which bills the same account as
the LLM calls. `uv run python scripts/gpu_jobs.py --dry-run` lists the GPU runs still to do and the most each
could cost, set by its time limit. Notebooks 10, 15 and 17 stop before their LLM calls pass $5, $12 and $80, set
in each notebook's first code cell; if prices have risen, raise the limit there and run again.

To spend less, run the first notebooks only: `scripts/run_pipeline.sh 01 02 03 04 05 06 07` downloads the data,
embeds it and runs those notebooks, for about $13 of LLM calls and no GPU. Or swap in cheaper models in
[configs/experiment.yaml](configs/experiment.yaml).

### Steps

```bash
git switch -c my-run                  # the run rewrites outputs/, the notebooks and README.md
uv sync
cp .env.example .env                  # set TAXONOMY_DATA_ROOT and TAXONOMY_JOBS_DATASET
mkdir -p /path/to/taxonomy_test_sets  # the data folder named in .env must exist
uv run hf auth login                  # or set HF_TOKEN; the token is never written to this repository
uv run python scripts/check_setup.py  # the data folder, disk space, the login and the jobs dataset
scripts/run_pipeline.sh               # everything, in order
```

`scripts/run_pipeline.sh` stops at the first failure, and runs again from where it stopped. In order, it:

1. checks the setup and downloads the test sets, checking each file (`scripts/check_setup.py`,
   `scripts/download_data.py`);
2. runs notebook 01, which reads them and downloads EuroVoc's tree;
3. embeds every document on your machine with BGE-M3 (`scripts/embed_corpus.py`);
4. runs notebooks 02 to 07, the LLM methods on samples of both test sets;
5. trains the local models (`scripts/train_local_models.py`);
6. uploads the reranker's inputs to your jobs dataset, launches the 19 GPU runs on Hugging Face Jobs, waits for
   them and records which commit of the dataset holds their outputs (`scripts/gpu_jobs.py`);
7. runs notebooks 08 to 17: the trained models, the combined method, the published benchmarks and HumSet 2.0.0;
8. exports the label browser and builds the report, which also rewrites the results section of this README
   (`scripts/export_label_browser.py`, `scripts/make_report.py`).

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

Given the same trained models and saved replies, everything else is repeatable: we checked that, replayed from
our own saved replies, which are not published, the pipeline rebuilds our report exactly.

The report's findings are written for our results, so `scripts/make_report.py` stops and names any finding your
run no longer supports, and then builds no report. Your results are still all there:
`git diff outputs/results` shows how your numbers differ from ours, and each notebook shows its own. Notebook 01
also stops if today's EuroVoc tree reproduces fewer than 95% of the dataset's top-level labels.

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
