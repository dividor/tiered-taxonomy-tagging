# Third-party notices

What this repository uses from others, and their terms as their own pages gave them on 16 November 2026 (the
Hugging Face licence tag of each model and dataset, and GitHub's licence field of each code repository). This
repository's own code is released under the MIT licence (see `LICENSE`).

## Data

| Source | Used for | Terms | Attribution |
|---|---|---|---|
| [MultiEURLEX](https://huggingface.co/datasets/coastalcph/multi_eurlex) (Chalkidis et al., 2021) | EU laws and their EuroVoc labels | CC BY 4.0 (dataset card); the laws themselves are © European Union, reusable under the Commission's reuse policy | Chalkidis, Fergadiotis and Androutsopoulos (2021), and "© European Union, https://eur-lex.europa.eu" |
| [EuroVoc](https://op.europa.eu/en/web/eu-vocabularies/dataset/-/resource?uri=http://publications.europa.eu/resource/dataset/eurovoc), its links between labels from the Publications Office's SPARQL endpoint | the label tree | © European Union, reusable under the Commission's reuse policy | "EuroVoc, © European Union" |
| [EuroVoc descriptors file](https://github.com/nlpaueb/multi-eurlex) of the MultiEURLEX repository | label names | no licence file in the repository; the names are EuroVoc's | as for EuroVoc |
| [HumSet](https://huggingface.co/datasets/nlp-thedeep/humset) (Fekih et al., 2022), versions 1.0.0 and 2.0.0 | humanitarian excerpts and their labels | Apache 2.0 (dataset card) | Fekih et al. (2022) |

The excerpts' source documents remain the property of their authors. This work is not affiliated with, or endorsed
by, the European Union, the United Nations, or any of the agencies whose documents appear in the test sets.

## Models

| Model | Used for | Licence |
|---|---|---|
| `BAAI/bge-m3` | embeddings of documents and labels | MIT |
| `nlpaueb/legal-bert-base-uncased` | fine-tuned classifier on EU laws | CC BY-SA 4.0 |
| `nlp-thedeep/humbert` | fine-tuned classifier on HumSet, and HumBERT's recipe | Apache 2.0 |
| `BAAI/bge-reranker-v2-m3` | fine-tuned reranker | Apache 2.0 |
| `Qwen/Qwen2.5-7B` | fine-tuned small LLM | Apache 2.0 |
| `fastino/GLiNER2.5-Decide`, `fastino/GLiNER2.5-multi-Decide` | fine-tuned GLiNER2.5 | Apache 2.0 |
| `google-bert/bert-base-uncased`, `google-bert/bert-base-multilingual-cased` | XR-Transformer's encoders | Apache 2.0 |
| `openai/gpt-oss-120b`, `openai/gpt-oss-20b` (through Hugging Face Inference Providers) | the LLM methods | Apache 2.0 |
| `google/gemma-3-12b-it`, `google/gemma-3-27b-it` | LLM-Ensemble's keyword writers | Gemma terms of use |
| `NousResearch/Hermes-3-Llama-3.1-70B` | LLM-Ensemble's keyword writer | Llama 3 community licence |
| `Qwen/Qwen3-14B` | LLM-Ensemble's keyword writer | Apache 2.0 |
| `meta-llama/Llama-3.1-8B-Instruct` | LLM-Ensemble's ranker | Llama 3.1 community licence |

The LLMs are called, not redistributed: their replies are kept in your data folder, which stays out of the
repository.

## Code

| Project | Used for | Licence |
|---|---|---|
| [Omikuji](https://github.com/tomtung/omikuji) | the Omikuji label-tree classifier | MIT |
| [PECOS](https://github.com/amzn/pecos) | XR-Transformer, in its GPU job | Apache 2.0 |
| [classifier-trainer](https://github.com/the-deep-nlp/classifier-trainer), at a pinned commit | HumBERT's recipe, rerun in its GPU job | MIT |
| [GLiNER2](https://github.com/fastino-ai/GLiNER2) | fine-tuning GLiNER2.5 | Apache 2.0 |

Python packages are listed with their versions in `uv.lock`; CI keeps a software bill of materials for each run.
