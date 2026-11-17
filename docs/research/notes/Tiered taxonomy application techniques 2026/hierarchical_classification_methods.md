# Classifying text against a supplied tiered taxonomy: techniques from May to November 2026

Searched on 2026-11-05. Window: first posted or published about May to November 2026. Older papers appear only as the origin of a technique, or where first posting falls before the window but publication falls inside, and each is marked.

**How to read each entry.** "First posted" is the arXiv v1 date, taken from the stamp on the v1 PDF or from the arXiv abstract page, never inferred from the ID. For example, arXiv 2608.14649 has a v1 stamp of 30 Jul 2026, and arXiv 2607.24783 was submitted on 22 Jun 2026. "Published" is the venue date. ACL Anthology gives only a month. "First posted: none found" means a DataCite title search of arXiv DOIs returned nothing, which does not prove that no preprint exists.

**Read status codes.**
- **FULL**: PDF converted to text and the whole body read.
- **PDF-S**: PDF converted to text and only the named sections read. Numbers were copied from the extracted text of tables.
- **ABS**: abstract only, from a primary record (arXiv, ACL Anthology, Crossref or OpenAlex).
- **META**: title and metadata only. Not read. The user may wish to obtain these.
- **SNIPPET**: seen only in a web search result or summary. Not verified beyond that, unless the entry says so.

**Earlier work reused.** Entries marked "(earlier notes)" were verified in `research_notes/Entity and ontology extraction papers 2026/mapping_to_supplied_vocabularies.md` and were not re-read unless the entry says so.

**Plain-language key.** "Flat" means predicting leaves directly, ignoring tiers. "Top-down" means choosing a tier-1 label, then a tier-2 label under it, and so on. "Path" means predicting a whole root-to-leaf route at once. "Rolling up" means taking a leaf's parents from the tree after the leaf is chosen.

---

## Starting point: what Kähler et al. (arXiv 2607.14882) actually does, and what builds on or competes with it

### Takeaway
It compares trained extreme classifiers with LLM pipelines on a very large vocabulary, and the winner depends on the measure: XR-Transformer leads on agreement with past librarians, and the LLM-Ensemble leads on graded usefulness and on rare labels. **It does not use the vocabulary's tiers at all.** The "tree" in XR-Transformer is a tree of clustered labels built for speed, and all labels are scored as one flat set, so it says nothing directly about top-down versus flat on a supplied hierarchy.

### Cited Findings

**Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature.** Maximilian Kähler, Katja Konermann, Lisa Kluge, Markus Schumacher (German National Library). First posted 16 Jul 2026, v2 17 Aug 2026. Published: arXiv comment says "Submitted to KONVENS 2026"; the ACL Anthology page for KONVENS 2026 returned 404, so publication is not confirmed. Read: FULL (body and appendices A to C, plus the start of D; plots seen only as captions). [arXiv 2607.14882](https://arxiv.org/abs/2607.14882)

*The setting* — [same paper](https://arxiv.org/abs/2607.14882)
- Label set: GND-204K, 204,056 entities across six types, of which 94,362 are subject headings. All entities are treated alike, which the authors list as a limitation.
- Two tasks: Book-Titles (title only) and Fulltext-30k (first 30,000 characters). The median full text in training is 504,671 characters, so about 6% of a median book is seen. The authors say an ablation with Omikuji showed only a moderate loss and hypothesise that tables of contents are the reason. Full-length books were "computationally out of scope".
- Training: 951,104 titles (193,921 distinct labels) or 167,281 full texts (67,757 labels). Test: 4,651 documents balanced over 20 subject groups, 8,978 distinct labels, 4.04 labels per document.
- Two kinds of measure. Binary: AUCpr (their primary measure), best-possible F1, and long-tail weighted versions (cps) that up-weight rare labels using inverse propensity. Graded: subject librarians rated the top 5 suggestions as false, slightly useful, useful or very useful, where "useful" means a good search term for finding the document. Each method was rated on its own separate slice of 524 to 586 documents, over three years, and only six methods were rated. KI-FSPrompt was not rated.

*How the methods work, in steps*
- **XR-Transformer (trained).** (1) Build a tree of labels by grouping labels, so inner nodes act as meta-labels. (2) Fine-tune a pretrained encoder level by level on those meta-label tasks. (3) Concatenate the tuned dense embedding with sparse TF-IDF features. (4) Train linear classifiers on that for final prediction. It uses no label text. In the Fulltext-30k task the authors hypothesise that most of its power comes from TF-IDF, since the encoder sees a short context window. [Kähler sec. 3.1.5](https://arxiv.org/abs/2607.14882). (The paper is brief on this; I did not read the XR-Transformer origin paper, Zhang et al. 2021.)
- **Omikuji, DiSMEC++, ZestXML, AttentionXML, NGAME (trained).** Omikuji uses partitioned label trees with linear classifiers and beam search at inference. DiSMEC++ trains one classifier per label on TF-IDF. ZestXML correlates text and label TF-IDF features, can add a label hierarchy to the label features, and can reach zero-shot labels. The paper does not say whether the GND hierarchy was fed in. NGAME embeds documents and label text together. AttentionXML failed on Fulltext-30k because it sees only the first few sentences. [Kähler sec. 3.1](https://arxiv.org/abs/2607.14882)
- **Lexical matching (MLLM) and embedding-based matching (EBM).** Find candidate labels by text match or vector search over the vocabulary, then rerank with a small model on simple signals: how often, where, and how spread out the match is. EBM uses encoders such as BGE-M3. [Kähler sec. 3.2](https://arxiv.org/abs/2607.14882)
- **LLM-Ensemble (no training, five stages).** [Kluge and Kähler 2025 (older, origin)](https://aclanthology.org/2025.semeval-1.148/), read: PDF-S (sections 1 to 4.2, plus selection text); benchmark use in [Kähler sec. 3.3.1](https://arxiv.org/abs/2607.14882).
  1. *Complete*: several off-the-shelf open models (Llama, Mistral, Mixtral, OpenHermes, Teuken) each generate free keywords from prompts with 8 to 12 fixed examples. The vocabulary is not shown to the model.
  2. *Map*: each keyword is embedded with BGE-M3 and matched to the nearest vocabulary term with hybrid vector and BM25 search in Weaviate. Low-similarity matches can be dropped by a threshold.
  3. *Summarise*: sum similarities for each term across model-and-prompt combinations, then normalise, giving an ensemble score.
  4. *Rank*: one more LLM (Llama-3.1-8B-Instruct in the benchmark) rates each term from 0 to 10 for relevance to the text.
  5. *Combine*: final score = 0.3 × ensemble + 0.7 × relevance.
  - The ensemble was chosen from 135 combinations, first narrowed to 50 by PR-AUC, then pruned further. The SemEval system made [20 LLM calls per document and used 234 GPU hours on that test set](https://aclanthology.org/2025.konvens-2.42/). Longer texts are "processed chunkwise", but the benchmark does not state chunk size or how chunk results are merged. [Kähler sec. 3.3.1](https://arxiv.org/abs/2607.14882)
- **KI-FSPrompt (cheaper version, four stages, two LLM calls).** [Kähler, Kluge, Konermann, KONVENS 2025 workshops, September 2025, pp. 455-464 (older, origin)](https://aclanthology.org/2025.konvens-2.42/). Read: PDF-S (sections 1 to 3.1). (1) *Retrieve* similar training documents of the same language by embedding search. (2) *Complete*: build the few-shot prompt from those documents and their gold terms. (3) *Map* as above. (4) *Rank* with an LLM. It was not evaluated on Fulltext-30k.

*Results* — [Kähler Tables 4, 5, 6](https://arxiv.org/abs/2607.14882)

| Method | Titles F1 / AUCpr | Titles cps-F1 / cps-AUCpr | Full text F1 / AUCpr | Full text cps-F1 / cps-AUCpr | g-F1@5 titles / full text | ms per title |
|---|---|---|---|---|---|---|
| XR-Transformer | 0.467 / 0.427 | 0.421 / 0.389 | 0.398 / 0.359 | 0.339 / 0.307 | 0.530 / 0.512 | 10.5 |
| DiSMEC++ | 0.426 / 0.402 | 0.386 / 0.368 | 0.361 / 0.354 | 0.305 / 0.306 | not rated | <1 |
| Omikuji | 0.417 / 0.395 | 0.370 / 0.354 | 0.343 / 0.329 | 0.277 / 0.273 | 0.499 / 0.525 | <1 |
| KI-FSPrompt | 0.453 / 0.340 | 0.436 / 0.339 | not run | not run | not rated | 271 |
| LLM-Ensemble | 0.382 / 0.308 | 0.379 / 0.316 | 0.364 / 0.333 | 0.362 / 0.345 | 0.576 / 0.601 | 998 |
| Embedding matching | 0.285 / 0.212 | 0.274 / 0.214 | 0.309 / 0.279 | 0.295 / 0.274 | 0.433 / 0.488 | 9.4 |
| Lexical (MLLM) | 0.274 / 0.148 | 0.266 / 0.146 | 0.303 / 0.268 | 0.277 / 0.253 | not rated | 23 |

- Timings are for titles only, on differing hardware footnoted per method. The authors say full-text times are "much higher", that LLM-Ensemble, EBM and MLLM scale linearly with length, and that the LLM-Ensemble's speed "must be judged as infeasible" for daily library use. [Kähler sec. 5.3](https://arxiv.org/abs/2607.14882)
- By label frequency: LLM-Ensemble and EBM do not improve notably on more frequent labels because they are untrained. KI-FSPrompt improves with frequency, as supervised models do, because it borrows training examples. LLMs "provide a wider coverage of the vocabulary". [Kähler App. C.2](https://arxiv.org/abs/2607.14882)
- KI-FSPrompt leads on the cps-F1 point estimate for titles, but XR-Transformer keeps the larger area under the curve, because KI-FSPrompt returns only 3 to 10 terms per record and cannot trade precision for recall. [Kähler App. C.1](https://arxiv.org/abs/2607.14882)
- The authors recommend building productive systems around Annif and say future work should refine KI-FSPrompt and research ensembling. [Kähler sec. 6-7](https://arxiv.org/abs/2607.14882)

*What builds on or competes with it*
- **Hauss et al., On the Transferability Between Extreme Multi-Label and Hierarchical Text Classification.** Florian Hauss, Tom Speier, Nerijus Bertalis, Paul Granse, Ferhat Gül, Leon Menkel, David Schüler, Lukas Galke Poech, Ansgar Scherp. Published 18 Aug 2026, ACM DocEng. First posted: none checked. Read: ABS via OpenAlex. Flattening the human hierarchy and running extreme classifiers is often competitive with specialised hierarchical models, especially on ranking measures; hierarchical models struggle at extreme scale through limited scale or lower performance. [DOI](https://doi.org/10.1145/3820755.3832808). Relevance: the strongest support for "flat extreme classifier as baseline". Abstract gives no numbers.
- **MATCH**, below in Q2, is the closest competitor at production scale. **Usta, R3, TAGHN** are in (earlier notes) Q2.
- **DualMLC.** Hui Ye, Jing Zhang, Xiulong Yang, Rajshekhar Sunderraman. LLM-Enhanced Dual-Branch Learning for Large-Scale Multi-Label Text Classification. First posted 11 Sep 2026. Preprint. Read: ABS and one sensitivity table. An LLM branch and an encoder branch each score all labels, then logits are merged. It reports state of the art on three benchmarks including EURLex-4K, Wiki10-31K and AmazonCat-13K; on Wiki10-31K the sensitivity rows sit around P@1 90.7, P@3 81.0, P@5 71.9. [arXiv 2609.12915](https://arxiv.org/abs/2609.12915). Relevance: needs training labels and does not use a supplied tree.

### Inferences
- The benchmark's chunkwise handling is unspecified, so it cannot be used as evidence for chunk-then-aggregate design.
- Kähler's LLM-Ensemble handles a closed vocabulary by not showing it to the model and mapping afterwards. This is a different design from every tree-navigation method in Q2. In the UZH task (earlier notes Q3) retrieval pre-filtering before the model lost rare tags, and post-hoc mapping avoids that specific problem, though whether it does better on a tiered taxonomy of hundreds to low thousands of leaves is untested.

### Gaps
- A paper first posted in July 2026 has had little time to be cited. I could not run citation tracking because Semantic Scholar and OpenAlex search were unavailable in earlier work and not retried here, so "builds on" is by topic, not by citation.
- Graded ratings used different document slices per method and took three years, so small graded differences should be read with care. The paper gives confidence intervals but does not test for rater drift.

---

## Q1. Top-down versus flat versus whole paths: what new head-to-head results show, how upper-tier errors spread, and how recovery works

### Takeaway
There is no single winner, and the pattern depends on training and on how strict the descent is. Hard two-stage routing lost to flat in the one plain head-to-head (0.332 vs 0.3659, with 30.2% of items stuck in the wrong branch), while a trained, multi-branch tree walk beat flat on rare leaves but only matched it on micro-F1 at equal training data. The most useful recovery ideas are: start from a retrieved anchor instead of the root, keep several branches open, look ahead to the leaves, and re-check with the full taxonomy visible.

### Cited Findings

#### Head-to-head results

| Paper | Task | Compared | Result |
|---|---|---|---|
| [Tashkees-AI](https://aclanthology.org/2026.abjadnlp-1.20/) | 82 categories under 12 broad domains, Arabic medical questions, trained encoders | Flat vs two-stage | Flat test F1 0.3659; two-stage 0.332 despite stage-1 F1 0.654. 30.2% of samples misrouted at stage 1 cannot recover. Two-stage predicts only 68 of 82 categories; flat predicts 79 of 82 |
| [XplaiNLP](https://aclanthology.org/2026.nslp-1.29/) | Multi-label climate narratives, CARDS taxonomy, Qwen3-8B | Flat prompt vs two-turn top-down vs tree-shaped single pass | Zero-shot flat macro 42.85, micro 74.51; two-turn top-down macro 43.13, **micro 58.01**, 700 s vs 187 s. Tree-shaped chain of thought in one pass with a reasoning model: macro 62.48, micro 84.42, first place (but 2,438 s) |
| [Graph-Constrained Policy](https://arxiv.org/abs/2607.11954) | ICD-10-CM, 15,761 leaves, MIMIC-IV | Trained tree walk vs flat trained | See below: macro gain is large, micro roughly ties at equal data |
| [ToT-ICD](https://aclanthology.org/2026.findings-acl.191/) | Primary diagnosis, 4-level ICD, 560 Chinese records | Retrieve-rerank vs tree agents | DeepSeek-V3.2 full-code accuracy: retrieval 34.64, retrieve-rerank 40.00, plain agent 39.46, look-ahead agent 43.75 |
| [Constraint-Aware Search](https://arxiv.org/abs/2607.10588) | Tariff and regulatory codes | Local tree search vs RAG systems | HS-Hard: LLM alone 17.93, flat evidence RAG + rerank 31.44, ours 58.29. Dual-Use: 78.04 vs 78.64, a very small gain |
| [Yoshimura and Kashima (older, Aug 2025)](https://arxiv.org/abs/2508.04219) | gpt-4o-mini prompts, WoS and Amazon | Leaf only (DL), whole path (DH), top-down multi-step (TMH) | WoS two levels, best DL at 5-shot (ACC2 0.440); TMH 3-shot 0.434. Amazon three levels, best DH at 20-shot (ACC3 0.532); TMH 20-shot 0.490; DL 20-shot 0.392. Trained HPT: 0.571 on WoS ACC2, 0.377 on Amazon ACC3 |
| [Alangari et al.](https://doi.org/10.3390/electronics15142988) | Arabic legal cases, 2 then 63 labels, 1,146 cases | Flat, multi-task, two-stage, hierarchy-aware encoders; GPT-4o | Hierarchy-aware AraBERTv2 best L2 (83.04% accuracy, 76.07 macro-F1); GPT-4o 5-shot macro-F1 74.30 without training |

Paper details for the table:

- **Tashkees-AI at AbjadMed 2026: Flat vs. Hierarchical Classification for Fine-Grained Arabic Medical QA.** Fatimah Mohamed Emad Eldin. First posted: none found. Published March 2026, AbjadNLP workshop, so just outside the window; included as the only plain flat-versus-two-stage test I found. Read: PDF-S (Table 1, results section). Ranked 26th, single author, low absolute scores. [ACL Anthology](https://aclanthology.org/2026.abjadnlp-1.20/)
- **XplaiNLP @ ClimateCheck 2026 Task 2: Comparing Hierarchical Approaches for Fine-Grained Climate Disinformation Narrative Classification.** Arthur Hilbert, Jing Yang, Vera Schmitt. First posted: none found. Published May 2026, NSLP workshop. Read: PDF-S (Tables 2, 3, prompt definitions). Times and CO2 are measured with CodeCarbon. Limitation stated: single-run scores and no hierarchical partial credit. [ACL Anthology](https://aclanthology.org/2026.nslp-1.29/)
- **Graph-Constrained Policy Learning for Extreme Clinical Code Prediction.** Amritpal Singh, Sebastian Torres, Khawar Shakeel, Syed Ahmad Chan Bukhari. First posted 11 Jul 2026. Preprint. Read: PDF-S (abstract, method tools, Table IV, results text, limitations). [arXiv 2607.11954](https://arxiv.org/abs/2607.11954)
  - Full label space (Table IV), micro-F1 / macro-F1: LAAT flat 0.4829 / 0.0988; PLM-ICD flat 0.1958 / 0.0018; CAML flat 0.1938 / 0.0018; traversal on 2,000 notes (SFT-1) 0.4775 / 0.2172; traversal on 5,000 notes (SFT-1+) 0.5266 / 0.2562; three-specialist cascade SFT-3+ 0.5126 / 0.2602. Path-F1 for SFT-1+ is 0.6225.
  - **Caution on the headline.** The abstract's "+0.044 micro-F1" compares the 5,000-note policy with LAAT. The results text says SFT-1 on 2,000 trajectories "matches" LAAT, and describes the flat baselines with "only 2,000 training notes". So at what appears to be equal data, micro-F1 is a tie and the real gain is in macro-F1. The paper does not state the baseline budget in a table, so check before repeating.
  - Zero-shot, the small 4B model on the traversal protocol scores 0.0012 micro-F1, called "protocol-following failure". This shows the method is not zero-shot on small models.
  - The three-specialist cascade overflows a 32,000-token cap on 28 to 32% of full-space notes, a memory limit of their one GPU.
  - Label space is codes observed in the cohort, and the graph is pruned to those codes and their ancestors. All results come from one 4B model and one hospital.
- **Travel on the ICD Tree: Benchmarking Agentic Reasoning for ICD Coding from Chinese Electronic Medical Records (ToT-ICD).** Xinjie Xu, Yongqi Fan, Shuang-shuang Chen, Qi Ye, Weibin Guo, Xinxuan Hu. First posted: none found. Published July 2026, Findings of ACL, pp. 3927-3943. Read: PDF-S (method, Tables 1 to 4, 6). [ACL Anthology](https://aclanthology.org/2026.findings-acl.191/). Single runs per configuration.
- **Constraint-Aware Hierarchical Search for Regulation-Driven Fine-Grained Classification.** Siyu Wang, Wei Tan, Lulu Chen. First posted 12 Jul 2026. Preprint for Elsevier. Read: PDF-S (intro, Tables 1, 3, 4, ablation text). [arXiv 2607.10588](https://arxiv.org/abs/2607.10588). Mean of three runs. 270 to 370 instances per dataset, DeepSeek as LLM.
- **Hierarchical Text Classification Using Black Box Large Language Models.** Kosuke Yoshimura, Hisashi Kashima. First posted 6 Aug 2025 (older; the clearest origin of the three-strategy comparison). Read: PDF-S (abstract, Tables 2, 3). [arXiv 2508.04219](https://arxiv.org/abs/2508.04219). Model: gpt-4o-mini-2024-07-18, per the PDF.
- **Hierarchical Classification of Arabic Legal Cases Using Transformer Architectures and Large Language Models.** Nourah Alangari, Nouf Alshenaifi, Huda Almuzaini. Published 8 Jul 2026, Electronics 15(14). First posted: not checked. Read: ABS via Crossref (truncated at the end). [DOI](https://doi.org/10.3390/electronics15142988)

#### How upper-tier errors spread: numbers by tier

- ToT-ICD, DeepSeek-V3.2, look-ahead agent: chapter 81.61, category 65.36, subcategory 51.07, full code 43.75. Retrieve-rerank: 78.75, 63.57, 48.75, 40.00. So roughly one in five records is already lost at tier 1 for every method, and the agent's gain is modest at each tier. [ACL Anthology, Table 6](https://aclanthology.org/2026.findings-acl.191/)
- Plain tree agents collapse on mid-size models: on Qwen3-30B the plain agent reaches 18.43 at full code against 30.36 for retrieve-rerank and 31.43 with look-ahead. [ToT-ICD Tables 1, 6](https://aclanthology.org/2026.findings-acl.191/)
- Constraint-Aware Search, HS-Hard: level 1 84.77, level 2 72.97, level 3 58.29. HS-Simple: 96.95, 95.31, 90.44. [arXiv 2607.10588, Table 4](https://arxiv.org/abs/2607.10588)
- AraGenre 2026 third place: broad macro-F1 0.8913, specific 0.5114, "most remaining errors concentrate in fine-grained genre choices rather than broad-category mistakes". [NAMAA](https://aclanthology.org/2026.arabicnlp-sharedtasks.33/). Read: ABS. Phoenix, zero-shot: 0.749 broad, 0.411 specific. [Phoenix](https://aclanthology.org/2026.arabicnlp-sharedtasks.31/). Read: ABS.
- Canadian HTS, 10 digits, on 3,300 expert-labelled records: Gemini-3.1-Pro goes from 74.31% chapter accuracy to 47.92% suffix, and 40.97% exact match. [Nguyen et al., arXiv 2606.16987](https://arxiv.org/abs/2606.16987). Authors: Truong Thanh Hung Nguyen, Khanh Van Quynh Nguyen, Hoang-Loc Cao, Tri Duong, Phuc Ho, Van Pham, Loc Nguyen, Hung Cao. First posted 15 Jun 2026. Read: PDF-S (abstract, level tables). Method: multi-agent retrieval, element-wise voting across code parts, and human escalation.
- MMM-Bench, five-level business document taxonomy, 5,990 documents from 12 Alibaba domains. Claude-Opus-4.6 accuracy L1 to L5: 95.11, 83.77, 74.07, 71.57, 86.67; Qwen-3.5-27B: 93.40, 77.19, 57.91, 57.84, 53.33. Accuracy does not fall smoothly, and L5 is sparse. [Ma et al., arXiv 2605.10550, table on level accuracy](https://arxiv.org/abs/2605.10550). Authors: Denghao Ma, Qing Liu, Zulong Chen, Chuanfei Xu, Jia Xu, Zhibo Yang, Wei Shao, Zhao Li. First posted 11 May 2026 (v2 14 May). Read: PDF-S (abstract, one table).
- MAX-EVAL-11: exact match on ICD-11 is 0 to 4.8% for all LLMs; the best weighted score 0.433 (Claude 4 Sonnet) comes mostly from credit for upper tiers. [Singh et al., BioNLP at ACL 2026, July, pp. 282-291](https://aclanthology.org/2026.bionlp-1.23/). Ujjwal Singh, Sarthak Deshwal, Nitish Dube, Arjun Sharma. Read: ABS. This is the clearest warning about unaided generation of leaf codes.
- Errors follow the tree. In CVE-to-CWE mapping the main confusions follow the CWE hierarchy and are shared across three encoders (Pearson r > 0.92), suggesting taxonomy design drives errors more than the model does. [Kelm et al., arXiv 2607.07573](https://arxiv.org/abs/2607.07573). Ana Schwengber Kelm, Christian Bockermann, Jörg Frochte. First posted 8 Jul 2026. Read: ABS.
- The AraGenre organisers warn that "a hard broad-first constraint may propagate an incorrect broad decision", and that their shared task cannot say whether hierarchical constraints help because per-system prediction pairs were not kept. [El-Haj et al., pp. 231-245](https://aclanthology.org/2026.arabicnlp-sharedtasks.27/). Mo El-Haj, Saad Ezzini, Shadi Abudalfa, Mustafa Jarrar, Nguyen Minh Chi, Quan Minh Nguyen. First posted 23 Sep 2026 ([arXiv 2609.27387](https://arxiv.org/abs/2609.27387)). Published October 2026. Read: PDF-S (abstract, consistency sections).

#### Recovery techniques, in concrete steps

- **Several branches open.** The tree walk selects a *subset* of children at each level, and an agenda expands all pending nodes together. An empty selection is allowed, so a whole level can be skipped. [Graph-Constrained Policy, tools `expand_level` and `select`](https://arxiv.org/abs/2607.11954).
- **Look-ahead.** Within the current subtree, vector search takes the top 40 nodes, reports which direct children lead to the best leaves, and also shows the top 3 leaves. The agent sees where each branch ends before committing. [ToT-ICD sec. 3.3.2](https://aclanthology.org/2026.findings-acl.191/). Its navigation or drift errors are only 9.8% of its failures; the largest share is 45.4% initial judgement errors. [Table 3](https://aclanthology.org/2026.findings-acl.191/)
- **Explicit backtracking.** A `validate_current_node` tool checks the current path against the text, and `backtry_path` moves back up after a conflict. The full tool set also has `get_child_node`, `select_next_node` and `finish_selection`. Cost: retrieve-rerank 2 steps and 32K tokens per case; plain agent 14 steps and 323K; look-ahead agent 18 steps and 408K. [ToT-ICD sec. 3.3.2, 6.1](https://aclanthology.org/2026.findings-acl.191/)
- **Start from an anchor, not the root.** Search all nodes globally, pick an entry point, then search locally downwards. On HS-Hard, this raised accuracy from 50.54% to 58.29% and cut average path length from 2.89 to 1.55. [Constraint-Aware Search sec. 4.6.1](https://arxiv.org/abs/2607.10588)
- **Verify each level.** HCRE builds a tree over relations and uses multi-view verification at each level to limit error propagation. [Ma et al., Findings of ACL, July 2026, pp. 39895-39913](https://aclanthology.org/2026.findings-acl.1985/); first posted 9 Apr 2026, before the window ([arXiv 2604.07937](https://arxiv.org/abs/2604.07937)). Guoqi Ma, Liang Zhang, Hongyao Tu, Hao Fu, Hui Li, Yujie Lin, Longyue Wang, Weihua Luo, Jinsong Su. Read: ABS. It concerns relation extraction, not documents.
- **Agreement-gated correction with the whole taxonomy visible.** The first system runs family-restricted judging, then two models re-predict the specific label while seeing all 74 definitions, and a label is changed only when both agree against the base. This raised the top result from 0.7139 to 0.7352 after later steps, and "Religious" topic-trap calls fell 394 → 233 → 119. [Thakaa, pp. 300-312](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/). Hassan Alqaeri, Meshal Alamr, Abdullah Aldahlawi. Published October 2026. Read: PDF-S (abstract, sec. 1, pipeline lines). The authors disclose hand-written cues and leaderboard-based selection.
- **Margin-gated widening.** When the shortlist's top scores are close, widen it. Also rerank siblings separately. [Phoenix, Nguyen Minh Chi, pp. 261-265](https://aclanthology.org/2026.arabicnlp-sharedtasks.31/). Read: ABS.
- **Beam search.** Standard inside trained label trees (Omikuji, per [Kähler sec. 3.1.2](https://arxiv.org/abs/2607.14882)). I found no new paper in the window that runs a named beam search with an LLM over a supplied taxonomy and reports a comparison. Two older items came up only in a search result and are not verified: an unnamed [single-pass LLM paper that takes the top three classes per level](https://payberah.github.io/files/download/papers/llm_classification.pdf) (SNIPPET) and a non-LLM study with beam size 5 at [arXiv 2005.02473](https://arxiv.org/abs/2005.02473) (SNIPPET).

#### Whole paths and semantic codes

- **SemICD.** Yihang Cheng, Qian Zhang, Huang Yonghui, Baoshuo Kan, Junyi Shen. Beyond ICD Codes: Semantic IDs for Generative Medical Coding. First posted 13 Oct 2026. Read: ABS and result lines. Instead of generating original codes, build Semantic IDs by hierarchically clustering embeddings of code text, so related codes share prefixes, then train the model to generate those IDs. Micro-F1 gains are 1.34 to 8.00 points over direct code generation across four LLMs; Gemma3-1B on MIMIC-IV goes from 48.54 to 56.54. [arXiv 2610.19301](https://arxiv.org/abs/2610.19301). Relevance: it replaces the supplied tree for generation purposes and needs training. The abstract states that the ICD hierarchy does not always reflect clinical relationships.
- **MERIT.** Abhinav Mahajan, Arindam Sarkar, Prakash Mandayam Comar. Generative extreme classification over 250k+ categories at an e-commerce firm. Early generated labels steer later ones towards correlates, and the training objective is changed to fix this. First posted: not checked. Published October 2026, EMNLP Industry, pp. 3638-3656. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.emnlp-industry.218/). Relevance: names a specific path-generation failure, namely exposure bias.

### Inferences
- The tier-1 loss rates here (about 15 to 25% on harder tasks) would, under strict top-down, set a hard ceiling. This fits Tashkees-AI and ToT-ICD but is a comparison across tasks, not a controlled test.
- Whole-path versus top-down reversed between datasets in the 2025 study, and XplaiNLP's two-turn prompt lost a lot of micro-F1, which suggests a multi-step prompt should not be assumed better. Why the micro score fell is not established. The paper shows recall rising and precision falling on macro numbers, which is only suggestive.

### Gaps
- I found no in-window paper that compares flat, top-down and whole-path on one dataset **with LLMs and hundreds to thousands of multi-label leaves**. The 2025 study has that design at small scale.
- No in-window paper measures top-down versus flat on long report chunks, or chunk-level versus document-level labels.
- Most papers report final-tier scores only. Few report what share of final errors began at each tier. ToT-ICD's failure table and Tashkees-AI's 30.2% are the closest.

---

## Q2. New methods that use taxonomy structure and label descriptions

### Takeaway
Five working patterns stand out: retrieve across all leaves then prune by tier (MATCH, with two million articles in production), generate label knowledge once and reuse it (SLKR), put the tree into the prompt (XplaiNLP, TaxoAgent), walk the tree with tools (ToT-ICD, graph policy), and encode the hierarchy in training (RelaClass, HiCo-EL). MATCH is the only one shown at a scale near the use case.

### Cited Findings

#### Taxonomy-aware retrieval and pruning

- **MATCH: Multi-stage Article Tagging to Concept Hierarchies.** Daniele Raimondi, Feichi Lu, Oliver Grun, Mariia Eremina, Andrea Perlato (MDPI). First posted 19 Oct 2026. Preprint. Read: PDF-S (abstract, sec. 3 to 3.5, deployment, evaluation rows). [arXiv 2610.25948](https://arxiv.org/abs/2610.25948)
  - Taxonomy: five tiers, 4 Domains, 26 Fields, 252 Subfields, 4,516 Topics, 113,892 Concepts. Each Concept has a short LLM-generated explanation. The same name can sit under several Topics as separate entries. [sec. 3.1](https://arxiv.org/abs/2610.25948)
  - *Stage 1, multi-view retrieval.* Embed Concept names and explanations once with SPECTER2. Split each article into eight views: full title, title word groups, sliding three-word phrases, single abstract sentences, three-sentence windows, adaptive splits, word segments, author keywords. Short pieces match names, long pieces match explanations, each with its own threshold (0.87 to 0.94) and top-k (30 to 50). Merge by Concept, keep the best cosine, add a small boost for repeated hits: `s_final = s_best (1 + 0.1 log h)`. Keep the top 250. Settings were chosen by hand.
  - *Stage 2, tier-based pruning, no LLM.* For Concepts with a single hit, drop those in rare Fields (up to 15% of candidates) and require the Subfield to rank in the top 60% by count or by share. Then merge duplicates and near-duplicates (Jaccard 0.50 on name tokens, union-find), preferring the article's dominant Fields.
  - *Stage 3, LLM rerank.* Either one step with GPT-5.4 mini, or three steps with self-hosted Qwen3 32B (two filter steps, one rank step). JSON is constrained, and returned Concepts must belong to the candidate set. Invalid output is retried up to three times. At most ten Concepts per article, each keeping its full path and its supporting text segments.
  - *Evidence.* Blind review by 22 experts, 11 fields, 110 articles, 219 assessments. Self-hosted Qwen3 32B / API GPT-5.4 mini: P@1 95.9 / 94.5, P@5 88.7 / 88.9, precision 81.5 / 83.3, recall 96.0 / 95.6, F1 86.7 / 87.8, no significant difference. Recall counts only Concepts experts listed as missing, so it is likely generous. Deployed on over two million articles; median five Concepts; covers about 94% of Concepts; estimated 1,900 articles per hour, 14 hours a day, over about two and a half months.
  - *Limits.* Input is title, abstract and keywords only; authors say transfer beyond MDPI is unevaluated. Authors evaluate their own system.
  - *Relevance.* Highest in this set: a similar corpus size, a tiered taxonomy, a self-hosted route, and traceable evidence. Its leaf count is far above the use case's.

- **TAGHN, HiCo-EL, OntologyAligner, Sci-ZSEL.** See (earlier notes) Q1 and Q2. TAGHN picks a narrower term where an unstructured reranker settles for the broader parent, which is directly relevant to deep tiers.
- **BioHiCL.** Mengfei Lan, Lecheng Zheng, Halil Kilicoglu. MeSH hierarchy gives graded positives for contrastive training of small retrievers. ACL 2026 Short, July, pp. 759-769. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.acl-short.62/). First posted before the window ([arXiv 2604.15591](https://arxiv.org/abs/2604.15591), registered 20 Apr 2026 per DataCite; v1 not checked). Relevance: a way to tune embeddings if labelled data exists.

#### Generated label definitions

- **SLKR: Label-Centric Knowledge Retrieval for Zero-Shot Text Classification.** Daoping Wu, Dong Wang, Siyu Liu, Chen Liang, Jian Geng, Ziqi Li, Shuguang Xiong, Ying Zhang. First posted 12 Oct 2026. Preprint under review. Read: PDF-S (abstract, Tables 3, 4, 6). [arXiv 2610.16182](https://arxiv.org/abs/2610.16182)
  - Steps: (1) offline, a flagship LLM writes for each label a definition, retrieval keywords, and boundary rules for confusable pairs. (2) Online, hybrid search finds candidate labels (K = 15). (3) Boundary rules relevant to those candidates are retrieved. (4) A small LLM classifies using only that context.
  - Evidence: WOS-143 accuracy on Qwen3.5-4B: label only 57.10, label plus definition 57.37, 1-shot per label 58.67, SLKR 60.17. On the smaller Ministral-3-3B: 29.04 → 55.83. First-stage recall at K = 15 is 95.9 to 98.3%. Cost on BANKING77: direct flagship 86.01 accuracy at $4.111 per 1,000; SLKR 77.45 at $0.0596 online plus $3.4 build, or $0.4276 per 1,000 amortised over 9,240 classifications.
  - Relevance: high. Build cost depends on the number of labels, not on 1.9 million chunks. It is single-label, and the smallest gain is on the large-model case. It was not tried on multi-label or deep tiers.
- **Mosaic.** See (earlier notes) Q2: generates descriptions for a 140-class taxonomy from unlabelled text.
- **TELEClass (older, origin).** Yunyi Zhang, Ruozhen Yang, Xueqiang Xu and others (first page only). LLM-enriched class terms and pseudo-labels from class names, using a top-down candidate search. arXiv v3 stamp 5 Feb 2025; ID suggests 2024, v1 not checked. Read: title and first lines only, plus description in [TaxoAgent related work](https://aclanthology.org/2026.emnlp-main.1006/). [arXiv 2403.00165](https://arxiv.org/abs/2403.00165). Newer work compares against it.
- **RelaClass.** Wenting He, Qiang Yan, Bing Li, Huaming Liao, Jiafeng Guo, Xueqi Cheng. Published June 2026, IJCNN. Read: ABS via OpenAlex. Uses class names only: the taxonomy becomes a hierarchy of codebooks that decomposes a document step by step into residuals, and LLM-generated prototypes anchor each node. Beats weak-supervision and zero-shot LLM prompting on two benchmarks per the abstract, with no numbers. [DOI](https://doi.org/10.1109/ijcnn69854.2026.11730021)
- **TEAMMix.** (earlier notes) Q2; LLM-generated minority samples.
- **TaxMorph (just outside the window).** Jonas Golde, Nicolaas Jedema, Ravi Krishnan, Phong Le. First posted 26 Jan 2026. Published March 2026, EACL, pp. 214-228. Read: ABS. An LLM rewrites the whole taxonomy (rename, merge, split, reorder), gaining up to 2.9 points of F1 over human-curated trees. LLM-refined trees are harder to separate in embedding space but better fit the model's confusions. [ACL Anthology](https://aclanthology.org/2026.eacl-long.10/). Relevance: if the taxonomy can be edited, editing may help, but the user's taxonomy is supplied, so this may only be a diagnostic.

#### In-prompt navigation and policy training

- **TaxoAgent: Utility-Aligned Policy Optimization for Hierarchical Semantic Classification.** Zhiming Lin, Yihao Zhong, Liwei Hou, Saisai Hu, ShaoYan He, Chen Rongru, Xuan Yang, Siyue Liu. First posted: none found. Published October 2026, EMNLP main, pp. 23149-23171. Read: PDF-S (intro, sec. 3.1 to 3.3, Tables 1, 2, 11 to 13). [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1006/)
  - Steps: (1) the model writes a three-part output: parent labels, child labels given parents, verification. (2) Scores combine correctness, parent-child validity, semantic closeness and path coherence. (3) Reinforcement training favours higher-scoring outputs for the same input. (4) At inference, a fixed step adds missing ancestors and removes out-of-vocabulary labels.
  - Evidence: Micro/Macro-F1 on IOS-APP-ZH 77.35 / 35.21 (best prior micro 74.86), Google-APP-EN 82.50 / 80.70, RCV1 micro 89.38. Direct prompting with ChatGPT-4o gets 37.42 / 27.12 and 73.40 / 70.20. Same-model ladder on IOS: raw 31.38; +supervised 73.51; +projection 73.89; +path prompt 74.18; +verifier rerank 75.62; +DPO 76.41; full 77.35. Mean semantic error distance falls from 0.266 to 0.148, and near-miss rate rises from 51.6% to 82.0%. Inference 1.27 s and 630 tokens, against 0.24 s and 80 tokens raw (Qwen3-14B, one A100).
  - Needs labelled training data and 5 to over 20 hours on 8 GPUs. Average gold children per item is 1.09 to 1.17, so it is barely multi-label.
  - Note: its related work cites many papers on unrelated topics, such as vision and robotics. I did not assess this further.
  - Relevance: the ladder is useful. Most gain comes from supervised data. Enforcing validity after the fact adds only 0.38 points.
- **YNU-HPCC, HiPP.** Hierarchical taxonomy prompting for political evasion (Canning Wen, Jin Wang, Xuejie Zhang; [SemEval 2026, July, pp. 308-313](https://aclanthology.org/2026.semeval-1.45/), ABS, 6th and 5th place). Hierarchical prompting with fine-grained labels predicted first and then aggregated (Lukas Stähelin, Veronika Solopova, Max Upravitelev, David Kaplan, Premtim Sahitaj, Ariana Sahitaj, Charlott Jakob, Sebastian Möller, Vera Schmitt; [Findings of ACL 2026, July, pp. 2887-2902](https://aclanthology.org/2026.findings-acl.139/); [arXiv 2605.13663](https://arxiv.org/abs/2605.13663), first posted 13 May 2026; ABS) helps most after fine-tuning and on ambiguous, low-agreement taxonomies. HiPP is bottom-up, which is interesting.
- **A Deterministic Agentic Workflow for HS Tariff Classification.** Yu Zhang, Dongjiang Zhuang, Qu Zhou, Zheng Huang, Junhe Wu, Jing Cao, Kai Chen. First posted 14 May 2026. Read: ABS. A fixed six-stage flow, not a self-planning agent, with LLM calls confined to narrow stages and verbatim citation of the notes. HSCodeComp: 75.0% top-1 and 91.5% top-3 at four digits, 64.2% and 78.3% at six digits with Qwen3.6-plus. An open-weight 27B model agrees with it 84.2% and 77.4%. [arXiv 2605.14857](https://arxiv.org/abs/2605.14857). Relevance: shows a fixed flow can work with rule-heavy taxonomies. Top-3 is far above top-1, which argues for carrying several options forward (my inference).
- **Hierarchical Classification Framework for Earth Science Data Based on LLMs and Label Graph Constraints.** Le Zhao, Zugang Chen, Guoqing Li, Hengliang Guo, Jing Li. Published 23 May 2026, Applied Sciences. Read: ABS. RAG plus knowledge graph plus hierarchical constraints, on a two-level NASA metadata benchmark: +8.68 points micro-F1 and +29.92 points macro-F1 over BERT-BiLSTM on full paths. [DOI](https://doi.org/10.3390/app16115230)
- **Retrieval-Augmented LLMs and Encoder Models for Multi-Label Climate Disinformation Narrative Classification.** Neda Foroutan, Alexandra Tsiakalou, Vera Schmitt. Published May 2026, NSLP, pp. 225-234. Read: ABS. Retrieval narrows the CARDS labels before the LLM, giving macro-F1 59.72%, second place. [ACL Anthology](https://aclanthology.org/2026.nslp-1.22/). Compare with XplaiNLP's 62.48 first place on the same task with no retrieval narrowing described.

#### Tool-based tree walking

- **ToT-ICD** and **Graph-Constrained Policy**: tool sets described in Q1.
- **Understanding the Limits of Agentic ICD Coding.** Eng Chong Yock, Yushi Cao, Yiming Chen, Kezhi Mao, Hongchao Jiang. First posted 12 Sep 2026 ([arXiv 2609.13806](https://arxiv.org/abs/2609.13806)). Published October 2026, EMNLP main, pp. 36028-36037. Read: PDF-S (abstract, Tables 1, 2, tool text). [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1583/)
  - Nine tools over official references: index lookup with drill-down, tables, child browsing, verification, guideline retrieval, and a submission tool that rejects any code not previously discovered.
  - "The bottleneck is selection": Claude Opus 4.6 discovers 67K codes, submits 2.8% of them, with 56% precision and 8.7 tool calls per correct code. More tool calls do not raise precision. [Table 2](https://aclanthology.org/2026.emnlp-main.1583/)
- **Taxonomy Exploration with Large Language Model Reasoning and Tool Calling.** Fedor Sadkovskii, Mikhail Tikhomirov, Natalia Loukachevitch. Published 13 Jul 2026, Russian Digital Libraries Journal; English version October 2026, Automatic Documentation and Mathematical Linguistics. Read: ABS. A ReAct agent for *taxonomy expansion*, not text classification. Navigating from the root gives MAP@3 24.6%, but using it as a verifier of hypotheses from another model adds 9.5 points for FastText and 1.1 for TaxoYandexGPT-5-Lite. [DOI](https://doi.org/10.26907/1562-5419-2026-29-4-1212-1234); [English DOI](https://doi.org/10.3103/s0005105526700780). Relevance: walking from the root was weak, and the agent as checker worked better, on a different task.
- **PaperRouter-Agent.** Keshen Zhou, Lintao Wang, Suqin Yuan, Zhuqiang Lu, Yu Luo, Zhiyong Wang. First posted 13 Jul 2026. Read: ABS. Training-free. Narrows the folder tree, retrieves folder evidence, inspects member papers, and learns from past rejections. Recall@1 0.39 → 0.61, Recall@3 0.57 → 0.83, in what the authors call a formative study. [arXiv 2607.11564](https://arxiv.org/abs/2607.11564). Relevance: when node names are vague, judge a node by the documents already in it.
- **MedDCR.** Jiyang Zheng, Islam Nassar, Thanh Vu, Xu Zhong, Yang Lin, Tongliang Liu, Long Duong, Yuan-Fang Li. Learns the workflow design itself, with Designer, Coder and Reflector agents. Findings of ACL 2026, July, pp. 12878-12893. Read: ABS, no numbers. [ACL Anthology](https://aclanthology.org/2026.findings-acl.627/)

#### Eval-relevant: choosing among many options

- **Lost in Choice: Evaluating LLMs Under Extreme Multiple-Choice Scaling.** Alessio Cocchieri, Luca Ragazzi, Giuseppe Tagliavini, Gianluca Moro. Published October 2026, Findings of EMNLP, pp. 6931-6963. First posted: not found in my searches. Read: ABS and lines of the introduction. Up to 1,000 options, 31,000 instances, six reasoning LLMs. Accuracy often collapses at late-middle and last positions, and comparative questions degrade much more. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.328/). Relevance: direct evidence for why listing a whole leaf set in one prompt is risky, and why position should be shuffled.
- **dLLM-SetScore.** Pawan Kumar. First posted 30 Jul 2026. Read: ABS. Putting all labels in one masked prompt led to a position effect: first answer slot positive on 99.4% of GoEmotions and 100% of Reuters. Asking about each label separately removes it. [arXiv 2608.14649](https://arxiv.org/abs/2608.14649). Relevance: applies to diffusion models tested here, though the same design question applies to any LLM (my inference).
- **GHADAI.** Manar Ali, Ghader Kurdi. Option rotation, five rotations over the top 15: hierarchical macro-F1 0.4213 bi-encoder, 0.5006 with Fanar-9B reranking, 0.6050 with Qwen-32B. Published October 2026, Arabic NLP, pp. 277-280. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.arabicnlp-sharedtasks.34/)

### Inferences
- Comparing SLKR and MATCH, offline label work appears cheap relative to per-chunk work at this scale. Neither paper tests this at 1.9 million chunks.
- Understanding Limits and ToT-ICD agree that finding the right candidate is easier than choosing it. This matches the entity-linking finding in (earlier notes) Q1.

### Gaps
- No in-window paper applies any of these patterns to UN, development or evaluation reports, or to DAC or UNBIS tiers.
- Tool-walking papers report tokens only on ICD-like tasks, on very long inputs. Cost on short chunks is unmeasured.

---

## Q3. Zero-shot and few-label options, and how they compare with trained extreme models on rare leaves

### Takeaway
On rare labels, untrained LLM methods repeatedly beat trained extreme classifiers, and they may beat them overall when the labels or training data shift. Where training data is plentiful and the label set is stable, trained models lead on overall agreement. Few-example retrieval (KI-FSPrompt) sits between the two: it scores near XR-Transformer but inherits a dependence on training frequency.

### Cited Findings

- **Kähler et al.** Long-tail cps-F1 on full text: LLM-Ensemble 0.362, XR-Transformer 0.339, DiSMEC++ 0.305. Even embedding matching (0.295) and lexical matching (0.277) meet or beat Omikuji (0.277). On titles, KI-FSPrompt 0.436 vs XR-Transformer 0.421. [Kähler Table 4](https://arxiv.org/abs/2607.14882)
- **Understanding the Limits.** Micro-F1 by rarity. PLM-ICD (trained): rare 0.227, common 0.656, a 0.43 gap. CodeSeeker on Opus 4.6: rare 0.560, common 0.607. Plain tool agent, Opus 4.6: rare 0.354, common 0.577. On multi-step injury codes CodeSeeker falls to 0.010 and PLM-ICD holds at 0.256. "No single system dominates." [Table 1](https://aclanthology.org/2026.emnlp-main.1583/)
- **Graph-Constrained Policy** (needs training but no rare-label data at the node level). Macro-F1 0.2562 vs LAAT 0.0988, on the full label space. [Table IV](https://arxiv.org/abs/2607.11954). Note that macro here is averaged across gold codes.
- **AstroConcepts.** (earlier notes) Q2: tail F1 0.198 for a vocabulary-constrained LLM vs 0.081 for astroBERT.
- **AraGenre 2026** tests pure label shift. 74 specific labels are absent from training, only English definitions are given. 27,972 Arabic segments. Top: Thakaa 73.52, HP 71.69, NAMAA 70.13. [Overview](https://aclanthology.org/2026.arabicnlp-sharedtasks.27/). NAMAA's fine-tuned bi-encoders reached 0.93 in development and fell as low as 0.22 on test; zero-shot LLM prompting matched the best tuned ensemble at 0.6899 with no training, and self-consistency sampling raised this to 0.7013. [NAMAA, Hassan Barmandah, Israa Elhosiny, Yousra Alaa El-Ghawi, Omer Nacar, pp. 271-276](https://aclanthology.org/2026.arabicnlp-sharedtasks.33/). Read: ABS.
  - Definition-only dense retrieval with BGE-M3 reports 0.6125 on development, with no tuning. The broad, specific and hierarchical scores are identical in the abstract, which is odd, and it is a development score. [Shlkamy and Zaghouani, pp. 257-260](https://aclanthology.org/2026.arabicnlp-sharedtasks.30/). Read: ABS.
  - Phoenix, retrieval plus reranker: 0.5800 on test. Per-genre z-normalisation of scores gave the largest build-up gain (+0.095), and LLM reranking +0.082. [Phoenix](https://aclanthology.org/2026.arabicnlp-sharedtasks.31/). Read: ABS.
- **BTZSC (just outside the window).** Ilias Aarab. First posted 12 Mar 2026. 22 datasets, 38 checkpoints. Rerankers (Qwen3-Reranker-8B) macro-F1 0.72, small LLMs up to 0.67, strong embeddings give the best trade-off between accuracy and latency. Read: ABS via DataCite record. [arXiv 2603.11991](https://arxiv.org/abs/2603.11991). Relevance: a useful reminder that reranking models may beat small LLMs, though label counts here are generally small.
- **Hierarchical Classification of Management Variables: An LLM Approach.** Isaiah James Lynch, Frank A. Bosco. Published July 2026, Academy of Management Proceedings. Read: ABS via OpenAlex. Zero-shot on nearly 3,700 variables against a construct taxonomy: Gemini-3.0-Pro micro-F1 0.90 vs SBERT 0.77. [DOI](https://doi.org/10.5465/amproc.2026.18478poster). Very short text units, poster.
- **Multi-Stage Framework with Taxonomy-Guided Reasoning for Occupation Classification.** Palakorn Achananuparp, Ee-Peng Lim. Published 25 May 2026, ICWSM. Read: ABS. Inference, then retrieval, then reranking with taxonomy-guided reasoning examples, offered as a cheaper option than GPT-4o. No numbers in the abstract. Preprint status not checked. [DOI](https://doi.org/10.1609/icwsm.v20i1.42622)
- **EMMETT and IRENE (extreme meta-classification).** Sachin Yadav, Deepak Saini, Anirudh Buvanesh, Bhawna Paliwal, Kunal Dahiya, Siddarth Asokan, Yashoteja Prabhu, Jian Jiao, Manik Varma. First posted 23 Jun 2026. Read: ABS. Synthesises classifiers for new labels from classifiers of known labels, adding up to 15 points of Recall@10 on zero-shot items. [arXiv 2606.25237](https://arxiv.org/abs/2606.25237). Relevance: the trained extreme classification field is itself moving towards zero-shot, though this work targets retrieval and advertising.
- **Earlier notes**, Q2: Hydropower (BERT F1 0 on unseen classes, hybrid 0.524 micro-F1), Thakaa, Mosaic, SCOTTA, RelaClass.

### Inferences
- Rough arithmetic of my own, using titles timings from Kähler, so it understates chunk time: KI-FSPrompt at 271 ms would take about 6 days over 1.9 million items, against about 22 days for LLM-Ensemble (earlier notes) and about 5.5 hours for XR-Transformer. Hardware differs.
- MATCH's reported 1,900 articles per hour implies, by my arithmetic, about 1,000 hours for 1.9 million chunks, or about 71 days at 14 hours a day. Chunks may be longer than abstracts, and its hardware was not read.
- ToT-ICD's 408K tokens per case would total about 7.8 × 10^11 tokens over 1.9 million chunks, though its inputs are full medical records. Tree walking on every chunk looks costly; using it on uncertain cases alone is an obvious option, but no paper here tests that.

### Gaps
- I found no in-window head-to-head of LLM methods and trained extreme models **on a tiered taxonomy of hundreds to low thousands of leaves** in general text. Most are ICD (tens of thousands) or small benchmarks (under 150).
- Whether few-example retrieval helps depends on labelled material existing per tier. Kähler has large catalogues. How it behaves with a small labelled seed is not tested.

---

## Q4. How hierarchical consistency is enforced when a chunk can belong to several branches

### Takeaway
Four mechanisms are in use: restrict choices at generation time (tree walk, constrained JSON), repair afterwards (add missing ancestors), roll up from leaf paths, and reconcile across models. Evidence favours building validity in rather than relying on repair for accuracy: repair added only 0.38 points in the one ablation, though it is a cheap guarantee. Consistency is not accuracy, and multi-branch correctness is rarely measured separately.

### Cited Findings

- **Validity by construction, with several branches.** The walk's `select(children=[...])` accepts any subset of children from the last expansion, invalid IDs are rejected, and an empty list is allowed. So a note can descend several branches at once, and every output is a valid path. [Graph-Constrained Policy](https://arxiv.org/abs/2607.11954)
- **Repair afterwards.** Closure projection adds missing ancestors and removes labels outside the vocabulary. TaxoAgent trains on raw output so that violations are still penalised. Raw hierarchy inconsistency on IOS-APP-ZH falls from 10.2% (raw model) to 5.3% (full method). Projection adds 0.38 points on IOS and 0.55 on Google. Weighting hierarchy and semantic rewards too heavily lowers violations by predicting fewer children (recall 76.6 → 70.3), so a coverage term is needed. [TaxoAgent Tables 11, 13](https://aclanthology.org/2026.emnlp-main.1006/)
- **Roll up from leaf paths.** Each Concept carries a full path, so tiers follow automatically. Multi-branch outcomes are possible because the same name under different Topics is a separate entry. But MATCH's filters prefer dominant Fields and remove rare ones among single-hit Concepts, so a real secondary topic with a single hit may be cut. [MATCH sec. 3.1, 3.3](https://arxiv.org/abs/2610.25948). (The last clause is my reading of the design. The authors do not report it as a measured loss.)
- **Candidate-set validation.** MATCH requires returned Concepts to belong to the candidate set, with retries. [sec. 3.5](https://arxiv.org/abs/2610.25948). Understanding Limits has a submission tool that refuses undiscovered codes. [EMNLP](https://aclanthology.org/2026.emnlp-main.1583/). Profiling Hallucinations (earlier notes) shows why: unconstrained models invent codes.
- **Decoding for single-path tasks.** Thakaa forces broad = parent(specific), scoring only siblings of a gated family for small families and top-k for large ones. It notes that broad accuracy then equals family accuracy of the specific label, and that broad and specific macro scores fail differently, so correction must be tier-specific. [Thakaa](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/). This is not multi-label.
- **Reconciling an ensemble.** Flügel et al. propose layered weighting and a specialised inconsistency-resolution algorithm across deep and symbolic models, on ChEBI, 1,742 classes. Simon Flügel, Martin Glauer, Janna Hastings, Till Mossakowski, Christopher J. Mungall, Charlotte Tumescheit, Fabian Neuhaus, Aditya Ganesh Khedekar. First posted 25 May 2026, Research Square. Read: ABS. [DOI](https://doi.org/10.21203/rs.3.rs-9023090/v1). Not text and not an LLM, but one of few works on multi-label reconciliation.
- **Label-wise scoring with a set rule.** OPINE (Mengting Zhang, Gaofeng Pan, Zhixiong Zhang, Yang Li, Guangyin Zhang) scores each label independently, calibrates with section priors, then uses a quantile rule for how many labels to keep, and pairwise refinement for confusable pairs. 18 labels, micro-F1 63.20. Findings of ACL 2026, July, pp. 32313-32333. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.findings-acl.1617/). Not hierarchical, but useful for multi-label chunks.
- **Measuring consistency.** AraGenre defines a rate that counts missing or invalid output as failures, and stresses that a pair can be consistent yet wrong at both levels, or inconsistent yet right at one. [Overview App. C](https://aclanthology.org/2026.arabicnlp-sharedtasks.27/). Path-F1 scores each prediction together with its ancestors, giving credit for a nearby leaf. [Graph-Constrained Policy](https://arxiv.org/abs/2607.11954). MAX-EVAL-11 gives partial credit across five levels. [BioNLP](https://aclanthology.org/2026.bionlp-1.23/). Hierarchical Utility Calibration shows that errors from different tree nodes can cancel, so a small overall calibration error can hide large errors at some nodes (Futoshi Futami, Jerry Huang, Ichiro Takeuchi; first posted 29 Sep 2026; read: ABS). [arXiv 2609.36532](https://arxiv.org/abs/2609.36532). Calibration in multi-label is in (earlier notes) Q4.
- **Default to single label.** LLMs can default to one label unless the output format is changed. See Choo and Hong in (earlier notes) Q4. This is a real risk for several-branch assignment.

**Not read, listed for the user to obtain (META):**
- *When Constraints Go Both Ways: Dynamic Hierarchical Multi-label Classification Networks.* Gelsomina Di Palma, Zhiming Wang, Eleonora Giunchiglia. Published June 2026, IJCNN. No abstract available to me. Likely the most direct work on the logical side. [DOI](https://doi.org/10.1109/ijcnn69854.2026.11730020)
- *Evaluating robustness of hierarchical and multi-label requirements classification.* Nam Nguyen Van, Moataz Ahmed. Information and Software Technology, October 2026 issue; Crossref record created 4 Jun 2026. [DOI](https://doi.org/10.1016/j.infsof.2026.108212)
- *Top-Down Hyperbolic Case Retrieval for Few-Shot Hierarchical Ordinal Text Classification.* Vihanga Ashinsana Wijayasekara, Kyle Martin, Nirmalie Wiratunga. Published 1 Aug 2026, LNCS. An accepted manuscript is listed by OpenAlex. [DOI](https://doi.org/10.1007/978-3-032-33865-5_15)
- *A large language model-augmented hierarchical text classification approach for intelligent failure data management.* Yi Ding, Feng Zhu, He Li, Min Xie. EAAI, October 2026 issue. [DOI](https://doi.org/10.1016/j.engappai.2026.115750) (also listed in earlier notes)
- *Relative contrastive pretraining of heterogeneous graph aware language model for zero-shot multi-label document classification.* Quan Fang, Yufei Li, Jiawei Wu, Yu Long, Jun Hu, Shengsheng Qian. Knowledge-Based Systems, May 2026 issue; Crossref record created 9 Mar 2026. OpenAlex links an SSRN version. [DOI](https://doi.org/10.1016/j.knosys.2026.115711)
- *Coherent Hierarchical Multi-Label Learning to Defer for Medical Imaging.* Joshua Strong, Pramit Saha, Emma Sun, Helen Higham, Alison Noble. First posted around 4 May 2026 per arXiv search listing. On images. Title only. [arXiv 2605.02734](https://arxiv.org/abs/2605.02734)
- *JCLDE* and *L-SIMCL*: hierarchical contrastive models, Knowledge-Based Systems and IP&M 2026, title only: [JCLDE DOI](https://doi.org/10.1016/j.knosys.2026.116155), [L-SIMCL DOI](https://doi.org/10.1016/j.ipm.2026.104818).

### Inferences
- For several-branch chunks, the two approaches with direct mechanisms are the subset tree walk and leaf-retrieval with roll-up. They fail differently: the walk can miss a branch at tier 1, and leaf retrieval can miss a branch when pruning favours a dominant discipline. This is my reading of the designs, not a measured comparison.

### Gaps
- I found no paper that reports recall on *secondary* branches specifically, which is the central question for multi-topic chunks.
- Datasets tested for hierarchical validity have low label counts per item (TaxoAgent about 1.1 to 1.2 children). Chunk-level multi-branch evaluation remains unstudied in what I found.

---

## Q5. Which techniques would be most compelling to demonstrate

### Takeaway
My pick for a demo, as an untested suggestion: show one chunk going three ways (flat with roll-up, tree walk, retrieve-prune-rerank) on the same screen, with a tier-by-tier chart underneath. The evidence for each panel exists in the papers below, and the contrast makes the top-down question visible.

### Cited Findings
Facts that make each idea demonstrable:
- A tree walk produces a readable trace of steps and tool calls, including backtracking. [ToT-ICD Fig. 2 and tools](https://aclanthology.org/2026.findings-acl.191/); [Graph-Constrained Policy tools](https://arxiv.org/abs/2607.11954).
- A funnel from 113,892 to at most 250 to at most 10, with evidence text highlighted for each Concept and a map of retrieved Concepts. [MATCH Figs. 2, 3](https://arxiv.org/abs/2610.25948).
- Tier-by-tier decline charts: [ToT-ICD Table 6](https://aclanthology.org/2026.findings-acl.191/), [HTS](https://arxiv.org/abs/2606.16987), [MMM-Bench](https://arxiv.org/abs/2605.10550).
- Error maps showing near-miss versus far-away errors, as ridgeline plots of semantic error distance. [TaxoAgent Fig. 4](https://aclanthology.org/2026.emnlp-main.1006/).
- Label cards with definition, keywords and "not to be confused with" rules, plus a clear cost table. [SLKR Table 6](https://arxiv.org/abs/2610.16182).
- "Topic trap": the prompt with the whole taxonomy visible moves errors away (394 → 119). [Thakaa](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/). This is a very tellable story.
- A "disagree with the catalogue but useful" result: the same methods change rank when judged by experts. [Kähler Table 5](https://arxiv.org/abs/2607.14882).
- Taxonomy-problem detection: tree-search scoring can flag issues in the tree. [Duraj et al., eBay, first posted 1 Jan 2026, just outside the window, read: abstract](https://arxiv.org/abs/2601.00510). In UZH, 20 of 178 reviewed paragraphs had genuine taxonomy gaps (earlier notes Q3).
- Position effects: shuffle the labels and watch answers change. [Lost in Choice](https://aclanthology.org/2026.findings-emnlp.328/); [dLLM-SetScore](https://arxiv.org/abs/2608.14649).

### Inferences
All items below are **my own untested hypotheses**, not findings.

Ranked by interest to a reader, in my judgement:
1. **Three-way race on a single chunk.** Same chunk, same taxonomy. Panel A: flat leaf scoring with roll-up. Panel B: tree walk, animated. Panel C: MATCH-style funnel. Show where they disagree.
2. **Tier waterfall.** On a small gold set, chart accuracy at each tier for each method, and mark how many final errors began at each tier. This answers the top-down question directly, and it is rarely reported, so it would be a real contribution to the blog post.
3. **Rescue moves.** Show the same walk with and without an anchor start, then look-ahead. The HS-Hard numbers suggest a visible effect, but it is untested on this data.
4. **Label-card maker.** Generate a definition and boundary rules for each leaf, then show classification improving, and show the small build cost against the very large run cost. It is cheap and appealing for a taxonomy with bare names.
5. **Multi-branch map.** Take a real multi-topic chunk, such as gender and agriculture, and show the branches lit up on a treemap, to raise the secondary-branch recall question.
6. **Gap finder.** List chunks where every method abstains or disagrees, as candidates for missing taxonomy nodes.

Suggested pipeline to test on the use case:
- Fix a small gold set of chunks across agencies and languages, with graded judgements, as Kähler does.
- Run retrieve across all leaves with several chunk views, then prune by tier, then LLM rerank with constrained output, on a sample. Call the tree-walk only on low-margin cases. Selective calling is supported by SelectED and DuCEI in (earlier notes) Q1, but not on this task.
- Compare with a flat extreme classifier trained on publisher tags where they exist, as a baseline, since Hauss et al. say this is strong.
- Test BGE-M3 or similar multilingual embeddings given French and Spanish. The evidence here is AraGenre's Arabic text with English definitions, not French or Spanish.
- PDF parsing noise is a further unknown: none of these papers tests it.

### Gaps
- None of the demo ideas has been tried on evaluation-report chunks.
- I did not view figures as images in most papers; figure references rely on captions and surrounding text.

### Search coverage and tool limits (applies to all questions)
- **Worked.** ACL Anthology event pages for ACL, EMNLP, EACL, LREC, SemEval and Findings 2026 (22,740 titles collected, filtered by keyword, about 240 titles inspected, about 30 abstracts read). DataCite API for arXiv DOIs (10.48550), about 70 title and abstract queries, 354 records filtered to 133 and then reviewed by title. Crossref API, 15 queries. arXiv PDFs and abstract pages for v1 dates. OpenAlex lookup by DOI.
- **Partly worked.** arXiv API returned 429 after a short run of queries, and I stopped using it. arXiv advanced search completed 15 of 56 planned queries before 429 errors recurred, and I stopped rather than press on. Two web searches gave mostly older work, recorded above as SNIPPET.
- **Not tried or unavailable.** SIGIR, CIKM and KDD 2026 were not enumerated directly: covered only through arXiv and Crossref titles. KONVENS 2026 and NAACL 2026 event pages returned 404 at ACL Anthology, so GermEval 2026 was not checked (the earlier notes found no 2026 LLMs4Subjects edition). No bot check was met or bypassed. NeurIPS 2026 accepted lists were not checked.
- DataCite "registered" dates are not first-posted dates; I used them only for selection, and where they are quoted above the entry says so.
- Keyword filters on titles will miss papers that use other words. This list is verified but not exhaustive.
