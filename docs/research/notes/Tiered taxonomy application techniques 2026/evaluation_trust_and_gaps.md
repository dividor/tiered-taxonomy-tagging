# Trusting and improving a supplied tiered taxonomy once applied to text: evaluation, evidence, abstention, gaps, languages

Searched on 2026-11-05. Window: first posted or published about May to November 2026. Older items are marked **[older]** and are included only as the origin of a technique, or because their first posting is old and their publication falls inside the window (the entry says which).

**Read status codes.**
- **FULL**: I downloaded the HTML or PDF, converted it to text myself and read the named sections directly, not through a summarising tool. I did not read every line.
- **ABS**: abstract only, from arXiv, the ACL Anthology, Crossref or OpenAlex. "ABS (cut)" means the record I read was cut off before the end of the abstract, so later numbers may exist.
- **META**: title and metadata only. Not read. The user can obtain these.
- **SECONDARY**: known only through another paper's description. Not checked against the original.

**Dates.** "First posted" is the arXiv v1 date from the arXiv API `published` field or the abstract page, never inferred from the ID. Several IDs mislead: 2608.18097 was first posted 8 Jun 2026, and 2609.22133 on 24 Aug 2026. "Published" is the venue date; the ACL Anthology gives only a month (taken from each paper's BibTeX record). "No arXiv title match" means an arXiv title search returned nothing, which can miss a preprint with a different title. Crossref-only records give the Crossref date.

**Search-tool leads.** A general web search tool was used to find leads only. Nothing below rests on it: each entry was then opened at a primary source, with the read status stated.

Earlier notes are referred to as **[mapping notes]**: `/Users/matthewharris/git/nvidia-autoontology-demo/research_notes/Entity and ontology extraction papers 2026/mapping_to_supplied_vocabularies.md`. Entries reused from there are by reference and carry that file's read status. I did not modify that folder.

Coverage and tool limits are at the end of Q6.

---

## Q1. Hierarchical evaluation: metrics that credit near misses by tier, and whether metric choice changes conclusions

### Takeaway
Four usable families exist: match at a coarser tier (exact versus concept, broad versus specific), fixed credit by tree distance, a similarity matrix over labels, and expert graded judgement. Metric choice changed the winner in the two papers that tested it most directly (Kähler et al.; LCSHBench), and Kähler's reversal is real as printed but rests on different documents per method and has no reported inter-rater figure. I found no paper that compares all four families on one dataset, and no paper that I could confirm builds on Kähler et al.

### Cited Findings

#### The Kähler reversal, step by step

**Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature.** Maximilian Kähler, Katja Konermann, Lisa Kluge, Markus Schumacher. First posted 16 Jul 2026, v2 17 Aug 2026. Preprint, marked as submitted to KONVENS 2026; no published version found (the Anthology has no KONVENS 2026 event page). Read: FULL (sections 2, 4, 5, 6, 7; appendices A.1, C.3, D.4). [arXiv 2607.14882](https://arxiv.org/abs/2607.14882). Metadata and basic numbers also in [mapping notes] Q2.

How the grading was done:
1. The 4,651-document test set was cut into six slices with the same subject-category mix, and held without gold annotations before the evaluation phase ([arXiv 2607.14882, appendix A.1](https://arxiv.org/abs/2607.14882)).
2. Six methods were rated: AttentionXML, embedding matching, LLM-Ensemble, Omikuji, XR-Transformer, ZestXML. Each got its own slice (524 to 586 documents), so that an expert saw each record once.
3. Over three years, professional subject experts rated each method's top 5 suggestions on an ordinal scale: 0 false, 1 slightly useful, 2 useful, 3 very useful. Usefulness was defined as suitability as a search query for finding the document, deliberately different from following cataloguing rules.
4. The experts then completed the record under the cataloguing rules, adding what the machine missed. That produced the gold standard for all 4,651 documents, so gold was created after seeing machine suggestions.
5. Scoring follows Kekäläinen and Järvelin ([JASIST 2002, DOI 10.1002/asi.10137](https://doi.org/10.1002/asi.10137), **[older]**, origin of generalised precision and recall; confirmed on Crossref, not read). Ratings are normalised to 0 to 1. Delta-rel is the sum of ratings over top-k suggestions that are not in gold. g-Prec = (tp + Δrel) / (tp + fp). A suggestion that matches gold keeps full credit, whatever its rating, so a g-score can only be at or above the binary score.
6. Flag: g-Rec as rendered in the HTML is (tp + Δrel) / (tp + fp + Δrel). I would have expected fn in place of fp. Check the PDF before reproducing it.

The reversal as printed (Tables 3, 4 and 5):

| Method | Titles F1* | Titles cps-F1* | Titles g-F1@5 [95% CI] | Full text F1* | Full text cps-F1* | Full text g-F1@5 [95% CI] |
|---|---|---|---|---|---|---|
| XR-Transformer | 0.467 | 0.421 | 0.530 [0.520, 0.545] | 0.398 | 0.339 | 0.512 [0.502, 0.521] |
| LLM-Ensemble | 0.382 | 0.379 | 0.576 [0.570, 0.583] | 0.364 | 0.362 | 0.601 [0.595, 0.601] |
| Omikuji | 0.417 | 0.370 | 0.499 [0.492, 0.521] | 0.343 | 0.277 | 0.525 [0.513, 0.531] |
| AttentionXML | not captured | not captured | 0.523 [0.514, 0.536] | not captured | not captured | 0.337 [0.331, 0.353] |

(F1* is the best F1 reachable with an optimal limit and threshold. cps is a long-tail weighting based on inverse propensity scores.)

- The authors explain the gap through the rating distribution: LLM-Ensemble's share of "very useful" is only slightly larger, but it has many more "useful" and "slightly useful" and fewer "false" ([arXiv 2607.14882, section 5.2](https://arxiv.org/abs/2607.14882)).
- Limits the paper states: only six methods could be rated, only the top 5, and the most promising LLM prototype (KI-FSPrompt, titles F1* 0.453) was excluded from grading ([section 6](https://arxiv.org/abs/2607.14882)).
- Limits I observed in the text:
  - Graded scores for different methods come from different documents, so they are not paired comparisons.
  - A keyword search of the text found no inter-rater figure.
  - F1* is at an optimally tuned operating point, while g-F1 is fixed at k=5, so the two columns are not at the same operating point.
  - The printed full-text CI for LLM-Ensemble has an upper bound equal to its point estimate (0.601); reproduced as printed.
- Relevance: the closest worked example for a demo, because publisher tags in the use case are also a rule-bound human product. It credits near misses by asking people, and it does not use the tree at all. No tier is involved.

**What has followed Kähler?** I could not establish this. Semantic Scholar returned 429 throughout, so no citation list was available. The later papers I checked by text search do not cite it: [Reading the Neighborhood, arXiv 2610.20048](https://arxiv.org/abs/2610.20048) and [SHELF, arXiv 2609.03047](https://arxiv.org/abs/2609.03047). Its nearest parallel is earlier: LCSHBench (below), posted six weeks before, which describes a prior DNB result (Kluge and Kähler 2025, SemEval-2025 Task 5) as ranking fourth on F1 and first on expert rating. That description is SECONDARY: I did not confirm it against the SemEval overview or system paper.

#### Exact versus concept match, read as a tier

**LCSHBench: A Multilingual, Consensus-Grounded Benchmark for Library of Congress Subject Heading Assignment.** Kwok Leong Tang. First posted 3 Jun 2026. Preprint. Read: FULL (sections 2, 3.3, 3.10, 3.11, 4, 5, 6). [arXiv 2606.04382](https://arxiv.org/abs/2606.04382)
- Technique (steps):
  1. Admit a record only if at least two independent cataloguing agencies assigned headings.
  2. Score every metric twice: exact (the full subdivided heading) and concept (the base term before the first `--`).
  3. Break every metric down by language and heading type.
  4. Report a human reference by treating each library as a predictor of the consensus of the other two.
- Data: 22,346 books in 15 languages, about 515,000 labels; neural runs on a 2,002-record development subset. Concordance study of 465,187 works catalogued by three libraries.
- Evidence on agreement: identical heading sets 39.4% exact and 49.8% concept. At least one shared heading 80.7% and 93.3%. Median three-way Jaccard 0.50 and 0.86. Human agreement reference 86.9% exact, 93.0% concept.
- Evidence on flips: the fine-tuned on-device embedder leads on exact recall@200 (0.659 vs 0.623), while text-embedding-3-large leads on concept recall@200 (0.762 vs 0.732, paired p = 0.0001). LLM generation F1 is 0.161 exact and 0.384 concept.
- Limits per the author: development subset only, first-stage retrieval only, expert track not done, not on held-out test.
- Relevance: gives a measured reason to score at a coarser tier, since most human disagreement sits below the concept level. Library catalogues, not reports.

#### Fixed credit by tree distance

**MAX-EVAL-11: A Large Scale Benchmark for Evaluating Large Language Models on Full-Spectrum ICD-11 Medical Coding.** Ujjwal Singh, Sarthak Deshwal, Nitish Dube, Arjun Sharma. Published Jul 2026, BioNLP 2026, pp. 282-291. No arXiv title match. Read: FULL (sections 3.4, 4.4, 5). [ACL Anthology 2026.bionlp-1.23](https://aclanthology.org/2026.bionlp-1.23/)
- Technique:
  1. Assign credit h(p, g) by the deepest shared level of a 5-level tree: 1.0 for exact, falling by 0.1 per level to 0.6 for a chapter match, 0 for no shared chapter.
  2. Take the best match per prediction.
  3. Combine: Score = 0.5·Exact + 0.3·Clinical Precision + 0.15·Points Earned + 0.05·Hierarchical Match.
- Evidence: on 500 samples, exact match is 0.000 to 0.048 and final scores 0.245 to 0.433. Pilot validation against 2 expert coders on 50 cases gives Spearman ρ=0.79 (the sentence is broken in my text extraction, so check what it is compared against).
- Gold labels are mapped from ICD-9, with κ 0.67 on the high-confidence subset and 0.52 on the full set.
- Observations from Table 4 (my reading): Claude 4 Sonnet leads the final score (0.433) yet Claude 3.7 Sonnet has higher exact match (0.048 vs 0.047) and Claude 3.5 Haiku has higher hierarchical match (0.402 vs 0.375). So the leader depends on the component.
- Flags:
  - The paper's worked example gives 0.8 for a one-level miss, which does not fit the stated 0.1 step.
  - My arithmetic: with the printed weights, the printed 0.433 needs Points Earned of about 1.74. That is possible only if the term can exceed 1, because it sums over predictions and divides by gold count. Points Earned is not printed in the part I read.
- Relevance: the plainest recipe for tier credit, and a warning to check whether a near-miss term rewards over-generation.

#### Equal-weight tier averages

**AraGenre 2026: A Hierarchical Definition-Guided Arabic Genre Classification Shared Task.** Mo El-Haj, Saad Ezzini, Shadi Abudalfa, Mustafa Jarrar, Nguyen Minh Chi, Quan Minh Nguyen. First posted 23 Sep 2026. Published Oct 2026, Arabic NLP 2026 shared tasks, pp. 231-245. Read: ABS plus metric section. [ACL Anthology 2026.arabicnlp-sharedtasks.27](https://aclanthology.org/2026.arabicnlp-sharedtasks.27/); [arXiv 2609.27387](https://arxiv.org/abs/2609.27387)
- Technique: Hierarchical Macro F1 is the equal-weight average of two separately scored tasks, broad and specific. The overview itself says it is not path-aware: a right broad label earns credit with a wrong specific one, and inconsistent pairs get no extra penalty.
- Data and evidence: 27,972 segments, 6 broad and 74 specific labels, English definitions on Arabic text, 17 ranked entries. Winner 73.52. For the third-placed team, Specific Macro F1 is 0.5114 vs Broad 0.8913 ([NAMAA, 2026.arabicnlp-sharedtasks.33](https://aclanthology.org/2026.arabicnlp-sharedtasks.33/), ABS).
- Relevance: the most direct tier-by-tier gap figure in the set, and it shows a tuned system collapsing under a taxonomy shift: NAMAA's encoder reached 0.93 on development and as low as 0.22 on test.

#### Similarity-matrix credit

**A Unified Subject Map for 130 Years of Physics.** Khoa Nguyen, Pragyan Pandey, Sophie Li, Eric Y. Ma. First posted 12 Jun 2026. Preprint. Read: FULL (results and appendix metrics, not all). [arXiv 2606.14043](https://arxiv.org/abs/2606.14043)
- Technique:
  1. An LLM assigns a modern vocabulary of over 3,000 concepts across the whole APS archive, in batches, with few-shot examples.
  2. Non-vocabulary outputs are retried until valid.
  3. Disciplines (coarse tier) are scored by exact sample F1. Concepts (fine tier) are scored by semantic F1, where near misses earn partial credit through a concept-similarity matrix, against author-assigned labels on recent papers.
- Evidence: on a 1,000-paper held-out benchmark, concept semantic F1 80.9% and discipline F1 77.6%.
- Relevance: an applied retrospective tagging of a large archive, close in shape to the use case. Note the fine tier scores higher only because a softer metric is used on it.

**Semantic F1 Scores: Fair Evaluation Under Fuzzy Class Boundaries.** Georgios Chochlakis, Jackson Trager, Vedant Jhaveri, Nikhil Ravichandran, Alexandros Potamianos, Shrikanth Narayanan. First posted 25 Sep 2025 **[older]**. Read: ABS. [arXiv 2509.21633](https://arxiv.org/abs/2509.21633). Origin of the metric above. Soft precision and recall through a label similarity matrix, for label sets of any size, without forcing matches between dissimilar labels.

#### Older origins worth citing
- **[older]** Kosmopoulos, Partalas, Gaussier, Paliouras, Androutsopoulos. *Evaluation measures for hierarchical classification: a unified view and novel approaches.* Data Mining and Knowledge Discovery, Crossref date 6 Sep 2014. Confirmed on Crossref, not read. [DOI 10.1007/s10618-014-0382-x](https://doi.org/10.1007/s10618-014-0382-x)
- **[older]** Falis, Dong, Birch, Alex. *CoPHE: A Count-Preserving Hierarchical Evaluation Metric in Large-Scale Multi-Label Text Classification.* EMNLP 2021. Confirmed on Crossref, not read. [DOI 10.18653/v1/2021.emnlp-main.69](https://doi.org/10.18653/v1/2021.emnlp-main.69)
- **[older]** Plaud, Labeau, Saillenfest, Bonald. *Revisiting Hierarchical Text Classification: Inference and Metrics.* First posted 2 Oct 2024, CoNLL 2024. Read: ABS. Metric choice and inference method interact, and simple baselines are often competitive. [arXiv 2410.01305](https://arxiv.org/abs/2410.01305)

#### Other reversals and metric warnings in the window
- **High Scores, Hidden Failures: Evaluating Codebook-Based LLM Content Analysis.** Elena Senger, Yuri Campbell, Rob van der Goot, Barbara Plank. First posted 26 Oct 2026. Read: ABS. [arXiv 2610.35179](https://arxiv.org/abs/2610.35179). Four datasets and five models. High aggregate scores coexist with label-level disparities, instability across runs and domain sensitivity, which accuracy and micro-F1 do not show.
- **When the Ruler is Broken: Parsing-Induced Suppression in LLM-Based Security Log Evaluation.** Chaitanya Vilas Garware, Sharif Noor Zisad. First posted 8 May 2026. Read: ABS. [arXiv 2605.07293](https://arxiv.org/abs/2605.07293). A strict regex scorer reported 0% threat accuracy and a fuzzy one 76% on identical outputs, on 50 examples. A reminder to test the scorer itself.
- UZH shared task: F1 and LLM-judge rankings diverge (one team 5th on F1, 1st on the judge). By reference: [mapping notes] Q3; [ACL Anthology 2026.argmining-1.10](https://aclanthology.org/2026.argmining-1.10/).
- *Hierarchy-Aware Hyperbolic and Semantic Reranking* (Labbe et al., BioNLP 2026) introduces hierarchy-aware evaluation beyond exact match. By reference: [mapping notes] Q4; [ACL Anthology 2026.bionlp-1.5](https://aclanthology.org/2026.bionlp-1.5/).

### Inferences
All untested hypotheses, not findings.
- Untested hypothesis: a scorecard with exact, parent and top-tier match per language would show most errors as fine-tier errors in this corpus, as AraGenre and LCSHBench show in theirs.
- Untested hypothesis: Kähler's reversal could be partly a usefulness-versus-rules effect and partly an operating-point effect. A paired re-rating at the same k on shared documents would separate them.
- Untested hypothesis: near-miss terms that sum over predictions can reward over-generation, so any tier credit should be reported with a label-count column.

### Gaps
- No paper in the window compares hierarchical F1, tree distance, similarity-matrix credit and graded judgement on one dataset. I found no in-window study measuring how far tree-distance credit agrees with expert grades, other than MAX-EVAL-11's 50-case pilot.
- The arXiv query for "hierarchical precision" returned 0 results in the window, and "tree distance" with classification also returned 0. Other wording could be missed.
- No citation data for Kähler et al.; the question of what followed stays open.

---

## Q2. Evaluating without a gold standard: silver labels, LLM judges, inter-model agreement, sampling for review

### Takeaway
Agreement among models is repeatedly shown not to establish correctness, and scores can fall steeply when moved from operational labels to independent review. The most defensible design in the window is a small human sample combined with cheap model scores through prediction-powered inference, which corrects bias in the estimate rather than assuming the model is right. Model disagreement is useful for choosing what humans review.

### Cited Findings

#### How far silver labels can flatter

**Evaluation Sovereignty in Metadata-Driven Classification: A Multi-Track Framework for Weakly Supervised Information Systems.** Raymond Vasquez. First posted 11 Jun 2026. Preprint. Read: FULL (introduction, sections 3.3, 5.1, 5.2, data statement). [arXiv 2606.13436](https://arxiv.org/abs/2606.13436)
- Technique: vary the training and evaluation label sources as tracks. Track F trains and tests on operational labels. Track R trains on operational labels and tests on an independently reviewed subset. Track P uses a high-authority source (PubMed/MeSH). Track X (inverted authority) is defined but not evaluated.
- Evidence: hierarchical multi-label classification on OSTI metadata. Level 1 Micro-F1 about 0.64 falls to 0.37. Level 2 about 0.54 falls to 0.03.
- Caveats the author states: the Level 2 fall combines two changes, because Track F uses a Top-N ranking constraint and Track R does not. The gold subset is 1,000 records from a single reviewer with no inter-annotator agreement, described as a structured audit. Single-author preprint.
- Relevance: the closest match to the use case's mixed publisher and LLM tags, and it reports per tier. The 0.03 should not be quoted without that caveat; the Level 1 change is the cleaner figure.

**When Ground Truth Disagrees: A Human-in-the-Loop Audit of Annotation Errors in High-Stakes Crash Narratives.** Md Sajjad Hossain, Lin Li, Judy A. Perkins, John Clary, Joel Meyer. Published Jul 2026, LAW XX, pp. 241-256. No arXiv title match. Read: ABS. [ACL Anthology 2026.law-main.18](https://aclanthology.org/2026.law-main.18/). 9,387 crash reports. 49.4% show a discrepancy between structured and narrative labels, mostly unsupported structured assignments. Narrative-based annotation agrees with adjudication at κ=0.990. Annotator-reported uncertainty predicts difficulty: uncertain records are nearly nine times more likely to disagree with structured labels. Relevance: a published template for auditing administrative tags against the text.

**Can LLMs Reliably Annotate Bioassay Metadata to Improve Data Readiness?** Laura van Weesep, Riccardo Tedoldi, Jens Sjölund, Hossein Azizpour, Susanne Winiwarter, Ola Engkvist, Jon Paul Janet, Samuel Genheden, Juan Viguera Diez. First posted 1 Oct 2026; accepted to the AIDaR workshop at NeurIPS. Read: ABS. [arXiv 2610.01616](https://arxiv.org/abs/2610.01616). About 2 million PubChem bioassays: 36% lack an assay format, over 99.9% lack any ontology-mapped term. Recall at least 0.96 for the two main formats against silver. Many disagreements trace to inconsistency between silver sources. A senior curator revised some of their own labels after seeing LLM evidence.

**FrenchNews-7: Benchmarking Cross-Publisher French News Editorial Desk Classification.** Amr Sobhy. First posted 8 Jun 2026, v2 15 Oct 2026; accepted at AACL-IJCNLP 2026 Main (no Anthology record checked). Read: ABS. [arXiv 2608.18097](https://arxiv.org/abs/2608.18097). Labels from publisher URL slugs plus LLM annotation, audited with 2 humans and 2 LLMs (pairwise κ at least 0.766; human-human κ 0.806). Economie recall 0.517 is indistinguishable from blinded human agreement (0.55, n=40), so the author reads it as a limit of the label, not the model. Relevance: publisher-derived labels, French, and a per-label reliability table as a published artefact.

#### LLM judges

**Large Language Models as Judges for Human Subject Indexing** (Włodarczyk, Zdunek, Aug 2026) and **Beyond the gold standard** (Golub et al., Sep 2026): by reference, [mapping notes] Q4. LLMs are weak at catching incorrectly assigned descriptors, and F1 on full text is much lower than on metadata in fiction.

**Reliability of LLM Judges for Evaluating Entity Alignment.** Vaibhava Lakshmi Ravideshik, Mayank Kejriwal. First posted 7 Oct 2026. Published Oct 2026, EMNLP 2026 main, pp. 39175-39200. Read: ABS. [ACL Anthology 2026.emnlp-main.1717](https://aclanthology.org/2026.emnlp-main.1717/); [arXiv 2610.09554](https://arxiv.org/abs/2610.09554). Three frontier models, three datasets, four systems. When the judge can see the system's decision label, discrimination collapses (ROC-AUC 0.12 to 0.87). A label-free protocol recovers 0.93 to 1.00. Stronger judges show more label sensitivity. Blinded human check on 102 pairs, κ=0.902. Relevance: the most actionable single rule here: do not show the judge the proposed label as the system's answer. Tested on entity alignment, not tagging.

**The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment.** Sourabrata Mukherjee, Hamna Abid, Kalika Bali, Sunayana Sitaram. First posted 2 Jun 2026. Published Oct 2026, Findings of EMNLP 2026, pp. 36237-36266. Read: ABS (cut). [ACL Anthology 2026.findings-emnlp.1696](https://aclanthology.org/2026.findings-emnlp.1696/); [arXiv 2606.03043](https://arxiv.org/abs/2606.03043). 42 judges, eight languages. On subjective rubrics judges agree with each other as much as humans do yet reach only 58 to 66% of human agreement. On the one verifiable rubric most of the gap closes. Ensembles converge on the judges' shared axis rather than the human one.

**Agreement Without Alignment: Label-Free Rubric Clarification Increases LLM-Judge Consensus but Not Agreement with Humans.** Sangdo Han, Moonsoo Kim. First posted 9 Oct 2026. Read: ABS (cut). [arXiv 2610.13303](https://arxiv.org/abs/2610.13303). Pre-registered, 129,770 judge requests. Krippendorff's α up by 0.069 (95% CI 0.046 to 0.092), with no detectable change in agreement with experts (Kendall τ difference -0.007).

**Beyond the Illusion of Consensus.** Mingyang Song, Mao Zheng. Published Oct 2026, EMNLP 2026 main, pp. 35416-35433. First-posted date not looked up. Read: ABS (cut). [ACL Anthology 2026.emnlp-main.1552](https://aclanthology.org/2026.emnlp-main.1552/). 105,600 evaluations. Moving from a shared rubric to independently generated rubrics cut agreement by 62% for the two top-scoring models.

#### Inter-model and repeat agreement

**How Reliable Are LLM Annotators for Online Manipulation? Evidence from Phishing and Abuse Corpora.** Senaid Popovic, Damien Riquet, Maxime Meyer, Fabien Lauer, Yannick Parmentier. Published Oct 2026, WOAH, pp. 298-306. No arXiv title match. Read: FULL (sections on Tables 1 to 3, 6). [ACL Anthology 2026.woah-1.26](https://aclanthology.org/2026.woah-1.26/)
- Technique: annotate at the finest tier, then collapse both human and model labels to the top tier by disjunction over children, and report κ at each resolution with a held-out-human control.
- Evidence (100 emails, 9 top-level, 23 sub-labels, three annotators):

| Rater vs majority gold | Top-level κ | Sub-label κ |
|---|---|---|
| Gemma 3-27B | 0.689 | 0.372 |
| Claude Sonnet 4.5 | 0.648 | 0.469 |
| Qwen 2.5-7B | 0.456 | 0.232 |
| Human pairwise average | 0.509 | 0.349 |

- Model order changes by tier: Gemma leads at the top, Claude at sub-label. Sub-label LLM-LLM agreement is κ=0.269, below human agreement.
- Nine sub-labels have under five positives. Dropping them lifts Claude to 0.632 and humans to 0.450, and changes the held-out conclusion.
- On a separate synthetic corpus, models agree with each other more than with gold.
- Relevance: the single best demonstration that agreement is a per-tier property. Small corpus.

**Reliable but Design-Sensitive: Instrument Uncertainty in LLM Annotation.** Thomas Reiter, Christoph Kern, Fedor Miasnikov, Sofiia Nikolenko, Rob Chew, Stephanie Eckman, Frauke Kreuter. First posted 22 Sep 2026. Published Oct 2026, UncertaiNLP, pp. 312-327. Read: ABS. [ACL Anthology 2026.uncertainlp-main.20](https://aclanthology.org/2026.uncertainlp-main.20/); [arXiv 2609.35824](https://arxiv.org/abs/2609.35824). Same design repeated: median Fleiss κ 0.91. Changed design: median Cohen κ 0.76. Design and model choice raise prevalence variance by factors of 76.7 and 110.6 over sampling variance. Confidence tracks repeat outputs better than human labels.

**Agreement Without Accuracy: Six Systematic Defects an LLM Annotation Study Could Not See in Its Own Reliability Statistics.** Hon-Ren Lin. Posted 25 Sep 2026, SocArXiv (OSF). Read: ABS (cut) via Crossref. [DOI 10.31235/osf.io/n3zhf_v1](https://doi.org/10.31235/osf.io/n3zhf_v1). 1,902 items. A prompt flag was wrong on 64.7% of cases at κ=.971. Only a blind reading of 40 random items gave an anchored estimate.

**Observational Equivalence of LLM and Human Annotation.** Kentaro Nakamura, Jing Ling Tan, George Yean. First posted 24 Aug 2026, updated 28 Oct 2026. Preprint. Read: FULL (selected results and supplementary tables). [arXiv 2609.22133](https://arxiv.org/abs/2609.22133)
- Design: replications of text classification from 14 political science studies, 13 LLMs, 3 PhD coders, 165 crowd coders, same codebooks.
- Evidence: where LLMs agree, experts agree 93% of the time; where LLMs disagree, expert-expert agreement falls to 60%. Mean crowd Gini is 0.242 vs 0.379 between the 140 least and 140 most contested texts (difference +0.137, 95% CI 0.100 to 0.173).
- Technique: use cross-LLM disagreement to pick cases for human review and to refine codebooks, and report ambiguity-aware bounds.
- Tension with the papers above: this one argues human labels should not automatically take precedence, while others show consensus is not alignment. They differ in task and in what they measure. Binary or small codebooks, not a large taxonomy.

**Systematic Failure Analysis of Open-Weight LLM Annotators.** By reference, [mapping notes] Q4: agreement across prompts gives 72 to 94% accuracy at 71 to 90% coverage on clean tasks, and can make things worse where bias is shared.

**Can LLMs Evaluate What They Cannot Annotate?** Paloma Piot, David Otero, Patricia Martin-Rodilla, Javier Parapar. First posted 10 Dec 2025 **[older posting]**. Published May 2026, LREC 2026, pp. 4358-4370. Read: FULL (section on Table 7 setup only; table values not captured). [ACL Anthology 2026.lrec-1.341](https://aclanthology.org/2026.lrec-1.341/); [arXiv 2512.09662](https://arxiv.org/abs/2512.09662). LLM labels diverge from humans per item yet preserve the ranking of classifiers, tested with synthetic classifiers made by flipping predictions, using Kendall's τ. Note the classifiers are simulated.

#### Sampling and statistical correction

**Which Metrics Save the Most Human Annotation? Prediction-Powered Evaluation and Meta-Evaluation.** Mingqi Gao, Anthony Sicilia, Weiyan Shi. First posted 27 Aug 2026. Published Oct 2026, EMNLP 2026 main, pp. 19215-19254. Read: FULL (sections 5 and framing). [ACL Anthology 2026.emnlp-main.827](https://aclanthology.org/2026.emnlp-main.827/); [arXiv 2608.26638](https://arxiv.org/abs/2608.26638)
- Technique:
  1. Score everything with the cheap metric.
  2. Have humans judge a sample.
  3. Use the human sample to correct the bias of the cheap metric.
  4. Report PPSR, the saving ratio. A PPSR of 0.4 means the same statistical conclusion with about 40% fewer human-annotated segments.
- Evidence: six WMT machine translation datasets, not classification. Authors reframe metrics as a way to cut annotation cost, not replace judgement.

**Statistically Reliable LLM-Based Ranking Evaluation via Prediction-Powered Inference.** Abhishek Divekar. First posted 3 Jun 2026; accepted at the GEM workshop, ACL 2026. Read: ABS. [arXiv 2606.05308](https://arxiv.org/abs/2606.05308). 30 human annotations plus LLM judgments cut the standard error of Precision@4 from 4.45 to 3.50 (21%). In a production system, 100 human labels picked the best of three variants, confirmed by A/B test.

**[older] Prediction-powered inference.** Angelopoulos, Bates, Fannjiang, Jordan, Zrnic. Science, 10 Nov 2023. Origin. Confirmed on Crossref, not read. [DOI 10.1126/science.adi6000](https://doi.org/10.1126/science.adi6000)

**QUORUM: QUality-Optimized Routing Using Multiple annotators.** Antonio Purificato, Maria Sofia Bucarelli, Andrea Bacciu, Amin Mantrach, Fabrizio Silvestri. First posted 28 Aug 2026. Published Oct 2026, EMNLP 2026, pp. 26873-26890 (from the arXiv journal reference). Read: ABS. [arXiv 2608.27974](https://arxiv.org/abs/2608.27974). Routes each instance to human or LLM under a budget, using instance difficulty features rather than model confidence. Quality up by as much as 34.4%, cost down 8.8%, including multilingual settings.

**Reference points for what agreement to expect.** LCSHBench 93.3% concept vs 80.7% exact (Q1). UZH κ=0.540 intra-annotator and 20 of 178 reviewed paragraphs with gaps ([mapping notes] Q3, Q4). ATT&CK multi-label annotation κ=0.68 ([arXiv 2606.18166](https://arxiv.org/abs/2606.18166), ABS).

### Inferences
All untested hypotheses.
- Untested hypothesis: treating publisher tags and pipeline LLM tags as two separate silver tracks, against one blind reviewed sample, would give a more honest picture than merging them. The method comes from Vasquez, but no paper applies it to evaluation reports.
- Untested hypothesis: prediction-powered estimates would work per tier and per language if the human sample is stratified on both. PPE and PRECISE test neither taxonomy tiers nor large label sets.
- Untested hypothesis: review by cross-model disagreement would find taxonomy problems faster than random review, but random review is still needed for an unbiased estimate.

### Gaps
- I found no reliability figures for LLM judges on tier-level taxonomy tagging of long policy documents. Most judge papers score free text, translation or entity pairs.
- No published sample-size guidance for per-tier estimates over a taxonomy with hundreds to thousands of leaves. Rare leaves will have too few samples under any reasonable budget.
- Inter-rater figures for development taxonomies remain missing, as in [mapping notes] Q4.

---

## Q3. Evidence and explanation: quoted spans per label, verification, and effects on accuracy and trust

### Takeaway
A separate verification step that checks a candidate label against a located text span and the label's definition has the strongest measured accuracy effect found (precision 82.15% with it, 55.55% without, in one cybersecurity study). Whether merely asking for quotes helps is not isolated by any paper I read. On trust, I found no in-window user study of quoted evidence for taxonomy labels.

### Cited Findings

**CTI2TTP: Evidence-Grounded TTP Extraction from Cyber Threat Intelligence via Diverge-then-Converge Verification.** Yutong Cheng, Changze Li, Raihan Sultan Pasha Basuki, Qian Cui, Wei Ding, Peng Gao. First posted 25 May 2026, v2 13 Oct 2026. Preprint. Read: FULL (design overview, section 4.3). [arXiv 2605.25836](https://arxiv.org/abs/2605.25836)
- Technique:
  1. Split the report into atomic behaviours.
  2. Propose candidate labels broadly for each.
  3. Locate a text window for each behaviour with a deterministic step, not the LLM.
  4. Keep only candidates supported by both that window and the official label definition.
- Evidence (ablation on CTI2TTP-Bench with GPT-4o):
  - Removing verification: precision 82.15% to 55.55%, F1 down 23.0 points, recall down 16.1.
  - Removing span location: precision down 8.9, recall down 18.3.
  - Removing decomposition: recall 95.50% to 76.30%.
- Overall macro-F1 76.48% and 87.39% on the two datasets, and 63.5% and 29.4% better than chain-of-thought with the same model.
- Relevance: best measured evidence for per-label verification, on a supplied taxonomy (MITRE ATT&CK) in long reports. The closest analogue to a demo step. Security reports, and the authors' own new benchmark.

**[older] From Documents to Spans: Span-Centric Learning for LLM-based Explainable Medical Coding.** Xu Zhang, Wenxin Ma, Chenxu Wu, Rongsheng Wang, Zhiyang He, Xiaodong Tao, Kun Zhang, S. Kevin Zhou. First posted 16 Mar 2026, v3 14 Oct 2026. Read: FULL (selected lines). [arXiv 2603.15270](https://arxiv.org/abs/2603.15270). Model may answer "No Evidence Found" when a code cannot be substantiated. Adding evidence reasoning raises Micro-F1 from 36.8 to 43.9 and Macro-F1 from 13.9 to 17.9. Adding extracted spans raises them to 48.9 and 22.7. These gains involve training on evidence annotations, so they do not show what prompting for quotes alone does.

**UZH Shared Task and POINTERS.** POINTERS required a quoted phrase for every tag decision, used no retrieval pre-filtering, and placed first on tag micro-F1 at 0.459 (overview, [mapping notes] Q3).
- [POINTERS at UZH Shared Task 2026, Sohom Sen, Avina Nakarmi, Xun Song, Aritra Dasgupta, Jul 2026, ArgMining, pp. 145-150, 2026.argmining-1.16](https://aclanthology.org/2026.argmining-1.16/). Read: FULL (abstract, introduction).
- The quote and the absence of pre-filtering are confounded, so this does not show that quoting raises accuracy.
- Discrepancy: this paper describes a 126-code vocabulary, while the overview as recorded in [mapping notes] gives 141 tags. Unresolved.
- Relevance: UN text, a supplied taxonomy, a sub-8B local model.

**Agreement-gated correction with the full taxonomy in view.** [Thakaa at AraGenre 2026, Hassan Alqaeri, Meshal Alamr, Abdullah Aldahlawi, Oct 2026, pp. 300-312, 2026.arabicnlp-sharedtasks.38](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/). Read: FULL (section 4). Two models re-predict with all 74 definitions, and a label changes only where both agree against the base. The base scores 0.7139 and the full submitted system 0.7352, but that system also includes later post-processing, so the gain is not from this step alone. Of 1,178 proposed cross-branch moves, 460 fell in author-screened directions and were applied. The authors stress that components were picked on the public leaderboard and that agreement here is a conservatism device, not evidence of an ensemble effect.

**From Rubrics to Reliable Scores (RULERS).** Yihan Hong, Huaiyuan Yao, Bolin Shen, Wanpeng Xu, Hua Wei, Yushun Dong. First posted 13 Jan 2026 **[older posting]**. Published Oct 2026, EMNLP 2026 main, pp. 13001-13021. Read: ABS. [ACL Anthology 2026.emnlp-main.566](https://aclanthology.org/2026.emnlp-main.566/); [arXiv 2601.08654](https://arxiv.org/abs/2601.08654). Locks the rubric, requires evidence-grounded judgments and calibrates to human scales. Ablation attributes the gain to the combination, not to calibration alone. Scoring, not tagging.

**Medical coding with rationales.**
- [ICDAGENT, Ziyi Yin, Yuanpu Cao, Ting Wang, Jinghui Chen, Fenglong Ma, Jul 2026, ACL 2026 Long, pp. 14131-14149](https://aclanthology.org/2026.acl-long.643/). Read: ABS (cut). A coding agent proposes and a critic agent verifies.
- [KREL, Xubin Chen et al., first posted 21 Aug 2026, Oct 2026, Findings of EMNLP, pp. 32024-32039](https://aclanthology.org/2026.findings-emnlp.1504/). Read: ABS (cut). Also in [mapping notes] Q2.
- [Evaluation and LLM-Guided Learning of ICD Coding Rationales, Mingyang Li et al., Mar 2026, EACL Long, pp. 4969-5003](https://aclanthology.org/2026.eacl-long.232/) **[older]**. Read: ABS (cut). Separates faithfulness from plausibility, on a new annotated dataset.
- [CPTCoder, Benlu Wang et al., Jul 2026, ACL Demo, pp. 605-614](https://aclanthology.org/2026.acl-demo.60/). Read: ABS. Per-code probabilities from n-best reweighting flag uncertain cases for review, with fewer near-miss errors. Micro-F1 0.61 and 0.51, 12 and 5 points above the strongest baseline.

**Are self-explanations good evidence?**
- [A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification](https://aclanthology.org/2026.trustnlp-main.44/). Stephanie Brandl, Oliver Eberle. First posted 4 Oct 2024 **[older posting]** ([arXiv 2410.03296](https://arxiv.org/abs/2410.03296)). Published Jul 2026, TrustNLP, pp. 563-583. Read: ABS. Alignment with human rationales depends heavily on text length and task complexity, though self-explanations give faithful subsets of tokens.
- [Lost in Interpretation: The Plausibility-Faithfulness Trade-off in Cross-Lingual Explanations](https://arxiv.org/abs/2605.19274). Somnath Banerjee, Pranav Jha, Rima Hazra, Animesh Mukherjee. First posted 19 May 2026. Read: ABS. Three tasks, 5 languages. English-pivot explanations can match human spans better while being less causally grounded, with comprehensiveness up to 5.7 times worse than native-language conditions, while task accuracy holds. Authors advise auditing in the input language.
- [Recognition Is Not Grounding (MULTIFINGROUND)](https://arxiv.org/abs/2610.19432). Lingfei Qian et al. First posted 14 Oct 2026. Read: ABS. 90 issuers, English, French, Greek. Eight models reach 97.3 to 99.4% concept recognition yet miss at least one target fact in 26.6 to 41.9% of correctly recognised groups. Right label does not mean complete evidence.

**Trust and human reaction.**
- The curator in [arXiv 2610.01616](https://arxiv.org/abs/2610.01616) revised some of their own labels on seeing LLM evidence. One expert, qualitative.
- The review in [ReLTEx](https://arxiv.org/abs/2608.10970) (Q5) is about taxonomy edits, not labelling.
- [Believing without Seeing, He et al., ACL 2026 Long](https://aclanthology.org/2026.acl-long.673/). Read: ABS (cut). Adjacent, about vision-language models: prior work found that explanations can convince users that wrong predictions are correct, and the paper proposes quality scores to counter overreliance. Not a tagging study.

### Inferences
All untested hypotheses.
- Untested hypothesis: a deterministic check that each quote actually appears in the chunk would catch fabricated evidence cheaply. That check is close to CTI2TTP's deterministic location step, but no paper measures it in isolation for tagging.
- Untested hypothesis: on noisy PDF text, exact-substring checks will reject true quotes broken by parsing, so the failure rate should be measured before treating a failed quote as a wrong label.
- Untested hypothesis: for French and Spanish chunks, showing the quote in the source language is both more faithful (Banerjee et al.) and more useful to reviewers, with an English gloss added.

### Gaps
- **No in-window paper isolates the effect of requiring quotes on tagging accuracy with everything else held fixed.** The nearest are ablations in CTI2TTP and the older span-learning paper.
- **No in-window user study on whether quoted evidence raises appropriate trust in taxonomy labels.** My search of ACL Anthology abstracts for user studies with explanations and trust found only four hits, none about labels. arXiv search was limited by rate limits.

---

## Q4. Abstention and "no label fits": thresholds, conformal prediction on hierarchies, back-off to a parent, over-assignment

### Takeaway
Backing off to a parent is well developed for classifiers, through hierarchical conformal prediction, but I found no paper applying it to LLM taxonomy tagging of text. Conformal guarantees are fragile for LLMs: changing a prompt or the dataset erodes coverage, so a calibration set must match deployment. Over-assignment is real but varies by model, and LLMs can also under-assign, so output format and thresholds need measuring per model.

### Cited Findings

#### Back-off to a parent

**[older] Hierarchical Conformal Classification (HCC).** Floris den Hengst, Inès Blin, Majid Mohammadi, Syed Ihtesham Hussain Shah, Taraneh Younesian. First posted 18 Aug 2025. Read: FULL (results, user study). [arXiv 2508.13288](https://arxiv.org/abs/2508.13288)
- Technique:
  1. Run a calibration step on held-out labelled data.
  2. Build a prediction set that may mix leaves and ancestor nodes, trading set size against how many leaves are covered, using a parameter β.
  3. Keep the coverage guarantee over the leaves.
- Evidence: coverage met; significantly smaller prediction sets than baselines on the text set (DBpedia14) and others, at a moderate cost in covered leaves. Four annotators on 500 samples preferred HCC over standard conformal in about 57% of cases for text and 71% for images (Fleiss κ 0.47 and 0.53).
- Relevance: the origin of the "answer at parent tier with a guarantee" idea, and a demo-friendly output. Used a BERT classifier, not an LLM. Needs scores across labels, and the authors note optimisation scale limits.

**[older] Conformal Prediction in Hierarchical Classification with Constrained Representation Complexity.** Thomas Mortier, Alireza Javanmardi, Yusuf Sale, Eyke Hüllermeier, Willem Waegeman. First posted 31 Jan 2025, v3 10 Apr 2026. Read: ABS. [arXiv 2501.19038](https://arxiv.org/abs/2501.19038). Returns internal nodes as prediction sets, with a variant allowing smaller combinations of nodes.

**Hierarchical Group-Conditional Conformal Risk Control for Selective Prediction in Language Models (HG-CRC).** Murilo Salem, Luísa Böhm, Daniel Pontes, Anderson Ferrugem. First posted 27 Jul 2026. Preprint. Read: FULL (introduction, summary). [arXiv 2607.24562](https://arxiv.org/abs/2607.24562)
- Technique: calibrate a separate risk threshold for each node of a group hierarchy with a Bonferroni correction, then at inference use the deepest certified node whose threshold accepts the input, falling back to coarser nodes.
- Evidence: standard risk control violates its budget in up to 47% of trials under mild group shift. HG-CRC shows 0% empirical violation on ARC for two models, read by the authors as upper bounds. On MMLU-Pro models either abstain entirely or retain a worst-group excess risk of 0.014. It costs 22 to 37 percentage points of coverage.
- Important: the hierarchy here is of user or domain groups, not of labels, and the tasks are multiple choice. It is an analogy for tier back-off, not a test of it.

**Coherent Hierarchical Multi-Label Learning to Defer for Medical Imaging.** Joshua Strong, Pramit Saha, Emma Sun, Helen Higham, Alison Noble. First posted 4 May 2026. Read: ABS. [arXiv 2605.02734](https://arxiv.org/abs/2605.02734). Deferring each label independently can contradict the taxonomy. Proposes a decoder that removes such contradictions exactly. Imaging, but the incoherence problem applies directly to per-leaf abstention.

**TaxoAgent: Utility-Aligned Policy Optimization for Hierarchical Semantic Classification.** Zhiming Lin, Yihao Zhong, Liwei Hou, Saisai Hu, ShaoYan He, Chen Rongru, Xuan Yang, Siyue Liu. Published Oct 2026, EMNLP 2026 main, pp. 23149-23171. No arXiv title match. Read: FULL (abstract, section on metrics and out-of-scope). [ACL Anthology 2026.emnlp-main.1006](https://aclanthology.org/2026.emnlp-main.1006/). Parent-to-child trajectory scored for correctness, consistency, semantic proximity and coherence. Out-of-scope F1 up by as much as 3.68 points (intent data). Requires training, and reports a hierarchy inconsistency rate, a useful scorecard column. Mainly a prediction method; other researchers cover that.

#### Conformal and risk control for LLMs: limits

**Conformalized Large Language Models under Configuration Shift.** Yuqicheng Zhu, Jialin Yu, Lin Li, Gengyuan Zhang, Zhen Yang, Steffen Staab, Puneet K. Dokania, Philip Torr, Jie Tang, Evgeny Kharlamov. First posted 2 Aug 2026. Published Oct 2026, EMNLP 2026 main, pp. 18261-18289. Read: ABS. [ACL Anthology 2026.emnlp-main.786](https://aclanthology.org/2026.emnlp-main.786/); [arXiv 2608.01460](https://arxiv.org/abs/2608.01460). 9 LLMs, 4 datasets, 4 scores. Changing prompt template, temperature or quantisation often drives coverage below target, while set size is largely preserved. Bound-based recalibration and calibration ensembles recover much of it.

**When Can Conformal Risk Control Certify LLM Outputs?** Varun Kotte. First posted 27 Jun 2026, v2 7 Sep 2026. Read: ABS. [arXiv 2606.29054](https://arxiv.org/abs/2606.29054)
- Feasibility test: when base risk μ exceeds target α, any distribution-free method must abstain on at least (μ-α)/(M-α) of inputs.
- Heuristic abstention violates targets on 7.5 to 12.5% of settings. Under cross-dataset shift the target fails on 14 of 16 transfers.
- Relaxing the target to α=0.40 certifies only 19% of classification settings.
- Relevance: gives a calculation to run before promising a guarantee. See also PASC, same author, [mapping notes] Q4.

**Uncertainty-Aware Multi-Label Routing of Clinical Text.** By reference for metadata, [mapping notes] Q4. Read: ABS. [ACL Anthology 2026.bionlp-1.16](https://aclanthology.org/2026.bionlp-1.16/). At α=0.10, Mondrian calibration gave 0.891 mean recall coverage, mean set size 1.70, candidate-label precision 0.642 and 0.61 false-positive labels per report. Lowest AURC 0.084 under min-margin abstention. Authors treat set-valued output as candidates for review, not automatic selection. Relevance: the best model for how to report abstention, with review burden next to coverage.

**Benchmarking System One decision models against trained classifiers and language models for automated decision gates.** Amir Rafe, Subasish Das. First posted 29 Sep 2026. Read: ABS (cut). [arXiv 2610.00346](https://arxiv.org/abs/2610.00346). A guaranteed 5 percent risk leaves 0.528 of intent decisions automated, and an in-scope threshold still accepts 0.310 of out-of-scope requests.

**BOLT: Benchmarking Open-World Learning for Text Classification.** Chuan Qin, Xi Chen, Jinpeng Li, Hengshu Zhu. Published Jul 2026, Findings of ACL, pp. 13624-13658. No arXiv title match. Read: ABS. [ACL Anthology 2026.findings-acl.667](https://aclanthology.org/2026.findings-acl.667/). 12 datasets, 22 methods. Most methods overfit training distributions and struggle with unseen classes.

**One Score, Two Decisions** (Jiang et al., [mapping notes] Q4): top-two margin helps with fixed candidates but can remove the signal for "the list has no answer". Directly relevant: the score that picks between leaves may be wrong for deciding that no leaf fits.

#### Calibration and thresholds
- **How to Estimate Whether You Have Found Several Needles in a Haystack** (Henning et al., [mapping notes] Q4). Standard calibration binning misleads when most labels are negative.
- **When High Accuracy Hides Poor Calibration.** Guilherme Fonseca, Gabriel Prenassi, Washington Cunha, Leonardo Chaves Dutra da Rocha, Marcos André Gonçalves. Published Jul 2026, ACL Long, pp. 45888-45900. No arXiv title match. Read: ABS (cut). [2026.acl-long.2128](https://aclanthology.org/2026.acl-long.2128/). ECE and Brier become biased when most predictions are right; proposes Balanced Brier Score.
- **Beyond Global Calibration.** Claudio Moisés Valiense de Andrade et al. Published Oct 2026, UncertaiNLP, pp. 683-692. No arXiv title match. Read: ABS. [2026.uncertainlp-main.41](https://aclanthology.org/2026.uncertainlp-main.41/). Low global error coexists with class-specific errors, and high-confidence errors concentrate in specific confusions.
- **RAPT: Retrieval-Augmented Post-hoc Thresholding for Multi-Label Classification.** Lasal Jayawardena, Nirmalie Wiratunga, Ikechukwu Nkisi-Orji, Darren Nicol. First posted 15 May 2026. Read: ABS. [arXiv 2605.16535](https://arxiv.org/abs/2605.16535). Sets the threshold per document from similar documents' outcomes, naming OCR noise and varying label count as motives. Beat global and label-wise static thresholds across one industrial and six public datasets (0.87 Macro-F1 in the industrial setting). Requires labelled past cases.
- **OPINE.** Mengting Zhang, Gaofeng Pan, Zhixiong Zhang, Yang Li, Guangyin Zhang. Published Jul 2026, Findings of ACL, pp. 32313-32333. Read: ABS (cut). [2026.findings-acl.1617](https://aclanthology.org/2026.findings-acl.1617/). Independent per-label scores, then a quantile decoding rule and pairwise refinement for confusable labels, instead of direct label generation.
- **When Models Hesitate.** Jasper Meynard Arana et al. Published Jul 2026, ACL SRW, pp. 816-826. Read: ABS (cut). [2026.acl-srw.72](https://aclanthology.org/2026.acl-srw.72/). Variation across repeated generations is a black-box uncertainty signal, competitive with semantic entropy.
- **To Predict or Not to Predict?** Nouran Khallaf, Serge Sharoff. Published May 2026, LREC, pp. 152-168. Read: ABS. [2026.lrec-1.11](https://aclanthology.org/2026.lrec-1.11/). Multilingual. Softmax methods decline in low-resource or shifted settings. Abstaining on the 10% most uncertain raised macro-F1 from 0.81 to 0.85 in one task.
- **Selection Effect.** Dmitry Tikhomirov, Andrey Savchenko. Published Oct 2026, EMNLP main, pp. 24095-24114. Read: ABS. [2026.emnlp-main.1049](https://aclanthology.org/2026.emnlp-main.1049/). No compact LLM verifier improved the quality-cost frontier behind an encoder confidence gate, because the gate forwards a slice on which compact LLMs are no better. Encoder-only gating and split-conformal acceptance were preferable. Emotion labels only.

#### Over-assignment and under-assignment
- **Evaluating Open-Source LLMs for Multi-Label ATT&CK Technique Classification on CTI Reports.** Ahmed Ryan, Saad Sakib Noor, Md Erfan, Shaswata Mitra, Sudip Mittal, Md Rayhanur Rahman. First posted 16 Jun 2026. Read: FULL (error analysis lines). [arXiv 2606.18166](https://arxiv.org/abs/2606.18166). Best micro-F1 0.22 over 114 labels. One model over-predicts (micro recall 0.21 to 0.22, precision 0.09 to 0.10). Another under-predicts (precision 0.12 to 0.27, recall 0.00 to 0.04). 795 of 2,076 sentences are negatives, included to measure false positives: a useful design to copy.
- **Modulating Multi-Label Tendency** (Choo, Hong, [mapping notes] Q4). Single-label defaults of 96% and 54% fall to 19% and 0.0% with a different output format. So output format moves label count.
- **[older] Large Language Models Do Multi-Label Classification Differently.** EMNLP 2025. Title confirmed on the ACL Anthology, paper not read: [2025.emnlp-main.126](https://aclanthology.org/2025.emnlp-main.126/). META.
- **LLMs as Annotators.** Yahui Zhao, Clemencia Siro, Laura Hollink. Published May 2026, LREC, pp. 10883-10896. Read: ABS. [2026.lrec-1.852](https://aclanthology.org/2026.lrec-1.852/). Over-predict for identity and colonial terms, under-predict for shifted or figurative uses, in historical Dutch. So over-assignment can be concentrated in particular labels.
- Golub et al. ([mapping notes] Q4): LLMs often choose broader concepts and ignore indexing policy; 37.5% and 56.1% fully incorrect terms.

**[older] Conformal Labeling (Model-agnostic Selective Labeling with Provable Statistical Guarantees).** Huipeng Huang, Wenbo Liao, Huajun Xi, Hao Zeng, Mengchen Zhao, Hongxin Wei. First posted 16 Oct 2025, v3 14 Feb 2026. Read: FULL (introduction). [arXiv 2510.14581](https://arxiv.org/abs/2510.14581). Selects which AI labels to accept while controlling the false discovery rate, leaving the rest to humans. Example: 63.73% of one task labelled at FDR 9.85%. Tested on non-taxonomy labelling tasks. Relevance: maps well onto "auto-accept these, send the rest to review".

### Inferences
All untested hypotheses.
- Untested hypothesis: a simple rule, accept a leaf only if its score clears a calibrated threshold, otherwise emit the parent if the children's combined score clears, otherwise abstain, will give a useful coverage curve per tier. This is a plain cousin of HCC and HG-CRC, and no paper tests it on LLM tags.
- Untested hypothesis: an LLM without per-label scores could use repeat-run stability (Arana et al.) or per-label yes/no scores (OPINE) as the score to calibrate. Whether that is exchangeable enough for conformal validity is unknown, and Zhu et al. warn the calibration must use the same prompt and model.
- Untested hypothesis: an explicit "none fits" option plus negative chunks in the test set (as in the ATT&CK dataset) is the cheapest way to measure over-assignment.

### Gaps
- **No in-window paper applies conformal prediction over a label hierarchy to LLM text tagging.** Searches: arXiv with "conformal" plus hierarch*, taxonomy, multi-label and text classification (11, 16, 5 and 0 hits in the window); ACL Anthology abstracts. Most hits use "hierarchical" in other senses, such as grouped data ([Generalized Hierarchical Conformal Prediction, arXiv 2608.15500](https://arxiv.org/abs/2608.15500)).
- No figures found for how often LLMs choose "none of the above" correctly across a large taxonomy. arXiv "none of the above" returned 4 in the window, not screened as relevant.
- No direct measure of over-assignment across taxonomy size.

---

## Q5. Taxonomy gaps and evolution: text the taxonomy cannot place, new nodes, overlapping or unused labels, review loops

### Takeaway
There is usable material on each piece but no paper covers the whole loop on a supplied taxonomy: detecting unplaced text, proposing a node and human review. The closest measured rate is UZH's 20 of 178 reviewed paragraphs with genuine gaps. Reference-free taxonomy metrics and corpus-grounded exhaustivity scores could drive a gap report without gold labels.

### Cited Findings

#### Measuring gaps directly
- **UZH overview.** 20 of 178 expert-reviewed paragraphs (about 11%) have genuine taxonomy gaps, on UN education resolutions with 141 tags. By reference: [mapping notes] Q3. Most relevant figure for domain, from a small sample.
- **Human-LLM Collaborative Inductive Coding for Conceptualizing K-12 Educator AI Use.** Alex Liu, Min Sun, Lief Esbenshade, Michael Xiao, Victor Tian, Zachary Zhang, Kevin He. First posted 30 Jul 2026. Read: ABS. [arXiv 2607.28889](https://arxiv.org/abs/2607.28889). 45,000 messages. LLMs drafted labels, and humans kept authority over definitions and merges. Three trained coders then applied the codebook to 2,560 messages and added five codes the LLM phases had not surfaced. Final: 72 items, 19 categories, six domains. So human coding of a sample found gaps the LLM missed.
- **Monitoring Global Aid Flows** (Luo et al. 2025 **[older]**): 36 aid areas not tracked in the OECD taxonomy. By reference: [mapping notes] Q3. Closest domain analogue.
- **Awareness Without Synthesis** (Bossler et al., May 2026): hidden vocabulary fragmentation in knowledge graphs, α=0.752. By reference: [mapping notes] Q4.
- **Cattle reporting categories** (Raymond et al., Sep 2026): target vocabulary lacks needed granularity. By reference: [mapping notes] Q3.

#### Scoring a taxonomy without labels

**Reference-Free Evaluation of Taxonomies.** Pascal Wullschleger, Majid Zarharan, Donnacha Daly, Marc Pouly, Jennifer Foster. First posted 16 May 2025 **[older posting]** ([arXiv 2505.11470](https://arxiv.org/abs/2505.11470)). Published Jul 2026, Findings of ACL, pp. 25489-25507. Read: FULL (sections on CSC, NLIV, Figures 4 and 5 text; correlation values are in figures and not captured). [ACL Anthology 2026.findings-acl.1273](https://aclanthology.org/2026.findings-acl.1273/). Extends [mapping notes] Q4.
- Technique 1, Concept Similarity Correlation:
  1. Compute taxonomic similarity between all concept pairs from their root paths.
  2. Embed each concept's description.
  3. Report Kendall's τ between the two similarity lists. Low τ suggests misplaced concepts.
- Technique 2, NLI validity:
  1. Turn each parent-child edge into sentences using Hearst patterns.
  2. Ask an entailment model if the child entails the parent (strong) or at least does not contradict it (weak).
- Evidence: five taxonomies. Both correlate with F1 against ground truth, and with macro F1 of downstream hierarchical classification, significant at α=0.001.
- Relevance: can be run on a candidate UN taxonomy before any tagging, and needs good concept descriptions.

**TaxCE: A Framework for Automated Taxonomy Construction and Evaluation at Scale.** Sandeep Sricharan Mukku, Albert Aristotle Nanda, Rohit Pyati. First posted 31 Aug 2026. Published Oct 2026, EMNLP 2026 Industry, pp. 1280-1301. Read: FULL (metric definitions). [ACL Anthology 2026.emnlp-industry.76](https://aclanthology.org/2026.emnlp-industry.76/); [arXiv 2608.30614](https://arxiv.org/abs/2608.30614)
- Exclusivity (overlap): 100 × (1 − mean pairwise cosine similarity across leaf embeddings).
- Exhaustivity (coverage): extract actionable segments from the corpus, independently of any taxonomy. A segment is covered if its cosine similarity to some leaf exceeds 0.65, with sensitivity analysis in an appendix. Score is the share covered.
- Granularity is the third score (definition not read).
- Average improvement over the strongest baseline: 11.8, 20.5 and 15.7 points. A metric-driven refinement loop applies targeted corrections. Built for constructing a taxonomy, but the metrics apply to a supplied one.
- Relevance: uncovered segments are the raw material of a gap report, and high-overlap leaf pairs are a merge list. The 0.65 threshold is embedding-specific.

#### Proposing and validating new nodes

**ReLTEx: Reliable LLM-based Taxonomy Expansion.** Zeinab Ghamlouch, Mehwish Alam. First posted 11 Aug 2026. Preprint. Read: FULL (sections 4, 6, tables 2, 3, ablation). [arXiv 2608.10970](https://arxiv.org/abs/2608.10970)
- Technique:
  1. Walk the tree depth-first.
  2. For each node, prompt with only the root-to-node path and its existing children, asking for k new children at the same granularity.
  3. Remove lexical duplicates.
  4. Validate each parent-child edge. Three strategies compared: semantic similarity, an LLM judge, and a classifier trained on existing edges. The classifier performed best.
  5. Recurse on accepted nodes.
- Evidence:
  - Masked-recovery recall is modest (SemEval Environment R@K 28.85 to 44.23; Schema.org 11.06 to 23.28).
  - Three human annotators, on 20% of eligible parents with models anonymised, rated hierarchical correctness 0.96 to 0.98 on SemEval Environment and 0.898 to 0.988 for the Schema.org values I captured.
  - A local-context prompt beat wider context.
- Relevance: a clean validator recipe. It generates from the taxonomy, not from unplaced text, so it does not itself show what the corpus needs.

**Event Ontology Expansion via LLM-Based Conceptualization (ConceptE).** Weicheng Ren, Zixuan Li, Long Bai, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng. First posted 19 Jun 2026. Published Oct 2026, Findings of EMNLP, pp. 15473-15489. Read: ABS (cut). [ACL Anthology 2026.findings-emnlp.737](https://aclanthology.org/2026.findings-emnlp.737/); [arXiv 2606.21048](https://arxiv.org/abs/2606.21048). The LLM gives each unmatched instance a short concept name and description, clusters those rather than raw text, then attaches clusters to the ontology. The closest method to going from text to a proposed node.

**Other generation and discovery work.**
- [BEATS, Yung-Yu Shih et al., first posted 3 Jun 2026, SIGIR 2026 Industry](https://arxiv.org/abs/2606.04909). Read: ABS (cut). Iterative rounds: developer quality check, then domain-expert annotation, with prompts revised after each round.
- [TLSA, Wenxi Xu et al., Jul 2026, ACL Long, pp. 19030-19046](https://aclanthology.org/2026.acl-long.869/). Read: ABS (cut). Category discovery with LLM naming of new clusters and forced renaming of overlapping ones.
- [SC-Taxo, Shiqiang Cai et al., first posted 1 May 2026](https://arxiv.org/abs/2605.00620). Read: ABS (cut). Taxonomy generation with consistency between levels.
- [VITC, Yuhang Niu et al., Jul 2026, ACL Long, pp. 6092-6107](https://aclanthology.org/2026.acl-long.275/). Read: ABS (cut). Taxonomy completion with synthesised images. Low relevance.
- [DeLFGCD, Henry Peng Zou et al., Mar 2026, EACL Long](https://aclanthology.org/2026.eacl-long.358/) **[older]**. Read: ABS (cut).
- [SkiLLens, Alessia De Santo et al., Mar 2026, EACL Industry, pp. 877-885](https://aclanthology.org/2026.eacl-industry.65/) **[older]**. Read: ABS. Multilingual human-in-the-loop pipeline finding novel skills across European job ads and linking to ESCO. A strong analogue for multilingual gap detection, with no numbers in the abstract.
- [GRACE, John Seon Keun Yi et al., first posted 3 Sep 2026](https://arxiv.org/abs/2609.04442). Read: ABS (cut). Defers a claim to expert review only when priority-weighted uncertainty exceeds the verification cost, and promotes verified claims. A costed review rule, not about taxonomies.

#### Fixing label boundaries and overlaps

**Mosaic: Automated Intent Description Generation for Multi-Label Classification.** Ajay Chatterjee, Eric Crestan, Raunak Saha, Rotem Mordoch, Swarnangsu Acharyya. Published Oct 2026, EMNLP 2026 Industry, pp. 545-556. No arXiv title match. Read: ABS plus introduction. [ACL Anthology 2026.emnlp-industry.33](https://aclanthology.org/2026.emnlp-industry.33/)
- Technique: encode unlabelled text with a multilingual encoder, cluster it, associate clusters with an existing taxonomy, and write discriminative descriptions from that evidence, with light human review.
- Evidence: on a 140-class production system, approaches expert descriptions and cuts taxonomy revision from 2-4 weeks to under 30 minutes. On public NLU++ and Multi³NLU++ it matches or exceeds expert descriptions; cluster grounding adds a further 2.6 pp on one measure.
- Relevance: assumes the taxonomy exists, as in the use case. Short utterances, not reports.

**CoDA: A Human-LM Collaborative Codebook Development Assistant.** Jaspreet Ranjit, Swaminathan Chellappa, Augusto Rivas Costante, Chandan Manjunath, Laura Onasch-Vera, Hailey Winetrobe Nadel, Eric Rice, Swabha Swayamdipta. First posted 16 Oct 2026. Published Oct 2026, EMNLP Demos, pp. 547-557. Read: ABS. [ACL Anthology 2026.emnlp-demos.53](https://aclanthology.org/2026.emnlp-demos.53/); [arXiv 2610.24157](https://arxiv.org/abs/2610.24157). Experts review targeted wrong predictions and feed changes into the codebook. Comparable to expert codebooks, development time from several weeks to two hours, 5× fewer annotated examples. A demo paper, so a good reference for a review interface.

**[older] TaxMorph (Hierarchical Text Classification with LLM-Refined Taxonomies).** Jonas Golde, Nicolaas Paul Jedema, RaviKiran Krishnan, Phong Le. First posted 26 Jan 2026, published Mar 2026, EACL Long, pp. 214-228. Read: ABS. [ACL Anthology 2026.eacl-long.10](https://aclanthology.org/2026.eacl-long.10/); [arXiv 2601.18375](https://arxiv.org/abs/2601.18375). LLM renames, merges, splits and reorders. Up to +2.9pp F1 over human taxonomies. Warning for the use case: this optimises for the model, and a supplied taxonomy may have to stay fixed.

**Codebook schema testing.** [From DQI to C-DQI, Esther Shizgal et al., first posted 20 Oct 2026](https://arxiv.org/abs/2610.27937). Read: ABS. Inductive schemas gain higher inter-model agreement than top-down ones. In light of Q2, that agreement alone is not proof of quality; the authors also use a validated external proxy.

**FrenchNews-7** (Q2): Societe (precision 0.577) absorbs boundary ambiguity. A published example of diagnosing an overlapping label by confusion and blinded human agreement.

### Inferences
All untested hypotheses.
- Untested hypothesis: a gap report built from three streams would work: chunks where all labels were abstained, chunks whose nearest leaf falls below an exhaustivity threshold, and cluster names in the ConceptE style. Each proposed node then gets a ReLTEx-style validation and human review.
- Untested hypothesis: unused labels will be simple to find with a count per tier, and overlapping labels by combining high description similarity (TaxCE) with high model confusion or co-assignment. I found no in-window paper that measures unused labels directly.
- Untested hypothesis: the gap rate per 1,000 chunks could be reported with a reviewed sample, using UZH's 11% only as a rough prior, since it comes from 178 selected paragraphs of a different corpus.

### Gaps
- No in-window end-to-end paper on a supplied large taxonomy going from unplaced text to reviewed new nodes with a precision figure for proposed nodes.
- No in-window paper with a dedicated method for detecting unused labels. Overlap is covered only indirectly.
- arXiv "taxonomy expansion" gave 1 hit and "taxonomy completion" 0 in the window, which seems low. arXiv search coverage may be limited by exact wording.
- [TEAMMix, arXiv 2608.11044](https://arxiv.org/abs/2608.11044) (taxonomy enrichment for weak supervision) was seen in [mapping notes], and not re-read here.

---

## Q6. Multilingual behaviour: does a tier approach hold across languages, does translating help, and which items make the best demo

### Takeaway
Direct evidence on tiers across languages is thin. Language gaps are real and uneven, and a translated test set can overstate performance slightly (about +1.3 accuracy points in the one paired study). I found no in-window paper that directly compares translating before tagging with tagging in the source language against a taxonomy. For the demo, the evidence best supports a tier-by-language scorecard with a human agreement line, together with a gap report; hypotheses on which is most compelling are listed separately.

### Cited Findings

#### Tiers across languages

**From Synthetic to Localized: Benchmarking Multilingual Intent Classification in Logistics Customer Service.** Haoyu He, Jinyu Zhuang, Haoran Chu, Shuhang Yu, Hao Wang, Kunpeng Han. First posted 24 Mar 2026 **[older posting]**, under the title *From Synthetic to Native* ([arXiv 2603.23172](https://arxiv.org/abs/2603.23172)). Published Oct 2026, EMNLP 2026 Industry, pp. 2078-2092. Read: FULL (sections 5.1, 5.2, Tables 1, 2, 6). [ACL Anthology 2026.emnlp-industry.121](https://aclanthology.org/2026.emnlp-industry.121/)
- Technique: 13 parents and 17 leaves. Run 13-way, 17-way flat and 17-way hierarchical. Pair machine-translated and human-localized test items from the same source, 674 per language. Test zero-shot on languages not seen in training.
- Evidence: machine-translated input gives higher accuracy in 42 of 45 settings, mean +1.319 points (95% CI 0.159 to 2.479). Macro-F1 in 41 of 45, mean +1.819. Qwen3-Max: 94.77 at 13-way, 91.15 flat 17, 92.92 hierarchical 17.
- Relevance: the only paired tier-and-language design found. Only 30 short-text labels and no French. It tests evaluation on translated text, not translating before tagging.

**LCSHBench** (Q1). Per-language exact recall@200, fine-tuned vs text-embedding-3-large: English 0.696 vs 0.710; German 0.704 vs 0.651; Russian 0.505 vs 0.543; Chinese 0.618 vs 0.600; Japanese 0.593 vs 0.537; Korean 0.668 vs 0.517; Arabic 0.689 vs 0.540. The ranking changes with language. Table 3 as I read it has no French or Spanish row, and per-language scores are for exact match only.
- Note also that the language gap was fixed by low-cost fine-tuning on the benchmark's own data: 0.407 to 0.659.

**AraGenre 2026** (Q1). English definitions applied to Arabic text. Broad 0.89 vs specific 0.51 macro-F1 for one team. Only one source language.

**MOSAIC radiology, BioELX, LongBEL, MDPO, Volkov SDG lexicon, Fallatah.** By reference, [mapping notes] Q3 and Q4. These give English-first gaps (for example BioELX EN 91.2, ES 69.6; LongBEL EN 80.0, FR 77.6) for linking, not tier tagging. Volkov: assignment disagreement on translated pairs en-es 33.0%.

#### Does translation help, or a language gap persist?

**Do Generative Large Language Models Close the Language Gap in Multilingual Discourse Classification?** Frederik Weber Henriksen. Published 16 Sep 2026 per Crossref (OpenAlex gives 20 Aug 2026), Computational Communication Research. Read: ABS via OpenAlex; publisher page returned 403. [DOI 10.5117/ccr2026.1.11.henr](https://doi.org/10.5117/ccr2026.1.11.henr). 6,000 posts in Danish, Swedish, German; 33 model settings. Configurations on German or on English translations reliably beat those for Danish and Swedish. Fine-tuned encoders set the ceiling at macro-F1 about 0.85 in German. Also a useful validation method: group items by model agreement, then model failures with mixed effects. High-agreement share 76.6% vs 52.3% across tasks.

**Quantifying the Impact of Translation Errors on Multilingual LLM Evaluation.** Klaudia Thellmann, Bernhard Stadler, Michael Färber, Jens Lehmann. First posted 24 May 2026. Published Jul 2026, ACL Long, pp. 41311-41334. Read: ABS. [ACL Anthology 2026.acl-long.1916](https://aclanthology.org/2026.acl-long.1916/); [arXiv 2605.24904](https://arxiv.org/abs/2605.24904). Target-side translation errors are associated with percentage-point drops in translated accuracy, even after controlling for English correctness. Numbers not in the abstract.

**Predicting Multilingual Classification and Translation Performance of LLMs with Cross-Lingual Alignment.** Adnan Al Ali, Kathy Hämmerl, Jindřich Libovický, Alexander Fraser. First posted 4 Aug 2026, v2 24 Aug 2026; accepted to EMNLP 2026. Read: ABS. [arXiv 2608.03446](https://arxiv.org/abs/2608.03446). 27 alignment scores. Alignment with English predicts performance, supporting the idea of English as an internal pivot.

**Judges across languages.**
- [Rank Reversal in Multilingual LLM Judges](https://arxiv.org/abs/2608.22432). Alhasan Mahmood, Samir Abdaljalil, Hasan Kurban. First posted 23 Aug 2026, v2 6 Sep 2026. Read: ABS. Top-ranked model alternates across eight languages including Spanish. Label-free double-centring raises rank consistency τ from 0.650 to 0.902 and panel agreement with human gold from 68.7% to 76.6% (7 languages).
- [Cross-Lingual Stability of LLM Judges](https://aclanthology.org/2026.mme-main.8/). Isaac Chung, Linda Freienthal. First posted 2 Feb 2026, Mar 2026 **[older]**. Read: ABS. Rank inversions on pragmatic judgments across Finno-Ugric languages.

**Other language items.**
- Banerjee et al., Q3: explanations in English are less faithful.
- MULTIFINGROUND, Q3: English, French and Greek evidence.
- FrenchNews-7, Q2: CamemBERT on full text beats zero-shot LLMs on recall (0.799).
- UZH: most teams relied mainly on English translations of French ([mapping notes] Q3). No controlled comparison.
- **[older] Multilingual hierarchical classification of job advertisements.** Maciej Beręsewicz, Marek Wydmuch, Herman Cherniaiev, Robert Pater. First posted 6 Nov 2024. Read: ABS (cut). [arXiv 2411.03779](https://arxiv.org/abs/2411.03779). Narrow from a 1-digit occupation group down to 6 digits. A search summary claims translation helped for some languages; not confirmed in what I read.
- **[older] Learning the Topic, Not the Language.** Nasuto et al. First posted 8 Aug 2025. Read: ABS. [arXiv 2508.06435](https://arxiv.org/abs/2508.06435). Fine-tuning in one or two languages generalises to topic classification in others, while stance benefits from more. A search summary claims translation reduces accuracy; the abstract does not state this, so treat it as unconfirmed.
- **Speak Beyond English: Multilingual Prompts Improve Query Classification in Small Language Models.** Pratyay Banerjee, Panthadeep Bhattacharjee, Angshuman Jana. Published 19 Jul 2026 per Crossref, SIGIR 2026. Read: META, not read. [DOI 10.1145/3805712.3808528](https://doi.org/10.1145/3805712.3808528)
- **Conformal Prediction for Statistically Reliable LLM-as-a-Judge in Recommender Systems.** Maddalena Amendola et al. Crossref date 6 Nov 2026, CIKM 2026. Read: META, not read. [DOI 10.1145/3799682.3839966](https://doi.org/10.1145/3799682.3839966). Relevant to Q2 and Q4.
- **A two-stage hybrid intelligence framework for subject indexing.** Tian Xia, Xin Yang, Wenjing Wu, Ying-Hsang Liu, Junzhi Jia. Published 15 Jul 2026 per Crossref, Journal of Information Science. Read: ABS via Crossref. [DOI 10.1177/01655515261449401](https://doi.org/10.1177/01655515261449401). Cross-lingual retrieval plus LLM refinement on TIBKAT. Mostly a prediction method.

#### Candidates for the demo: what the evidence supports

| Showpiece | Published model to follow | What a viewer sees | Evidence strength |
|---|---|---|---|
| Tier-by-tier scorecard with human line | LCSHBench exact vs concept; DIDECO κ by tier; AraGenre broad vs specific | Score at each tier, next to human agreement at that tier | Good: several papers, directly on tiers |
| Metric flip | Kähler; LCSHBench | The winner changes when the ruler changes | Good, with Kähler's caveats |
| Silver versus independent audit | Vasquez; law-main.18 | Same system scored on publisher tags, LLM tags, reviewed sample | Moderate: single-reviewer study, confounded at level 2 |
| Evidence verification | CTI2TTP; POINTERS | Label, quote, check result; precision with and without | Good on one domain |
| Back-off coverage curve | HCC; bionlp-1.16 | Leaf, parent or abstain, with review load | Method well defined; **untested on LLM tagging** |
| Gap report | UZH; TaxCE; ConceptE; ReLTEx; Mosaic | Unplaced clusters, proposed nodes, overlapping and unused labels | Pieces only; whole loop untested |
| Language panel | LCSHBench; He et al. | Tier scores by language; translated vs source | Moderate: no French or Spanish tier study |

### Inferences
All untested hypotheses, my own suggestions.
- Untested hypothesis, the strongest demo pairing: (1) a tier-by-language scorecard with human agreement on a blind sample, and (2) a gap report. The first shows whether to trust the tags, the second shows what to change, and both can be read by non-specialists.
- Untested hypothesis, a third showpiece with most surprise value: an ablation of evidence verification, showing how many labels disappear when a quote and definition check is added. Over-assignment would be visible.
- Untested hypothesis: a metric-flip chart using the user's own systems (publisher tags, pipeline LLM tags, the new system) would be a strong opening for the blog, if the sample supports it.
- Untested hypothesis for translation: a small paired test on the same French and Spanish chunks, as source, as machine translation, and with the taxonomy kept in English, scored per tier on a blind human sample. That would be new, since I found no such published comparison. Sample sizes per tier need planning, because leaf counts will be small.
- Constraint from this project's `CLAUDE.md`: report text must not be copied into the repository, and notebook outputs must stay small. Any evidence-quote showpiece must therefore be displayed without committing quotes, or use short agreed excerpts.

### Gaps
- **No in-window paper directly tests whether a tier-based tagger holds up across English, French and Spanish, and none tests translate-then-tag vs source-language tagging against a taxonomy, with numbers.** He et al. is the nearest for tiers, Henriksen for translated input.
- **No controlled study of noisy PDF parsing on tagging quality**, as also noted in [mapping notes] Q4. RAPT names OCR noise as a motive, without a controlled noise test in the abstract.
- Whether a viewer finds a scorecard or a gap report more compelling is not something any paper measures.

### Search coverage and tool limits (applies to all questions)
- **Worked:**
  - arXiv API by first-submission date: 42 phrase queries, 506 unique records, title-screened, with abstracts read for the shortlist. Slow, with repeated 429s, and I waited rather than forcing it.
  - ACL Anthology event pages for ACL, EMNLP, EACL and LREC 2026, plus Findings and co-located workshops including UncertaiNLP: 16,930 abstracts, filtered by regular expressions.
  - Crossref API: 23 queries, 407 candidates after a broad title filter.
  - OpenAlex lookup by DOI worked once, for one record.
- **Limited:** the arXiv advanced search web page returned 429 after about ten queries, and I stopped. Only a partial set of searches was completed by that route.
- **Did not work:** Semantic Scholar returned 429 throughout, so no citation graph, including for Kähler. KONVENS 2026 has no Anthology event page. The AUP publisher page returned 403. I met no bot check and did not try to bypass any block.
- **Not searched directly:** ICML 2026, NeurIPS 2026, JCDL 2026 and TPDL 2026 proceedings. Crossref title queries are the only coverage of those, and no relevant JCDL or TPDL paper surfaced, which is not proof of absence. arXiv queries would catch preprints. AACL 2026 was only partly covered, through the Findings page.
- **Keyword filters miss other wordings.** This list is verified but not exhaustive.
- **Items I could not fully verify:**
  - Kluge and Kähler 2025 is known only through LCSHBench.
  - The 2025 EMNLP multi-label paper was confirmed by title only.
  - Numbers from figures were not captured in the reference-free paper and the LREC paper.
  - Published dates for accepted-only preprints (Kähler, FrenchNews-7, the CLA paper, PRECISE) are not confirmed.

### Summary
The notes are complete within the limits above. Strongest findings are that scores depend on which tier and label source you use, that agreement between models is not proof of correctness, and that a separate evidence check can strongly raise precision. There is no published test of parent-tier back-off on LLM tagging, or of translation against source-language tagging on a taxonomy, so those would be new work and the demo should present them as experiments.
