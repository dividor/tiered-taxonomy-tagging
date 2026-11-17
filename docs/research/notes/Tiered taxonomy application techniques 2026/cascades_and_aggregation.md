# Applying a tiered taxonomy to a very large chunked corpus at affordable cost, and rolling chunk labels up to document labels

Searched on 2026-11-05. Window: first posted or published about May to November 2026. Anything older is marked **[older]** and included only because newer work builds on it or because it is the nearest direct evidence on a question.

Use case: about 1.9 million chunks from about 13,000 long UN evaluation reports, mostly English with French, Spanish and others, noisy PDF parsing, taxonomy of hundreds to low thousands of leaf labels. That is about 146 chunks per report (my arithmetic).

**How to read each entry.** "First posted" is the earliest public date I could establish (arXiv v1 from the arXiv record, never inferred from the ID; or the preprint server's posted date). "Published" is the venue date. ACL Anthology gives only a month. Where no first-posted date is given for an Anthology paper, none was looked up, except where the entry names an arXiv version.

**Read status codes.**
- **PDF**: I converted the PDF to text and read the abstract, method description and the results tables named in the entry. I did not read every line.
- **PDF-part**: PDF converted to text; I read the abstract plus passages found by keyword. Treat as named sections only.
- **ABS**: abstract only, from a primary record (arXiv listing, ACL Anthology page, or Crossref record).
- **ABS-OA**: abstract only, from the OpenAlex DOI record, because Crossref had none or the publisher page was not opened.
- **META**: title and metadata only. Not read. The user may wish to obtain these.
- **SNIPPET**: rests on a search-tool summary, not a primary record. Treat as unverified.

**Earlier notes.** Entries marked "see earlier notes" are verified in `research_notes/Entity and ontology extraction papers 2026/mapping_to_supplied_vocabularies.md` and are not repeated in full. Model names and prices are as the papers state them. I did not check them against provider price lists.

---

## Q1. New cascade and routing designs: what comes first, what decides escalation, and what savings and quality changes are measured

### Takeaway
A two-tier design is enough in most tests: one paper finds a third model adds nothing in 118 of 135 comparisons. Reported savings are large (LLM on about 2% to 40% of items at similar or better quality), but the escalation signal is the weak point: confidence barely beats random on some tasks, and it can send one language up to 30 points more often than another, so a per-language quota and a held-out gold set matter as much as the choice of router.

### Cited Findings

#### Foundations that newer work builds on

- **Task Cascades for Efficient Unstructured Data Processing [older].** Shreya Shankar, Sepanta Zeighami, Aditya G. Parameswaran. First posted 9 Jan 2026. Published Proc. ACM Manag. Data 4(1), SIGMOD, Article 88, February 2026 (DOI as printed in the preprint). Read: PDF, Section 7 and Tables 3 and 4. [arXiv 2601.05536](https://arxiv.org/abs/2601.05536); [DOI 10.1145/3786702](https://doi.org/10.1145/3786702)
  - How it works: (1) On a small sample, the expensive model marks which parts of each document matter. (2) Documents are reordered so relevant text comes first, and each stage reads only the top fraction. (3) An LLM agent proposes simpler stand-in questions (for example "does this mention money?") that a cheap model can answer. (4) A greedy search assembles a sequence of (model, question, document fraction) stages, each with confidence thresholds. (5) Remaining documents go to the expensive model. Cached prefix tokens are priced separately in the cost model.
  - Evidence: eight document-classification tasks at a 90% accuracy target, measured against the expensive model. Average 36% lower end-to-end cost than a two-model cascade at production scale (41% on inference; 48.5% with guarantees). Largest gains where documents hold much irrelevant text: Table 3 shows 0.09× to 0.14× of the two-model cascade cost on Enron emails. Offline optimisation is 7.9× to 27× the two-model cascade cost, repaid after 2,986 documents on average (678 for the Lite version). Savings per million documents have a median of $257 to $327. The authors note high run-to-run variance.
  - Relevance: **the most directly relevant design found**. It changes how much of each document is read, not only which model reads it. It targets one question at a time, so it looks a better fit for a top-tier decision than for thousands of leaf labels (my inference).

- **Cut Costs, Not Accuracy: LLM-Powered Data Processing with Guarantees (BARGAIN) [older].** Sepanta Zeighami, Shreya Shankar, Aditya Parameswaran. Published Proc. ACM Manag. Data 3(6), SIGMOD, Article 311, December 2025 (first-posted date not established). Read: PDF-part, abstract and introduction. [DOI 10.1145/3769776](https://doi.org/10.1145/3769776)
  - How it works: samples records adaptively, estimates quality with tight statistical bounds, and sets a threshold so the cheap model's answers are used only where a chosen accuracy, precision or recall target can be certified.
  - Evidence: eight datasets; reduces cost on average by up to 86% more than the prior method; that prior method missed its target more than 75% of the time in some settings.

- **ScaleDoc: Scaling LLM-based Predicates over Large Document Collections [older].** Hengrui Zhang, Yulong Hui, Yihao Liu, Huanchen Zhang. First posted 16 Sep 2025 (v2 3 Mar 2026). SIGMOD 2026 per the authors' PDF name; issue and date not confirmed. Read: PDF-part. [arXiv 2509.12610](https://arxiv.org/abs/2509.12610)
  - How it works: a small LLM embeds each document once, offline. For each new question, a light proxy is trained on those embeddings with contrastive learning, filters most documents, and sends ambiguous ones to the LLM.
  - Evidence: BigPatent, PubMed, GovReport: more than 2× end-to-end speed-up and up to 85% fewer LLM calls. PROBE's comparison finds it delegates 59.8% of rows for geometric-mean F1 0.530 on very selective queries, so results depend strongly on selectivity.

- **100x Cost & Latency Reduction: Performance Analysis of AI Query Approximation using Lightweight Proxy Models [older].** Yeounoh Chung et al. (12 authors, Google Cloud). First posted 16 Mar 2026 (v6 14 Apr 2026). Accepted to SIGMOD 2026 per the paper. Read: PDF-part. [arXiv 2603.15970](https://arxiv.org/abs/2603.15970). Vendor-authored.
  - How it works: (1) embed all rows; (2) label a sample with an LLM (200 examples in the worked spam experiment); (3) fit scikit-learn logistic regression with default settings and balanced class weights; (4) check it against the LLM labels and deploy only if within a quality threshold, otherwise use the LLM.
  - Evidence: AI.IF on online proxies 329× latency and 728× cost; pre-trained 991× and 792×. Evaluated up to 10 million rows.
  - Relevance: the simplest possible baseline for frequent labels, and a good demo contrast. It is one binary question at a time, so a thousand labels would mean a thousand proxies, and very rare labels would lack positives in a small sample (my inference; see PROBE).

#### New cascade designs that use a cheap classifier or retrieval first

- **HIRA: A Human-in-the-Loop Retrieval-Augmented Cascade for Document Classification in Regulated Industries.** Shangxuan Tian, Yanhui Chen, Carlos Queiroz. First posted 22 Aug 2026. Published CIKM 2026 (7-11 Nov 2026), DOI as printed in the PDF. Read: PDF-part, Sections 6-7. [arXiv 2608.21792](https://arxiv.org/abs/2608.21792); [DOI 10.1145/3799682.3840098](https://doi.org/10.1145/3799682.3840098)
  - How it works: (1) fuse BM25 over OCR text, dense embeddings and image features with weighted reciprocal-rank fusion. (2) If the score passes a threshold swept on validation (0.675), accept. (3) Otherwise a local LLM sees OCR text, retrieved exemplars, label descriptions and confusion-specific terms. (4) If still unsure (below 0.95), a human decides. (5) Each correction becomes a weighted retrieval exemplar and updates a confusion graph. No weights are retrained.
  - Evidence: Tobacco-3482: macro-F1 0.9423 with LLM calls on about 40% of documents, +17.4 points over zero-shot LLM. Private 80-class trade-finance stream of 30,233 documents: humans correct 6.4%, macro-F1 from 0.6218 to 0.8548. Raising the threshold above 0.85 adds LLM cost with no gain; the authors attribute this to the verifier rarely overturning borderline retrieval answers.
  - Relevance: high. OCR-noisy documents, cold start, no retraining. It works at document level, with far fewer labels than the use case.

- **CARE: Bounded Selective Evidence-to-Judgment Routing for Cost-Aware Document Security Classification.** Yeseul Chang, Woosang Lim, Rahul Kailasa, Simon Shim, Jaewoo Lee. Published October 2026, EMNLP 2026 Industry, pp. 4423-4433. Read: ABS plus PDF-part. [ACL Anthology 2026.emnlp-industry.263](https://aclanthology.org/2026.emnlp-industry.263/)
  - How it works: (1) a retrieval vote over similar past documents, no LLM. (2) A small learned gate predicts whether the vote is wrong; if risk is low, commit. (3) Otherwise one verifier model. (4) Only if verifier and evidence disagree, a panel of models.
  - Evidence: WikiLeaks: macro-F1 0.9824, 44 of 2,863 documents reach any LLM, 55 invocations, 1.9% of one-call-per-document. DISC: macro-F1 0.8785, 496 invocations vs 789 for an always-LLM policy (63%), with rare TOP SECRET F1 0.8444.
  - Relevance: an unusually clean recipe, including an explicit guard on the rare class. Few classes only.

- **A Classifier That Teaches Itself: SIFT.** Bogdan Raduta, Horia Velicu, Alexandru Preda, Serban Chiricescu. First posted 20 Jul 2026 (updated 1 Sep 2026). Read: ABS. [arXiv 2607.18358](https://arxiv.org/abs/2607.18358)
  - How it works: SPLADE sparse features into LightGBM on CPU; low-confidence pages go to an LLM judge; verdicts become training labels; a retrained model is promoted only if it passes a critical-label F1 check and a frozen golden set it never trains on.
  - Evidence: describes an "illustrative" deployment; no benchmark numbers in the abstract. Worth citing for the frozen-gate idea, not for results.

- **L2D-Clinical: Learning to Defer for Adaptive Model Selection in Clinical Text Classification.** Rishik Kondadadi, John E. Ortega. Published May 2026, ClinicalNLP at LREC 2026, pp. 38-48. Read: ABS. [ACL Anthology 2026.clinicalnlp-1.5](https://aclanthology.org/2026.clinicalnlp-1.5/)
  - Evidence: learns when BERT should defer to an LLM. ADE: F1 0.928 (+1.7 over BERT) deferring 7%. MIMIC: F1 0.980 (+9.3) deferring 16.8%. The combination beat both parts even when one part was clearly weaker. MIMIC's ground truth is multi-LLM consensus, which favours the LLM.

- **Cost-Efficient Intent Classification via Semantic-Candidate Adaptation and Selective Capability Routing.** Jiawei Sun, Yongsheng Sang. Published October 2026, EMNLP 2026 Industry, pp. 4338-4348. Read: ABS. [ACL Anthology 2026.emnlp-industry.257](https://aclanthology.org/2026.emnlp-industry.257/)
  - Evidence (CLINC150): direct centroid retrieval 80.71%; task-trained encoder 90.05%; small LLM reranking top 5 with one example per candidate 76.13%, below plain retrieval. Sending the lowest retrieval scores to a large model: 82.11-82.89% at a 10% budget and 87.33-88.07% at 20%. No policy wins everywhere. **A useful honest result: always-on small-LLM reranking can be worse than the embedding alone.**

- **The Embedder's Dilemma: LLMs Are Better, but at What Cost?** Adnan El Assadi, Niklas Muennighoff, Jinhyuk Lee. First posted 13 Aug 2026. COLM 2026 per a search result showing the PDF header (SNIPPET for the venue only). Read: ABS. [arXiv 2608.12875](https://arxiv.org/abs/2608.12875)
  - Evidence: ten LLMs and 26 embedding models on 37 tasks. Best LLM 77.6 vs best embedding 77.2 overall, and embedding models lead on classification. An LLM costs up to 1,431× more for comparable quality ($154 vs $0.11 per benchmark pass). Reasoning tokens are 28-81% of LLM cost.
  - Relevance: strong support for embeddings as the first stage.

- **AutoRelAnnotator: Calibrated Model Cascades for Cost-Efficient Relevance Evaluation in Sponsored Search.** Md Omar Faruk Rokon, Shasvat Desai, Hong Yao, Kuang-chih Lee. First posted 24 Jun 2026; SIGIR 2026 E-commerce workshop. Read: ABS. [arXiv 2606.25871](https://arxiv.org/abs/2606.25871). Vendor-authored.
  - Evidence: 150M+ annotations in production. Fine-tuning adds 20 accuracy points; cascading is about accuracy-neutral but halves compute; per-class isotonic calibration adds +0.6 points. **Clear scale evidence, and a clean way to explain what each part buys.**

- **Others with numbers (all ABS or ABS-OA):**
  - *Cost-Aware LLM Cascades for Zero-Shot Multi-Class Mental Health Classification on Reddit.* Mahmoud Abusaqer, Jamil Saquer. ICMLA 2026, October 2026 (ABS-OA). 20,997 posts, seven classes: cascade 92.87% F1 at $26.00 vs Claude Sonnet 4.6 alone 93.91% at $96.29. [DOI](https://doi.org/10.1109/icmla69890.2026.00102)
  - *A Sketch Calibrated Cascade for Cost Bounded Text Classification on Large Review Streams.* Yuvaraju Chinnam et al. ICACT 2026, September 2026 (ABS-OA). Linear scorer then DistilBERT; threshold set by conformal risk control; target 0.50 points, realised 0.376; 25.3% escalated; 74.7% less GPU time on 400,000 records. Also alarms on feature drift. [DOI](https://doi.org/10.1109/icact70625.2026.11748740)
  - *A Hybrid ML-LLM Cascade System for Cost-Efficient Phishing Webpage Detection.* Yusuf Özaslan, M. Fatih Adak. YEF-ECE 2026, June 2026 (ABS-OA). XGBoost F1 0.9574, Gemini 0.9339, cascade 0.9662 with 5.2% escalated. [DOI](https://doi.org/10.1109/yef-ece70590.2026.11614634)
  - *MECH.* Yancui Li, Xiaoyu Zhou, Guoyi Miao, Fang Kong. ACL 2026 Long, pp. 22460-22473. 78.55% accuracy, 44.4% lower API cost; routes on a task signal, motivated by confidence failing on long-tail items. [ACL Anthology](https://aclanthology.org/2026.acl-long.1028/)
  - *Prune First, Decide Fast: JEVDB.* Zhengle Wang, Hanxu Yan, Fuheng Zhao, Chunwei Liu. First posted 1 Oct 2026. Typed decision models first, escalating uncertain cases; removes 87.4% of candidate pairs before evaluation and cuts reasoning escalations by 55.2%. [arXiv 2610.02046](https://arxiv.org/abs/2610.02046)
  - *Do We Need an LLM to Know Why Sources Conflict?* Parv Khatri, Harshul Jain, Mehak Kapoor, Tanya Sah. First posted 27 Oct 2026. 10 of 11 decision models match or beat Gemini 2.5 Flash at 4-39× lower billed cost; sending the least confident quarter to a stronger model ties it for 8 of 11 at 25-37% of its cost. Decomposing the taxonomy into yes/no questions lowered accuracy for every model. [arXiv 2610.36915](https://arxiv.org/abs/2610.36915)
  - *System1-Arena.* Tianyu Chen, Xinyang Liu, Ruyi An, Zhendong Wang, Mingyuan Zhou. First posted 15 Oct 2026. 92k questions, 105 tasks, 30 models. Large one-pass decision models match or beat fast closed LLMs on non-agent accuracy at a quarter of the latency; a reasoning LLM is most accurate at about 70× the cost. [arXiv 2610.21244](https://arxiv.org/abs/2610.21244)
  - See earlier notes: SelectED (LLM on about 20%), DuCEI (27.8%), invoice filtering (49.6% fewer calls), Jev-as-a-Judge, Circuits of LLM Classifiers.

#### Which models, how many tiers, and which signal

- **Is Escalation Worth It? On the Depth of LLM Cascades.** Dylan Bouchard. First posted 7 May 2026 (substantially revised, updated 5 Oct 2026). Read: ABS. [arXiv 2605.06350](https://arxiv.org/abs/2605.06350)
  - How it works: derives that at the best thresholds, accuracy gained per downstream cost is equal across boundaries, and shows the gain over random escalation splits into two AUROC terms.
  - Evidence: five benchmarks, nine scores, pool of eight models. Two-model cascades add 2.1 to 8.2 points over the best single model. A third model does not help in 118 of 135 comparisons and adds at most 0.43 points. Simulated scores that also reflect difficulty shared across models do not unlock depth, even at the same AUROC.
  - Relevance: **argues against over-engineering.**

- **Who Pays for Efficient Inference? Confidence-Based Small-to-Large LLM Routing Across Tasks and Languages.** Md Sabbir Hossen, Anichur Rahman, Pabon Shaha, Airin Afroj Aishi, Andrew H. Sung, Md Shohel Rana. First posted 16 Oct 2026. Read: ABS. [arXiv 2610.23249](https://arxiv.org/abs/2610.23249)
  - Evidence: 72 configurations, 42,288 predictions, four languages. Confidence's correction-detection AUC: 0.705 (SIB-200 topic classification), 0.612 (Belebele), 0.510 (MMLU-ProX-Lite, about random). At 50% escalation, gains over random are 2.54-5.63 points on topic classification. **Global routing creates language escalation gaps up to 30.33 points; equal-quota routing removes them at little accuracy cost.** The abstract does not name the four languages, and I did not read the full text.
  - Relevance: the one paper that directly tests the multilingual risk here. An equal per-language quota is easy to demo.

- **UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing.** Varun Kotte. First posted 11 May 2026. Read: ABS. [arXiv 2605.18796](https://arxiv.org/abs/2605.18796). Single author.
  - How it works: token margin mapped to error probability by isotonic regression, then threshold chosen by constrained cost minimisation.
  - Evidence: 75,000 production NER queries, 4B and 12B models on H100s: 31% cost cut (95% CI 27-35%) at micro-F1 0.91; ECE 0.12 to 0.03; beats entropy, split-conformal and a FrugalGPT-style threshold, using measured latency.

- **Calibrated Deferral Routing for Cost-Efficient Guardrails.** Giandomenico Cornacchia et al. (10 authors). Published October 2026, EMNLP 2026 Industry, pp. 4264-4282. Read: ABS. [ACL Anthology 2026.emnlp-industry.252](https://aclanthology.org/2026.emnlp-industry.252/)
  - Evidence: 18 models, 10 benchmarks. ECE out of the box 6-28%; below 2.5% after isotonic, Platt or temperature scaling in most cases. **Calibration makes thresholds deployable rather than improving rank ordering**, so it will not fix a score that cannot separate right from wrong.

- **When to Defer: Scope-Aware Cascades for Structured Attribute Generation in E-commerce Product Catalogs.** Fatemeh Mansoori, Andrea Scarinci, Aditya Aggarwal, Ashwin Chandramouli, Suleiman A. Khan. Published October 2026, EMNLP 2026 Industry, pp. 3149-3160. Read: ABS. [ACL Anthology 2026.emnlp-industry.187](https://aclanthology.org/2026.emnlp-industry.187/)
  - How it works: an eight-feature gradient-boosted tree over token confidence plus which task it is.
  - Evidence: at a fixed precision target accepts about 11 points more values than a global confidence threshold. The gain comes from per-task reliability, not a better transformation of confidence.
  - Relevance: the direct analogue would be a per-label or per-branch reliability prior (my inference, not tested on taxonomies).

- **Conformal Cascade.** Yifan Dou, Shikan Lian, Shibo Li. First posted 27 Jul 2026. Read: ABS. [arXiv 2607.25018](https://arxiv.org/abs/2607.25018). Accept when the calibrated prediction set has a single answer, otherwise defer; finite-sample guarantee on 18 multiple-choice benchmarks. Black-box API only. The set-size rule needs a defined answer set, which retrieved candidates would give (my inference).

- **Circuits of LLM Classifiers.** See earlier notes. Read here: ABS plus PDF-part. On prompts where every call is correct, it matches majority voting (within 1.3%) with 38-86% fewer calls. Its calibration error is at most 2.1% on six tasks and 5.2% on BIG-Bench Hard, against 4.8-24.6% for majority voting. [arXiv 2610.30076](https://arxiv.org/abs/2610.30076)

- **Other signals (ABS):**
  - *Flexible Routing via Uncertainty Decomposition.* Charlotte Peale, Siddartha Devic, Parikshit Gopalan, Udi Wieder, Aravind Gollakota. First posted 8 May 2026. Route when uncertainty can be reduced, abstain when it cannot. Needs several annotations per input. [arXiv 2605.07805](https://arxiv.org/abs/2605.07805)
  - *Two Uncertainties Are Not One.* Karan Yadav. First posted 23 Oct 2026. Human disagreement and decision risk call for different escalation signals; audit on 142,658 comments, a cheap model at about $0.000025 per call has AUROC 0.83, not separable from 13 of 14 frontier systems on a 56-item probe. [arXiv 2610.33258](https://arxiv.org/abs/2610.33258)
  - *Large Language Models as Automatic Annotators and Annotation Adjudicators.* Gaurav Negi, M.A. Waskow, John Philip McCrae, Omnia Zayed, Paul Buitelaar. UncertaiNLP at EMNLP 2026, pp. 570-589. Agreement among LLM annotators is a calibrated predictor of correctness. [ACL Anthology](https://aclanthology.org/2026.uncertainlp-main.35/)
  - *QUORUM.* Antonio Purificato, Maria Sofia Bucarelli, Andrea Bacciu, Amin Mantrach, Fabrizio Silvestri. EMNLP 2026 Main, pp. 26873-26890. Uses text features rather than confidence, under a fixed budget; annotation quality up to +34.4% and cost -8.8%; includes multilingual tasks. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1172/)
  - *The Differential Reasoning Router.* Cheng Lyu, Jingyue Zhang, Vinny DeGenova, Mengwei Li, Yuanli Pei. EMNLP 2026 Industry, pp. 2815-2829. Estimates success for direct and reasoning models separately, and sends likely double failures to humans; more than 60% saving on reasoning tokens at parity. [ACL Anthology](https://aclanthology.org/2026.emnlp-industry.166/)

#### Negative and cautionary results

- **Reliable but Design-Sensitive: Instrument Uncertainty in LLM Annotation.** Thomas Reiter, Christoph Kern, Fedor Miasnikov, Sofiia Nikolenko, Rob Chew, Stephanie Eckman, Frauke Kreuter. Published October 2026, UncertaiNLP at EMNLP 2026, pp. 312-327. Read: ABS. [ACL Anthology 2026.uncertainlp-main.20](https://aclanthology.org/2026.uncertainlp-main.20/)
  - Evidence: same setup repeated κ=0.91; changing the task design κ=0.76. Task design and model raise the variance of estimated prevalence by factors of 76.7 and 110.6 over sampling variance. **Confidence tracked repeat outputs more closely than agreement with humans, and grouping six tweets in one prompt lowered mean confidence by 660 basis points.**
  - Relevance: a warning for corpus-wide statistics such as "x% of evaluations address gender", and for mixing batch prompting with confidence gates.

- **Pre-Inference Routing for Cost-Efficient Document Field Extraction.** Sreerekha Rajendran. First posted 6 Aug 2026. Read: ABS. [arXiv 2608.06607](https://arxiv.org/abs/2608.06607). Routing helps only if the cheap model fails often enough and failures are predictable from visible features. 31-33% saved on receipts, 77% on degraded forms; none on clean invoices. Router does not transfer across datasets. A small labelled pilot predicted correctly in the two cases tested.

- **Cheap First, Ask the LLM Later?** Soumya Patel. First posted 27 Oct 2026, Preprints.org. Read: ABS. [DOI](https://doi.org/10.20944/preprints202610.2704.v1). A small test (16 syllabi), but the shape is instructive: rule-first routing saved 57.8% of tokens and dropped F1 from 0.657 to 0.370. "Every token saved cost accuracy."

- **Fast LLM-Based Semantic Filtering: From a Unified Framework to an Adaptive Two-Phase Method.** Kyoungmin Kim, Martin Catheland, Anastasia Ailamaki. First posted 6 Jun 2026. Read: ABS. [arXiv 2606.08090](https://arxiv.org/abs/2606.08090)
  - How it works: (1) try model-free clustering first; (2) train an online proxy only if needed, sharing LLM calls across phases; (3) use the LLM's per-document confidence as a soft training label, not just yes/no; (4) add a safety margin only where the labelled sample is sparse.
  - Evidence: 90% accuracy target on three 10K-document corpora: 1.6-2.0× faster than the best prior method on each corpus, meeting the target on 95% of queries. The same confidence signal gives a lower bound on the LLM calls any proxy cascade needs, indicating about 4-20× further headroom.
  - Relevance: soft labels from LLM confidence are a cheap upgrade for any silver-label classifier (my inference for this corpus).

- **PROBE: Learned Proxy-Based Execution for Highly Selective Multimodal Semantic Queries.** Haikai Zhao, Haisheng Fu, Zhengjie Miao, Zhenman Fang. First posted 1 Nov 2026. The PDF uses a PVLDB template with blank page numbers, so acceptance is not confirmed. Read: PDF-part. [arXiv 2611.01454](https://arxiv.org/abs/2611.01454)
  - How it works: adaptive sampling to find rare positives, a light proxy, correction over a similarity graph, joint calibration on a separate sample, then send to the LLM the candidates whose re-check would most raise estimated F1.
  - Evidence: positive rates as low as 2.17 × 10⁻⁵; 10% sampling budget; geometric-mean F1 0.762 vs 0.797 for exhaustive LLM; 4.87× mean speed-up. Prior systems at a 0.9 target: LOTUS sends 54.6% for F1 0.485, ScaleDoc 59.8% for F1 0.530. The authors point out that a classifier can reach high accuracy while missing all matches.
  - Relevance: **this is the rare-leaf-label problem stated in database terms** (my reading). Accuracy targets can hide failure, since nearly everything is negative.

- **Not read, listed for the user to obtain (META, or abstract without numbers):** *When to Trust: Margin-Calibrated Routing Between LLM-as-a-Judge and Human Experts* (Xinran Zhang et al., [EMNLP Industry](https://aclanthology.org/2026.emnlp-industry.201/); no abstract retrieved). *GenGate* (Yanqiu Xu, Han Luo, Lei Feng, [EMNLP Main](https://aclanthology.org/2026.emnlp-main.2071/); ABS, no figures). *CascadeDebate* (Raeyoung Chang et al., [ACL Industry](https://aclanthology.org/2026.acl-industry.93/); ABS). *Larch* (Fuheng Zhao et al., first posted 6 Jun 2026; 3-19× lower token overhead than Palimpzest and Quest for filter ordering; ABS, [arXiv 2606.07923](https://arxiv.org/abs/2606.07923)). *Routing Should Pay for Itself* (Guannan Lai et al., first posted 29 Sep 2026; break-even volume 1.9-9.5× lower; ABS, [arXiv 2609.37402](https://arxiv.org/abs/2609.37402)). *CRE-Router* (Yasmin Moslem et al., first posted 25 Jun 2026; ABS, [arXiv 2606.27457](https://arxiv.org/abs/2606.27457)). *Cost-Aware LLM Invocation Strategies in Large-Scale Enterprise Systems* (Sai Chaitanya Kumar Paidi, ICARAI 2026; [DOI](https://doi.org/10.1109/icarai70085.2026.11635457); abstract only partly read).

### Inferences
- A sensible default for the use case is two tiers, not three: embedding or trained classifier first, one strong LLM second. Bouchard's result, the embedder benchmark and the industrial cascades all point the same way, though none tests a taxonomy of this size.
- Thresholds should be per label branch or per language rather than global. Evidence: scope-aware routing (+11 points) and Who Pays (30.33-point language gap). This combination has not been tested together; hypothesis.
- Several papers report a cascade beating both of its parts (L2D, phishing, SelectED). One hypothesis is that the two tiers make different errors, but I have not seen that measured across these papers.

### Gaps
- I found no cascade evaluated on a taxonomy with hundreds to thousands of leaf labels at chunk level. The tests I read mostly have 2 to 150 classes (Flipkart at 351 labels in Q2 is a retrieval test, not a cascade).
- No paper reports how noisy PDF parsing changes escalation rates. HIRA and RAPT use OCR text but do not vary noise.
- Who Pays does not report French or Spanish versus English in the abstract.

---

## Q2. Retrieve-then-rerank against a taxonomy: how many candidates are needed, and what does narrowing too hard cost on rare labels

### Takeaway
Measured gold-label recall at 10 to 15 candidates is about 86 to 98% on taxonomies up to 351 labels. In the UN resolutions task, systems with tight candidate pools scored far below one with no filter (0.226 at 20 candidates vs 0.459), though these are different teams' systems, not a controlled test. Adding the labels a model is known to confuse beats simply raising K (98.1% recall at about 18 candidates, where plain retrieval needed K=50), but it raises prompt size about 5 to 8 times. No paper reports recall-at-K broken down by label frequency on a large document taxonomy.

### Cited Findings

| Source | Labels | Candidates vs result |
|---|---|---|
| [Confusion to Clarity, EMNLP Industry 2026](https://aclanthology.org/2026.emnlp-industry.125/), Table 3 | Flipkart 351, three levels | Recall at L3: K=10 86.0%, K=20 92.4%, K=10 plus confusion partners (18 on average) 98.1%; plain retrieval needs K=50 to match. L3 F1: 74.2 with partners vs 72.2 at K=20 |
| [SLKR, arXiv 2610.16182](https://arxiv.org/abs/2610.16182), Table 3 | 38 to 150 | R@3 74-86, R@5 81-92, R@10 94-97, R@15 95.9-98.3. K=15 chosen |
| [UZH overview, ArgMining 2026](https://aclanthology.org/2026.argmining-1.10/) (earlier notes) | 141, UN resolutions | Tag micro-F1 across different teams: 20 candidates 0.226; 40 with caps per dimension 0.396; no pre-filter 0.459 |
| [SelectED, UncertaiNLP 2026](https://aclanthology.org/2026.uncertainlp-main.9/) (earlier notes) | entity linking | k=20 uses 37.6% more tokens than k=2, k=10 13.1%, with no accuracy gain |
| [Student-Guided Teacher Distillation, arXiv 2610.02516](https://arxiv.org/abs/2610.02516) (earlier notes) | 60 | Coverage@16 of 91-100%, preliminary, 200 examples |
| [InfluMatch, arXiv 2607.05968](https://arxiv.org/abs/2607.05968), ABS | KOL search | 50 retrieved, reranked to 10: half the token spend and 14 points better than reasoning over all 50. Only 11 queries |

- **From Confusion to Clarity: Confusion-Aware Retrieval and Knowledge Injection for Text Classification.** Manish Gupta, Chaitanya Giri, Jayasimha Talur (Amazon). First posted 1 Sep 2026. Published October 2026, EMNLP 2026 Industry, pp. 2149-2164. Read: PDF, Tables 1, 3 and 11. [ACL Anthology](https://aclanthology.org/2026.emnlp-industry.125/); [arXiv 2609.01564](https://arxiv.org/abs/2609.01564)
  - How it works: (1) run retrieval plus LLM on labelled training data and collect mistakes; (2) find label pairs the model mixes up; (3) at inference, whenever one label of a pair is retrieved, add its partner; (4) have an LLM write short rules separating each pair, offline; (5) inject only rules for labels in the candidate set.
  - Evidence with Qwen3-32B, macro-F1 (my reading of the extracted table columns, LEDGAR / WOS L1, L2 / Flipkart L1, L2, L3): zero-shot all labels 57.1 / 80.5, 56.6 / 84.4, 73.3, 65.5. Retrieval K=20: 56.3 / 78.4, 52.4 / 91.0, 81.8, 72.3. Full method: 58.9 / 81.5, 57.7 / 92.5, 84.4, 76.7. **Both prompt optimisers over the full label set scored below zero-shot on every column (GEPA 52.4 / 68.3, 32.7 / 61.4, 61.1, 58.0; MIPROv2 47.2 / 61.4, 35.9 / 73.8, 67.0, 62.4).** Error decomposition on Flipkart L3: retrieval misses fall from 14% to 1.9%, but LLM errors rise from 15.5% to 21.9%, and rules bring them back to 17.1%. Rules transfer to 2B-20B models, gaining up to 11.5 points.
  - Cost: offline 3.26M to 15.52M tokens per taxonomy for 455-658 rules. Input tokens per query go from about 1,200-1,400 to 7,100-10,100, and median latency from about 400 ms to 900-1,200 ms.
  - Relevance: **high. It is hierarchical, has a clear demo story, and is measured.** It needs a labelled set to find confusions, and the token overhead is large at 1.9M chunks.

- **Label-Centric Knowledge Retrieval for Zero-Shot Text Classification (SLKR).** Daoping Wu, Dong Wang, Siyu Liu, Chen Liang, Jian Geng, Ziqi Li, Shuguang Xiong, Ying Zhang. First posted 12 Oct 2026. Preprint under review. Read: PDF, Tables 2-6. [arXiv 2610.16182](https://arxiv.org/abs/2610.16182)
  - How it works: (1) a strong LLM writes a definition, keywords and boundary rules for confusable pairs, from the label list alone with no training data. (2) Hybrid retrieval picks top-K labels. (3) Rules involving only those labels are filtered, then the M most relevant to the input are chosen. (4) A small LLM classifies in one call.
  - Evidence (accuracy / macro-F1, Qwen3.5-4B): WOS-143 60.17 / 58.20, BANKING77 77.45 / 76.63, CLINC150 89.81 / 89.33, multilingual Book-38 70.58 / 70.25, vs label-only zero-shot accuracy 57.10, 68.77, 80.36, 64.25. Average 1.4k input tokens vs 4.7k for one example per label. Selecting rules uses 0.35k tokens vs 2.51k for all rules. Cost on BANKING77: build $3.40; online $0.0596 per 1k vs flagship $4.111 per 1k; but flagship accuracy is higher (86.01 vs 77.45).
  - Relevance: **highest for the demo. No labels needed, a multilingual long-document dataset, and an explicit cost table.** Book-38 has only 719 items and 38 labels, and construction can grow with the square of the label count.

- **RAPT: Retrieval-Augmented Post-hoc Thresholding for Multi-Label Classification.** Lasal Jayawardena, Nirmalie Wiratunga, Ikechukwu Nkisi-Orji, Darren Nicol. First posted 15 May 2026. Read: ABS. [arXiv 2605.16535](https://arxiv.org/abs/2605.16535)
  - How it works: for each document, find similar past cases and adapt the threshold or the label count from their outcomes, rather than using one global cut-off.
  - Evidence: OCR-noisy industrial set plus six public benchmarks. Beats global and per-label thresholds. Metric-learning encoders reach 0.87 macro-F1; fine-tuned transformers 0.775, which is 2× few-shot LLM baselines, with at least 115× less inference time.
  - Relevance: also a candidate for setting how many labels a report should get (see Q4).

- **Lost in Choice: Evaluating LLMs Under Extreme Multiple-Choice Scaling.** Alessio Cocchieri, Luca Ragazzi, Giuseppe Tagliavini, Gianluca Moro. Published October 2026, Findings of EMNLP 2026, pp. 6931-6963. Read: ABS plus PDF-part; degradation table not extracted. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.328/)
  - Evidence: 31,000 instances, up to 1,000 options, six reasoning LLMs. Questions that need comparing options degrade far more; accuracy often collapses for gold answers at late positions, rather than recovering at the end.
  - Relevance: the best evidence against pasting a full taxonomy into a prompt. Position effects suggest shuffling candidates, which is my inference and not a remedy I saw them test.

- **Rare-label evidence, all indirect:**
  - [SNOMED CT Concept Recommendation, arXiv 2609.17855](https://arxiv.org/abs/2609.17855) (earlier notes): Recall@10 7.74% for concepts in one or two training notes vs 43.90% for concepts in over ten.
  - [Retrieval-Augmented Classification of Hydropower Documents, arXiv 2608.23241](https://arxiv.org/abs/2608.23241) (earlier notes): 135 categories, 40 with no examples; BERT F1 is 0 on unseen, hybrid micro-F1 0.524.
  - [RESOLVENOW, ArgMining 2026](https://aclanthology.org/2026.argmining-1.12/) (earlier notes): embedding retrieval "dropped relevant tags before the LLM saw them", and the team reverted to a single full-taxonomy prompt with a checklist.
  - [Thakaa at AraGenre](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/) (earlier notes): full-taxonomy prompt won, with 74 specific labels.
  - [Retrieval-Augmented LLMs and Encoder Models for climate narratives](https://aclanthology.org/2026.nslp-1.22/) (Neda Foroutan, Alexandra Tsiakalou, Vera Schmitt; May 2026; ABS): narrowing candidates before instruction tuning gave their best result, macro-F1 59.72%, second place. So narrowing is not always harmful.
  - [HCMUS at AbjadMed](https://aclanthology.org/2026.abjadnlp-1.7/) (March 2026, outside the window; ABS): BERT plus k-NN retrieval for the long tail, macro-F1 0.4902.

- **Multilingual retrieval:** *A Cost-Efficient Routing Pipeline for Multilingual Short-Text Classification Using Small Language Models.* Wajdi Ben Saad, Safa Madiouni. ACIT 2026, September 2026. Read: ABS-OA. [DOI](https://doi.org/10.1109/acit69831.2026.11700134). Translating only low-resource languages into English before zero-shot matching lifts low-tier macro-F1 from 0.4632 to 0.6828 on a seven-way task, and from 0.2143 to 0.4417 on 60 intents. High-tier languages are unchanged on the seven-way task. The tiers tested are not stated in what I read.

- **Label wording matters as much as model choice:** *Between the Embedding and the Prompt: Systematic Design Effects in LLM-based Occupation Coding.* Olga Kononykhina, Anna-Carolina Haensch, Frauke Kreuter. First posted 19 May 2026, SocArXiv. Read: ABS, from Crossref. [DOI 10.31235/osf.io/g6wjy_v1](https://doi.org/10.31235/osf.io/g6wjy_v1). 119 experiments on German survey data. How the categories are written changes retrieval accuracy by 8-23 points and classification by 11-21. Variation in how respondents describe their work matters more than any model or design choice (ICC = 0.76).

- **Very large taxonomies, in brief (the other researcher covers core methods):** *Graph-Constrained Policy Learning for Extreme Clinical Code Prediction.* Amritpal Singh, Sebastian Torres, Khawar Shakeel, Syed Ahmad Chan Bukhari. First posted 11 Jul 2026. Read: ABS. [arXiv 2607.11954](https://arxiv.org/abs/2607.11954). One model walks the code tree level by level instead of retrieving flat candidates: 0.527 micro-F1 on 15,761 codes, +0.157 macro-F1 over the best flat baseline, with the authors suggesting this eases the rare-code problem. A three-specialist cascade overflowed the context window on 28-32% of full-space notes. Needs supervised training data.

- **Scoring many rare labels with one-vs-rest classifiers:** *Odds-Shift Slippage in One-vs-Rest Rankers.* Akifumi Goto. First posted 12 Sep 2026. Read: ABS, partly. [arXiv 2609.13810](https://arxiv.org/abs/2609.13810). Heavy positive-class weights damaged top-K ranking (MAP@7 0.808 to 0.117 on Santander). Per-label isotonic regression restored it to 0.784, but only if labels with no calibration positives are mapped to their prior. Tabular data, so relevant by analogy, and a practical warning for per-label proxy classifiers on rare leaves.

- **Occupation coding analogue:** *From 4-Digit Codes to Social Class.* Erdem Yilmaz, Elias H. Kruithof, Pieter-Paul Verhaeghe. First posted 25 Jul 2026, SocArXiv (v2 11 Sep 2026). Read: ABS, partly. [DOI](https://doi.org/10.31235/osf.io/xv2gf_v2). Multilingual ISCO-08, supervised, retrieval and LLM reranking compared; supervised reaches 59.2% exact four-digit accuracy, comparable to human agreement of about 42-53%, and rises to about 80% at coarser levels (77.9-87.8% in the part I read). **A strong argument for reporting at each tier: the authors find errors are mostly near-misses.**

### Inferences
- A workable starting point is K of about 15 to 40 from a hybrid retriever, tested by measuring gold-in-candidates recall, split by label frequency and language, before any LLM spending. The evidence for that number comes from taxonomies far smaller than the use case, so it must be measured.
- For a tiered taxonomy, confusion partners and retrieving per top-level branch both try to protect recall cheaply. UZH's "caps per dimension" is the only direct evidence I found for per-branch quotas, and it comes from a different pipeline. Hypothesis, not tested.
- Rules cost tokens at inference. For a corpus this size, SLKR's selective approach (0.35k rule tokens) looks far more affordable than Confusion to Clarity's full approach (about 7-10k input tokens), though they were tested on different datasets and with different models.

### Gaps
- **I found no paper that reports recall@K by label frequency on a large document taxonomy, so "the cost of narrowing too hard on rare labels" is only indirectly measured.**
- No test at thousands of labels combined with chunk-length input.
- Kähler et al.'s retrieval depth was not examined here; the other researcher covers core methods.

---

## Q3. Can a small labelled sample bootstrap a classifier? Silver labels, active learning, weak supervision, prompt optimisation

### Takeaway
LLM silver labels work well when the LLM is already good and classes are few: student models come within 3 points of human-trained ones. They fail when agreement with experts is low: one careful replication at 0.39 agreement finds no gain at any size, and rare-label coverage gets worse. Whether silver labels suit a large taxonomy therefore depends on measuring agreement per label on a gold sample first, and on keeping gold labels strictly apart.

### Cited Findings

#### Silver labels: positive

- **SemDistill: Bootstrapping a Low-Latency, Low-Cost Semantic Table Annotator from Noisy LLM.** Yuhang Ge, Zhangyan Ye, Yuren Mao, Yunjun Gao. Published 24 Sep 2026, Proc. ACM Manag. Data. Read: ABS, from Crossref. [DOI 10.1145/3837124](https://doi.org/10.1145/3837124)
  - How it works: (1) keep a few trusted anchor labels; (2) measure how the LLM systematically over-generalises concepts, and correct that using anchors; (3) find stubborn high-confidence errors from how they behave during training; (4) train on trusted, semi-trusted and noisy tiers with different objectives.
  - Evidence: beats its GPT-5-mini teacher and the baseline, while cutting inference cost 3800× and latency 2600×.
  - Relevance: names the two error types to expect: ontology drift towards general labels, and confident hallucination. On column annotation, not prose.

- **LLM-Based Weak Annotation for Cybersecurity Tweet Classification.** Ferry Afit Kurniawan, Kazushi Okamoto, Koki Karube, Kei Harada, Atsushi Shibata. First posted 15 Oct 2026. Read: ABS plus PDF-part. [arXiv 2610.21254](https://arxiv.org/abs/2610.21254)
  - Evidence: 1,897 gold tweets, 11,451 weakly labelled; 400 used to pick the teacher. Students above F1 0.92, within 3 points of human-supervised models.

- **HP at AraGenre 2026.** Hoàng Phong Đinh, Thìn Đặng Văn. Published October 2026, Arabic NLP shared tasks, pp. 252-256. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.arabicnlp-sharedtasks.29/)
  - Evidence: 140 labelled examples; a teacher scoring 0.9413 in five-fold validation labelled about 28,000 segments; MARBERTv2 trained on those reaches hierarchical macro-F1 0.7169. The winning team, judging directly with an LLM, reached 0.7352 (earlier notes). These are different teams with different extra features, so the roughly 2-point gap is only a loose price on distillation. It labelled the test set itself, so this is transductive.

- **Best of Both Worlds: Fine-Tuned Small Models Complement LLMs.** Shivam Mathur, Sri Harish Popuri, Priya Krishnan Sundararajan, Kevin Horecka, Michael Mugo. First posted 16 Oct 2026. Read: ABS. [arXiv 2610.24260](https://arxiv.org/abs/2610.24260). Survey text, 16 classes. RoBERTa F1 96% vs GPT-3.5 83%; ensemble 97.5% at 4.4% of GPT-3.5's inference cost; batch prompting saves about 70%. Uses an old model.

- **Scale references:** *Scaling Dense Retrieval with LLM-Annotated Training Data* (Rokon et al., first posted 22 Jun 2026; SIGIR 2026 workshop; ABS; [arXiv 2606.23911](https://arxiv.org/abs/2606.23911)): 240M+ training examples, three-model cascade at 89.1% agreement with humans, +5.1% NDCG@10 and a live A/B test. *CAPC-CG* (Bolun Sun et al.; ACL 2026 Long, pp. 944-966; ABS; [ACL Anthology](https://aclanthology.org/2026.acl-long.42/)): 3.3 million paragraphs of Chinese policy, expert-directed LLM annotation method, human κ=0.86, the closest in shape to the use case, though with only a five-colour typology. The abstract does not state how many paragraphs the LLM itself labelled.

- **A Cheap Lunch.** Shutao Chen, Piek T.J.M. Vossen. Published May 2026, LREC, pp. 10353-10364. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.lrec-1.813/). A domain encoder trained on manual plus synthetic annotations beats manual alone on WHO ICF categories, and can extend to new categories with little expert effort. Relevant because agency taxonomies change.

- See earlier notes: Kasliwal and Singh, LLM distillation with 112 topics (discriminative students lose 35% macro-F1; a 1B generative student wins on long text); Sci-ZSEL (filtered LLM aliases as pseudo-pairs, 72.91% to 90.32% accuracy).

#### Silver labels: negative or cautionary

- **Mapping CVEs to MITRE ATT&CK Techniques: A Curated Gold-Set Classifier and the Limits of LLM-Assisted Label Expansion.** Cédric Bonhomme, Alexandre Dulaunoy. First posted 28 Jul 2026. Read: ABS plus PDF-part. [arXiv 2607.25572](https://arxiv.org/abs/2607.25572)
  - Evidence: 1,207 expert CVEs. A first result suggested LLM labels helped, but it was an artefact of choosing checkpoints on a small test split, which alone moved recall@5 by up to 0.05 between identical runs. At about 0.39 agreement with experts, adding 100 to 984 LLM-labelled items gives no reliable gain, and about 1,000 lower macro-F1 by 0.04. More expert data always helps. Gold-only recall@5 is 0.673 ± 0.019.
  - Relevance: **the most important caution here: multi-label, a real taxonomy, rare-label harm, and an evaluation pitfall that would affect a demo.**

- **Divide, Conquer, and Distill**, Bo Yuan, Yulin Chen, Jie Cao, Mei Gao; Findings EMNLP 2026, pp. 30847-30864 ([link](https://aclanthology.org/2026.findings-emnlp.1447/)). **DeCo**, Bo Yuan, Yulin Chen, Yin Zhang; Findings ACL 2026, pp. 30504-30542 ([link](https://aclanthology.org/2026.findings-acl.1526/)). **GRIP**, Ziqing Tian, Bo Yuan; Findings EMNLP 2026, pp. 40508-40535 ([link](https://aclanthology.org/2026.findings-emnlp.1901/)). Read: ABS. Together they address clean tail items and mislabelled items looking alike under high loss, so plain noise filtering may delete rare classes. GRIP reports +9.76 accuracy points on TREC.

- **How Noisy is Automatic Annotation?** Alba Pérez-Montero, Elena Lloret, Paloma Moreda Pozo. Published October 2026, W-NUT, pp. 141-153. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.wnut-1.10/). Spanish text: rule-based errors are mostly omissions (63.99%), LLM errors lean towards additions (over 51%). Consistent with LLMs over-assigning labels (see Charalampous et al. in earlier notes).

- **ESG-SASB Label Stability.** Yufei Li, Tianhao Chen, Wei Ke, Patrick Pang. Published 3 Jul 2026, Informatics 13(7):106. Read: ABS, from Crossref. [DOI 10.3390/informatics13070106](https://doi.org/10.3390/informatics13070106). 6,460 sentences. Label reproduction is strongest at coarser levels; blind annotation flags 40.3% of held-out sentences as ambiguous; **aggregate profiles remain sensitive to label source**.

#### Active learning and weak supervision

- **Who Gets Annotated? Impact-Aware Active Learning for Robust Humanitarian Text Classification.** Divyajot Singh. Published October 2026, NLP4PI at EMNLP 2026, pp. 605-614. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.nlp4pi-1.44/)
  - How it works: picks a batch by combining expected criticality, uncertainty, diversity in embedding space and balance across events.
  - Evidence: budgets of 50-500 labels. On untouched CrisisMMD: area under the learning curve up by 0.089 for critical-category recall, 0.026 macro-F1, 0.032 worst-event macro-F1. Critical share of the labelled set from 3.4% to 26.8%. Gains do not all transfer between embedding models and datasets; no setting dominates. A 512-dimension embedding halves storage and stays within 0.009.
  - Relevance: **the closest domain match. "Balance across events" maps to balance across agencies, or languages (my inference).** Crisis text rather than long reports.

- **EvoPool: Evolutionary Programmatic Annotation.** Tianyi Xu, Yaolun Zhang, Xuan Ouyang, Huazheng Wang. Published October 2026, Findings of EMNLP 2026, pp. 15617-15656. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.743/)
  - How it works: agents write labelling functions as code; a small validation set scores them; a gate keeps only those adding value; votes become soft labels for a student.
  - Evidence: runs 4,500 to 31,000× faster than LLM annotation on 100K examples; beats the best LLM annotation on 7 of 8 weak-LLM tasks by +0.132 macro-F1 on average, including dense multi-label biomedical classification (+0.268 on PubMed).
  - Relevance: a very interesting cost idea for keyword-rich labels. Code-written labellers may be weak on abstract themes (my inference).

- **How many labels before returns flatten?** *EURO-5K* (Marios Koniaris, Vasileios Kotronis, Eugenia Giannini, Panayiotis Tsanakas; first posted 1 Jun 2026; ABS; [arXiv 2606.02971](https://arxiv.org/abs/2606.02971)): sentence-level obligations in 136 EU legislative acts; generic and legal BERT reach 0.89 F1, and all approaches converge at around 3K samples. *Where A Small Language Model Helps in Invoice Categorisation* (Emma Ceccherini, Daniel Lawson, Anjulika Salhan; first posted 18 Aug 2026; ABS; [arXiv 2608.18033](https://arxiv.org/abs/2608.18033)): a fine-tuned small sentence encoder reaches 0.96 accuracy, above a zero-shot LLM, and 0.9 F1 with about 100 client-specific invoices. Neither involves a large taxonomy.
- **Measure Before You Train.** Federico Lucca. First posted 23 Oct 2026. Read: ABS, partly. [arXiv 2610.32955](https://arxiv.org/abs/2610.32955). 231 real emails in Italian and English. With about 200 human labels, TF-IDF plus logistic regression beats every zero-shot system tested on every field (type 0.79 vs best 0.74), at about three orders of magnitude lower latency. But once decisions are composed into actions, it misses 18% of emails needing action against 4% for the best generative model. A small test, with a useful lesson: per-item accuracy may not be the metric that matters.
- *A Low-Cost Human-in-the-Loop Investigation of Toxicity on GitHub at Scale* (Rahat Rizvi Rahman, Mia Mohammad Imran, Kostadin Damevski; first posted 28 Jul 2026; ABS; [arXiv 2607.25946](https://arxiv.org/abs/2607.25946)): a small local LLM gives a label and interpretable category scores, then a Random Forest flags likely mislabels for human review. It beats confidence-based and multi-LLM baselines, applied to over 124,000 conversations.
- *Large Language Model Selection with Limited Annotations* (Yavuz Durmazkeser et al.; first posted 24 May 2026; ABS; [arXiv 2605.24981](https://arxiv.org/abs/2605.24981)): picks the most informative items to label for choosing a model; annotation cost down by up to 81.8%. Could cut the cost of choosing the demo's models (my inference).
- **HIRA** (Q1) is effectively active learning without retraining: 518 corrections (24.8% of the pool) match a fully labelled index on Tobacco-3482.
- *Beyond Keywords* (Paul Trust; first posted 25 Sep 2026; [arXiv 2609.35856](https://arxiv.org/abs/2609.35856)): weak supervision with LLM labels for hierarchical economic policy uncertainty in public-sector news. ABS without numbers.
- *Aggregating Crowd of LLMs for Cost-Effective Data Annotation* (Jiacheng Liu, Xiaofeng Hou; March 2026, Findings EACL; outside the window; [link](https://aclanthology.org/2026.findings-eacl.125/)): 12,277 segments; small open models surpass MTurk and approach GPT-4; aggregating beats each alone.

#### Prompt optimisation

- **From Errors to Rules: ERGO.** Yueying Cui, Renhao Xue, Yi Zhang, Mukul Prasad. First posted 15 Jun 2026 per arXiv search record. Read: ABS. [arXiv 2607.20497](https://arxiv.org/abs/2607.20497). Compares methods on 2 to 150 classes: no method dominates. Error-driven rules win where errors concentrate in confused pairs (TREC 90.0%, CLINC150 94.4%), exploration wins on many-class intent, and demonstrations win on coverage. The result fits SLKR and Confusion to Clarity: pairwise rules recur as a useful lever.
- **Confusion to Clarity** (Q2): MIPROv2 and GEPA over a full label list underperform plain retrieval and zero-shot. This is three datasets and one model, and ERGO reports that exploration-based search can win on many-class intent, so the result is not general.
- **LLM Prompt Duel Optimizer.** Yuanchen Wu et al. (nine authors). Findings ACL 2026, pp. 10066-10089. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.findings-acl.490/). Prompt selection with no labels by LLM judge comparison; tested on BIG-bench Hard and MS MARCO, not large-label classification.
- *ClusterFewshot* (Omri Bar Haim, Shahar Katz, Lior Wolf; first posted 22 Sep 2026; [arXiv 2609.25939](https://arxiv.org/abs/2609.25939)) and *A Reproducible Optimisation Protocol* (Teo Susnjak; first posted 7 May 2026; [arXiv 2605.06937](https://arxiv.org/abs/2605.06937); DSPy with GEPA, small student and large reflection model). Read: ABS.
- *Mosaic* (earlier notes): generates label descriptions for a 140-class taxonomy from unlabelled text.
- *Refining and Reusing Annotation Guidelines for LLM Annotation* (Kon Woo Kim, Jin-Dong Kim, Akiko Aizawa; ACL 2026 Long, pp. 37951-37964; [link](https://aclanthology.org/2026.acl-long.1760/)). Read: ABS. Reusing written guidelines helps LLMs match benchmark conventions, on biomedical NER.

### Inferences
- A defensible sequence: (1) human-label a few hundred to a few thousand chunks, stratified by top-level branch, language and agency; (2) measure LLM agreement per label; (3) use silver labels only for labels above an agreement threshold; (4) keep gold entirely out of training and selection. I found no evidence for what that threshold should be. The CVE failure at 0.39 is a data point, not a bound.
- For head labels, an embedding classifier trained on silver labels is probably the cheapest tier. For rare leaves, PROBE and the noise papers suggest silver alone will not be enough. Inference across papers.
- Validation with small test sets can mislead: CVE showed 0.05 swings between identical runs. Any demo comparison should average several seeds and select on a separate split.

### Gaps
- **I found no paper in the window that trains a classifier on LLM silver labels for hundreds or thousands of classes and evaluates it against human gold.** Kasliwal and Singh at 112 topics is nearest (only abstract read).
- No active learning study at large taxonomy scale or on long reports.
- No new result found on human-annotated samples from evaluation reports.

---

## Q4. Chunk to document: how to aggregate, whether a summary or the first pages is enough, and whether anyone tests chunk-then-aggregate against a document-level gold standard

### Takeaway
**The earlier "not found" needs correcting: direct tests exist, but none at report length or with a large taxonomy.** In the window, *Count Evidence, Not Sentences* compares direct prediction, majority vote, soft vote and certainty-weighted log-odds on 8,358 human-labelled posts: equal-weight voting does not consistently beat reading the whole post, while the weighted rule gains 4.5 accuracy points at about three times the token cost (part of that benchmark is synthetic or LLM-edited). An older, tiny test (10 documents, 17 labels) found any-chunk rollup beat whole-document prompting against human raters. The closest match in shape, the AI Risk Observatory (9,821 annual reports), rolls passage labels up to reports by rule but validates only at passage level. **I found no test on long reports against report-level gold, with hundreds of labels, of section weighting, or of summary versus all chunks, and that remaining gap is the opportunity for the blog post.**

### Cited Findings

#### Direct tests against a document-level human reference

- **Count Evidence, Not Sentences: Tempered Evidence Fusion of LLM Judgments for Long-Text Value Measurement (TEF).** Yuhe Wu, Rui Qian, Guangyu Wang, Yuran Chen, Yuanchao Zhu, Junjie Yang, Zhengheng Li, Jiulin Cai, Tianyi Zhang, Zihan Dong, Jiaxin Liu, Yujie Chen, Guang Zhang. First posted 22 Sep 2026. Preprint; its code repository name mentions ICASSP 2027, but I could not confirm any acceptance. Read: PDF, Sections 3-5, Tables 2-4 (Table 3 partly, by extracted columns). [arXiv 2609.27165](https://arxiv.org/abs/2609.27165)
  - How it works: (1) split the post into sentences. (2) Ask the LLM once per sentence with a single-token answer and keep the probability for each category. (3) Give each sentence a weight of 1 minus its entropy divided by log K, so an unsure sentence counts for almost nothing. (4) For each category, sum weight × log(p / (1 − p)) across sentences, clipping extreme values (ε = 10⁻⁶, log-odds at ±10). (5) Pick the highest score. No training.
  - Evidence: MIND benchmark, 5,474 Chinese and 2,884 English posts, six value dimensions, five LLMs. Average +4.5 accuracy and +4.6 macro-F1 over the best of Direct, Majority Vote and Soft Vote. Best or tied in 116 of 120 dimension comparisons, never more than 2.6 points behind. **"Segmentation alone does not ensure improvement": equal-weight voting does not consistently improve on Direct.** Gains concentrate on smaller models: Chinese culture accuracy on Qwen2.5-7B is Direct 44.6, MV 44.4, SV 43.5, TEF 72.1, but on Qwen3-14B it is 73.7, 69.5, 73.7, 73.8. Cost on Qwen2.5-7B: Direct 0.5M tokens and $0.09; sentence methods 2.7M tokens and $0.27-0.28. Removing either the entropy weight or the log-odds drops DeepSeek-V3.2 below Soft Vote on average, which shows the two parts work together on that model.
  - Limits: labels came from three annotators per round with two needing to agree, but a later stage **adds LLM-generated samples and has an LLM edit ambiguous texts to make the labelled orientation explicit**, so the test data is partly machine-made and may favour rules that reward decisive sentences (my reading; the paper does not test this). Social media posts, not reports. Single-label per dimension with few categories. The 28-point jump on one model from a change of fusion rule alone is large enough that it should be replicated before being relied on (my view). Calibration was shown on one model only.
  - Relevance: **the most useful aggregation result found, and simple to demo: it needs only per-chunk probabilities, and the rule fits in a few lines.** Applying it per label across a taxonomy (one yes/no per candidate) is my extension and is untested. Sentence units also differ from the use case's chunks.

- **Text Chunking for Document Classification for Urban Systems Management using Large Language Models [older].** Joshua Rodriguez, Om Sanan, Guillermo Vizarreta-Luna, Steven A. Conrad. First posted 31 Mar 2025. The arXiv record links DOI 10.1049/smc2.70023 (journal record not opened). Read: PDF, Algorithms 1-2, Tables 2-3. [arXiv 2504.00274](https://arxiv.org/abs/2504.00274)
  - How it works: (1) ask for each of 17 characteristics whether the paper shows it. (2) Whole-text: one call per characteristic per document. Chunk: one call per chunk, marking the document TRUE if **any** chunk says TRUE. (3) Repeat 15 times and take the most common answer. (4) Compare with the consensus of three human raters.
  - Evidence (Table 2, whole / chunk): GPT-4o accuracy 74.12 / 87.65, recall 76.58 / 95.09, precision 91.68 / 90.63. GPT-4o-mini accuracy 77.65 / 84.12, recall 82.61 / 86.52, precision 90.17 / 93.64. o1-mini accuracy 72.35 / 87.06, recall 72.18 / 94.91, precision 93.85 / 89.90. Recall differences significant at 1% overall; precision not significant. Consensus κ with humans for GPT-4o-mini: 0.346 whole, 0.437 chunk; humans alone κ=0.434.
  - Limits: 10 documents, 170 judgments, older models, and the authors warn against blanket use. Humans themselves agree only moderately.
  - Relevance: **the nearest direct evidence. The surprising part is that any-chunk rollup did not hurt precision here.** With 146 chunks per report and thousands of labels, false positives from any-chunk would be expected to grow; hypothesis, not measured.

- **An auditable pipeline for fuzzy full-text screening in systematic reviews.** Arya Rahgozar, Pouria Mortezaagha. Published 26 May 2026, Artificial Intelligence Review. Read: ABS-OA. [DOI 10.1007/s10462-026-11599-2](https://doi.org/10.1007/s10462-026-11599-2)
  - How it works: (1) overlapping chunks, embedded. (2) For each criterion, similarity to inclusion text minus similarity to exclusion text, plus a vagueness margin. (3) A fuzzy controller produces graded inclusion. (4) An LLM judges only highlighted spans, with confidence and rationale. (5) Weak evidence lowers membership rather than excluding.
  - Evidence: 16 full texts, 3,208 chunks, document-level recall 75.00-87.50% by criterion vs 56.25-75.00% statistical baselines and 43.75-81.25% crisp baselines. Human-machine agreement 96.07%. **All-positive gold set, so precision is not measurable, as the authors state.**
  - Relevance: **in the window and the closest design: chunk evidence, multi-label, document gold. Too small and one-sided to settle anything.**

- **Asymmetric Effects of Cross-Segment Contextualization in Long Legal Document Classification.** Meng-Luen Wu, Chao-Wei Liu. First posted 19 Oct 2026, Research Square. Read: ABS. Preprint, not peer reviewed as far as I can tell. [DOI 10.21203/rs.3.rs-10632286/v1](https://doi.org/10.21203/rs.3.rs-10632286/v1)
  - Evidence: ECtHR binary violation, document-level labels. Equal-weight aggregation of segments macro-F1 83.0; adding BiGRU 82.5 (recall 80.4 to 81.3, precision 90.2 to 84.5); attention alone 82.8; BiGRU plus attention 84.6. Overlap of 100 tokens best; larger overlaps hurt.
  - Relevance: a controlled aggregator comparison, where **equal weighting is beaten only by the combination of both added parts**. Encoders trained on document labels; no LLM chunk labels.

- **Beyond Chunking: Efficient Global Pooling for Holistic Long-Document Representation (SATPool).** Hoin Jung, Xiaoqian Wang. Published October 2026, EMNLP 2026 Main, pp. 47372-47397. Read: PDF-part, Table 2. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2080/)
  - How it works: chunk, encode, then let tokens attend across the whole document, then compress to one vector. Encoder stays frozen; only the pooling and head are trained.
  - Evidence (F1, EURLEX with BERT, my reading of extracted columns): [CLS] 0.5645, mean of chunks 0.6856, late chunking 0.6763, PARADE 0.6318, TrLDC 0.6277, SATPool 0.7124. Multi-label document gold. **Plain mean pooling beats two hierarchical aggregators in that column.** English datasets only, as the authors note.
  - Relevance: a cheap baseline, if chunk embeddings already exist for this corpus.

- **Classifying long legal documents using short random chunks [older].** Luis Adrián Cabrera-Diego. First posted 31 Dec 2025. Read: PDF-part, Table 3. [arXiv 2512.24997](https://arxiv.org/abs/2512.24997)
  - How it works: encode a random sample of 48 chunks of at most 128 tokens with DeBERTa v3, run an LSTM over them, and redraw the sample each epoch. Runs on CPU.
  - Evidence: expert-labelled 18 classes, a corpus spanning 25 languages. Best model weighted F 0.898; median 0.891-0.893 in the rows I read. 498 seconds per 100 files on CPU. "Even as few as 20 chunks" works, with some loss (figure not extracted).
  - Relevance: **strong evidence that sampling chunks can replace reading all of them for a coarse label.** 18 classes only; long-tail labels may behave very differently.

- **Learning Long-Document Embeddings via Chunk-Context Entailment (CPE).** Waheed Ahmed Abro, Naïm Es-Sebbani, Zied Bouraoui. Published May 2026, LREC, pp. 7405-7414. Read: PDF-part. [ACL Anthology](https://aclanthology.org/2026.lrec-1.586/). Self-supervised, needs no labels. On ECHR, SCOTUS, EURLEX, MIMIC, it gains about +2-4 macro-F1 over SimCSE and ESimCSE, and is more compute-efficient than prompt-only LLMs at matched token budgets (specific figures not extracted).

#### Closest in shape to the use case, but not validated at report level

- **The AI Risk Observatory: What Can We Learn from AI Disclosures in Annual Reports About Societal Resilience?** Bart Jaworski, Andrew Sutton. First posted 1 Oct 2026. Read: PDF, Sections 3-4.1 and 5.3 limitations. [arXiv 2610.02281](https://arxiv.org/abs/2610.02281)
  - How it works: (1) normalise reports to markdown, keeping headings and section boundaries. (2) A keyword gate: only passages holding a listed AI term continue, taken with two paragraphs of context either side. (3) Stage 1, Gemini 3 Flash assigns mention types (multi-label, confidence at least 0.2 kept). (4) Stage 2, further classifiers assign a ten-category risk taxonomy, adoption types, vendor, and a 1-3 substantiveness rating. (5) **Report-level rollup in code: a report "mentions" a type if any passage has it, and its substantiveness is the band held by the most passages, ties going to the higher band.**
  - Evidence: 9,821 annual reports from 1,362 companies reduced to 24,189 passages (about 2.5 per report, my arithmetic). Against 474 human-annotated passages from 30 reports: mean Jaccard 0.73 for Stage 1, 0.41 for adoption types, 0.33 for the ten-category risk taxonomy. Stage 1 recall 0.91-0.97, with lower precision. All 40 passages tagged as harm were false positives, which prompted a manual review of every harm tag. **The any-passage rule makes breadth look high: in 2025, 41.2% of reports mention AI risk, but only 4.3% have disclosure rated substantive.** No money or token figures found in my keyword search of the text.
  - Limits, as stated by the authors: a single annotator, and adjudication reviewed disagreements with the classifier's output (which could favour the model, my reading). The validation set is large-cap 2023-2024 filings only. The keyword gate creates false negatives, and the net direction of error is not quantified. **There is no report-level gold standard, so the rollup itself is not tested.** A narrow single-topic gate, not a broad taxonomy.
  - Relevance: **the best published template for the corpus: long reports, a tiered label scheme, passage-level labels and a rule-based rollup, with data released under CC BY 4.0.** It also shows agreement falling as the taxonomy gets finer (0.73 to 0.33). It makes a good foil for the blog post: same design, plus the report-level test it lacks. A keyword gate per label or branch is my extrapolation, and false negatives on implicit themes would need measuring.

#### Aggregation design, without a document-level classification test

- **Model Diversity Over Model Size: Unanimous LLM Ensembles Correct Over-Classification in Survey Coding.** Chris Soria. First posted 5 Jun 2026, SocArXiv (Crossref lists a v2 on 25 Aug 2026 and a further record on 3 Nov 2026). Read: ABS, from Crossref (v1). [DOI 10.31235/osf.io/er6mz_v1](https://doi.org/10.31235/osf.io/er6mz_v1)
  - How it works: require several models from different providers to agree before a category is assigned.
  - Evidence: four survey questions, 3,208 responses, six categories each, 16 models, human-coded truth. On the most ambiguous categories, false positives fall from 50% to 3% and precision triples; clear categories show no benefit. Cross-provider access was the most reliable source of differing errors; temperature and model size within a family were not. Three diverse lower-tier models were enough to exceed GPT-5 reliably.
  - Relevance: short responses, not chunks, but it speaks directly to the over-assignment risk in any-chunk rollup. Agreement could be spent only on a label's strongest chunks (my suggestion, untested).

- **Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information [older].** Rui Ai, Yuqi Pan, David Simchi-Levi, Milind Tambe, Haifeng Xu. arXiv ID from TEF's reference list; October 2025 and ICML 2026 (PMLR 306) rest on a search summary, not a primary record I opened. Read: SNIPPET. [arXiv 2510.01499](https://arxiv.org/abs/2510.01499). Weights models by estimated reliability and correlation instead of counting votes. It fuses models rather than chunks, so it is a neighbouring idea only.

- **A Model for Imbalanced Label Aggregation.** Gabriel Singer, Samuel Gruffaz, Olivier Vo Van, Nicolas Vayatis, Argyris Kalogeratos. First posted 27 Jul 2026. Read: ABS. [arXiv 2607.24622](https://arxiv.org/abs/2607.24622). Crowdsourcing, 33 datasets; models annotator skill per class and item difficulty, and consistently gets the highest minority recall. The authors show majority voting asymptotically preserves class proportions. The rare-label angle transfers as an idea; chunks are not annotators.

- **Consensus Drift: Semantic Salience Can Override Majority Evidence in LLM Aggregation.** Gokul Srinath Seetha Ram, Rashmi Elavazhagan, Anirudh Vasudevan. Published October 2026, EMNLP 2026 Main, pp. 38411-38428. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1689/)
  - Evidence: when jurors give only YES/NO, minority override is near zero. When they give evidence statements, a strong-sounding minority can override a weak majority, depending on position, and more so on GPT-4.1 than smaller models. Synthetic strengths.
  - Relevance: **an LLM "reduce" step over chunk evidence may be swayed by wording and order** (my extrapolation). Counting in code has no such problem.

- **GraphReduce: Coverage-Preserving LLM Aggregation for E-commerce Review Insights.** Hao Jiang, Xin Li, Yichi Zhang, Weisi Lin. Published October 2026, EMNLP 2026 Industry, pp. 1545-1557. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.emnlp-industry.91/)
  - How it works: extract tuples per item; cluster using frozen embeddings plus a small projection; let the LLM only name and rank clusters; keep track of membership outside the model.
  - Evidence: flat LLM baselines look fine on ranking metrics, yet omit substantial evidence.
  - Relevance: **a map-reduce pattern that suits traceable report tags: every report label keeps its supporting chunk IDs** (my application). Reviews, not reports.

- **When Chunks Disagree.** Weichao Zhou, Chuanfu Zhang. EECCT 2026, September 2026. Read: ABS-OA. [DOI](https://doi.org/10.1109/eecct71142.2026.11746532). Chunk-independent extraction assigns inconsistent types across chunks. Re-checking only conflicts against compressed evidence lifts the conflict resolution rate from 58.33% to 81.82% for GPT-5.5. The end-to-end gain on 43 real conflicts is not significant.

- **RAPT** (Q2): per-document thresholds or label counts derived from similar cases. **This is the closest thing found to a learned evidence threshold.** It uses document-level scores, not chunk rollups.

- **Evidence dilution and bias in document embeddings:**
  - *Lost in a Single Vector.* Shanshan Lyu, Yiwei Wang, Yujun Cai, Jiafeng Guo, Shenghua Liu. First posted 17 Jun 2026. Read: ABS. [arXiv 2606.18781](https://arxiv.org/abs/2606.18781). Retrieval, not classification. Aggregating chunk vectors instead of embedding one long text lifts Passkey above 4k tokens from 30.0 to 90.0, on one model. Lower dilution in 92.8% of 12,779 samples.
  - *Information Representation Fairness in Long-Document Embeddings.* Elias Schuhmacher, Andrianos Michail, Juri Opitz, Rico Sennrich, Simon Clematide. Published July 2026, Findings of ACL 2026, pp. 4996-5028. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.findings-acl.246/). **Early segments and segments in higher-resource languages such as English are over-represented.** Attention calibration at inference helps later segments.
  - Relevance: **a concrete reason to be wary of embedding whole reports, and a warning for French or Spanish sections in mixed documents** (my application).

- *Explainable Long-Document Resume Classification.* Mst. Bithi Akter Sarmin et al. PECCII 2026, June 2026. Read: ABS-OA, partly. [DOI](https://doi.org/10.1109/peccii70991.2026.11661871). 512-token overlapping chunks, max pooling, 50 classes, F1 0.9337 cross-validated. No comparison with other pooling rules in what I read.

#### Summary, first pages, or partial reading

- **Task Cascades** (Q1): the most relevant large-scale test of reading only part of a document. Table 3 includes baselines such as naive RAG filtering and restructuring to the top 25%. Results vary by task: for example Top-25% costs 0.85× of the two-model cascade on Enron but 2.67× on AG News. Reference is the oracle LLM rather than humans.

- **Summarization-Aware Inference Pipelines for Cost-Efficient Suicide Risk Classification.** Aditya Tekale, Mohammad Masum. COMPSAC 2026, July 2026. Read: ABS-OA. [DOI 10.1109/compsac69091.2026.00037](https://doi.org/10.1109/compsac69091.2026.00037)
  - Evidence: misclassification is significantly associated with input length. Summarising long posts cuts length about 44%, improves small models by up to 10%, and cuts inference cost by 30-54%. Balanced accuracy peaks at only 0.53 before summarisation. Long Reddit posts, far shorter than reports.

- **HyTRAM: A hybrid text classification framework for fire accident investigation reports via LLM-based text compression and GCN-LSTM.** Ying Wang, Heng Peng, Yuping Han. Results in Engineering, December 2026 issue; Crossref record created 7 Aug 2026. Read: ABS-OA. [DOI 10.1016/j.rineng.2026.112355](https://doi.org/10.1016/j.rineng.2026.112355). Compresses reports to causal summaries before classifying; precision 85.76% on FIREAR, above pretrained baselines. Accident reports are the closest document genre found. **No comparison with classifying all chunks in what I read.**

- **Budget-Aware Routing for Long Clinical Text.** Khizar Qureshi, Geoffrey Martin, Yifan Peng. Published July 2026, Findings of ACL 2026, pp. 42583-42598. Read: ABS plus PDF-part. [ACL Anthology](https://aclanthology.org/2026.findings-acl.2114/)
  - How it works: treats picking segments under a token budget as a knapsack problem, comparing segmenting by sentence, section, window or cluster, and selection by lead text, relevance or diversity.
  - Evidence: **selector choice matters more than how you segment**. Lead text wins at low budgets in extractive tasks; the authors attribute this to clinical notes front-loading content. Diversity (MMR) improves LLM generation. Selection at 512 tokens can match full-context ROUGE-1. **Summarisation evaluation, not classification.**

- **Page classification before the LLM:** *NLP-based Page Classification for Efficient LLM Extraction from Brazilian Public Tender Documents.* Pedro Campos, Ivo de Medeiros, Adailton de Araújo. PROPOR 2026, month not checked. Read: ABS from PDF. [ACL Anthology](https://aclanthology.org/2026.propor-1.61.pdf). 11,190 pages from 350 documents. XGBoost with engineered features 97.75% F1, beating fine-tuned BERT by over 4 points. Tokens down 64-88% with extraction completeness retained.
  - *Operationalizing Document AI* (Yao Fehlis et al.; first posted 12 May 2026; [arXiv 2605.18818](https://arxiv.org/abs/2605.18818)): abstract read; numbers (page classification at 92%, fallback on 4% of pages, about 10× lower cost) are **SNIPPET only**, check before use.

- **More text is not reliably better:**
  - [Kähler et al.](https://arxiv.org/abs/2607.14882) (earlier notes): XR-Transformer F1 0.467 titles vs 0.398 full text; LLM-Ensemble 0.382 vs 0.364, though training set sizes differ.
  - [Golub et al.](https://doi.org/10.1108/jd-06-2026-0363) (earlier notes): metadata F1 44% vs full text 12% (14 works of fiction).
  - *SumRank* (Jincheng Feng, Wenhan Liu, Zhicheng Dou; EMNLP 2026 Main; [link](https://aclanthology.org/2026.emnlp-main.1451/); ABS): a summariser trained for the ranking task. By analogy, a summary might best be written for the taxonomy, a hypothesis not tested in classification.

- **Not read, may be worth obtaining:** *Hierarchical Compositional Hypergraphs Encode Document Structure for Classification* (Madjid Eshaghi Gordji, Mohamadali Berahman; Research Square; ABS; small gain on 20 Newsgroups; [DOI](https://doi.org/10.21203/rs.3.rs-10829259/v1)). *Generative LLM for adaptive taxonomy-informed multi-label classification in construction safety* (Qing Shuang, Lujie Qi, Huaiyuan Zhai; Automation in Construction; META, no abstract available; [DOI](https://doi.org/10.1016/j.autcon.2026.107297)). *Ex-ante Identification of Disruptive Technologies* (Fangyuan Ding, Dongyuan Lu; Research Square; rolls patent evidence up to technology-level themes; ABS, partly; [DOI](https://doi.org/10.21203/rs.3.rs-9585548/v1)). *Evaluation of Embedding-Based and Generative Methods for LLM-Driven Document Classification* ([arXiv 2604.04997](https://arxiv.org/abs/2604.04997); April 2026 by search result; first page only; SNIPPET).

### Inferences
- **Suggested experiment (my proposal, not found in the literature).** Take a sample of reports with human document-level labels, including publisher tags as a weaker reference. Compare on one taxonomy:
  1. all chunks with LLM labels, then (a) any-chunk, (b) at least n chunks, (c) share of chunks above a threshold per label, (d) certainty-weighted log-odds as in TEF, (e) calibrated noisy-OR over chunk probabilities;
  2. embedding mean of chunks plus a classifier;
  3. first pages or executive summary only;
  4. a random sample of 20 and 48 chunks;
  5. a taxonomy-aware LLM summary of the report, then one classification call.
  Report by label frequency, tier, language and cost. At report length and taxonomy size this would be new, as far as I can find.
- Keep aggregation in code, and keep chunk IDs as evidence for each report label. TEF, Consensus Drift and GraphReduce support this, though none tests report classification.
- A log-odds sum grows with the number of chunks, so long reports may pile up small amounts of evidence for many labels. TEF's weighting reduces this but was tested on posts, not on 146-chunk reports. Hypothesis, to be measured; a normalising step or a threshold tuned on gold may be needed.
- Noisy-OR and count thresholds are my suggestions of candidates; I found no recent paper that evaluates either on LLM chunk labels. Noisy-OR and TEF both rely on chunk probabilities, and Reiter et al. warn that confidences are design-sensitive, so probabilities from batched prompts should not be mixed with those from single prompts (my inference).
- Evaluation reports have predictable sections (executive summary, findings, recommendations). Section weighting seems natural but is **untested**. Budget-Aware Routing suggests that selection matters more than segmentation, but in a different task.

### Gaps
- **Found, correcting the earlier notes:** a comparison of direct prediction and aggregation rules on LLM segment labels against human document labels (TEF), on 8,358 posts.
- **Still unfound, after searching arXiv, ACL Anthology 2026, Crossref and web search:** the same comparison on long reports, or with a taxonomy of hundreds of labels or more, or for multi-label output, or including map-reduce or section weights.
- **Still unfound:** a head-to-head of summary, first pages and all chunks for multi-label classification of long reports. Web search on first page versus whole document also found none in the window.
- **Section-type weighting:** nothing evaluated. Evidence thresholds per label: only RAPT at the document-score level.
- Coverage caveat: TEF surfaced only when an unquoted query used different words ("sentence level labels document level label aggregation"), having been missed by earlier phrase queries. Other papers using other vocabulary may also have been missed, so these "not found" statements are not proof of absence.

---

## Q5. Batch APIs, prompt caching and other serving tricks reported in research papers, with numbers

### Takeaway
Few papers report these with real bills. The best is PromptPack: batch API alone saves 50%, prompt caching a further 70-80% against uncached batch where the two combine, and in-prompt batching of 20 items saves 89% with AUC unchanged, with the shared prompt making up 94% of input tokens. A long taxonomy prompt has the same shape, but in-prompt batching can affect quality and confidence, and caching does not speed up generation.

### Cited Findings

- **PromptPack: Scaling LLM Annotation Agents for Online Recommendation.** Sebastian Koralewski, Merwan Barlier, Yulia Stolin, Blaž Škrlj (Teads). First posted 10 Jul 2026. AI4IR 2026 per page header. Read: PDF, Sections 6-7, Tables 3-4. [arXiv 2607.20528](https://arxiv.org/abs/2607.20528). Vendor-authored.
  - How it works: (1) one 8,500-token system prompt containing the taxonomy, sent once per call; (2) items inside a strict XML envelope with row IDs; (3) a correction layer detects missing or malformed rows and retries them singly; (4) asynchronous batch API tier on top.
  - Evidence:
    - Shared instructions are 94% of billed input tokens.
    - Batch API: 50% off with no loss of accuracy.
    - Prompt caching combined with batch: cached input 80-90% cheaper, overall 70-80% lower than uncached batch. In-prompt batch 20 adds a further 52-57%.
    - gpt-4.1-nano at batch size 20: **89% lower cost**, AUC 0.609 vs 0.608, 28 vs 71 seconds per 1,000. Worst degradation across models 0.012 AUC.
    - Caching barely changes wall-clock time (claude-haiku-4.5 at batch 20: 50.0 s with cache vs 48.7 s without). The authors explain that it only speeds prefill, and about 2,600 output tokens still have to be generated.
    - Reported constraint at the time of writing: OpenAI restricts caching on the async batch tier to the gpt-5 family. Check current provider terms.
    - Permutation ensembling buys about 0.05 AUC, reducing savings to 67.1% (K=3) or 45.1% (K=5).
  - Relevance: **the best single source for this question, with a taxonomy in the prompt. The quality metric is downstream ranking AUC, not label accuracy, so label-level harm could be hidden.**

- **Towards Cost-effective LLMs Routing with Batch Prompting (RoBatch).** Haotian Xu, Kangfei Zhao, Jiadong Xie. First posted 27 May 2026. Read: ABS. [arXiv 2605.28268](https://arxiv.org/abs/2605.28268). Chooses both the model and batch size per query under a budget, by upgrading greedily along the Pareto front. Six benchmarks, better cost-performance front than either alone; no figures in the abstract.

- **Cascaded Batch Prompting.** Sho Hoshino, Peinan Zhang. First posted 27 Aug 2026. Published October 2026, Findings of EMNLP 2026, pp. 2462-2469. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.119/); [arXiv 2608.27038](https://arxiv.org/abs/2608.27038). Separates reasoning from answer assignment in two stages; speed-up proportional to batch size while beating single prompting. Tested on multiple choice and NLI.

- **From Generation to Scoring: Scaling LLM-Based Semantic Labeling at LinkedIn.** Jian Sheng et al. (15 authors). First posted 26 Oct 2026. Read: ABS plus PDF-part. [arXiv 2610.36383](https://arxiv.org/abs/2610.36383). Vendor-authored.
  - How it works: instead of asking for YES/NO across many labels in one generated list, fine-tune to score one candidate at a time from the prefill pass with no decoding, then score hundreds of candidates in one sequence with attention masked between them.
  - Evidence: reordering candidates changes at most 0.01% of decisions; matches packed generation's agreement with about 4.7× fewer training pairs; throughput 1.24× offline, 1.9× serving. Open implementation proposed for Ray Data and SGLang.
  - Relevance: **a direct design for labelling one chunk against many candidates, and low sensitivity to order is relevant to the position effects in Lost in Choice** (my link). It needs self-hosting and fine-tuning.

- **Cache-Aware Prompt Compression.** Yan Song. First posted 16 Jul 2026. Read: ABS. [arXiv 2607.15516](https://arxiv.org/abs/2607.15516). Single author. On Sonnet 4.6, cache hit rate plateaus near 0.83 below about 3,500 tokens. Per-query compression breaks prefix caching. Query-independent compression plus explicit cache control: mean 49% saving over cache only and 90% over no caching on LongBench-v2. **Practical lesson (my reading): keep the shared taxonomy prefix stable and large enough, and put the chunk last.**

- **Silent and Costly: Prompt-Caching Mis-Integration.** Ying Zhang et al. (seven authors). First posted 26 Oct 2026. Read: ABS. [arXiv 2610.34817](https://arxiv.org/abs/2610.34817). 120 caching-related pull requests in 10 agent systems: most errors fail silently. Log cache hit tokens, rather than assume.

- **Token Reduction Is Not Cost Reduction.** Sarel Weinberger, Amir Hozez. First posted 13 Jul 2026. Read: ABS. [arXiv 2607.12161](https://arxiv.org/abs/2607.12161). Coding agents, so an indirect result: cutting tool output 38.4% raised billed cost 6.8%. Report billed cost, not tokens.

- **Cost-Efficient LLM-Assisted Title and Abstract Screening.** Dimitri Belenki et al. (seven authors). First posted 4 Sep 2026, Research Square. Read: ABS. [DOI 10.21203/rs.3.rs-10909832/v1](https://doi.org/10.21203/rs.3.rs-10909832/v1). gpt-4.1-mini: sensitivity 96.4%, workload down 80.4%, about **$1.0 per 1,000 records on the batch API**. A rare published batch price point.

- **Reusing judgments across runs:**
  - *OCTOPUS.* GuangXue Zhang, Nada Lahjouji, Yiming Lin, Ashwin Colaco, Sharad Mehrotra. First posted 23 Oct 2026. Read: ABS. [arXiv 2610.32586](https://arxiv.org/abs/2610.32586). Writes every model judgment back to the database as a tag, so later queries reuse it; 10-108× fewer rows sent to the model on one SemBench scenario.
  - *Semantic Views.* Oktie Hassanzadeh, Dharmashankar Subramanian. First posted 8 Oct 2026. Read: ABS. [arXiv 2610.12768](https://arxiv.org/abs/2610.12768). Reuses earlier LLM labels to order items, then certifies with fresh stratified samples, so a stale label can cost calls but not accuracy. Replayed on 725,000 recorded labels: meets precision and recall targets on 99.63% of queries, saving 28-45% of calls.
  - *SemReuse.* Phat T. Tran-Truong et al. First posted 12 Oct 2026. Read: ABS. [arXiv 2610.16035](https://arxiv.org/abs/2610.16035). Reuses earlier filter results with a statistical audit; up to 15.4× reductions.
  - *NousDB.* Christos Tsapelas, Georgia Koutrika, Yannis Ioannidis. First posted 20 Oct 2026. Read: ABS. [arXiv 2610.28551](https://arxiv.org/abs/2610.28551). Token histograms to set safe batch size under context limits; up to 86× speed-up.
  - Relevance: if a taxonomy changes, stored chunk labels and their evidence could make re-tagging cheap (my inference). Hierarchy may help, since a NO on a branch can settle its children, but this was not tested on taxonomies here.

- **Other:**
  - *CaBSALLM.* Mohammadsadegh Abolhasani, Reza Mousavi, Paul Jen-Hwa Hu. ACL 2026 Short, pp. 615-636. Read: ABS. Dynamic batching with light context; no figures in the abstract. [Link](https://aclanthology.org/2026.acl-short.51/)
  - *LLM Cache Bandits with Query Heterogeneity* ([EMNLP Main](https://aclanthology.org/2026.emnlp-main.1849/)), *Continuous Semantic Caching for Low-Cost LLM Serving* ([DOI](https://doi.org/10.1145/3842721.3849656)), *Semantic Caching for Low-Cost LLM Serving* ([INFOCOM 2026, DOI](https://doi.org/10.1109/infocom59046.2026.11571467)), *Prompt Context Caching Architecture* ([IJISAE, DOI](https://doi.org/10.17762/ijisae.v14i1s.8385); ABS-OA, no measured result in the part I read). META except as stated. Semantic caching seems less useful for near-unique chunks, but boilerplate may repeat across UN reports, and duplicates could be removed before labelling (my inference, not measured on this corpus).

### Inferences

**Everything below is my own estimate, not a finding. It is an order-of-magnitude guide only.**

Assumptions: 1.9 million chunks; prices are those printed in the papers, not checked against current provider lists; BANKING77 queries in SLKR are probably shorter than a UN chunk, so real input costs would be higher (perhaps 1.2 to 2 times, my guess); escalation rates are assumed, not measured; embedding and retrieval costs excluded; no batch or cache discounts applied unless stated.

| Scenario | Basis | Estimated cost |
|---|---|---|
| Strong LLM on every chunk | SLKR Table 6, $4.111 per 1k | about $7,800 |
| Same, with Abusaqer and Saquer's standalone Sonnet price | $96.29 for 20,997 posts | about $8,700 |
| Small LLM reader on every chunk | SLKR, $0.0596 per 1k | about $110 |
| Two tiers, 5% escalated | $0.0596 + 5% of $4.111 per 1k | about $500 |
| Two tiers, 10% escalated | as above | about $900 |
| Two tiers, 20% escalated | as above | about $1,700 |
| Cascade at Abusaqer and Saquer's rate | $26 for 20,997 posts | about $2,400 |
| Mini model via batch API | Belenki et al., about $1.0 per 1k | about $1,900 |
| Silver labelling a 20,000-chunk sample with a strong LLM | $4.111 per 1k | about $80 |
| Label-knowledge build, 2,000 labels | $3.40 for 77 labels, scaled linearly to about $90; scaled quadratically (worst case) to about $2,300 | $90 to $2,300 once |

Reading fewer chunks per report (13,000 reports):
- 5 per report: 65,000 calls (3.4% of the corpus).
- 20 per report: 260,000 calls (13.7%).
- 48 per report: at most 624,000 calls (32.8%).

These are upper bounds where a report has fewer chunks than that. Whether quality holds at these levels on a large taxonomy is the unanswered question in Q4.

**Techniques I would shortlist for the demo and blog, judged on evidence and on how easily they can be shown:**
1. **Offline label knowledge plus a small reader** (SLKR). No labels needed, clear cost table, multilingual test.
2. **Two-tier cascade with a calibrated gate and a per-language quota** (UCCI, Bouchard, Who Pays). The per-language chart should make a good visual.
3. **Candidate recall chart, with confusion partners** (Confusion to Clarity). It shows retrieval misses and LLM misses as separate failures.
4. **Chunk-to-document experiment, with TEF against voting and direct prediction** (Q4 proposal). The strongest blog angle: published tests stop well short of long reports with a large taxonomy, and the rule is cheap once chunk probabilities exist.
5. **Prompt layout for caching and batching** (PromptPack, Cache-Aware Prompt Compression). Cheap to do and to report honestly, in billed cost.
6. **A frozen gold set and a silver-label agreement check** (SIFT, CVE paper). This is what makes the other results believable.

Caveat: these numbers come from different datasets and were never combined, so a combined saving cannot be predicted by multiplying them.

### Gaps
- Almost no paper in the set reports a real bill on a corpus of millions of documents. AutoRelAnnotator and the Walmart paper give scale but no dollar figures in their abstracts.
- Batch API and caching numbers on large-label classification rest on PromptPack alone.
- I did not find a study on the effect of in-prompt batching on chunk-level multi-label accuracy.
- **Search coverage and tool limits.**
  - *arXiv:* API for about 20 queries, then rate limited (HTTP 429). Advanced search by first-submission date, about 130 queries in all, also rate limited for a time. Another research process was using the same endpoints then, though I did not establish that as the cause. I stopped, waited and resumed at a slow pace rather than try to get around the limits, and all planned queries were eventually run.
  - *Query quality:* many multi-term phrase queries returned zero results, and a few of my early queries used cut-off words that matched nothing. Unquoted queries were re-run for the main topics, and they found TEF and the AI Risk Observatory, both missed at first. This shows the search is sensitive to wording. Results beyond the 50 shown per query were not seen, and broad queries were noisy.
  - *Other sources:* ACL Anthology event pages for EMNLP, ACL, LREC and EACL 2026 (16,883 titles filtered by keyword, abstracts read for about 50). NAACL 2026 page returned 404. Crossref API (34 queries, title filter), with a brief 429. OpenAlex lookup by DOI.
  - *Web search:* used to find leads, then checked against primary records wherever an entry does not say SNIPPET.
  - *Not checked:* SIGMOD 2026, VLDB 2026, KDD 2026 and CIKM 2026 tables of contents were not enumerated. Those venues are covered only through arXiv, Crossref and citations. Papers that use different words in their titles may be missed. This list is verified but not exhaustive.
  - *Dates:* first-posted dates for the Anthology entries are not established unless an arXiv version is named. Publication dates for SIGMOD papers come from the ACM reference printed in the PDFs, and I did not open the ACM pages.
