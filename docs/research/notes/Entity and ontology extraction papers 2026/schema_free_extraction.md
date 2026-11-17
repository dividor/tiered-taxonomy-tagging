# Schema-free entity and relation extraction and graph building from text chunks (papers from May to November 2026)

Notes compiled 5 November 2026. Use case: about 13,000 long UN evaluation reports from 31 agencies, about 1.9 million chunks, mostly English with French and Spanish, noisy PDF parsing with tables.

## Objective: verified list of papers

### Takeaway
72 papers are listed below with identifiers checked against primary sources, plus 11 just outside the window. 19 were read in full text through targeted sections, most of the rest by abstract only, and 8 by title, metadata or part of an abstract only. Four ACM papers that look highly relevant could not be opened and need obtaining by the user.

### Cited Findings

**How this list was built and verified**
- arXiv: the query API rate-limited me early on (HTTP 429), so I crawled the monthly listing pages for cs.CL, cs.IR, cs.DB and cs.AI from May to November 2026 (47,999 IDs), filtered titles by keyword, then opened the abstract page of each candidate. A later API check confirmed additional first-posted dates by title. [Source](https://arxiv.org/list/cs.CL/2026-06)
- ACL Anthology: I downloaded the full 2026 volumes for ACL (long, short, Findings), EMNLP (main, Findings), EACL and TACL (over 10,000 papers) and filtered titles. Abstracts come from those pages. Month of publication was confirmed from the BibTeX of a sample of papers in each volume (ACL July, EMNLP October, EACL March) and applied to the rest of that volume. [Source](https://aclanthology.org/volumes/2026.acl-long/)
- SIGIR, KDD, CIKM, PVLDB and others: Crossref DOI records. ACM pages returned 403, and most ACM records carry no abstract. [Source](https://api.crossref.org/works/10.1145/3770855.3818058)
- First-posted dates come from the arXiv version 1 stamp, never from the ID. This matters: IDs and dates disagree in several cases. For example 2606.07525 has a v1 date of 20 April 2026, so it was dropped, and 2607.22592 has a v1 date of 10 June 2026. [Source](https://arxiv.org/abs/2606.07525)
- "No arXiv version found" means the arXiv title search returned nothing. A preprint may still exist under a different title.
- Names of models such as GPT-6 Luna or GPT-5.4 are given as written in the papers. I did not check them separately.
- Reading labels: "Full text, targeted" means I downloaded the PDF and read the named tables and passages, not every page. "Abstract only" abstracts were read as served, and a few were cut off near the end in my view, so absence of a number does not prove the paper lacks one.

#### A. Schema-free extraction and graph-building pipelines

**A1. AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora**
- Authors: Jiaxin Bai, Wei Fan, Qi Hu, Qing Zong, Chunyang Li, Hong Ting Tsang, Hongyu Luo, Yauwai Yim, Haoyu Huang, Xiao Zhou, Feng Qin, Tianshi Zheng, Xi Peng, Xin Yao, Huiwen Yang, Leijie Wu, JI Yi, Gong Zhang, Renhai Chen, Yangqiu Song
- First posted: 29 May 2025 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: ACL 2026 (long).
- Date note: First posted a year before the window; included because publication falls inside it.
- Links: [ACL Anthology 2026.acl-long.942](https://aclanthology.org/2026.acl-long.942/); [arXiv:2505.23628](https://arxiv.org/abs/2505.23628)
- What I read: Abstract only
- Method: LLM extracts entity and event triples and induces the schema from the same text, with no predefined schema; conceptualisation groups instances into categories.
- Evidence: ATLAS graphs from over 50 million documents: 900+ million nodes, 5.9 billion edges. Schema induction reported at 92% semantic alignment with human-made schemas. LLM-call cost per document not in the abstract. [Source](https://aclanthology.org/2026.acl-long.942/)
- Relevance: The only paper found at a scale larger than the user's corpus. High relevance as a design reference; full text should be read for cost.

**A2. SocraticKG: Knowledge Graph Construction via QA-Driven Fact Extraction**
- Authors: Sanghyeok Choi, Woosang Jeon, Kyuseok Yang, Taehyeong Kim
- First posted: 15 Jan 2026 (arXiv v1). Published: July 2026 (ACL Anthology). Venue: Findings of ACL 2026.
- Date note: First posted before the window.
- Links: [ACL Anthology 2026.findings-acl.1951](https://aclanthology.org/2026.findings-acl.1951/); [arXiv:2601.10003](https://arxiv.org/abs/2601.10003)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Tables 1, 4, 7, 8.
- Method: Generates 5W1H question-answer pairs from each document, extracts triples from each pair, then canonicalises. The question step makes references explicit before extraction.
- Evidence: MINE factual retention: 73.4% (Qwen2.5-7B) to 96.3% (Claude-4), against GraphRAG 48.5-59.7%, KGGen 44.3-69.1%, direct extraction 66.5-86.8%. HotpotQA hard-bridge 3-hop accuracy on Claude-4: 56.38 vs GraphRAG 52.12 vs naive RAG 47.88. Tokens to extraction over 100 articles: 333,917 (Qwen) and 553,530 (Claude), against direct extraction 145,262 and 150,569. Human-rated factual precision on Qwen in Table 8: 84.21 +/- 6.33 vs direct 86.42 +/- 4.21 (the prose gives 84.38, a small internal mismatch). [Source](https://aclanthology.org/2026.findings-acl.1951/)
- Relevance: High. Direct evidence on the coverage-versus-fragmentation trade-off, with a small-model column and exact token counts. MINE articles are short, so per-chunk behaviour on long reports is untested.

**A3. LLMs as Knowledge Graph Refiners: Mitigating Factual Inconsistencies in Generative Knowledge Extraction**
- Authors: Donghyun Kim, Hyeongjun Yang, Seokju Hwang, Kyong-Ho Lee, Chanhee Lee
- First posted: Not established. Published: July 2026 (ACL Anthology). Venue: ACL 2026 (long).
- Date note: First posted: no arXiv version found by title search; may exist under another title.
- Links: [ACL Anthology 2026.acl-long.1353](https://aclanthology.org/2026.acl-long.1353/)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Table 1 and setup.
- Method: Fine-tuned LLM (setup names Llama-2-7B) reviews each drafted triple and chooses to keep, delete, edit or rewrite, using training data built from human-categorised error types.
- Evidence: BL F1 from drafts built with GPT-4o-mini. GenWiki: EDC 62.92, KGGen 55.88, GraphJudge 63.40, GraphRefine 65.23. DocRED: 39.88, 41.07, 47.57, 52.94. SciERC: 55.08, 52.33, 52.91, 55.43. CDR: 33.84, 38.38, 40.16, 42.73. [Source](https://aclanthology.org/2026.acl-long.1353/)
- Relevance: High for clean-up of extraction errors. Requires a fine-tuning step and gold-derived training data.

**A4. AutoGraph-R1: End-to-End Reinforcement Learning for Knowledge Graph Construction**
- Authors: Hong Ting Tsang, Jiaxin Bai, Haoyu Huang, Qiao Xiao, Tianshi Zheng, Baixuan Xu, Shujie Liu, Yangqiu Song
- First posted: 17 Oct 2025 (arXiv v1). Published: July 2026 (ACL Anthology). Venue: ACL 2026 (long).
- Date note: First posted before the window.
- Links: [ACL Anthology 2026.acl-long.1070](https://aclanthology.org/2026.acl-long.1070/); [arXiv:2510.15339](https://arxiv.org/abs/2510.15339)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): main results table only.
- Method: Trains an LLM graph constructor with reinforcement learning, using the graph's usefulness in a RAG pipeline as the reward.
- Evidence: With Qwen2.5-3B and 7B constructors, HippoRAG and HippoRAG2 F1 gains of about 0.4-2.7 points and Recall@5 gains up to 13.5 points across six results columns. [Source](https://aclanthology.org/2026.acl-long.1070/)
- Relevance: Medium. Shows small models can be tuned for extraction, but needs a QA training set, which the user does not yet have.

**A5. MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation**
- Authors: Chuanjie Wu, Zhishang Xiang, Yunbo Tang, Zerui Chen, Qinggang Zhang, Jinsong Su
- First posted: 30 May 2026 (arXiv v1). Published: 8 August 2026 (Crossref). Venue: KDD 2026 (arXiv comment and Crossref agree).
- Links: [arXiv:2606.00610](https://arxiv.org/abs/2606.00610); [DOI 10.1145/3770855.3818074](https://doi.org/10.1145/3770855.3818074)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Sections 2 and 5.
- Method: Extraction agents share a memory holding corpus-wide context, so they can resolve conflicts and keep the graph connected instead of treating each fragment separately.
- Evidence: Motivating test: removing 40% of low-frequency triples slightly raised accuracy (65.28% vs 64.85%). GraphRAG-Bench complex reasoning: recall 90.42, relevance 82.64. Abstract claims efficiency comparable to baselines; I did not extract cost numbers. [Source](https://arxiv.org/abs/2606.00610)
- Relevance: High. Directly addresses cross-chunk consistency. Whether a shared memory stays workable at 1.9 million chunks is untested.

**A6. RAGA: Reading-And-Graph-building-Agent for Autonomous Knowledge Graph Construction and Retrieval-Augmented Generation**
- Authors: Chengrui Han, Zesheng Cheng
- First posted: 16 May 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2605.17072](https://arxiv.org/abs/2605.17072)
- What I read: Abstract only
- Method: ReAct agent with create, read, update and delete tools over the graph. It searches the existing graph and verifies against source before writing, and links each entry to its source text.
- Evidence: Preliminary only: a subset of QASPER; beats zero-shot baselines. No large-scale cost figures in the abstract. [Source](https://arxiv.org/abs/2605.17072)
- Relevance: Medium. Targets cross-chunk relations and disambiguation, but a per-chunk agent loop looks expensive at this scale (my inference).

**A7. RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM**
- Authors: Mikhail Komarov, Ivan Bondarenko, Stanislav Shtuka, Oleg Sedukhin, Roman Shuvalov, Yana Dementyeva, Matvey Solovyov, Nikolay O. Nikitin
- First posted: 13 Jul 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2607.11683](https://arxiv.org/abs/2607.11683)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): sections on results and cost.
- Method: Two-stage typed extraction, DBSCAN deduplication, LLM summarisation and Leiden communities, driven by a compact 7B extractor (Meno-Lite-0.1).
- Evidence: About 8k tokens per document at about 2k tokens/s; about $0.001/doc on rented GPU against about $0.10/doc for commercial APIs, so about $100 vs $10,000 per 100k documents (the authors' estimate). Outperforms Qwen2.5-32B on construction (+12.5% relative harmonic mean). GraphRAG-Bench Medical evidence recall up to 0.84 vs at most 0.76. BioASQ answer correctness 72.9 vs HippoRAG2 72.4 once answer format is controlled. [Source](https://arxiv.org/abs/2607.11683)
- Relevance: High. Closest match to the budget question: separates extraction from consolidation and reports cost per document with a small model. Its extraction uses entity and relation types, so check how open it really is.

**A8. From Extraction to Governed Memory: Multi-Agent Knowledge Graph Construction with Domain-Expert Review**
- Authors: Pranav Bykampadi, Neel Mokaria, Vishesh Narayan, Faizan Wajid, Ashok Agrawala
- First posted: 12 Aug 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2608.28642](https://arxiv.org/abs/2608.28642)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Table 2, cost appendix text (not its numbers), QA results.
- Method: A classifier induces entity and relation types from content, then candidate triples are assigned to domain-owner agents that check evidence and approve, reject or revise them, with an audit log.
- Evidence: SciERC strict triple F1: flat insertion 0.106, global reviewer 0.143, MAGG 0.156. Mapped F1: 0.192, 0.271, 0.290. Admitted triples fall from 2,881 to 1,077. Blinded review of 120 triples favours governed triples. MuSiQue: 9.0 EM and 11.2 F1 points above Microsoft GraphRAG. Costs more LLM calls than flat insertion; all runs on GPT-5. [Source](https://arxiv.org/abs/2608.28642)
- Relevance: High on quality control, but note how low the absolute strict F1 is even after governance.

**A9. Better Later Than Sooner: Neuro-Symbolic Knowledge Graph Construction via Ontology-grounded Post-extraction Correction**
- Authors: Lorenzo Loconte, Timothy Hospedales, Cristina Cornelio
- First posted: 27 May 2026 (arXiv v1). Published: October 2026 (ACL Anthology). Venue: Findings of EMNLP 2026.
- Links: [ACL Anthology 2026.findings-emnlp.1867](https://aclanthology.org/2026.findings-emnlp.1867/); [arXiv:2605.29168](https://arxiv.org/abs/2605.29168)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Tables 1-3.
- Method: Open-domain extraction first, then embedding-based mapping of types and predicates, then LLM correction only where ontology constraints are violated.
- Evidence: HotpotQA (Qwen3-30B): triples 127,108, ontology consistency 96.8% for OAK+MEND vs 63.4% before correction. Token cost 59% (Qwen3) and 79% (GPT-OSS-120B) of Wikontic. QA EM/F1 on HotpotQA with GPT-OSS-120B: vector RAG 58.6/72.9, HippoRAGv2 38.8/57.0, KGGen 38.9/48.5, Wikontic 40.4/50.4, OAK 46.4/57.8. With Qwen3: vector RAG 42.9/56.5, HippoRAGv2 50.0/64.1, OAK 47.0/58.6. [Source](https://aclanthology.org/2026.findings-emnlp.1867/)
- Relevance: High, and its QA table is important: plain vector retrieval beat every graph with the stronger model on HotpotQA. Ontology-grounded, so it borders the other researcher's scope. Post-hoc correction ordering carries over to a no-ontology setting only partly.

**A10. DiffIE: Diffusion-based Open Information Extraction**
- Authors: Konstantin Fedorov, Valentin Malykh
- First posted: 2 Sep 2026 (arXiv v1). Published: October 2026 (ACL Anthology). Venue: EMNLP 2026 (main).
- Links: [ACL Anthology 2026.emnlp-main.1851](https://aclanthology.org/2026.emnlp-main.1851/); [arXiv:2609.02315](https://arxiv.org/abs/2609.02315)
- What I read: Abstract only
- Method: Non-LLM OpenIE: conditional discrete diffusion over per-token role tags yields a pool of candidates, which are clustered and ranked.
- Evidence: Reports new best CaRB (1-1) F1 and AUC, beats ClausIE on BenchIE, best average over CaRB, CaRB (1-1), BenchIE and WiRe57. Exact scores not in the abstract. [Source](https://aclanthology.org/2026.emnlp-main.1851/)
- Relevance: Medium. Rare recent non-LLM OpenIE result, but sentence level and English benchmarks.

**A11. ReaORE: Reasoning-Guided Progressive Open Relation Extraction Empowered by Large Reasoning Models**
- Authors: Xin Lin, Liang Zhang, Guoqi Ma, Hongyao Tu, Jinsong Su
- First posted: 25 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.26986](https://arxiv.org/abs/2606.26986)
- What I read: Abstract only
- Method: Open relation extraction in two steps: reason to filter a candidate relation set, then compare candidates to choose the label.
- Evidence: Beats baselines on two OpenRE datasets; no numbers in the abstract. [Source](https://arxiv.org/abs/2606.26986)
- Relevance: Low to medium. Assumes entity pairs are given.

**A12. F-GraphRAG: Preserving Fine-Grained Facts in Graph Retrieval-Augmented Generation**
- Authors: Shenmeng Chen, Kang Zhang
- First posted: Not established. Published: October 2026 (ACL Anthology). Venue: Findings of EMNLP 2026.
- Date note: First posted: no arXiv version found by title search.
- Links: [ACL Anthology 2026.findings-emnlp.2253](https://aclanthology.org/2026.findings-emnlp.2253/)
- What I read: Abstract only
- Method: Extractor designed to keep precise quantities, dates and attributes, with a filter that resists favouring only globally important facts. Adds stage-wise loss metrics.
- Evidence: +12.6% accuracy over GraphRAG with 33.8% fewer relationships on a novels-based suite; +5.3-11.2% across nine public benchmarks. [Source](https://aclanthology.org/2026.findings-emnlp.2253/)
- Relevance: High. Evaluation reports are full of figures and dates, which is the failure this paper targets.

**A13. CycleIE: Robust Document Information Extraction through Iterative Verification and Refinement**
- Authors: Zhengxuan Zhang, Yin WU, Zhuowen Liang, Haixun Wang, Yuyu Luo, Nan Tang
- First posted: Not established. Published: October 2026 (ACL Anthology). Venue: Findings of EMNLP 2026.
- Date note: First posted: no arXiv version found by title search.
- Links: [ACL Anthology 2026.findings-emnlp.418](https://aclanthology.org/2026.findings-emnlp.418/)
- What I read: Abstract only
- Method: Iterative multi-agent extraction (ReAct plus Monte Carlo Tree Search) that retrieves, structures, extracts and repairs under verification.
- Evidence: >10% relative improvement over baselines on document QA benchmarks. [Source](https://aclanthology.org/2026.findings-emnlp.418/)
- Relevance: Low at this scale because of the search loop (my inference); useful as a targeted repair idea.

**A14. AutoPKG: An Automated Framework for Dynamic E-commerce Product-Attribute Knowledge Graph Construction**
- Authors: Pollawat Hongwimol, Haoning Shang, Chutong Wang, Zhichao Wan, Yi Gao, Yuanming Li, Lin Gui, Wenhao Sun, Cheng Yu
- First posted: 18 April 2026 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: Findings of ACL 2026.
- Date note: First posted 18 April 2026, just before the window.
- Links: [ACL Anthology 2026.findings-acl.766](https://aclanthology.org/2026.findings-acl.766/); [arXiv:2604.16950](https://arxiv.org/abs/2604.16950)
- What I read: Abstract only
- Method: Multi-agent system that induces product types and attribute keys on demand; a central decision agent keeps one canonical graph.
- Evidence: Lazada catalogue: 0.953 WKE for types, 0.724 for keys, 0.531 edge F1. +0.152 edge exact-match F1 on three public benchmarks. [Source](https://aclanthology.org/2026.findings-acl.766/)
- Relevance: Medium. Different domain, but a working example of central canonicalisation with an evaluation protocol for it.

**A15. Format-Constraint Coupling in Knowledge Graph Construction from Statistical Tables**
- Authors: Jingxuan Qi, Zhiqiang Ye, Yuxiang Feng
- First posted: 21 May 2026 (arXiv v1). Published: None found. Venue: Preprint (comment says submitted to EMNLP 2026 via ARR).
- Links: [arXiv:2605.21974](https://arxiv.org/abs/2605.21974)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): abstract and contributions only.
- Method: Factorial study of serialisation format against schema constraints when building graphs from statistical CSV tables.
- Evidence: Schema-format interaction up to +1.180 (2x2, 6 datasets). A mismatched schema drops fact coverage below the unconstrained baseline on 4 of 6 datasets. Three standard retrieval modes hide construction differences (at most 1 point), while direct graph access shows gaps up to +47.6 points. CSVFidelity-Bench has 1,892 gold facts. [Source](https://arxiv.org/abs/2605.21974)
- Relevance: High. Speaks to both tables in reports and to whether end-to-end QA can be trusted to judge graph quality.

**A16. Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline**
- Authors: Kirill Solovev, Jana Lasser
- First posted: 25 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.27347](https://arxiv.org/abs/2606.27347)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): introduction, pipeline and evaluation passages.
- Method: Open-weight multilingual pipeline: span NER, a linking cascade to Wikidata IDs (37 million aliases in 36 languages), then ontology-constrained relation extraction with guided decoding.
- Evidence: Graphs from 300,000 Austrian and 500,000 Polish articles. 3,491-relation gold check: 68.2% strict to 93.7% lenient textual correctness. 63.5% LLM-adjudicated semantic F1 on 74 German development articles with a 397B model. [Source](https://arxiv.org/abs/2606.27347)
- Relevance: High for multilingual entity identity at scale. Relations use a fixed ontology, so not schema-free at that stage.

**A17. Faithful and Complete LLM-Assisted Automated Construction of Hierarchical Hardware Knowledge Graphs**
- Authors: Haotian Qiao, Robert P. Dick
- First posted: 8 Oct 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2610.12762](https://arxiv.org/abs/2610.12762)
- What I read: Abstract only
- Method: Chunk-based property extraction, hierarchical routing and iterative refinement with provenance, on datasheets dense with tables and figures.
- Evidence: 20 datasheets: 98.8% property faithfulness vs 76.1% for one-shot. Fully covers 83.6% of the baseline's faithful properties; the baseline fully covers only 10.5% of the pipeline's. [Source](https://arxiv.org/abs/2610.12762)
- Relevance: Medium. Evidence that chunked extraction beats one-shot on long table-heavy documents, on a small sample.

**A18. Every Edge Needs a Quote: Building a Quote-Grounded Knowledge Graph of Healthcare Software Adoption with Language-Model Agents**
- Authors: Juan Felipe Duran, Arthur Tobler, Martin Vallières
- First posted: 23 Oct 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2610.32781](https://arxiv.org/abs/2610.32781)
- What I read: Abstract only
- Method: Each edge must carry a verbatim source quote. A permissive extractor proposes; deterministic checks and a five-call vote on nine criteria decide.
- Evidence: 61 vendors: 0.94 precision, 0.71 recall against a page-level key over 6,428 candidates, whose verdicts were themselves model-drafted. [Source](https://arxiv.org/abs/2610.32781)
- Relevance: High as a hallucination control. The quote check is cheap, the five-call vote is not.

**A19. FineREX: Fine-Tuned NER-RE for Human Smuggling Knowledge Graphs**
- Authors: Elijah Feldman, Dipak Meher, Carlotta Domeniconi
- First posted: 18 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.19710](https://arxiv.org/abs/2606.19710)
- What I read: Abstract only
- Method: Replaces coreference rewriting with a fine-tuned NER-RE model for court documents. Schema is domain-defined.
- Evidence: 512 annotated chunks: +15.50 entity F1 and +31.46 relation F1 vs a larger general model. Node duplication on long documents 17.78% to 11.17%. Processing time down 50%. [Source](https://arxiv.org/abs/2606.19710)
- Relevance: Medium. Shows a small annotated chunk set can pay off, and that fine-tuning can reduce duplication.

**A20. DeepRefine: Agentic Knowledge Refinement via Reinforcement Learning**
- Authors: Haoyu Huang, Jiaxin Bai, Shujie Liu, Yang Wei, Huihao Jing, Hong Ting Tsang, Yisen Gao, Zhongwei Xie, Yufei Li, Yangqiu Song
- First posted: 11 May 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2605.10488](https://arxiv.org/abs/2605.10488)
- What I read: Abstract only
- Method: Reinforcement-learned agent that diagnoses defects (missing links, weak claims, coreference issues) from query interactions and edits an existing knowledge base.
- Evidence: Consistent downstream gains; no numbers in the abstract. [Source](https://arxiv.org/abs/2605.10488)
- Relevance: Medium. Post-hoc repair, needs a query stream.

**A21. AgentsKG: A Hierarchical Multi-Agent Framework for Open-Domain Knowledge Graph Construction**
- Authors: Shilong Liu, Yongqiang Liu, Jiye Liu, Xuan Guo, Tianpeng Li, Qiyao Peng, Wenjun Wang
- First posted: Not established. Published: 8 August 2026 (Crossref). Venue: KDD 2026.
- Date note: First posted: no arXiv version found by title search.
- Links: [DOI 10.1145/3770855.3818058](https://doi.org/10.1145/3770855.3818058)
- What I read: Title and metadata only (NOT READ; publisher page returned 403, Crossref holds no abstract). User can obtain it from the DOI.
- Method: Unknown beyond the title (hierarchical multi-agent, open-domain).
- Evidence: Not read. [Source](https://doi.org/10.1145/3770855.3818058)
- Relevance: Title suggests high relevance. Worth obtaining.

**A22. Less Is More: Knowledge Graph Construction and Schema-Guided Canonicalization with LLMs**
- Authors: Hwangseon Park, Hayeon Yang, Dongyoung Son, Myoungseok Song, Hayoung Oh
- First posted: Not established. Published: 6 November 2026 (Crossref; one day after this note). Venue: CIKM 2026.
- Date note: First posted: no arXiv version found by title search.
- Links: [DOI 10.1145/3799682.3839949](https://doi.org/10.1145/3799682.3839949)
- What I read: Title and metadata only (NOT READ; publisher page returned 403, Crossref holds no abstract). User can obtain it from the DOI.
- Method: Unknown beyond the title (construction plus schema-guided canonicalisation).
- Evidence: Not read. [Source](https://doi.org/10.1145/3799682.3839949)
- Relevance: Title suggests high relevance. Worth obtaining.

**A23. Efficient LLM Adaptation for Opinion Knowledge Graph Construction: Lessons from the Telecom Industry**
- Authors: Nai-Chi Yang, Yu-Ming Hsieh, Wei-Yun Ma, Kuo-Wei Chang
- First posted: Not established. Published: 19 July 2026 (Crossref). Venue: SIGIR 2026.
- Links: [DOI 10.1145/3805712.3808450](https://doi.org/10.1145/3805712.3808450)
- What I read: Title and metadata only (NOT READ; publisher page returned 403, Crossref holds no abstract). User can obtain it from the DOI.
- Method: Unknown beyond the title (telecom opinion graphs).
- Evidence: Not read. [Source](https://doi.org/10.1145/3805712.3808450)
- Relevance: Unknown. Possibly useful as an industry cost report.

#### B. Cross-chunk links, coreference, entity resolution, canonicalisation

**B1. Beyond Chunk-Local Extraction: Cross-Chunk Graph Augmentation for GraphRAG**
- Authors: Jiaming Zhang, Yibo Zhao, Jing Yu, Jianxiang Yu, Xiang Li
- First posted: 27 May 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2605.28004](https://arxiv.org/abs/2605.28004)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Tables 1-4, limitations.
- Method: Offline step: a graph neural network scores subgraphs for missing cross-chunk relations using self-supervised corruption, and the LLM is called only on high-scoring regions, given evidence passages.
- Evidence: Base indexes of 6,119-11,656 chunks and 64,523-123,308 triples gain 5,048-8,901 triples. HippoRAG2 F1 +1.33 on MuSiQue and +1.09 on LiteraryQA. Random selection under the same budget loses up to 1.33 F1. LLM-judged correctness of added triples 84.35% (human-LLM kappa 0.765). LinearRAG with no LLM extraction had the best 2Wiki F1 (65.16). [Source](https://arxiv.org/abs/2605.28004)
- Relevance: High. The most direct attack on the cross-chunk gap with a bounded LLM budget. Gains are small, and most of the index is still chunk-local.

**B2. ImCoref-CeS: An Improved Lightweight Pipeline for Coreference Resolution with LLM-based Checker-Splitter Refinement**
- Authors: Kangyang Luo, Yuzhuo Bai, Shuzheng Si, Cheng Gao, Zhitong Wang, Yingli Shen, Wenhao Li, Zhu Liu, Yufeng Han, Jiayi Wu, Cunliang Kong, Maosong Sun
- First posted: 11 October 2025 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: ACL 2026 (long).
- Date note: First posted before the window.
- Links: [ACL Anthology 2026.acl-long.1122](https://aclanthology.org/2026.acl-long.1122/); [arXiv:2510.10241](https://arxiv.org/abs/2510.10241)
- What I read: Abstract only
- Method: Supervised coreference model improved for long text, with an LLM acting only as checker and splitter of candidate mentions and clusters.
- Evidence: Claims better than state of the art; numbers not in the abstract. [Source](https://aclanthology.org/2026.acl-long.1122/)
- Relevance: High as a pattern: cheap model first, LLM to check. Test data is not evaluation reports.

**B3. MEIC-DT: Memory-Efficient Incremental Clustering for Long-Text Coreference Resolution with Dual-Threshold Constraints**
- Authors: Kangyang Luo, Shuzheng Si, Yuzhuo Bai, Cheng Gao, Zhitong Wang, Cheng Huang, Yingli Shen, Yufeng Han, Wenhao Li, Cunliang Kong, Maosong Sun
- First posted: 31 December 2025 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: Findings of ACL 2026.
- Date note: First posted before the window.
- Links: [ACL Anthology 2026.findings-acl.881](https://aclanthology.org/2026.findings-acl.881/); [arXiv:2512.24711](https://arxiv.org/abs/2512.24711)
- What I read: Abstract only
- Method: Non-LLM incremental clustering for long-text coreference under a set memory budget, using eviction and cluster condensing rules.
- Evidence: Competitive performance under tight memory; numbers not in the abstract. [Source](https://aclanthology.org/2026.findings-acl.881/)
- Relevance: Medium to high. Non-LLM option for within-report coreference.

**B4. Curate Before You Connect: Identity and Ontology Tagging in a Production Knowledge Graph**
- Authors: Vaibhav Dangaich, Kevin Lewis, Kundeshwar Pundalik
- First posted: 11 Aug 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2608.10644](https://arxiv.org/abs/2608.10644)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): abstract, contributions, measurement provenance table.
- Method: Production ingestion layer. Identity is decided by a ladder over identifier columns, name columns and position; the graph write merges on exact canonical name; entity resolution only flags candidates and never merges.
- Evidence: 537,157 entities and 2,198,567 relationships from 98,795 government documents. One wrong merge of two surface forms corrupted a correct record and deleted eight entities from an unrelated document. Anchored evidence cut role assignments from 36 to 4, all correct, on an enriched sample. The authors state no identity benchmark is reported and over-merges under the key are undetectable. [Source](https://arxiv.org/abs/2608.10644)
- Relevance: High. Closest corpus type (government documents, tables) and honest about what is unmeasured. Treat as engineering experience, not as a validated method.

**B5. Keep Your Receipts: Trustworthy Cleanup of Agent-Built Knowledge Graphs**
- Authors: Vishisht Choudhary
- First posted: 13 Oct 2026 (arXiv v1). Published: Workshop, date not checked. Venue: NeurIPS 2026 workshop (arXiv comment).
- Links: [arXiv:2610.17719](https://arxiv.org/abs/2610.17719)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): abstract and Table 1.
- Method: The graph builder records, for each merge, edge and summary, which source bundles support it, so deleting a source becomes set arithmetic with no model calls.
- Evidence: Two builds of the same 300-document corpus differ in 29,475 artifacts, five times the 6,248 changed by removing 61 retracted documents. Deleting only directly extracted facts lowers contaminated queries from 32.2% to 28.0%. Merge receipts agree with re-derivation 68% and 85% of the time; summary receipts are wrong about two-thirds of the time. [Source](https://arxiv.org/abs/2610.17719)
- Relevance: High. The run-to-run instability figure is a warning for any reproducibility claim.

**B6. Better Retrieval, Limited Clustering Gains: A Controlled Study of Multilingual Company Entity Resolution**
- Authors: Yijiashun Qi, Yuxuan Li, Hanzhe Guo
- First posted: 4 Oct 2026 (arXiv v1). Published: Date not checked. Venue: ICEKIM 2026 (arXiv comment).
- Links: [arXiv:2610.05573](https://arxiv.org/abs/2610.05573)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): results sections.
- Method: Controlled study of multilingual company-name resolution: adapted E5 encoders, unchanged pair classifier, guarded Union-Find clustering.
- Evidence: 3,633 names with silver labels: pair recall 53.74% to 76.98%, but cluster recall only 32.54% to 33.33%. Most new positives fail the classifier threshold. Cross-script coverage 385/691 to 553/691. [Source](https://arxiv.org/abs/2610.05573)
- Relevance: High. Organisation names across languages are the user's exact problem; shows retriever gains may not reach the final entity groups. Labels are silver and independent gold is left for later.

**B7. Adaptive Graph Refinement and Label Propagation with LLMs for Cost-Effective Entity Resolution**
- Authors: Hongtao Wang, Renchi Yang, Haoran Zheng, Xiangyu Ke
- First posted: 25 May 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2605.25814](https://arxiv.org/abs/2605.25814)
- What I read: Abstract only
- Method: Merges matching and clustering into iterative label propagation on a changing graph, spending a fixed LLM query budget via a greedy rule.
- Evidence: Beats cascaded pipelines on eight benchmarks; no numbers in the abstract. [Source](https://arxiv.org/abs/2605.25814)
- Relevance: High in principle: budgeted LLM calls for deduplication. Benchmarks are record data, not extracted entity names.

**B8. Entity Resolution via Batched Oracle Queries**
- Authors: Lorenzo Balzotti, Donatella Firmani, Luca Gagliardelli, Giovanni Simonini
- First posted: 23 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.24407](https://arxiv.org/abs/2606.24407)
- What I read: Abstract only
- Method: Treats an LLM as an oracle that clusters a limited batch; picks batches to maximise recall per call. Proves optimal batching is NP-hard and gives an optimal solution under an entity-size condition.
- Evidence: Six datasets; beats baselines; no numbers in the abstract. [Source](https://arxiv.org/abs/2606.24407)
- Relevance: High. Matches the case where entity groups are far larger than one prompt.

**B9. Entity Resolution in Practice: Lessons from a Self-Serve Pipeline**
- Authors: Kaushik Pavani, Ganga Aluri, Pravin Jadhav, Neeraj Prasad, Kiran Sanka
- First posted: 28 Jul 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2607.26298](https://arxiv.org/abs/2607.26298)
- What I read: Abstract only
- Method: Self-serve resolution system with an automatic comparison of matching families, rule vetoes for precision and diverse candidate retrieval for recall.
- Evidence: Six benchmarks from 864 to 5M records. Main lesson: treating matches as transitive lets one false link chain hundreds of records, so cross-group merges must be re-checked. [Source](https://arxiv.org/abs/2607.26298)
- Relevance: High. Practitioner evidence at a comparable scale.

**B10. Same Entity, Different Script: Diagnosing Cross-Script Entity Matching Failures in Language Models**
- Authors: Athira Gopal, Patil Prathamesh Pravin, Ankush Kadu, Arpit Jain, Ashwanth Krishnan
- First posted: 22 Oct 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2610.31407](https://arxiv.org/abs/2610.31407)
- What I read: Abstract only
- Method: Tests name-identity judgement when the same person's name appears in different scripts.
- Evidence: Qwen2.5-7B same-person accuracy falls from 35.1% in Latin to 8.5% in Devanagari (Marathi prompts) and 5.3% in Malayalam, while different-person accuracy stays above 98%. 601 and 374 audited pairs. [Source](https://arxiv.org/abs/2610.31407)
- Relevance: Medium to high. Warns small models under-merge across scripts, relevant to Arabic or Cyrillic names (not tested here).

**B11. Entity Matching using Decision Models**
- Authors: Aaron Steiner, Ralph Peeters, Christian Bizer
- First posted: 25 Oct 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2610.34626](https://arxiv.org/abs/2610.34626)
- What I read: Abstract only
- Method: Zero-shot matching with models that return a probability per answer, rather than generating text.
- Evidence: Eight benchmarks: four strongest close to GPT-6 Luna mean F1, best slightly above, faster and mostly cheaper, well calibrated. Sending unsure pairs to an LLM ensemble adds 1.9-4.0 F1 points. [Source](https://arxiv.org/abs/2610.34626)
- Relevance: High. Cheap, calibrated pair decisions suit a very large candidate set.

**B12. Labeling Training Data for Entity Matching Using Large Language Models**
- Authors: Aaron Steiner, Christian Bizer
- First posted: 27 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.28823](https://arxiv.org/abs/2606.28823)
- What I read: Abstract only
- Method: An LLM teacher labels pairs to train a small BERT-class matcher.
- Evidence: Students perform similarly to benchmark-trained matchers, with 34-459 times less inference time than direct LLM matching. $5.86-$8.11 per training set at GPT-5.2 prices. Weaker on unseen products. Includes non-English data. [Source](https://arxiv.org/abs/2606.28823)
- Relevance: High. A concrete non-LLM-at-inference route, with a stated weakness on unseen entities.

**B13. CaRL-EM: Cost-Aware Reinforcement Learning for Entity Matching with LLMs**
- Authors: Chaohui Guo, Michel C. A. Klein, Zhisheng Huang
- First posted: 1 Sep 2026 (arXiv v1). Published: July 2026 (ACL Anthology). Venue: ACL 2026 (long).
- Date note: arXiv copy posted after the conference date.
- Links: [ACL Anthology 2026.acl-long.1258](https://aclanthology.org/2026.acl-long.1258/); [arXiv:2609.01195](https://arxiv.org/abs/2609.01195)
- What I read: Abstract only
- Method: Reinforcement-learned controller chooses which matching operator and model size to use per record.
- Evidence: 7 benchmarks; better quality-cost trade-off than baselines; numbers not read. [Source](https://aclanthology.org/2026.acl-long.1258/)
- Relevance: Medium.

**B14. LLM-CER: An Interactive System for In-Context Clustering-Based Entity Resolution with Large Language Models**
- Authors: Haoyu Wang, Haitong Tang, Jiajie Fu, Arijit Khan, Sharad Mehrotra, Xiangyu Ke, Yunjun Gao
- First posted: Not established. Published: 1 August 2026 (Crossref). Venue: PVLDB (system paper).
- Links: [DOI 10.14778/3827998.3828112](https://doi.org/10.14778/3827998.3828112)
- What I read: First part of the abstract via Crossref only.
- Method: Interactive in-context clustering for entity resolution with LLMs, rather than pair matching.
- Evidence: Not read. [Source](https://doi.org/10.14778/3827998.3828112)
- Relevance: Medium. Follow up for its clustering approach.

**B15. Generalized Entity Matching with Adaptivity via Large Language Models**
- Authors: Xingguang Chen, Yimin Shi, Xiaokui Xiao
- First posted: Not established. Published: 18 May 2026 (Crossref). Venue: Proceedings of the ACM on Management of Data.
- Links: [DOI 10.1145/3802066](https://doi.org/10.1145/3802066)
- What I read: First part of the abstract via Crossref only.
- Method: Entity matching across structured, semi-structured and unstructured sources with LLMs.
- Evidence: Not read. [Source](https://doi.org/10.1145/3802066)
- Relevance: Medium to high: covers matching extracted text mentions to records.

**B16. Self-Evolving Ontology via Cumulative Definition Enrichment for Entity Canonicalization**
- Authors: Seungtaek Oh, Ilhwan Kim, Jaewon Moon
- First posted: Not established. Published: 6 November 2026 (Crossref). Venue: CIKM 2026.
- Date note: First posted: no arXiv version found by title search.
- Links: [DOI 10.1145/3799682.3840115](https://doi.org/10.1145/3799682.3840115)
- What I read: Title and metadata only (NOT READ; publisher page returned 403, Crossref holds no abstract). User can obtain it from the DOI.
- Method: Unknown beyond the title.
- Evidence: Not read. [Source](https://doi.org/10.1145/3799682.3840115)
- Relevance: Title suggests high relevance. Worth obtaining.

**B17. Reliability of LLM Judges for Evaluating Entity Alignment**
- Authors: Vaibhava Lakshmi Ravideshik, Mayank Kejriwal
- First posted: 7 Oct 2026 (arXiv v1). Published: October 2026 (ACL Anthology). Venue: EMNLP 2026 (main).
- Links: [ACL Anthology 2026.emnlp-main.1717](https://aclanthology.org/2026.emnlp-main.1717/); [arXiv:2610.09554](https://arxiv.org/abs/2610.09554)
- What I read: Abstract only
- Method: Benchmarks three frontier models as judges of four alignment systems on three datasets.
- Evidence: Showing the system's label collapses judge discrimination (J-ROC-AUC 0.12-0.87); a label-free prompt recovers 0.93-1.00 on distinctive names and 0.93-0.95 on biomedical pairs. [Source](https://aclanthology.org/2026.emnlp-main.1717/)
- Relevance: High for evaluation design: never show the judge the proposed merge decision.

**B18. Can we trust LLM Self-Explanations for Entity Resolution?**
- Authors: Tommaso Teofili, Donatella Firmani, Nick Koudas, Paolo Merialdo, Divesh Srivastava
- First posted: 31 May 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.01210](https://arxiv.org/abs/2606.01210)
- What I read: Abstract only
- Method: Large evaluation of LLM explanations for matches, across three LLMs and ten datasets.
- Evidence: Explanations are often unstable and weakly faithful. [Source](https://arxiv.org/abs/2606.01210)
- Relevance: Medium. Do not use LLM rationales as evidence for a merge.

**B19. AgenticER: the next frontier in Entity Resolution**
- Authors: George Papadakis, Panos Korovesis, Manolis Koubarakis, Themis Palpanas
- First posted: 29 Jul 2026 (arXiv v1). Published: None found. Venue: Vision paper (comment: submitted to VLDB).
- Links: [arXiv:2607.27435](https://arxiv.org/abs/2607.27435)
- What I read: Abstract only
- Method: Argues for resolution as a sequential decision process with evidence gathering and cost control.
- Evidence: No experiments. [Source](https://arxiv.org/abs/2607.27435)
- Relevance: Low. Framing only.

#### C. Quality against gold, hallucination, small and non-LLM models

**C1. Ranked by the Matcher: A Reproducibility Audit of Knowledge Graph Extraction from Threat Reports**
- Authors: Safayat Bin Hakim, Houbing Herbert Song
- First posted: 1 Sep 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2609.01671](https://arxiv.org/abs/2609.01671)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): abstract and RQ sections.
- Method: Reproducibility audit of graph-extraction tools for threat reports, focused on how predicted triples are matched to gold.
- Evidence: Matching rule reproducible for 5 of 12 systems. Eight protocols reverse 11 of 45 pairwise orderings. One fixed prediction set spans 0.16-0.70 F1. No mechanical matcher agrees with human reviewers more than 7 in 10 times; an LLM judge agrees far more. 149 documents, 1,824 gold triples. [Source](https://arxiv.org/abs/2609.01671)
- Relevance: High. Most F1 comparisons in this note should be read in this light.

**C2. From Unreliable to Usable: Decomposing Failure Modes and Confidence Signals in LLM Entity Extraction - Evidence from Cyber Threat Intelligence**
- Authors: Jiahong Qian
- First posted: 14 Oct 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2610.20417](https://arxiv.org/abs/2610.20417)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): abstract and failure analysis.
- Method: Preregistered audit of whole-document entity extraction by two low-cost models and one flagship.
- Evidence: 354 campaigns, 630 evaluations. Micro precision 0.03-0.34 across tiers; calibration error 0.53-0.94. False positives back-project to 65.6% background mentions, 19.9% gold-scope gaps, 1.2% unsupported. Second annotators were LLMs (kappa 0.79). [Source](https://arxiv.org/abs/2610.20417)
- Relevance: High. In this domain the main failure was correct strings in the wrong context, not invented ones. Reports citing past programmes and other countries could show the same; hypothesis, not tested.

**C3. KGCQual: An Interpretable Framework for Evaluating the Knowledge Graph Construction Quality from Text**
- Authors: Nipun Misra, Vikranth Udandarao, Aanchal Gupta, Yogender Kumar, Manuj Mukherjee, Raghava Mutharaju
- First posted: 11 Jul 2026 (arXiv v1). Published: None found. Venue: Preprint (under review per comment).
- Links: [arXiv:2607.10212](https://arxiv.org/abs/2607.10212)
- What I read: Abstract only
- Method: Gold-free intrinsic quality score using entity completeness, resolution and connectivity plus predicate preservation via dependency parses and negation handling.
- Evidence: Tested on WebNLG, TinyButMighty and BenchIE; scores correlate with link prediction. Numbers not in the abstract. [Source](https://arxiv.org/abs/2607.10212)
- Relevance: High. A quality gauge that needs no gold data.

**C4. When Does Consistency Pay Off? Relational Constraints for LLM-based Document-Level Relation Extraction under Incomplete Annotations**
- Authors: Mingxuan Sun
- First posted: 27 Jul 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2607.24312](https://arxiv.org/abs/2607.24312)
- What I read: Abstract only
- Method: Defines a measure of how closed gold annotations are under repair rules (Delta-closure) and tests constraint-based repair for document-level RE.
- Evidence: Rescoring identical predictions on Re-DocRED reverses the ranking. On DWIE, deterministic closure beats reflection at 3.6 times lower cost. A +0.025 RL gain vanishes after rescoring. [Source](https://arxiv.org/abs/2607.24312)
- Relevance: Medium to high. Incomplete gold can make good repairs look bad.

**C5. Sub-Billion, Super-Frontier: Small Language Models Rival Zero-Shot Frontier LLMs on General and Literary Relation Extraction**
- Authors: Despina Christou, Grigorios Tsoumakas
- First posted: 21 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.22606](https://arxiv.org/abs/2606.22606)
- What I read: Abstract only
- Method: Compares five small models (360M-3B), tuned, with zero-shot frontier LLMs and a RoBERTa baseline on relation extraction across nine benchmarks.
- Evidence: Qwen2.5-0.5B fine-tuned: micro-F1 0.83, vs GPT-5.4 0.69 and Claude Sonnet 4.6 0.66 zero-shot. RoBERTa also beats both frontier models. Literary: 0.92 vs 0.83. Fixed relation sets. [Source](https://arxiv.org/abs/2606.22606)
- Relevance: High for the small-model question, with the caveat that tuning needs labels and relation types.

**C6. LLMs Underperform Graph-Based Parsers on Supervised Relation Extraction for Complex Graphs**
- Authors: Paolo Gajo, Domenic Rosati, Hassan Sajjad, Alberto Barrón-Cedeño
- First posted: 9 April 2026 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: ACL 2026 (short).
- Date note: First posted 9 April 2026, just before the window.
- Links: [ACL Anthology 2026.acl-short.17](https://aclanthology.org/2026.acl-short.17/); [arXiv:2604.08752](https://arxiv.org/abs/2604.08752)
- What I read: Abstract only
- Method: Four LLMs against a graph-based parser on six supervised RE datasets of varying graph complexity.
- Evidence: The parser pulls further ahead as relations per document increase. Scores not in the abstract. [Source](https://aclanthology.org/2026.acl-short.17/)
- Relevance: Medium to high. Supervised setting. Suggests dense passages are where LLMs lose ground.

**C7. Assessment of Generative Named Entity Recognition in the Era of Large Language Models**
- Authors: Qi Zhan, Yile Wang, Hui Huang
- First posted: 25 January 2026 (arXiv API, v1). Published: October 2026 (ACL Anthology). Venue: Findings of EMNLP 2026.
- Date note: First posted before the window.
- Links: [ACL Anthology 2026.findings-emnlp.1706](https://aclanthology.org/2026.findings-emnlp.1706/); [arXiv:2601.17898](https://arxiv.org/abs/2601.17898)
- What I read: Abstract only
- Method: Eight open LLMs on four standard NER datasets, comparing output formats and memorisation.
- Evidence: With light fine-tuning and bracketed or XML output, competitive with encoder models. Scores not in the abstract. [Source](https://aclanthology.org/2026.findings-emnlp.1706/)
- Relevance: Medium.

**C8. Energy-Aware Benchmarking of Generative Information Extraction**
- Authors: Andrei Politov, Iryna Vergunova, Rene Jäkel, Michael Färber
- First posted: Not established. Published: October 2026 (ACL Anthology). Venue: Findings of EMNLP 2026.
- Date note: First posted: no arXiv version found by title search.
- Links: [ACL Anthology 2026.findings-emnlp.2002](https://aclanthology.org/2026.findings-emnlp.2002/)
- What I read: Abstract only
- Method: Benchmark for energy per true positive and F1 per kilojoule across decoding and serving settings.
- Evidence: Efficiency depends on interactions between settings, not model size alone. Numbers not in the abstract. [Source](https://aclanthology.org/2026.findings-emnlp.2002/)
- Relevance: Medium for cost planning on owned GPUs.

**C9. The Right Information Extraction Pipeline Depends on the Document: Accuracy-Energy Trade-offs for Small, Local Models**
- Authors: Christoph Walser, Mauricio Fadel Argerich, Jonathan Fürst
- First posted: 25 Sep 2026 (arXiv v1). Published: Date not checked. Venue: DocInsights at EMNLP 2026 (workshop, per comment).
- Links: [arXiv:2609.31341](https://arxiv.org/abs/2609.31341)
- What I read: Abstract only
- Method: Accuracy and energy for models up to 8B parameters, using page images or parsed text.
- Evidence: Batching cuts energy per page 38-85% at no accuracy cost. Neural OCR uses 17 times the energy of classical OCR. Vision models win on layout-rich forms, small text models with a parser on near-plain text. [Source](https://arxiv.org/abs/2609.31341)
- Relevance: Medium to high. Reports are mixed, so routing by page type is worth considering (my inference).

**C10. Trust the Edge, Doubt the Arrow: Auditing Validation Signals for Text-Mined Clinical Causal Knowledge Graphs**
- Authors: Su Lan, Shicheng Zhou, Yiqi Wang, Yanming Zhu, Xuefei Yin, Alan Wee-Chung Liew
- First posted: 17 Oct 2026 (arXiv v1). Published: Date not checked. Venue: PRICAI 2026 (per comment).
- Links: [arXiv:2610.24390](https://arxiv.org/abs/2610.24390)
- What I read: Abstract only
- Method: Audits validation signals on a graph from 328,839 discharge summaries using LLM extraction and UMLS normalisation.
- Evidence: Literature corroboration covers only 30.4% of candidate relations. Evidence-unsupported answers fall from 17.3% to 6.1% with conservative demotion. [Source](https://arxiv.org/abs/2610.24390)
- Relevance: Medium. Rare corpus-scale audit of relation direction.

#### D. Do schema-free graphs beat retrieval or schema-guided graphs?

**D1. WildGraphBench: Benchmarking GraphRAG with Wild-Source Corpora**
- Authors: Pengyu Wang, Benfeng Xu, Licheng Zhang, Shaohan Wang, Mingxuan Du, Chiwei Zhu, Zhendong Mao
- First posted: 2 Feb 2026 (arXiv v1). Published: July 2026 (ACL Anthology). Venue: Findings of ACL 2026.
- Date note: First posted before the window.
- Links: [ACL Anthology 2026.findings-acl.679](https://aclanthology.org/2026.findings-acl.679/); [arXiv:2602.02053](https://arxiv.org/abs/2602.02053)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Tables 2 and 6 and analysis.
- Method: Benchmark using Wikipedia's long, mixed external references as the corpus and cited statements as truth. Abstract says 1,100 questions; body says 1,197 (internal mismatch).
- Evidence: Average accuracy: HippoRAG2 64.33, NaiveRAG 59.79, LightRAG 56.76, MS GraphRAG global 54.54, LinearRAG 44.87, MS GraphRAG local 38.23, BM25 36.83. Multi-fact: MS GraphRAG global 47.64 vs NaiveRAG 35.08. Summary F1: NaiveRAG 15.84, best of all. Indexing seconds for flat methods, 28 minutes to 21.5 hours for graphs; tokens 0 (LinearRAG) to 90M (LightRAG) on one domain. [Source](https://aclanthology.org/2026.findings-acl.679/)
- Relevance: Very high. The closest benchmark to the user's long, heterogeneous documents, and it shows graphs helping only on aggregation.

**D2. Dissecting GraphRAG: A Modular Analysis of Knowledge Structuring for Factoid Question Answering**
- Authors: Noriki Nishida, Rumana Ferdous Munne, Shanshan Liu, Narumi Tokunaga, Yuki Yamagata, Fei Cheng, Kouji Kozaki, Yuji Matsumoto
- First posted: Not established. Published: 2026, month not stated (Crossref DOI record created 6 May 2026). Venue: TACL 2026.
- Date note: First posted: no arXiv version found by title search.
- Links: [ACL Anthology 2026.tacl-1.29](https://aclanthology.org/2026.tacl-1.29/)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Tables 3-5 and analysis.
- Method: Varies triple extraction, clustering and report generation one at a time on two factoid QA sets, with gold triples as an upper bound.
- Evidence: Default GraphRAG slightly under document RAG. Answer recall on CDR-QA: gold triples 11.7, tuned small model 7.7, GPT-4o-mini few-shot 4.8. DocRED-QA: 18.3, 18.2, 14.5. Relation precision 66.0 vs 38.2 and 66.5 vs 10.8 for small model vs LLM. Template reports beat LLM summaries and take 0.005 vs 46.129 minutes. [Source](https://aclanthology.org/2026.tacl-1.29/)
- Relevance: Very high. Measures that extraction precision limits the result, that a tuned small model can beat few-shot GPT-4o-mini, and that community summaries can be skipped.

**D3. Structure Over Scale: Schema-Constrained Causal Graphs for RAG**
- Authors: Marc Saouda, Rajprakash Bale, Eren Aldis, Cloves Almeida
- First posted: 10 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2607.22592](https://arxiv.org/abs/2607.22592)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): Tables 1, 2 and 15.
- Method: Replaces open extraction with an automatically induced, fixed vocabulary of causal variables in a compact two-tier graph.
- Evidence: Medical benchmark, 2,062 questions. Answer correctness: NaiveRAG .608, MS-GraphRAG local .654, LightRAG .673, HCG-RAG M .679, Fast-GraphRAG .690. Build LLM calls: MS-GraphRAG 17,638, LightRAG 4,423, Fast-GraphRAG 2,882, HCG-RAG M 1,167, XS 131. Query prompt tokens: NaiveRAG 7,500, MS-GraphRAG global 2,339,848. EpilepsyQA: HCG tiers .604-.611 vs NaiveRAG .397 and LightRAG .334. [Source](https://arxiv.org/abs/2607.22592)
- Relevance: Very high. Best single cost table found. Its schema is induced rather than supplied, so it is a middle option for the user.

**D4. When Is Graph Structure Worth Its Cost? The Case for Structure Pricing in Retrieval-Augmented Generation**
- Authors: Yuzhong Zhang, Haoyang Ma, Chao Peng, Lionel Briand, Boxi Yu, Jialun Cao
- First posted: 16 Sep 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2609.18099](https://arxiv.org/abs/2609.18099)
- What I read: Abstract only
- Method: Uses the graph only to find passages and answers from original text; a non-LLM filter skips low-salience chunks.
- Evidence: UltraDomain, 120 questions: preferred over LightRAG-hybrid on 93, LightRAG on 7, 20 split. Cost $0.952 down to $0.408; 4.2 and 4.5 times cheaper at 10 and 20 documents per domain. [Source](https://arxiv.org/abs/2609.18099)
- Relevance: High. Chunk skipping is a direct cost lever at this scale, though tested on small corpora with an LLM judge.

**D5. EGT-KG: Evidence-Grounded Typed KG Retrieval for Practical Scientific QA with Small Language Models**
- Authors: Muran Yu, Jiechao Gao, Yuandong Pan, Barney H. Miao, Andrew C. Lesh, Kincho H. Law, Jie Wang, Michael D. Lepech
- First posted: 31 Aug 2026 (arXiv v1). Published: Date not checked. Venue: EMNLP 2026 Industry track (per comment).
- Links: [arXiv:2609.00479](https://arxiv.org/abs/2609.00479)
- What I read: Abstract only
- Method: Compares vanilla RAG with a typed graph using an automatic relation schema or an expert schema, on local small models.
- Evidence: With llama3:8b: automatic schema 70.37 (+14.67%), expert schema 68.82 (+12.14%) vs vanilla RAG. [Source](https://arxiv.org/abs/2609.00479)
- Relevance: High. A direct automatic-versus-expert comparison, but on one narrow scientific literature set.

**D6. Is GraphRAG Needed? From Basic RAG to Graph-/Agentic Solutions with Context Optimization**
- Authors: Long Chen, Ryan Razkenari, Yuxuan Zhou, Yuan Tian, Rahul Ghosh, Venkatesh Pappakrishnan, Disha Ahuja, Vidya Sagar Ravipati
- First posted: 24 Jun 2026 (arXiv v1). Published: Date not checked. Venue: ACL 2026 GEM Workshop (per comment).
- Links: [arXiv:2606.25656](https://arxiv.org/abs/2606.25656)
- What I read: Abstract only
- Method: Nine standard RAG scenarios on semi-structured knowledge bases.
- Evidence: Context engineering cuts tokens 19-53%. Wider retrieval does not proportionally improve answers, so retrieval metrics can overstate benefit. [Source](https://arxiv.org/abs/2606.25656)
- Relevance: Medium to high.

**D7. ContextRAG: Extraction-Free Hierarchical Graph Construction for Retrieval-Augmented Generation**
- Authors: Roman Prosvirnin, Sergei Kuznetsov, Seungmin Jin
- First posted: 19 May 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2605.19735](https://arxiv.org/abs/2605.19735)
- What I read: Abstract only
- Method: Builds hierarchy from chunk embeddings using fuzzy formal concept analysis, with no LLM entity or relation extraction.
- Evidence: 130-task UltraDomain subset: 30 LLM calls and 22,073 tokens to index. A HiRAG reproduction used 870 calls and 3.54M tokens on 20 tasks before failing. F1 33.6% overall, not directly compared. [Source](https://arxiv.org/abs/2605.19735)
- Relevance: Medium. Extreme cost floor; the graph has no named relations, so it does not serve entity analytics.

**D8. Geometric Gain Graph: Zero-Token Graph Construction for Multi-Hop RAG**
- Authors: Zeliang Li, Xiaofen Xing, Kailing Guo, Wenyu Tao, Xiangmin Xu
- First posted: 17 Sep 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2609.19622](https://arxiv.org/abs/2609.19622)
- What I read: Abstract only
- Method: Zero-token chunk graph: edge weights from embedding geometry, trading relevance against novelty.
- Evidence: Abstract claims significant gains with no LLM cost at construction; I did not read the numbers. [Source](https://arxiv.org/abs/2609.19622)
- Relevance: Medium, for retrieval only.

**D9. EHRAG: Bridging Semantic Gaps in Lightweight GraphRAG via Hybrid Hypergraph Construction and Retrieval**
- Authors: Yifan Song, Xingjian Tao, Zhicheng Yang, Yihong Luo, Jing Tang
- First posted: 19 April 2026 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: Findings of ACL 2026.
- Date note: First posted 19 April 2026, just before the window.
- Links: [ACL Anthology 2026.findings-acl.1233](https://aclanthology.org/2026.findings-acl.1233/); [arXiv:2604.17458](https://arxiv.org/abs/2604.17458)
- What I read: Full text, targeted (PDF downloaded; I read the methods and result tables named here, not every page): efficiency sections, partly (key chart is a figure).
- Method: Hypergraph from sentence co-occurrence with light entity extraction plus clusters of entity embeddings; no tokens at construction.
- Evidence: Four datasets; 1.6% over the strongest baseline on the domain-specific corpus; linear indexing. [Source](https://aclanthology.org/2026.findings-acl.1233/)
- Relevance: High as the low-cost baseline to beat.

**D10. Collision to Cognition: Hash-Driven Graph Construction for Efficient RAG**
- Authors: Chuang Zhou, Zheng Yuan, Linhao Luo, Zhaozhuo Xu, Yilin Xiao, Junnan Dong, Siyu An, di Yin, Xing Sun, Xiao Huang
- First posted: Not established. Published: July 2026 (ACL Anthology). Venue: ACL 2026 (long).
- Date note: First posted: no arXiv version found by title search.
- Links: [ACL Anthology 2026.acl-long.1156](https://aclanthology.org/2026.acl-long.1156/)
- What I read: Abstract only
- Method: Builds structure from hash collisions, with bitwise operations and no triple extraction.
- Evidence: Outperforms baselines with no GPU required for construction. Exact figures not extracted. [Source](https://aclanthology.org/2026.acl-long.1156/)
- Relevance: Medium. Same caveat as other extraction-free graphs.

**D11. Beyond Chunks and Graphs: Retrieval-Augmented Generation through Triplet-Driven Thinking**
- Authors: Shengbo Gong, Xianfeng Tang, Qi He, Carl Yang, Wei Jin
- First posted: 4 August 2025 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: Findings of ACL 2026.
- Date note: First posted before the window.
- Links: [ACL Anthology 2026.findings-acl.1310](https://aclanthology.org/2026.findings-acl.1310/); [arXiv:2508.02435](https://arxiv.org/abs/2508.02435)
- What I read: Abstract only
- Method: Atomic triplet store with no graph; decomposes questions into triplets with placeholders and resolves them iteratively.
- Evidence: Up to 11% average gain across six datasets; retrieval cost down up to 45%. [Source](https://aclanthology.org/2026.findings-acl.1310/)
- Relevance: Medium to high. Suggests triples may be useful without graph traversal.

**D12. MiniRAG: A Lightweight RAG system with Small Language Models**
- Authors: Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang
- First posted: 12 January 2025 (arXiv API, v1). Published: July 2026 (ACL Anthology). Venue: ACL 2026 (long).
- Date note: First posted January 2025 under a different title.
- Links: [ACL Anthology 2026.acl-long.1721](https://aclanthology.org/2026.acl-long.1721/); [arXiv:2501.06713](https://arxiv.org/abs/2501.06713)
- What I read: Abstract only
- Method: Heterogeneous graph of chunks and named entities, designed for small language models.
- Evidence: Comparable to LLM-based methods with small models; 25% of the storage. [Source](https://aclanthology.org/2026.acl-long.1721/)
- Relevance: High on the small-model question, but the abstract gives limited numbers.

**D13. GRASP: Graph Agentic Search over Propositions for Multi-hop Question Answering**
- Authors: Stockton Jenkins, Ramya Korlakai Vinayak, Junjie Hu
- First posted: 15 May 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2605.16598](https://arxiv.org/abs/2605.16598)
- What I read: Abstract only
- Method: Agentic search over entities, propositions and passages.
- Evidence: Highest open-retrieval accuracy on MuSiQue and 2Wiki with 40-50% fewer tokens than IRCoT+HippoRAG2. [Source](https://arxiv.org/abs/2605.16598)
- Relevance: Low to medium. Mostly a query-time result.

**D14. Reducing Hallucinations in Complex Question Answering using Simple Graph-based Retrieval-Augmented Generation (long version)**
- Authors: Christopher J. Wedge, Joshua Stutter, Danny Dixon, Jacek Cała
- First posted: 4 Jun 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2606.05901](https://arxiv.org/abs/2606.05901)
- What I read: Abstract only
- Method: Agent with vector and graph tools over a simple graph schema on curated Wikipedia.
- Evidence: MoNaCo: graph tools can halve hallucinated answers with modest token increase. [Source](https://arxiv.org/abs/2606.05901)
- Relevance: Medium. Pro-graph evidence, but a light fixed schema.

**D15. Operationalizing Cyber Threat Intelligence with GraphRAG**
- Authors: Atul Kabra, Prakhar Paliwal, Manjesh K. Hanawal
- First posted: 13 Aug 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2608.13050](https://arxiv.org/abs/2608.13050)
- What I read: Abstract only
- Method: Same report, prompt and model; only retrieval differs (Microsoft GraphRAG vs naive RAG).
- Evidence: One case: 100% vs 29% of detections kept firing after rotation of indicators. Nine reports: GraphRAG plans reach harder-to-evade levels, sometimes close on total score. Prompt wording matters almost as much. [Source](https://arxiv.org/abs/2608.13050)
- Relevance: Low to medium. Very small sample.

**D16. Scan or Index? large Evidence Coverage, Answer Quality, and Cost in Long-Document RAG**
- Authors: Juan Terven, Diana-Margarita Córdova-Esparza, Julio-Alejandro Romero-González, Edgar Chavez-Urbiola, Juan Irving Vasquez, Julian Andres Ramirez-Bautista, Silvia L. Chaparro-Cárdenas, Gendry Alfonso-Francia
- First posted: 20 Oct 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2610.28957](https://arxiv.org/abs/2610.28957)
- What I read: Abstract only
- Method: 11 retrieval configurations on 396 QASPER and 80 book questions, with no graph.
- Evidence: Hybrid with reranking has highest coverage (94.1%), and more evidence did not always give better answers. [Source](https://arxiv.org/abs/2610.28957)
- Relevance: Medium. A strong non-graph baseline to compare against.

**D17. Reading Between the Chapters: A Graph-Free RAG Rebuilt With AI and Human Feedback**
- Authors: Dipto Sumit, Syeda Fabiayee Jahan, Farig Sadeque
- First posted: 31 Oct 2026 (arXiv v1). Published: None found. Venue: Preprint.
- Links: [arXiv:2611.00752](https://arxiv.org/abs/2611.00752)
- What I read: Abstract only
- Method: Graph-free index of passages, summaries and book-level cards.
- Evidence: CrossChapterQA, 1,853 questions over 14 books. About $0.41 to index a book and under a cent per answer. [Source](https://arxiv.org/abs/2611.00752)
- Relevance: Medium. Evidence cross-section questions may not need a graph.

**D18. PILAR: A Page-Grounded Unified Evidence Representation via an Entity-Linked Assertion Graph for Open-Domain QA Agents over Multimodal Document Corpora**
- Authors: Joongmin Shin, Gyuho Shim, Jung-Hun Lee, Jaehyung Seo
- First posted: 26 Sep 2026 (arXiv v1). Published: October 2026 (ACL Anthology). Venue: Findings of EMNLP 2026.
- Links: [ACL Anthology 2026.findings-emnlp.128](https://aclanthology.org/2026.findings-emnlp.128/); [arXiv:2609.32895](https://arxiv.org/abs/2609.32895)
- What I read: Abstract only
- Method: Entity-linked assertion graph across text, tables and figures, used as a linking layer over page retrieval.
- Evidence: +1.6 EM over flat retrieval overall, +2.9 on compositional, +5.9 on 3-hop questions. Gains mostly from the text slice. [Source](https://aclanthology.org/2026.findings-emnlp.128/)
- Relevance: High. Handles tables and keeps pages as ground truth.

**D19. Graph-Based Retrieval-Augmented Generation: Applications, Challenges, Solutions, and Opportunities**
- Authors: Angela Bonifati, Yixiang Fang, Sibo Wang, Yingli Zhou
- First posted: Not established. Published: 1 August 2026 (Crossref). Venue: PVLDB.
- Links: [DOI 10.14778/3827998.3828147](https://doi.org/10.14778/3827998.3828147)
- What I read: Title page and first part of the abstract only.
- Method: Tutorial surveying graph-based RAG from a data management view.
- Evidence: None; survey. [Source](https://doi.org/10.14778/3827998.3828147)
- Relevance: Medium. Good route into VLDB work I may have missed.

**D20. RAG vs. GraphRAG: A Systematic Evaluation and Key Insights**
- Authors: Haoyu Han, Li Ma, Yu Wang, Harry Shomer, Kai Guo, Yongjia Lei, Zhisheng Qi, Zhigang Hua, Bo Long, Hui Liu, Charu Aggarwal, Jiliang Tang
- First posted: 17 February 2025 (arXiv API, v1). Published: 8 August 2026 (Crossref). Venue: KDD 2026.
- Date note: First posted February 2025, well before the window.
- Links: [arXiv:2502.11371](https://arxiv.org/abs/2502.11371); [DOI 10.1145/3770855.3817575](https://doi.org/10.1145/3770855.3817575)
- What I read: Title and metadata only for this version (NOT READ in this session). Summary below rests on a search snippet.
- Method: Systematic comparison of RAG and GraphRAG.
- Evidence: Search snippet only, unverified: RAG does well on single-hop factual questions, GraphRAG on multi-hop reasoning. [Source](https://arxiv.org/abs/2502.11371)
- Relevance: High, because it is the widely cited comparison now peer reviewed.

#### E. Just outside the window, listed because they are clearly important

- **Wikontic: Constructing Wikidata-Aligned, Ontology-Aware Knowledge Graphs with Large Language Models.** Alla Chepurova, Aydar Bulatov, Mikhail Burtsev, Yuri Kuratov. First posted 29 Nov 2025 (arXiv API); EACL 2026, March 2026. Abstract only. 96% of MuSiQue generated triplets contain the correct answer entity, 86% on MINE-1, under 1,000 output tokens, about 3 times fewer than AriGraph and under 1/20 of GraphRAG. Uses Wikidata constraints, so not schema-free. [Source](https://aclanthology.org/2026.eacl-long.388/)
- **ATOM: AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs.** Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, Rémy Cazabet, Pierre Cléau. First posted 26 Oct 2025 (arXiv API); Findings of EACL 2026, March 2026. Abstract only. Splits text into atomic facts, then merges in parallel: 18% higher exhaustivity, 33% better stability, over 90% latency reduction. High relevance for parallel merging. [Source](https://aclanthology.org/2026.findings-eacl.49/)
- **Zero-Shot Open-Schema Entity Structure Discovery (ZOES).** Xueqiang Xu, Jinfeng Xiao, James Barry, Mohab Elkaref, Jiaru Zou, Pengcheng Jiang, Yunyi Zhang, Max Giammona, Geeth de Mel, Jiawei Han. First posted 4 Jun 2025 (arXiv API); EACL 2026. Abstract only. Extracts entity attributes without schema or examples through enrichment, refinement and unification. [Source](https://aclanthology.org/2026.eacl-long.354/)
- **DIAL-KG: Schema-Free Incremental Knowledge Graph Construction via Dynamic Schema Induction and Evolution-Intent Assessment.** Weidong Bao, Yilin Wang, Ruyu Gao, Fangling Leng, Yubin Bao, Ge Yu. First posted 20 Mar 2026; DASFAA 2026 per comment. Abstract only. Closed loop of extraction, governance and schema evolution; incremental. [Source](https://arxiv.org/abs/2603.20059)
- **Are Large Language Models Effective Knowledge Graph Constructors? (D2A-HKG).** Ruirui Chen, Weifeng Jiang, Chengwei Qin, Bo Xiong, Kaiwen Wei, Fiona Liausvia, Pei Fang Tan, Ker Yung Chua, Dongkyu Choi, Mukkesh Kumar, Evelyn C. Law, Dennis Wang, Boon Kiat Quek. First posted 13 Oct 2025. Abstract only. Seven frontier LLMs, zero-shot and schema-free, on paediatric research articles (CMW-Lit); finds relevant, document-faithful triples with limited hallucination. [Source](https://arxiv.org/abs/2510.11297)
- **LINK-KG: LLM-Driven Coreference-Resolved Knowledge Graphs for Human Smuggling Networks.** Dipak Meher, Carlotta Domeniconi, Guadalupe Correa-Cabrera. First posted 30 Oct 2025; ICKG 2025. Abstract only. A prompt cache tracks references across chunks: 45.21% less node duplication and 32.22% fewer noisy nodes. [Source](https://arxiv.org/abs/2510.26486)
- **Democratizing GraphRAG: Linear, CPU-Only Graph Retrieval for Multi-Hop QA (SPRIG).** Qizhi Wang. v1 stamped 27 Dec 2025 despite the 2602 ID. Abstract only. Token-free, NER co-occurrence graph with Personalized PageRank. It also finds strong lexical hybrids can be sufficient. [Source](https://arxiv.org/abs/2602.23372)
- **LiteSemRAG: Lightweight LLM-Free Semantic-Aware Graph Retrieval for Robust RAG.** Xiao Yue, Guangzhi Qu, Lige Gan. v1 stamped 16 Mar 2026 despite the 2604 ID. Abstract only. [Source](https://arxiv.org/abs/2604.16350)
- **Don't Worry About Quality: Robust Multi-Hop Retrieval despite Issues in Knowledge Graphs (CS-RAG).** Yizhuo Ma, Ke Qin, Jinchuan Xu, Tao Wen, Rongzheng Wang, Jiakai Li, Nan Chen, Lunxu Liu, Deqiang Ouyang, Shuang Liang, Man-Fai Leung. First posted 16 Mar 2026. Abstract only. Finds similar extraction error patterns across graph builders: spurious noise and incomplete information. A search result listed this ID under a different title and system name; the arXiv page is what is given here. [Source](https://arxiv.org/abs/2603.14828)
- **FinReflectKG -- HalluBench: GraphRAG Hallucination Benchmark for Financial Question Answering Systems.** Mahesh Kumar, Bhaskarjit Sarmah, Stefano Pasquali. First posted 11 Mar 2026. Abstract only. 755 examples over SEC filings; LLM judges reach F1 0.82-0.86 when clean, and MCC falls 44-84% with noisy triples for most methods. [Source](https://arxiv.org/abs/2603.20252)
- **Lost in Formatting: How Output Formats Skew LLM Performance on Information Extraction.** Rishi Ravikumar, Nuhu Ibrahim, Riza Batista-Navarro. First posted: no arXiv version found by title search. EACL 2026. Abstract only. Over 280 experiments; changing only the output format can shift F1 by over 40%. [Source](https://aclanthology.org/2026.eacl-long.256/)

#### F. Seen but not assessed (title or partial abstract only; treat dates as unverified unless stated)

These may be worth a look: [2605.05476](https://arxiv.org/abs/2605.05476) (benchmark for graph construction methods with GNNs; abstract read, biomedical), [2605.22304](https://arxiv.org/abs/2605.22304) (KGI-Bench for integration pipelines; abstract partly read), [2606.31718](https://arxiv.org/abs/2606.31718) (cross-lingual RE on Romanian; abstract partly read), [2607.21610](https://arxiv.org/abs/2607.21610) (SCOPE and SCION schema induction benchmark, v1 20 May 2026; likely the ontology researcher's scope), [2607.16201](https://arxiv.org/abs/2607.16201) (Generative Ontology Induction, v1 1 May 2026; same), [2606.01208](https://arxiv.org/abs/2606.01208) (ANCHOR; ontology-supplied CTI), [2608.14055](https://arxiv.org/abs/2608.14055) (HERMES; structured extraction from 55 volumes of ultra-long documents, including tables, F1 about 0.90, with domain constraints; abstract read), [2606.28351](https://arxiv.org/abs/2606.28351) (HyperSU), [2609.29695](https://arxiv.org/abs/2609.29695) (EvLink), [2610.31684](https://arxiv.org/abs/2610.31684) (KG quality framework). From the ACL Anthology: [TH-RAG](https://aclanthology.org/2026.acl-long.1740/), [TagRAG](https://aclanthology.org/2026.findings-acl.321/), [HyperRAG-IHM](https://aclanthology.org/2026.emnlp-main.898/), [Beyond Static Retrieval](https://aclanthology.org/2026.emnlp-main.1897/). From Crossref: [Core-based Hierarchies for Efficient GraphRAG, KDD 2026](https://doi.org/10.1145/3770855.3818007), [Lost in Extraction: RE-ErrorTax, CIKM 2026](https://doi.org/10.1145/3799682.3840080), [QA-GraphRAG, PVLDB](https://doi.org/10.14778/3819518.3819546).

### Inferences
- The list is biased toward papers whose titles use standard terms, because filtering was on titles.
- Publication and first posting often differ by many months. Papers such as AutoSchemaKG and MiniRAG are old preprints with new venues, so their ideas are less new than their publication dates suggest.

### Gaps
- OpenAlex search returned 503 or 429, Semantic Scholar 429, and DBLP served a bot check, so none could be used for discovery. Results rest on arXiv listings, ACL Anthology and Crossref.
- arXiv categories other than the four crawled (for example cs.LG, cs.CR, cs.SE) were not crawled. ANCHOR in cs.CR was found only through web search, so others may be missing.
- No ISWC 2026 or TKDE paper on the topic surfaced from Crossref title queries; this does not show none exists. COLING and OpenReview were not systematically checked. NeurIPS 2026 workshops appear only where an arXiv comment names them.
- ACM papers could not be opened, so AgentsKG, Less Is More, Self-Evolving Ontology and RE-ErrorTax are unread and need the user to fetch them.

## What are the new methods for open IE, open NER and LLM-based document-level graph construction, including GraphRAG-style indexing?

### Takeaway
Work has shifted from extracting better triples to managing what happens after extraction: separating extraction from consolidation, reviewing or repairing triples after the fact, and skipping LLM extraction altogether. Classic open IE is now a small strand, and I found almost no new open NER work as such.

### Cited Findings
**Multi-stage LLM pipelines**
- Question-first extraction: SocraticKG turns text into 5W1H question-answer pairs before extracting triples, retaining 73.4-96.3% of facts on MINE against 48.5-59.7% for GraphRAG — [Source](https://aclanthology.org/2026.findings-acl.1951/)
- Extract then consolidate: RAGU uses two-stage typed extraction, DBSCAN deduplication, summarisation and Leiden communities around a 7B model — [Source](https://arxiv.org/abs/2607.11683)
- Shared corpus memory during extraction: MemGraphRAG agents consult a common memory to resolve conflicts — [Source](https://arxiv.org/abs/2606.00610)
- Review before admission: MAGG induces types from content, then domain agents approve, reject or revise triples — [Source](https://arxiv.org/abs/2608.28642)
- Web-scale automatic schema: AutoSchemaKG reports 50 million documents and 900+ million nodes — [Source](https://aclanthology.org/2026.acl-long.942/)
- Detail preservation: F-GraphRAG targets lost quantities, dates and attributes, and reports +12.6% accuracy with 33.8% fewer relationships — [Source](https://aclanthology.org/2026.findings-emnlp.2253/)
- Training the builder on downstream usefulness: AutoGraph-R1 uses reinforcement learning with RAG performance as reward — [Source](https://aclanthology.org/2026.acl-long.1070/)

**Open IE and open relation extraction**
- DiffIE is a new non-LLM OpenIE model using discrete diffusion over per-token role tags, reporting a new best on CaRB (1-1) — [Source](https://aclanthology.org/2026.emnlp-main.1851/)
- ReaORE narrows then compares candidate relations using a reasoning model; it reports gains on two OpenRE datasets without numbers in the abstract — [Source](https://arxiv.org/abs/2606.26986)

**Graphs without LLM extraction**
- EHRAG builds hyperedges from sentence co-occurrence and entity-embedding clusters at zero construction tokens — [Source](https://aclanthology.org/2026.findings-acl.1233/)
- MeshRAG builds structure from hash collisions — [Source](https://aclanthology.org/2026.acl-long.1156/)
- ContextRAG builds a concept hierarchy over chunk embeddings — [Source](https://arxiv.org/abs/2605.19735)
- G3RAG weights chunk-to-chunk edges from embedding geometry — [Source](https://arxiv.org/abs/2609.19622)

### Inferences
- The field now treats single-pass extraction per chunk as the baseline to beat, not as a finished method.
- Extraction-free graphs answer a different need. They index passages for retrieval but give no named entities or relations, so they would not support entity analytics or an ontology step.

### Gaps
- I found no in-window paper presenting open NER as its own task. Zero-shot NER papers (for example [SAM-NER](https://arxiv.org/abs/2605.03706), seen by title only) still appear to assume a label set; this rests on titles and I did not confirm it.
- No paper tested these pipelines on UN or development evaluation reports.

## How do they handle cross-chunk problems: coreference, entity resolution, canonicalisation and hallucinated or low-quality extractions?

### Takeaway
Cross-chunk relations are now an explicit research target, but measured gains are small. The strongest practical message concerns merging: a wrong merge does lasting damage, and better candidate retrieval may not produce better final entity groups.

### Cited Findings
**Cross-chunk relations**
- CrossAug adds 5,048-8,901 triples to base indexes of 64,523-123,308 by calling the LLM only on subgraphs a GNN flags as incomplete; F1 rises by +1.33 on MuSiQue and +1.09 on LiteraryQA with HippoRAG2 — [Source](https://arxiv.org/abs/2605.28004)
- RAGA has an agent search the existing graph before writing, with only preliminary evidence on a QASPER subset — [Source](https://arxiv.org/abs/2605.17072)

**Coreference**
- ImCoref-CeS pairs a supervised long-text model with an LLM that checks mentions and splits bad clusters — [Source](https://aclanthology.org/2026.acl-long.1122/)
- MEIC-DT does incremental long-text coreference without an LLM under a memory budget — [Source](https://aclanthology.org/2026.findings-acl.881/)
- FineREX dropped its coreference rewriting stage in favour of a fine-tuned extractor and lowered node duplication on long documents from 17.78% to 11.17% — [Source](https://arxiv.org/abs/2606.19710)
- Just outside the window: LINK-KG tracks references across chunks with a prompt cache, cutting node duplication 45.21% against its baselines — [Source](https://arxiv.org/abs/2510.26486)

**Entity resolution**
- Retrieval gains may not reach clusters: on 3,633 multilingual company names, pair recall rose from 53.74% to 76.98% while cluster recall moved from 32.54% to 33.33% — [Source](https://arxiv.org/abs/2610.05573)
- A single wrong link can chain hundreds of records when matches are treated as transitive, on benchmarks of up to 5M records — [Source](https://arxiv.org/abs/2607.26298)
- In a government-document graph of 2.2M relationships, one merge corrupted a correct record and deleted eight entities from an unrelated document, after which resolution was limited to flagging candidates. The authors state they report no identity benchmark — [Source](https://arxiv.org/abs/2608.10644)
- Budgeted LLM use: Alper spends a fixed query budget during label propagation — [Source](https://arxiv.org/abs/2605.25814); Batched ER picks batches to maximise recall per call — [Source](https://arxiv.org/abs/2606.24407); decision models give calibrated probabilities, and sending unsure pairs to an LLM ensemble adds 1.9-4.0 F1 points — [Source](https://arxiv.org/abs/2610.34626)
- LLM-labelled training data lets small matchers run 34-459 times faster than direct LLM matching at $5.86-$8.11 per training set, though they weaken on unseen products — [Source](https://arxiv.org/abs/2606.28823)
- Multilingual identity: mapping mentions to Wikidata IDs via 37 million aliases in 36 languages was applied to 800,000 articles from two countries — [Source](https://arxiv.org/abs/2606.27347). Small models can under-merge across scripts: Qwen2.5-7B same-person accuracy fell from 35.1% to 8.5% in Devanagari — [Source](https://arxiv.org/abs/2610.31407)

**Canonicalisation**
- OAK+MEND maps predicates and types by embedding, then corrects only violations with an LLM, reaching 96.8% consistency from 63.4% at 59-79% of Wikontic's token cost; it relies on an ontology — [Source](https://aclanthology.org/2026.findings-emnlp.1867/)
- AutoPKG keeps one canonical graph through a central decision agent, scoring 0.953 WKE on types and 0.724 on keys — [Source](https://aclanthology.org/2026.findings-acl.766/)
- Two CIKM 2026 papers look directly relevant but are not read: [Less Is More](https://doi.org/10.1145/3799682.3839949) and [Self-Evolving Ontology for Entity Canonicalization](https://doi.org/10.1145/3799682.3840115)

**Hallucinated and low-quality extractions**
- GraphRefine goes beyond deletion by editing or rewriting triples, and raises BL F1 over GraphJudge on all four datasets (for example 52.94 vs 47.57 on DocRED) — [Source](https://aclanthology.org/2026.acl-long.1353/)
- Requiring a verbatim quote for each edge reached 0.94 precision and 0.71 recall on 61 vendors, against a key that was itself model-drafted — [Source](https://arxiv.org/abs/2610.32781)
- In a preregistered audit of threat reports, only 1.2% of false positives were unsupported, while 65.6% were correct strings from background context — [Source](https://arxiv.org/abs/2610.20417)
- Removing 40% of low-frequency triples slightly raised accuracy in MemGraphRAG's motivating test (65.28% vs 64.85%) — [Source](https://arxiv.org/abs/2606.00610)
- Two builds of the same 300-document corpus differed in 29,475 artifacts, five times the change from removing 61 retracted documents — [Source](https://arxiv.org/abs/2610.17719)
- Self-explanations from LLMs for matches are often unstable and weakly faithful — [Source](https://arxiv.org/abs/2606.01210)

### Inferences
- For this corpus, entity merging should default to flagging, with automatic merges only on strong evidence. Two independent papers point that way (Curate Before You Connect and Entity Resolution in Practice), though neither measures it on comparable data.
- Linking to external identifiers such as Wikidata looks like the most tested route for French, Spanish and English mentions of the same organisation or place. It will not cover programme or project names missing from Wikidata (my inference; not measured).
- Hypothesis, not tested: "right string, wrong context" may be the main error in evaluation reports, because they cite past programmes and other countries at length. A sample audit should separate this from invention, since quote-checking catches invention but not misplaced context.
- Frequency filtering should be used with care. Rare triples may be noise in QA benchmarks yet be the only record of a small programme in this corpus (my inference).

### Gaps
- I found no paper measuring coreference across chunks at corpus scale with gold labels.
- I found no in-window study of relation-name canonicalisation at scale without an ontology. The two CIKM papers may cover this.
- No paper reports entity resolution on extracted mentions from mixed English, French and Spanish reports.

## What is the reported quality against gold data, and the cost or LLM calls per document? Are small models or non-LLM alternatives compared?

### Takeaway
Reported scores vary widely and depend heavily on the scoring rule, so they are hard to compare across papers. Costs differ by two orders of magnitude between systems, and well-tuned small models repeatedly match or beat zero-shot use of large ones, given some labelled data.

### Cited Findings
**Quality against gold**
- The same prediction set can span 0.16-0.70 F1 depending on the matching rule, and 11 of 45 pairwise system orderings reverse under different protocols — [Source](https://arxiv.org/abs/2609.01671)
- MAGG's best strict triple F1 on SciERC is 0.156, rising to 0.290 with mapped scoring — [Source](https://arxiv.org/abs/2608.28642)
- Incomplete gold can reverse rankings: rescoring identical predictions on Re-DocRED moved one method from fifth to first — [Source](https://arxiv.org/abs/2607.24312)
- Whole-document entity extraction reached micro precision of only 0.03-0.34, including on a flagship model — [Source](https://arxiv.org/abs/2610.20417)
- A gold-free alternative: KGCQual scores entity and relation fidelity without reference triples and correlates with link prediction — [Source](https://arxiv.org/abs/2607.10212)
- LLM judges collapse when shown the system's label (J-ROC-AUC 0.12-0.87) and recover to 0.93-1.00 when not — [Source](https://aclanthology.org/2026.emnlp-main.1717/)

**Cost** (figures are as reported, on different corpora and in different units, so they cannot be compared directly)
- HCG-RAG table, medical benchmark build calls: MS-GraphRAG 17,638; LightRAG 4,423; Fast-GraphRAG 2,882; HCG-RAG M 1,167; XS 131; NaiveRAG 0 — [Source](https://arxiv.org/abs/2607.22592)
- WildGraphBench, one domain: flat indexing in seconds, graphs 28 minutes to 21.5 hours, tokens 0 to 90M — [Source](https://aclanthology.org/2026.findings-acl.679/)
- RAGU: about 8k tokens per document, about $0.001 per document on rented GPU vs about $0.10 on commercial APIs (authors' estimate) — [Source](https://arxiv.org/abs/2607.11683)
- SocraticKG: 333,917-553,530 tokens per 100 MINE articles, against 145,262-150,569 for direct extraction and about 331,000-334,000 for GraphRAG — [Source](https://aclanthology.org/2026.findings-acl.1951/)
- ContextRAG: 30 LLM calls and 22,073 tokens for a 130-task subset — [Source](https://arxiv.org/abs/2605.19735)
- EffiRAG: total $0.408 vs $0.952 for LightRAG-hybrid on UltraDomain, with a non-LLM filter that skips low-salience chunks at larger sizes — [Source](https://arxiv.org/abs/2609.18099)
- Template community reports took 0.005 minutes vs 46.129 minutes for LLM summaries, with better results — [Source](https://aclanthology.org/2026.tacl-1.29/)
- Governance raises cost: MAGG reports more calls and prompt tokens than flat insertion — [Source](https://arxiv.org/abs/2608.28642)

**Small and non-LLM comparisons**
- A tuned small model beat GPT-4o-mini few-shot on GraphRAG answer recall (7.7 vs 4.8 and 18.2 vs 14.5), driven by relation precision of 66.0 vs 38.2 and 66.5 vs 10.8 — [Source](https://aclanthology.org/2026.tacl-1.29/)
- Qwen2.5-0.5B fine-tuned reached micro-F1 0.83 vs 0.69 for GPT-5.4 and 0.66 for Claude Sonnet 4.6 zero-shot, on fixed relation sets — [Source](https://arxiv.org/abs/2606.22606)
- A graph-based parser pulls ahead of four LLMs as relations per document increase — [Source](https://aclanthology.org/2026.acl-short.17/)
- SocraticKG's weakest backbone, Qwen2.5-7B, retained 73.4% against 96.3% with Claude-4, so model size still matters for zero-shot extraction — [Source](https://aclanthology.org/2026.findings-acl.1951/)
- LinearRAG, with no LLM extraction, had the best F1 on 2WikiMultiHopQA in CrossAug's comparison (65.16), but lost on the other three benchmarks — [Source](https://arxiv.org/abs/2605.28004)
- On 8B-or-smaller models, batching cut energy per page 38-85% with no accuracy loss, and neural OCR used 17 times the energy of classical OCR — [Source](https://arxiv.org/abs/2609.31341)

### Inferences
- Rough scale check, my arithmetic rather than a reported result: 1.9 million chunks across 13,000 reports is about 146 chunks per report, so one call per chunk means at least 1.9 million LLM calls. RAGU's quoted rates work out to about $0.125 and $12.50 per million tokens. If a chunk averages 400 tokens (an assumption; the real figure should be measured from the corpus), one input pass is about 760 million tokens, or about $95 and $9,500, before prompts, output, retries or multi-stage steps. SocraticKG's multi-stage overhead of about 2.3-3.7 times direct extraction suggests the multiplier to budget for.
- A small labelled set is likely to pay for itself. FineREX used 512 annotated chunks, and Dissecting GraphRAG and Sub-Billion both favour tuned small models. These are different tasks and domains, so a pilot on the UN chunks would be needed to confirm it.
- Any gold set for this project should fix its matching rule first and report more than one, following the Matcher audit.

### Gaps
- Few papers report LLM calls per document in a directly usable form. The MAGG cost table exists in its appendix but I did not extract the numbers.
- I did not find a paper reporting cost at millions of chunks with a measured quality figure. AutoSchemaKG may do so; I read only the abstract.
- No quality data on French or Spanish open extraction was found in the papers read.

## What do new evaluations say about schema-free graphs versus plain retrieval or schema-guided graphs?

### Takeaway
The new evaluations do not support graphs as a general upgrade. Graphs help on multi-fact aggregation and on some multi-hop tasks, while plain retrieval stays competitive on single facts and summaries, and a stronger answer model can erase the gap. The limited evidence on schema guidance favours compact or induced structure, not expert schemas.

### Cited Findings
**Graph vs plain retrieval**
- On long, mixed web sources, NaiveRAG averaged 59.79 against HippoRAG2 64.33, LightRAG 56.76, MS GraphRAG global 54.54 and local 38.23. MS GraphRAG global led on multi-fact questions (47.64 vs 35.08), and NaiveRAG led on summary F1 (15.84) — [Source](https://aclanthology.org/2026.findings-acl.679/)
- Default GraphRAG slightly trailed document RAG on factoid questions, and only a well-configured version pulled ahead — [Source](https://aclanthology.org/2026.tacl-1.29/)
- On HotpotQA with GPT-OSS-120B, vector RAG scored 58.6/72.9 EM/F1, above HippoRAGv2 (38.8/57.0), KGGen, Wikontic and OAK. With the smaller Qwen3-30B the order flipped: HippoRAGv2 50.0/64.1 vs vector RAG 42.9/56.5 — [Source](https://aclanthology.org/2026.findings-emnlp.1867/)
- On a medical benchmark, answer correctness was NaiveRAG .608, MS-GraphRAG global .573 and local .654, LightRAG .673, Fast-GraphRAG .690, while on EpilepsyQA NaiveRAG (.397) and LightRAG (.334) both fell well behind HCG-RAG (.604-.611) — [Source](https://arxiv.org/abs/2607.22592)
- SocraticKG beat naive RAG by 8.5 points on hard-bridge questions with Claude-4 (56.38 vs 47.88) — [Source](https://aclanthology.org/2026.findings-acl.1951/)
- PILAR improved on flat retrieval by +1.6 EM overall and +5.9 on 3-hop questions over multimodal documents — [Source](https://arxiv.org/abs/2609.32895)
- A graph-free index answered cross-chapter questions at about $0.41 to index a book — [Source](https://arxiv.org/abs/2611.00752)
- Wider retrieval does not improve answers proportionally, so retrieval metrics can overstate benefit — [Source](https://arxiv.org/abs/2606.25656)
- Measurement warning: standard retrieval modes hid graph construction differences (at most 1 point), while direct graph access showed gaps up to 47.6 points — [Source](https://arxiv.org/abs/2605.21974)

**Schema-free vs schema-guided**
- HCG-RAG uses an automatically induced fixed vocabulary, with answer quality matching entity-relation systems at 8-135 times fewer build calls than MS-GraphRAG. It argues that what is placed in the graph matters more than how many nodes it contains — [Source](https://arxiv.org/abs/2607.22592)
- An automatic relation schema scored 70.37 and an expert schema 68.82 against vanilla RAG on one literature set with llama3:8b — [Source](https://arxiv.org/abs/2609.00479)
- A schema can reduce fidelity: on statistical tables, a mismatched schema dropped fact coverage below the unconstrained baseline on 4 of 6 datasets — [Source](https://arxiv.org/abs/2605.21974)
- OAK (ontology-grounded) and KGGen (no ontology) can be compared on HotpotQA F1: 57.8 vs 48.5 with GPT-OSS-120B, and 58.6 vs 48.7 with Qwen3 — [Source](https://aclanthology.org/2026.findings-emnlp.1867/)
- On SciERC, MAGG's review step improved strict F1 over flat insertion (0.156 vs 0.106). This tests governance, not schema-free against schema-guided — [Source](https://arxiv.org/abs/2608.28642)

**Effect of graph defects**
- Extraction quality limits downstream results: gold triples gave answer recall of 11.7 on CDR-QA, a tuned small model 7.7 and GPT-4o-mini 4.8 — [Source](https://aclanthology.org/2026.tacl-1.29/)
- Just outside the window: most detectors degrade sharply when noisy triples are introduced, with MCC falling 44-84% — [Source](https://arxiv.org/abs/2603.20252)

### Inferences
- For a question-answering product on this corpus, chunk retrieval with hybrid search is the baseline any graph has to beat. WildGraphBench is the closest precedent.
- The stronger case for building a graph here is corpus analytics and ontology induction, which retrieval cannot do, not better QA. I found no paper testing this directly.
- The evidence on schema-free versus schema-guided is thin and mostly comes from papers promoting one approach, so treat it as suggestive. The OAK-KGGen gap partly reflects different pipelines, not only the schema.
- Multi-fact aggregation across reports is where graph-style retrieval seems most likely to add value for evaluation synthesis questions, though no study covers such questions on this kind of corpus.

### Gaps
- I found no independent, head-to-head study of schema-free, induced-schema and expert-schema graphs on one corpus with one protocol. RAG vs. GraphRAG (KDD 2026) was not read in this session.
- Almost all evaluations use QA benchmarks, which by the Format-Constraint result may hide graph quality differences.
- Benchmarks use clean Wikipedia-style or curated text. None combines long documents, noisy PDF parsing, tables and several languages.
