# LLM-based ontology learning and schema or taxonomy induction from text, with no ontology supplied (window: May to November 2026, checked 2026-11-05)

How to read this file:

- **First-posted dates** for arXiv papers are the `[v1]` submission date read from each paper's arXiv abstract page. **Published dates** come from Crossref records or the publisher page. They are reported separately.
- **The arXiv ID month is not the first-posted month.** Four papers show this: 2607.16201 (v1 1 May 2026), 2607.21610 (v1 20 May 2026), 2608.12326 (v1 30 May 2026), 2606.01208 (v1 31 May 2026). The reason for the gap was not investigated.
- **"Read" labels are exact:**
  - **FT-main**: I read the paper's main body.
  - **FT-part**: I retrieved the full text and read only the named sections, plus keyword extracts.
  - **ABS**: abstract only.
  - **META**: title and metadata only.
- **"Not read"** covers paywalled or blocked items. ACM DL and Sage returned HTTP 403 to automated fetches. Springer returned a login redirect. These papers can still be obtained by the user.
- **All numbers are the authors' own unless stated.** I re-ran nothing.
- **Author names** follow arXiv or Crossref. Semantic Scholar's author lists differed in places (for example 2608.09049 and 2606.01208), so I did not use them.

## Which real papers induce an ontology, taxonomy, type hierarchy or KG schema from text?

### Takeaway
Yes, there is a substantial new body of work. The most directly relevant papers for a chunked, heterogeneous corpus are SCOPE/SCION, GrOIL, TaxCE, the thematic-corpora method (2610.01463), CORTEX, TERMNET and AutoSchema (KDD). AutoSchemaKG (ACL 2026) and TRACE-KG sit just outside the window. Several papers with "schema-agnostic" or "ontology" in the title actually assume a supplied ontology; these are listed separately.

### Cited Findings

#### A. Induction from text with no full ontology supplied (in window)

**1. SCOPE and SCION: A Benchmark and an Auditable Reference Pipeline for Schema Induction and Fusion from Text**
- Authors: Miaobo Hu, Xiaobo Guo, Shuhao Hu, Bokun Wang, Rui Chen, Xin Wang, Daren Zha, Jun Xiao.
- First posted: 2026-05-20. Published: no venue found. Link: [arXiv:2607.21610](https://arxiv.org/abs/2607.21610). Read: FT-main plus Appendix Tables 27 and 28.
- Method: chunks text at 1,200 whitespace tokens with no overlap and aggregates chunk evidence into a source-level candidate package. Constrained LLM modules then name, merge and filter only inside that candidate space, under a strict JSON contract with evidence pointers. Induces entity, relation, event and role types — [source](https://arxiv.org/abs/2607.21610).
- Evidence: macro F1 over 24 sources of 0.7518 Literal, 0.9298 Fuzzy, 0.8909 Continuous, 0.7888 Graph. The strongest baseline, extract-then-aggregate (ETA), scored 0.7063, 0.9141, 0.8759, 0.7629. An RL-trained Qwen3-8B variant reached 0.8025, 0.9310, 0.9101, 0.8228 — [source](https://arxiv.org/abs/2607.21610).
- Relevance: the closest match to chunk-to-schema consolidation. Includes zh/en sources. Not tested on long noisy reports.

**2. GrOIL: Graph-Grounded Domain Ontology Induction with Constrained LLM Mediation**
- Authors: Maruf Ahmed Mridul, Abid Talukder, Oshani Seneviratne.
- First posted: 2026-08-22. Published: CIKM '26, Crossref online date 2026-11-06 (one day after this check). Links: [arXiv:2608.22135](https://arxiv.org/abs/2608.22135), [DOI 10.1145/3799682.3840910](https://doi.org/10.1145/3799682.3840910). Read: FT-part (introduction, related work, method through Stage 3, Evaluation sections 4.1 to 4.6).
- Method: seven-stage pipeline from documents to discourse hypergraphs to an OWL TBox. The LLM only makes local bounded decisions such as naming clusters and resolving domain and range. Classes, properties and restrictions are clustered as separate streams with embedding-based agglomerative clustering — [source](https://arxiv.org/abs/2608.22135).
- Evidence: SPARQL competency-question (CQ) coverage 0.8527 and 0.7694, against 0.6304 and 0.4027 (direct LLM) and 0.6196 and 0.4383 (multi-agent). Keyphrase coverage 86.2% to 88.7%. 91% accuracy on 300 CQs, judged by an LLM. 86.38% on the gap-and-overlap benchmark, against 87.76% for Claude Sonnet 4.6 reading the contracts directly — [source](https://arxiv.org/abs/2608.22135).
- Relevance: strong consolidation design and full provenance. One domain only, and the 100-contract corpus is synthetic.

**3. TaxCE: A Framework for Automated Taxonomy Construction and Evaluation at Scale**
- Authors: Sandeep Sricharan Mukku, Albert Aristotle Nanda, Rohit Pyati.
- First posted: 2026-08-31. Published: EMNLP 2026 Industry Track (Crossref issued year 2026 only). Links: [arXiv:2608.30614](https://arxiv.org/abs/2608.30614), [DOI 10.18653/2026.emnlp-industry.76](https://doi.org/10.18653/2026.emnlp-industry.76). Read: FT-part (abstract, section 6, cost appendix, keyword extracts).
- Method: a local small LLM extracts actionable segments per document. These are grouped semantically, condensed into concepts, standardised corpus-wide, turned into leaf topics, built bottom-up into a hierarchy and refined iteratively against metrics — [source](https://arxiv.org/abs/2608.30614).
- Evidence: tested on ~194K reviews (5 languages, 20% machine-translated), ~4M complaints and ~167K questions. Average improvements of 11.8, 20.5 and 15.7 points over the best baseline on exclusivity, exhaustivity and granularity. Human ratings 4.4, 4.2 and 4.3 out of 5 (Fleiss' κ 0.68 to 0.74) — [source](https://arxiv.org/abs/2608.30614).
- Relevance: the best cost evidence at millions of documents. Output is a topic taxonomy, not typed relations.

**4. Learning to structure data from user-generated thematic corpora**
- Authors: Elishay Avram, Oren Glickman, Elad Yom-Tov.
- First posted: 2026-10-01. Published: none found. Link: [arXiv:2610.01463](https://arxiv.org/abs/2610.01463). Read: FT-part (abstract, section 3 consolidation and type assignment, section 5).
- Method: iterative attribute discovery with explicit LLM consolidation, type assignment, then value extraction. Four consolidation strategies are compared.
- Evidence: 61% agreement with human attributes, against 62% between annotators. 82% type accuracy across 50 attributes. Value extraction F1 0.8. Sample is 1,000 documents per corpus with k=10 attributes — [source](https://arxiv.org/abs/2610.01463).
- Relevance: gives a direct comparison of consolidation strategies and a usable stopping rule. Small scale, and attributes rather than relations.

**5. CORTEX: High-Quality Cross-Domain Organization of Web-Scale Corpora through Ontological Corpus Graph**
- Authors: Chengtao Gan, Xiaoke Guo, Yushan Zhu, Zhaoyan Gong, Zhiqiang Liu, Songze Li, Huajun Chen, Wen Zhang.
- First posted: 2026-06-29. Published: EMNLP 2026 Main (Crossref year 2026 only). Links: [arXiv:2606.30175](https://arxiv.org/abs/2606.30175), [DOI 10.18653/2026.emnlp-main.985](https://doi.org/10.18653/2026.emnlp-main.985). Read: FT-part (sections 3.2, 3.3, 4.1, 4.2, part of 4.3).
- Method: **not seed-free.** Starts from 196 Wikipedia-derived concept chains across 12 domains. A frontier LLM processes a small document sample one at a time and proposes a new chain only when no existing chain reaches a confidence threshold. Evolution stops after P documents with no proposal, then candidates are filtered by an admission score. Final set: 403 chains — [source](https://arxiv.org/abs/2606.30175).
- Evidence: 73.15M documents preprocessed, 24.14B tokens refined. Validation is indirect, through continual pre-training and a downstream benchmark rather than a gold ontology — [source](https://arxiv.org/abs/2606.30175).
- Relevance: a practical pattern of evolving on a sample, freezing, then assigning everything else by embedding. Instantiated on Chinese web text.

**6. Term-Centric Hierarchy Induction from Heterogeneous Corpora (TERMNET)**
- Authors: Elena Senger, Yuri Campbell, Jan-Peter Bergmann, Rob van der Goot, Barbara Plank.
- First posted: 2026-06-25. Published: none found. Link: [arXiv:2606.26963](https://arxiv.org/abs/2606.26963). Read: FT-part (abstract, sections 5 and 6, keyword extracts).
- Method: maps documents from different sources into shared space through automatic term extraction, then builds a seed-guided hierarchy. **The top two layers are expert-curated**, so it is not fully ontology-free.
- Evidence: over 1M EN and DE documents (publications, patents, funding records). Precision 67.70, intruder accuracy 94.44, source entropy 36.70, expert Structure and Validity both 4.5. For comparison, K-Means on raw text has intruder accuracy 84.30 and entropy 24.50, and K-Means on summaries reaches intruder accuracy 91.25 — [source](https://arxiv.org/abs/2606.26963).
- Relevance: the only paper here built specifically around cross-source style differences and policy analysis use. Term extraction is what closes the source gap in their data.

**7. AutoSchema: Self-Prompted Schema Induction and Evidence-Grounded Extraction for Materials Science Literature**
- Authors: Mingfang Zhu, Yixin Chen, Zhiling Zheng.
- First posted: no arXiv version found. Published: KDD 2026, online 2026-08-08. Link: [DOI 10.1145/3770855.3817812](https://doi.org/10.1145/3770855.3817812). Read: **ABS. Not read** (ACM returned 403; Crossref shows a licence and Semantic Scholar marks it hybrid open access under CC BY).
- Method: from 3 to 5 seed PDFs, expands the corpus by retrieval, induces and refines a schema over rounds, then freezes it for extraction.
- Evidence: across five seed sets, normalisation raises mean pairwise Jaccard from 19.97% to 43.41% and field overlap from 32.50% to 59.91%. 1,656-paper corpus, 200-paper sample — [source](https://doi.org/10.1145/3770855.3817812) (abstract text retrieved via Semantic Scholar).
- Relevance: the clearest published measure of schema instability across runs. Worth obtaining.

**8. AgentsKG: A Hierarchical Multi-Agent Framework for Open-Domain Knowledge Graph Construction**
- Authors: Shilong Liu, Yongqiang Liu, Jiye Liu, Xuan Guo, Tianpeng Li, Qiyao Peng, Wenjun Wang.
- First posted: no arXiv version found. Published: KDD 2026, online 2026-08-08. Link: [DOI 10.1145/3770855.3818058](https://doi.org/10.1145/3770855.3818058). Read: **ABS. Not read** (403).
- Method: verification committee voting, contextual profiling for ambiguous mentions, a linker that merges entities and relations, and a logic auditor. Open-domain, no schema.
- Evidence: the abstract gives no numbers — [source](https://doi.org/10.1145/3770855.3818058).
- Relevance: possible design for merging, but evidence is unknown until the paper is read.

**9. SC-Taxo: Hierarchical Taxonomy Generation under Semantic Consistency Constraints using Large Language Models**
- Authors: Shiqiang Cai, Nianhong Niu, Shizhu He, Kang Liu, Jun Zhao.
- First posted: 2026-05-01. Published: none found. Link: [arXiv:2605.00620](https://arxiv.org/abs/2605.00620). Read: FT-part (abstract, results extracts).
- Evidence: on TaxoBench, Purity 64.8, NMI 50.2, CEDS 28.7 (nearly 5 points above best baseline). A Chinese-corpus benchmark is also reported — [source](https://arxiv.org/abs/2605.00620).
- Relevance: hierarchy labelling method, scientific literature only.

**10. Generative Ontology Induction: Domain-Agnostic Schema Discovery from Document Corpora Using Large Language Models**
- Author: Sergei Sergienko.
- First posted: 2026-05-01. Published: none found. Link: [arXiv:2607.16201](https://arxiv.org/abs/2607.16201). Read: FT-part (abstract, sections 3 and start of 4).
- Method: a single multi-document prompt per document class, sampling documents to fit the token budget. Exports six node types and seven edge types.
- Evidence: Node Coverage Score of 95 to 100%, on four ontologies. **The metric is the author's own and measures generated outputs against the induced schema, not against gold** — [source](https://arxiv.org/abs/2607.16201).
- Relevance: weak evidence and no consolidation step. The "dimension" node type may suit report structure (hypothesis, not tested).

**11. Security and Privacy Taxonomy Generation from Mobile App Reviews (TaxoScale)**
- Authors: Moghis Fereidouni, Krishna Upadhyay, Umar Farooq, A. B. Siddique.
- First posted: 2026-08-10. Published: none found. Link: [arXiv:2608.09049](https://arxiv.org/abs/2608.09049). Read: FT-part (abstract, section IV, Tables IV and V).
- Evidence: over 600K reviews. Unseeded scores: path 0.7980, level 0.7258, coverage 0.9537, novelty 0.1933. Baselines could not process the full label set and were run on a 100K sample. Ward clustering hits 29 GB and 1h 24m at 60,000 points, so they cap sub-clusters at 40,000 — [source](https://arxiv.org/abs/2608.09049).
- Relevance: concrete engineering limits for recursive clustering at scale.

**12. Building a Custom Taxonomy of AI Skills and Tasks from the Ground Up with Job Postings (TaxonomyBuilder)**
- Authors: Stephen Meisenbacher, Peter Norlander.
- First posted: 2026-05-20. Published: CustomNLP4U workshop 2026 (Crossref year only). Links: [arXiv:2605.21029](https://arxiv.org/abs/2605.21029), [DOI 10.18653/v1/2026.customnlp4u-1.11](https://doi.org/10.18653/v1/2026.customnlp4u-1.11). Read: FT-part (abstract, data section, sections 6 and 7).
- Evidence: 32.12M postings (NLx) and 1.54M (USAJOBS, average 5,054 words). Filtering improves domain coverage over unfiltered input. Exact scores are in their Table 1, which I did not transcribe — [source](https://arxiv.org/abs/2605.21029).
- Relevance: long documents and "less data is clearer"; suggests sampling or filtering chunks before induction.

**13. How Well Do LLMs Generate Taxonomies in the SE Domain? A Multi-perspective Evaluation Framework**
- Authors: Sota Nakashima, Yuta Ishimoto, Masanari Kondo, Tao Xiao, Yasutaka Kamei.
- First posted: 2026-08-03. Published: ASE 2026, online 2026-10-07. Links: [arXiv:2608.01592](https://arxiv.org/abs/2608.01592), [DOI 10.1145/3832783.3837551](https://doi.org/10.1145/3832783.3837551). Read: FT-part (abstract, RQ4 tables, keyword extracts).
- Evidence: independent comparison of TnT-LLM and CLIMB. CLIMB is 15 to 40 times faster and 8 to 49 times cheaper; TnT-LLM has higher quality but over-complex output — [source](https://arxiv.org/abs/2608.01592).
- Relevance: independent cost and quality trade-off.

**14. ATLAS: Agentic Taxonomy of Large-Scale Software Ecosystems**
- Authors: Junyi Lu, Mengyao Lyu, Jiahui Wu, Lei Yu, Chengwei Liu, Fengjun Zhang, Li Yang, Chun Zuo, Yang Liu.
- First posted: 2026-06-19. Published: ASE 2026, online 2026-10-07. Links: [arXiv:2606.21597](https://arxiv.org/abs/2606.21597), [DOI 10.1145/3832783.3834361](https://doi.org/10.1145/3832783.3834361). Read: ABS.
- Evidence: 54,387 repositories. Taxonomy Quality F-score 83.13% on a 2,001-repository benchmark, about 73.0% on the full corpus. Classification failures drive revision of split dimensions — [source](https://arxiv.org/abs/2606.21597).
- Relevance: failure-driven refinement loop.

**15. TEMPLAR: agentic induction and evolution of standardized radiology reporting templates from large-scale clinical corpora**
- Authors: Xiaotian Hu, Mingxuan Liu, Zhonghan Wang, Xinfeng Zhang, Yiming Huang, Ziang Wang, Kasidit Anmahaepong, Yijin Li, Yifei Chen, Hongjia Yang, Zihan Li, Qiyuan Tian.
- First posted: 2026-10-04. Published: none found. Link: [arXiv:2610.05247](https://arxiv.org/abs/2610.05247). Read: FT-part (abstract, start of section 4, keyword extracts).
- Method: canonical slots come from dual-view clustering of extracted atoms, then a persistent template is revised against downstream structuring feedback. Uses anatomy graphs as a constraint.
- Evidence: on PETCT (n=49), average template coverage 0.8912 against 0.8374 for the ASTAR baseline. Output JSON truncation is a real failure: with one screened model (Doubao seed-1.6), only 6 of 22 batches (27%) succeeded without retrying, and the merge call was still truncated after 17 retry rounds — [source](https://arxiv.org/abs/2610.05247).
- Relevance: report-structure induction and practical output-length limits, medical only.

**16. Ontology-Free Biomedical Knowledge Graph Induction (OF-Bio-KG), and A Web-Based Pipeline for Ontology-Free Biomedical Knowledge Graph Construction and Exploration**
- Authors: Ahlem Ben Younes, Baha Eddine Kalai, Laila Ben Ayed, Chaima Kachroud (first paper). The second paper's Crossref record lists the same first authors; I did not capture its full list.
- First posted: none found. Published: IEEE COMPSAC 2026 (Crossref 2026-07). Links: [DOI 10.1109/COMPSAC69091.2026.00263](https://doi.org/10.1109/COMPSAC69091.2026.00263), [DOI 10.1109/COMPSAC69091.2026.00287](https://doi.org/10.1109/COMPSAC69091.2026.00287). Read: **ABS. Not read.**
- Method: cluster embeddings, then LLaMA-3 8B labels clusters as concepts and infers relations.
- Evidence: 1,068 EMA leaflets, over 131M tokens, 256,148 leaf nodes, 3,742 concept nodes. **No evaluation yet**: a human review is only proposed — [source](https://doi.org/10.1109/COMPSAC69091.2026.00287).
- Relevance: long regulatory documents and cluster-first design, but quality is unmeasured.

**17. DSE-RE: Dynamic Schema Evolution with LLM for Iterative Relation Extraction**
- Authors: Tongyao Wang, Donghui Shi, Jose Aguilar, Jozef Zurada.
- First posted: none found. Published: Human-Centric Intelligent Systems, 2026-06-09. Link: [DOI 10.1007/s44230-026-00158-1](https://doi.org/10.1007/s44230-026-00158-1). Read: **ABS. Not read** (Springer login redirect, although Crossref shows a licence).
- Evidence: graph-based clustering merges synonymous types. F1 rises about 3% on FOBIE and 6.8% on HSA. The authors state it has **not been evaluated on highly heterogeneous multilingual corpora** — [source](https://doi.org/10.1007/s44230-026-00158-1).
- Relevance: explicit schema-drift control, small data.

**18. Text2Onto-Agent: An LLM Agent-Based End-to-End Automated Ontology Construction Method and a Case Study of Green Building Domain Modeling**
- Authors: Juntao He, Hainan Chen, Hai Yang, Pengyang Li, Zhuang Zheng.
- First posted: none found. Published: J. Construction Engineering and Management, 2026-09-01. Link: [DOI 10.1061/jcemd4.coeng-18479](https://doi.org/10.1061/jcemd4.coeng-18479). Read: **ABS. Not read.**
- Evidence: graph memory replaces long-context prompts. Concept disambiguation F1 0.77, parent-child accuracy 85.22%, sibling accuracy 90.35% — [source](https://doi.org/10.1061/jcemd4.coeng-18479). Code: [github.com/pipiyapi/T2OA](https://github.com/pipiyapi/T2OA) (link from the abstract, not checked).
- Relevance: persistent ontology memory is one answer to drift across chunks.

**19. AOCG-LLM+: automatic ontology construction and generation method**
- Authors: Zhenhai Lu, Jungang Yang, Chao Zhang, Yao Li.
- First posted: none found. Published: Complex & Intelligent Systems, online 2026-07-16. Link: [DOI 10.1007/s40747-026-02372-7](https://doi.org/10.1007/s40747-026-02372-7). Read: **ABS. Not read.**
- Evidence: 92.5% entity precision and 88.3% relation F1. Hierarchy through dynamically adjusted radius clustering. The abstract's evaluation is partly LLM-calibrated — [source](https://doi.org/10.1007/s40747-026-02372-7).

**20. Extended NeOn-GPT: Advancing LLM-Powered Ontology Learning Through Ontology Reuse and Automated Verification**
- Authors: Nadeen Fathallah, Arunav Das, Stefano De Giorgis, Andrea Poltronieri, Peter Haase, Liubov Kovriguina, Albert Meroño-Peñuela, Elena Simperl, Steffen Staab, Alsayed Algergawy.
- First posted: none found. Published: Semantic Web journal, online 2026-06-26. Link: [DOI 10.1177/22104968261453138](https://doi.org/10.1177/22104968261453138). Read: **ABS. Not read** (Sage 403).
- Evidence: four domains, mostly 0.5 to 0.8 entity and triple similarity against gold — [source](https://doi.org/10.1177/22104968261453138).
- Relevance: methodology-led prompting with competency questions and repair. Adds an ontology reuse step.

**21. LightOL: A Lightweight Ontology Learning Framework with Large Language Models**
- Authors: Yongqiang Liu, Binrong Liu, Bowen Dong, Jiawei Chen, Xuewei Li, Wenjun Wang.
- First posted: none found. Published: ICASSP 2026, 2026-05-03. Link: [DOI 10.1109/ICASSP55912.2026.11464515](https://doi.org/10.1109/ICASSP55912.2026.11464515). Read: **META. Not read**; no abstract available in Semantic Scholar.

**22. AgriEvoKG: Empirical Failure Modes and Remedies in LLM-Based Cross-Media Agricultural Knowledge Graph Construction**
- Authors: Xin Zeng, Wang Guo, Yisheng Miao, Huaji Zhu, Huarui Wu.
- First posted: none found. Published: ICCK Transactions on Emerging Topics in AI, 2026-11-02. Link: [DOI 10.62762/tetai.2026.882738](https://doi.org/10.62762/tetai.2026.882738). Read: **ABS. Not read.**
- Evidence: 199,418 entities, 277,053 relations. **Sectors with free relation labels hold 23,727 and 19,139 distinct labels.** 62.5% of relations added from 50 new passages used new relation types. Exact-name fusion merged 6,805 entity names across sectors but only 7 relations — [source](https://doi.org/10.62762/tetai.2026.882738).
- Relevance: the most direct evidence of what schema drift looks like at scale. I could not assess the venue's standing.

**23. EventForge: Optimizable Event Schema Induction with Hybrid RAG+KG Storage**
- Author: Harshil Lodhiya.
- First posted: none found. Published: World Journal of Advanced Research and Reviews, 2026-08-31. Link: [DOI 10.30574/wjarr.2026.31.2.2105](https://doi.org/10.30574/wjarr.2026.31.2.2105). Read: **ABS. Not read.**
- Evidence: 12 PDFs, 393 events, 286 relationships, 497.8 seconds on gpt-4o-mini, 6.2% near-duplicate rejection. No gold evaluation — [source](https://doi.org/10.30574/wjarr.2026.31.2.2105). Weak evidence; I could not assess the journal's review standards.

**24. Overview of methods for automated creation of ontologies for scientific domains**
- Authors: G.V. Filippov, A.V. Rykalin, A.A. Khoroshilov, S.E. Churilov.
- First posted: none found. Published: Modelling and Data Analysis, 2026-06-30. Link: [DOI 10.17759/mda.2026160205](https://doi.org/10.17759/mda.2026160205). Read: **ABS. Not read.**
- Evidence: hybrid method proposal over 3,000 aviation documents, described as preliminary with no metrics in the abstract — [source](https://doi.org/10.17759/mda.2026160205).

**25. Peeling the Matryoshka (PEEL) and 26. SPARROW: taxonomy from a term list, with no corpus**
- PEEL: Eunchan Kim, Inwhee Joe. First posted 2026-10-30. Published EMNLP 2026 Main (Crossref year only). [arXiv:2610.41785](https://arxiv.org/abs/2610.41785), [DOI 10.18653/2026.emnlp-main.1305](https://doi.org/10.18653/2026.emnlp-main.1305). Read: FT-part (abstract, metrics and results extract). Evidence: Graph F1 .885 on wiki-ol and .931 on arXiv-ol (zero-shot transfer), with OLLM at .644 and .633 as a cross-task reference that does more work — [source](https://arxiv.org/abs/2610.41785).
- SPARROW: Yirui Zhang, Yixuan Tang, Yandong Sun, Mong-Li Lee, Anthony Kum Hoe Tung. First posted 2026-09-07. Published EMNLP 2026 Main (Crossref year only). [arXiv:2609.07307](https://arxiv.org/abs/2609.07307), [DOI 10.18653/2026.emnlp-main.1502](https://doi.org/10.18653/2026.emnlp-main.1502). Read: FT-part (abstract, section 5, Tables 1 to 5). Evidence: at 1K nodes with GPT-5, Ancestor F1 0.623 (CCS) and 0.745 (Google). At 10K MeSH concepts, 0.487 against 0.011 and 0.015. SPARROW is weaker than LLMscorer on Edge F1 on these 1K runs — [source](https://arxiv.org/abs/2609.07307).
- Relevance: these start from concepts, so they fit the **second** stage, after types are found.

**27. Generation from text with a partial or supplied starting vocabulary (useful but not fully ontology-free)**
- **LLM-Assisted Ontology Engineering and Construction of a French Legal Knowledge Graph.** Génesis Montenegro, Mokhtar Boumedyen Billami, Catherine Faron, Fabien Gandon, Pierre Monnin. First posted 2026-07-27. SEMANTiCS 2026 posters and demos, per the paper and arXiv journal reference. [arXiv:2607.24551](https://arxiv.org/abs/2607.24551). Read: FT-main. Starts from SEMLEG core classes; induces relations. Evidence in the consolidation section below — [source](https://arxiv.org/abs/2607.24551).
- **LLM-Guided Ontology-Driven Knowledge Graph Construction from Unstructured Text.** Abdelhadi Belfadel, Maxence Gagnant, Joseph Kattan, Sana Tmar. First posted 2026-09-15. IJCKG 2026, per arXiv journal reference. [arXiv:2609.31663](https://arxiv.org/abs/2609.31663). Read: FT-part (PDF, baseline results and schema-description section). French power-grid incident reports, 80 annotated. Best baseline scores: Mistral-Small3.2 24B, Partial F1 0.58, Strict 0.49, relation F1 0.53. The prompts I read target defined entity and relation types — [source](https://arxiv.org/abs/2609.31663).
- **An Agentic Hybrid Top-Down and Bottom-Up Approach to Knowledge Graph Generation.** Emma Jouffroy, Warren Jouanneau, Marc Palyart. First posted 2026-08-07. No venue found. [arXiv:2608.07023](https://arxiv.org/abs/2608.07023). Read: FT-part (section 4, keyword extracts). Wikidata-grounded, multilingual skills graph. 36,037 strings to 13,298 canonical nodes, 79.7% alignment, 19.1% wrong-guess rate. **The final consolidation stage is not yet benchmarked** — [source](https://arxiv.org/abs/2608.07023).
- **APOLO: Automatic Prompt Optimization for Ontology Learning.** Huu Tan Mai, Roman Kochnev, Cuong Xuan Chu, Lukas Lange, Heiko Paulheim, Daria Stepanova. First posted 2026-09-28. ISWC 2026 Posters and Demos, per arXiv comment. [arXiv:2609.34540](https://arxiv.org/abs/2609.34540). Read: FT-part (abstract, Table 1). Trains prompts on text-ontology pairs generated from curated ontologies. Autoregressive with Qwen3.5-27B: DOID axiom P .838 and R .689, but PO only .174 and .159. Single run — [source](https://arxiv.org/abs/2609.34540).
- **Specific Domain Ontology Construction Using Large Language Models.** Vivian Magri Alcaldi Soares, Renata Wassermann. First posted 2026-06-14, but presented at NeLaMKRR@KR 2025 and uses GPT-3.5 and GPT-4, so it is older work. [arXiv:2606.20691](https://arxiv.org/abs/2606.20691). Read: ABS. Hierarchy from an initial concept, no corpus — [source](https://arxiv.org/abs/2606.20691).
- **Utilization of advanced machine learning based on financial ontology induction for adaptive multilingual method in document recognition.** Bingqian Li. Published Scientific Reports, 2026-09-09. [DOI 10.1038/s41598-026-70314-x](https://doi.org/10.1038/s41598-026-70314-x). Read: ABS. Cross-lingual induction under label noise, semi-supervised, no numbers in the abstract, and LLM use not stated — [source](https://doi.org/10.1038/s41598-026-70314-x).

#### B. Just outside the window, clearly important

- **AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora.** Jiaxin Bai and 19 co-authors. First posted **2025-05-29**, published **ACL 2026 (July 2026)**, pp. 20557–20584. [arXiv:2505.23628](https://arxiv.org/abs/2505.23628), [ACL Anthology](https://aclanthology.org/2026.acl-long.942/), DOI 10.18653/v1/2026.acl-long.942. Read: FT-part (abstract, keyword extracts). Reports 92% semantic alignment with human schemas and 95%+ triple precision. Note TRACE-KG's independent run below finds high text leakage — [source](https://arxiv.org/abs/2505.23628).
- **Traceable Context-Enriched Knowledge Graph Generation with Induced Schemas (TRACE-KG).** Mohammad Sadeq Abolhasani, Yang Ba, Yixuan He, Rong Pan. First posted **2026-04-03** (three versions; v1 had a different title). LoG 2026, PMLR 338, per arXiv comment. [arXiv:2604.03496](https://arxiv.org/abs/2604.03496). Read: FT-part (abstract, Tables 1 to 3 and 11, keyword extracts). Induces a schema jointly with the graph on long technical documents. Retrieval accuracy 90.2% against AutoSchemaKG's 95.1%, but entity-text leakage 1.3% against 36.5% — [source](https://arxiv.org/abs/2604.03496).
- **Towards Automated Ontology Generation from Unstructured Text: A Multi-Agent LLM Approach.** Abid Talukder, Maruf Ahmed Mridul, Oshani Seneviratne. First posted **2026-04-25**. [arXiv:2604.23090](https://arxiv.org/abs/2604.23090). Read: FT-part. Overall quality 3.33 and 3.00 against direct generation's 1.33 and 2.33, judged by five LLMs — [source](https://arxiv.org/abs/2604.23090).
- **TextMineX** (humanitarian mine action). arXiv 2509.15098, reported in Findings of EACL 2026. **Rests on a search snippet only; not verified or read.** It uses a newly constructed ontology on 120 humanitarian reports. It is the only humanitarian-sector work my searches surfaced — [search result](https://arxiv.org/html/2509.15098v3).

#### C. Checked and set aside (ontology or existing taxonomy is an input, or the input is not text)

- **ANCHOR**, "Schema-Agnostic Knowledge Graph Construction via Hybrid Ontology Discovery" ([arXiv:2606.01208](https://arxiv.org/abs/2606.01208), first posted 2026-05-31): "discovery" means navigating a **supplied** UCO, STIX or MALOnt schema. ABS. Title is misleading for this question.
- **Better Later Than Sooner** ([arXiv:2605.29168](https://arxiv.org/abs/2605.29168), first posted 2026-05-27; [Findings of EMNLP 2026, DOI 10.18653/2026.findings-emnlp.1867](https://doi.org/10.18653/2026.findings-emnlp.1867), found by Crossref title match). Uses Wikidata's ontology. FT-part. Its post-extraction canonicalisation is still useful: up to 98.4% triple consistency at 59% and 79% of Wikontic's token cost.
- **An Ontology-Guided, Deduplication-Aware Extraction Layer** ([arXiv:2607.28662](https://arxiv.org/abs/2607.28662), first posted 2026-07-22). Curated ontology supplied. ABS. Relevant for its deterministic deduplication: search recall reported up from about 70% to about 95%, with no false merges.
- **Hamilton – Interactive Ontology Learning** ([DOI 10.14778/3827998.3828059](https://doi.org/10.14778/3827998.3828059), PVLDB): from relational databases. ABS.
- **Making order from chaos through semantics** (Journal of Big Data, DOI 10.1186/s40537-026-01451-3, Semantic Scholar metadata only): from SQL databases. ABS.
- **OntoExtend** ([2607.17963](https://arxiv.org/abs/2607.17963)), **ReLTEx** ([2608.10970](https://arxiv.org/abs/2608.10970)), **SCALE** ([2608.07254](https://arxiv.org/abs/2608.07254)): extension of an existing structure. META.
- **ArticleMiner** ([2609.25607](https://arxiv.org/abs/2609.25607)), **SciKGExtract** ([2610.06322](https://arxiv.org/abs/2610.06322), ABS), **BEACON** ([2608.28394](https://arxiv.org/abs/2608.28394), ABS, anchored to MITRE ATT&CK), **CodeGraph** ([2609.29474](https://arxiv.org/abs/2609.29474), ABS, Wikidata-grounded, 167M files): guided by an existing structure.

#### D. Seen by title in Semantic Scholar, not assessed

These rest on Semantic Scholar metadata only. I did not confirm them at the DOI and did not read them, and each may merit a look:

- An LLM-Based Agentic AI System for Automated Construction of Knowledge Taxonomies in Data Science Problem… (L@S, 10.1145/3774398.3811567)
- Automated semantic ontology construction for foresight studies using large language models (10.20535/srit.2308-8893.2026.2.09)
- Large language models for automated Islamic ontology construction and knowledge integration (10.20935/acadai8357)
- LLM-driven automated literature review and taxonomy generation for powder metallurgy (10.1016/j.matdes.2026.116630)
- LLM-Assisted Ontology Engineering for Intangible Cultural Heritage in a Lower-Resourced Language (10.1109/IS71416.2026.11727428)
- Comparing LLM-Based Knowledge Graph Extraction Approaches on Literary Studies in Spanish (NSLP@LREC, 10.63317/5kjis3dy8i7k)
- From Unstructured Procedural Text to Knowledge Graphs: Comparing LLM-Based Extraction Strategies (10.34190/eckm.27.2.4713)
- Unsupervised Construction of Multimodal Knowledge Graphs for the China Telecom (10.1109/ICAIBD69640.2026.11637266)
- Taxonomy Exploration with Large Language Model Reasoning and Tool Calling (10.3103/S0005105526700780)
- Automated Tree Knowledge Graph Construction using Ontology Expansion and Retrieval from Vietnamese History (arXiv 2609.00763)
- From DQI to C-DQI: Automatic Codebook Generation (arXiv 2610.27937)

### Inferences
- Taxonomy-style methods (TaxCE, TERMNET, TaxoScale, CORTEX) have reached millions of documents, while typed-schema methods with gold evaluation (SCION, GrOIL) have not, at least per source. The UN goal needs typed relations, so no single paper covers both needs.
- Inference on venue: peer-reviewed status is verified only for the DOIs shown. "Accepted" comments on arXiv are author statements.

### Gaps
- I did not obtain the full text of AutoSchema (KDD), AgentsKG, DSE-RE, Text2Onto-Agent, AOCG-LLM+, Extended NeOn-GPT, the OF-Bio-KG papers or LightOL. All should be obtainable by the user, and several appear to be open access.
- arXiv first-posted date could not be established for those because no arXiv version turned up. That is not proof none exists.

## How do methods induce classes and hierarchies from chunks, and consolidate across a corpus (duplicate types, drift)?

### Takeaway
The recurring recipe is: extract locally, restrict the candidate vocabulary, cluster by embedding, have an LLM name and merge only inside clusters, then **freeze** the schema before full extraction. Direct evidence on drift across chunks is thin, but the measured instability is large where it is reported.

### Cited Findings
- **Aggregate before calling the LLM.** SCION's main runs do not call the LLM per chunk; chunk evidence becomes a source-level candidate package — [SCION](https://arxiv.org/abs/2607.21610). Its margin over plain extract-then-aggregate is small: +0.0455 Literal, +0.0157 Fuzzy, +0.0150 Continuous and +0.0259 Graph F1 — [SCION](https://arxiv.org/abs/2607.21610).
- **Separate streams, then cluster.** GrOIL clusters class, property and restriction candidates separately, with average-linkage cosine clustering and origin-aware embedding templates, so that a string used as both class and predicate is handled in each role — [GrOIL](https://arxiv.org/abs/2608.22135).
- **Frequency-ordered incremental merging won a direct comparison.** On a held-out corpus: incremental F1 0.32, bottom-up 0.13, global grouping 0.08, pairwise 0.08. Absolute scores are low for all four — [2610.01463](https://arxiv.org/abs/2610.01463).
- **Stopping rule.** The top-k attribute set stabilised in 3 of 5 corpora within 10 iterations (4, 5 and 8); TTC and Diabetes hit the cap of 10 — [2610.01463](https://arxiv.org/abs/2610.01463).
- **Threshold merging within a type signature.** The French legal pipeline merges entity labels per class and predicates per (subject class, object class) pair at cosine 0.7, using the most frequent label as canonical. Mistral: entities 74,035 to 20,827, properties 2,643 to 500. OpenAI: entities 53,804 to 38,339, properties 2,649 to 2,031. Property coverage rises from 82.51% to 96.42% (Mistral) and 81.52% to 85.11% (OpenAI) — [2607.24551](https://arxiv.org/abs/2607.24551).
- **The same thresholds gave very different results by embedding model and LLM**, and two LLMs induced different vocabularies from the same input: 75 against 44 properties, sharing 21 — [2607.24551](https://arxiv.org/abs/2607.24551).
- **Over-merging is a stated risk.** Predicates can hide distinctions in obligations and prohibitions, and some French predicates appeared despite instructions to use English — [2607.24551](https://arxiv.org/abs/2607.24551).
- **Bounded LLM resolution inside neighbourhoods.** TRACE-KG uses HDBSCAN neighbourhoods on multi-field embeddings, at most 10 items per prompt, and stops class resolution after three consecutive runs with no accepted changes — [TRACE-KG](https://arxiv.org/abs/2604.03496).
- **Joint processing reduces duplication.** Over three ISO standards, joint against separate runs: entity classes 124 to 64 (48.4% fewer), relation classes 113 to 62 (45.1% fewer), with 39 of 64 entity classes occurring in two or more documents — [TRACE-KG](https://arxiv.org/abs/2604.03496).
- **Standardisation matters most for overlap.** Removing TaxCE's concept standardisation costs 14.4 exclusivity points; removing refinement costs 7.6 exhaustivity points — [TaxCE](https://arxiv.org/abs/2608.30614).
- **Evolve on a sample, then freeze.** CORTEX adds a chain only when no existing chain reaches a confidence threshold, stops after P documents with no proposal, then applies an admission score that uses hit counts and confidence — [CORTEX](https://arxiv.org/abs/2606.30175). AutoSchema likewise freezes the induced schema for extraction — [AutoSchema](https://doi.org/10.1145/3770855.3817812).
- **Measured instability.** AutoSchema reports only 19.97% mean pairwise Jaccard across five seed sets before normalisation and 43.41% after — [AutoSchema](https://doi.org/10.1145/3770855.3817812).
- **Measured drift when labels are open.** AgriEvoKG: 23,727 and 19,139 distinct relation labels in two sectors, and 62.5% of relations from 50 new passages used new types — [AgriEvoKG](https://doi.org/10.62762/tetai.2026.882738).
- **Saturation.** GrOIL shows decelerating vocabulary growth from 10 to 100 contracts, though on synthetic contracts of one domain, and I saw no asymptote figure in what I read — [GrOIL](https://arxiv.org/abs/2608.22135).
- **Taxonomy merge step.** SPARROW treats block-level relations as constraints rather than truth; without the merge stage Ancestor F1 on CCS drops from 0.623 to 0.465 — [SPARROW](https://arxiv.org/abs/2609.07307).
- **Post-hoc repair evidence conflicts on the surface.** LLMs4OL 2026 teams found post-hoc canonicalisation gave little benefit, whereas retrieval and fine-tuning helped — [overview](https://doi.org/10.52825/ocp.v11i.3728). Against this, French legal, TRACE-KG and TaxCE all report clear value from merging — [2607.24551](https://arxiv.org/abs/2607.24551), [TRACE-KG](https://arxiv.org/abs/2604.03496), [TaxCE](https://arxiv.org/abs/2608.30614).
- **Heterogeneity across sources.** TERMNET's term-based input raised source entropy from 24.50 (K-Means raw text) to 30.00 (K-Means terms), and 36.70 for the full method — [TERMNET](https://arxiv.org/abs/2606.26963).

### Inferences
- The repair conflict is likely a difference in what is measured, not a true contradiction (hypothesis, not tested). LLMs4OL scores per-document match to a **hidden gold vocabulary**, which a repair step cannot know about. The merging papers measure corpus-level compactness or their own coverage. Corpus-level consolidation is simply not what the LLMs4OL score tests.
- For about 13,000 reports: a sensible hypothesis to test is a local small-model pass per chunk, or on a stratified sample of chunks across agencies and languages. Follow with type-signature-grouped embedding clustering, LLM merge decisions inside clusters, a frozen schema, and an admission score using support across **agencies** rather than raw frequency. Each step has support in a different paper, but no paper has tested the combination.
- Term extraction ahead of clustering may help offset differences between agency writing styles (TERMNET's finding on publications, patents and funding records). This is untested on evaluation reports.
- A top-down seed of the types you already named (programmes, interventions, beneficiary groups, findings, recommendations) would make this CORTEX-like rather than strictly ontology-free. That is a choice for you to make, since the brief says no ontology is supplied.

### Gaps
- I found no paper that measures type drift **between chunks of the same long document**. The nearest are cross-document (TRACE-KG) and cross-run (AutoSchema).
- No paper I read measured the effect of noisy PDF parsing on induced schemas.
- Multilingual consolidation is barely tested. TaxCE used machine-translated reviews and translation at extraction; SCOPE includes zh/en labels with bge-m3; TERMNET covers EN and DE; 2608.07023 covers five locales but through Wikidata. None tests French and Spanish against English in natural reports.

## What evaluation do they use, and what are the actual numbers?

### Takeaway
Evaluation is uneven and rarely comparable. Gold schema match is the strongest for schema papers, competency questions are the most useful for task fitness, and many papers rely on LLM judges or their own metrics. The headline LLMs4OL score overstates what a schema-free pipeline can do, because non-taxonomic relations are nearly absent from results.

### Cited Findings
- **LLMs4OL 2026 Flagship leaderboard** (11 teams), Graph Similarity (GS) averaged over exact, fuzzy and semantic matching:
  - Genuineknowledge-KEOL 0.7762, BigDipperAI 0.7559, LaboroAI 0.7474, Semantic-Swingers 0.7436, XuanJi 0.6877, pro-team 0.6553, LIST-CortAIx 0.5326, Volt-IQ 0.5218, ELLMO 0.4958, LettRAGraph 0.4539, VUL337-OL 0.4167 — [overview](https://doi.org/10.52825/ocp.v11i.3728).
  - Best term typing F1 0.6967 (LaboroAI), best taxonomy discovery F1 0.7502 (KEOL), **best non-taxonomic relation F1 0.0308** — [overview](https://doi.org/10.52825/ocp.v11i.3728).
  - 15 of 19 Flagship and Reuse entries scored exactly 0.0000 on non-taxonomic relations, largely by deliberate omission, because the composite score rewards it — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **Supplying a vocabulary helps much more than model size.** Best GS rises from 0.7762 (Flagship) to 0.9806 (Reuse) on the same text distribution. The organisers call vocabulary induction a stronger bottleneck than relation prediction — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **LLMs4OL text is synthetic.** Documents are LLM-written passages verbalising batches of 30 relations from curated ontologies, and scores are computed per sample — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **Taxonomy task is unstable by target.** Best F1 ranges from 0.5437 (Hylex) to 0.7442 (Phyra); ELLMO scores 0.5116 on Archeon and 0.0075 on Biora — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **Team results.** XuanJi, with one GPT-5.4 in-context call: GS 0.7790 semantic and 0.5485 exact — [XuanJi](https://doi.org/10.52825/ocp.v11i.3682). BigDipperAI placed second with a QLoRA-adapted Qwen3-8B, ahead of systems using GPT-5.4 and GPT-5.5 — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **Gold schema graphs.** SCOPE uses Literal, Fuzzy, Continuous and Graph F1, following the OLLM metrics. Released human schemas themselves score Literal 0.5724 and Fuzzy 0.8613, and the authors warn that beating this reference does not mean a better human ontology. Fuzzy threshold 0.45 is described as intentionally permissive. They also warn that memorisation of popular public schemas cannot be fully ruled out — [SCION](https://arxiv.org/abs/2607.21610).
- **Downstream extraction test.** Fixed extractor, varying the schema: macro F1 0.5633 (released reference), 0.6523 (ETA), 0.6800 (SCION-lite), 0.6900 (with fusion) — [SCION](https://arxiv.org/abs/2607.21610).
- **Competency questions.** GrOIL figures are listed in the first section. Those 300 CQs were themselves LLM-generated and LLM-judged — [GrOIL](https://arxiv.org/abs/2608.22135). French legal uses SPARQL CQ tests without a reported score in the section I read — [2607.24551](https://arxiv.org/abs/2607.24551).
- **CQ4OE numbers.** Global term F1 59.1% to 66.5%; mean structural F1 26.7% to 33.7%; generation strategies differ within 3 points; classes recover better than properties, 67.1% against 55.2% — [CQ4OE](https://arxiv.org/abs/2609.26029).
- **Human review.** 61% against 62% inter-annotator agreement on attributes — [2610.01463](https://arxiv.org/abs/2610.01463). TERMNET expert ratings, κ 0.43 — [TERMNET](https://arxiv.org/abs/2606.26963). TaxCE human ratings — [TaxCE](https://arxiv.org/abs/2608.30614). OF-Bio-KG has no result yet — [OF-Bio-KG](https://doi.org/10.1109/COMPSAC69091.2026.00287).
- **LLM judges.** GrOIL reports Krippendorff's α of 0.57 to 0.71 by metric, and cautions that narrow differences on its 1 to 5 scale should not be over-interpreted — [GrOIL](https://arxiv.org/abs/2608.22135). TaxoScale validates its LLM metrics with a human study; those metrics are judged by GPT-4o — [TaxoScale](https://arxiv.org/abs/2608.09049).
- **Self-referencing metrics.** TaxCE's exhaustivity reference set comes from its own extraction stage, though the authors argue this is not circular — [TaxCE](https://arxiv.org/abs/2608.30614). GOI's coverage metric is the author's own — [GOI](https://arxiv.org/abs/2607.16201).
- **Semantic loss test.** Answering from the learned ontology, rather than source text, loses 8.4 to 27.4 points. Averages: LLMs4OL method −21, NeOn-GPT −13, NeOn-CoT −16, on legal merger agreements across six models — [2608.12326](https://arxiv.org/abs/2608.12326). Accepted at KES 2026, per arXiv comment.
- **Model size is a poor predictor.** Term typing F1 of 58.14% (Qwen3.5-27B) against 39.44% (a larger 122B MoE model), and the larger model wins on taxonomy at 23.05% against 12.13%, on MatWerk — [2608.31118](https://arxiv.org/abs/2608.31118).
- **Taxonomy retrieval ceiling.** OntoLearner reports top retrievers at 9 to 37% recall on taxonomy discovery, rising to about 47.9% with LLM-augmented retrieval; term typing is 40 to 83% F1 and non-taxonomic relations 0.2 to 95.7% by domain — [OntoLearner](https://arxiv.org/abs/2607.01977).
- **Independent caution on AutoSchemaKG.** TRACE-KG's comparison reports AutoSchemaKG at 95.1% retrieval accuracy with 36.5% entity-text leakage, meaning much of the text is retained in long entity strings. TRACE-KG is a competitor, so this is not neutral — [TRACE-KG](https://arxiv.org/abs/2604.03496).

### Inferences
- For the UN corpus, gold-standard scoring is not available, so a practical package would be: a human-written competency question set, a task-based semantic loss test using held-out questions, stability across samples (AutoSchema's Jaccard), and human review of merges. Each has precedent above. Because of LLMs4OL's scoring and synthetic text, its leaderboard is weak guidance on how well a system performs on real long reports.
- Non-taxonomic relations are your core interest ("evaluates", "targets", "recommends"), and they are precisely what the main benchmark leaves unmeasured.

### Gaps
- No shared metric lets these papers be compared with each other. SCION, GrOIL, TaxCE and LLMs4OL figures must not be ranked against one another.
- I found no competency-question evaluation done with real domain users on an induced schema from a real corpus at scale.

## Which report scale, cost or LLM calls, and which were tested on long documents or at corpus scale?

### Takeaway
Cost reporting is now common but inconsistent. AutoSchemaKG is the only typed-schema system at far larger scale than your corpus, at a high GPU cost. TaxCE shows a cheap two-tier design at 1M documents, but for topics. Most careful schema evaluations remain at tens of thousands of chunks or fewer.

### Cited Findings
- **AutoSchemaKG.** 9.599M, 7.918M and 35.040M text chunks on three corpora; 14,300, 11,800 and 52,300 GPU hours with Llama-3-8B, about 78,400 in total, processing 1,024-token chunks in batches — [AutoSchemaKG](https://arxiv.org/abs/2505.23628).
- **TaxCE.** 1M documents in under 10 hours on one 24 GB GPU plus a commercial API. About 80% of the time is the per-document stage, and about 140 frontier API calls per taxonomy **regardless of corpus size**. Per-document cost falls 25 to 30% from 100K to 10M documents. Local throughput 15.8 to 40.1 docs per second depending on the model; Qwen3-4B 36.2. The authors note the document-level stage can be expensive for very large corpora, and absolute API spend was not given in the passage I read — [TaxCE](https://arxiv.org/abs/2608.30614).
- **SCION.** Across 24 sources with 38,417 total chunks: average 4 LLM calls, 35.5 seconds and $0.12 per source; LLM-only 2 calls and $0.18 (GPT-5.2) — [SCION](https://arxiv.org/abs/2607.21610). Its table gives average tokens of 1,200 in and 800 out, while another table shows 2,450 and 2,379 for a representative run; I could not reconcile these from what I read, so check the run manifests.
- **TRACE-KG.** 876 LLM calls, 3.60M input and 0.42M output tokens, 2.8 hours for 0.61M characters (83 pages, 339 chunks of 100 to 200 tokens). The authors say this **does not establish scalability** to larger or more heterogeneous corpora — [TRACE-KG](https://arxiv.org/abs/2604.03496).
- **SE taxonomy study.** Hundreds of instances. TnT-LLM $1.09 to $27.34 on average by model, CLIMB $0.13 to $0.96; worst case $36.52 for 483 instances — [2608.01592](https://arxiv.org/abs/2608.01592).
- **SPARROW.** Tokens at 1K concepts: TaxoGPT 9,813, Chain-of-Layer 62,038, SPARROW 62,312, LLMscorer 407,518 — [SPARROW](https://arxiv.org/abs/2609.07307).
- **CORTEX.** 2.61B raw pages to 73.15M documents to 15.13M high-quality; a 0.3B student model handles scoring at scale in place of the teacher LLMs — [CORTEX](https://arxiv.org/abs/2606.30175).
- **French legal.** 6,370 articles, 1,389 sampled for ontology engineering (about 10% per stratum as stated by the authors), and a 5% triple sample per property in 15 batches — [2607.24551](https://arxiv.org/abs/2607.24551).
- **Long document evidence.**
  - TRACE-KG: technical standards, 83 pages across three documents — [TRACE-KG](https://arxiv.org/abs/2604.03496).
  - GrOIL: insurance contracts, synthetic — [GrOIL](https://arxiv.org/abs/2608.22135).
  - TaxonomyBuilder: USAJOBS postings averaging 5,054 words — [TaxonomyBuilder](https://arxiv.org/abs/2605.21029).
  - OF-Bio-KG: 1,068 leaflets, 131M tokens, unevaluated — [OF-Bio-KG](https://doi.org/10.1109/COMPSAC69091.2026.00287).
  - TEMPLAR: peak prompt-plus-output estimates of 38,204 to 43,624 tokens — [TEMPLAR](https://arxiv.org/abs/2610.05247).
- **Practical limits.** TaxoScale's Ward clustering reaches 13.5 GB at 40,000 points and 29 GB at 60,000 — [TaxoScale](https://arxiv.org/abs/2608.09049). In SPARROW, Chain-of-Layer could not complete on Llama 3 8B because prompts exceeded the context window — [SPARROW](https://arxiv.org/abs/2609.07307).

### Inferences
These are my own rough calculations, not reported figures. Your chunk sizes may differ.

- AutoSchemaKG-style full extraction at its ATLAS-Wiki rate: 1.9M chunks × (14,300 GPU h ÷ 9.599M chunks) ≈ 2,830 GPU hours, assuming 1,024-token chunks.
- TRACE-KG-style processing at 876 calls per 339 chunks ≈ 2.6 calls per chunk, or about 4.9M LLM calls for 1.9M chunks. This would need a sampling strategy.
- TaxCE-style local pass: 1.9M chunks at 36.2 per second ≈ 14.6 hours on one GPU, if chunks resemble their documents in length.
- Treating agencies as SCION "sources" (31 sources × about 4 calls) would be very cheap, but pushing 1.9M chunks through an aggregated candidate package is untested and may lose rare types.
- These figures point towards inducing the schema from a sample and using it to guide full extraction (as French legal and AutoSchema KDD do), rather than running open induction on every chunk.

### Gaps
- No paper reports running a typed, gold-evaluated schema induction on heterogeneous long documents at a scale close to 1.9M chunks.
- Dollar costs at scale are mostly absent. TaxCE gives shares, not an absolute spend, in what I read.
- GrOIL's cost and LLM-call count were not in the sections I read.

## Are there new benchmarks, shared tasks and surveys from this window?

### Takeaway
Yes. LLMs4OL 2026 is the main shared task, and its proceedings came out in early November 2026. SCOPE, CQ4OE and OntoLearner are new resources. One peer-reviewed systematic review (Semantic Web journal) is the main survey.

### Cited Findings
- **LLMs4OL 2026** (3rd edition, ISWC 2026, Bari, 25 to 29 October). 13 scored teams: 11 Flagship, 8 Reuse, 5 Taxonomy — [site](https://sites.google.com/view/llms4ol2026/challenge-overview), [preface, DOI 10.52825/ocp.v11i.3738](https://doi.org/10.52825/ocp.v11i.3738).
- **Tasks.** A Flagship (raw text to primitive ontology, with **no ontology given**), B Reuse (partial ontology given), C Taxonomy (types given, unseen domain), D Ontologizer (not evaluated) — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **Data.** 36 source ontologies; 4,303 train and 2,018 test Flagship documents; 2,774 and 1,799 Reuse; 35,667 gold test triples; synthetic text from Qwen3.6-35B-A3B — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **Metric.** GS is the mean of edge F1, neighbourhood Jaccard and ancestor-descendant Jaccard, averaged over exact, fuzzy (0.90) and semantic (0.90) matching. Data repository as cited by the organisers: [github.com/sciknoworg/LLMs4OL-Challenge](https://github.com/sciknoworg/LLMs4OL-Challenge) — [overview](https://doi.org/10.52825/ocp.v11i.3728).
- **Overview paper**: Hamed Babaei Giglou, Sören Auer, Jennifer D'Souza, published 2026-11-02, Open Conference Proceedings vol. 11, [DOI 10.52825/ocp.v11i.3728](https://doi.org/10.52825/ocp.v11i.3728). Read: FT-part (sections 1, 2, Table 2, Flagship part of section 3, section 4).
- **Participant papers** (Open Conference Proceedings vol. 11; Crossref dates; authors and DOIs verified at Crossref; PDFs downloaded from TIB; **unless stated, I read the Crossref or Semantic Scholar abstract and only the introduction or extracts**). No arXiv version unless noted:
  - [Genuineknowledge-KEOL](https://doi.org/10.52825/ocp.v11i.3661), Lvting Liu, Yubin Luo, Ji Qi, Chun Zheng, Chengli Zhong, Yi Zhou, 2026-11-02. Training-free, domain-aware staged pipeline, Claude Opus 4.6 for Task A. **Rank 1.**
  - [BigDipperAI](https://doi.org/10.52825/ocp.v11i.3657), Guojian Zhao, Yuxuan Tang, Zhao Li, 2026-11-03. Introduction read only. Adapted Qwen3-8B.
  - [LaboroAI](https://doi.org/10.52825/ocp.v11i.3656), Xinyi Zhao, Jiachen Xu, 2026-11-03. PDF retrieved, not read; results via overview only.
  - [Semantic-Swingers](https://doi.org/10.52825/ocp.v11i.3681), Matias Vizcaíno, Jagdsh Chand, Andy Symonds, 2026-11-02. Negative results catalogue.
  - [XuanJi](https://doi.org/10.52825/ocp.v11i.3682), Jiaqi Jin, 2026-11-02.
  - [pro-team](https://doi.org/10.52825/ocp.v11i.3663), Shivam Mishra, Dhannu Ram Meena, Muneendra Ojha, Krishna Pratap Singh, Kuldeep Singh, 2026-11-04, also [arXiv:2608.27101](https://arxiv.org/abs/2608.27101), first posted 2026-08-27.
  - [LIST-CortAIx](https://doi.org/10.52825/ocp.v11i.3658), Sondes Bannour Souihi, Sandra Kara, 2026-11-03. Stage-wise analysis of where errors enter the graph; worth reading in full.
  - [Volt-IQ](https://doi.org/10.52825/ocp.v11i.3662), Erban Ku, Ding Nie, 2026-11-04. The challenge site lists Erban Ku alone and a shorter title.
  - [ELLMO](https://doi.org/10.52825/ocp.v11i.3664), Ryn Gray, Portia Cooper, Ena Nayak, Ryan Roche, Brendan Donohoe, Jaimie Murdock, Douglas C. Crowder, 2026-11-02.
  - [LettRAGraph](https://doi.org/10.52825/ocp.v11i.3655), Pasquale Lisena, Julien Plu, Oscar Moreno Escobar, Edouard Trouillez, Raphael Troncy, 2026-11-03. PDF retrieved, not read.
  - [VUL337-OL](https://doi.org/10.52825/ocp.v11i.3665), Chuhan Deng, Yuqi Fan, Can Zeng, Chao Zhang, 2026-11-02.
  - [HybridAI-Squared](https://doi.org/10.52825/ocp.v11i.3660), Upal Bhattacharya, Maaike De Boer, Sergey Sosnovsky, 2026-11-02 (Reuse only).
  - [SEMA](https://doi.org/10.52825/ocp.v11i.3659), Miquel Canal Esteve, Yoan Gutiérrez, José Abreu Salas, Fernando Llopis, 2026-11-04 (Taxonomy only). PDF retrieved, not read.
- **SCOPE**: 24 public IE sources, train-text-only input, gold schema graphs for evaluation only. Code and packages at [github.com/wandugu/paper_scion](https://github.com/wandugu/paper_scion), link from the paper, not checked — [SCION](https://arxiv.org/abs/2607.21610).
- **CQ4OE**: Jiayi Li, Ziyuan Wang, Daniel Garijo, María Poveda-Villalón. First posted 2026-09-22. CQ2Term over 99 CQs and CQ2Onto over 118 CQs, nine LLMs. Input is competency questions, not corpus text. Read: FT-part (abstract, results extracts) — [arXiv:2609.26029](https://arxiv.org/abs/2609.26029).
- **OntoLearner**: Hamed Babaei Giglou, Jennifer D'Souza, Andrei Aioanei, Nandana Mihindukulasooriya, Sören Auer. First posted 2026-07-02. Under review at Nature Communications, per arXiv comment. 180 ontologies, 22 domains, 22 retrieval models, 12 LLMs. MIT licence, per abstract. Read: FT-part (abstract, results extracts) — [arXiv:2607.01977](https://arxiv.org/abs/2607.01977).
- **When Does Bigger Help?**: Hamed Babaei Giglou, Sören Auer, Jennifer D'Souza. First posted 2026-08-31. WOP 2026 at ISWC, per arXiv comment. 13 models on four ontologies. Read: FT-part — [arXiv:2608.31118](https://arxiv.org/abs/2608.31118).
- **On Measuring Semantic Preservation in Legal Ontology Learning**: Albert Sadowski, Jarosław A. Chudziak. First posted 2026-05-30. KES 2026, per arXiv comment. Read: FT-part (abstract, Table 1, Figure 1 caption) — [arXiv:2608.12326](https://arxiv.org/abs/2608.12326).
- **Other new resources:**
  - MeSH-Rel-4K, 4K relations, with fine-tuning raising small-model average F1 by 34.1 points. Tanay Aggarwal, Angelo Salatino, Francesco Osborne, Enrico Motta, first posted 2026-07-20. ABS — [arXiv:2607.17902](https://arxiv.org/abs/2607.17902).
  - MIOE, a causal-abstraction benchmark for entity typing and subsumption. Bohui Zhang, Albert Meroño-Peñuela, Elena Simperl, first posted 2026-10-28. ABS — [arXiv:2610.38326](https://arxiv.org/abs/2610.38326).
  - GrOIL's 100-contract corpus — [GrOIL](https://arxiv.org/abs/2608.22135).
  - TERMNET's EN and DE multi-source benchmark — [TERMNET](https://arxiv.org/abs/2606.26963).
- **Surveys.**
  - **Large Language Models for Ontology Engineering: A Systematic Literature Review.** Jiayi Li, Daniel Garijo, María Poveda-Villalón. Semantic Web journal, online 2026-07-13. 36 papers and 49 task-level studies; finds little standardisation of tasks, datasets and metrics, and limited reproducibility. Read: **ABS**; Sage returned 403. A likely preprint with matching title and authors, dated 2025, is at [semantic-web-journal.net](https://semantic-web-journal.net/system/files/swj4039.pdf); I checked only its first page — [DOI 10.1177/22104968261465514](https://doi.org/10.1177/22104968261465514).
  - **Building Knowledge Graphs for Theses: A Survey.** Donia Ruelas, Elmer Coyla, Miguel Aceituno, Elwin Huaman. First posted 2026-10-05. 46 studies from 2021 to 2025. ABS — [arXiv:2610.28648](https://arxiv.org/abs/2610.28648).
  - **Large Language Models for Health Knowledge Modelling in Data Interoperability: A Scoping Review.** Omid Pournik and co-authors. Published 2026-05-21 (Medical Informatics Europe). 20 of 166 records. ABS — [DOI 10.3233/SHTI260406](https://doi.org/10.3233/SHTI260406).
  - **Enhancing Ontology Engineering with Large Language Models: A Stage-Wise Human-in-the-Loop Study.** Xuefei Ding and co-authors. Published 2026-05-21. An empirical study rather than a survey: human input consistently improves quality, and LLMs do better at early stages than later formalisation. ABS — [DOI 10.3233/SHTI260304](https://doi.org/10.3233/SHTI260304).

### Inferences
- LLMs4OL Flagship is usable as a smoke test, and OntoLearner gives the harness, but SCOPE is closer to your question because it scores corpus-to-schema induction from real IE dataset text.
- A UN-specific gold schema does not exist as far as I found, so a small hand-annotated subset per agency and language would likely be needed.

### Gaps
- **Coverage limits of this search.**
  - OpenAlex returned HTTP 429 on all queries and DBLP was blocked by a bot check, so neither contributed.
  - Google Scholar was not used.
  - I searched arXiv and Semantic Scholar with date filters, Crossref for verification, and web search, and found no ISWC 2026 or EKAW 2026 main-track or ESWC 2026 main-track paper on this topic. That does not show none exists. Those proceedings should be browsed directly.
  - TKDE returned nothing notable, but I did not query it directly.
  - ACL and EMNLP coverage comes through Semantic Scholar and Crossref, not by browsing the Anthology.
- The official LLMs4OL leaderboard page did not expose its tables to my fetch, so the scores come from Table 2 of the overview paper.
- A web search snippet claimed that Executable Schema Contracts ([arXiv:2606.05415](https://arxiv.org/abs/2606.05415), first posted 2026-06-03) recovers 10 to 15% more entity types with a second pass. I could not find this figure in the HTML text I retrieved, so I do not rely on it. That paper does report two LLM calls per source and 0.5 to 2.4 minutes for schema generation, and evaluates through QA rather than schema gold.
- I found no peer-reviewed paper on LLM ontology learning from UN or development evaluation reports. One dedicated web search returned only the mine action work noted above.
