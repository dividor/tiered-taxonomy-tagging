# Mapping text onto a supplied ontology or taxonomy: papers from May to November 2026

Searched on 2026-11-05. Window: first posted or published about May to November 2026. Items outside the window are placed under "Just outside the window" sub-headings.

**How to read each entry.** "First posted" is the earliest public date I could establish (arXiv v1, or the preprint server's posted date). "Published" is the venue or journal date. They are given separately. ACL Anthology gives only a month. An arXiv date is taken from the arXiv listing or abstract page, never inferred from the ID: for example arXiv 2607.20430 has an ID that suggests July, but its abstract page says "[v1] Sun, 10 May 2026". Where a paper's first posting falls before the window but its publication falls inside (BioELX, AstroConcepts, RHC, the XMTC Library Dataset), the entry says so.

**First-posted dates for ACL Anthology papers.** An arXiv title search (from October 2025 onwards) found no preprint for HiCo-EL, DuCEI, SPUR-XMC, SCOTTA, Before You Trust the Count, Profiling Hallucinations, Context-Aware SNOMED CT Entity Linking, the Transferability paper or SelectED, so for these only the publication date is known. A title search can miss a preprint with a different title. Other Anthology papers that give no first-posted date were not looked up, and their first-posted date is unknown.

**Read status codes.**
- **PDF**: I downloaded the PDF, converted it to text, and read the abstract, the results tables and the conclusion. I did not read every line.
- **HTML**: full text read through a summarising fetch tool. Numbers marked "quoted" were then re-requested as verbatim quotes from the page. Others should be checked before being reused.
- **ABS**: abstract only, from a primary record (arXiv, ACL Anthology, Crossref or OpenAlex DOI record).
- **META**: title and metadata only. Not read. The user may wish to obtain these.

**Identifier verification.** Every identifier below was resolved against arXiv, ACL Anthology, the Crossref API or the OpenAlex DOI record, unless the entry says otherwise. No entry rests only on a search snippet unless it says so.

---

## Q1. New methods for LLM-based entity linking and concept normalisation: retrieve-then-rerank, large vocabularies, and abstention (NIL)

### Takeaway
The strongest new pattern is to keep retrieval simple, let an LLM choose among candidates, and call the LLM or other costly step only on uncertain cases: selective designs report their best results on general benchmarks while running that step on only 20 to 28% of mentions. Abstention is now being measured, but it is still weak (best reported recall on "none of the candidates" cases is 45.3%), and several benchmarks contain no true NIL cases at all.

### Cited Findings

#### Retrieval plus LLM selection, with abstention

- **Select, Don't Train: The Benefits of Modular Entity Disambiguation with LLM-Based Selection.** Fina Polat, Daniel Daza, Pengyu Zhang, Klim Zaporojets, Paul Groth. First posted 17 Aug 2026. Accepted to ISWC 2026 Research Track (proceedings page not checked). Read: HTML, abstention row quoted. [arXiv 2608.27470](https://arxiv.org/abs/2608.27470)
  - Method: compares BM25, Wikipedia API search and a trained dense retriever, all feeding the same LLM selector. Abstention is prompt-based: the selector may answer "None of the candidates". There is no separate retrieval-failure detector.
  - Evidence on nine ZELDA datasets: inKB micro-F1 rises from 82.3 (prior best) to 86.3 with training-free BM25 + GPT-5.4-mini, and to 88.5 with a trained dense retriever. Abstention-aware F1 is 90.7. The quoted Table 3 row gives: gold "none of the candidates" 9.6%, model abstains 5.4%, precision 75.0, recall 45.3, F1 53.8. Recall@16: BM25 90.4, dense 95.1 (not quoted verbatim). No token or cost numbers are reported.
  - Relevance: the most direct evidence that a vocabulary can be mapped without training a retriever, which matters when agency vocabularies change. The abstention recall shows the LLM misses more than half of cases where the correct concept was not retrieved.

- **SelectED: Selective LLM Refinement for Entity Disambiguation.** Susanna Rücker, Alexander Reinicke, Alan Akbik. Published October 2026, UncertaiNLP workshop at EMNLP 2026, pp. 111-137. Read: PDF. [ACL Anthology 2026.uncertainlp-main.9](https://aclanthology.org/2026.uncertainlp-main.9/)
  - Method: a dual encoder answers first. The LLM sees only mentions with a small gap between the top two candidates (10 candidates shown).
  - Evidence on ZELDA: accuracy improves by 4.3% with LLM calls on about 20% of mentions, and it beats calling the LLM on every mention. In tuning, the best threshold gave 84.83% accuracy with 19.4% of spans sent to the LLM. Candidate count: k=20 uses 37.6% more tokens than k=2, against 13.1% for k=10, with no accuracy gain (81.42 vs 81.47).
  - Limit stated by the authors: "our benchmarks do not contain genuine NIL cases", and true NIL prediction is left as future work.
  - Relevance: directly usable cost design. The finding that selective use beats full use suggests the LLM also introduces errors on easy cases.

- **DuCEI: When to Introduce Metadata in Cross-Entity Interaction for Entity Disambiguation.** Jinwoo Min, Roseline Nyange, Wonkee Lee, Hyun-Je Song, Seung-Hoon Na. Published October 2026, Findings of EMNLP 2026, pp. 27453-27472. Read: ABS. [ACL Anthology 2026.findings-emnlp.1298](https://aclanthology.org/2026.findings-emnlp.1298/)
  - Method: compares candidates on titles first, and uses detailed metadata only for the remaining candidates. No LLM.
  - Evidence: Confidence-based selective refinement keeps nearly all the two-stage performance while running Stage 2 on 27.80% of mentions, cutting cost from 1.90× to about 1.27× of Stage 1.
  - Relevance: non-LLM confirmation of the "expensive step only when uncertain" design, and a template for adding definitions from a thesaurus only late in the pipeline.

- **Beyond Single-Negative Preference: Multi-Negative DPO for LLM-Centric Historical Entity Linking.** Tien Nam Nguyen, Emanuela Boros, Ahmed Hamdi, Adam Jatowt, Mickael Coustaty, Antoine Doucet. First posted 10 Sep 2026 per arXiv listing. Published October 2026, Findings of EMNLP 2026, pp. 39970-39985. Read: PDF. [ACL Anthology 2026.findings-emnlp.1873](https://aclanthology.org/2026.findings-emnlp.1873/); [arXiv 2609.07379](https://arxiv.org/abs/2609.07379)
  - Method: trains the LLM selector against all retrieved wrong candidates, not one. Evaluates on OCR-noisy historical newspapers in French, German, English, Swedish and Finnish, on hipe-2020 and newseye.
  - Evidence: micro-F1 for Multi-DPO is FR 72.4, DE 65.6, EN 73.8 (hipe-2020) and FR 68.7, DE 59.2, SV 65.0, FI 62.7 (newseye). This compares with supervised fine-tuning at 60.3, 55.0, 46.2 and 44.8, 36.6, 48.2, 38.4. In the rows of the oracle table I read, gold-in-candidate-set recall is 82.6 to 88.5, and perfect selection among retrieved candidates would add +7.7 to +23.3 points, so selection still leaves room as well as retrieval. The abstract claims strong gains on NIL mentions; I did not extract a NIL-specific number.
  - Relevance: the closest match found to "noisy text, several languages, NIL present". It is mention-level, not chunk-level.

#### Cross-lingual and long-context linking

- **BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision.** Yi Wang, Corina Dima, Liangyu Zhong, Steffen Staab. First posted 9 Apr 2026 per the v1 abstract page, just before the window (the ID 2605.27380 suggests May). Published October 2026, Findings of EMNLP 2026, pp. 23462-23488. Read: PDF. [ACL Anthology 2026.findings-emnlp.1109](https://aclanthology.org/2026.findings-emnlp.1109/); [arXiv 2605.27380](https://arxiv.org/abs/2605.27380)
  - Method: retrieve then rerank. Extends SapBERT with Wikidata-derived cross-lingual aliases, then uses a pretrained LLM reranker with the mention marked in the prompt. No task-specific linking annotations are needed.
  - Evidence: Recall@1 gain of 4.8 to 18.2 points over prior best on four cross-lingual benchmarks. See Q4 for per-language numbers. Prompt design mattered more than fine-tuning: on EMEA, a vanilla prompt gave 35.8, a refined prompt 63.2, and refined plus fine-tuning 61.5.
  - Relevance: the alias trick (mining cross-lingual aliases from Wikidata) is transferable to UNBIS or AGROVOC labels, as a hypothesis not tested by the paper.

- **LongBEL: Long-Context and Document-Consistent Biomedical Entity Linking.** Adam Remaki, Xavier Tannier, Christel Gérardin. First posted 13 May 2026. No venue stated. Read: HTML. [arXiv 2605.13451](https://arxiv.org/abs/2605.13451)
  - Method: generative linking using the full document plus a memory of earlier predictions, trained on cross-validated predictions rather than gold labels so that it learns to handle its own errors.
  - Evidence: Recall@1 for the ensemble is 80.0 (English MedMentions), 77.6 (French QUAERO-EMEA), and 63.3, 65.8, 71.0 (three Spanish clinical sets). Sentence-level baseline is 75.0, 73.8, 61.7, 63.2, 68.3. Gains on the French and Spanish sets are often within confidence intervals.
  - Relevance: document-level consistency is the relevant idea for chunked reports. Documents here average only about 1,500 to 5,000 characters, far shorter than UN evaluation reports.

- **Medical Concept Normalization of German Clinical Expressions to SNOMED CT.** Helena Adam, Akhila Abdulnazar, Roland Roller, Stefan Schulz, Markus Kreuzthaler. Published 17 Sep 2026, Studies in Health Technology and Informatics. Read: ABS. [DOI 10.3233/shti260995](https://doi.org/10.3233/shti260995)
  - Evidence: GPT-5.4 alone: Recall@1 0.235. medBERT.de bi-encoder retrieval: 0.681. Retrieval plus LLM reranking: 0.771. Recall@5 is 0.303, 0.812, 0.812.
  - Relevance: the cleanest non-English numbers showing an LLM without retrieval is far worse. Reranking helps at rank 1 and cannot help beyond retrieval's ceiling.

- **Entity Linking for Faroese Using Large Language Models with Web Search.** Annika Simonsen, Iben Nyholm Debess, Hafsteinn Einarsson. Published May 2026, RESOURCEFUL workshop at LREC 2026, pp. 32-43. Read: ABS. [ACL Anthology 2026.resourceful-4.4](https://aclanthology.org/2026.resourceful-4.4/); [DOI](https://doi.org/10.63317/2myt23gboqw3)
  - Evidence: GPT-5 with web search and a language fallback reaches 87.5% precision and 87.3% recall by human evaluation on 1,010 examples. A Wikipedia API baseline reaches F1 0.57 to 0.60.
  - Relevance: fallback across language versions of a knowledge base is a useful pattern for French and Spanish, though web search cannot be applied to a closed thesaurus.

#### Large vocabularies and ontology structure

- **Bridging Lexical Divergence: LLM-Assisted, Cost-Efficient, Zero-shot Scientific Entity Linking (Sci-ZSEL).** Md Rasel Khondokar, Qiao Qiao, Farjana Sultana Samia, Nhat Le, Yuepei Li, Qi Li. First posted 31 Aug 2026. Published October 2026, Findings of EMNLP 2026, pp. 27231-27249. Read: PDF. [ACL Anthology 2026.findings-emnlp.1286](https://aclanthology.org/2026.findings-emnlp.1286/); [arXiv 2609.00228](https://arxiv.org/abs/2609.00228)
  - Method: LLM generates aliases per ontology entity for a selected subset, not per mention, then an ontology-aware filter drops aliases that drift towards neighbouring concepts. The aliases become pseudo-training pairs.
  - Evidence: the filter raises pseudo-pair accuracy from 72.91% to 90.32% (NCBI) and 72.55% to 91.4% (BC5CDR). Evaluated on five benchmarks, including a new animal-science one linked to three livestock trait ontologies.
  - Relevance: high. Agricultural ontologies, zero human labels, and alias sets that can be reused across any corpus linked to the same ontology.

- **HiCo-EL: Encoding Ontology Structure into Bi-Encoder Entity Linking.** Albert Lee, Prasanth Bathala, Puru Malhotra, Christophe Ye, Cassie S. Mitchell. Published October 2026, Findings of EMNLP 2026, pp. 2113-2123. Read: ABS. [ACL Anthology 2026.findings-emnlp.103](https://aclanthology.org/2026.findings-emnlp.103/)
  - Evidence: beats ArboEL by +3.7 and +1.8 Recall@1 on MedMentions and NCBI Disease; deep hierarchies add +9.6 Recall@1 on unseen (zero-shot) entities.

- **OntologyAligner: Ontology-Aligned Retrieval and Hierarchy-Guided LLM Reranking for Biomedical Ontology Normalization.** Jie Song, Zhichuan Xu, Ziyu Lu, Meng Xiao, Cheng Bi, Yuxin Zhang, Xin Zheng, Xiaoran Li, Qiongfang Cao, Hao Yang, Bairong Shen. First posted 9 Sep 2026. Read: ABS. [arXiv 2609.10055](https://arxiv.org/abs/2609.10055). Not to be confused with OntoAligner (Q5).
  - Evidence: 88.78% macro and 86.75% micro Top-1 on PhenoNormBench (13,390 samples from seven HPO datasets), +4.85 and +5.07 points over the strongest baseline. Also applied to MONDO, MEDIC and NCBITaxon.
  - Relevance: "selective hierarchy-guided refinement" is a good model when a thesaurus has broader and narrower terms.

- **BeLink: Biomedical Entity Linking Meets Generative Re-Ranking.** Darya Shlyk, Stefano Montanelli, Lawrence Hunter. First posted 21 May 2026. Accepted to ACM SIGIR 2026. Read: ABS. [arXiv 2605.22501](https://arxiv.org/abs/2605.22501)
  - Evidence: set-wise instruction-tuned reranker improves linking accuracy by 3% to 24% and reduces inference time against the state of the art.

- **BELXTR: Biomedical Entity Linking via Contextualized Token Retrieval.** Samuele Garda, Ulf Leser. First posted 22 Sep 2026. Read: ABS. [arXiv 2609.25859](https://arxiv.org/abs/2609.25859)
  - Evidence: multi-vector embedding model over ten corpora and five knowledge bases; average +5 points recall@1 in half the corpora. It beats an LLM retrieve-and-rerank pipeline on cross-species gene linking.
  - Relevance: an alternative when LLM reranking is too costly at very large scale, the authors' stated case (PubMed-scale).

- **SNO-PI: Label-Scarce SNOMED CT Entity Linking with Pre-trained Encoders and Curated Terminology Knowledge.** Piyawoot Songsiritat. First posted 19 Oct 2026. Read: ABS. [arXiv 2610.26425](https://arxiv.org/abs/2610.26425)
  - Evidence: macro character IoU 0.5387 vs reference system 0.4582. 272 annotated notes cover 6,595 of about 219,000 eligible concepts. Concepts never seen in training remain the main source of error.

- **Context-Aware SNOMED CT Entity Linking for Clinical Text.** Provia Kadusabe, Demian Gholipour Ghalandari, Lauren Cassidy, Jack Boylan, Chris Hokamp. Published May 2026, CLINICALNLP at LREC 2026, pp. 290-299. Read: ABS. [ACL Anthology 2026.clinicalnlp-1.33](https://aclanthology.org/2026.clinicalnlp-1.33/)
  - Evidence: context helps; agentic reranking with a state-of-the-art LLM "only marginally" improves results, and the authors suggest the benchmark may be near its practical ceiling.

- **Profiling Hallucinations in Frontier LLMs for Entity Linking to Medical Ontologies.** Logan Born, Nishant Kambhatla, Uliyana Kubasova, Maryam Siahbani, Andrei Vacariu, Timothy W. O'Connell, Anoop Sarkar. Published May 2026, CLINICALNLP at LREC 2026, pp. 394-413. Read: PDF. [ACL Anthology 2026.clinicalnlp-1.41](https://aclanthology.org/2026.clinicalnlp-1.41/)
  - Evidence: Claude Opus 4.6 end-to-end invents codes in 13.61% of outputs (macro 42.21%). The abstract says constraining models to correct mention spans makes this worse, but the Opus rows I read (13.61% end-to-end, 13.57% with oracle spans) do not by themselves show that, so check before repeating the claim. RAG brings the rate to 0 to 0.003%, but F1 only reaches 18.67 to 22.64 and IoU 11.66 to 14.56. Their taxonomy of errors includes deprecated codes and codes from other ontologies.
  - Caution: most authors are affiliated with the vendor whose system is reported as the comparison (84.45% IoU).
  - Relevance: RAG removes invented codes but does not by itself produce good mappings. Any closed vocabulary should be validated in code.

- **Context-aware LLM extraction and controlled-vocabulary normalization of GEO transcriptomic sample metadata.** Mateusz Szczepaniak, Jonathan Wren. First posted 31 Aug 2026, Research Square. Read: ABS. [DOI 10.21203/rs.3.rs-10844781/v1](https://doi.org/10.21203/rs.3.rs-10844781/v1)
  - Method: local LLM extraction, document-level inference of missing labels, then linking to MeSH and Cellosaurus, with clustering of concepts outside the vocabulary.
  - Evidence on 804,427 samples: linking accuracy 0.985, 0.993, 0.982 where a vocabulary answer exists. An identifier is assigned to 80.69%, 62.91%, 23.13% of labels, vs 31.68%, 35.01%, 7.46% for exact MeSH match.
  - Relevance: large-scale, messy metadata, and a clear handling of out-of-vocabulary concepts. The high accuracy applies only to the subset with a vocabulary answer.

- **LLM-Assisted Clinical Data Harmonization: Combining Automated ETL Generation with Semantic Vocabulary Mapping for OMOP CDM.** Falk Meyer-Eschenbach, Martin Vogel, Philipp Jacob, Severin Kohler, Louis Agha-Mir-Salim, Loreen Ruhm, Christof von Kalle, Elias Grünewald, Oliver Kumpf, Felix Balzer. Published 29 Jun 2026, Studies in Health Technology and Informatics. Read: ABS. [DOI 10.3233/shti260883](https://doi.org/10.3233/shti260883)
  - Evidence: LLM refinement after vector search raised medication precision from 60.8% to 93.9% (n=148) and laboratory precision from 62.0-88.6% to 78.5-96.8% (n=158), judged by clinical experts.

- **DIMAS-OMOP: A Deliberative Intelligence-Based Multi-Agent System for Chinese Medical Text Standardization toward OMOP.** Hanlin Lv, Xiao Wang, Kesong Wu, Lei Li, Lei Wang. Published July 2026, MELLM workshop at ACL 2026, pp. 108-118. Read: ABS. [ACL Anthology 2026.mellm-1.10](https://aclanthology.org/2026.mellm-1.10/)
  - Evidence: 95.2% mapping accuracy on 1,250 reports, +21.8 points over rules and +8.1 over single-AI baselines, 1,200 reports per hour.

- **Extraction of Human Phenotype Ontology (HPO) Concepts from Clinical Notes Utilizing LLMs with Model Context Protocol.** Michael Larsen, Ian M. Campbell, Lori A. Orlando, Peter Robinson, Nephi A. Walton. First posted 25 May 2026, medRxiv. Read: ABS. [DOI 10.64898/2026.05.23.26353963](https://doi.org/10.64898/2026.05.23.26353963)
  - Evidence: four synthetic notes, five LLMs, over 2,000 runs: real-time ontology lookup tool raises mean F1 from 0.46 to 0.72, and cuts mapping errors from 40.9% to 7.8%. Note the small and synthetic test set.

- **Entity Linking in Invoices: Reducing Large Language Model Calls by Using Knowledge-Guided Heuristic Filtering.** Pedro H. Azevedo, Renato Fileto, André W. Zibetti, Simone S. Werner. Published 8 Sep 2026, SBBD 2026. Read: ABS. [DOI 10.5753/sbbd.2026.249286](https://doi.org/10.5753/sbbd.2026.249286)
  - Evidence on noisy Brazilian medicine invoices: filters reduce LLM calls by 49.60% and tokens per call by 42.4%; qwen2.5:7b reaches 84.48% accuracy, over 21% above a filterless baseline.

- **SCODEL: Shared Canonical Ontology Definitions for Food Entity Linking and Ontology-Grounded Recipe Knowledge Graphs.** Nuhu Ibrahim, Rishi Ravikumar, Ayuba Dauda, Fuqi Xu, Robert Stevens, Riza Batista-Navarro. First posted 20 Oct 2026. Read: ABS. [arXiv 2610.27637](https://arxiv.org/abs/2610.27637)
  - Evidence: mentions and concepts learn one shared canonical definition. Hit@1 88.33% on seen concepts and 83.15% on unseen concepts, beating the strongest LLM baseline on each.

- **A Comparative Benchmark of Biomedical Language Models for Concept Normalization from Real-World Text.** Anshul Verma, Abhijay, Manan Vangani, Satyartha Prakash, Kumardeep Chaudhary. First posted 21 Sep 2026, bioRxiv. Read: ABS. [DOI 10.64898/2026.09.19.752843](https://doi.org/10.64898/2026.09.19.752843)
  - Evidence: 15 encoders and 12 LLMs across 12,713 instances. SapBERT Top-5 63.8% for SNOMED CT and 58.0% for MedDRA, rising to 69.4% and 63.2% when an LLM corrects the input text first.
  - Relevance: a measured gain from cleaning text before retrieval, which may apply to noisy PDF parsing (untested on PDF noise).

- **Other new entries, read at abstract level, no numbers in abstract or lower relevance:**
  - PILOT, *Neighborhood-Aware Dual Biomedical Entity Linking*. Yicheng Tao, Jie Liu. First posted 4 Aug 2026. [arXiv 2608.04144](https://arxiv.org/abs/2608.04144)
  - *Training-Free BioNEL-RAG*. Ziyi Chen, Hui Liu, Ziyang Ling, Ronald X. Xu, Mingzhai Sun. First posted 10 Oct 2026. [arXiv 2610.14789](https://arxiv.org/abs/2610.14789)
  - *LELA: An End-to-end LLM-based Entity Linking Framework with Zero-shot Domain Adaptation*. Samy Haffoudhi, Nikola Dobričić, Fabian Suchanek, Nils Holzenberger. First posted 26 May 2026; IJCAI-ECAI 2026 demo per arXiv record. [arXiv 2605.26956](https://arxiv.org/abs/2605.26956)
  - *Improving clinical data standardization in surgical notes through biomedical entity linking with contextual evidence from LLMs*. Chaiho Shin, Dareen Eom, Hahyun You, Kwangsoo Kim, Siun Kim, Hyung-Jin Yoon. Published 3 Oct 2026, BMC Med Inform Decis Mak; +2.7 Recall@1. [DOI](https://doi.org/10.1186/s12911-026-03877-4)
  - *Think Before You Link*. Parinthapat Pengpun, Simran Khanuja, Graham Neubig. First posted 9 Sep 2026; EMNLP 2026 main, pp. 52464-52486. Despite the title it concerns multimodal linking (text and images): +6.9% on MERLIN over five languages, up to +23.3% on rare entities. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2309/); [arXiv 2609.10745](https://arxiv.org/abs/2609.10745)
  - *TELLER: Dual-Path Iterative Preference Optimization for Table Entity Linking*. Yixin Peng, Kehao Li, Stefan Decker. First posted 29 Jul 2026; OM-2026. [arXiv 2607.28680](https://arxiv.org/abs/2607.28680). Title and metadata only.

- **Not read, listed for the user to obtain (META):**
  - *Collective Entity Linking with Large Language Models via Mention-Aware Question Answering*. Nan Wei, Wei-Ning Weng, Xiao-Long Ren, Jin-Yun Fang. J. Computer Science and Technology, July 2026 issue; Crossref online date 5 Nov 2026. [DOI](https://doi.org/10.1007/s11390-026-5822-8)
  - *Quality-aware multi-source data fusion and enhancement for Medical Concept Normalization using LLMs*. Yuhan Zhou, Ruochi Li, Ana Cleveland, Junhua Ding, Kewei Sha, Haihua Chen. J. Biomedical Informatics, October 2026 issue. OpenAlex links an SSRN version. [DOI](https://doi.org/10.1016/j.jbi.2026.105093)

### Just outside the window (Q1)
- *Generative models for clinical entity linking in Spanish*. Elena Zotova, Montse Cuadros, German Rigau. Array, July 2026 issue, but the Crossref record was created 8 Apr 2026. Read: ABS. Generating canonical ICD-10 descriptions and then matching them outperforms retrieve-and-rerank, and encoder-decoder models outperform decoder-only LLaMA. Unusual counter-example. [DOI](https://doi.org/10.1016/j.array.2026.100805)
- *Integrating semantic retrieval and chain-of-thought reasoning in small language models for SNOMED CT normalization*. Pilar López-Úbeda, Teodoro Martín-Noguerol, Antonio Luna. Int. J. Medical Informatics, May 2026 issue; Crossref record created 10 Feb 2026. Not read (META). [DOI](https://doi.org/10.1016/j.ijmedinf.2026.106340)
- *It's All About the Confidence: An Unsupervised Approach for Multilingual Historical Entity Linking using Large Language Models*. Cristian Santini, Marieke van Erp, Mehwish Alam. EACL 2026, March 2026, pp. 3939-3954. Not read beyond title. [ACL Anthology](https://aclanthology.org/2026.eacl-long.184/)
- *A comprehensive survey on medical concept normalization*. Haihua Chen, Yuhan Zhou, Ruochi Li, Aryan Murthy Illa, Ana Cleveland, Junhua Ding. J. Biomedical Informatics, June 2026 issue; OpenAlex date 2 Mar 2026. Not read. [DOI](https://doi.org/10.1016/j.jbi.2026.105005)

### Inferences
- Retrieval recall sets a hard ceiling, and LLM reranking mainly improves rank 1. German SNOMED, MDPO and SelectED all show this shape. For the use case, measuring gold-in-candidate-set recall per vocabulary and per language should come before any LLM tuning.
- Prompted abstention is a useful start but should not be relied on as the only NIL control, since recall was 45.3% in the one paper with direct numbers. A hypothesis, not tested in these papers: combining abstention with a retrieval score threshold could do better. *One Score, Two Decisions* (Q4) warns that the top-two margin may remove the signal needed to detect a missing answer.
- Nearly all this work maps short mentions. Whether it transfers to mapping a whole chunk to a topic is not shown by these papers. Chunk-to-theme mapping is closer to subject indexing (Q2).

### Gaps
- I found no entity linking paper in the window evaluated on UN, development or humanitarian text.
- Very few papers report cost in tokens or money, and none in this set reports the cost of mapping a corpus at the scale of 1.9 million chunks.
- NIL performance is reported on differing definitions, so numbers cannot be compared across papers.

---

## Q2. New work in hierarchical and extreme multi-label classification against large or changing taxonomies, zero-shot or few labels, and cost-saving designs

### Takeaway
No single approach wins. In the best benchmark, supervised extreme classifiers lead on agreement with past human indexing and LLM methods lead on librarians' graded judgements and rare labels, at about 100 times the inference time. Retrieval pre-filtering is the common cost control, but the UN resolutions shared task shows it can drop rare labels, so the candidate pool size has to be tested rather than assumed.

### Cited Findings

#### Benchmarks that directly compare LLMs and supervised classifiers

- **Does generative AI supersede supervised XMLC? A Benchmark Study on Automated Subject Indexing with German Scientific Literature.** Maximilian Kähler, Katja Konermann, Lisa Kluge, Markus Schumacher. First posted 16 Jul 2026 (v2 17 Aug 2026). Submitted to KONVENS 2026. Read: HTML, key rows quoted. [arXiv 2607.14882](https://arxiv.org/abs/2607.14882)
  - Data: German National Library, GND vocabulary of 204,056 entities. Test set of 4,651 documents across 20 subject groups. Trained on up to 951,104 titles or 167,281 full texts.
  - Evidence (quoted rows). XR-Transformer: F1 0.467 titles, 0.398 full text. LLM-Ensemble: F1 0.382 and 0.364. Graded relevance by subject experts (g-F1 at k=5): XR-Transformer 0.530 and 0.512; LLM-Ensemble 0.576 and 0.601. On the long-tail weighted cps-F1 for full text, LLM-Ensemble 0.362 vs XR-Transformer 0.339. Inference time per record: XR-Transformer 10.5 ms; LLM-Ensemble 998 ms; DiSMEC++ and Omikuji under 1 ms.
  - Relevance: the most useful single paper for the design decision. It shows the answer depends on whether past human labels are treated as truth, which matters because UN publisher vocabularies are inconsistent.

- **On the Transferability Between Extreme Multi-Label and Hierarchical Text Classification.** Florian Hauss, Tom Speier, Nerijus Bertalis, Paul Granse, Ferhat Gül, Leon Menkel, David Schüler, Lukas Galke Poech, Ansgar Scherp. Published 18 Aug 2026, ACM DocEng 2026. Read: ABS via OpenAlex; ACM page returned 403. [DOI 10.1145/3820755.3832808](https://doi.org/10.1145/3820755.3832808)
  - Evidence: extreme classification methods transfer well to hierarchical datasets and are often competitive; hierarchical methods struggle at extreme scale. No numbers in the abstract.

- **AstroConcepts: A Large-Scale Multi-Label Classification Corpus for Astrophysics.** Atilla Kaan Alkan, Felix Grezes, Sergi Blanco-Cuaresma, Jennifer Lynn Bartlett, Daniel Chivvis, Anna Kelbert, Kelly Lockhart, Alberto Accomazzi. First posted 2 Apr 2026, just before the window. Published May 2026, NSLP workshop at LREC 2026, pp. 1-12. Read: PDF. [ACL Anthology 2026.nslp-1.1](https://aclanthology.org/2026.nslp-1.1/); [arXiv 2604.02156](https://arxiv.org/abs/2604.02156)
  - Data: 21,702 abstracts, 2,367 Unified Astronomy Thesaurus concepts, 76% of concepts with under 50 examples.
  - Evidence: zero-shot, vocabulary-constrained DeepSeek F1 0.3770, vs astroBERT 0.3243, SciBERT 0.2127, k-NN 0.2017. On rare labels (tail F1), DeepSeek 0.198 vs astroBERT 0.081.
  - Relevance: shows thesaurus-constrained LLMs handle the long tail best here, and supports reporting head, middle and tail separately.

- **Overview of the UZH Shared Task 2026 on Reconstructing the Reasoning in United Nations Resolutions.** See Q3. On retrieval pre-filtering: 20 candidates gave micro-F1 0.226, 40 candidates with per-dimension caps gave 0.396, and quoting evidence for each tag without retrieval pre-filtering gave 0.459.

#### Reranking with the thesaurus and hybrids

- **Reading the Neighborhood: Text-Attentive Graph Reranking for Automated Subject Indexing (TAGHN).** Lakshmi Rajendram Bashyam, Ralf Krestel (Crossref lists only the first author). First posted 14 Oct 2026. Published 13 Oct 2026, JCDL 2026 proceedings per Crossref. Read: ABS. [arXiv 2610.20048](https://arxiv.org/abs/2610.20048); [DOI](https://doi.org/10.1145/3805696.3846044)
  - Evidence on STW economics thesaurus (6,198 descriptors): macro-F1 0.380, vs fine-tuned cross-encoder 0.346 and fixed-vocabulary classifier 0.278. It can select a narrower gold term where an unstructured reranker settles for the broader parent.
  - Relevance: STW is one of the closest vocabularies to UN policy subject matter. Note: this reranker uses no LLM per the abstract.

- **Automated subject indexing in large-scale library collections: a minimal semantic enhancement strategy at the National Library of Spain.** Gökhan Usta. Published 26 Aug 2026, Library Hi Tech. Read: ABS. [DOI 10.1108/lht-10-2025-0269](https://doi.org/10.1108/lht-10-2025-0269)
  - Evidence on 2.4 million Spanish records: blending Omikuji (97%) with Qwen3 embeddings (3%) gains 7.3% nDCG@5 overall, +24.9% on medium and +40.1% on long titles.
  - Relevance: a large-scale, Spanish, cheap design. The odd 97/3 weight deserves independent checking.

- **A two-stage hybrid intelligence framework for subject indexing via semantic embedding and LLM collaborative optimization (R3).** Tian Xia, Xin Yang, Wenjing Wu, Ying-Hsang Liu, Junzhi Jia. Published 15 Jul 2026, Journal of Information Science. Read: ABS. [DOI 10.1177/01655515261449401](https://doi.org/10.1177/01655515261449401)
  - Method: find similar documents by cross-lingual embeddings, take their subjects as candidates, then have an LLM refine. No model fine-tuning.
  - Evidence on TIBKAT: MAP from 41.19% to 45.24%. Recall 65.68% (core) and 58.56% (all) in LLMs4Subjects.
  - Relevance: finding candidates through similar documents is attractive given that about a quarter of reports already have tags, though those tags may carry their publisher's biases (my inference).

- **Retrieval-Augmented Classification of Environmental Mitigations in Hydropower Licensing Documents.** Hong-Jun Yoon, Tom Ruggles, Joanna Lee, Debjani Singh. First posted 24 Aug 2026. Read: ABS via summarising fetch. [arXiv 2608.23241](https://arxiv.org/abs/2608.23241)
  - Evidence on 5,860 paragraphs, 135 categories, with 40 having no training examples: micro-F1 hybrid 0.524, BERT only 0.477, RAG only 0.416. BERT scores F1 0 on unseen classes.
  - Relevance: the closest structural match found: long government documents, paragraph-level, a mid-size taxonomy, patchy labels, category definitions used at inference.

- **Constraint-Aware Hierarchical Search for Regulation-Driven Fine-Grained Classification.** Siyu Wang, Wei Tan, Lulu Chen. First posted 12 Jul 2026. Read: ABS. [arXiv 2607.10588](https://arxiv.org/abs/2607.10588)
  - Method: turns regulatory hierarchies into a searchable tree and moves one level at a time using definitions and exclusions. Reports best mean accuracy on four new datasets; no numbers in the portion read.
  - Relevance: useful for DAC purpose codes, which have explicit inclusion rules (my inference).

#### Zero-shot, changing taxonomies, and few labels

- **Thakaa at AraGenre 2026: Definition-Guided LLM Judging with Full-Taxonomy Agreement-Gated Correction for Hierarchical Arabic Genre Classification.** Hassan Alqaeri, Meshal Alamr, Abdullah Aldahlawi. Published October 2026, Arabic NLP shared tasks, pp. 300-312. Read: ABS. [ACL Anthology 2026.arabicnlp-sharedtasks.38](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/)
  - Evidence: first of 18 teams, hierarchical macro-F1 0.7352 over 27,972 segments, 6 broad and 74 specific labels with specific labels disjoint from training, given only as English definitions. Edits are made only where two models agree against the first pass. Authors disclose 74 hand-written cues and leaderboard-based selection.
  - Relevance: showing models the whole taxonomy reduced "topic traps" where subject displaced genre, a useful counterweight to always narrowing candidates.

- **Mosaic: Automated Intent Description Generation for Multi-Label Classification.** Ajay Chatterjee, Eric Crestan, Raunak Saha, Rotem Mordoch, Swarnangsu Acharyya. Published October 2026, EMNLP 2026 Industry, pp. 545-556. Read: ABS. [ACL Anthology 2026.emnlp-industry.33](https://aclanthology.org/2026.emnlp-industry.33/)
  - Evidence: generates label descriptions for an existing 140-class taxonomy from unlabelled text; approaches experts' quality; reduces revision from 2-4 weeks to under 30 minutes.
  - Relevance: agency theme lists often have only names, not definitions. Generating definitions may be a cheap first step.

- **Reasoning for Hierarchical Text Classification: The Case of Patents (RHC).** Lekang Jiang, Wenjun Sun, Stefan Goetz. First posted 8 Oct 2025, well before the window. Published July 2026, Findings of ACL 2026, pp. 11127-11142. Read: ABS. [ACL Anthology 2026.findings-acl.541](https://aclanthology.org/2026.findings-acl.541/); [arXiv 2510.07167](https://arxiv.org/abs/2510.07167). About +3% in accuracy and macro-F1 over supervised fine-tuning; needs training.

- **Structured Confidence-Guided Online Adaptation for LLM-based Multi-Label Classification (SCOTTA).** Pengyu Xu, JingRen Hou, Liping Jing, Jian Yu. Published July 2026, Findings of ACL 2026, pp. 16671-16686. Read: ABS. [ACL Anthology 2026.findings-acl.822](https://aclanthology.org/2026.findings-acl.822/). Measures confidence per label, keeps a cache of in-context examples that balances label coverage, no parameter updates. Best gains on rare labels; no numbers in abstract.

- **RelaClass: A Residual Tree with LLM-Augmented Prototypes for Hierarchical Text Classification.** Wenting He, Qiang Yan, Bing Li, Huaming Liao, Jiafeng Guo, Xueqi Cheng. Published June 2026, IJCNN 2026. Read: ABS. [DOI 10.1109/ijcnn69854.2026.11730021](https://doi.org/10.1109/ijcnn69854.2026.11730021). Uses only class names as supervision.

- **TEAMMix.** Jian Zhang, Zhuohao Yang, Songlin Lei, Bangli Liu, Ziwei Wang, Xufeng Weng, Gehan Amaratunga, Yu Lin, Hongwei Wang. First posted 11 Aug 2026, though Crossref dates the IEEE CSCWD 2026 paper 13 May 2026. Read: ABS. [arXiv 2608.11044](https://arxiv.org/abs/2608.11044); [DOI](https://doi.org/10.1109/cscwd68734.2026.11582679). Weakly supervised, LLM-generated pseudo-samples for rare labels.

- **Push and Pull: Training Sentence Encoders with Contrastive Losses for Distance-Based Multi-Label Text Classification.** Jens Van Nooten, Andriy Kosar. Published May 2026, LREC 2026, pp. 7359-7379. Read: ABS. [ACL Anthology 2026.lrec-1.583](https://aclanthology.org/2026.lrec-1.583/). Label-to-text similarity improved by contrastive training on six datasets including one Dutch; embedding-only and therefore cheap.

#### Cost-saving designs: cheap model first, or teacher and student

- **Multi-Label Topic Assignment via LLM Distillation: A Comparative Analysis of Generative vs. Discriminative Student Models.** Sourabh Kasliwal, Shubhranshu Singh. First posted 6 Oct 2026. Read: ABS via summarising fetch. [arXiv 2610.09063](https://arxiv.org/abs/2610.09063)
  - Evidence: LLM labels train small students. With 112 topics, discriminative baselines lose 35% Macro-F1, while a 1B generative student beats them on long multi-turn conversations.
  - Relevance: one route to labelling 1.9 million chunks: label a sample with a strong LLM, then train a small model.

- **Circuits of LLM Classifiers.** Hadi Khalaf, Sajani Vithana, Kunal Talreja, Haewon Jeong, Viveck R. Cadambe, Flavio Calmon. First posted 21 Oct 2026. Read: ABS. [arXiv 2610.30076](https://arxiv.org/abs/2610.30076). Treats LLM classifiers as noisy instruments and organises them as an adaptive sequence of queries to maximise accuracy at a given cost. Improves the cost-accuracy frontier over routing and cascading baselines on seven tasks. No figures in the abstract. The most formal treatment of cheap-model-first I found.

- **Jev-as-a-Judge for RAG Claim Verification: Confidence-Gated Cascading Between a Non-Generative Verifier and a Frontier LLM.** Alden Do Rosario, Kirollos Atef. First posted 30 Sep 2026 per arXiv listing. Read: ABS. [arXiv 2610.20634](https://arxiv.org/abs/2610.20634). A different task (claim verification) but a useful caution: on 495 claims the two systems tie on the usual metric at a 187-fold cost difference, yet disagree on 15.2%, and the cheaper one admits 23.2% of unsupported claims against 14.1%. A tie on one aggregate score does not show that a cheap model is an equal.

- **Student-Guided Teacher Distillation for Efficient LLM Task Routing.** Haifeng Wu, Srinivasan Manoharan, Jian Wan, Fangbo Tu, Junhua Zhao, Xin Chen. First posted 1 Oct 2026. Read: ABS. [arXiv 2610.02516](https://arxiv.org/abs/2610.02516). 60-category taxonomy: student narrows candidates for a larger reranker; 77.5% agreement on a small 200-example set, Coverage@16 of 91 to 100%. Warns that truncated top-k scores should not be used as soft targets.

- **SelectED, DuCEI, invoice filtering, DRAMA.** See Q1 and Q5: LLM or second stage used on about 20%, 27.8%, half, and 9 to 20% of candidates respectively.

- **LLMs4OL 2026 Overview.** Hamed Babaei Giglou, Sören Auer, Jennifer D'Souza. Published 2 Nov 2026, Open Conference Proceedings. Read: ABS. [DOI 10.52825/ocp.v11i.3728](https://doi.org/10.52825/ocp.v11i.3728). Constraining the label space through a supplied vocabulary mattered more than model scale; retrieval grounding and adaptation of open-weight models transferred better than correction after the fact. Companion paper: **semantic-swingers at LLMs4OL 2026** (Matias Vizcaíno, Jagdsh Chand, Andy Symonds; [DOI](https://doi.org/10.52825/ocp.v11i.3681)): all after-the-fact repairs failed; the vocabulary must be supplied at generation time; a fine-tuned open 9B model beat a stronger paid API on unseen extraction. Read: ABS. No numeric scores in abstract.

#### Medical coding as a large-taxonomy analogue

- **KREL: Automatic Medical Coding via Knowledge-Guided Reasoning over Clinical Evidence with LLMs.** Xubin Chen, Yipeng Zhou, Wen Sun, Chengkai Huang, Xiaoming Fu, Quan Z. Sheng. First posted 21 Aug 2026; accepted to Findings of EMNLP 2026. Read: ABS. [arXiv 2608.20887](https://arxiv.org/abs/2608.20887). Handles long notes and adds coding guidelines; no numbers in abstract.
- **Can Post-Training Turn LLMs into Good Medical Coders?** Ziqing Wang, Weihao Li, Shijie Chen, Yuan Luo, Kaize Ding. First posted 11 Jun 2026. Read: ABS via summarising fetch. [arXiv 2606.13940](https://arxiv.org/abs/2606.13940). Prompt-only evaluation greatly underestimates LLMs; supervised fine-tuning gives the largest gain.

#### Lower relevance or not read

- *LGSA*. Shuai Zhang, Weibo Xu, Jiahao Nie, Kecheng Huang. ACL 2026 Long, label geometry for hierarchical classification, needs supervised training. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.acl-long.1322/)
- *SPUR-XMC*. Ratri Mukherjee, Shailesh Dahal, Kishlay Jha. Findings of EMNLP 2026, tail labels in biomedical literature. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1597/)
- *HASTE: Hardware-Aware Dynamic Sparse Training for Large Output Spaces*. Nasib Ullah, Jinbin Zhang, Jean Lucien Randrianantenaina, Erik Schultheis, Rohit Babbar. First posted 31 May 2026; ICML 2026 per arXiv comment. Up to 4.4× forward and 25× backward speedups. Read: ABS. [arXiv 2606.01117](https://arxiv.org/abs/2606.01117)
- *A Skill-Based AI Agentic Pipeline for Library of Congress Subject Indexing*. Eric H. C. Chow. First posted 5 May 2026. Only ten titles evaluated. Read: ABS. [arXiv 2605.03537](https://arxiv.org/abs/2605.03537)
- *Research Entity Extraction and Topic Detection from UKRI Grant Proposals*. Xingran Ruan, Angelo Salatino, Rosa Filgueira, Kara Moraw, Alexandru Marcoci, Gemma Derrick, Sarah Callaghan. First posted 29 Jun 2026. Mistral mapped to OpenAlex Topics: 90.5% on only 42 proposals. Read: ABS. [arXiv 2606.30304](https://arxiv.org/abs/2606.30304)
- **Not read (META):** *On the Influence of Hyperparameters in Tree-Based Linear Methods for Extreme Multi-Label Text Classification* (Kuan-Ting Chen, Hung-Chih Chiang, Chih-Jen Lin; CIKM 2026, Crossref 6 Nov 2026) [DOI](https://doi.org/10.1145/3799682.3840104); *Multi-label text classification with category-constrained prompting and reflection mechanisms* (Wandong Xue, Weiyan Zhang, Jianjun Zeng, Tong Ruan; Applied Soft Computing, October 2026 issue) [DOI](https://doi.org/10.1016/j.asoc.2026.115777); *A large language model-augmented hierarchical text classification approach for intelligent failure data management* (Yi Ding, Feng Zhu, He Li, Min Xie; EAAI, October 2026 issue) [DOI](https://doi.org/10.1016/j.engappai.2026.115750); *SIAGXC* (Zihao Yin, Zhihai Wang, Xiaokang Wang, Haiyang Liu, Pengfei Yu, Shijiang Li; Applied Intelligence, August 2026) [DOI](https://doi.org/10.1007/s10489-026-07400-0); *Overcoming data sparsity in eXtreme Multi-Label classification using word embedding-based models* (Purvi Prajapati, Shruti Patil, Amit Thakkar, Nirav Bhatt, Nikita Bhatt; Discover Computing, 31 May 2026) [DOI](https://doi.org/10.1007/s10791-026-10158-1).

### Just outside the window (Q2)
- *An Extreme Multi-label Text Classification (XMTC) Library Dataset*. Jennifer D'Souza, Sameer Sadruddin, Maximilian Kaehler, Andrea Salfinger, Luca Zaccagna, Francesca Incitti, Lauro Snidaro, Osma Suominen. First posted 11 Mar 2026 (arXiv v1). Published May 2026 at LREC, pp. 169-184. So publication is in the window, first posting is not. English and German, 136,569 records with GND labels. Read: PDF sections. [ACL Anthology 2026.lrec-1.12](https://aclanthology.org/2026.lrec-1.12/); [arXiv 2603.10876](https://arxiv.org/abs/2603.10876)
- *Hierarchical Text Classification with LLM-Refined Taxonomies*. Jonas Golde, Nicolaas Paul Jedema, RaviKiran Krishnan, Phong Le. EACL 2026, March 2026, pp. 214-228. Title only. [ACL Anthology](https://aclanthology.org/2026.eacl-long.10/)
- *CrisisSense-LLM*. Kai Yin, Bo Li, Chengkai Liu, Ali Mostafavi, Xia Hu. In Advanced Engineering Informatics, October 2026 issue, but the preprint dates from 2024 per the arXiv ID 2406.15477 shown by OpenAlex. Multi-label classification of disaster social media. Not read. [DOI](https://doi.org/10.1016/j.aei.2026.104836)
- The GermEval 2025 edition of LLMs4Subjects was the second edition. I found no 2026 edition, at SemEval or GermEval. [Task site](https://sites.google.com/view/llms4subjects-germeval)

### Inferences
- Rough arithmetic of my own: at the benchmark's rate of 998 ms per record, 1.9 million chunks would take about 22 days of sequential inference on their setup, compared with about 5.5 hours at 10.5 ms. The hardware differs by method, so this is only an order of magnitude. It supports running a cheap classifier or embedding step first.
- Candidate narrowing has a measured cost and benefit in the UZH task. Twenty candidates was too few for 141 tags. The evidence favours testing several pool sizes, including rare-label recall, for each vocabulary.
- A mid-size taxonomy (SDGs with 17 goals, DAC sectors) may not need retrieval at all, since the whole list fits in a prompt. Thakaa's result is consistent with this but tests only one task. Fallatah (Q5) shows the opposite risk: showing a full list with no real "none" path forces a choice.

### Gaps
- No paper in this set measures a chunk-then-aggregate design, that is, classifying chunks and rolling up to a report, against a document-level gold standard. This is the central design question for the use case.
- SIGIR 2026 and CIKM 2026 were not enumerated in full, as ACM pages returned 403 and DBLP showed a bot check. Those venues are covered only through arXiv and Crossref.

---

## Q3. New work specifically on SDG tagging of development, policy or humanitarian text, and on multilingual mapping

### Takeaway
New SDG work is mostly about scholarly or news text, not development reports, and it repeatedly warns that SDG labels depend on the measuring tool: LLMs and keyword systems disagree heavily at the document level. The one in-window task on real UN text (UNESCO resolutions, French and English) shows tagging on a 141-label education taxonomy is still hard, with best micro-F1 0.459.

### Cited Findings

#### UN and policy text

- **Overview of the UZH Shared Task 2026 on Reconstructing the Reasoning in United Nations Resolutions.** Anastassia Shaitarova, Yingqiang Gao, Fatma-Zohra Rezkellah, Reto Gubelmann, Patrick Montjouridès. Published July 2026 (workshop date 3 Jul 2026), ArgMining at ACL 2026, pp. 107-118. Read: PDF. [ACL Anthology 2026.argmining-1.10](https://aclanthology.org/2026.argmining-1.10/)
  - Task: paragraph type, and tags from a fixed taxonomy of 141 tags in 15 education policy dimensions, on UNESCO conference resolutions (1934-2008) in French with machine English translation. Models restricted to open weights, at most 8B parameters, so these scores reflect small models. 2,695 unlabelled UN resolutions for training; 45 test documents.
  - Evidence: paragraph type is largely solved by rules (best macro-F1 0.936). The silver type labels themselves agree about 97% with two human annotators on 715 paragraphs. Tag micro-F1: POINTERS 0.459 (macro 0.357), which requires quoting a phrase for each tag and does no retrieval pre-filtering. LLM-Instruct 0.396 (40 candidates). TypeCoT 0.280, macro 0.278, using 15 calls per paragraph. Prompteam 0.226 (20 candidates).
  - Label quality: tag gold data is silver: the intersection of GPT-5.4 and Claude Opus 4.6, with expert approval on 178 paragraphs. The expert found 20 of those paragraphs have genuine gaps in the taxonomy, so a perfect system cannot reach 1.0. F1 and LLM-judge rankings diverge substantially (Prompteam: 5th on F1 and 1st on the judge). Most teams relied mainly on English translations, using French only in a supporting role.
  - Relevance: highest in the set for domain. It also demonstrates a taxonomy-gap rate (about 11% of reviewed paragraphs) that should be expected with agency vocabularies. Only 178 paragraphs were human-reviewed, so numbers depend on silver labels.

- **LLM-INSTRUCT at UZH Shared Task 2026: Constraint-Aware Retrieval and Selective Debate for Paragraph-Level Argument Mining.** Phuong Huu Vu Tran, Long Minh Vo, Son Nguyen Minh Le, Hoang Van. First posted 10 May 2026 (quoted from abstract page; the ID 2607.20430 is misleading). Published July 2026, pp. 119-124. Read: ABS via summarising fetch. [arXiv 2607.20430](https://arxiv.org/abs/2607.20430); [ACL Anthology](https://aclanthology.org/2026.argmining-1.11/). First overall. Metadata-aware retrieval, constrained decoding, debate for uncertain cases. Development micro-F1 for tags rose from 35.83% to 40.08% through configuration search.

- **RESOLVENOW at UZH Shared Task 2026.** Vedant Gupta, Rahul Bhatia, Vaibhav Varshney, Manjunatha Naik. Published July 2026, pp. 125-128. Read: ABS. [ACL Anthology 2026.argmining-1.12](https://aclanthology.org/2026.argmining-1.12/). A practitioner's account: an embedding-retrieval system "dropped relevant tags before the LLM saw them"; per-dimension prompting was accurate but too slow; the final system uses a single prompt with the full taxonomy and an eight-point checklist. Deterministic French-English rules reached 0.910 on paragraph type.

- **Other teams in the same task, titles and metadata only:** Argchestrators ([2026.argmining-1.13](https://aclanthology.org/2026.argmining-1.13/)), Prompteam ([2026.argmining-1.14](https://aclanthology.org/2026.argmining-1.14/)), HybridArguer, on bilingual UN resolutions ([2026.argmining-1.17](https://aclanthology.org/2026.argmining-1.17/)).

- **Automated Analysis of Global AI Safety Initiatives: A Taxonomy-Driven LLM Approach.** Takayuki Semitsu, Naoto Kiribuchi, Kengo Zenitani. Published May 2026, PoliticalNLP at LREC 2026, pp. 284-301. Read: ABS. [ACL Anthology 2026.politicalnlp-1.30](https://aclanthology.org/2026.politicalnlp-1.30/). Maps policy documents to a shared taxonomy. Model choice substantially changes results, and model scores still differ from three experts.

- **Ontology-Guided Query Expansion for Evidence Retrieval in ESRS E1 Climate Disclosures.** Tatsuya Ishisaka. Published October 2026, ClimateNLP at EMNLP 2026, pp. 116-123. Read: ABS. [ACL Anthology 2026.climatenlp-1.8](https://aclanthology.org/2026.climatenlp-1.8/). A lightweight ontology built from a reporting standard improves page-level retrieval on corporate reports: Recall@10 0.842 vs BM25 0.702, with only 114 gold labels. Useful as an inverse framing: for each concept, find the chunks.

#### SDG classification

- **Can Large Language Models Replace Keyword-Based Mappings for SDG Classification? A Zero-Shot Evaluation Against the Elsevier 2025 Standard.** Antonis Charalampous, Nicolas Tsapatsoulis, Constantinos Djouvas. Published 10 Oct 2026, Sustainability 18(20):10316. Read: ABS, from Crossref; MDPI returned 403. [DOI 10.3390/su182010316](https://doi.org/10.3390/su182010316)
  - Evidence: five LLMs on 5,000 OSDG texts, multi-label. All over-classify against Elsevier, assigning 1.15 to 1.81 times as many labels. Of 85 model-goal pairs, 64 are significantly above and 11 below. Embedding matching narrows the gap on 9 of 17 goals and overshoots on 8. Conclusion: the tested zero-shot setups cannot directly replace the mapping; hybrids look more tractable. Elsevier is treated as a reference, not truth.

- **Alignment and Boundary Shifts in SDG Classification: Evaluating LLMs against Rule-Based Systems without a Gold Standard.** Hongren Wang, Chi-Un Lei, Dickson K.W. Chiu. First posted 15 Jun 2026, Research Square. Not peer reviewed as far as I can tell. Read: ABS. [DOI 10.21203/rs.3.rs-9413287/v1](https://doi.org/10.21203/rs.3.rs-9413287/v1)
  - Evidence: GPT-5-mini over a pool of 1.76 million English articles from Aurora and Elsevier retrievals. Very low document-level overlap with the rule systems, yet the macro-level profile is preserved. Recommends a funnel: rules for broad recall, an LLM as a core filter. The abstract gives no overlap figure.
  - Conflict: this paper finds the LLM *tightens* compared with rules; Charalampous et al. find LLMs *over-assign* compared with Elsevier. The two start from different text and different candidate pools. I have not established why they differ.

- **Before You Trust the Count: Auditing Multilingual Sustainable Development Goal News.** Sergey Volkov. Published October 2026 (workshop date 28 Oct 2026), NLP4PI at EMNLP 2026, pp. 67-76. Read: PDF. [ACL Anthology 2026.nlp4pi-1.7](https://aclanthology.org/2026.nlp4pi-1.7/)
  - Data: 16,060,006 candidate titles in English, Spanish, Russian and Chinese.
  - Evidence: on the OSDG and SDGi public benchmarks, supervised TF-IDF has balanced accuracy 0.610 and 0.767. "Always accept" gets OSDG F1 0.973 and AP 0.949, which exposes a trap in OSDG-based reporting. Paired translated headlines give assignment disagreement of 33.0% (en-es), 46.6% (en-ru), 72.9% (en-zh) on the expanded lexicon. On a controlled benchmark (Table 6; I read the columns as false-release rate, correct-claim retention, release rate, in the caption's order), the full audit scores 1.5%, 25.6%, 16.9% on the OSDG semantic track and 0.0%, 37.1%, 27.3% on SDGi, against 36.1% and 28.3% false releases without specifications. On the engineering track retention falls to 1.0%, which is the over-abstention the abstract admits.
  - Relevance: an evaluation template for SDG counts without labels. It tests lexicons and weak signals on news titles, not an LLM on reports, and it is single-authored.

- **UK charities that work internationally: an open dataset with validated SDG classification.** Jack Chen. First posted 21 Jul 2026, SocArXiv. Read: ABS. [DOI 10.31235/osf.io/tky7s_v1](https://doi.org/10.31235/osf.io/tky7s_v1)
  - Evidence: LLM assigned primary SDG for 19,688 charities. It matched a hand-labelled, blind, stratified sample of 150 at 77.3% (95% CI 70.0 to 83.3), against a 56% benchmark from earlier machine learning. The single labeller is also the author.
  - Relevance: rare example of development-adjacent text with a pre-registered labelling approach and open labels.

- **Quantifying SDG curriculum alignment using a GenAI-assisted, expert validated mapping approach.** Edoardo Bertone, Sherif Mostafa, Ivan Gratchev, Tommy Soesmanto, Jo McCormack, Harry Kanasa. Published 16 Jul 2026, Int. J. Sustainability in Higher Education. Read: ABS. [DOI 10.1108/ijshe-12-2025-1514](https://doi.org/10.1108/ijshe-12-2025-1514). LLM scores (0 to 1) on 241 course profiles, revised by eight experts; "high consistency" reported without a number in the abstract.

- **ITL: Interpretable Document Alignment with Structured Reference Frameworks.** Raúl Giráldez, Dayrelis Mena, Jesús S. Aguilar-Ruiz. First posted 27 Aug 2026. Read: ABS. [arXiv 2608.27031](https://arxiv.org/abs/2608.27031). Term profiles built from a framework document, giving traceable scores per concept. Tested only for internal consistency on the 17 SDG statements, so accuracy on real documents is not shown.

#### Humanitarian text

- **Causal Evidence Extraction and Triangulation in Crisis Reports using Large Language Models: A ReliefWeb-based Study.** Yuanjun Zhang, Mourad Oussalah. First posted 5 Aug 2026. Read: ABS. [arXiv 2608.04576](https://arxiv.org/abs/2608.04576). Not vocabulary mapping as such: it extracts intervention-outcome records, restricted by intervention class, and links each to supporting text. Weighted F1 90.73% (best closed LLM) and 94.15% (fine-tuned Llama-3.1-8B) on 100 expert-annotated reports. Relevance: it describes reports much like the use case's as "long, noisy, and multi-topic", and query-conditioned extraction is a route to mapping by concept. Overlaps with the extraction researcher's scope.

- **Are LLMs Ready for Conflict Monitoring? Empirical Evidence from West Africa.** Hoffmann Muki, Olukunle Owolabi. First posted 5 May 2026. Read: ABS. [arXiv 2605.04177](https://arxiv.org/abs/2605.04177). Classification of conflict events against ACLED, with systematic directional errors: Gemma misclassifies up to 18.29% of legitimate battles as civilian-targeted violence with zero errors the other way. Domain adaptation helps direction but not actor bias. Relevance: errors in humanitarian classification may be one-sided, so aggregate F1 can hide them.

#### AGROVOC

- **LLM-Generated RDF Triples for Agricultural Species: A Comparative Evaluation Using AGROVOC Grounding.** Leonardo Vianna do Nascimento. Published 19 Jul 2026, EnCompIF 2026. Read: ABS. [DOI 10.5753/encompif.2026.21874](https://doi.org/10.5753/encompif.2026.21874). With the correct AGROVOC URIs supplied in a file, URI precision on 38 species was 100% for Claude, 16.2% for ChatGPT and 8.1% for Gemini. It is a small test, and the results depend on prompt and version, but it reinforces validating identifiers in code.

#### Multilingual mapping

- **MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification in Low-Resource Settings.** Alice Schiavone, Marco Fraccaro, Lea Marie Pehrson, Silvia Ingala, Rasmus Bonnevie, Michael Bachmann Nielsen, Vincent Beliveau, Melanie Ganz. Published May 2026, CLINICALNLP at LREC 2026, pp. 313-323. Read: ABS. [ACL Anthology 2026.clinicalnlp-1.35](https://aclanthology.org/2026.clinicalnlp-1.35/). A 4B model across seven datasets in English, Spanish, French and Danish; mean macro-F1 88 on five chest X-ray datasets; 80 labelled samples reach weighted F1 82 on Danish.
- **Extracting Semantics from Cattle Reporting Categories for Data Interoperability and Findability.** Kassy Raymond, Andrew Hamilton-Wright, Deborah Stacey. First posted 8 Sep 2026. Read: ABS. [arXiv 2609.09381](https://arxiv.org/abs/2609.09381). Found through an AGROVOC search. Cattle terms from five datasets capture age, sex and production detail not present in AGROVOC. Argues for using the terms already in the data rather than forcing them into a vocabulary. A useful reminder that a target vocabulary may lack the needed granularity.
- See also BioELX, LongBEL, MDPO, German SNOMED and Faroese in Q1; Fallatah, DH and MultiFarm in Q5.

### Just outside the window (Q3)
- *Classifying and Tracking International Aid Contribution Towards SDGs*. Sungwon Park, Dongjoon Lee, Kyeongjin Ahn, Yubin Choi, Junho Lee, Meeyoung Cha, Kyung Ryul Park. arXiv v1 21 May 2025, v2 24 Jun 2025. Important as the nearest SDG work on aid text. Metadata and one-line abstract only. [arXiv 2505.15223](https://arxiv.org/abs/2505.15223)
- *Monitoring Global Aid Flows: A Novel Approach Using Large Language Models*. Xubei Luo, Arvind Balaji Rajasekaran, Andrew Conner Scruggs. World Bank Policy Research Working Paper 11248, 2025 (RePEc record). Read: ABS. Unsupervised clustering of nearly 400,000 records from the 2021 OECD CRS into 841 activity clusters and 80 subsectors, finding 36 aid areas not tracked in the current taxonomy, validated against CRS benchmarks. It discovers categories, not maps to supplied ones, but is the closest work to DAC sectors I found. [RePEc](https://ideas.repec.org/p/wbk/wbrwps/11248.html)
- *TextMineX: Data, Evaluation Framework and Ontology-guided LLM Pipeline for Humanitarian Mine Action*. Chenyue Zhou, Gürkan Solmaz, Flavio Cirillo, Kiril Gashteovski, Jonathan Fürst. Findings of EACL 2026, March 2026, pp. 2505-2523. Title only. [ACL Anthology](https://aclanthology.org/2026.findings-eacl.130/)
- *Polarity Detection of Sustainable Development Goals in News Text (SDG-POD)*. Andrea Cadeddu, Alessandro Chessa, Vincenzo De Leo, Gianni Fenu, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino, Luca Secchi. First posted 24 Sep 2025, v4 on 3 Jul 2026. Metadata only. [arXiv 2509.19833](https://arxiv.org/abs/2509.19833)

### Inferences
- SDG labels from different instruments should be treated as differing measurements, not noisy versions of one truth. Reporting how much each instrument disagrees may be more honest than a single accuracy number.
- Since OSDG-based accuracy can look good for a do-nothing classifier (Volkov), the use case should build a small gold set of its own, from evaluation reports, rather than rely on OSDG.
- Rule-based steps for structure, and LLM steps for meaning, is what top teams converged on in UZH. This may also suit report sections, though no paper tests it on evaluation reports.

### Gaps
- **I found no paper in the window on mapping UN evaluation reports, or on UNBIS Thesaurus or OECD DAC sector classification with LLMs.** Searches covered arXiv, Crossref and ACL Anthology titles. OpenAlex search was not available.
- The MDPI full text (Charalampous et al.) could not be opened, so the identity of the five LLMs and any F1 numbers are unknown to me. The user can obtain it; it is open access per the journal.
- Comparisons of SDG tagging across French and Spanish on the same documents do not appear in what I found.

---

## Q4. How mapping quality is evaluated, and the numbers, including across languages, label noise, and human agreement

### Takeaway
Evaluation is moving away from exact match against old labels towards graded human judgement, frequency-split reporting and benchmark diagnostics, because old labels are known to be incomplete. Cross-language gaps are large and consistent, with English far ahead: for example Recall@1 91.2 for English vs 35.9 for Finnish in the best system. Human agreement on these tasks is only moderate, so any target near 1.0 is unrealistic.

### Cited Findings

#### Across languages

| Source | Task | Numbers |
|---|---|---|
| [BioELX, Findings EMNLP 2026, Table 1](https://aclanthology.org/2026.findings-emnlp.1109/) | Recall@1, XL-BEL, retrieve + rerank | EN 91.2, ES 69.6, DE 53.0, FI 35.9, RU 56.6, TR 60.3, KO 39.6, ZH 44.5, JA 39.1, TH 48.6; average 53.8. Prior best averages about 35.6. |
| [LongBEL, arXiv 2605.13451](https://arxiv.org/abs/2605.13451) | Recall@1, ensemble | EN 80.0, FR 77.6, ES 63.3 / 65.8 / 71.0 (different KBs and datasets, so not strictly comparable) |
| [MDPO, Findings EMNLP 2026, Table 1](https://aclanthology.org/2026.findings-emnlp.1873/) | Micro-F1, historical text with OCR noise | hipe-2020: FR 72.4, DE 65.6, EN 73.8. newseye: FR 68.7, DE 59.2, SV 65.0, FI 62.7 |
| [Volkov, NLP4PI 2026, Table 5](https://aclanthology.org/2026.nlp4pi-1.7/) | SDG lexicon, assignment disagreement on translated pairs | en-es 33.0%, en-ru 46.6%, en-zh 72.9% (expanded lexicon). Neither language treated as truth. |
| [Fallatah, OM-2026, Table 1](https://om.ontologymatching.org/2026/papers/om2026-STpaper3.pdf) | Arabic-English ontology matching F1 | Same ontology: Claude 4.6 0.892, GPT-4o 0.836. Different ontologies: 0.198, 0.187. LSMatch-ML 0.663 and 0.631. |

#### Human agreement and graded evaluation

- **Large Language Models as Judges for Human Subject Indexing.** Bartłomiej Włodarczyk, Magdalena Zdunek. Published 14 Aug 2026, Cataloging & Classification Quarterly. Read: ABS via OpenAlex; publisher page returned 403. [DOI 10.1080/01639374.2026.2714375](https://doi.org/10.1080/01639374.2026.2714375). Forty Polish papers. Metrics range 0.56 to 0.96. Of AI-gold disagreements, 66.67% (Gemini) and 39.13% (ChatGPT) coincided with disagreements among human evaluators. LLMs are weak at catching incorrectly assigned descriptors.

- **Beyond the gold standard: evaluating AI for subject indexing of Swedish LGBTQ+ fiction.** Koraljka Golub, Olof Falk, Siska Humlesjö. Published 22 Sep 2026, Journal of Documentation. Read: ABS. [DOI 10.1108/jd-06-2026-0363](https://doi.org/10.1108/jd-06-2026-0363). F1 on metadata 44%, on full text 12%. Fully incorrect terms 37.5% and 56.1%. LLMs often choose broader concepts and do not follow thesaurus definitions or indexing policy. 14 works.
  - Relevance: full text giving much worse results than metadata is a warning for mapping from long documents, though fiction differs greatly from reports.

- **Kähler et al.** See Q2. Graded judgement reverses the ranking seen against old labels: XR-Transformer leads on F1 and LLM-Ensemble leads on graded g-F1. Summary read reports no inter-rater number.

- **UZH overview.** See Q3. For the argument relation subtask, intra-annotator agreement on 233 blind re-annotations is κ=0.540, used as the "human ceiling"; the best model reaches κ=0.354. For tags, 20 of 178 reviewed paragraphs have genuine ontology gaps.

- **Awareness Without Synthesis: LLM-Powered Detection and Alignment of Hidden Vocabulary Fragmentation in Scholarly Knowledge Graphs.** Ana Bossler, Enric Bas, Andrés Fullana. First posted 13 May 2026, Research Square. Read: ABS. [DOI 10.21203/rs.3.rs-9588669/v1](https://doi.org/10.21203/rs.3.rs-9588669/v1). Human annotation α=0.752. See Q5.

- **Systematic Failure Analysis of Open-Weight LLM Annotators for Classification Tasks.** Nahid Hossain, Siam Rahman Karip, Md Faisal Kabir, Swakkhar Shatabda, Salekul Islam. Published October 2026, EMNLP 2026 main, pp. 46409-46466. Read: ABS. [ACL Anthology 2026.emnlp-main.2037](https://aclanthology.org/2026.emnlp-main.2037/). 14 models, 51,189 instances, 10 datasets. Agreement across prompts reaches 72 to 94% accuracy at 71 to 90% coverage on clean tasks, but can make things worse where bias is shared.

#### Rare labels, hierarchy and benchmark quality

- **What Do Biomedical NER and Entity Linking Benchmarks Measure? A Corpus-Centric Diagnostic Framework.** Robert Leaman, Rezarta Islamaj, Zhiyong Lu. First posted 19 May 2026. Published July 2026, BioNLP at ACL 2026, pp. 801-811. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.bionlp-1.66/); [arXiv 2605.20537](https://arxiv.org/abs/2605.20537). Nine corpora; apparently similar tasks differ substantially in train-test overlap and terminology coverage.

- **Hierarchy-Aware Hyperbolic and Semantic Reranking for Ontology-Based Phenotype Linking.** Thomas Labbe, Moussa Baddour, Axel Bonesteve, Paul Rollier, Marie De Tayrac, Olivier Dameron. Published July 2026, BioNLP at ACL 2026, pp. 45-63. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.bionlp-1.5/). Introduces hierarchy-aware evaluation that goes beyond exact match. Hyperbolic embeddings alone do not beat semantic retrieval but add a complementary signal.

- **OntologyBench: Can Dense Retrieval Satisfy Structured Biomedical Constraints?** Xiao Yu Cindy Zhang, Wyeth Wasserman, Jian Zhu. First posted 7 Sep 2026. Published October 2026, EMNLP 2026 main, pp. 13707-13725. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.595/); [arXiv 2609.08174](https://arxiv.org/abs/2609.08174). 125,744 evaluation pairs. Reranking and LLM scoring gave little or no end-to-end improvement on these tiers, while ontology-derived fine-tuning helped on several tasks.

- **SNOMED CT Concept Recommendation from Masked Clinical Context.** Ali Noori. First posted 15 Sep 2026. Read: ABS. [arXiv 2609.17855](https://arxiv.org/abs/2609.17855). Recall@10 is 7.74% for concepts in one or two training notes, 43.90% for concepts in over ten. Sparse TF-IDF beats the retrieval-augmented hybrid.

- **MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification of Medical Device Adverse Events.** Raunak Agarwal, Markus A. Wenzel, Simon Baur, Jonas Zimmer, George Harvey, Jackie Ma. Published July 2026, ACL 2026 Long, pp. 46308-46328. Read: ABS (truncated). [ACL Anthology](https://aclanthology.org/2026.acl-long.2148/). Addresses saturated benchmarks and memorised training data.

- **Reference-Free Evaluation of Taxonomies.** Pascal Wullschleger, Majid Zarharan, Donnacha Daly, Marc Pouly, Jennifer Foster. Published July 2026, Findings of ACL 2026, pp. 25489-25507. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.findings-acl.1273/). Two metrics for judging a taxonomy with no labels, correlated with F1 and able to predict downstream hierarchical classification.

#### Calibration, abstention and uncertainty

- **How to Estimate Whether You Have Found Several Needles in a Haystack: Measuring Calibration in Multi-Label Text Classification.** Sophie Henning, Georg Hofmann, Alexander Schulte, Alexander Fraser, Annemarie Friedrich. First posted 22 Sep 2026. Read: ABS. [arXiv 2609.26468](https://arxiv.org/abs/2609.26468). Standard calibration binning misleads when most labels are negative. Proposes equal weight to positive and negative assignments. Calibrating LLM confidence for multi-label output remains open.

- **One Score, Two Decisions: Selective Prediction on the Rare-Disease Tail.** Zhaoyang Jiang, Zhizhong Fu, Yunsoo Kim, Zicheng Li, Xuanqi Peng, Fei Teng, Jiacong Mi, Honghan Wu. First posted 4 Aug 2026. Read: ABS. [arXiv 2608.14683](https://arxiv.org/abs/2608.14683). The top-two margin is a better gate for fixed candidates, but can remove the signal needed to detect that the list has no answer. It proves unlabelled scores alone cannot determine whether switching to the margin helps.

- **Uncertainty-Aware Multi-Label Routing of Clinical Text to Surveillance Pathways.** Agathe Zecevic, Sebastian Zeki, Angus Roberts. Published July 2026, BioNLP at ACL 2026, pp. 181-190. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.bionlp-1.16/). Compares threshold abstention with conformal prediction, reporting the review burden as well as accuracy; 1,773 reports, six labels; macro-F1 0.861.

- **PASC: Pipeline-Aware Conformal Prediction with Joint Coverage Guarantees.** Varun Kotte. First posted 12 May 2026. Read: ABS. [arXiv 2605.18812](https://arxiv.org/abs/2605.18812). 96.4% end-to-end coverage vs 86.5% for independent calibration on a three-stage pipeline. Independent calibration collapses to 59% under shift.

### Just outside the window (Q4)
- *Onomasiological Sense Alignment Across Dialect Dictionaries*. Nathalie Mederake, Nico Urbach, Hanna Fischer, Alfred Lameli. VarDial, March 2026. Read: ABS. Run-to-run model agreement κ=0.73 closely matches human agreement κ=0.74, degrading at finer levels of the taxonomy. [ACL Anthology](https://aclanthology.org/2026.vardial-1.10/)
- *Modulating Multi-Label Tendency in Zero-Shot LLM Coding*. Hyunwoo Choo, Sungsoo Hong. HeaLing, March 2026. Read: ABS. LLMs default to single labels (96% for Magistral, 54% for Qwen3) unless the output format is changed (down to 19% and 0.0%). Output structure is thus a quality control. [ACL Anthology](https://aclanthology.org/2026.healing-1.14/)

### Inferences
- A workable evaluation for the use case would combine several parts already demonstrated here: exact match against publisher tags, graded review by evaluators on a sample, frequency-split reporting, per-language reporting, and a count of "the vocabulary has no right concept" cases.
- Reporting agreement among instruments or runs (Mederake, Volkov) offers a check without a gold set, but agreement is not accuracy; Volkov states this explicitly.

### Gaps
- No study measures how noisy PDF parsing specifically changes mapping quality. OCR-noise results in MDPO are the nearest and are not a controlled test of noise level.
- Inter-rater agreement for tagging against SDGs, DAC or UNBIS on development reports was not found.

---

## Q5. New LLM ontology matching work that could help align agency vocabularies, including OAEI 2026

### Takeaway
OAEI 2026 results are out, and LLM use is now common among leading systems on biomedical and anatomy tracks: LogMap with an LLM oracle gained about 0.09 F1 on Bio-ML. But on the tracks most like aligning multilingual thesauri, scores remain low and LLM systems do not clearly lead, and a new Arabic-English study shows LLM precision collapses to about 0.11 when no true match exists among the options shown.

### Cited Findings

#### OAEI 2026 results (campaign pages; preliminary or recently frozen)

Campaign: [OAEI 2026](https://oaei.ontologymatching.org/2026/), results due 30 Sep 2026; [results index](https://oaei.ontologymatching.org/2026/results/). System papers due 12 Nov 2026, so I found no overview paper yet.

| Track | Most relevant finding | Source |
|---|---|---|
| Bio-ML, equivalence, repaired, macro F1 (frozen 3 Oct 2026) | Matcha-ML 0.636 (uses train maps), MetaMatch 0.634, LogMapLLM 0.623, LogMapBio-LLM 0.621, Matcha 0.612, GenOM 0.554, LogMap 0.534, mosaic 0.487. LogMapLLM precision 0.507, recall 0.812. | [Bio-ML results](https://bio-ml.oaei-ml.org/results/) (raw text read) |
| Anatomy | DRAMA 0.951, Matcha 0.941, ALIN 0.931 (LogMapBioLLM named third, F1 not stated in the text I read). Page notes CHARM, DRAMA, LogMapLLM, LogMapBioLLM and Mosaic runtimes vary with LLM API calls. 12 of 14 systems beat the string baseline. | [Anatomy results](http://oaei.ontologymatching.org/2026/results/anatomy/index.html) (via summarising fetch) |
| MultiFarm, different ontologies in different languages | LSMatch Multilingual 2.0: P 0.742, R 0.285, F1 0.412. SeCea F1 0.241. Matcha F1 0.231. LogMap family: P about 0.89, F1 0.009. No system is identified as LLM-based on the page. | [MultiFarm results](http://oaei.ontologymatching.org/2026/results/multifarm/index.html) (raw text read) |
| Digital Humanities, **thesaurus matching** including UNESCO-related thesauri | Average F1: LogMap KG 0.61, Matcha 0.59, MOSAIC 0.52, Agent-OM 0.39, TIM 0.34, SECEA 0.23. Agent-OM and MOSAIC results were author-provided and not verified with MELT. | [DH results](https://github.com/FelixFrizzy/DH-benchmark/blob/main/results/OAEI2026/results.md) (raw average row read) |
| Archaeology multilingual | Average F1: Agent-OM 0.36 (unverified), LogMap KG 0.24, Matcha 0.23, TIM 0.09, SECEA 0.01. Several systems returned empty alignments. | [Archaeology results](https://github.com/FelixFrizzy/DH-benchmark-multiling/blob/main/results/OAEI2026/results.md) (raw average row read) |

Assumption on Bio-ML: the block order follows the page's tab order of Macro, NCIT-DOID, SNOMED-FMA, SNOMED-NCIT. The raw text does not label each block individually.

#### OM-2026 workshop papers (ISWC 2026, 25 Oct 2026, Bari)

The [OM-2026 page](https://om.ontologymatching.org/2026/) lists accepted papers with PDFs. CEUR-WS publication was not confirmed. Published date below means the workshop date.

- **DRAMA: Debate-Driven Reasoning Agents for Matching Alignments.** Yannick Oefler, Sophia Hammes, Jan Portisch, Sven Hertling. OM-2026 long paper. Read: PDF. [PDF](https://om.ontologymatching.org/2026/papers/om2026-LTpaper2.pdf)
  - Evidence: F1 0.951 on Anatomy and 0.688 on Conference (P 0.647, R 0.734) with GPT-5.5. Mistral Small on Anatomy 0.912. A staged pipeline cuts the search space by 80 to 91% with reject precision above 0.997. With GPT-5.5 the critic raises objections but produces **zero decision changes**, so a stronger model beat the debate mechanism in their tests.
  - Relevance: cost design for alignment. Debate does not appear worth paying for on top of a strong model.

- **Which Model, Which Prompt? Selecting and Configuring LLMs as Oracles for Ontology Matching.** Jonathon Dilworth, Sviatoslav Lushnei, Dmytro Shumskyi, Severyn Shykula, Pedro Cotovio, Artur d'Avila Garcez, Ernesto Jiménez-Ruiz. OM-2026 long paper. Read: PDF. [PDF](https://om.ontologymatching.org/2026/papers/om2026-LTpaper5.pdf)
  - Evidence: 86 models over 3,792 runs. No configuration wins consistently. Demonstrations help cautious oracles (Gemini 2.5 Flash sensitivity 0.534 → 0.937 on Knowledge Graph at k=4). Asking equivalence as two-way subsumption improves specificity on permissive oracles at twice the number of calls. Table 1 reports cost, for example gemini-2.5-flash at $0.877 per 1,000 pairs, 10.3 s per call, F1 0.92 on human-mouse.
  - Relevance: the most practical guide found. LogMap's design only sends uncertain candidates to the LLM.

- **Evaluating Large Language Models for Arabic-English Ontology Matching.** Omaima Fallatah. OM-2026 short paper. Read: PDF. [PDF](https://om.ontologymatching.org/2026/papers/om2026-STpaper3.pdf)
  - Evidence: on 25 MultiFarm tasks, same-ontology F1 is up to 0.892 (Claude 4.6). Different ontologies: precision about 0.11, recall up to 0.805, F1 0.183 to 0.198, because LLMs rarely answer NO_MATCH given the full candidate list. LSMatch-ML: P 0.769, R 0.535, F1 0.631 on the different-ontology tasks. Claude 4.6 in the paper is as named by the author.
  - Relevance: direct warning for agency vocabulary alignment and for abstention: when a true match does not exist, the LLM chooses one anyway.

- **OntoAligner-Ensemble: Voting-Based Fusion across Heterogeneous Ontology Alignment Techniques.** Hamed Babaei Giglou, Sören Auer, Peio Popov, Mahsa Sanaei, Jennifer D'Souza. First posted 31 Aug 2026. Accepted to OM-2026. Read: ABS. [arXiv 2608.31137](https://arxiv.org/abs/2608.31137). Eight tasks from five OAEI tracks. Mixed-paradigm ensembles generally raise precision, homogeneous LLM ensembles more often reach higher F1. No figures in the abstract.

- **Listed on the OM-2026 page, not read (title and metadata only):** *Complex Ontology Matching with LLMs: Combining Topological Modularization and Graph Verbalization* (Igor Nascimento, Rinaldo Lima, Cassia Trojahn) [PDF](https://om.ontologymatching.org/2026/papers/om2026-LTpaper6.pdf); *Language Model Embeddings for Species Synonym Detection in Biodiversity* (Sefika Efeoglu, Jana Hoffmann) [PDF](https://om.ontologymatching.org/2026/papers/om2026-STpaper2.pdf); *Foundational Ontology Alignments For Information Sharing: A Case Study* (Rebecca Rafferty) [PDF](https://om.ontologymatching.org/2026/papers/om2026-LTpaper7.pdf); *Aligning the Czech Core Metadata Model with DataCite* (Ondřej Zamazal) [PDF](https://om.ontologymatching.org/2026/papers/om2026-poster.pdf); *A Controlled Semantic-Intervention Protocol for Construct Validation of Ontology-Alignment Justification Metrics* (Vitaliy Charushin) [PDF](https://om.ontologymatching.org/2026/papers/om2026-STpaper4.pdf).

#### Other new matching work

- **AgentMap: Joint Equivalence and Subsumption Discovery for Ontology Matching.** Yiping Song, Jiaoyan Chen, Renate Schmidt, Hui Yang, Wen Zhang. First posted 29 Jul 2026. Read: ABS. [arXiv 2607.27130](https://arxiv.org/abs/2607.27130). For each concept, finds the equivalent if one exists, else the most specific broader concept. Outperforms equivalence-only and subsumption-only baselines per the abstract, no figures.
  - Relevance: high in principle. Agency theme lists rarely match one-to-one, and "broader than" mappings are likely the common case when aligning to UNBIS (my inference).

- **Crowd-OM: Crowdsourcing for Ontology Matching Validation.** Zhangcheng Qiang, Weiqing Wang, Kerry Taylor. First posted 12 May 2026; a later version 7 Sep 2026. Read: ABS. [arXiv 2605.12226](https://arxiv.org/abs/2605.12226). LLM matchers produce more candidates than experts can review; proposes annotator weighting and consistency pre-filling.

- **Improving Interoperability among Defence and National Security Ontologies.** Jonathon Dilworth, Pedro Giesteira Cotovio, David Herron, Paul Cripps, Nigel Dewdney, Catia Pesquita, Ernesto Jiménez-Ruiz. First posted 6 Aug 2026; ISWC 2026 Resource Track. Read: ABS. [arXiv 2608.05867](https://arxiv.org/abs/2608.05867). New OAEI track of eight tasks. Its method is useful: consensus alignments across systems plus manual validation gives a silver standard. That is a method for building a gold set when none exists.

- **Surprising Effectiveness of Self-Demonstrations in Enhancing Schema-Ontology Mapping with LLMs.** Siddhesh Thombre, Manasi Patwardhan, Sunita Sarawagi. First posted 12 Sep 2026. Read: ABS. [arXiv 2609.13776](https://arxiv.org/abs/2609.13776). One-shot prompting and naive multi-stage pipelines perform poorly; symbolic constraints plus self-generated demonstrations reach 25 points F1 over baselines on three hard RODI scenarios. Maps database schemas rather than text, but is informative on how to generate demonstrations with no labels.

- **Open Ontologies: Tool-Augmented Ontology Engineering with Stable Matching Alignment.** Fabio Rovai. First posted 9 May 2026. Read: ABS. [arXiv 2605.09184](https://arxiv.org/abs/2605.09184). One-to-one stable matching gives F1 0.832 on Anatomy; removing it drops F1 to 0.728. An LLM reading raw OWL scored F1 0.323, worse than with no file (0.431); structured tool access gave 0.717. Single author; the F1 is lower than the leaders above.

- **Hybrid LLM and Embedding Ontology Matching with Reference Ontologies.** Sarah Dahir, Abderrahim El Qadi. Published July 2026, ICCSC 2026. Read: ABS via OpenAlex. [DOI 10.1109/iccsc71566.2026.11650532](https://doi.org/10.1109/iccsc71566.2026.11650532). One LLM call per element to assign a semantic type, then matching through a reference ontology by embeddings. No figures in the abstract.
  - Relevance: using a hub such as UNBIS to align agency lists fits this design (my inference).

- **Awareness Without Synthesis.** Bossler et al., see Q4. An LLM module aligns differing vocabularies across subdomains; automatic query expansion gains a mean of 636 papers (+39.9%) on a 3,138-publication corpus. Mean 1 − RBO = 0.976 shows almost entirely non-overlapping vocabularies.

- **A Framework for Dynamic Knowledge Graph Ontology Incremental Alignment based on LLM Agents (DOKA-Agent).** Hao Qin, Fusheng Wei, Bin Liang, Wobin Lin. First posted 3 Jul 2026, Research Square. Read: ABS. [DOI 10.21203/rs.3.rs-9503150/v1](https://doi.org/10.21203/rs.3.rs-9503150/v1). Addresses vocabularies that change, though evaluated on entity alignment in DBP15K, not vocabularies.

- **Medical Bilingualism: Cross-Ontology Graph Retrieval for Incommensurable Knowledge Bases (Xonto-RAG).** Zixuan Wang, Ke Lu, Zihan Wang, Junwei Lei, Junhao Dong, Jinyu Guo, Chaomeng Chen. Published October 2026, EMNLP 2026 main. Read: ABS. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.201/). Keeps two ontologies separate with LLM-verified typed bridges, instead of merging them, to avoid false equivalences. Relevant as a design principle for agency lists that truly differ in meaning.

### Just outside the window (Q5)
- *Large Language Models as Oracles for Ontology Alignment* (LogMapLLM). Sviatoslav Lushnei, Dmytro Shumskyi, Severyn Shykula, Ernesto Jiménez-Ruiz, Artur d'Avila Garcez. EACL 2026, March 2026, pp. 2435-2449; arXiv ID suggests August 2025 (not separately checked). Reported top-two overall in the OAEI 2025 Bio-ML track, per a search snippet of the paper. Not read in this pass. [ACL Anthology](https://aclanthology.org/2026.eacl-long.110/); [arXiv 2508.08500](https://arxiv.org/abs/2508.08500)
- *Improving LLM-based Ontology Matching with fine-tuning on synthetic data*. [arXiv 2511.22612](https://arxiv.org/html/2511.22612v1). November 2025 by ID. Search snippet only.
- *GenOM*: reported as accepted to World Wide Web (2026). Search snippet only; not verified.

### Inferences
- Comparing the tracks suggests that LLM oracles help most where a good structural or lexical matcher already supplies candidates and sends only hard cases to the LLM (LogMap's gain on Bio-ML). This is a comparison across systems, not a controlled test, so it remains a hypothesis.
- For agency vocabularies: use a two-way ("A is broader than B", "B is broader than A") test instead of only equivalence, always allow "no match", and start with an ensemble and human validation of the disputed mappings.
- A consensus method (as in the DISO track) offers a way to build a validation set for agency vocabularies without labelling everything by hand.

### Gaps
- No OAEI 2026 overview or per-system papers on LLM-specific numbers yet. System papers are due 12 Nov 2026 and the DH and Archaeology pages do not label which systems use LLMs. Which of MOSAIC, MetaMatch or Matcha use LLMs was not established.
- OM-2026 first-posted dates are not established where no preprint exists. CEUR-WS volume number not confirmed.
- I found no matching study on UNBIS, AGROVOC-to-agency lists or DAC-to-SDG crosswalks in the window.

### Search coverage and tool limits (applies to all questions)
- Worked: arXiv advanced search by first-submission date, up to 50 results per query, across about 40 topic queries. After rate limiting I re-ran the failed queries slowly; all were eventually completed. Broad queries such as "taxonomy classification LLM" hit the 50-result cap and were mostly noise, so items beyond the cap were not seen. ACL Anthology event pages for ACL, EMNLP, Findings, EACL, SemEval and LREC 2026 (thousands of titles filtered by keyword, then abstracts read). Crossref API (about 700 candidates, title filter). OAEI and OM-2026 pages.
- Did not work: OpenAlex search was paused for anonymous users ("Anonymous search is paused"); OpenAlex lookup by DOI did work. Semantic Scholar returned 429. DBLP showed a bot check, which I did not try to bypass. MDPI, ACM Digital Library and Taylor & Francis pages returned 403.
- SemEval 2026 title filter found no subject-tagging or ontology-mapping task; no LLMs4Subjects 2026 edition was found.
- Keyword filters on titles will miss papers whose titles use other words. This list is verified but not exhaustive.
