# Ontology-guided extraction: new papers, May to November 2026

Scope: papers in which a supplied ontology, SHACL shape, schema or competency-question set constrains extraction or populates a knowledge graph (KG) from text. Search date 2026-11-05. Entity linking and taxonomy classification are covered by another researcher.

How to read the access labels used below:

- **FULL**: I read the whole text.
- **PARTIAL**: I downloaded the full text and read the abstract, the evaluation tables and keyword-selected passages, but not every page.
- **ABSTRACT**: abstract only, from the publisher, repository or Crossref. Full text not read.
- **METADATA**: title, authors and venue only. **Not read.** The user can obtain these.
- Numbers marked "(fetch paraphrase)" come from a summarising web-fetch tool's reading of the publisher page. I did not see the raw text, so check them before quoting.

## Objective: verified paper catalogue

### Takeaway

I list 50 in-window papers, each checked against a primary page: 34 with at least the abstract read (Tiers A, B and C) and 16 as title and metadata only (Tier D, including three workshop talks known only from the programme). None was tested on UN evaluation reports. The closest documents are policy excerpts, statutes, safety reports, industrial incident reports and humanitarian sources.

### Cited Findings

#### Tier A: closest to the use case

**A1. LMKG: a flexible method for ontology-aware Knowledge Graph Extraction from text**
- Authors: Gianmarco Pappacoda, Paolo Torroni
- First posted: no preprint checked. Published: October 2026.
- Venue and link: AKBC 2026 workshop, [ACL Anthology 2026.akbc-1.23](https://aclanthology.org/2026.akbc-1.23/), DOI 10.18653/2026.akbc-1.23
- Read: PARTIAL
- Method: a modular pipeline with open and ontology-constrained modes. The model supplies an evidence quote, and a deterministic resolver maps it to a character span, leaving the span empty rather than guessing if no match exists. It offers two long-document strategies: "fold" accumulates a graph over chunks, and "enrich" extracts each chunk independently, merges with a base graph, then proposes implied edges.
- Evidence: see the head-to-head table under the third key question. [MINE-1 fact recovery was 84.87 for fold and 90.80 for enrich; downstream HotpotQA question answering was worse with graphs than with full passages (65.36 exact match with passages)](https://aclanthology.org/2026.akbc-1.23.pdf).
- Relevance: the cleanest same-system on/off ontology comparison, and it has an explicit long-document method. Its "enrich" completion step is not source-grounded, as the authors state.

**A2. From Documents to Decisions: Ontology-Driven Knowledge Graph Construction Using Iterative LLM Pipelines**
- Authors: Felix Neubürger, Ole Katzenberger, Salvador Cantuarias, Thomas Kopinski
- First posted: none found. Published: CEUR proceedings 2026-10-13, for SEMANTiCS, September 2026.
- Venue and link: SEMANTiCS 2026 posters and demos, [CEUR Vol-4293 paper 59](https://ceur-ws.org/Vol-4293/paper59.pdf). CEUR's listing gives DOI 10.83095/ceurws_04293_059, which Crossref did not resolve (404), so treat that DOI as unconfirmed.
- Read: FULL
- Method: an OWL ontology seeds competency questions per class via SPARQL. Hybrid retrieval (dense plus BM25 plus reranking) finds chunks for each question. Rules run first with an LLM fallback, Pydantic enforces structure with automatic retry, a second pass extracts relations constrained by domain, range and cardinality, SHACL validates before persistence, and coverage gaps generate follow-up questions until convergence.
- Evidence: [33 nuclear decommissioning documents, 3,004 segments, 58 classes and 120 object properties; going from single-pass to 5-pass raised entities from 377 to 572 and relation coverage from 41.3% to 74.3%, while class coverage rose only from 33.7% to 35.7%; SHACL score 0.937 to 0.942, and 0.953 with a law graph; the 31B model trailed the 2B model on class coverage, relation coverage and SHACL score at 2.9 times the build time; n = 3 runs](https://ceur-ws.org/Vol-4293/paper59.pdf).
- Limits stated by the authors: [factual correctness is not assessed against a gold set](https://ceur-ws.org/Vol-4293/paper59.pdf). I also note a reported relation coverage of 104.5% in one condition, above 100%, which the paper does not explain in the text I read.
- Relevance: the closest architecture to the use case, covering local models, mixed formats, safety and licensing documents, German statutes and a question-driven loop.

**A3. Schema-Agnostic Knowledge Graph Construction via Hybrid Ontology Discovery for Cyber Threat Intelligence (Anchor)**
- Authors: Seonwoo Kim, Jinwoo Kim, Daegyu Kang, Daeseong Kim, Insup Lee
- First posted: arXiv v1 stamped 31 May 2026. Published: no venue found.
- Link: [arXiv 2606.01208](https://arxiv.org/abs/2606.01208)
- Read: PARTIAL
- Method: extract without binding to the schema, then align each element to a class or property URI using embedding search, switching to LLM-guided recursive navigation of the hierarchy when search confidence is low. SHACL violations go back to the mapper, [capped at three retries; persistent violations are kept with a warning flag rather than discarded](https://arxiv.org/html/2606.01208).
- Evidence: [entity typing F1 on UCO (419 classes) was 0.7347 against 0.4521 for the best baseline; whole-schema prompting (CTINexus) fell from 0.6370 on MALOnt to 0.4439 on UCO; non-compliance on UCO was 5.2% against 43.9% to 75.8% for baselines; predicate typing was much harder at 0.5180](https://arxiv.org/html/2606.01208). [With ground-truth inputs, search-only reached 0.9184 on entities but only 0.3142 on predicates; hybrid reached 0.9364 and 0.7843](https://arxiv.org/html/2606.01208).
- Inconsistency in the paper: [its Table I gives schema sizes of up to 997 for UCO, 77 for STIX and 60 for MALOnt, while the evaluation text gives 419, 109 and 75 classes](https://arxiv.org/html/2606.01208). I did not establish whether Table I counts classes plus properties.
- Relevance: the best evidence on avoiding passing a whole ontology, and on the search and navigation split between classes and predicates.

**A4. An Ontology-Guided, Deduplication-Aware Extraction Layer for Knowledge Graph Construction from Heterogeneous Documents**
- Authors: Vaibhav Dangaich, Kevin Lewis, Kundeshwar Pundalik
- First posted: arXiv v1 stamped 22 Jul 2026. Published: no venue found.
- Link: [arXiv 2607.28662](https://arxiv.org/abs/2607.28662)
- Read: PARTIAL
- Method: a production system. Paragraph windows ranked by proper-noun density are embedded and matched against ontology class and predicate embeddings in Neo4j, with a PageRank penalty for generic classes, a score floor and a 5,000-token budget. Pydantic enforces output. [The model is told to mark types as novel when nothing fits; unmatched predicates are flagged for human review and can be promoted into the ontology](https://arxiv.org/html/2607.28662).
- Evidence: [catalogue injection fell from about 11,200 to about 700 tokens on one representative document, a 94% reduction](https://arxiv.org/html/2607.28662). [With the target ontology force-injected, conformance was 68 to 70% on Text2KGBench and 89% on OSKGC, triple F1 was 22 to 26% and 35 to 40%, and fine-grained type mapping accuracy was 42 to 50%](https://arxiv.org/html/2607.28662). [On a 10-page OCR document, chunks of 2 pages rather than 5 raised ground-truth recall from 29.4% to 58.8%, at a cost of 2:30 more latency](https://arxiv.org/html/2607.28662).
- Limits stated by the authors: [case studies rest on single documents, public-dataset samples are small, there is no significance testing, and thresholds are hand-tuned](https://arxiv.org/html/2607.28662). Note also that [a graph outage degrades to un-guided extraction rather than blocking](https://arxiv.org/html/2607.28662), so guidance is steering rather than a hard constraint.
- Relevance: the most detailed account of noisy PDF handling, per-page OCR routing, chunking and novel-type flagging. Its domain is intelligence documents.

**A5. Extraction-Oriented Ontology Modeling for Local LLM-Based Knowledge Graph Construction in Safety-Critical Domains**
- Authors: Sarmad Rezayat, Daniel Garcia Vasquez, Gerrit Burmester, Hui Ma, Sven Hartmann
- First posted: none found. Published: first online 5 October 2026 (the page also shows a 2027 copyright line).
- Venue and link: ER 2026, LNCS vol. 17081, [DOI 10.1007/978-3-032-40795-5_5](https://doi.org/10.1007/978-3-032-40795-5_5)
- Read: ABSTRACT, paywalled. **Full text not read.**
- Method: a six-stage approach that refines the ontology during KG construction, treating the extraction schema as a distinct artefact from the domain ontology.
- Evidence (fetch paraphrase): [on 643 aviation accident reports, the 915-class AIRM-O ontology failed to support extraction by local LLMs, while a 20-class schema gave stable extraction across six local LLMs; the authors conclude ontology design matters more than model choice](https://link.springer.com/chapter/10.1007/978-3-032-40795-5_5).
- Relevance: directly challenges passing a large ontology at all, on long real-world safety reports. High priority to obtain.

**A6. Evaluating Large Language Models for Strategic Knowledge Extraction in Capability-Based Planning**
- Authors: Hein Kolk, Julia García-Fernández, Julia Bronkhorst, Roos Bakker
- First posted: none found. Published: LREC workshop volume, May 2026 by the volume's date.
- Venue and link: KG-LLM at LREC 2026, [ACL Anthology 2026.kallm-1.20](https://aclanthology.org/2026.kallm-1.20.pdf)
- Read: PARTIAL
- Method: extracts strategic entities from policy excerpts and links them to a predefined capability list, comparing GPT-4.1, GPT-4.1-mini and GPT-5-mini with human experts using the alternative annotator test.
- Evidence: [the LLM failed at the skilled-annotator threshold (epsilon 0.2) for all concept types, with winning rate 0.0, while average advantage probability stayed above 0.5; one capability-linking set-up passed at the relaxed 0.3 threshold; about €0.64 of model cost against about €136 for one expert](https://aclanthology.org/2026.kallm-1.20.pdf).
- Relevance: the closest document type I found, namely vision statements and annual reports from a public institution, with a subjective task like coding evaluation findings. I did not record the corpus size.

**A7. Verifiable Disaster Storylines and Causal Knowledge Graphs: A Citation-Grounded Pipeline from Heterogeneous Humanitarian Sources**
- Authors: Ivan Decostanzi, Michele Ronco, Sergio Consoli, Christina Corbane, Lorenzo Bertolini, Indaco Biazzo, Daria Mihaila, Manuel Garcia-Herranz, Felix Schwebel, Yelena Mejova, Kyriaki Kalimeri
- First posted: arXiv v2 stamped 13 Sep 2026; v1 date not checked. Published: GoodIT '26, Crossref 2026-09-02, online 2026-10-06.
- Links: [arXiv 2609.00858](https://arxiv.org/abs/2609.00858), DOI 10.1145/3794786.3830759
- Read: PARTIAL
- Method: RAG over EM-DAT, ReliefWeb and European Media Monitor documents fills [a fixed 17-field schema, then builds causal triples with citation-grounded narratives](https://arxiv.org/html/2609.00858).
- Evidence: [three crisis cases with 9 expert and 9 non-expert evaluators, reporting high retrieval precision and a clear expert preference for citation-grounded components](https://arxiv.org/html/2609.00858). The excerpts I read gave no extraction F1.
- Relevance: the only paper found using ReliefWeb-type humanitarian sources. The schema is a fixed field list and causal predicates rather than a formal ontology.

**A8. LLM-Guided Ontology-Driven Knowledge Graph Construction from Unstructured Text (OntoConnectLM study)**
- Authors: Abdelhadi Belfadel, Maxence Gagnant, Joseph Kattan, Sana Tmar
- First posted: arXiv v1 stamped 15 Sep 2026. Published: no venue found.
- Link: [arXiv 2609.31663](https://arxiv.org/abs/2609.31663)
- Read: FULL, except that about 2,900 characters around the quantisation results were cut from my view
- Method: local 7B to 32B models. The prompt gives context, task, schema, one example and input paragraphs, and outputs JSON that becomes RDF and OWL. The schema is expert-designed with 16 classes and 21 relations.
- Evidence: [80 French power-grid incident reports; adding semantic descriptions raised Gemma3 strict NER F1 from 0.42 to 0.57 and relation classification F1 from 0.51 to 0.70; direct triple extraction scored 26% against 23% for the sequential pipeline](https://arxiv.org/pdf/2609.31663).
- Relevance: French operational reports written by many authors, on local models. Note that the "schema-guided" comparison is labels versus labels with descriptions, not guided versus free. The corpus is private, so it cannot be reproduced.

**A9. LLM-Assisted Ontology Engineering and Construction of a French Legal Knowledge Graph**
- Authors: Génesis Montenegro, Mokhtar Boumedyen Billami, Catherine Faron, Fabien Gandon, Pierre Monnin
- First posted: arXiv v1 stamped 27 Jul 2026. Published: SEMANTiCS 2026 posters and demos per the paper's footer; I did not confirm the proceedings entry.
- Link: [arXiv 2607.24551](https://arxiv.org/abs/2607.24551)
- Read: PARTIAL
- Method: starts from the SEMLEG core ontology, runs open extraction on a stratified sample, fuses labels by embedding, induces object properties with domain and range, then runs closed extraction over the full corpus and lifts to RDF with RML mappings.
- Evidence: [fewer than 20% of triples introduce unseen properties, and lower exact signature compliance points to new domain and range combinations for existing predicates rather than missing vocabulary](https://arxiv.org/html/2607.24551).
- Relevance: a directly reusable two-stage recipe for a supplied core ontology that needs extending from the corpus, on French-language legal text.

**A10. Ontology-Guided Knowledge Graph Construction from Structured Metadata and Climate Science Publications**
- Authors: Aayush Acharya, Qi Zhang, Eduard Dragut, Longin Jan Latecki
- First posted: arXiv v1 stamped 23 Oct 2026. Published: November 2026.
- Venue and links: IJCNLP-AACL 2026 long papers, pp. 5615 to 5642, [ACL Anthology 2026.aacl-long.252](https://aclanthology.org/2026.aacl-long.252/), [arXiv 2610.32725](https://arxiv.org/abs/2610.32725)
- Read: PARTIAL
- Method: triples with evidence spans go through three stages: a deterministic conformance test against [35 licensed type tuples, with no LLM](https://arxiv.org/html/2610.32725), evidence matching, and an LLM validator.
- Evidence: [2,000 papers; 133,990 nodes, 755,308 relations, 55 node types, 59 relation types; 98.96% judged correct by the LLM validator on 23,327 validated publication triplets; human and LLM kappa 0.640 to 0.700 on 1,000 triplets](https://arxiv.org/html/2610.32725). [Micro recall was 90.29% on the entities recoverable from text, and only 50.78% of curated gold entities appeared in the text at all](https://arxiv.org/html/2610.32725).
- Relevance: the largest corpus with a stage-by-stage ablation, and scientific reports with IPCC links. See the warning under the fourth key question about what the conformance filter discards.

#### Tier B: useful method or evidence, further from the domain

**B1. Better Later Than Sooner: Neuro-Symbolic Knowledge Graph Construction via Ontology-grounded Post-extraction Correction**
- Authors: Lorenzo Loconte, Timothy Hospedales, Cristina Cornelio
- First posted: not checked. Published: Findings of EMNLP 2026, October 2026.
- Link: [ACL Anthology 2026.findings-emnlp.1867](https://aclanthology.org/2026.findings-emnlp.1867/)
- Read: PARTIAL
- Method: open extraction, embedding-based canonicalisation of types and predicates, then targeted LLM correction of violations (OAK+MEND), under a Wikidata-derived ontology. I did not record the ontology size.
- Evidence: see the head-to-head table. Dataset is question-answering text, not reports.
- Relevance: tests the design question of whether to constrain before or correct after extraction.

**B2. Large Language Models for Knowledge Graph Extraction: A Schema-Constrained Evaluation Framework**
- Authors: Markus Ilves, Eduard Barbu, Jaan Übi
- First posted: none found. Published: May 2026.
- Venue and link: KG-LLM at LREC 2026, [ACL Anthology 2026.kallm-1.21](https://aclanthology.org/2026.kallm-1.21/), DOI 10.63317/3icoxh7axwxy
- Read: PARTIAL
- Evidence: [6 entity types and 96 relation types on DocRED with Pydantic validation; GPT-5.1 entity F1 84.0% but triple F1 21.7%; Mistral 7B 74.1% and 4.2%](https://aclanthology.org/2026.kallm-1.21.pdf). [Manual review of 10 documents found many valid triples absent from the gold standard, so the authors treat triple F1 as a lower bound](https://aclanthology.org/2026.kallm-1.21.pdf).
- Relevance: an evaluation warning, as low triple F1 can partly reflect incomplete gold labels.

**B3. Extracting ontology-compliant knowledge from scientific text describing irradiated materials using large language models (eolas)**
- Authors: Marco Luca Sbodio, Marcos Martínez Galindo, Vanessa Lopez, Blanca Biel, Pablo Canca, Pedro Delgado, Jesús I. Mendieta-Moreno, Raphael Tack, Maria J. Caturla
- First posted: arXiv v1 stamped 15 Sep 2026. Published: no venue found.
- Link: [arXiv 2609.17291](https://arxiv.org/abs/2609.17291)
- Read: PARTIAL
- Evidence: [168 experiments and 18,648 evaluated graphs; ontology of 14 classes, 111 instances, 13 object properties, 25 data properties; 111 evaluation passages; at error threshold 0.2 the best zero-shot success was 36.0% and the best few-shot 67.6%; a stricter threshold gave 15.3% and 45.9%](https://arxiv.org/html/2609.17291). [Labels-only schema verbalisation works only with examples, and few-shot prompts with good examples usually beat adding instances](https://arxiv.org/html/2609.17291).
- Relevance: the most systematic test of how to serialise a schema into a prompt.

**B4. Ontology-Guided Multi-Agent Extraction of Evaluation Objects from Academic Review Texts: Evidence from Chinese Library and Information Science**
- Authors: Haolin Chen, Hongyi Dong, Yu Zhu, Yijia Hong, Leiqing Niu, Jiyuan Ye
- First posted: arXiv listing 29 Aug 2026. Published: no venue found.
- Link: [arXiv 2608.29526](https://arxiv.org/abs/2608.29526)
- Read: PARTIAL
- Evidence: [precision 90.33%, recall 84.55%, entity F1 87.34%, strict typed F1 79.78%, type accuracy 91.35%; ontology of 4 top classes, 14 subclasses, 59 leaf types; 200 test documents](https://arxiv.org/pdf/2608.29526).
- Relevance: despite the name, it concerns objects being assessed in scholarly reviews, not programme evaluation. Evaluative text and the "Other" policy are nonetheless relevant. Texts are Chinese.

**B5. PolicyKG: An Agentic LLM Pipeline for Translating Institutional Policies into SHACL Knowledge Graphs**
- Authors: Ponkrit Kaewsawee, Chaklam Silpasuwanchai, Chutiporn Anutariya
- First posted: arXiv v1 stamped 10 Aug 2026. Published: under review at IJCKG 2026 per the arXiv comment.
- Link: [arXiv 2608.09028](https://arxiv.org/abs/2608.09028)
- Read: PARTIAL
- Method: policy PDFs to deontic labels to first-order logic to SHACL. A YAML "Corpus Adapter" grounds predicates in a target ontology.
- Evidence: [about 86.9% classification accuracy; shape correctness F1 .866 on 69 shapes; swapping in a GDPR registry raised property alignment from 1/15 to 11/15; Macro F1 fell from .810 to .370 on lease contracts](https://arxiv.org/html/2608.09028).
- Relevance: real institutional policy documents, and evidence that vocabulary registries transfer unevenly. It generates shapes rather than populating instances, so recommendations might fit better than findings.

**B6. ArticleMiner: Ontology-Guided Knowledge Graph Construction from Scientific Publications**
- Authors: Md Abrar Jahin, Craig A. Knoblock, Jay Pujara
- First posted: arXiv v1 stamped 22 Sep 2026. Published: Sci-K workshop at ISWC 2026, October 2026, per the arXiv journal reference.
- Link: [arXiv 2609.25607](https://arxiv.org/abs/2609.25607)
- Read: PARTIAL
- Evidence: [163 papers across four domains using human-written modules of 60 to 220 entries; paired means with Sonnet 4.6 against a same-LLM few-shot baseline: ChemTables 36.4 vs 27.6, MLTables 52.2 vs 37.8, DiSCoMaT 66.5 vs 51.1, GeoChem 63.2 vs 0.8](https://arxiv.org/html/2609.25607). [The authors say the geochemistry gain cannot be attributed to domain guidance alone, since that arm also had supplementary files](https://arxiv.org/abs/2609.25607).
- Relevance: tables in reports. Evaluation reports carry results tables that prose extractors miss.

**B7. Mapping Political-Elite Networks in Europe with a Multilingual Joint Entity-Relation Extraction Pipeline**
- Authors: Kirill Solovev, Jana Lasser
- First posted: 25 Jun 2026 (arXiv API record for v1); v4 stamped 11 Oct 2026. Published: no venue found.
- Link: [arXiv 2606.27347](https://arxiv.org/abs/2606.27347)
- Read: PARTIAL
- Evidence: [guided decoding against a domain ontology with Pydantic, vLLM and DSPy; 63.5% LLM-adjudicated semantic F1 on 74 German articles; a 100-article Polish reference; vocabulary built from literature plus recurrent types in unconstrained extraction, then frozen](https://arxiv.org/pdf/2606.27347).
- Relevance: the best multilingual constrained-decoding example, on news.

**B8. From Text to Triples: A Neuro-Symbolic Pipeline for Structured Claim Extraction**
- Authors: Massimiliano Fadda, Enrico Motta, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino
- First posted: none found. Published: CEUR Vol-4233, 2026-08-01; workshop May 2026.
- Link: [TEXT2KG 2026 paper](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_3.pdf)
- Read: PARTIAL
- Evidence: [F1 76.57 for DeepSeek-v3.1, 71.74 for qwen3-235b and 41.87 for gpt-4.1-nano, on annotated news about the war in Ukraine under the News Classification Ontology](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_3.pdf).
- Relevance: claim extraction is structurally close to findings extraction, with an agent and a statement. Small models did poorly here.

**B9. Ontology-Driven Knowledge Graph Construction from Text Using Large Language Models: A Nuclear Domain Case Study**
- Authors: Bokyeong Kim, Yonggyun Yu
- First posted: none found. Published: Transactions of the Korean Nuclear Society Spring Meeting, 7 to 8 May 2026. I found no DOI.
- Link: [KNS PDF](https://www.kns.org/files/pre_paper/55/26S-290-김보경.pdf)
- Read: PARTIAL
- Evidence: [DIAMOND has 615 classes, but the benchmark selected 58 concepts mapped to Wikidata; 89 test sentences; conformance above 0.93 for six 8B to 32B models; subject hallucination up to 0.292 even for models with high conformance](https://www.kns.org/files/pre_paper/55/26S-290-김보경.pdf).
- Relevance: a warning that conformance only checks that the relation is in the list, so hallucinated arguments can coexist with perfect conformance.

#### Tier C: abstract only (full text not read)

**C1. LLM-Based Knowledge Graph Construction for Cultural Heritage: An Extraction-Failure Diagnostic and CIDOC-CRM Benchmark.** Ruben Peeters, Xuemin Duan, Anastasia Dimou. First online 16 Oct 2026, EKAW 2026, LNCS vol. 16897, [DOI 10.1007/978-3-032-41614-8_7](https://doi.org/10.1007/978-3-032-41614-8_7). Paywalled. Evidence (fetch paraphrase): [post-hoc SHACL repair lifted an agent pipeline from F1 0.562 to about the single-prompt baseline's 0.585 on Text2KGBench; stage-level gating hurt 9 of 10 domains, by up to 37%; on 30 museum texts every variant was at or below F1 0.06, and repair raised parseable output to 27 of 30 without closing the accuracy gap; ten failure modes catalogued](https://link.springer.com/chapter/10.1007/978-3-032-41614-8_7). Relevance: the strongest counter-evidence to relying on validation.

**C2. From Text to Knowledge: An Ontology-Aware Triple Extraction Framework with LLMs.** Marco Calautti, Damiano Duranti, Paolo Giorgini. First online 29 Sep 2026, Advances in Conceptual Modeling (ER 2026), LNCS vol. 17125, [DOI 10.1007/978-3-032-40377-3_5](https://doi.org/10.1007/978-3-032-40377-3_5). Paywalled. Evidence (fetch paraphrase): [schema violations become feedback for repeated refinement; a 14B model with refinement beat a frontier model doing one pass on a Wikidata-derived benchmark, and the gains held on anonymised data](https://link.springer.com/chapter/10.1007/978-3-032-40377-3_5). The abstract gives no numbers.

**C3. Ontology-Aware Prompting for Knowledge Graph Construction from Text.** Balram Tiwari, Italo Lopes Oliveira, Asep Fajar Firmansyah, Hamada M. Zahera, Frank Hopfgartner, Axel-Cyrille Ngonga Ngomo. Published 2 Sep 2026, Studies on the Semantic Web vol. 63, pp. 87 to 105, [DOI 10.3233/SSW260009](https://ebooks.iospress.nl/doi/10.3233/SSW260009). Evidence: [multi-prompt candidates plus three rule-based filters (cross-prompt agreement, evidence grounding, surface-form alignment); up to 2.3 times LLM baselines on Text2KGBench; hallucination below 0.20 in most domains; conformance above 0.90 on most DBpedia-WebNLG domains](https://ebooks.iospress.nl/doi/10.3233/SSW260009). I did not confirm whether the volume is open access.

**C4. Deterministic and Trustworthy LLM-Driven Semantic Knowledge Graph Construction (DOPKG-SHACL).** Chukwudi Uwasomba, Nonso Nnamoko, Ioannis Korkontzelos. July 2026, IEEE ICCSC, DOI 10.1109/iccsc71566.2026.11650062. Abstract read from the [Edge Hill repository](https://research.edgehill.ac.uk/en/publications/deterministic-and-trustworthy-llm-driven-semantic-knowledge-graph/). Evidence: [deterministic IRI minting and ontology alignment had the strongest effect in an ablation; raw extraction varied substantially across runs](https://research.edgehill.ac.uk/en/publications/deterministic-and-trustworthy-llm-driven-semantic-knowledge-graph/). The abstract gives no figures.

**C5. Agentic schema-guided extraction of materials process knowledge from scientific literature (SciKGExtract).** Sameer Sadruddin, Jennifer D'Souza. Posted 28 Oct 2026, Research Square preprint, DOI 10.21203/rs.3.rs-11290953/v1, abstract read through Crossref. Evidence: [176 papers; best ZnO F1 rose from 0.591 to 0.805 with agentic refinement; best IGZO 0.344; a nested schema of 65 properties and 155 measurement nodes exposed process segmentation errors](https://doi.org/10.21203/rs.3.rs-11290953/v1).

**C6. Constraining LLM-Based RDF Generation with Ontology-Guided Prompting in Urban Agriculture.** Leonardo Vianna do Nascimento, José Palazzo Moreira de Oliveira. Published 8 Sep 2026, SBBD 2026, DOI 10.5753/sbbd.2026.249179, abstract read through Crossref. Evidence: [42 cultivar web pages with SHACL validation; the abstract says validity improved and hallucination fell, without figures](https://doi.org/10.5753/sbbd.2026.249179).

**C7. Large Language Model with Integrated Ontology and Inference Chain Constraints for Generative Information Extraction from Metallurgical Lifting Equipment Failure Reports.** Bin Zhou, Xingwang Shen, Jinsong Bao. Published 18 Jun 2026, Applied Sciences, DOI 10.3390/app16126178. The publisher page returned 403, so I read the abstract through Crossref. Evidence: [7 entity and 9 relation types, a stepwise chain from scene identification to triple output, character-level source localisation, and LoRA fine-tuning; no figures in the abstract](https://doi.org/10.3390/app16126178).

**C8. Ontology-Guided and LLM-Enhanced Knowledge Graph Construction from Unstructured Text.** Adham Kahlawi, William Pitchford, Sean Harkin, Nitish Malpotra. First online 1 Jun 2026, Intelligent Computing (CC 2026), LNNS vol. 1950, [DOI 10.1007/978-3-032-24807-7_1](https://doi.org/10.1007/978-3-032-24807-7_1). Paywalled. Evidence (fetch paraphrase): [scanned PDFs with printed and handwritten text, ontology-shaped prompts, page and context citations, a two-stage evaluation, and a medical case; no numbers in the abstract](https://link.springer.com/chapter/10.1007/978-3-032-24807-7_1). Relevance: noisy PDF input.

**C9. Extract, Align, Reflect: An LLM-Based Framework for Knowledge Graph Construction.** Hongfa Huang, Zhao Hong. First online 21 Jul 2026, ICIC 2026, LNCS vol. 16664, [DOI 10.1007/978-981-92-3426-4_13](https://doi.org/10.1007/978-981-92-3426-4_13). Paywalled. Evidence (fetch paraphrase): [open extraction, then schema alignment, then a reflection step, motivated by schema injection being limited by the context window; higher F1 with a schema and less redundancy without; no figures in the abstract](https://link.springer.com/chapter/10.1007/978-981-92-3426-4_13).

**C10. OntoExtend: A Framework for Requirement-driven and Scalable Ontology Extension with LLMs.** Anna Sofia Lippolis, Mohammad Javad Saeedizade, Stefan Schmid, Simon Blattner, Robin Keskisärkkä, Aldo Gangemi, Eva Blomqvist, Andrea Giovanni Nuzzolese. arXiv v1 stamped 20 Jul 2026, [arXiv 2607.17963](https://arxiv.org/abs/2607.17963). Evidence: [RAG over input ontologies and competency questions; 39 competency questions across Onto-DESIDE and a Bosch ontology; fragments passed all functional tests and needed minor to moderate revision by engineers](https://arxiv.org/html/2607.17963). Extension rather than extraction, but it is the natural partner for gap flagging.

**C11. Workshop abstracts from [CEUR Vol-4233, TEXT2KG 2026](https://ceur-ws.org/Vol-4233/), published 2026-08-01:** [CIExMAS](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_5.pdf) (Max Lautenbach, Sven Hertling) reached [up to 80% of the synthIE model's performance without fine-tuning, with property extraction as the main remaining challenge](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_5.pdf). [PrOntoNarr](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_12.pdf) (Francesco Orciuoli, Antonella Pascuzzo, Sabrina Senatore) populates a narrative ontology from political speeches. [DBpedia Hindi](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_18.pdf) (Sanju Tiwari and six others) links extracted predicates to DBpedia ontology properties. [Legal rule ontologies](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_10.pdf) (Galileo Sartor, Enrico Francesconi, Adam Wyner) use competency questions as validation while generating an ontology, not populating one.

**C12. DSE-RE: Dynamic Schema Evolution with LLM for Iterative Relation Extraction.** Tongyao Wang, Donghui Shi, Jose Aguilar, Jozef Zurada. Published 9 Jun 2026, Human-Centric Intelligent Systems 6:429-449, [DOI 10.1007/s44230-026-00158-1](https://doi.org/10.1007/s44230-026-00158-1). Open access, read through the fetch tool's summary of the first 100,000 of 126,990 characters. It [does not need a predefined ontology; type tables grow, are clustered and are fed back; F1 0.740 to 0.781 with GPT-4o on HSA and 0.741 to 0.771 on FOBIE; not tested on heterogeneous multilingual corpora](https://link.springer.com/article/10.1007/s44230-026-00158-1). Included as the contrast case to supplied schemas.

#### Tier D: title and metadata only, **not read**

All verified against Crossref on 2026-11-05 unless noted. The user can obtain these. CIKM entries carry a published date of 6 November 2026, one day after this search, with DOIs registered on 3 November.

| Paper | Authors | Date | Venue, identifier | Why obtain |
|---|---|---|---|---|
| Less Is More: Knowledge Graph Construction and Schema-Guided Canonicalization with LLMs | Hwangseon Park, Hayeon Yang, Dongyoung Son, Myoungseok Song, Hayoung Oh | 2026-11-06 | CIKM 2026, 10.1145/3799682.3839949 | Schema-guided canonicalisation; ACM page returned 403 |
| Schema-Driven Structured Information Extraction from Biomedical Literature via LLMs | Emanuele Cavalleri, Ali Rastegar Mojarad, J. Harry Caufield, Justin Reese, Christopher J. Mungall, Marco Mesiti | 2026-11-06 | CIKM 2026, 10.1145/3799682.3840835 | Schema-driven extraction |
| OntoForge: Continual Ontology and Knowledge Graph Construction via Multi-Agent Consensus and Self-Refining Procedures | Fang Guo, Li Zhu, Mengying Wu | 2026-11-06 | CIKM 2026, 10.1145/3799682.3839852 | Joint ontology and graph evolution |
| SHACL-in-the-Loop: Enhancing Structural Conformance in Agentic Knowledge Graph Construction | Vladislav Snytko, Rishabh Jakhar, Remzi Celebi | 26 Oct 2026 talk | [NLP4KGC at ISWC 2026 programme](https://5thnlp4kgc-code.github.io/5nlp4kgc/pages/programme.html), #10; no DOI or proceedings found | Directly on SHACL loops |
| Where, How, and What to Ground: Evaluating Ontology-Guided LLM-based Knowledge Graph Construction | Itisha Yadav, Roman Klinger, Diana Peters | 26 Oct 2026 talk | [Same programme](https://5thnlp4kgc-code.github.io/5nlp4kgc/pages/programme.html), #11; no DOI found | Appears to answer the design question directly |
| AutoKGen: Autonomous Knowledge Graph Construction through Agentic AI-driven Code Generation and Ontology-based Validation | Giorgos Anagnostou, Konstantinos Kotis | 26 Oct 2026 talk | [Same programme](https://5thnlp4kgc-code.github.io/5nlp4kgc/pages/programme.html), #8; no DOI found | Ontology validation |
| MOIE-KG: A multi-agent framework for ontology-guided open information extraction and knowledge graph construction | Xianhui Liu, Yueying Liu, Yiheng Zhuang, Wenlong Hou | Oct 2026 issue, registered 2026-09-29 | Knowledge-Based Systems, 10.1016/j.knosys.2026.117109 | Journal-level method |
| OntoGRC: ontology-guided mineral exploration knowledge extraction from geological texts using generate-reflect-correct prompting | Chunju Zhang, Chenxi Li, Shu Wang, Zhao Zhang, Yunqiang Zhu, RuiXue Xiong, Gang Wang, Maohong Chen, RenYi Chen, Yankai Yu | Aug 2026 issue, registered 2026-06-24 | Ore Geology Reviews, 10.1016/j.oregeorev.2026.107411 | Repair-style prompting |
| A Semantic Knowledge Graph Construction Method for AI Risks in Multi-Institutional Policy Texts | Wenjuan Sun | Sep 2026 | IEEE AICE, 10.1109/aice71085.2026.11736963 | Multi-institution policy text, close match |
| Verbatim-Anchored Knowledge Graph Construction via Semantic Functional Segmentation for Power Marketing Policy Documents | Long Peng, Wobin Lin, Yao Fu, Wenchong Guo | Jun 2026 | IEEE EESPE, 10.1109/eespe68405.2026.11648568 | Policy documents, source anchoring |
| Schema-Constrained Large Language Model Construction and Evolution of Electric Power Knowledge Graphs | Hui Xu, Libo Xing, Yuping Wang, Siyuan Leng, Kailing Guo | Sep 2026 | IEEE AICE, 10.1109/aice71085.2026.11736970 | Schema evolution |
| LLM-based Schema-Guided Extraction and Validation of Missing-Person Intelligence from Heterogeneous Data Sources | Joshua Castillo, Ravi Mukkamala | May 2026 | IEEE IDSS, 10.1109/idss69894.2026.11691381 | Heterogeneous sources |
| Ontology-Based Grounding and Feedback for Reliable LLM-Based Asset Administration Shell Generation | Martin B. Jensen, Tristan Schwörer, Sazvan Saeed, Zakka U. Rizqi, Casper Schou | Sep 2026 | IEEE ETFA, 10.1109/etfa68377.2026.11725111 | Feedback loop, industrial |
| Component-Aware Case Retrieval for Professional Ethics with Ontology-Constrained LLM Extraction | Christopher B. Rauch, Rosina O. Weber | 2026-08-01 | ICCBR proceedings per Crossref container, 10.1007/978-3-032-33865-5_13 | Ontology-constrained extraction |
| Ontology-Guided Relation Extraction from Financial Documents Using Fine-Tuned LLMs | Nader Bakir | 2026-10-26 | 10.51219/urforum.2026.nader-bakir; venue quality unassessed | Financial reports |
| Comparing LLM-Based Knowledge Graph Extraction Approaches on Literary Studies in Spanish | Federico Cortes | May 2026 | LREC-related volume, 10.63317/5kjis3dy8i7k | Spanish-language comparison |

#### Just outside the window, listed because they are clearly important

- **Wikontic: Constructing Wikidata-Aligned, Ontology-Aware Knowledge Graphs with LLMs.** Alla Chepurova, Aydar Bulatov, Mikhail Burtsev, Yuri Kuratov. ACL Anthology ID 2026.eacl-long.388 (EACL 2026, before May). It is the baseline used by [Loconte et al.](https://aclanthology.org/2026.findings-emnlp.1867.pdf) and [LMKG](https://aclanthology.org/2026.akbc-1.23.pdf). I read the abstract from a harvested Anthology record; date not independently checked.
- **OntoKG (arXiv 2604.02618), April 2026.** It treats schema extension as a feedback problem. **This rests on a search snippet only**; I did not open the page.
- **OntoLogX (arXiv 2510.01409; a Wiley version exists at DOI 10.1002/aisy.202501381).** SHACL feedback loop on cybersecurity logs. **Search snippet only**; publication date not verified.

### Inferences

- The field has moved past "put the ontology in the prompt" towards staged designs, with more of the work done by retrieval and deterministic checks.
- Workshop and low-profile venues hold most of the relevant work, so a sweep of only main conferences would miss most of it.

### Gaps

- **Coverage is incomplete.** The arXiv API returned 429 errors and my 32-query sweep never completed. OpenAlex returned 503, Semantic Scholar 429, and DBLP served a bot challenge. I worked from web search, from arXiv and Crossref and ACL Anthology records already harvested in this session by other researchers, and from direct volume pages.
- I did not sweep ISWC 2026 research track, ESWC 2026 main track, Journal of Web Semantics, TKDE or KDD 2026 proceedings. The NLP4KGC proceedings were not found.
- Some leads could not be verified as papers, so they are not catalogued, for example MFG-AgentGraph at ISWC 2026, seen only through a GitHub issue.
- arXiv IDs do not track first-posted months in this period: [2606.01208 is stamped 31 May](https://arxiv.org/abs/2606.01208). I used stamps rather than IDs throughout.

## What are the new methods for ontology-constrained LLM extraction: prompt design, retrieval of the relevant part of a large ontology, structured output, validation against SHACL or OWL, and repair loops?

### Takeaway

The common new shape is: extract or retrieve first, align to the ontology with search plus navigation, enforce structure with Pydantic or guided decoding, check deterministically, and repair only what fails. Evidence favours post-extraction correction and bounded repair over strict gating at every step, though validation does not rescue poor extraction.

### Cited Findings

Prompt design:
- [Adding semantic descriptions of types raised Gemma3 relation classification F1 from 0.51 to 0.70](https://arxiv.org/pdf/2609.31663).
- [Few-shot prompts with good examples usually beat adding ontology instances; labels-only schema listing is effective only with examples](https://arxiv.org/html/2609.17291).
- [Multiple prompts, then filtering on cross-prompt agreement and evidence](https://ebooks.iospress.nl/doi/10.3233/SSW260009), and [a stepwise reasoning chain ending in triple output](https://doi.org/10.3390/app16126178), are the main new patterns.

Structured output:
- [Pydantic with automatic retry](https://ceur-ws.org/Vol-4293/paper59.pdf), [Pydantic validation with schema-injected prompts](https://aclanthology.org/2026.kallm-1.21.pdf), and [JSON-schema guided decoding via vLLM and DSPy](https://arxiv.org/pdf/2606.27347) all appear.
- Structured output does not solve relations: [GPT-5.1 had entity F1 of 84.0% but triple F1 of 21.7% under a closed schema](https://aclanthology.org/2026.kallm-1.21.pdf).

Validation:
- [Conformance can be a plain set lookup against licensed type tuples, with no LLM](https://arxiv.org/html/2610.32725).
- [SHACL validation before persistence gave zero hard violations in all conditions](https://ceur-ws.org/Vol-4293/paper59.pdf), which measures structure and not truth.
- [Conformance can be perfect while subject hallucination stays at 0.138](https://www.kns.org/files/pre_paper/55/26S-290-김보경.pdf).
- [Evidence quotes resolved deterministically to character spans](https://aclanthology.org/2026.akbc-1.23.pdf) are a separate check from ontology validity.

Repair:
- [SHACL feedback capped at three retries, then keep and flag](https://arxiv.org/html/2606.01208) gave 5.2% non-compliance.
- [Violation feedback let a 14B model beat a frontier model doing a single pass](https://link.springer.com/chapter/10.1007/978-3-032-40377-3_5) (fetch paraphrase, no figures).
- [Post-hoc repair recovered F1 0.562 to about 0.585, whereas stage-level gating hurt 9 of 10 domains by up to 37%](https://link.springer.com/chapter/10.1007/978-3-032-41614-8_7) (fetch paraphrase).
- [Agentic refinement raised ZnO F1 from 0.591 to 0.805](https://doi.org/10.21203/rs.3.rs-11290953/v1), while the harder process only reached 0.344.
- [Deterministic IRI minting and ontology alignment had the strongest effect in an ablation](https://research.edgehill.ac.uk/en/publications/deterministic-and-trustworthy-llm-driven-semantic-knowledge-graph/), ahead of validation alone.

### Inferences

- Hypothesis, not tested here: gate late rather than early, because a chunk that fails an early stage loses everything downstream. Peeters et al. report this pattern, but the cause is their reading and I have not read the full paper.
- A SHACL pass rate is a poor headline metric for this project. Pair it with evidence checks and a sampled human truth check.

### Gaps

- No paper I read isolates each of prompt, retrieval, structured output and repair on one common benchmark. Yadav et al. may do this, but I have not read it.
- Most repair evidence is on sentence-level or short-text benchmarks.

## How large an ontology do they handle, and how do they avoid passing the whole ontology to the model?

### Takeaway

Successful use of a large ontology is rare. The largest with measured success in typing is 419 classes (Anchor), using retrieval and navigation, and a 915-class ontology reportedly failed outright until replaced with a 20-class schema. Most papers use small schemas of roughly 7 to 100 types.

### Cited Findings

Sizes reported:

| Paper | Size | How it reached the model |
|---|---|---|
| [Anchor](https://arxiv.org/html/2606.01208) | UCO 419 classes, STIX 109, MALOnt 75 | Embedding search, then recursive hierarchy navigation |
| [Rezayat et al.](https://link.springer.com/chapter/10.1007/978-3-032-40795-5_5) (fetch paraphrase) | 915 classes failed; 20 classes worked | Reduced to a purpose-built extraction schema |
| [Kim and Yu](https://www.kns.org/files/pre_paper/55/26S-290-김보경.pdf) | 615 classes, 58 selected | Permitted list in the prompt, restricted benchmark |
| [Dangaich et al.](https://arxiv.org/html/2607.28662) | 954 predicate label keys; class count not recorded | Per-document retrieval under a 5,000-token budget |
| [Neubürger et al.](https://ceur-ws.org/Vol-4293/paper59.pdf) | 58 classes, 120 object properties | Per-class questions; relation pass uses property definitions |
| [Acharya et al.](https://arxiv.org/html/2610.32725) | 55 node types, 59 relation types | Post-extraction check against licensed tuples |
| [Ilves et al.](https://aclanthology.org/2026.kallm-1.21.pdf) | 6 entity, 96 relation types | Whole schema injected |
| [Chen et al.](https://arxiv.org/pdf/2608.29526) | 4, 14 and 59 types by level | Coarse to fine classification |
| [Belfadel et al.](https://arxiv.org/pdf/2609.31663) | 16 classes, 21 relations | Whole schema injected |

Ways of avoiding the whole ontology:
- Retrieval by chunk: [11,200 tokens fell to 700 on one document, using windows ranked by proper-noun density, class and predicate embeddings, a PageRank penalty and subclass expansion](https://arxiv.org/html/2607.28662).
- Search plus navigation: [search alone worked for entities (0.9184) but not predicates (0.3142); hybrid reached 0.9364 and 0.7843](https://arxiv.org/html/2606.01208).
- Extract first, then align: [open extraction, then canonicalisation, then correction](https://aclanthology.org/2026.findings-emnlp.1867.pdf), and [Extract, Align, Reflect](https://link.springer.com/chapter/10.1007/978-981-92-3426-4_13).
- Reduce the schema: [a separate extraction schema from the domain ontology](https://link.springer.com/chapter/10.1007/978-3-032-40795-5_5).
- Ask by class: [one set of questions per class, each with its own chunk retrieval](https://ceur-ws.org/Vol-4293/paper59.pdf).

Scale effects:
- [The advantage of retrieval shrinks on small schemas: 62.5% on UCO, 0.7% on MALOnt, and Anchor is below a baseline on MALOnt predicates](https://arxiv.org/html/2606.01208).
- The authors attribute the whole-schema decline to context exhaustion and "lost in the middle". That is their explanation; they measured the score drop across different schemas, not the mechanism.

### Inferences

- For an evaluation ontology of perhaps tens of concept types, whole-schema injection may be fine for classes. Controlled vocabularies such as countries, programmes and SDG targets likely belong in a linking step rather than in the prompt. This is my reasoning, not a tested result.
- Predicates deserve separate treatment, since every paper that split them found them harder.

### Gaps

- No paper reports extraction with an ontology of tens of thousands of terms. Loconte et al. use a Wikidata-derived ontology, but I did not record its size.
- The 94% token figure is from one document, and the authors [say results lack significance testing](https://arxiv.org/html/2607.28662).

## What do head-to-head evaluations show comparing schema-guided against schema-free extraction, with numbers?

### Takeaway

On closed benchmarks, guidance wins clearly on conformance and moderately on F1, but those benchmarks favour guidance by definition. Applying the ontology after extraction gave validity as good as or better than putting it in the prompt in one study, with more triples on one model and fewer tokens on the other.

### Cited Findings

Same-system comparison, [LMKG Table 2, gpt-5.4-mini](https://aclanthology.org/2026.akbc-1.23.pdf):

| Text2KGBench split | Ontology | P | R | F1 | Conformance | Relation hallucination |
|---|---|---|---|---|---|---|
| Wikidata-TekGen | off | 0.10 | 0.08 | 0.08 | 0.11 | 0.89 |
| Wikidata-TekGen | on | 0.33 | 0.31 | 0.31 | 0.97 | 0.03 |
| DBpedia-WebNLG | off | 0.25 | 0.14 | 0.17 | 0.37 | 0.63 |
| DBpedia-WebNLG | on | 0.43 | 0.36 | 0.39 | 1.00 | 0.00 |

[The authors say part of this gap is definitional, because open runs produce their own labels and so score badly on measures defined against the ontology](https://aclanthology.org/2026.akbc-1.23.pdf).

When to apply the ontology, [Loconte et al. Table 1, HotpotQA](https://aclanthology.org/2026.findings-emnlp.1867.pdf):

| Model | Method | Triples | % valid | Weighted tokens (millions) |
|---|---|---|---|---|
| Qwen3-30B | KGGen (no ontology) | 155,391 | n/a | 28.9 |
| Qwen3-30B | Wikontic | 117,891 | 77.4 | 45.3 |
| Qwen3-30B | Ontology in context | 173,942 | 79.4 | 35.8 |
| Qwen3-30B | OAK before correction | 127,763 | 63.4 | 16.3 |
| Qwen3-30B | OAK+MEND | 127,108 | 96.8 | 26.8 |
| GPT-OSS-120B | Wikontic | 110,109 | 99.0 | 87.3 |
| GPT-OSS-120B | Ontology in context | 67,197 | 97.9 | 46.2 |
| GPT-OSS-120B | OAK+MEND | 112,737 | 98.4 | 69.2 |

Other comparisons:
- [Whole-schema prompting scored 0.4439 on UCO against 0.7347 for retrieval, with non-compliance 43.9% against 5.2%](https://arxiv.org/html/2606.01208). These are different systems, not one toggle.
- These compare pipelines, not guidance on and off: [on 25 ClimateIE papers, using the ClimateIE ontology for both systems, LLM-judged precision was 75.0% for ClimateIE and 92.9% for Acharya et al., but with 5,729 triplets against 637](https://arxiv.org/html/2610.32725). Against KARMA, [92.3% and 99.1%, with 3,820 and 17,141 triplets](https://arxiv.org/html/2610.32725).
- [A same-LLM few-shot baseline scored 51.1 against 66.5 on DiSCoMaT](https://arxiv.org/html/2609.25607), with [wide uncertainty on the smaller benchmarks](https://arxiv.org/abs/2609.25607).
- [EAR reports higher F1 with a schema and less redundancy without](https://link.springer.com/chapter/10.1007/978-981-92-3426-4_13), with no figures in the abstract.
- [Conformance reached 68 to 70% and 89% with near-zero hallucination, while triple F1 stayed at 22 to 26% and 35 to 40%](https://arxiv.org/html/2607.28662), which the authors call under-extraction.
- [Automatic triple F1 is a lower bound, given incomplete gold data](https://aclanthology.org/2026.kallm-1.21.pdf).
- A contrary caution: [supplying graphs rather than passages lowered downstream question answering](https://aclanthology.org/2026.akbc-1.23.pdf), and fact recovery on MINE-1 was measured in open-schema mode.

### Inferences

- Do not expect a schema-free run to be fairly scored on conformance. For this project, compare both modes on human-judged truth and on usefulness for intended questions, not on benchmark F1.
- Guidance improves consistency more certainly than it improves recall. Measure what is lost: [stricter pipelines can output far fewer triples](https://arxiv.org/html/2610.32725).

### Gaps

- I found no head-to-head on long reports with a human gold standard. Text2KGBench is sentence-level, and [LMKG ran it without chunking](https://aclanthology.org/2026.akbc-1.23.pdf).
- Yadav et al. and Calautti et al. may contain relevant numbers, but I have not read them.

## How do they handle text that does not fit the ontology, and ontology extension or feedback when gaps are found?

### Takeaway

Handling ranges from silently filtering to flagging for human review, and very little work closes the loop automatically into the ontology. Filtering can discard a lot of correct information, so it needs measuring.

### Cited Findings

- **Filtering:** [the conformance stage removed 11,050 triplets, of which 93.4% were judged textually correct, because their type signatures lay outside the ontology](https://arxiv.org/html/2610.32725). This is the most important single number on this question.
- **Flag for review:** [novel types and predicates are marked and surfaced as candidate extensions in an admin panel](https://arxiv.org/html/2607.28662).
- **Conservative mapping:** [predicates snap to the vocabulary only at a very high similarity, because generic predicates otherwise attract confident wrong matches](https://arxiv.org/html/2607.28662).
- **Generic fallback:** [mappings below confidence thresholds default to owl:Thing or generic properties, with schema augmentation left as future work](https://arxiv.org/html/2606.01208).
- **"Other" with evidence:** [the label is allowed only after considering context and rules, and must retain evidence and uncertainty to support later ontology refinement](https://arxiv.org/pdf/2608.29526).
- **Coverage-driven loop:** [gaps generate follow-up questions until improvement falls below 2% or answerability exceeds 80%](https://ceur-ws.org/Vol-4293/paper59.pdf). It deepens population without changing the schema, since [class coverage rose only 2 points, and co-evolution is future work](https://ceur-ws.org/Vol-4293/paper59.pdf). Coverage stayed under 40% of classes in every condition.
- **Corpus-driven extension before closed extraction:** [fewer than 20% of triples bring unseen properties, while new domain and range combinations are the larger issue](https://arxiv.org/html/2607.24551).
- **Requirements-driven extension:** [competency questions plus RAG over input ontologies produced fragments needing minor to moderate revision](https://arxiv.org/html/2607.17963).
- **Schema redesign:** [the ontology is refined during construction for extractability](https://link.springer.com/chapter/10.1007/978-3-032-40795-5_5).
- **Fully evolving schema:** [clustered type tables are fed back into prompts](https://link.springer.com/article/10.1007/s44230-026-00158-1), without a supplied ontology.
- **Upstream failures:** [the shortfall on museum text comes from flattened events and lost temporal, figurative and subjective content, not from validation](https://link.springer.com/chapter/10.1007/978-3-032-41614-8_7) (fetch paraphrase).

### Inferences

- A sensible design for this project, untested here: run an open pass on a stratified sample of agencies and languages, as in [Montenegro et al.](https://arxiv.org/html/2607.24551), then closed extraction with novel-type flagging, and log every rejected triple with its reason so the ontology team can review the volume.
- Evaluation findings are often qualified and judgemental, so the museum-text failure modes may be more predictive than Text2KGBench results. This is a hypothesis.

### Gaps

- No paper I read measured whether an automated gap-to-ontology loop improves later extraction.
- OntoForge and the Xu et al. schema evolution paper may address this, but are unread.

## Which of them were tested on long reports, or on policy, government, legal, scientific or industrial documents similar to evaluation reports?

### Takeaway

None was tested on UN evaluation reports or at anything near 1.9 million chunks. The largest corpora I saw are 2,000 scientific papers and 643 accident reports. The evidence on long documents and multilingual text is thin.

### Cited Findings

Policy and government:
- [Dutch National Police policy excerpts, vision statements and annual reports](https://aclanthology.org/2026.kallm-1.20.pdf).
- [Asian Institute of Technology policies, with GDPR and lease transfer](https://arxiv.org/html/2608.09028).
- [Humanitarian ReliefWeb and EM-DAT sources](https://arxiv.org/html/2609.00858).
- Unread but apparently close: Sun on multi-institutional AI risk policy texts, and Peng et al. on power marketing policy documents (Tier D).

Legal:
- [French maintenance regulations from Légifrance](https://arxiv.org/html/2607.24551).
- [17 German federal statutes](https://ceur-ws.org/Vol-4293/paper59.pdf), used as a domain-transfer demonstration without a gold standard.

Industrial and safety reports:
- [33 nuclear documents, including safety reports and licence applications](https://ceur-ws.org/Vol-4293/paper59.pdf).
- [643 aviation accident reports](https://link.springer.com/chapter/10.1007/978-3-032-40795-5_5).
- [80 French power-grid reports](https://arxiv.org/pdf/2609.31663), [metallurgical failure reports](https://doi.org/10.3390/app16126178).
- [A 45-page intelligence report and a 10-page naval OCR document](https://arxiv.org/html/2607.28662).

Scientific:
- [2,000 climate publications](https://arxiv.org/html/2610.32725), [163 papers with tables and supplementary files](https://arxiv.org/abs/2609.25607), [176 materials papers](https://doi.org/10.21203/rs.3.rs-11290953/v1), [111 passages on irradiated materials](https://arxiv.org/html/2609.17291).

Long-document and noisy-input methods:
- [Fold and enrich, with structure-aware chunks](https://aclanthology.org/2026.akbc-1.23.pdf). MINE-1 is 100 articles, not long reports.
- [8,000-character chunks with 1,000 overlap, an explicit overlap note in the prompt, per-page OCR routing and format-specific handlers](https://arxiv.org/html/2607.28662).
- [Scanned PDFs with handwriting, using page-level citations](https://link.springer.com/chapter/10.1007/978-3-032-24807-7_1) (fetch paraphrase).

Language coverage:
- French: [Belfadel et al.](https://arxiv.org/pdf/2609.31663) and [Montenegro et al.](https://arxiv.org/html/2607.24551). German: [Neubürger et al.](https://ceur-ws.org/Vol-4293/paper59.pdf). German and Polish: [Solovev and Lasser](https://arxiv.org/pdf/2606.27347). Hindi: [Tiwari et al.](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_18.pdf). Chinese: [Chen et al.](https://arxiv.org/pdf/2608.29526). OCR of German and Spanish scans: [Dangaich et al.](https://arxiv.org/html/2607.28662). Spanish literary studies: Cortes (unread).
- I found no paper mixing several languages under one ontology on report-style text. [DSE-RE's authors note it has not been tested on heterogeneous multilingual corpora](https://link.springer.com/article/10.1007/s44230-026-00158-1).

Model size on such corpora, where the results disagree:
- [2B multi-pass beat 31B on ontology-aligned metrics in one three-run study](https://ceur-ws.org/Vol-4293/paper59.pdf), while [lightweight models did much worse on claim extraction](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_3.pdf). Task and metric differ, and the first study did not assess factual correctness.

### Inferences

- At 1.9 million chunks, per-chunk multi-stage pipelines look costly. Question-driven retrieval, as in [Neubürger et al.](https://ceur-ws.org/Vol-4293/paper59.pdf), follows a different cost model, since it targets chunks per question rather than reading everything. But it would miss what no question asks about. I have no measured throughput at this scale.
- Findings and recommendations may need an initial step that locates the relevant passages before ontology-constrained extraction, since [chunk size strongly affected recall](https://arxiv.org/html/2607.28662) in the one study that measured it, on one document.
- A practical first test would combine policy-style human agreement measurement, as in [Kolk et al.](https://aclanthology.org/2026.kallm-1.20.pdf), with a small multilingual gold set across agencies.

### Gaps

- I found no peer-reviewed evaluation on UN, development or programme evaluation reports with ontology-guided extraction. Given the search limits above, this is "not found", not "does not exist".
- I found no reported cost or throughput figures relevant to millions of chunks. The climate paper has a cost appendix (A.11) that I did not read.
