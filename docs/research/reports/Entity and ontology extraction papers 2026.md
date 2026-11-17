# Read these 2026 papers before modelling UNEG chunks

New work is plentiful on both tracks, but the five research notes behind this report found no paper in the May to November 2026 window that builds an ontology or knowledge graph from UN evaluation reports. For discovering an ontology from chunks, start with **SCOPE and SCION**, the **French legal pipeline** and **GrOIL**. For mapping text to an ontology you supply, start with the **UZH shared task on UN resolutions**, UNDP's **AIDA** chapter, and the **German subject-indexing benchmark by Kähler et al.**, then the design papers by **Neubürger et al.** and **Kim et al. (Anchor)**. The evidence favours sampling chunks, inducing a vocabulary, freezing it and then extracting against it, but no paper has tested that sequence on this kind of corpus. Relations are the weak point on both tracks: the best non-taxonomic relation F1 in the main ontology-learning shared task is 0.0308, on synthetic text ([LLMs4OL overview](https://doi.org/10.52825/ocp.v11i.3728)). Large ontologies are also rarely handled well: the largest measured success in the notes is 419 classes ([Anchor](https://arxiv.org/html/2606.01208)). Coverage is incomplete because OpenAlex, Semantic Scholar, DBLP and the arXiv API each failed for at least some researchers, and several venues were not swept, so "not found" below never means "does not exist". Many of the most relevant papers are unread and are listed separately with their identifiers.

## Every entry separates first-posted from published dates and states how it was read

**Read status.** **Full** means the whole text was read. **Part** means only named sections, tables or extracts were read, including "main body" and "targeted" reads. **Abstract** means the abstract only. **Metadata** means title and bibliographic data only, so the paper is not read. Where researchers read different amounts, both are shown, fuller first, with the note that recorded each. Numbers come from the fuller read.

**Dates.** "First posted" is the earliest preprint date the notes established, from the arXiv v1 stamp or the preprint server, never from the arXiv ID. "Published" is the venue or journal date. **PRE** marks a first posting before May 2026. "None found" means a title search found nothing, which is not proof that nothing exists. "Not checked" means nobody looked.

**Flags on numbers.** **†** means the number or status passed through a summarising fetch tool, so check it against the paper before quoting. **‡** means it rests on a search snippet. All numbers are the authors' own unless labelled otherwise, and nobody re-ran anything.

**Note codes.** Each entry names its source in `research_notes/Entity and ontology extraction papers 2026/`: **OL** is `ontology_learning_from_text.md`, **OG** is `ontology_guided_extraction.md`, **SF** is `schema_free_extraction.md`, **MV** is `mapping_to_supplied_vocabularies.md`, and **DJ** is `domain_and_journal_sweep.md`.

## Track A: when no ontology is supplied, read SCION, the French legal pipeline and GrOIL first

The ranking below is my judgement of fit to long, heterogeneous report chunks. It is not a measured result, because none of these papers was tested on such a corpus.

| # | Paper | Why read it for UNEG-style chunks | Read | First posted | Published | Note |
|---|---|---|---|---|---|---|
| 1 | **SCOPE and SCION**, Hu et al. [arXiv 2607.21610](https://arxiv.org/abs/2607.21610) | Closest chunk-to-schema design. Chunks of 1,200 tokens are aggregated into a source-level candidate package before constrained LLM naming and merging. Macro Graph F1 0.7888 against 0.7629 for extract-then-aggregate over 24 sources, averaging $0.12 per source. Not tested on long noisy reports. | Part (main body, OL) / Abstract (DJ) | 2026-05-20 | None found | OL, DJ, SF |
| 2 | **LLM-Assisted Ontology Engineering of a French Legal KG**, Montenegro et al. [arXiv 2607.24551](https://arxiv.org/abs/2607.24551) | The most reusable recipe on real text, and it bridges both tracks: open extraction on a stratified sample, label fusion at cosine 0.7, then closed extraction. Two LLMs induced 75 and 44 properties, sharing only 21. | Part (main body, OL; partial, OG) | 2026-07-27 | SEMANTiCS 2026 posters and demos, per the paper; proceedings entry unconfirmed | OL, OG |
| 3 | **GrOIL**, Mridul et al. [arXiv 2608.22135](https://arxiv.org/abs/2608.22135), [DOI](https://doi.org/10.1145/3799682.3840910) | Documents to OWL TBox with provenance, and the LLM makes only local bounded decisions. SPARQL competency-question coverage 0.8527 and 0.7694 against 0.6304 and 0.4027 for a direct LLM. The 100-contract corpus is synthetic and covers one domain. | Part (OL) / Abstract (DJ) | 2026-08-22 | CIKM '26; Crossref online date 2026-11-06, a day after the search | OL, DJ |
| 4 | **TaxCE**, Mukku et al. [arXiv 2608.30614](https://arxiv.org/abs/2608.30614), [DOI](https://doi.org/10.18653/2026.emnlp-industry.76) | Best cost evidence at scale: a local small model per document plus about 140 frontier API calls per taxonomy regardless of corpus size; 1M documents in under 10 hours on one 24 GB GPU plus an API. Output is a topic taxonomy, not typed relations. | Part (OL) / Abstract (DJ) | 2026-08-31 | EMNLP 2026 Industry (Crossref year only) | OL, DJ |
| 5 | **Learning to structure data from user-generated thematic corpora**, Avram et al. [arXiv 2610.01463](https://arxiv.org/abs/2610.01463) | Direct comparison of consolidation strategies: frequency-ordered incremental merging F1 0.32, bottom-up 0.13, global grouping 0.08, pairwise 0.08, all low. Gives a usable stopping rule. Attributes, not relations; 1,000 documents per corpus. | Part | 2026-10-01 | None found | OL |
| 6 | **TERMNET**, Senger et al. [arXiv 2606.26963](https://arxiv.org/abs/2606.26963) | The one paper built around style differences between sources, with a policy-analysis use. Over 1M English and German documents. Term input lifts source entropy from 24.50 to 30.00, and the full method reaches 36.70. Its top two layers are expert-curated. | Part | 2026-06-25 | None found | OL |
| 7 | **Curate Before You Connect**, Dangaich et al. [arXiv 2608.10644](https://arxiv.org/abs/2608.10644) | Closest corpus type: 98,795 government documents with tables, 537,157 entities. One wrong merge deleted eight entities from an unrelated document, so resolution was limited to flagging. No identity benchmark is reported, so treat it as engineering experience. | Part | 2026-08-11 | None found | SF |
| 8 | **Beyond Chunk-Local Extraction (CrossAug)**, Zhang et al. [arXiv 2605.28004](https://arxiv.org/abs/2605.28004) | The most direct attack on relations that span chunks, with a bounded LLM budget. Adds 5,048-8,901 triples, but HippoRAG2 F1 rises only +1.33 and +1.09. | Part (SF) / Abstract (DJ) | 2026-05-27 | None found | SF, DJ |
| 9 | **LLMs4OL 2026 Overview**, Babaei Giglou et al. [DOI](https://doi.org/10.52825/ocp.v11i.3728) | The benchmark to read sceptically. Best graph similarity 0.7762 with no ontology and 0.9806 with a supplied vocabulary, but on synthetic short passages, and 15 of 19 entries scored exactly 0.0000 on non-taxonomic relations. | Part (OL, DJ) / Abstract (MV) | No preprint recorded | 2026-11-02 | OL, DJ, MV |
| 10 | **AutoSchema**, Zhu et al. [DOI](https://doi.org/10.1145/3770855.3817812) **(full text not read)** | Clearest measure of schema instability across runs: mean pairwise Jaccard across five seed sets is 19.97%, or 43.41% after normalisation. Obtain this one first. | Abstract | None found | KDD 2026, online 2026-08-08 | OL |
| 11 | **TRACE-KG**, Abolhasani et al. [arXiv 2604.03496](https://arxiv.org/abs/2604.03496) | Induces a schema jointly with the graph on long technical standards. Joint runs cut entity classes from 124 to 64. Authors say 876 LLM calls for 339 chunks does not establish scalability. | Part | **PRE** 2026-04-03 | LoG 2026, PMLR 338, per arXiv comment | OL |
| 12 | **AutoSchemaKG**, Bai et al. [arXiv 2505.23628](https://arxiv.org/abs/2505.23628), [ACL Anthology](https://aclanthology.org/2026.acl-long.942/) | The only typed-schema system well beyond your scale, about 78,400 GPU hours in total. A competitor's run reports 36.5% entity-text leakage, which is not a neutral finding. Old idea, new venue. | Part (OL) / Abstract (SF) | **PRE** 2025-05-29 | ACL 2026, Jul 2026 | OL, SF |

**Taken together, these papers describe a recurring recipe rather than a settled method.** They extract locally, restrict the candidate vocabulary, cluster by embedding, let an LLM name and merge only inside clusters, then freeze the schema before full extraction (OL). Drift is large wherever it is measured. AgriEvoKG found 23,727 and 19,139 distinct relation labels in two sectors, and 62.5% of relations from 50 new passages used new types ([AgriEvoKG](https://doi.org/10.62762/tetai.2026.882738), abstract only). No paper measures type drift between chunks of the same long document (OL), which is the exact condition your corpus creates.

**Scale and typing have not yet met in one paper.** Taxonomy-style methods have reached millions of documents, while typed-schema methods with gold evaluation have not (OL). LLMs4OL also shows a disagreement worth reading for yourself: its teams found after-the-fact canonicalisation gave little benefit, while the French legal, TRACE-KG and TaxCE papers report clear value from merging (OL). The OL researcher's explanation, that the two sides measure different things, is an untested hypothesis.

## Track B: when you supply the ontology, read the UN resolutions task, AIDA and Kähler first

| # | Paper | Why read it for UNEG-style chunks | Read | First posted | Published | Note |
|---|---|---|---|---|---|---|
| 1 | **UZH Shared Task 2026 on UN Resolutions**, Shaitarova et al. [ACL Anthology](https://aclanthology.org/2026.argmining-1.10/) | The one in-window mapping task on real UN text, in French with machine English translation. 141 tags; best micro-F1 0.459, from a system that quotes evidence per tag with no retrieval pre-filter. 20 of 178 reviewed paragraphs had real taxonomy gaps. Models were limited to 8B parameters and labels are silver. | Part | Not looked up | ArgMining at ACL 2026, Jul 2026 | MV |
| 2 | **AIDA: Assessment of development analytics**, Jones et al. [DOI](https://doi.org/10.4324/9781003799139-20) | The same document type: over 7,000 UNDP evaluation reports averaging about 70 pages, roughly 20% French or Spanish. Classifier F1 0.85 for findings, conclusions, recommendations and taxonomy tags, from 500 hand-labelled evaluations. No graph or ontology in the part read, and that F1 belongs to the older version. | Part | None found | 2026-08-18, *From Algorithms to Evidence* (Routledge) | DJ |
| 3 | **Does generative AI supersede supervised XMLC?**, Kähler et al. [arXiv 2607.14882](https://arxiv.org/abs/2607.14882) | The most useful single paper on the design choice. On full text against a 204,056-entity vocabulary, XR-Transformer F1 0.398 beats LLM-Ensemble 0.364, but expert-graded g-F1 flips it, 0.512 against 0.601. Inference 10.5 ms against 998 ms per record. | Full† (key rows quoted) | 2026-07-16 | Submitted to KONVENS 2026 | MV |
| 4 | **From Documents to Decisions**, Neubürger et al. [CEUR Vol-4293 paper 59](https://ceur-ws.org/Vol-4293/paper59.pdf) | Closest architecture: a local-model, question-driven loop with SHACL over 33 nuclear documents, 3,004 segments, 58 classes and 120 properties. Five passes lift relation coverage from 41.3% to 74.3% but class coverage only from 33.7% to 35.7%. Factual correctness was not assessed. | Full | None found | CEUR 2026-10-13, for SEMANTiCS Sep 2026 | OG |
| 5 | **Anchor**, Kim et al. [arXiv 2606.01208](https://arxiv.org/abs/2606.01208) | Best evidence on avoiding passing a whole ontology. On UCO (419 classes), entity typing F1 0.7347 against 0.4521; predicates are harder at 0.5180. With ground-truth inputs, search alone reached 0.9184 on entities but 0.3142 on predicates. Its "schema-agnostic" title is misleading. | Part (OG) / Abstract (OL) | 2026-05-31 | None found | OG, OL |
| 6 | **Ontology-Guided, Deduplication-Aware Extraction Layer**, Dangaich et al. [arXiv 2607.28662](https://arxiv.org/abs/2607.28662) | Most detail on noisy PDFs, per-page OCR routing, retrieving an ontology slice and flagging novel types. Catalogue tokens fell from about 11,200 to about 700 on one document. Chunks of 2 rather than 5 pages raised recall from 29.4% to 58.8% on one OCR document. No significance testing. | Part (OG) / Abstract (OL, DJ) | 2026-07-22 | None found | OG, OL, DJ |
| 7 | **Ontology-Guided KG from Climate Science Publications**, Acharya et al. [arXiv 2610.32725](https://arxiv.org/abs/2610.32725), [ACL Anthology](https://aclanthology.org/2026.aacl-long.252/) | The most important number on text that does not fit: the conformance stage removed 11,050 triplets, of which 93.4% were judged textually correct. 2,000 publications with a stage-by-stage ablation. | Part (OG) / Abstract (DJ) | 2026-10-23 | IJCNLP-AACL 2026, Nov 2026 | OG, DJ |
| 8 | **Select, Don't Train**, Polat et al. [arXiv 2608.27470](https://arxiv.org/abs/2608.27470) | Mapping without training a retriever: BM25 plus an LLM selector lifts inKB micro-F1 from 82.3 to 86.3. Abstention is weak: recall 45.3 on cases where no candidate is right. | Full† (abstention row quoted) | 2026-08-17 | Accepted to ISWC 2026 Research Track; proceedings page not checked | MV |
| 9 | **Causal Evidence Extraction in Crisis Reports**, Zhang and Oussalah [DOI](https://doi.org/10.18653/v1/2026.findings-acl.1626), [arXiv 2608.04576](https://arxiv.org/abs/2608.04576) | The most reusable extraction target for evaluation reports: intervention, outcome, direction and strength. Weighted F1 90.73% and 94.15% (fine-tuned Llama-3.1-8B) on 100 expert-annotated reports described as "long, noisy, and multi-topic". | Abstract | 2026-08-05 | Findings of ACL 2026, Jul 2026 | DJ, MV |
| 10 | **From UNDRR Reports to Event Records**, Andreozzi et al. [arXiv 2609.23853](https://arxiv.org/abs/2609.23853) | UN-hosted grey literature at real scale, with the best failure list. Pooled attribute F1 86.0% for GPT-5, but complete-match accuracy only 44.4%. Only the first 12,000 characters of each document were read. | Full† (headline numbers confirmed against the abstract) | 2026-09-20 | Not found | DJ |
| 11 | **Extraction-Oriented Ontology Modeling**, Rezayat et al. [DOI](https://doi.org/10.1007/978-3-032-40795-5_5) **(full text not read)** | Directly challenges passing a large ontology at all: on 643 aviation accident reports, a 915-class ontology failed with local LLMs while a 20-class schema worked†. High priority to obtain. | Abstract† (paywalled) | None found | ER 2026, online 2026-10-05 | OG |

**Guidance beats free extraction on closed benchmarks, but those benchmarks favour guidance by definition.** On LMKG's same-system comparison, F1 goes from 0.17 to 0.39 on DBpedia-WebNLG with the ontology on, and the authors say part of the gap is definitional ([LMKG](https://aclanthology.org/2026.akbc-1.23.pdf)). Applying the ontology after extraction also works: OAK+MEND reached 96.8% validity from 63.4% before correction, at 59% of Wikontic's token cost on Qwen3-30B ([Loconte et al.](https://aclanthology.org/2026.findings-emnlp.1867.pdf)). Validation does not rescue poor extraction, though. Conformance can be perfect while subject hallucination stays at 0.138 ([Kim and Yu](https://www.kns.org/files/pre_paper/55/26S-290-김보경.pdf)), and post-hoc repair reportedly lifted one pipeline only to about the single-prompt baseline† ([Peeters et al.](https://link.springer.com/chapter/10.1007/978-3-032-41614-8_7)).

**For vocabulary mapping, the strongest pattern is simple retrieval with costly steps kept for uncertain cases.** SelectED calls an LLM on about 20% of mentions and beats calling it on every mention ([SelectED](https://aclanthology.org/2026.uncertainlp-main.9/)). Candidate narrowing has a measured cost: in the UZH task, 20 candidates gave micro-F1 0.226 and 40 gave 0.396 (MV). Two warnings bear on agency vocabularies. LLMs over-assign SDG labels by 1.15 to 1.81 times against Elsevier's mapping ([Charalampous et al.](https://doi.org/10.3390/su182010316)). In Arabic-English matching between different ontologies, precision falls to about 0.11 because LLMs rarely answer NO_MATCH ([Fallatah](https://om.ontologymatching.org/2026/papers/om2026-STpaper3.pdf)).

## The fuller catalogue spreads across preprints, workshops and journals

Short-list papers are not repeated here. Papers known only by title are in the unread section, not here. Catalogue papers marked **Abstract** were not read in full, and those blocked by a publisher appear again in the unread section.

### Track A: ontology, schema and taxonomy induction

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| CORTEX, Gan et al. [arXiv 2606.30175](https://arxiv.org/abs/2606.30175), [DOI](https://doi.org/10.18653/2026.emnlp-main.985) | 2026-06-29 | EMNLP 2026 Main (Crossref year only) | Part | OL | Not seed-free: 196 Wikipedia-derived chains grow to 403 on a sample, then freeze. 73.15M documents; validated only downstream. |
| SC-Taxo, Cai et al. [arXiv 2605.00620](https://arxiv.org/abs/2605.00620) | 2026-05-01 | None found | Part (OL) / Abstract (DJ) | OL, DJ | Hierarchy labelling on scientific literature. TaxoBench Purity 64.8, NMI 50.2, CEDS 28.7. |
| Generative Ontology Induction, Sergienko [arXiv 2607.16201](https://arxiv.org/abs/2607.16201) | 2026-05-01 | None found | Part (OL) / Abstract (DJ) | OL, DJ, SF | Coverage of 95 to 100% uses the author's own metric, not gold. No consolidation step. Weak evidence. |
| TaxoScale, Fereidouni et al. [arXiv 2608.09049](https://arxiv.org/abs/2608.09049) | 2026-08-10 | None found | Part | OL | Over 600K reviews. Ward clustering hits 29 GB at 60,000 points, so sub-clusters are capped at 40,000. |
| TaxonomyBuilder, Meisenbacher and Norlander [arXiv 2605.21029](https://arxiv.org/abs/2605.21029), [DOI](https://doi.org/10.18653/v1/2026.customnlp4u-1.11) | 2026-05-20 | CustomNLP4U 2026 (Crossref year only) | Part | OL | 32.12M and 1.54M postings. Filtering improves coverage; exact scores not transcribed. |
| How Well Do LLMs Generate Taxonomies in SE?, Nakashima et al. [arXiv 2608.01592](https://arxiv.org/abs/2608.01592), [DOI](https://doi.org/10.1145/3832783.3837551) | 2026-08-03 | ASE 2026, online 2026-10-07 | Part | OL | Independent test: CLIMB is 15 to 40 times faster and 8 to 49 times cheaper; TnT-LLM has higher quality. |
| ATLAS, Lu et al. [arXiv 2606.21597](https://arxiv.org/abs/2606.21597), [DOI](https://doi.org/10.1145/3832783.3834361) | 2026-06-19 | ASE 2026, online 2026-10-07 | Abstract | OL | Failure-driven refinement. F-score 83.13% on 2,001 repositories, about 73.0% on the full corpus. |
| TEMPLAR, Hu et al. [arXiv 2610.05247](https://arxiv.org/abs/2610.05247) | 2026-10-04 | None found | Part | OL | Report-template induction, medical. Coverage 0.8912 against 0.8374 (n=49). Output truncation: 6 of 22 batches succeeded without retry on one model. |
| PEEL, Kim and Joe [arXiv 2610.41785](https://arxiv.org/abs/2610.41785), [DOI](https://doi.org/10.18653/2026.emnlp-main.1305) | 2026-10-30 | EMNLP 2026 Main (Crossref year only) | Part (OL) / Abstract (DJ) | OL, DJ | Taxonomy from a term list, no corpus. Graph F1 .885 and .931. Fits the second stage, after types are found. |
| SPARROW, Zhang et al. [arXiv 2609.07307](https://arxiv.org/abs/2609.07307), [DOI](https://doi.org/10.18653/2026.emnlp-main.1502) | 2026-09-07 | EMNLP 2026 Main (Crossref year only) | Part (OL) / Abstract (DJ) | OL, DJ | Ancestor F1 0.623 and 0.745 at 1K nodes; 0.487 at 10K MeSH concepts against 0.011 and 0.015. |
| Agentic Hybrid Top-Down and Bottom-Up KG, Jouffroy et al. [arXiv 2608.07023](https://arxiv.org/abs/2608.07023) | 2026-08-07 | None found | Part | OL | Wikidata-grounded multilingual skills graph. 36,037 strings to 13,298 nodes. Final consolidation not benchmarked. |
| APOLO, Mai et al. [arXiv 2609.34540](https://arxiv.org/abs/2609.34540) | 2026-09-28 | ISWC 2026 Posters and Demos, per arXiv comment | Part (OL) / Abstract (DJ) | OL, DJ | Prompt optimisation instead of fine-tuning. DOID axiom P .838 and R .689, but PO only .174 and .159. Single run. |
| Executable Schema Contracts [arXiv 2606.05415](https://arxiv.org/abs/2606.05415) | 2026-06-03 | Not recorded | Part (HTML) | OL | Two LLM calls per source, evaluated through QA. A snippet's claim about extra entity types was not found in the text, so it is not repeated here. |
| Specific Domain Ontology Construction, Soares and Wassermann [arXiv 2606.20691](https://arxiv.org/abs/2606.20691) | 2026-06-14, older work | Presented at NeLaMKRR@KR 2025 | Abstract | OL, DJ | Uses GPT-3.5 and GPT-4. Posted recently but not new. |
| Financial ontology induction, Bingqian Li [DOI](https://doi.org/10.1038/s41598-026-70314-x) | None recorded | *Scientific Reports*, 2026-09-09 | Abstract | OL, DJ | Cross-lingual induction claim, no figures in the abstract. Treat as a lead only. |
| AgentsKG, Liu et al. [DOI](https://doi.org/10.1145/3770855.3818058) | None found | KDD 2026, online 2026-08-08 | Abstract (OL) / Metadata (SF) | OL, SF | Committee voting and a logic auditor, open-domain. The abstract has no numbers. |
| DSE-RE, Wang et al. [DOI](https://doi.org/10.1007/s44230-026-00158-1) | None found | *Human-Centric Intelligent Systems*, 2026-06-09 | Part† (OG) / Abstract (OL) | OG, OL | Explicit schema-drift control. Not tested on heterogeneous multilingual corpora. Effect sizes differ between notes; see the disagreement table. |
| Text2Onto-Agent, He et al. [DOI](https://doi.org/10.1061/jcemd4.coeng-18479) | None found | *J. Construction Eng. and Management*, 2026-09-01 | Abstract | OL, DJ | Graph memory replaces long prompts. Disambiguation F1 0.77, parent-child accuracy 85.22%. |
| AOCG-LLM+, Lu et al. [DOI](https://doi.org/10.1007/s40747-026-02372-7) | None found | *Complex & Intelligent Systems*, online 2026-07-16 | Abstract | OL, DJ | 92.5% entity precision, 88.3% relation F1; evaluation partly LLM-calibrated. |
| Extended NeOn-GPT, Fathallah et al. [DOI](https://doi.org/10.1177/22104968261453138) | None found | *Semantic Web*, online 2026-06-26 | Abstract | OL, DJ | Competency questions, repair and reuse. Mostly 0.5 to 0.8 similarity across four domains. |
| OF-Bio-KG, Ben Younes et al. [DOI 00263](https://doi.org/10.1109/COMPSAC69091.2026.00263), [DOI 00287](https://doi.org/10.1109/COMPSAC69091.2026.00287) | None found | IEEE COMPSAC 2026, Jul 2026 | Abstract | OL | Two papers. 1,068 long regulatory leaflets, cluster-first design. No evaluation yet. |
| AgriEvoKG, Zeng et al. [DOI](https://doi.org/10.62762/tetai.2026.882738) | None found | ICCK Transactions, 2026-11-02 | Abstract | OL | Most direct evidence of drift at scale; 199,418 entities. Venue standing not assessed. |
| EventForge, Lodhiya [DOI](https://doi.org/10.30574/wjarr.2026.31.2.2105) | None found | WJARR, 2026-08-31 | Abstract | OL | 12 PDFs, no gold evaluation. Journal standards not assessed. |
| Overview of automated ontology creation, Filippov et al. [DOI](https://doi.org/10.17759/mda.2026160205) | None found | *Modelling and Data Analysis*, 2026-06-30 | Abstract | OL, DJ | Hybrid proposal over 3,000 aviation documents; preliminary. |
| LightOL, Liu et al. [DOI](https://doi.org/10.1109/icassp55912.2026.11464515) | None found | ICASSP 2026, 2026-05-03 | Abstract (DJ) / Metadata (OL) | DJ, OL | Builds ontologies from scratch; includes a self-built Natural Disaster Event dataset. No numbers in the abstract. |
| TaxoEvolve, Rubaiat et al. [DOI](https://doi.org/10.1145/3799682.3841091) | None recorded | CIKM 2026, issued 2026-11-06 | Abstract | DJ | Incremental taxonomy over growing memory. |
| Powder metallurgy taxonomy, Li et al. [DOI](https://doi.org/10.1016/j.matdes.2026.116630) | None recorded | *Materials & Design*, Crossref created 2026-07-21 | Abstract (DJ) / title (OL) | DJ, OL | 26,254 papers to 15 categories by stochastic sampling; expert agreement 90%. |
| Foresight ontology construction, Lupenko et al. [DOI](https://doi.org/10.20535/srit.2308-8893.2026.2.09) | None recorded | *SRIT*, 2026-06-30 | Abstract (DJ) / title (OL) | DJ, OL | Extract, embed, cluster, merge on social media input. |
| Intangible heritage ontology, Madanska [DOI](https://doi.org/10.1109/is71416.2026.11727428) | None recorded | IEEE IS, Sep 2026 | Abstract (DJ) / title (OL) | DJ, OL | Bulgarian bilingual ontology with a reusable taxonomy of LLM errors. |
| Patent field ontology, Qiu and Zhou [DOI](https://doi.org/10.1145/3803791) | None recorded | *TKDD*, online 2026-05-20 (Crossref created 2026-03-27) | Abstract | DJ | Static layer from a classification plus a dynamic layer from NER and clustering. |
| Multi-Agent Ontology Generation, Talukder et al. [arXiv 2604.23090](https://arxiv.org/abs/2604.23090) | **PRE** 2026-04-25 | Not recorded | Part | OL | Quality 3.33 and 3.00 against direct generation's 1.33 and 2.33, judged by five LLMs. |
| DIAL-KG, Bao et al. [arXiv 2603.20059](https://arxiv.org/abs/2603.20059) | **PRE** 2026-03-20 | DASFAA 2026, per comment | Abstract | SF | Closed loop of extraction, governance and schema evolution. |
| ZOES, Xu et al. [ACL Anthology](https://aclanthology.org/2026.eacl-long.354/) | **PRE** 2025-06-04 | EACL 2026 | Abstract | SF | Entity attributes with no schema or examples. |
| D2A-HKG, Chen et al. [arXiv 2510.11297](https://arxiv.org/abs/2510.11297) | **PRE** 2025-10-13 | Not recorded | Abstract | SF | Seven frontier LLMs, zero-shot and schema-free, on paediatric articles. |

### Track A: schema-free extraction and graph building

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| SocraticKG, Choi et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.1951/), [arXiv 2601.10003](https://arxiv.org/abs/2601.10003) | **PRE** 2026-01-15 | Findings of ACL 2026, Jul 2026 | Part | SF | 5W1H questions before triples. Fact retention 73.4-96.3% against GraphRAG 48.5-59.7%, at higher token cost. |
| LLMs as KG Refiners (GraphRefine), Kim et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.1353/) | Not established | ACL 2026, Jul 2026 | Part | SF | Keep, delete, edit or rewrite triples. DocRED 52.94 against GraphJudge 47.57. Needs fine-tuning data. |
| AutoGraph-R1, Tsang et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.1070/), [arXiv 2510.15339](https://arxiv.org/abs/2510.15339) | **PRE** 2025-10-17 | ACL 2026, Jul 2026 | Part | SF | RL with RAG usefulness as reward. Needs a QA training set. |
| MemGraphRAG, Wu et al. [arXiv 2606.00610](https://arxiv.org/abs/2606.00610), [DOI](https://doi.org/10.1145/3770855.3818074) | 2026-05-30 | KDD 2026, 2026-08-08 | Part | SF | Shared corpus memory for cross-chunk consistency. Untested at your scale. |
| RAGA, Han and Cheng [arXiv 2605.17072](https://arxiv.org/abs/2605.17072) | 2026-05-16 | None found | Abstract | SF | Agent checks the graph before writing. Preliminary, on a QASPER subset. |
| RAGU, Komarov et al. [arXiv 2607.11683](https://arxiv.org/abs/2607.11683) | 2026-07-13 | None found | Part | SF | Closest to the budget question: a 7B extractor, about 8k tokens per document, about $0.001 per document on rented GPU (authors' estimate). Check how open its typed extraction is. |
| From Extraction to Governed Memory (MAGG), Bykampadi et al. [arXiv 2608.28642](https://arxiv.org/abs/2608.28642) | 2026-08-12 | None found | Part | SF | Domain agents review triples. SciERC strict F1 0.156 against 0.106 flat; low even after review. |
| DiffIE, Fedorov and Malykh [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1851/), [arXiv 2609.02315](https://arxiv.org/abs/2609.02315) | 2026-09-02 | EMNLP 2026, Oct 2026 | Abstract | SF | Non-LLM OpenIE. Sentence-level, English benchmarks. |
| ReaORE, Lin et al. [arXiv 2606.26986](https://arxiv.org/abs/2606.26986) | 2026-06-25 | None found | Abstract | SF | Open relation extraction that assumes entity pairs are given. |
| F-GraphRAG, Chen and Zhang [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.2253/) | Not established | Findings of EMNLP 2026, Oct 2026 | Abstract | SF | Keeps quantities and dates. +12.6% accuracy with 33.8% fewer relationships. |
| CycleIE, Zhang et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.418/) | Not established | Findings of EMNLP 2026, Oct 2026 | Abstract | SF | Iterative verification and repair. Search loop looks costly at scale (researcher's inference). |
| AutoPKG, Hongwimol et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.766/), [arXiv 2604.16950](https://arxiv.org/abs/2604.16950) | **PRE** 2026-04-18 | Findings of ACL 2026, Jul 2026 | Abstract | SF | Central agent keeps one canonical graph. 0.953 WKE for types, 0.724 for keys. |
| Format-Constraint Coupling, Qi et al. [arXiv 2605.21974](https://arxiv.org/abs/2605.21974) | 2026-05-21 | Submitted to EMNLP 2026 via ARR, per comment | Part | SF | A mismatched schema dropped fact coverage below unconstrained on 4 of 6 table datasets. Retrieval modes hid construction gaps. |
| Hierarchical Hardware KGs, Qiao and Dick [arXiv 2610.12762](https://arxiv.org/abs/2610.12762) | 2026-10-08 | None found | Abstract | SF | Chunked extraction on table-heavy datasheets. 98.8% faithfulness against 76.1% one-shot, 20 datasheets. |
| Every Edge Needs a Quote, Duran et al. [arXiv 2610.32781](https://arxiv.org/abs/2610.32781) | 2026-10-23 | None found | Abstract | SF | Verbatim-quote check plus a five-call vote. 0.94 precision, 0.71 recall against a model-drafted key. |
| FineREX, Feldman et al. [arXiv 2606.19710](https://arxiv.org/abs/2606.19710) | 2026-06-18 | None found | Abstract | SF | 512 annotated chunks pay off. Node duplication 17.78% to 11.17% on long documents. Domain-defined schema. |
| DeepRefine, Huang et al. [arXiv 2605.10488](https://arxiv.org/abs/2605.10488) | 2026-05-11 | None found | Abstract | SF | Repair driven by query interactions. No numbers in the abstract. |
| HERMES [arXiv 2608.14055](https://arxiv.org/abs/2608.14055) | Not recorded | Not recorded | Abstract | SF | 55 volumes of ultra-long documents with tables, F1 about 0.90, with domain constraints. Seen, not assessed. |
| ATOM, Lairgi et al. [ACL Anthology](https://aclanthology.org/2026.findings-eacl.49/) | **PRE** 2025-10-26 | Findings of EACL 2026, Mar 2026 | Abstract | SF | Atomic facts merged in parallel; over 90% latency reduction. |
| LINK-KG, Meher et al. [arXiv 2510.26486](https://arxiv.org/abs/2510.26486) | **PRE** 2025-10-30 | ICKG 2025 | Abstract | SF | Prompt cache tracks references across chunks; 45.21% less node duplication. |

### Track A: coreference, entity resolution and merging

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| Better Retrieval, Limited Clustering Gains, Qi et al. [arXiv 2610.05573](https://arxiv.org/abs/2610.05573) | 2026-10-04 | ICEKIM 2026, per comment; date not checked | Part | SF | Multilingual organisation names. Pair recall 53.74% to 76.98%, but cluster recall only 32.54% to 33.33%. Silver labels. |
| Keep Your Receipts, Choudhary [arXiv 2610.17719](https://arxiv.org/abs/2610.17719) | 2026-10-13 | NeurIPS 2026 workshop, per comment | Part | SF | Two builds of one 300-document corpus differ in 29,475 artifacts. Warning for reproducibility claims. |
| Entity Resolution in Practice, Pavani et al. [arXiv 2607.26298](https://arxiv.org/abs/2607.26298) | 2026-07-28 | None found | Abstract | SF | One false link can chain hundreds of records. Benchmarks up to 5M records. |
| ImCoref-CeS, Luo et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.1122/) | **PRE** 2025-10-11 | ACL 2026, Jul 2026 | Abstract | SF | Cheap coreference model first, LLM as checker. |
| MEIC-DT, Luo et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.881/) | **PRE** 2025-12-31 | Findings of ACL 2026, Jul 2026 | Abstract | SF | Non-LLM long-text coreference under a memory budget. |
| Adaptive Graph Refinement (Alper), Wang et al. [arXiv 2605.25814](https://arxiv.org/abs/2605.25814) | 2026-05-25 | None found | Abstract | SF | Fixed LLM query budget. Record data, not extracted names. |
| Batched Oracle Queries, Balzotti et al. [arXiv 2606.24407](https://arxiv.org/abs/2606.24407) | 2026-06-23 | None found | Abstract | SF | Maximises recall per call when groups exceed one prompt. |
| Same Entity, Different Script, Gopal et al. [arXiv 2610.31407](https://arxiv.org/abs/2610.31407) | 2026-10-22 | None found | Abstract | SF | Same-person accuracy falls from 35.1% in Latin to 8.5% in Devanagari on Qwen2.5-7B. |
| Entity Matching using Decision Models, Steiner et al. [arXiv 2610.34626](https://arxiv.org/abs/2610.34626) | 2026-10-25 | None found | Abstract | SF | Cheap calibrated decisions. Sending unsure pairs to an ensemble adds 1.9-4.0 F1 points. |
| Labeling Training Data for Entity Matching, Steiner and Bizer [arXiv 2606.28823](https://arxiv.org/abs/2606.28823) | 2026-06-27 | None found | Abstract | SF | Students run 34-459 times faster at $5.86-$8.11 per training set. Weaker on unseen products. |
| CaRL-EM, Guo et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.1258/), [arXiv 2609.01195](https://arxiv.org/abs/2609.01195) | 2026-09-01, after the conference | ACL 2026, Jul 2026 | Abstract | SF | RL chooses operator and model per record. Numbers not read. |
| LLM-CER, Wang et al. [DOI](https://doi.org/10.14778/3827998.3828112) | Not established | PVLDB, 2026-08-01 | Abstract (first part only) | SF | In-context clustering rather than pair matching. Evidence not read. |
| Generalized Entity Matching, Chen et al. [DOI](https://doi.org/10.1145/3802066) | Not established | *PACMMOD*, 2026-05-18 | Abstract (first part only) | SF | Matches text mentions to records. Evidence not read. |
| Reliability of LLM Judges, Ravideshik and Kejriwal [ACL Anthology](https://aclanthology.org/2026.emnlp-main.1717/), [arXiv 2610.09554](https://arxiv.org/abs/2610.09554) | 2026-10-07 | EMNLP 2026, Oct 2026 | Abstract | SF | Showing the judge the system's label collapses J-ROC-AUC to 0.12-0.87; label-free recovers 0.93-1.00. |
| LLM Self-Explanations for ER, Teofili et al. [arXiv 2606.01210](https://arxiv.org/abs/2606.01210) | 2026-05-31 | None found | Abstract | SF | Explanations are often unstable, so they are poor evidence for a merge. |
| AgenticER, Papadakis et al. [arXiv 2607.27435](https://arxiv.org/abs/2607.27435) | 2026-07-29 | Submitted to VLDB, per comment | Abstract | SF | Vision paper, no experiments. |

### Track A: quality, small models and scoring rules

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| Ranked by the Matcher, Hakim and Song [arXiv 2609.01671](https://arxiv.org/abs/2609.01671) | 2026-09-01 | None found | Part | SF | One prediction set spans 0.16-0.70 F1 by matching rule. Read before trusting any F1 comparison here. |
| From Unreliable to Usable, Qian [arXiv 2610.20417](https://arxiv.org/abs/2610.20417) | 2026-10-14 | None found | Part | SF | Preregistered audit. Micro precision 0.03-0.34; 65.6% of false positives are background mentions, 1.2% unsupported. |
| KGCQual, Misra et al. [arXiv 2607.10212](https://arxiv.org/abs/2607.10212) | 2026-07-11 | Under review, per comment | Abstract | SF | Gold-free graph quality score. |
| When Does Consistency Pay Off?, Sun [arXiv 2607.24312](https://arxiv.org/abs/2607.24312) | 2026-07-27 | None found | Abstract | SF | Rescoring identical predictions on Re-DocRED reverses the ranking. |
| Sub-Billion, Super-Frontier, Christou and Tsoumakas [arXiv 2606.22606](https://arxiv.org/abs/2606.22606) | 2026-06-21 | None found | Abstract | SF | Fine-tuned Qwen2.5-0.5B micro-F1 0.83 against GPT-5.4 0.69 zero-shot, on fixed relation sets. |
| LLMs Underperform Graph-Based Parsers, Gajo et al. [ACL Anthology](https://aclanthology.org/2026.acl-short.17/) | **PRE** 2026-04-09 | ACL 2026 short, Jul 2026 | Abstract | SF | Parser pulls ahead as relations per document increase. |
| Generative NER assessment, Zhan et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1706/) | **PRE** 2026-01-25 | Findings of EMNLP 2026, Oct 2026 | Abstract | SF | Light fine-tuning is competitive with encoder models. |
| Energy-Aware Benchmarking, Politov et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.2002/) | Not established | Findings of EMNLP 2026, Oct 2026 | Abstract | SF | Energy per true positive for planning on owned GPUs. |
| Right IE Pipeline Depends on the Document, Walser et al. [arXiv 2609.31341](https://arxiv.org/abs/2609.31341) | 2026-09-25 | DocInsights at EMNLP 2026, per comment | Abstract | SF | Batching cuts energy per page 38-85%. Neural OCR uses 17 times classical OCR's energy. |
| Trust the Edge, Doubt the Arrow, Lan et al. [arXiv 2610.24390](https://arxiv.org/abs/2610.24390) | 2026-10-17 | PRICAI 2026, per comment | Abstract | SF | Corpus-scale audit over 328,839 summaries. Literature corroboration covers only 30.4% of relations. |
| Lost in Formatting, Ravikumar et al. [ACL Anthology](https://aclanthology.org/2026.eacl-long.256/) | None found | EACL 2026 (pre-window) | Abstract | SF | Output format alone can shift F1 by over 40%. |
| Don't Worry About Quality (CS-RAG), Ma et al. [arXiv 2603.14828](https://arxiv.org/abs/2603.14828) | **PRE** 2026-03-16 | Not recorded | Abstract | SF | Similar extraction errors across graph builders. A search result listed this ID under a different title. |
| FinReflectKG, Kumar et al. [arXiv 2603.20252](https://arxiv.org/abs/2603.20252) | **PRE** 2026-03-11 | Not recorded | Abstract | SF | MCC falls 44-84% with noisy triples for most methods. |

### Track A: whether a graph beats plain retrieval

These matter because an expensive ontology build must earn its keep against a retrieval baseline. Graphs help mostly on multi-fact aggregation, and plain retrieval stays competitive elsewhere (SF).

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| WildGraphBench, Wang et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.679/) | **PRE** 2026-02-02 | Findings of ACL 2026, Jul 2026 | Part | SF | Closest benchmark to long mixed documents. HippoRAG2 64.33, NaiveRAG 59.79, MS GraphRAG global 54.54; the graph leads only on multi-fact questions. |
| Dissecting GraphRAG, Nishida et al. [ACL Anthology](https://aclanthology.org/2026.tacl-1.29/) | Not established | TACL 2026; Crossref record created 6 May 2026 | Part | SF | Extraction precision limits results. A tuned small model beats few-shot GPT-4o-mini. Template reports beat LLM summaries. |
| Structure Over Scale (HCG-RAG), Saouda et al. [arXiv 2607.22592](https://arxiv.org/abs/2607.22592) | 2026-06-10 | None found | Part | SF | Best cost table: 17,638 build calls for MS-GraphRAG against 1,167 for HCG-RAG M, with similar answer correctness. Induced vocabulary is a middle option. |
| Structure Pricing (EffiRAG), Zhang et al. [arXiv 2609.18099](https://arxiv.org/abs/2609.18099) | 2026-09-16 | None found | Abstract | SF | Non-LLM filter skips low-salience chunks; $0.408 against $0.952. Small corpora, LLM judge. |
| EGT-KG, Yu et al. [arXiv 2609.00479](https://arxiv.org/abs/2609.00479) | 2026-08-31 | EMNLP 2026 Industry, per comment | Abstract | SF | Direct automatic versus expert schema: 70.37 and 68.82 on one literature set. |
| Is GraphRAG Needed?, Chen et al. [arXiv 2606.25656](https://arxiv.org/abs/2606.25656) | 2026-06-24 | ACL 2026 GEM Workshop, per comment | Abstract | SF | Context engineering cuts tokens 19-53%. |
| ContextRAG, Prosvirnin et al. [arXiv 2605.19735](https://arxiv.org/abs/2605.19735) | 2026-05-19 | None found | Abstract | SF | Extraction-free: 30 LLM calls to index. No named relations. |
| Geometric Gain Graph, Li et al. [arXiv 2609.19622](https://arxiv.org/abs/2609.19622) | 2026-09-17 | None found | Abstract | SF | Zero-token chunk graph. Numbers not read. |
| EHRAG, Song et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.1233/) | **PRE** 2026-04-19 | Findings of ACL 2026, Jul 2026 | Part | SF | Low-cost baseline to beat. |
| Collision to Cognition (MeshRAG), Zhou et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.1156/) | Not established | ACL 2026, Jul 2026 | Abstract | SF | Hash-built structure, no GPU for construction. |
| Beyond Chunks and Graphs, Gong et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.1310/) | **PRE** 2025-08-04 | Findings of ACL 2026, Jul 2026 | Abstract | SF | Triplet store with no graph. |
| MiniRAG, Fan et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.1721/) | **PRE** 2025-01-12 | ACL 2026, Jul 2026 | Abstract | SF | Small models, 25% of the storage. |
| GRASP, Jenkins et al. [arXiv 2605.16598](https://arxiv.org/abs/2605.16598) | 2026-05-15 | None found | Abstract | SF | Mostly a query-time result. |
| Reducing Hallucinations with Simple GraphRAG, Wedge et al. [arXiv 2606.05901](https://arxiv.org/abs/2606.05901) | 2026-06-04 | None found | Abstract | SF | Graph tools can halve hallucinated answers on MoNaCo. |
| Operationalizing CTI with GraphRAG, Kabra et al. [arXiv 2608.13050](https://arxiv.org/abs/2608.13050) | 2026-08-13 | None found | Abstract | SF | Very small sample. |
| Scan or Index?, Terven et al. [arXiv 2610.28957](https://arxiv.org/abs/2610.28957) | 2026-10-20 | None found | Abstract | SF | Hybrid with reranking has 94.1% coverage; more evidence did not always help. |
| Reading Between the Chapters, Sumit et al. [arXiv 2611.00752](https://arxiv.org/abs/2611.00752) | 2026-10-31 | None found | Abstract | SF | Graph-free, about $0.41 to index a book. |
| PILAR, Shin et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.128/), [arXiv 2609.32895](https://arxiv.org/abs/2609.32895) | 2026-09-26 | Findings of EMNLP 2026, Oct 2026 | Abstract | SF | Assertion graph across text and tables; +1.6 EM overall, keeping pages as ground truth. |
| Graph-Based RAG tutorial, Bonifati et al. [DOI](https://doi.org/10.14778/3827998.3828147) | Not established | PVLDB, 2026-08-01 | Abstract (first part only) | SF | Route into VLDB work. |
| SPRIG, Wang [arXiv 2602.23372](https://arxiv.org/abs/2602.23372) | **PRE** 2025-12-27 | Not recorded | Abstract | SF | CPU-only NER co-occurrence graph. |
| LiteSemRAG, Yue et al. [arXiv 2604.16350](https://arxiv.org/abs/2604.16350) | **PRE** 2026-03-16 | Not recorded | Abstract | SF | LLM-free graph retrieval. |

### Track A: benchmarks, shared tasks and surveys

LLMs4OL 2026 team papers all appeared in Open Conference Proceedings vol. 11 with no first-posted date unless shown. OL read abstracts plus introductions or extracts; DJ read abstracts or metadata.

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| [Genuineknowledge-KEOL](https://doi.org/10.52825/ocp.v11i.3661), Liu et al. | None found | 2026-11-02 | Abstract, extracts | OL, DJ | Rank 1, GS 0.7762. Training-free staged pipeline. |
| [BigDipperAI](https://doi.org/10.52825/ocp.v11i.3657), Zhao et al. | None found | 2026-11-03 | Part (introduction) | OL | Second, adapted Qwen3-8B, ahead of GPT-5.4 and GPT-5.5 systems. |
| [LaboroAI](https://doi.org/10.52825/ocp.v11i.3656), Zhao and Xu | None found | 2026-11-03 | Abstract (DJ); PDF retrieved, not read (OL) | DJ, OL | One call per document with retrieved examples beat decomposition and ensembles. |
| [Semantic-Swingers](https://doi.org/10.52825/ocp.v11i.3681), Vizcaíno et al. | None found | 2026-11-02 | Abstract, extracts | OL, MV, DJ | Negative-results catalogue: every after-the-fact repair tested failed. |
| [XuanJi](https://doi.org/10.52825/ocp.v11i.3682), Jin | None found | 2026-11-02 | Abstract, extracts | OL | One GPT-5.4 call: GS 0.7790 semantic, 0.5485 exact. |
| [pro-team](https://doi.org/10.52825/ocp.v11i.3663), Mishra et al., [arXiv 2608.27101](https://arxiv.org/abs/2608.27101) | 2026-08-27 | 2026-11-04 | Abstract, extracts | OL, DJ | GS 0.6553. |
| [LIST-CortAIx](https://doi.org/10.52825/ocp.v11i.3658), Souihi and Kara | None found | 2026-11-03 | Abstract, extracts | OL, DJ | Where errors enter the graph; OL says read it in full. |
| [Volt-IQ](https://doi.org/10.52825/ocp.v11i.3662), Ku and Nie | None found | 2026-11-04 | Abstract, extracts | OL | The challenge site lists one author and a shorter title. |
| [ELLMO](https://doi.org/10.52825/ocp.v11i.3664), Gray et al. | None found | 2026-11-02 | Abstract, extracts (OL) / Metadata (DJ) | OL, DJ | GS 0.4958. |
| [LettRAGraph](https://doi.org/10.52825/ocp.v11i.3655), Lisena et al. | None found | 2026-11-03 | Abstract (DJ); PDF retrieved, not read (OL) | DJ, OL | GS 0.4539. |
| [VUL337-OL](https://doi.org/10.52825/ocp.v11i.3665), Deng et al. | None found | 2026-11-02 | Abstract, extracts (OL) / Metadata (DJ) | OL, DJ | GS 0.4167. |
| [HybridAI-Squared](https://doi.org/10.52825/ocp.v11i.3660), Bhattacharya et al. | None found | 2026-11-02 | Abstract, extracts (OL) / Metadata (DJ) | OL, DJ | Reuse task only. |
| [SEMA](https://doi.org/10.52825/ocp.v11i.3659), Esteve et al. | None found | 2026-11-04 | Abstract; PDF retrieved, not read | OL | Taxonomy task only. |
| LLMs4OL preface [DOI](https://doi.org/10.52825/ocp.v11i.3738) | None found | Vol. 11 | Not recorded in the notes | OL | 13 scored teams: 11 Flagship, 8 Reuse, 5 Taxonomy. |
| CQ4OE, Li et al. [arXiv 2609.26029](https://arxiv.org/abs/2609.26029) | 2026-09-22 | None found | Part (OL) / Abstract (DJ) | OL, DJ | Competency-question benchmark. Term F1 59.1-66.5%, structural F1 26.7-33.7%. Input is not corpus text. |
| OntoLearner, Babaei Giglou et al. [arXiv 2607.01977](https://arxiv.org/abs/2607.01977), [DOI](https://doi.org/10.21203/rs.3.rs-9539494/v1) | 2026-06-25 (Research Square), 2026-07-02 (arXiv) | Under review at *Nature Communications*, per comment | Part (OL) / Abstract (DJ) | OL, DJ | 180 ontologies. Taxonomy recall 9 to 37% from top retrievers. |
| When Does Bigger Help?, Babaei Giglou et al. [arXiv 2608.31118](https://arxiv.org/abs/2608.31118) | 2026-08-31 | WOP 2026 at ISWC, per comment | Part (OL) / Abstract (DJ) | OL, DJ | Model size is a poor predictor: term typing 58.14% on Qwen3.5-27B against 39.44% on a larger model. |
| On Measuring Semantic Preservation, Sadowski and Chudziak [arXiv 2608.12326](https://arxiv.org/abs/2608.12326) | 2026-05-30 | KES 2026, per comment | Part | OL | Answering from the learned ontology loses 8.4 to 27.4 points against source text. |
| MeSH-Rel-4K, Aggarwal et al. [arXiv 2607.17902](https://arxiv.org/abs/2607.17902) | 2026-07-20 | None recorded | Abstract | OL | Fine-tuning raises small-model average F1 by 34.1 points. |
| MIOE, Zhang et al. [arXiv 2610.38326](https://arxiv.org/abs/2610.38326) | 2026-10-28 | None recorded | Abstract | OL | Causal-abstraction benchmark. |
| Systematic Literature Review, Li, Garijo and Poveda-Villalón [DOI](https://doi.org/10.1177/22104968261465514) | Likely 2025 preprint, first page only | *Semantic Web*, online 2026-07-13 | Abstract | OL, DJ | 36 papers and 49 task-level studies; little standardisation. Best route to further primary work. |
| Knowledge Graphs for Theses survey, Ruelas et al. [arXiv 2610.28648](https://arxiv.org/abs/2610.28648) | 2026-10-05 | None recorded | Abstract | OL, DJ | 46 studies. Long structured documents as an analogue. |
| Health Knowledge Modelling scoping review, Pournik et al. [DOI](https://doi.org/10.3233/SHTI260406) | None recorded | 2026-05-21 | Abstract | OL | 20 of 166 records. |
| Stage-Wise Human-in-the-Loop Study, Ding et al. [DOI](https://doi.org/10.3233/SHTI260304) | None recorded | 2026-05-21 | Abstract | OL | Human input consistently improves quality. |
| Diagnosing the limits of LLMs for evidence extraction, Tan et al. [DOI](https://doi.org/10.1007/s00799-026-00449-9) | None recorded | *IJDL*, online 2026-10-28 | Abstract | DJ | Better on single entities than on bindings among entities and numbers. |
| Knowledge Engineering capability assessment, Tsaneva et al. [DOI](https://doi.org/10.1177/22104968261499449) | None recorded | *Semantic Web*, online 2026-10-23 | Abstract | DJ | Ontology property restrictions. |
| Meta-survey of ontology development methodologies, Sinha and Dutta [DOI](https://doi.org/10.1177/01655515261462106) | None recorded | *JIS*, online 2026-07-07 | Abstract | DJ | 45 studies, mostly not LLM era. |
| Joint entity-relation extraction survey, Lin and Chang [DOI](https://doi.org/10.3724/jtke-20250008) | None recorded | *JTKE*, 2026-06-01 | Abstract | DJ | Bronze open access per OpenAlex. |
| RAG for Curated Thematic Corpora, Monika et al. [DOI](https://doi.org/10.3390/info17070660) | None recorded | *Information*, 2026-07-07 | Abstract | DJ | Argues for exploiting expert-authored structures. |
| RAG and knowledge graphs bibliometric review, Chansanam et al. [DOI](https://doi.org/10.47909/ijsmc.382) | None recorded | 2026-07-28 | Abstract | DJ | Record counts look inconsistent; verify before citing. |
| Graph-Mediated LLM Systems, Liu et al. [arXiv 2611.05953](https://arxiv.org/abs/2611.05953) | 2026-11-04 | None recorded | Abstract | DJ | Benchmarks organised by role of graph operations. |

### Track A: discovery on UN, policy and humanitarian text

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| Governing the evidence base, Turobov and Coyle [DOI](https://doi.org/10.1017/dap.2026.10100) | None found | *Data & Policy*, online 2026-09-24 | Full† | DJ | Closest discovery work on UN material: 63 UN AI policy documents, human-in-the-loop 713 codes and 12 themes against API 473 codes. Tables and graphs excluded. |
| Verifiable Disaster Storylines, Decostanzi et al. [arXiv 2609.00858](https://arxiv.org/abs/2609.00858), [DOI](https://doi.org/10.1145/3794786.3830759) | 2026-09-01 (DJ); v1 not checked (OG) | GoodIT '26, online 2026-10-06 | Full† (DJ) / Part (OG) | DJ, OG | ReliefWeb sources, free-form causal graphs. Experts rate causal graphs 2.78 of 5 and citations 4.33 of 5†. |
| Looking Complete Is Not Being Grounded, Kumar [arXiv 2610.18762](https://arxiv.org/abs/2610.18762) | 2026-10-13 | LT4CPR at AACL-IJCNLP 2026, per comment | Abstract | DJ | 15-slot schema derived from human reports. Machine reports reproduce only 5.4 to 11.0% of same-day official figures. |
| Humanitarian situation reporting, Decostanzi et al. [DOI](https://doi.org/10.1038/s41598-026-68010-x) | None recorded (received 2025-12-03) | *Scientific Reports*, 2026-08-28, early version | Abstract | DJ | Semantic clustering across over 1,100 documents; output is reports, not an ontology. |
| Discourse-Aware Policy Analysis (Apaf), Vasileiou and Derendiaeva [arXiv 2607.13260](https://arxiv.org/abs/2607.13260) | 2026-07-14 | None recorded | Abstract | DJ | Argument graphs over 100 sub-documents of disaster-risk policy. |
| Causal Relations in Social Media, Jeong et al. [arXiv 2605.11348](https://arxiv.org/abs/2605.11348) | 2026-05-12 | None recorded | Abstract | DJ | Marginal: social media input. |
| Graphing the European Green Deal, Arkadopoulou et al. [DOI](https://doi.org/10.3390/su18126193) | None recorded | *Sustainability*, 2026-06-16 | Abstract (MDPI 403) | DJ | 17 policy documents, an entity type set for policy text. No numbers in the abstract. |
| MGAL long-context benchmark, Li et al. [arXiv 2608.20853](https://arxiv.org/abs/2608.20853) | 2026-08-21 | None recorded | Part† | DJ | The only multilingual benchmark on UN reports found, six official languages, 8K to 128K tokens. Not extraction. |
| Multimodal CoLRAG-TF, Yasuno [arXiv 2607.20517](https://arxiv.org/abs/2607.20517) | 2026-07-07 | None recorded | Abstract | DJ | 43 Japanese disaster PDFs, 11,414 OpenIE triples. |

### Track B: ontology-guided extraction

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| LMKG, Pappacoda and Torroni [ACL Anthology](https://aclanthology.org/2026.akbc-1.23/) | No preprint checked | AKBC 2026, Oct 2026 | Part | OG | Cleanest on/off comparison, and two long-document strategies. MINE-1 fact recovery 84.87 (fold), 90.80 (enrich). |
| Better Later Than Sooner, Loconte et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1867/), [arXiv 2605.29168](https://arxiv.org/abs/2605.29168) | 2026-05-27 (OL, SF); not checked (OG) | Findings of EMNLP 2026, Oct 2026 | Part | OG, OL, SF | Constrain before or correct after. Plain vector RAG beat every graph on HotpotQA with the stronger model. |
| Strategic Knowledge Extraction in Capability-Based Planning, Kolk et al. [PDF](https://aclanthology.org/2026.kallm-1.20.pdf) | None found | KG-LLM at LREC 2026, May 2026 | Part | OG | Closest document type and task: public-institution policy text, subjective coding. LLM failed the skilled-annotator threshold for all concept types. |
| LLM-Guided Ontology-Driven KG, Belfadel et al. [arXiv 2609.31663](https://arxiv.org/abs/2609.31663) | 2026-09-15 | IJCKG 2026, per arXiv (OL, DJ); no venue found (OG) | Full, 2,900 characters unseen (OG) / Part (OL, DJ) | OG, OL, DJ | French incident reports, 80 annotated. Descriptions raise Gemma3 strict F1 from 0.42 to 0.57. Private corpus. |
| Schema-Constrained Evaluation Framework, Ilves et al. [ACL Anthology](https://aclanthology.org/2026.kallm-1.21/), [DOI](https://doi.org/10.63317/3icoxh7axwxy) | None found | KG-LLM at LREC 2026, May 2026 | Part | OG, DJ | Entity F1 84.0% but triple F1 21.7% on GPT-5.1. Triple F1 is a lower bound given incomplete gold. |
| eolas, Sbodio et al. [arXiv 2609.17291](https://arxiv.org/abs/2609.17291) | 2026-09-15 | None found | Part | OG | Most systematic test of serialising a schema into a prompt. Best zero-shot 36.0%, best few-shot 67.6%. |
| Evaluation Objects from Academic Review Texts, Chen et al. [arXiv 2608.29526](https://arxiv.org/abs/2608.29526) | 2026-08-29 listing | None found | Part | OG | Not programme evaluation, despite the name. Strict typed F1 79.78%, with an evidence-keeping "Other" policy. Chinese. |
| PolicyKG, Kaewsawee et al. [arXiv 2608.09028](https://arxiv.org/abs/2608.09028) | 2026-08-10 | Under review at IJCKG 2026, per comment | Part (OG) / Abstract (DJ) | OG, DJ | Institutional policy to SHACL. Macro F1 .810 falls to .370 on a different domain. |
| ArticleMiner, Jahin et al. [arXiv 2609.25607](https://arxiv.org/abs/2609.25607) | 2026-09-22 | Sci-K at ISWC 2026, per arXiv | Part | OG, OL | Tables in reports. MLTables 52.2 against 37.8 for few-shot. |
| Political-Elite Networks, Solovev and Lasser [arXiv 2606.27347](https://arxiv.org/abs/2606.27347) | 2026-06-25 | None found | Part (OG, SF) / Abstract (DJ) | OG, SF, DJ | Best multilingual constrained decoding. 37 million aliases in 36 languages; 68.2% strict correctness. News, not reports. |
| From Text to Triples, Fadda et al. [PDF](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_3.pdf) | None found | CEUR Vol-4233, 2026-08-01 | Part | OG | Claim extraction resembles findings extraction. F1 76.57 to 41.87 by model. |
| Nuclear Domain Case Study, Kim and Yu [PDF](https://www.kns.org/files/pre_paper/55/26S-290-김보경.pdf) | None found | KNS Spring Meeting, 7 to 8 May 2026; no DOI found | Part | OG | Conformance above 0.93 while subject hallucination reaches 0.292. |
| KG Construction for Cultural Heritage, Peeters et al. [DOI](https://doi.org/10.1007/978-3-032-41614-8_7) | None found | EKAW 2026, online 2026-10-16 | Abstract† (paywalled) | OG | Strongest counter-evidence on validation. Stage-level gating hurt 9 of 10 domains†. Museum texts at or below F1 0.06†. |
| Ontology-Aware Triple Extraction, Calautti et al. [DOI](https://doi.org/10.1007/978-3-032-40377-3_5) | None found | Advances in Conceptual Modeling (ER 2026), online 2026-09-29 | Abstract† (paywalled) | OG | Refinement let a 14B model beat a one-pass frontier model†. No figures. |
| Ontology-Aware Prompting, Tiwari et al. [DOI](https://ebooks.iospress.nl/doi/10.3233/SSW260009) | None found | *Studies on the Semantic Web*, 2026-09-02 | Abstract (truncated, DJ) | OG, DJ | Multi-prompt candidates plus rule filters. Up to 2.3 times LLM baselines on Text2KGBench. |
| DOPKG-SHACL, Uwasomba et al. [repository](https://research.edgehill.ac.uk/en/publications/deterministic-and-trustworthy-llm-driven-semantic-knowledge-graph/), DOI 10.1109/iccsc71566.2026.11650062 | None found | IEEE ICCSC, Jul 2026 | Abstract | OG | Deterministic IRI minting had the strongest effect. No figures. |
| SciKGExtract, Sadruddin and D'Souza [DOI](https://doi.org/10.21203/rs.3.rs-11290953/v1), [arXiv 2610.06322](https://arxiv.org/abs/2610.06322) | 2026-10-28 (Research Square) | None recorded | Abstract | OG, OL | 176 papers. Agentic refinement raises ZnO F1 from 0.591 to 0.805; IGZO only 0.344. |
| RDF Generation in Urban Agriculture, Vianna do Nascimento and Palazzo [DOI](https://doi.org/10.5753/sbbd.2026.249179) | None found | SBBD 2026, 2026-09-08 | Abstract | OG | 42 web pages with SHACL. No figures. |
| Metallurgical Failure Reports, Zhou et al. [DOI](https://doi.org/10.3390/app16126178) | None found | *Applied Sciences*, 2026-06-18 | Abstract (publisher 403) | OG | Stepwise chain to triple output with source localisation. No figures. |
| Ontology-Guided and LLM-Enhanced KG, Kahlawi et al. [DOI](https://doi.org/10.1007/978-3-032-24807-7_1) | None found | CC 2026, online 2026-06-01 | Abstract† (paywalled) | OG | Scanned PDFs with handwriting, with page citations. No numbers. |
| Extract, Align, Reflect, Huang and Hong [DOI](https://doi.org/10.1007/978-981-92-3426-4_13) | None found | ICIC 2026, online 2026-07-21 | Abstract† (paywalled) | OG | Higher F1 with a schema, less redundancy without†. No figures. |
| OntoExtend, Lippolis et al. [arXiv 2607.17963](https://arxiv.org/abs/2607.17963) | 2026-07-20 | None found | Abstract (OG, listed in its abstract tier though it cites HTML) / Metadata (OL) | OG, OL | Extension from competency questions. Fragments needed minor to moderate revision. |
| CIExMAS, Lautenbach and Hertling [PDF](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_5.pdf) | None found | CEUR Vol-4233, 2026-08-01 | Abstract | OG | Up to 80% of synthIE without fine-tuning. |
| PrOntoNarr, Orciuoli et al. [PDF](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_12.pdf) | None found | CEUR Vol-4233, 2026-08-01 | Abstract | OG | Narrative ontology from political speeches. |
| DBpedia Hindi, Tiwari et al. [PDF](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_18.pdf) | None found | CEUR Vol-4233, 2026-08-01 | Abstract | OG | Links extracted predicates to ontology properties. |
| Legal rule ontologies, Sartor et al. [PDF](https://ceur-ws.org/Vol-4233/Text2KG_Paper_ID_10.pdf) | None found | CEUR Vol-4233, 2026-08-01 | Abstract | OG | Competency questions validate generation, not population. |
| BEACON [arXiv 2608.28394](https://arxiv.org/abs/2608.28394) | Not recorded | Not recorded | Abstract | OL | Anchored to MITRE ATT&CK. |
| CodeGraph [arXiv 2609.29474](https://arxiv.org/abs/2609.29474) | Not recorded | Not recorded | Abstract | OL | Wikidata-grounded, 167M files. |
| Citation knowledge graphs, Salman and Ihsan [DOI](https://doi.org/10.1007/s11192-026-05825-5) | 2026-05-20 ([Research Square](https://doi.org/10.21203/rs.3.rs-9583368/v1)) | *Scientometrics*, online 2026-09-28 | Abstract | DJ | Compact mapping: small local LLM with a subset of an ontology. |
| Few-Shot Guided KG Population (WikiTabLLM), Gottschalk et al. [DOI](https://doi.org/10.1145/3799682.3840636) | None recorded | CIKM 2026, Crossref created 2026-11-03 | Abstract | DJ | KG population from clean Wikipedia tables, not noisy ones. |
| Wikontic, Chepurova et al. [ACL Anthology](https://aclanthology.org/2026.eacl-long.388/) | **PRE** 2025-11-29 | EACL 2026, Mar 2026 | Abstract | OG, SF | The baseline several papers use. Uses Wikidata constraints. |
| OntoKG (arXiv 2604.02618) and OntoLogX ([DOI](https://doi.org/10.1002/aisy.202501381), arXiv 2510.01409) | **PRE**‡ (April 2026 for OntoKG) | Not verified | Metadata; snippet only‡ | OG | Schema extension and SHACL feedback. Pages not opened. |

### Track B: entity linking and concept normalisation

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| SelectED, Rücker et al. [ACL Anthology](https://aclanthology.org/2026.uncertainlp-main.9/) | None found | UncertaiNLP at EMNLP 2026, Oct 2026 | Part | MV | LLM on about 20% of mentions gains 4.3% accuracy. Benchmarks have no real NIL cases. |
| DuCEI, Min et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1298/) | None found | Findings of EMNLP 2026, Oct 2026 | Abstract | MV | No LLM. Second stage on 27.80% of mentions cuts cost from 1.90× to about 1.27×. |
| Multi-Negative DPO, Nguyen et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1873/), [arXiv 2609.07379](https://arxiv.org/abs/2609.07379) | 2026-09-10 | Findings of EMNLP 2026, Oct 2026 | Part | MV | Closest match to noisy multilingual text with NIL. French micro-F1 72.4 and 68.7. Mention-level. |
| BioELX, Wang et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1109/), [arXiv 2605.27380](https://arxiv.org/abs/2605.27380) | **PRE** 2026-04-09 | Findings of EMNLP 2026, Oct 2026 | Part | MV | Wikidata cross-lingual aliases. XL-BEL Recall@1 EN 91.2, ES 69.6. Prompt design mattered more than fine-tuning. |
| LongBEL, Remaki et al. [arXiv 2605.13451](https://arxiv.org/abs/2605.13451) | 2026-05-13 | No venue stated | Full† | MV | Document-level consistency. FR 77.6, EN 80.0. Documents are far shorter than UN reports. |
| German SNOMED normalisation, Adam et al. [DOI](https://doi.org/10.3233/shti260995) | None recorded | 2026-09-17 | Abstract | MV | LLM alone Recall@1 0.235, retrieval 0.681, both 0.771. |
| Faroese entity linking, Simonsen et al. [ACL Anthology](https://aclanthology.org/2026.resourceful-4.4/) | None recorded | RESOURCEFUL at LREC 2026, May 2026 | Abstract | MV | Language fallback; 87.5% precision. Web search does not transfer to a closed thesaurus. |
| Sci-ZSEL, Khondokar et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1286/), [arXiv 2609.00228](https://arxiv.org/abs/2609.00228) | 2026-08-31 | Findings of EMNLP 2026, Oct 2026 | Part (MV) / Abstract (DJ) | MV, DJ | Per-entity LLM aliases, no human labels. Filter lifts pseudo-pair accuracy from 72.91% to 90.32%. Agricultural ontologies. |
| HiCo-EL, Lee et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.103/) | None found | Findings of EMNLP 2026, Oct 2026 | Abstract | MV | Hierarchy adds +9.6 Recall@1 on unseen entities. |
| OntologyAligner, Song et al. [arXiv 2609.10055](https://arxiv.org/abs/2609.10055) | 2026-09-09 | None recorded | Abstract | MV | Hierarchy-guided refinement; 86.75% micro Top-1. Not OntoAligner. |
| BeLink, Shlyk et al. [arXiv 2605.22501](https://arxiv.org/abs/2605.22501) | 2026-05-21 | Accepted to SIGIR 2026 | Abstract | MV | Accuracy gains of 3% to 24%. |
| BELXTR, Garda and Leser [arXiv 2609.25859](https://arxiv.org/abs/2609.25859) | 2026-09-22 | None recorded | Abstract | MV | Alternative when LLM reranking is too costly at scale. |
| SNO-PI, Songsiritat [arXiv 2610.26425](https://arxiv.org/abs/2610.26425) | 2026-10-19 | None recorded | Abstract | MV | Unseen concepts are the main error source. |
| Context-Aware SNOMED CT Linking, Kadusabe et al. [ACL Anthology](https://aclanthology.org/2026.clinicalnlp-1.33/) | None found | CLINICALNLP at LREC 2026, May 2026 | Abstract | MV | Agentic reranking helps only marginally. |
| Profiling Hallucinations, Born et al. [ACL Anthology](https://aclanthology.org/2026.clinicalnlp-1.41/) | None found | CLINICALNLP at LREC 2026, May 2026 | Part | MV | RAG removes invented codes but F1 only reaches 18.67 to 22.64. Most authors work for the comparison vendor. |
| GEO metadata normalisation, Szczepaniak and Wren [DOI](https://doi.org/10.21203/rs.3.rs-10844781/v1) | 2026-08-31 | Research Square | Abstract | MV | 804,427 samples, clear handling of out-of-vocabulary concepts. |
| OMOP harmonisation, Meyer-Eschenbach et al. [DOI](https://doi.org/10.3233/shti260883) | None recorded | 2026-06-29 | Abstract | MV | LLM refinement raises medication precision from 60.8% to 93.9%. |
| DIMAS-OMOP, Lv et al. [ACL Anthology](https://aclanthology.org/2026.mellm-1.10/) | None recorded | MELLM at ACL 2026, Jul 2026 | Abstract | MV | 95.2% mapping accuracy, 1,200 reports per hour. |
| HPO extraction with MCP, Larsen et al. [DOI](https://doi.org/10.64898/2026.05.23.26353963) | 2026-05-25 | medRxiv | Abstract | MV | Real-time lookup raises mean F1 from 0.46 to 0.72. Four synthetic notes. |
| Entity Linking in Invoices, Azevedo et al. [DOI](https://doi.org/10.5753/sbbd.2026.249286) | None recorded | SBBD 2026, 2026-09-08 | Abstract | MV | Filters cut LLM calls by 49.60%. |
| SCODEL, Ibrahim et al. [arXiv 2610.27637](https://arxiv.org/abs/2610.27637) | 2026-10-20 | None recorded | Abstract | MV | Hit@1 83.15% on unseen concepts. |
| Biomedical concept normalisation benchmark, Verma et al. [DOI](https://doi.org/10.64898/2026.09.19.752843) | 2026-09-21 | bioRxiv | Abstract | MV | Cleaning text before retrieval raises SNOMED Top-5 from 63.8% to 69.4%. |
| PILOT, Tao and Liu [arXiv 2608.04144](https://arxiv.org/abs/2608.04144) | 2026-08-04 | None recorded | Abstract | MV | Lower relevance. |
| Training-Free BioNEL-RAG, Chen et al. [arXiv 2610.14789](https://arxiv.org/abs/2610.14789) | 2026-10-10 | None recorded | Abstract | MV | Lower relevance. |
| LELA, Haffoudhi et al. [arXiv 2605.26956](https://arxiv.org/abs/2605.26956), [DOI](https://doi.org/10.24963/ijcai.2026/966), [LNCS DOI](https://doi.org/10.1007/978-3-032-42026-8_1) | 2026-05-26 (MV); see disagreement table | IJCAI 2026 demo; LNCS online 2026-10-23 | Abstract (demo) / Metadata (LNCS) | MV, DJ | Zero-shot entity linking framework with code. |
| Surgical notes linking, Shin et al. [DOI](https://doi.org/10.1186/s12911-026-03877-4) | None recorded | *BMC MIDM*, 2026-10-03 | Abstract | MV | +2.7 Recall@1. |
| Think Before You Link, Pengpun et al. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2309/), [arXiv 2609.10745](https://arxiv.org/abs/2609.10745) | 2026-09-09 | EMNLP 2026 main | Abstract | MV, DJ | Multimodal linking, despite the title. |
| CAREL, Yang et al. [DOI](https://doi.org/10.1145/3805696.3846031) | None recorded | JCDL 2026, 2026-10-13 | Abstract | DJ | Direct mapping method, including "no match" at F1 88. African-studies keywords, not running text. |
| Generating MeSH terms, Lee [DOI](https://doi.org/10.1177/01655515261490025) | None recorded | *JIS*, 2026-10-12 | Abstract | DJ | Best F1 58.4% on 1,730 articles. |
| Generative clinical EL in Spanish, Zotova et al. [DOI](https://doi.org/10.1016/j.array.2026.100805) | Crossref record 2026-04-08 (**PRE**) | *Array*, Jul 2026 issue | Abstract | MV | Generate then match beats retrieve-and-rerank here. |

### Track B: subject indexing and multi-label mapping

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| Hydropower Licensing Classification, Yoon et al. [arXiv 2608.23241](https://arxiv.org/abs/2608.23241) | 2026-08-24 | None recorded | Abstract† | MV | Closest structural match: long government documents, paragraph-level, 135 categories, 40 with no examples. Hybrid micro-F1 0.524. |
| Transferability Between XML and HTC, Hauss et al. [DOI](https://doi.org/10.1145/3820755.3832808) | None found | ACM DocEng 2026, 2026-08-18 | Abstract (ACM 403) | MV | Extreme methods transfer well. No numbers. |
| AstroConcepts, Alkan et al. [ACL Anthology](https://aclanthology.org/2026.nslp-1.1/), [arXiv 2604.02156](https://arxiv.org/abs/2604.02156) | **PRE** 2026-04-02 | NSLP at LREC 2026, May 2026 | Part | MV | Constrained zero-shot LLM leads on rare labels: tail F1 0.198 against 0.081. |
| TAGHN, Bashyam et al. [arXiv 2610.20048](https://arxiv.org/abs/2610.20048), [DOI](https://doi.org/10.1145/3805696.3846044) | 2026-10-14 | JCDL 2026, 2026-10-13 per Crossref | Abstract | MV | No LLM. STW economics thesaurus, macro-F1 0.380. |
| National Library of Spain indexing, Usta [DOI](https://doi.org/10.1108/lht-10-2025-0269) | None recorded | *Library Hi Tech*, 2026-08-26 | Abstract | MV | 2.4 million Spanish records; the odd 97/3 blend deserves checking. |
| R3 two-stage indexing, Xia et al. [DOI](https://doi.org/10.1177/01655515261449401) | None recorded | *JIS*, 2026-07-15 | Abstract | MV, DJ | Candidates from similar documents. MAP from 41.19% to 45.24%. |
| Constraint-Aware Hierarchical Search, Wang et al. [arXiv 2607.10588](https://arxiv.org/abs/2607.10588) | 2026-07-12 | None recorded | Abstract | MV | Definitions and exclusions, level by level. |
| Thakaa at AraGenre, Alqaeri et al. [ACL Anthology](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/) | None recorded | Arabic NLP shared tasks, Oct 2026 | Abstract | MV | Full taxonomy reduced topic traps. Macro-F1 0.7352. Authors disclose hand-written cues. |
| Mosaic, Chatterjee et al. [ACL Anthology](https://aclanthology.org/2026.emnlp-industry.33/) | None recorded | EMNLP 2026 Industry, Oct 2026 | Abstract | MV | Generates label descriptions for a 140-class taxonomy. |
| RHC, Jiang et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.541/), [arXiv 2510.07167](https://arxiv.org/abs/2510.07167) | **PRE** 2025-10-08 | Findings of ACL 2026, Jul 2026 | Abstract | MV | About +3% over supervised fine-tuning. |
| SCOTTA, Xu et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.822/) | None found | Findings of ACL 2026, Jul 2026 | Abstract | MV | Per-label confidence with no parameter updates. |
| RelaClass, He et al. [DOI](https://doi.org/10.1109/ijcnn69854.2026.11730021) | None recorded | IJCNN 2026, Jun 2026 | Abstract | MV | Class names only as supervision. |
| TEAMMix, Zhang et al. [arXiv 2608.11044](https://arxiv.org/abs/2608.11044), [DOI](https://doi.org/10.1109/cscwd68734.2026.11582679) | 2026-08-11 | IEEE CSCWD 2026, Crossref 2026-05-13 | Abstract | MV | LLM pseudo-samples for rare labels. |
| Push and Pull, Van Nooten and Kosar [ACL Anthology](https://aclanthology.org/2026.lrec-1.583/) | None recorded | LREC 2026, May 2026 | Abstract | MV | Embedding-only, therefore cheap. |
| LLM Distillation for Topics, Kasliwal and Singh [arXiv 2610.09063](https://arxiv.org/abs/2610.09063) | 2026-10-06 | None recorded | Abstract† | MV | Route to labelling millions: strong LLM on a sample, then a small student. |
| Circuits of LLM Classifiers, Khalaf et al. [arXiv 2610.30076](https://arxiv.org/abs/2610.30076) | 2026-10-21 | None recorded | Abstract | MV | Most formal treatment of cheap-model-first. |
| Jev-as-a-Judge, Do Rosario and Atef [arXiv 2610.20634](https://arxiv.org/abs/2610.20634) | 2026-09-30 | None recorded | Abstract | MV | Tie at 187-fold cost difference, yet they disagree on 15.2%. |
| Student-Guided Teacher Distillation, Wu et al. [arXiv 2610.02516](https://arxiv.org/abs/2610.02516) | 2026-10-01 | None recorded | Abstract | MV | Student narrows candidates for a reranker. |
| KREL, Chen et al. [arXiv 2608.20887](https://arxiv.org/abs/2608.20887) | 2026-08-21 | Accepted to Findings of EMNLP 2026 | Abstract | MV | Medical coding on long notes. No numbers. |
| Post-Training for Medical Coders, Wang et al. [arXiv 2606.13940](https://arxiv.org/abs/2606.13940) | 2026-06-11 | None recorded | Abstract† | MV | Prompt-only evaluation underestimates LLMs. |
| LGSA, Zhang et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.1322/) | None recorded | ACL 2026 | Abstract | MV | Needs supervised training. |
| SPUR-XMC, Mukherjee et al. [ACL Anthology](https://aclanthology.org/2026.findings-emnlp.1597/) | None found | Findings of EMNLP 2026 | Abstract | MV | Tail labels in biomedical literature. |
| HASTE, Ullah et al. [arXiv 2606.01117](https://arxiv.org/abs/2606.01117) | 2026-05-31 | ICML 2026, per comment | Abstract | MV | Training speedups for large output spaces. |
| LoC Subject Indexing Pipeline, Chow [arXiv 2605.03537](https://arxiv.org/abs/2605.03537) | 2026-05-05 | None recorded | Abstract | MV | Only ten titles evaluated. |
| UKRI Grant Proposals, Ruan et al. [arXiv 2606.30304](https://arxiv.org/abs/2606.30304) | 2026-06-29 | None recorded | Abstract | MV | 90.5% on only 42 proposals. |
| XMTC Library Dataset, D'Souza et al. [ACL Anthology](https://aclanthology.org/2026.lrec-1.12/), [arXiv 2603.10876](https://arxiv.org/abs/2603.10876) | **PRE** 2026-03-11 | LREC, May 2026 | Part | MV | 136,569 English and German records with GND labels. |
| Golub et al., Swedish fiction indexing [DOI](https://doi.org/10.1108/jd-06-2026-0363) | None recorded | *J. Documentation*, 2026-09-22 | Abstract | MV, DJ | F1 44% on metadata against 12% on full text: a warning for long documents, though on fiction. |

### Track B: UN, SDG, policy and humanitarian mapping

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| LLM-INSTRUCT at UZH, Tran et al. [arXiv 2607.20430](https://arxiv.org/abs/2607.20430), [ACL Anthology](https://aclanthology.org/2026.argmining-1.11/) | 2026-05-10 | ArgMining, Jul 2026 | Abstract† | MV | First overall. Constrained decoding, debate for uncertain cases. |
| RESOLVENOW at UZH, Gupta et al. [ACL Anthology](https://aclanthology.org/2026.argmining-1.12/) | None recorded | ArgMining, Jul 2026 | Abstract | MV | Embedding retrieval "dropped relevant tags before the LLM saw them". |
| AI Safety Initiatives taxonomy, Semitsu et al. [ACL Anthology](https://aclanthology.org/2026.politicalnlp-1.30/) | None recorded | PoliticalNLP at LREC 2026, May 2026 | Abstract | MV | Model choice substantially changes results. |
| ESRS E1 Query Expansion, Ishisaka [ACL Anthology](https://aclanthology.org/2026.climatenlp-1.8/) | None recorded | ClimateNLP at EMNLP 2026, Oct 2026 | Abstract | MV | Inverse framing: for each concept, find chunks. Recall@10 0.842 against 0.702, 114 labels. |
| Can LLMs Replace Keyword-Based SDG Mappings?, Charalampous et al. [DOI](https://doi.org/10.3390/su182010316) | None recorded | *Sustainability*, 2026-10-10 | Abstract (MDPI 403) | MV, DJ | Over-assignment on 5,000 OSDG texts; not a direct replacement. |
| Alignment and Boundary Shifts in SDG, Wang et al. [DOI](https://doi.org/10.21203/rs.3.rs-9413287/v1) | 2026-06-15 | Research Square | Abstract | MV | 1.76 million articles. Here the LLM tightens against rules, the opposite direction. |
| Before You Trust the Count, Volkov [ACL Anthology](https://aclanthology.org/2026.nlp4pi-1.7/) | None found | NLP4PI at EMNLP 2026, Oct 2026 | Part | MV | "Always accept" gets OSDG F1 0.973, a trap. Translated pairs disagree 33.0% (en-es). |
| UK charities SDG dataset, Chen [DOI](https://doi.org/10.31235/osf.io/tky7s_v1) | 2026-07-21 | SocArXiv | Abstract | MV | 77.3% on a blind sample of 150. Single labeller. |
| SDG curriculum mapping, Bertone et al. [DOI](https://doi.org/10.1108/ijshe-12-2025-1514) | None recorded | 2026-07-16 | Abstract | MV | 241 course profiles, eight experts. |
| ITL, Giráldez et al. [arXiv 2608.27031](https://arxiv.org/abs/2608.27031) | 2026-08-27 | None recorded | Abstract | MV | Tested only for internal consistency. |
| Conflict Monitoring in West Africa, Muki and Owolabi [arXiv 2605.04177](https://arxiv.org/abs/2605.04177) | 2026-05-05 | None recorded | Abstract | MV | One-sided errors: aggregate F1 can hide them. |
| AGROVOC-grounded RDF, Vianna do Nascimento [DOI](https://doi.org/10.5753/encompif.2026.21874) | None recorded | EnCompIF 2026, 2026-07-19 | Abstract | MV | URI precision 100%, 16.2%, 8.1% by LLM on 38 species. Validate identifiers in code. |
| MOSAIC radiology, Schiavone et al. [ACL Anthology](https://aclanthology.org/2026.clinicalnlp-1.35/) | None recorded | CLINICALNLP at LREC 2026, May 2026 | Abstract | MV | 4B model across English, Spanish, French and Danish. |
| Cattle Reporting Categories, Raymond et al. [arXiv 2609.09381](https://arxiv.org/abs/2609.09381) | 2026-09-08 | None recorded | Abstract | MV | A target vocabulary may lack the needed granularity. |
| From Policy Documents to Survey Responses, Cole et al. [DOI](https://doi.org/10.1007/978-981-92-2480-7_20), [arXiv 2609.29370](https://arxiv.org/abs/2609.29370) | 2026-09-24, after online publication | LNCS, online 2026-07-09 | Full† | DJ | Strongest mapping example on multi-country policy: overlap 84-95%, on a lenient measure. LLM assigned 2,336 policy-instrument labels against 1,281 by humans. |
| Experimenting with AI in CGIAR, Corsetti et al. [DOI](https://doi.org/10.4324/9781003799139-19) | None found | 2026-08-18 | Part (opening) | DJ | Reflective account, no ontology seen. |
| World Bank IEG chapter, Anuj et al. [DOI](https://doi.org/10.4324/9781003799139-9) | None found | 2026-08-18 | Part (excerpts) | DJ | Screening F1 0.67 on a test set of 30. Precision reported inconsistently. |
| From ritual to reflex, Martin et al. [DOI](https://doi.org/10.4324/9781003799139-6) | None found | 2026-08-18 | Part (excerpts) | DJ | Scorecard indicators; LLMs varied widely. |
| Partnership principle assessment, Lyubashenko [DOI](https://doi.org/10.4324/9781003799139-16) | None found | 2026-08-18 | Part (excerpts) | DJ | No formal accuracy metrics. |
| Sifting for gold, Sandnes et al. [DOI](https://doi.org/10.4324/9781003799139-14) | None found | 2026-08-18 | Part | DJ | Nearest audit example, Norwegian. 22 models, median accuracy 81.5%. |
| Other book chapters: [-8](https://doi.org/10.4324/9781003799139-8), [-29](https://doi.org/10.4324/9781003799139-29), [-27](https://doi.org/10.4324/9781003799139-27) | None found | 2026-08-18 | Part (openings), abstract for -27 | DJ | Not directly about extraction. |
| Extracting Dataset Mentions in FCV Documents, Macalaba et al. [arXiv 2609.12107](https://arxiv.org/abs/2609.12107) | 2026-09-10 | Not found | Abstract† | DJ | Small model proposes, LLM validates, labels retrain the small model. Precision 74.1%, recall 70.5%. |
| Fair Geolocation Extraction, Cafferata et al. [DOI](https://doi.org/10.1145/3774905.3795457) | None recorded | WWW Companion, online 2026-05-28 | Abstract (closed) | DJ | Few-shot NER on HumSet. No numbers. |
| Power Marketing Policy, Peng et al. [DOI](https://doi.org/10.1109/eespe68405.2026.11648568) | None recorded | IEEE EESPE, Jun 2026 | Abstract (DJ) / Metadata (OG) | DJ, OG | Filters boilerplate first; verbatim consistency 0.976 against 0.412. 47 documents. |
| AI Risks in Multi-Institutional Policy Texts, Sun [DOI](https://doi.org/10.1109/aice71085.2026.11736963) | None recorded | IEEE AICE, Sep 2026 | Abstract (DJ) / Metadata (OG) | DJ, OG | NER F1 91.96% on a self-built dataset. |
| Indicator-Centric KG, Sanovia et al. [DOI](https://doi.org/10.3390/data11090242) | None recorded | *Data*, 2026-09-17 | Abstract | DJ | Hand-designed SDG indicator graph. Not built from text. |
| IPCC scenario comparison, Warin and Bisson [DOI](https://doi.org/10.1371/journal.pclm.0000965) | None recorded | *PLOS Climate*, 2026-07-14 | Abstract | DJ | Qualitative, not a graph. |
| Cities' Climate Action Plans, Peddi et al. [DOI](https://doi.org/10.1109/icetci71772.2026.11710631) | None recorded | IEEE ICETCI, Aug 2026 | Abstract | DJ | Layout-aware; F1 0.931 for mitigation vs adaptation. |
| Climate equity policy, Choi and Lee [DOI](https://doi.org/10.1007/s43762-026-00279-0) | arXiv 2601.06703 by ID (**PRE**; not checked) | 2026-07-28 | Abstract | DJ | RAG semantic analysis. |
| CliReNER, Poleksić and Martinčić-Ipšić [DOI](https://doi.org/10.1007/s10584-026-04324-z) | 2026-05-06 ([Research Square](https://doi.org/10.21203/rs.3.rs-9445054/v1)) | *Climatic Change*, online 2026-11-03 | Abstract | DJ | Realistic ceiling for fine-grained typing: best strict micro F1 0.62, 28 entity types. |
| Annual-report evidence packs, Gao and Zhao [DOI](https://doi.org/10.7717/peerj-cs.4152) | None recorded | *PeerJ CS*, 2026-10-25 | Abstract | DJ | Spearman 0.910, but paragraph recall@30 only 0.419. |
| Longitudinal Policy Measures, Galvez-Hernandez [DOI](https://doi.org/10.66534/hp.2026.0015) | None recorded | *Health Politics*, 2026-09-30 | Abstract | DJ | Eligibility conditions were less accurate. |
| Prompt Optimization for Policy Classification, Sajid [DOI](https://doi.org/10.31222/osf.io/ja7bs_v1) | 2026-09-16 | MetaArXiv | Abstract | DJ | Neither prompt type reached F1 70%. |
| GovText, Mehta et al. [DOI](https://doi.org/10.1145/3805696.3848487) | None recorded | JCDL 2026, 2026-10-13 | Abstract | DJ | 9.02 million US government PDFs; a parse-quality reference. |
| ESG Disclosure Quality, Park [DOI](https://doi.org/10.1002/bse.71277) | None recorded | *BSE*, online 2026-07-07 | Abstract | DJ | Adjacent. |
| LLM and Rule-Based ESG Assessment, Jung et al. [DOI](https://doi.org/10.3390/su18136882) | None recorded | *Sustainability*, 2026-07-06 | Abstract | DJ | Adjacent. |
| ESG Report RAG Framework, Solanakis et al. [DOI](https://doi.org/10.1109/iisa71656.2026.11705446) | None recorded | IEEE IISA, Jul 2026 | Abstract | DJ | 20,000 ESG PDFs chunked; no extraction metrics. |
| Pathway for Assessing Grey Literature, Salatino et al. [arXiv 2608.24926](https://arxiv.org/abs/2608.24926) | 2026-08-19 | STI-ENID 2026, per comment | Abstract | DJ | Calls for papers mapped to OpenAlex and others. |
| COCI demo, Salatino et al. [arXiv 2608.24559](https://arxiv.org/abs/2608.24559) | 2026-08-25 | ISWC 2026 demo, per comment | Abstract | DJ | Companion to the pathway paper. |
| Grey-Literature Synthesis Pipeline, Starace et al. [DOI](https://doi.org/10.1093/eurpub/ckag169.272) | None recorded | *Eur. J. Public Health*, online 2026-10-26 | Abstract | DJ | 39% of 125 verdicts confirmed, 22% rewritten. |
| Ecological Metadata Extraction, Moorthy et al. [DOI](https://doi.org/10.1111/2041-210x.70403) | None recorded | *Methods in Ecology and Evolution*, online 2026-08-26 | Abstract | DJ | Fields from abstracts, not full reports. |
| Audit data review, Weiyi Li [DOI](https://doi.org/10.54097/a2mdak19) | None recorded | 2026-08-18 | Abstract | DJ | No numbers. |
| LLMs in Public Policy Research, Lan et al. [DOI](https://doi.org/10.1057/s41599-026-08575-0) | None recorded | *HSSC*, 2026-08-03 | Abstract | DJ | Conceptual framework. |
| AI in Policymaking, Lnenicka et al. [DOI](https://doi.org/10.1016/j.giq.2026.102138) | Crossref record 2026-04-01 (**PRE**) | *GIQ*, Jun 2026 issue | Abstract | DJ | Fewer AI actions target monitoring and evaluation. |
| Campbell Review of Reviews, Wei et al. [DOI](https://doi.org/10.1177/18911803261454702) | None recorded | *Campbell*, online 2026-06-03 | Abstract | DJ | Evidence supports supervised screening; other tasks limited. |
| Living Rapid Review of AI for Evidence Synthesis (authors not retrieved) [DOI](https://doi.org/10.1017/rsm.2026.10118) | None recorded | *Research Synthesis Methods*, 2026-10-08 | Abstract | DJ | LLM data extraction median accuracy 63% across 134 studies. |
| Knowledge Synthesis Review Framework, Shafqat et al. [arXiv 2608.12741](https://arxiv.org/abs/2608.12741) | 2026-08-13 | None recorded | Abstract | DJ | 244 documents including policy; screening accuracy 82.8% best. |
| Scientific Literature and Topic Modeling, Smetana and Khazanovich [DOI](https://doi.org/10.1007/s11192-026-05643-9) | arXiv 2510.16152 by ID (**PRE**; not checked) | *Scientometrics*, online 2026-06-05 | Abstract | DJ | Over 1,500 PNAS articles. |
| Power Industry Policy Conflict Detection, Huang et al. [DOI](https://doi.org/10.1109/CCSB71176.2026.11716430) | None recorded | IEEE, Aug 2026 | Abstract | DJ | Simulation data. |
| Scholarly KG over ETDs, Chekuri and Fox [DOI](https://doi.org/10.1145/3805696.3846035) | None recorded | JCDL, 2026-10-13 | Abstract | DJ | Formal ontology over chapters, figures and captions. |
| ESG-Kor [arXiv 2610.21368](https://arxiv.org/abs/2610.21368) | 2026-10-15 posting of older work | Findings of EMNLP 2024, per its arXiv record | Not read | DJ | Not treated as new. |
| Monitoring Global Aid Flows, Luo et al. [RePEc](https://ideas.repec.org/p/wbk/wbrwps/11248.html) | **PRE** 2025 working paper | World Bank WP 11248 | Abstract | MV | Closest work to DAC sectors; it discovers categories rather than mapping. |

### Track B: noisy PDFs, tables and languages

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| SOLE, Fouss [DOI](https://doi.org/10.1145/3805696.3846009) | None recorded | JCDL 2026, 2026-10-13 | Abstract | DJ | Most applicable noise numbers. Raw open-weight 0.216 partial-mention F1 on OCR; about 0.78 entity-level F1 with a pipeline, on different measures. |
| Scope3Trace, Zheng et al. [arXiv 2607.17122](https://arxiv.org/abs/2607.17122) | 2026-07-19 | EMNLP venue unconfirmed | Part† | DJ | Best measured table effect: removing reconstruction cuts exact match from 0.96 to 0.75. A narrow numeric task. |
| From Pixels to Pairs, Anvari [DOI](https://doi.org/10.21203/rs.3.rs-11199953/v1) | 2026-10-05 | Research Square | Abstract | DJ | Rankings reverse between clean and OCR text. No numbers in the abstract. |
| Historical Documents in KGs, Nicolau et al. [arXiv 2607.24475](https://arxiv.org/abs/2607.24475) | 2026-07-27 | ICDAR 2026, per comment | Full† | DJ | Query robustness on marriage records, not extraction. |
| Reframing Historical Text Extraction, Viana [DOI](https://doi.org/10.3390/info17080722) | None recorded | *Information*, 2026-07-25 | Abstract | DJ | Measurement expressions stay risky, especially on table pages. |
| Beyond Accuracy for Vision-Language Models, Patel et al. [arXiv 2609.15706](https://arxiv.org/abs/2609.15706) | 2026-09-14 | None recorded | Abstract | DJ | Synthetic checks. |
| Layout-Aware Parsing and KG, Jain et al. [DOI](https://doi.org/10.1109/ICISC69558.2026.11681334) | None recorded | IEEE ICISC, Aug 2026 | Abstract (truncated) | DJ | Aimed at education. |
| PDF Strategies for RAG Objectives, Heo et al. [DOI](https://doi.org/10.14801/jkiit.2026.24.9.59) | None recorded | *J. KIIT*, 2026-09-30 | Abstract | DJ | Simple text was better for evidence recall. |
| Cheap First, Ask the LLM Later?, Patel [DOI](https://doi.org/10.20944/preprints202610.2704.v1) | 2026-10-27 | Preprints.org | Abstract | DJ | Native PDF F1 0.708, extracted text 0.657; not significant. |
| Query-Driven Multimodal Extraction, Gao et al. [arXiv 2608.22214](https://arxiv.org/abs/2608.22214) | 2026-08-23 | None recorded | Abstract | DJ | 2,455 pages, 316 queries. |
| Long-Document Retrieval survey, Li et al. [DOI](https://doi.org/10.1145/3849382) | None recorded | *ACM TOIS*, 2026-09-24 | Abstract | DJ | Survey. |
| Multilingual Disaster Analysis, Karaca et al. [DOI](https://doi.org/10.1109/acit69831.2026.11700021) | None recorded | IEEE ACIT, Sep 2026 | Abstract | DJ | Turkish and English social posts, not reports. |
| KGs and multilingual SSH corpora, Faci et al. [arXiv 2607.05956](https://arxiv.org/abs/2607.05956) | 2026-07-07 | LLMs4SSH at LREC 2026, per comment | Abstract | DJ | Ongoing use case, no results. |
| League of Nations Index Cards, Cafiero and Mallard [arXiv 2606.24895](https://arxiv.org/abs/2606.24895) | 2026-06-04 | None recorded | Abstract | DJ | A curiosity. Possible third author; check the page. |

### Track B: ontology matching for aligning agency vocabularies

OAEI 2026 results are preliminary or recently frozen, and system papers were due 12 Nov 2026, so no overview paper exists yet (MV). LogMapLLM reached macro F1 0.623 on Bio-ML against 0.534 for LogMap, but on tracks most like multilingual thesauri scores stay low: LSMatch Multilingual 2.0 has F1 0.412 on MultiFarm, and the Digital Humanities track's best average is 0.61 ([Bio-ML](https://bio-ml.oaei-ml.org/results/), [MultiFarm](http://oaei.ontologymatching.org/2026/results/multifarm/index.html), [DH](https://github.com/FelixFrizzy/DH-benchmark/blob/main/results/OAEI2026/results.md)). Anatomy figures passed through a summarising fetch† ([Anatomy](http://oaei.ontologymatching.org/2026/results/anatomy/index.html)). Bio-ML block order is the researcher's assumption from the tab order.

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| DRAMA, Oefler et al. [PDF](https://om.ontologymatching.org/2026/papers/om2026-LTpaper2.pdf) | None established | OM-2026, 25 Oct 2026 | Part | MV | Staged pipeline cuts search space 80 to 91%. The debate step changed zero decisions with GPT-5.5. |
| Which Model, Which Prompt?, Dilworth et al. [PDF](https://om.ontologymatching.org/2026/papers/om2026-LTpaper5.pdf) | None established | OM-2026 | Part | MV | Most practical guide: 86 models over 3,792 runs, and no configuration wins consistently. |
| Arabic-English Ontology Matching, Fallatah [PDF](https://om.ontologymatching.org/2026/papers/om2026-STpaper3.pdf) | None established | OM-2026 | Part | MV | Same-ontology F1 up to 0.892; different ontologies 0.183 to 0.198. |
| OntoAligner-Ensemble, Babaei Giglou et al. [arXiv 2608.31137](https://arxiv.org/abs/2608.31137) | 2026-08-31 | Accepted to OM-2026 | Abstract | MV | Mixed-paradigm ensembles raise precision. |
| AgentMap, Song et al. [arXiv 2607.27130](https://arxiv.org/abs/2607.27130) | 2026-07-29 | None recorded | Abstract | MV | Finds an equivalent, otherwise the most specific broader concept. |
| Crowd-OM, Qiang et al. [arXiv 2605.12226](https://arxiv.org/abs/2605.12226) | 2026-05-12 | None recorded | Abstract | MV | LLM matchers produce more candidates than experts can review. |
| Defence and National Security Ontologies, Dilworth et al. [arXiv 2608.05867](https://arxiv.org/abs/2608.05867) | 2026-08-06 | ISWC 2026 Resource Track | Abstract | MV | Consensus plus manual validation builds a silver standard. |
| Self-Demonstrations for Schema-Ontology Mapping, Thombre et al. [arXiv 2609.13776](https://arxiv.org/abs/2609.13776) | 2026-09-12 | None recorded | Abstract | MV | Database schemas, not text. |
| Open Ontologies, Rovai [arXiv 2605.09184](https://arxiv.org/abs/2605.09184) | 2026-05-09 | None recorded | Abstract | MV | Reading raw OWL scored F1 0.323, worse than no file at 0.431. |
| Hybrid LLM and Embedding Matching, Dahir and El Qadi [DOI](https://doi.org/10.1109/iccsc71566.2026.11650532) | None recorded | ICCSC 2026, Jul 2026 | Abstract | MV, DJ | Matching through a reference ontology. No figures. |
| Awareness Without Synthesis, Bossler et al. [DOI](https://doi.org/10.21203/rs.3.rs-9588669/v1) | 2026-05-13 | Research Square | Abstract | MV | Vocabulary fragmentation; human annotation α=0.752. |
| DOKA-Agent, Qin et al. [DOI](https://doi.org/10.21203/rs.3.rs-9503150/v1) | 2026-07-03 | Research Square | Abstract | MV | Evaluated on entity alignment, not vocabularies. |
| Xonto-RAG, Wang et al. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.201/) | None recorded | EMNLP 2026 main, Oct 2026 | Abstract | MV | Keeps ontologies separate with typed bridges. |
| Survey on Embedding Methods for Matching, Sousa et al. [DOI](https://doi.org/10.1145/3805799) | None recorded | *ACM Computing Surveys*, online 2026-05-15 | Abstract | DJ | Survey. |

### Track B: how mapping quality is measured

| Paper | First posted | Published | Read | Note | What it offers |
|---|---|---|---|---|---|
| LLMs as Judges for Subject Indexing, Włodarczyk and Zdunek [DOI](https://doi.org/10.1080/01639374.2026.2714375) | None recorded | *CCQ*, 2026-08-14 | Abstract (publisher 403) | MV | LLMs are weak at catching wrong descriptors. |
| Failure Analysis of LLM Annotators, Hossain et al. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.2037/) | None recorded | EMNLP 2026 main, Oct 2026 | Abstract | MV | Agreement across prompts can make things worse where bias is shared. |
| What Do Biomedical Benchmarks Measure?, Leaman et al. [ACL Anthology](https://aclanthology.org/2026.bionlp-1.66/), [arXiv 2605.20537](https://arxiv.org/abs/2605.20537) | 2026-05-19 | BioNLP at ACL 2026, Jul 2026 | Abstract | MV | Similar tasks differ substantially in overlap. |
| Hierarchy-Aware Phenotype Linking, Labbe et al. [ACL Anthology](https://aclanthology.org/2026.bionlp-1.5/) | None recorded | BioNLP at ACL 2026, Jul 2026 | Abstract | MV | Hierarchy-aware evaluation beyond exact match. |
| OntologyBench, Zhang et al. [ACL Anthology](https://aclanthology.org/2026.emnlp-main.595/), [arXiv 2609.08174](https://arxiv.org/abs/2609.08174) | 2026-09-07 | EMNLP 2026 main, Oct 2026 | Abstract | MV | LLM scoring gave little end-to-end gain. |
| SNOMED CT Concept Recommendation, Noori [arXiv 2609.17855](https://arxiv.org/abs/2609.17855) | 2026-09-15 | None recorded | Abstract | MV | Recall@10 is 7.74% for rare concepts, 43.90% for common. |
| MADE, Agarwal et al. [ACL Anthology](https://aclanthology.org/2026.acl-long.2148/) | None recorded | ACL 2026, Jul 2026 | Abstract (truncated) | MV | Saturated benchmarks. |
| Reference-Free Evaluation of Taxonomies, Wullschleger et al. [ACL Anthology](https://aclanthology.org/2026.findings-acl.1273/) | None recorded | Findings of ACL 2026, Jul 2026 | Abstract | MV | Judges a taxonomy with no labels. |
| Several Needles in a Haystack, Henning et al. [arXiv 2609.26468](https://arxiv.org/abs/2609.26468) | 2026-09-22 | None recorded | Abstract | MV | Standard calibration misleads when most labels are negative. |
| One Score, Two Decisions, Jiang et al. [arXiv 2608.14683](https://arxiv.org/abs/2608.14683) | 2026-08-04 | None recorded | Abstract | MV | The top-two margin can remove the signal for "no answer". |
| Uncertainty-Aware Routing, Zecevic et al. [ACL Anthology](https://aclanthology.org/2026.bionlp-1.16/) | None recorded | BioNLP at ACL 2026, Jul 2026 | Abstract | MV | Reports review burden as well as accuracy. |
| PASC, Kotte [arXiv 2605.18812](https://arxiv.org/abs/2605.18812) | 2026-05-12 | None recorded | Abstract | MV | Pipeline-aware conformal prediction. |
| Sense Alignment Across Dialect Dictionaries, Mederake et al. [ACL Anthology](https://aclanthology.org/2026.vardial-1.10/) | None recorded | VarDial, Mar 2026 (**PRE** publication) | Abstract | MV | Model agreement κ=0.73, human κ=0.74. |
| Multi-Label Tendency in Zero-Shot Coding, Choo and Hong [ACL Anthology](https://aclanthology.org/2026.healing-1.14/) | None recorded | HeaLing, Mar 2026 (**PRE** publication) | Abstract | MV | Output format changes how many labels LLMs give. |

## Unread papers to obtain: nothing below has been read in full

**These are not excluded for being unread.** Part 1 lists papers where no researcher read the body and the abstract was seen. Part 2 lists papers known only by title and metadata, or by snippet. In Part 1, content claims come from abstracts only, some through a fetch tool. In Part 2, nothing is claimed beyond the title unless stated. Where there is no DOI, the available identifier is given and the row says so.

### Part 1: abstract seen, full text blocked or not opened (obtain first)

| Paper | Identifier | Why it is unread | Why obtain |
|---|---|---|---|
| AutoSchema (KDD 2026) | [10.1145/3770855.3817812](https://doi.org/10.1145/3770855.3817812) | ACM 403; Crossref shows a licence and Semantic Scholar marks it open access | Schema instability |
| Extraction-Oriented Ontology Modeling (ER 2026) | [10.1007/978-3-032-40795-5_5](https://doi.org/10.1007/978-3-032-40795-5_5) | Paywalled; figures† | 915 against 20 classes |
| LLM KG for Cultural Heritage (EKAW 2026) | [10.1007/978-3-032-41614-8_7](https://doi.org/10.1007/978-3-032-41614-8_7) | Paywalled; figures† | Counter-evidence on validation |
| Ontology-Aware Triple Extraction (ER 2026) | [10.1007/978-3-032-40377-3_5](https://doi.org/10.1007/978-3-032-40377-3_5) | Paywalled | Refinement against model size |
| Ontology-Guided and LLM-Enhanced KG | [10.1007/978-3-032-24807-7_1](https://doi.org/10.1007/978-3-032-24807-7_1) | Paywalled | Scanned PDFs |
| Extract, Align, Reflect | [10.1007/978-981-92-3426-4_13](https://doi.org/10.1007/978-981-92-3426-4_13) | Paywalled | Extract then align |
| Causal Evidence Extraction in Crisis Reports | [10.18653/v1/2026.findings-acl.1626](https://doi.org/10.18653/v1/2026.findings-acl.1626) | Abstract only; body not read | Reusable extraction target |
| Ontology-Aware Prompting | [10.3233/SSW260009](https://ebooks.iospress.nl/doi/10.3233/SSW260009) | Open access not confirmed | Filtering design |
| Text2Onto-Agent | [10.1061/jcemd4.coeng-18479](https://doi.org/10.1061/jcemd4.coeng-18479) | Not opened | Persistent ontology memory |
| AOCG-LLM+ | [10.1007/s40747-026-02372-7](https://doi.org/10.1007/s40747-026-02372-7) | Springer | Hierarchy clustering |
| Extended NeOn-GPT | [10.1177/22104968261453138](https://doi.org/10.1177/22104968261453138) | Sage 403 | Methodology-led prompting |
| Systematic Literature Review (Semantic Web) | [10.1177/22104968261465514](https://doi.org/10.1177/22104968261465514) | Sage 403; a likely preprint is at [semantic-web-journal.net](https://semantic-web-journal.net/system/files/swj4039.pdf), first page checked | Reference list |
| OF-Bio-KG, two papers | [10.1109/COMPSAC69091.2026.00263](https://doi.org/10.1109/COMPSAC69091.2026.00263), [10.1109/COMPSAC69091.2026.00287](https://doi.org/10.1109/COMPSAC69091.2026.00287) | Not opened | Cluster-first on long regulatory text |
| AgriEvoKG | [10.62762/tetai.2026.882738](https://doi.org/10.62762/tetai.2026.882738) | Not opened | Drift evidence |
| EventForge | [10.30574/wjarr.2026.31.2.2105](https://doi.org/10.30574/wjarr.2026.31.2.2105) | Not opened | Weak evidence; low priority |
| Overview of automated ontology creation | [10.17759/mda.2026160205](https://doi.org/10.17759/mda.2026160205) | Not opened | Hybrid proposal |
| Can LLMs Replace SDG Mappings? | [10.3390/su182010316](https://doi.org/10.3390/su182010316) | MDPI 403; open access per the journal | Identity of models and F1 unknown |
| Graphing the European Green Deal | [10.3390/su18126193](https://doi.org/10.3390/su18126193) | MDPI 403 | Policy entity type set |
| Metallurgical Failure Reports | [10.3390/app16126178](https://doi.org/10.3390/app16126178) | Publisher 403 | Source localisation |
| Transferability Between XML and HTC | [10.1145/3820755.3832808](https://doi.org/10.1145/3820755.3832808) | ACM 403 | Method transfer |
| LLMs as Judges for Subject Indexing | [10.1080/01639374.2026.2714375](https://doi.org/10.1080/01639374.2026.2714375) | Publisher 403 | Human agreement |
| Fair Geolocation Extraction | [10.1145/3774905.3795457](https://doi.org/10.1145/3774905.3795457) | Closed per OpenAlex | HumSet |
| TaxoEvolve (CIKM 2026) | [10.1145/3799682.3841091](https://doi.org/10.1145/3799682.3841091) | ACM | Incremental taxonomy |
| WikiTabLLM (CIKM 2026) | [10.1145/3799682.3840636](https://doi.org/10.1145/3799682.3840636) | ACM | Table population |

The AIDA chapter was read in part, but its last pages were not, so also finish it at [its DOI](https://doi.org/10.4324/9781003799139-20) in case they hold version 2 metrics (DJ).

### Part 2: title and metadata only, or snippet only

| Paper | Authors | Date recorded | Identifier | Note |
|---|---|---|---|---|
| Less Is More: KG Construction and Schema-Guided Canonicalization | Park et al. | CIKM 2026, 2026-11-06 | [10.1145/3799682.3839949](https://doi.org/10.1145/3799682.3839949) | OG, SF metadata; DJ abstract |
| OntoForge | Guo, Zhu, Wu | CIKM 2026, 2026-11-06 | [10.1145/3799682.3839852](https://doi.org/10.1145/3799682.3839852) | OG metadata; DJ abstract |
| Self-Evolving Ontology via Cumulative Definition Enrichment | Oh, Kim, Moon | CIKM 2026, 2026-11-06 | [10.1145/3799682.3840115](https://doi.org/10.1145/3799682.3840115) | SF metadata; DJ abstract: Korean government complaints |
| Schema-Driven Structured IE from Biomedical Literature | Cavalleri et al. | CIKM 2026, 2026-11-06 | [10.1145/3799682.3840835](https://doi.org/10.1145/3799682.3840835) | OG |
| Lost in Extraction: RE-ErrorTax | Bharadwaj and Anand | CIKM 2026, Crossref created 2026-11-03 | [10.1145/3799682.3840080](https://doi.org/10.1145/3799682.3840080) | SF unread; DJ abstract |
| Where, How, and What to Ground | Yadav, Klinger, Peters | Talk 26 Oct 2026 | No DOI found; [programme](https://5thnlp4kgc-code.github.io/5nlp4kgc/pages/programme.html) #11 | OG: appears to answer the design question |
| SHACL-in-the-Loop | Snytko, Jakhar, Celebi | Talk 26 Oct 2026 | No DOI found; same programme #10 | OG |
| AutoKGen | Anagnostou and Kotis | Talk 26 Oct 2026 | No DOI found; same programme #8 | OG |
| MOIE-KG | Liu et al. | Created 2026-09-29; issue Oct 2026 | [10.1016/j.knosys.2026.117109](https://doi.org/10.1016/j.knosys.2026.117109) | OG, DJ: high priority |
| Constructing knowledge graphs by LLMs (apparent survey) | Wu et al. | Created 2026-09-08; issue Oct 2026 | [10.1016/j.knosys.2026.116963](https://doi.org/10.1016/j.knosys.2026.116963) | DJ: high priority |
| Knowledge graphs and LLMs (apparent survey) | Syeda and Reddy B. | Created 2026-07-04; issue Oct 2026 | [10.1016/j.knosys.2026.116601](https://doi.org/10.1016/j.knosys.2026.116601) | DJ: high priority |
| OntoGRC | Zhang et al. | Registered 2026-06-24; issue Aug 2026 | [10.1016/j.oregeorev.2026.107411](https://doi.org/10.1016/j.oregeorev.2026.107411) | OG |
| Schema-Constrained Electric Power KGs | Xu et al. | Sep 2026 | [10.1109/aice71085.2026.11736970](https://doi.org/10.1109/aice71085.2026.11736970) | OG |
| Missing-Person Intelligence | Castillo and Mukkamala | May 2026 | [10.1109/idss69894.2026.11691381](https://doi.org/10.1109/idss69894.2026.11691381) | OG |
| Asset Administration Shell Generation | Jensen et al. | Sep 2026 | [10.1109/etfa68377.2026.11725111](https://doi.org/10.1109/etfa68377.2026.11725111) | OG |
| Component-Aware Case Retrieval | Rauch and Weber | 2026-08-01 | [10.1007/978-3-032-33865-5_13](https://doi.org/10.1007/978-3-032-33865-5_13) | OG |
| Relation Extraction from Financial Documents | Bakir | 2026-10-26 | [10.51219/urforum.2026.nader-bakir](https://doi.org/10.51219/urforum.2026.nader-bakir) | OG: venue unassessed |
| KG Extraction on Literary Studies in Spanish | Cortes | May 2026 | [10.63317/5kjis3dy8i7k](https://doi.org/10.63317/5kjis3dy8i7k) | OG, OL |
| Efficient LLM Adaptation for Opinion KG | Yang et al. | SIGIR 2026, 2026-07-19 | [10.1145/3805712.3808450](https://doi.org/10.1145/3805712.3808450) | SF |
| RAG vs. GraphRAG | Han et al. | **PRE** 2025-02-17; KDD 2026-08-08 | [10.1145/3770855.3817575](https://doi.org/10.1145/3770855.3817575), [arXiv 2502.11371](https://arxiv.org/abs/2502.11371) | SF: summary is snippet only‡ |
| Core-based Hierarchies for GraphRAG | Not recorded | KDD 2026 | [10.1145/3770855.3818007](https://doi.org/10.1145/3770855.3818007) | SF |
| QA-GraphRAG | Not recorded | PVLDB | [10.14778/3819518.3819546](https://doi.org/10.14778/3819518.3819546) | SF |
| Seen but not assessed | Not recorded | Treat dates as unverified | [2605.05476](https://arxiv.org/abs/2605.05476), [2605.22304](https://arxiv.org/abs/2605.22304), [2606.31718](https://arxiv.org/abs/2606.31718), [2606.28351](https://arxiv.org/abs/2606.28351), [2609.29695](https://arxiv.org/abs/2609.29695), [2610.31684](https://arxiv.org/abs/2610.31684), [TH-RAG](https://aclanthology.org/2026.acl-long.1740/), [TagRAG](https://aclanthology.org/2026.findings-acl.321/), [HyperRAG-IHM](https://aclanthology.org/2026.emnlp-main.898/), [Beyond Static Retrieval](https://aclanthology.org/2026.emnlp-main.1897/), [SAM-NER](https://arxiv.org/abs/2605.03706) | SF: some abstracts partly read |
| Agentic KG Taxonomies in Data Science | Not recorded | L@S | [10.1145/3774398.3811567](https://doi.org/10.1145/3774398.3811567) | OL title only |
| Islamic ontology construction | Not recorded | Not recorded | [10.20935/acadai8357](https://doi.org/10.20935/acadai8357) | OL title only |
| Unstructured Procedural Text to KGs | Not recorded | Not recorded | [10.34190/eckm.27.2.4713](https://doi.org/10.34190/eckm.27.2.4713) | OL title only |
| China Telecom multimodal KGs | Not recorded | Not recorded | [10.1109/ICAIBD69640.2026.11637266](https://doi.org/10.1109/ICAIBD69640.2026.11637266) | OL title only |
| Taxonomy Exploration with LLM Reasoning | Not recorded | Not recorded | [10.3103/S0005105526700780](https://doi.org/10.3103/S0005105526700780) | OL title only |
| Tree KG from Vietnamese History | Not recorded | Not recorded | arXiv 2609.00763; no DOI recorded | OL title only |
| From DQI to C-DQI | Not recorded | Not recorded | arXiv 2610.27937; no DOI recorded | OL title only |
| ReLTEx; SCALE | Not recorded | Not recorded | [2608.10970](https://arxiv.org/abs/2608.10970), [2608.07254](https://arxiv.org/abs/2608.07254) | OL: extension of an existing structure |
| Making order from chaos through semantics | Not recorded | *J. Big Data* | [10.1186/s40537-026-01451-3](https://doi.org/10.1186/s40537-026-01451-3) | OL: from SQL databases, Semantic Scholar metadata only |
| Hamilton: Interactive Ontology Learning | Not recorded | PVLDB | [10.14778/3827998.3828059](https://doi.org/10.14778/3827998.3828059) | OL abstract: from relational databases, not text |
| Collective Entity Linking via Question Answering | Wei et al. | Crossref online 2026-11-05 | [10.1007/s11390-026-5822-8](https://doi.org/10.1007/s11390-026-5822-8) | MV |
| Quality-aware fusion for Concept Normalization | Zhou et al. | Issue Oct 2026 | [10.1016/j.jbi.2026.105093](https://doi.org/10.1016/j.jbi.2026.105093) | MV: SSRN version per OpenAlex |
| TELLER | Peng, Li, Decker | First posted 2026-07-29 | [arXiv 2607.28680](https://arxiv.org/abs/2607.28680) | MV |
| UZH teams: Argchestrators, Prompteam, HybridArguer | See Anthology | Jul 2026 | [2026.argmining-1.13](https://aclanthology.org/2026.argmining-1.13/), [-14](https://aclanthology.org/2026.argmining-1.14/), [-17](https://aclanthology.org/2026.argmining-1.17/) | MV: no DOI recorded |
| Hyperparameters in Tree-Based Linear Methods | Chen, Chiang, Lin | CIKM 2026 | [10.1145/3799682.3840104](https://doi.org/10.1145/3799682.3840104) | MV |
| Category-constrained prompting | Xue et al. | Issue Oct 2026 | [10.1016/j.asoc.2026.115777](https://doi.org/10.1016/j.asoc.2026.115777) | MV |
| LLM-augmented HTC for failure data | Ding et al. | Issue Oct 2026 | [10.1016/j.engappai.2026.115750](https://doi.org/10.1016/j.engappai.2026.115750) | MV |
| SIAGXC | Yin et al. | Aug 2026 | [10.1007/s10489-026-07400-0](https://doi.org/10.1007/s10489-026-07400-0) | MV |
| Overcoming data sparsity in XML | Prajapati et al. | 2026-05-31 | [10.1007/s10791-026-10158-1](https://doi.org/10.1007/s10791-026-10158-1) | MV |
| OM-2026 unread papers | Nascimento et al.; Efeoglu and Hoffmann; Rafferty; Zamazal; Charushin | 25 Oct 2026 | No DOI; PDFs: [LT6](https://om.ontologymatching.org/2026/papers/om2026-LTpaper6.pdf), [ST2](https://om.ontologymatching.org/2026/papers/om2026-STpaper2.pdf), [LT7](https://om.ontologymatching.org/2026/papers/om2026-LTpaper7.pdf), [poster](https://om.ontologymatching.org/2026/papers/om2026-poster.pdf), [ST4](https://om.ontologymatching.org/2026/papers/om2026-STpaper4.pdf) | MV |
| Crisis knowledge generation | Han et al. | Created 2026-04-22 (**PRE** record); issue Nov 2026 | [10.1016/j.ipm.2026.104808](https://doi.org/10.1016/j.ipm.2026.104808) | DJ |
| Toward trustworthy engineering IE using RAG | Shen, Zhu, Zhang | Created 2026-06-12; issue Aug 2026 | [10.1016/j.knosys.2026.116418](https://doi.org/10.1016/j.knosys.2026.116418) | DJ: ScienceDirect 403 |
| VRQA | Huang et al. | Created 2026-05-01; issue Sep 2026 | [10.1016/j.eswa.2026.132613](https://doi.org/10.1016/j.eswa.2026.132613) | DJ |
| DynKGRAG | Zhou et al. | Created 2026-05-25; issue Oct 2026 | [10.1016/j.neucom.2026.134094](https://doi.org/10.1016/j.neucom.2026.134094) | DJ: title indicates numerical and textual data |
| CMOMgen | Silva, Faria, Pesquita | Created 2026-07-08; issue Aug 2026 | [10.1016/j.websem.2026.100884](https://doi.org/10.1016/j.websem.2026.100884) | DJ |
| Sixty years of EU forest policy discourse | Aggestam et al. | Created 2026-07-31; issue Oct 2026 | [10.1016/j.forpol.2026.103880](https://doi.org/10.1016/j.forpol.2026.103880) | DJ: domain relevant |
| Dam defects ontology and KG | Not retrieved | Issue Nov 2026 | [10.1016/j.aei.2026.104987](https://doi.org/10.1016/j.aei.2026.104987) | DJ |
| Digesting massive public comments | Dong et al. | Created 2026-04-22 (**PRE** record); issue Nov 2026 | [10.1016/j.ipm.2026.104821](https://doi.org/10.1016/j.ipm.2026.104821) | DJ |
| Systematic review of KGs for academic papers | Ramírez Molina et al. | Created 2026-04-23 (**PRE** record); issue Nov 2026 | [10.1016/j.ipm.2026.104842](https://doi.org/10.1016/j.ipm.2026.104842) | DJ |
| Cross-modal extraction for KG construction | Xu et al. | Created 2026-04-24 (**PRE** record); issue Jul 2026 | [10.1016/j.datak.2026.102609](https://doi.org/10.1016/j.datak.2026.102609) | DJ |
| MAKES-QA | Borrelli et al. | Created 2026-03-21 (**PRE** record); issue Jul 2026 | [10.1016/j.eswa.2026.132156](https://doi.org/10.1016/j.eswa.2026.132156) | DJ |
| OntoLLM | Venkatesh and Krishna P | Created 2026-02-04 (**PRE** record); issue Jun 2026 | [10.1016/j.eswa.2026.131505](https://doi.org/10.1016/j.eswa.2026.131505) | DJ |
| SEGA low-resource multilingual NER | Tulajiang et al. | Created 2026-01-22 (**PRE** record); issue Jun 2026 | [10.1016/j.ipm.2026.104627](https://doi.org/10.1016/j.ipm.2026.104627) | DJ |
| NER for Chinese aviation accident reports | Not retrieved | Created 2026-02-03 (**PRE** record); issue Jul 2026 | [10.1016/j.ipm.2026.104663](https://doi.org/10.1016/j.ipm.2026.104663) | DJ |
| Survey on KG-augmented LLM reasoning | Wu et al. | Online 2026-09-20; issue Dec 2026 | [10.1007/s10115-026-02880-5](https://doi.org/10.1007/s10115-026-02880-5) | DJ: closed per OpenAlex |
| Entity semantic graph for document-level event extraction | Not retrieved | Online 2026-06-02 | [10.1007/s10115-026-02805-2](https://doi.org/10.1007/s10115-026-02805-2) | DJ |
| Frugal Encoder-Decoder Models benchmark | Ringwald et al. | Online 2026-06-03 | [10.1177/22104968261452987](https://doi.org/10.1177/22104968261452987) | DJ |
| SAFAR-Onto | Hendrik et al. | 2026-10-22 | [10.1177/15705838261497585](https://doi.org/10.1177/15705838261497585) | DJ |
| Document-level relation extraction in *Evaluation Review* | Zhang et al. | 2026-09-18 | [10.1177/0193841x261491411](https://doi.org/10.1177/0193841x261491411) | DJ: looks out of scope for that journal |
| LLM agents for web automation and schema-guided extraction | Not retrieved | 2026-10-28 | [10.1007/s41060-026-01335-y](https://doi.org/10.1007/s41060-026-01335-y) | DJ |
| ESG insight | Zhou et al. | Online 2026-05-28 | [10.1007/s13042-026-03146-w](https://doi.org/10.1007/s13042-026-03146-w) | DJ |
| Climate-Risk Signals from ESG Documents | Theang and Thuon | Online 2026-10-01 | [10.1007/978-3-032-39385-2_31](https://doi.org/10.1007/978-3-032-39385-2_31), green copy at [HAL](https://hal.science/hal-05667766v1) | DJ |
| Corporate audit decision support | Not retrieved | 2026-05-19 | [10.1038/s41598-026-52652-y](https://doi.org/10.1038/s41598-026-52652-y) | DJ |
| MechaScreener | Not retrieved | 2026-10-15 | [10.1017/rsm.2026.10124](https://doi.org/10.1017/rsm.2026.10124) | DJ: screening, not extraction |
| Prompting strategies for screening | Not retrieved | 2026-10-19 | [10.1017/rsm.2026.10110](https://doi.org/10.1017/rsm.2026.10110) | DJ: screening, not extraction |
| Prompt engineering for screening | Not retrieved | 2026-06-22 | [10.1017/rsm.2026.10104](https://doi.org/10.1017/rsm.2026.10104) | DJ: screening, not extraction |
| The transformation of evaluation practice | Nielsen, Bruce, Gandhi | 2026-08-18 | [10.4324/9781003799139-1](https://doi.org/10.4324/9781003799139-1) | DJ: book introduction |
| GenAI and the evolution of evidence in evaluation | Bruce, Gandhi, Nielsen | 2026-08-18 | [10.4324/9781003799139-34](https://doi.org/10.4324/9781003799139-34) | DJ: book closing chapter |
| Semantic retrieval and chain-of-thought for SNOMED CT | López-Úbeda et al. | Crossref record 10 Feb 2026 (**PRE**); May 2026 issue | [10.1016/j.ijmedinf.2026.106340](https://doi.org/10.1016/j.ijmedinf.2026.106340) | MV |
| Unsupervised Multilingual Historical Entity Linking | Santini et al. | EACL 2026, Mar 2026 (**PRE**) | [Anthology](https://aclanthology.org/2026.eacl-long.184/) | MV |
| Comprehensive survey on concept normalization | Chen et al. | OpenAlex 2 Mar 2026 (**PRE**); Jun 2026 issue | [10.1016/j.jbi.2026.105005](https://doi.org/10.1016/j.jbi.2026.105005) | MV |
| HTC with LLM-Refined Taxonomies | Golde et al. | EACL 2026, Mar 2026 (**PRE**) | [Anthology](https://aclanthology.org/2026.eacl-long.10/) | MV |
| CrisisSense-LLM | Yin et al. | Preprint 2024 per arXiv ID (**PRE**); Oct 2026 issue | [10.1016/j.aei.2026.104836](https://doi.org/10.1016/j.aei.2026.104836) | MV |
| SDG-POD | Cadeddu et al. | **PRE** 24 Sep 2025; v4 3 Jul 2026 | [arXiv 2509.19833](https://arxiv.org/abs/2509.19833) | MV |
| Tracking Aid Contribution Towards SDGs | Park et al. | **PRE** 21 May 2025 | [arXiv 2505.15223](https://arxiv.org/abs/2505.15223) | MV: nearest SDG work on aid text |
| LLMs as Oracles for Ontology Alignment (LogMapLLM) | Lushnei et al. | arXiv ID suggests Aug 2025 (**PRE**); EACL 2026, Mar 2026 | [Anthology](https://aclanthology.org/2026.eacl-long.110/), [arXiv 2508.08500](https://arxiv.org/abs/2508.08500) | MV: OAEI 2025 ranking is snippet only‡ |
| MFG-AgentGraph | Not recorded | ISWC 2026, per a GitHub issue | No identifier | OG: not verified as a paper |
| TextMineX | Zhou et al. (MV) | **PRE**; Findings of EACL 2026, Mar 2026 | [Anthology](https://aclanthology.org/2026.findings-eacl.130/), [arXiv 2509.15098](https://arxiv.org/abs/2509.15098) | OL, DJ snippet‡; MV title only. Closest prior work on ontology plus humanitarian reports |
| Disaster Storylines and KGs from Global News | Ronco et al. | **PRE**; *Scientific Data*, 2026-03-17 | [10.1038/s41597-026-07036-2](https://doi.org/10.1038/s41597-026-07036-2) | DJ snippet‡: main precursor to Decostanzi |
| Snippet-only matching leads | Not recorded | **PRE**‡ | [arXiv 2511.22612](https://arxiv.org/html/2511.22612v1); GenOM, no identifier | MV: not verified |

## The notes disagree on read status, identifiers and a few numbers

I have not resolved these silently. The catalogue shows both values where they differ, and "my view" below is labelled as such.

**A general pattern explains most read-status differences.** The DJ researcher read many papers at abstract level by design, while OL, OG and SF read the same papers further. These are differences in depth, not contradictions, and they are marked "Part / Abstract" above. The table covers the cases that matter more.

| Paper | What the notes say | Status in this report |
|---|---|---|
| AgentsKG | OL read an abstract through Semantic Scholar. SF says Crossref holds no abstract and records metadata only. | Both shown. Treat as unread. |
| LightOL | OL says no abstract was available in Semantic Scholar (metadata only). DJ records `abstract only` and describes content. | Both shown. |
| Less Is More, OntoForge, Self-Evolving Ontology, RE-ErrorTax | OG and SF say metadata only because ACM returned 403. DJ read abstracts via OpenAlex or Semantic Scholar. | Listed as unread, with DJ's detail noted. |
| Sun; Peng et al. | OG lists them under metadata only. DJ read abstracts and gives figures. | Both shown. |
| DSE-RE | OL: abstract, Springer login redirect. OG: open access, read through a fetch-tool summary of the first 100,000 of 126,990 characters, yet filed in its abstract-only tier. | Part† / Abstract. |
| DSE-RE effect size | OL reports F1 rising about 3% on FOBIE and 6.8% on HSA. OG reports 0.741 to 0.771 and 0.740 to 0.781 with GPT-4o†. The HSA figures do not obviously agree. | Not reconciled. Check the paper. |
| Decostanzi et al. | OG: partial, v1 date not checked. DJ: whole text via fetch tool, first posted 2026-09-01. OG calls it a fixed schema; DJ calls its graphs free-form. | Date from DJ. Track assignment is open: fixed storyline fields, free-form graph nodes. |
| Belfadel et al. | OG: full bar about 2,900 characters, "no venue found". OL and DJ: part, and IJCKG 2026 per arXiv journal reference. | Both shown. |
| Loconte et al. | OG: first posted "not checked". OL and SF: 2026-05-27. | 2026-05-27. |
| OntoLearner | OL: first posted 2026-07-02. DJ: Research Square 2026-06-25, then arXiv 2026-07-02. | Both dates shown. The earlier is the first public posting. |
| LELA | MV: arXiv 2605.26956, first posted 26 May 2026. DJ: OpenAlex links arXiv 2601.05192 as a green copy, by ID January 2026, not checked. | Both flagged. If DJ's link holds, this is pre-window. |
| SciKGExtract | OL gives arXiv 2610.06322. OG gives Research Square DOI posted 28 Oct 2026. | Both listed. Not established whether they are the same version. |
| TextMineX | OL and DJ: snippet only, via arXiv 2509.15098. MV: title only, via the Anthology. Only DJ's snippet carries the 44.2% and 22.5% figures‡. | Unread. Figures not relied on. |
| SPARROW DOI | OL treats it as a Crossref match. DJ says it came from Semantic Scholar and was not checked in Crossref. | DOI shown; verify. |
| Neubürger et al. DOI | CEUR lists 10.83095/ceurws_04293_059, and Crossref returned 404. | Unconfirmed, so the CEUR link is used. |
| ELLMO, HybridAI-Squared, VUL337-OL | OL says verified at Crossref. DJ says not individually checked there. | Treat OL's as the stronger check. |
| AutoSchemaKG scale | SF reports "over 50 million documents". OL reports 9.599M, 7.918M and 35.040M text chunks on three corpora. | Unit not reconciled. Do not quote as either without reading. |
| LLMs4OL document counts | OL: 4,303 train and 2,018 test Flagship; 2,774 and 1,799 Reuse. DJ: 7,172 training and 4,036 test. | Not reconciled. May include other tasks. |
| Search services | See the coverage section, where DBLP and Semantic Scholar reports differ. | Reported by researcher. |

**Some papers contradict themselves.** These come from the notes and have not been resolved. SCION's token figures of 1,200 and 800 sit beside 2,450 and 2,379 for a representative run (OL). Anchor's Table I gives schema sizes of up to 997, 77 and 60 while its evaluation text gives 419, 109 and 75 classes (OG). Neubürger reports relation coverage of 104.5% in one condition, above 100% (OG). SocraticKG's table gives 84.21 and its prose 84.38 (SF). WildGraphBench says 1,100 questions in its abstract and 1,197 in its body (SF). The IEG chapter gives test-set precision of both 0.60 and 0.67 (DJ). On the French legal paper, "about 10% per stratum" sits oddly beside 1,389 sampled of 6,370 articles (OL; that observation is mine, so check how the strata are defined). The abstract of Profiling Hallucinations claims constraining spans worsens hallucination, but the Opus rows read show 13.61% and 13.57% (MV).

**Findings also conflict, and readers should weigh these themselves.** After-the-fact repair failed in LLMs4OL teams' tests but helped elsewhere: Anchor's capped retries reached 5.2% non-compliance, and OAK+MEND reached 96.8% validity. SDG tagging over-assigns in Charalampous et al. and tightens in Wang et al., with different text and candidate pools, and the MV researcher did not establish why. A 2B multi-pass model beat 31B in Neubürger's three-run study, while lightweight models did far worse in Fadda et al. Narrowing candidates helps cost, but a full taxonomy reduced errors in Thakaa, and a full list with no real "none" path forces wrong picks in Fallatah.

**Numbers that passed through a summarising fetch tool (†)** include the Rezayat, Peeters, Calautti, Kahlawi and Huang figures, all in OG. They also include Kähler, Select Don't Train and LongBEL, where MV re-requested only the rows marked as quoted, plus Yoon, Kasliwal, Wang (post-training), LLM-INSTRUCT and the OAEI Anatomy page in MV. In DJ they include Turobov and Coyle, Decostanzi, Andreozzi, Cole, Nicolau, Scope3Trace and MGAL, whose "full text" readings may also reflect the researcher's own text extraction. **Numbers resting on search snippets (‡)** are TextMineX's 44.2% and 22.5%, the Disaster Storylines description, the RAG vs. GraphRAG summary, the OntoKG and OntoLogX descriptions, LogMapLLM's OAEI 2025 ranking, and details of the Routledge book's publication and structure. A snippet claiming Executable Schema Contracts recovers extra entity types could not be found in the paper's text, so it is excluded.

## Several bibliographic services failed, so "not found" never means "does not exist"

Researchers worked in the same session and saw different failures, which is itself a warning about reproducing this search.

| Service | What happened | Recorded by |
|---|---|---|
| OpenAlex search | HTTP 429 (OL); 503 or 429 (OG, SF); "Anonymous search is paused" (MV, DJ). DOI lookups worked (MV, DJ). | All |
| Semantic Scholar | Used successfully with date filters (OL) and 20 boolean queries (DJ); 429 for OG, SF and MV. | All |
| DBLP | Bot check for OL, OG, SF and MV; DJ says it was reachable but not queried. It contributed nothing. | All |
| arXiv API | 429 for most. SF crawled monthly listings for cs.CL, cs.IR, cs.DB and cs.AI (47,999 IDs). OG's 32-query sweep never completed. DJ's search pages served 10 of 55 queries. MV's roughly 40 queries eventually completed, capped at 50 results each. | OG, SF, DJ, MV |
| Publisher pages | ACM, Sage, MDPI, Taylor & Francis, ScienceDirect refused automated access; Springer redirected to login. | OL, OG, SF, MV, DJ |
| Google Scholar | Not used. | OL |

**Some venues and repositories were not swept.** These are the ISWC 2026 research track, the ESWC 2026 main track, the Journal of Web Semantics, TKDE and KDD 2026 proceedings (OG), plus NLP4KGC proceedings, which were not found, and SIGIR and CIKM 2026, which were covered only through arXiv and Crossref (MV). COLING and OpenReview were not systematically checked, and arXiv categories beyond the four crawled were not crawled, including cs.LG, cs.CR and cs.SE (SF). World Bank Open Knowledge, UNEG, 3ie and evalforward were not swept beyond search hits (DJ). OL's statement that it found no ISWC, EKAW or ESWC main-track paper applies to ontology learning only: OG catalogues Peeters at EKAW 2026, and MV lists Polat et al. at ISWC 2026, so proceedings should be browsed directly.

**Discovery was biased towards English and title vocabulary.** Filtering was on titles, so papers using other terms were missed (SF, MV), and survey searches used English terms only (DJ). The evaluation-journal scan by ISSN found almost no relevant AI titles, but it matches titles only, and *Data & Policy* and *IDS Bulletin* returned no records even though *Data & Policy* papers exist (DJ).

**Four "not found" statements matter most to you, and all carry these limits.** They are no ontology or graph from UN evaluation reports (OL, OG, SF, DJ), no mapping of UN evaluation reports to UNBIS or DAC sectors with LLMs (MV), no comparison of extraction on the same UN content in English, French and Spanish (DJ), and no chunk-then-aggregate test against a document-level gold standard (MV).

## The researchers' estimates and pipelines are untested hypotheses

**Everything in this section is a hypothesis or the researcher's own arithmetic, not a finding.** No paper reports these figures for your corpus, and they assume your chunks resemble each paper's inputs. The corpus figures of about 13,000 reports, 31 agencies and about 1.9 million chunks come from the SF note.

| Estimate (untested) | Researcher's reasoning | Note |
|---|---|---|
| About 2,830 GPU hours for AutoSchemaKG-style extraction | 1.9M chunks at the ATLAS-Wiki rate, assuming 1,024-token chunks | OL |
| About 4.9M LLM calls for TRACE-KG-style processing | 876 calls per 339 chunks, or about 2.6 per chunk | OL |
| About 14.6 hours for a TaxCE-style local pass on one GPU | 36.2 chunks per second, if chunks resemble its documents | OL |
| About $95 to $9,500 for one input pass | RAGU's rates, an assumed 400 tokens per chunk, about 760 million tokens; before prompts, output and retries | SF |
| 2.3-3.7 times direct extraction for multi-stage pipelines | SocraticKG's token overhead | SF |
| About 22 days against about 5.5 hours of sequential inference | Kähler's 998 ms and 10.5 ms per record; hardware differs by method | MV |

**Suggested pipelines, all untested in combination.** OL proposes a small-model pass per chunk or on a stratified sample, type-signature-grouped embedding clustering, LLM merges inside clusters, a frozen schema, and an admission score using support across agencies rather than raw frequency. It notes that each step has support in a different paper but no paper has tested the combination. OG proposes an open pass on a stratified sample of agencies and languages, then closed extraction with novel-type flagging and logging of every rejected triple. DJ proposes a dual track, with discovery constrained to a small extraction target such as intervention, outcome, direction and strength. MV proposes retrieving candidates per vocabulary, always allowing "no match", and testing both directions of "broader than" when aligning agency lists.

**Suggested explanations, also untested.** Post-hoc repair may appear to fail in LLMs4OL because it scores against a hidden gold vocabulary, not corpus compactness (OL). Gating late may beat gating early because a chunk failing an early stage loses everything downstream (OG). "Right string, wrong context" may be the main error in evaluation reports, because they cite past programmes and other countries at length (SF). Museum-text failure modes may predict evaluation findings better than Text2KGBench (OG). Cross-lingual aliases from Wikidata may transfer to UNBIS or AGROVOC labels (MV). Prompted abstention plus a retrieval score threshold may beat either alone (MV). LLM oracles may help most where a good matcher already supplies candidates (MV). Scope3Trace's table gain may overstate gains on your corpus, since its targets are narrow numeric fields (DJ). A supplied seed of types such as programmes, interventions, beneficiary groups, findings and recommendations would make the discovery track seeded rather than ontology-free, which is a choice for you to make (OL).

## Conclusion

The practical surprise is how much of the useful evidence sits outside the main conferences. The closest document types appear in workshop papers, a Routledge book chapter and the CEUR proceedings, and many of the most relevant papers are behind ACM and Elsevier walls, so the unread list is part of the answer rather than a footnote. I would obtain AutoSchema, Rezayat et al., Peeters et al., Yadav et al. and the two *Knowledge-Based Systems* surveys first, because each speaks directly to a design choice the notes could not settle from what was read.

The evidence shifts the question from "which extraction method wins" to "what will you measure on your own chunks". Scoring rules alone move one prediction set between 0.16 and 0.70 F1, the main benchmark leaves non-taxonomic relations largely unmeasured, and supplied-vocabulary scores come from clean synthetic text. Since no shared metric compares these papers and none used evaluation reports, a small hand-labelled set across agencies and languages is the first thing the literature cannot give you.

In plain terms: there is plenty of new work worth reading, but nothing tested on UN evaluation reports, and many of the best-looking papers are still unread, so please obtain the priority items listed in the unread section. All cost estimates and suggested pipelines here are untested ideas, not results. The search was also incomplete because several services failed, so a missing paper may simply be one the search could not reach.
