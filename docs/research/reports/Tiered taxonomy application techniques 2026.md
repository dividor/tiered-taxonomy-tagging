# Test the tiers before showcasing them

Build the showcase around five techniques: a Kähler-style bake-off of trained flat classifiers against LLM pipelines with expert graded judgement, scored tier by tier; a retrieve-prune-rerank funnel over all leaves with generated label cards (MATCH plus SLKR); a tree walk with recovery moves as the top-down contrast (ToT-ICD, Constraint-Aware Search, Graph-Constrained Policy); a per-label evidence check (CTI2TTP); and a chunk-to-report rollup test using certainty-weighted log-odds (TEF). **Kähler et al. does not use the vocabulary's tiers and scores labels as one flat set**, so it supports the comparison of trained models with LLMs and the graded evaluation, not tiering as such. The tiering evidence comes from other papers and it is mixed: hard two-stage routing lost to flat in the one plain head-to-head (0.332 against 0.3659, with 30.2% of items stuck in the wrong branch), while a trained, multi-branch tree walk gained on macro-F1 and only tied on micro-F1 at what appears to be equal training data. The new notes correct one earlier finding directly: direct tests of chunk-then-aggregate were found, but none at report length or with a large taxonomy, and that remaining gap is the blog's best original contribution. No technique here was found tested on UN evaluation reports, so every application to this corpus below is an untested hypothesis or estimate, and search coverage was incomplete, so "not found" means only that these searches did not surface it.

**Key to tags.** *Read*: **Full** is the whole body. **Part** is named sections or tables only. **Abs** is abstract only, and **Abs-cut** means the record was cut off before the abstract ended. **Meta** is title and metadata only, **Sec** is known only through another paper, and **Snip** is seen only in a search result. *Numbers*: **X** came from a researcher's own PDF or HTML text extraction. **X?** came from extracted text where the researcher flagged doubt about a column or a broken sentence. **A** came from an abstract record. **F** passed through a summarising fetch tool. **S** rests on a search snippet. *Dates*: first-posted and published dates are always separate. **PRE** means first posted before about May 2026 and published in the window. **OLD** means older than the window with no in-window publication recorded. *Notes*: **H**, **C**, **E** and **B** are `hierarchical_classification_methods.md`, `cascades_and_aggregation.md`, `evaluation_trust_and_gaps.md` and `benchmarks_on_our_test_sets.md` in `research_notes/Tiered taxonomy application techniques 2026/`. **M** and **D** are `mapping_to_supplied_vocabularies.md` and `domain_and_journal_sweep.md` in `research_notes/Entity and ontology extraction papers 2026/`. All paper numbers are the authors' own and nobody re-ran anything. Strength ratings are my judgement of the notes, not a published score.

## Five techniques make the top-down question visible

The selection below is my recommendation, and it draws on three researchers whose own picks differ (see the disagreements section). Each "published evidence" passage reports other corpora. None of it is a result on UN chunks.

### 1. Run Kähler's bake-off, then add the tier columns it lacks

**What it would demonstrate.** "Which method wins" depends on the ruler. Agreement with past human tags favours the trained model, while graded usefulness favours the LLM pipeline. Adding tier columns would then show where in the tree each method fails.

**What Kähler does and does not support.** The paper treats all 204,056 GND entities alike, which the authors list as a limitation. The "tree" in XR-Transformer is a tree of clustered labels built for speed, not the vocabulary's hierarchy. ZestXML can add a label hierarchy to its features, but the paper does not say whether the GND hierarchy was fed in. The graded evaluation asks people about usefulness and does not use the tree at all. So the paper says nothing directly about top-down against flat on a supplied hierarchy ([Kähler et al.](https://arxiv.org/abs/2607.14882), Full in H, Part in E, X).

**How it works.** The trained route, XR-Transformer, (1) groups labels into a tree whose inner nodes act as meta-labels, (2) fine-tunes a pretrained encoder level by level on those meta-label tasks, (3) concatenates the tuned embedding with sparse TF-IDF features, and (4) trains linear classifiers on that. It uses no label text. The untrained route, LLM-Ensemble, (1) has several open models generate free keywords from prompts with 8 to 12 fixed examples, without showing the vocabulary, (2) maps each keyword to the nearest vocabulary term with BGE-M3 and hybrid vector and BM25 search, (3) sums similarities across model-and-prompt combinations into an ensemble score, (4) has one more LLM rate each term from 0 to 10, and (5) sets the final score to 0.3 × ensemble + 0.7 × relevance. KI-FSPrompt, the cheaper version, retrieves similar labelled documents, builds a few-shot prompt from them, maps the output and ranks it with an LLM, in two LLM calls ([KI-FSPrompt](https://aclanthology.org/2025.konvens-2.42/), OLD, Part). The grading (1) cuts the test set into six slices with the same subject mix, (2) gives each of six methods its own slice of 524 to 586 documents, (3) has professional subject experts rate the top 5 suggestions as false, slightly useful, useful or very useful, where useful means a good search term, and (4) scores generalised precision and recall, in which a suggestion matching gold keeps full credit.

| Method | Titles F1* | Full text F1* | Full text cps-F1* (long tail) | Titles g-F1@5 | Full text g-F1@5 | ms per title |
|---|---|---|---|---|---|---|
| XR-Transformer | 0.467 | 0.398 | 0.339 | 0.530 | 0.512 | 10.5 |
| Omikuji | 0.417 | 0.343 | 0.277 | 0.499 | 0.525 | <1 |
| LLM-Ensemble | 0.382 | 0.364 | 0.362 | 0.576 | 0.601 | 998 |
| KI-FSPrompt | 0.453 | not run | not run | not rated | not rated | 271 |

**Published evidence and its limits.** The test set is 4,651 documents, with 951,104 titles or 167,281 full texts for training (X). Graded scores for different methods come from different documents, so they are not paired comparisons. A keyword search of the text found no inter-rater figure, F1* is at an optimally tuned operating point while g-F1 is fixed at k=5, and gold was created after experts saw machine suggestions (E). The timings are **for titles only**, and the authors say full-text times are "much higher". The authors judge LLM-Ensemble's speed infeasible for daily library use and recommend building productive systems around Annif, so the paper does not conclude that LLMs win (H). The tier columns come from elsewhere. LCSHBench scores every metric exact and at concept level, and finds the fine-tuned embedder leads exact recall@200 (0.659 against 0.623) while text-embedding-3-large leads concept recall@200 (0.762 against 0.732) ([LCSHBench](https://arxiv.org/abs/2606.04382), Part, X). Popovic et al. annotate at the finest tier, collapse upwards and report κ at each level: Gemma 3-27B leads at the top (0.689) and Claude Sonnet 4.5 at sub-label (0.469), beside human pairwise averages of 0.509 and 0.349, on only 100 emails ([Popovic et al.](https://aclanthology.org/2026.woah-1.26/), Part, X). Hauss et al. back the flat baseline: flattening a human hierarchy and running extreme classifiers is often competitive, especially on ranking measures ([Hauss et al.](https://doi.org/10.1145/3820755.3832808), Abs, no numbers).

**Strength.** Strong for the trained-against-LLM comparison, on German library data. Moderate for tier-level scoring, from small or differently shaped datasets. None for tiering from Kähler itself. Two metric traps should go in the design. AraGenre's Hierarchical Macro F1 is not path-aware by the organisers' own account ([AraGenre overview](https://aclanthology.org/2026.arabicnlp-sharedtasks.27/), Part). MAX-EVAL-11's printed 0.433 needs a Points Earned term of about 1.74 by the researcher's arithmetic, which suggests a near-miss term can reward over-generation ([MAX-EVAL-11](https://aclanthology.org/2026.bionlp-1.23/), Part in E, Abs in H, X?).

### 2. Retrieve across all leaves, prune by tier, then rerank with label cards

**What it would demonstrate.** A readable funnel from the full taxonomy to a few labels, with tiers used for pruning and roll-up rather than for a hard first decision, and every label keeping its supporting text. Label cards show how little offline work can raise a small reader's score, and they suit a taxonomy with bare names.

**How it works.** MATCH (1) embeds Concept names and explanations once with SPECTER2, (2) splits each article into eight views and matches short pieces to names and long pieces to explanations, with thresholds of 0.87 to 0.94 and top-k of 30 to 50, (3) merges by Concept using `s_final = s_best (1 + 0.1 log h)` and keeps the top 250, (4) prunes with no LLM, dropping single-hit Concepts in rare Fields and requiring the Subfield to rank in the top 60%, then merges near-duplicates, and (5) reranks with one GPT-5.4 mini step or three steps of self-hosted Qwen3 32B. JSON is constrained, returned Concepts must belong to the candidate set, invalid output is retried up to three times, and at most ten Concepts survive, each with its full path and supporting text. Tiers follow by rolling up from the leaf path. SLKR (1) has a flagship LLM write, offline and from the label list alone, a definition, retrieval keywords and boundary rules for confusable pairs for each label, (2) retrieves candidates by hybrid search with K = 15, (3) selects the relevant boundary rules, and (4) has a small LLM classify in one call.

**Published evidence.** MATCH covers five tiers with 113,892 Concepts. Blind review by 22 experts on 110 articles gave P@5 of 88.7 (self-hosted) and 88.9 (API), with no significant difference between them, and it is deployed on over two million articles at an estimated 1,900 articles per hour ([MATCH](https://arxiv.org/abs/2610.25948), Part, X). Its recall counts only Concepts experts listed as missing, so it is likely generous. Input is title, abstract and keywords only, and the authors evaluate their own system. SLKR on WOS-143 with Qwen3.5-4B scores 57.10 on label only, 57.37 with a definition, 58.67 with one example per label and 60.17 in full. On the smaller Ministral-3-3B it goes from 29.04 to 55.83, and first-stage recall at K = 15 is 95.9 to 98.3% ([SLKR](https://arxiv.org/abs/2610.16182), Part, X). On BANKING77 the flagship used directly is more accurate (86.01 against 77.45) at $4.111 against $0.0596 per 1,000 online, plus a $3.4 build. From Confusion to Clarity adds a measured upgrade on a three-level, 351-label taxonomy: adding the labels a model is known to confuse reaches 98.1% recall at about 18 candidates, where plain retrieval needs K=50, but raises input tokens per query from about 1,200-1,400 to 7,100-10,100 ([Confusion to Clarity](https://aclanthology.org/2026.emnlp-industry.125/), Part, X?).

**Strength.** Moderate to strong for MATCH, the only pattern shown at a scale near this corpus, though a preprint that evaluates itself. Moderate for label cards, which were tested on single-label tasks of 38 to 150 labels, with the smallest gain on the larger model. Narrowing is contested: in the UN resolutions task, tag micro-F1 ran 0.226 at 20 candidates, 0.396 at 40 with caps per dimension and 0.459 with no pre-filter, but those are different teams' systems ([UZH overview](https://aclanthology.org/2026.argmining-1.10/), Part, X). Against that, a climate-narrative system reached its best result by narrowing before the LLM ([Foroutan et al.](https://aclanthology.org/2026.nslp-1.22/), Abs, A). So candidate recall needs to be its own chart, and always-on small-LLM reranking needs a control, because it fell below plain retrieval in one test (76.13% against 80.71%) ([Sun and Sang](https://aclanthology.org/2026.emnlp-industry.257/), Abs, A).

### 3. Walk the tree with recovery moves, as the top-down contrast

**What it would demonstrate.** Why strict descent fails and which repairs help, through a trace a viewer can follow, plus a tier-by-tier decline chart. It answers the top-down question directly rather than assuming the answer.

**How it works.** Four moves are published. *Start from an anchor*: search all nodes globally, pick an entry point, then search locally downwards ([Constraint-Aware Search](https://arxiv.org/abs/2607.10588), Part, X). *Keep several branches open*: select a subset of children at each level, expand all pending nodes together from an agenda, allow an empty selection, and reject invalid IDs, so every output is a valid path even across branches ([Graph-Constrained Policy](https://arxiv.org/abs/2607.11954), Part in H, Abs in C, X). *Look ahead*: within the current subtree, vector search takes the top 40 nodes, reports which direct children lead to the best leaves and shows the top 3 leaves. *Backtrack*: `validate_current_node` checks the path against the text and `backtry_path` moves back up ([ToT-ICD](https://aclanthology.org/2026.findings-acl.191/), Part, X).

| Comparison | Result | Setting and limits |
|---|---|---|
| Hard two-stage against flat ([Tashkees-AI](https://aclanthology.org/2026.abjadnlp-1.20/), OLD, just outside the window, Part, X) | 0.332 against 0.3659; stage-1 F1 0.654; 30.2% misrouted and cannot recover; predicts 68 of 82 categories against 79 of 82 | Arabic medical questions, trained encoders, ranked 26th, single author, low absolute scores |
| Two-turn top-down against flat prompt ([XplaiNLP](https://aclanthology.org/2026.nslp-1.29/), Part, X) | Macro 43.13 against 42.85; **micro 58.01 against 74.51**; 700 s against 187 s | Qwen3-8B, single-run scores. A tree-shaped single pass with a reasoning model reached macro 62.48 and micro 84.42 at 2,438 s, so model and shape are confounded |
| Trained tree walk against flat ([Graph-Constrained Policy](https://arxiv.org/abs/2607.11954), Part, X) | 5,000-note policy 0.5266 / 0.2562 against LAAT 0.4829 / 0.0988 (micro / macro); 2,000-note policy 0.4775 / 0.2172 | 15,761 leaves, one 4B model, one hospital. The results text says the 2,000-note policy "matches" LAAT, so at what appears to be equal data micro is a tie and the gain is macro. The baseline budget is not in a table, so check it before repeating. Zero-shot it scores 0.0012 micro |
| Look-ahead agent against retrieve-rerank ([ToT-ICD](https://aclanthology.org/2026.findings-acl.191/), Part, X) | Full code 43.75 against 40.00 at 18 steps and 408K tokens against 2 steps and 32K | 560 Chinese records, DeepSeek-V3.2, single runs. On Qwen3-30B the plain agent falls to 18.43 against 30.36 for retrieve-rerank and 31.43 with look-ahead |
| Anchor against root start ([Constraint-Aware Search](https://arxiv.org/abs/2607.10588), Part, X) | HS-Hard 50.54% to 58.29%; path length 2.89 to 1.55 | Mean of three runs, 270 to 370 instances per dataset. Against RAG on Dual-Use the gain is very small: 78.04 against 78.64 |
| Leaf, whole path, top-down ([Yoshimura and Kashima](https://arxiv.org/abs/2508.04219), OLD, Part, X) | WoS best is leaf only, ACC2 0.440 against top-down 0.434. Amazon best is whole path, ACC3 0.532 against top-down 0.490 and leaf 0.392 | gpt-4o-mini. The winner reverses between datasets |

**Published evidence on tier decline.** ToT-ICD's look-ahead agent falls from chapter 81.61 through 65.36 and 51.07 to 43.75 at full code, and retrieve-rerank falls from 78.75 to 40.00, so roughly one in five records is already lost at tier 1 for every method. Constraint-Aware Search on HS-Hard falls 84.77, 72.97, 58.29. Canadian HTS codes on Gemini-3.1-Pro fall from 74.31% chapter accuracy to 47.92% at suffix ([Nguyen et al.](https://arxiv.org/abs/2606.16987), Part, X). MMM-Bench shows the decline need not be smooth: Claude-Opus-4.6 scores 95.11, 83.77, 74.07, 71.57 and 86.67 across five levels, with a sparse L5 ([MMM-Bench](https://arxiv.org/abs/2605.10550), Part, X). ToT-ICD's navigation or drift errors are only 9.8% of its failures against 45.4% for initial judgement errors, which on my reading suggests a walk's weak point is often the first read and not the descent. One logistics study points the other way on small trees: Qwen3-Max scores 91.15 at flat 17-way and 92.92 at hierarchical 17-way ([He et al.](https://aclanthology.org/2026.emnlp-industry.121/), PRE, Part, X).

**Strength.** Mixed. The direction of the effect depends on training, on how strict the descent is and on the model. Reports of what share of final errors began at each tier are rare, with ToT-ICD's failure table and Tashkees-AI's 30.2% the closest (H). Beam search with an LLM over a supplied taxonomy was not found in a window paper that reports a comparison. On cost, a tree walk on every chunk looks unaffordable, so it belongs on a sample or on uncertain cases, and that selective use is a hypothesis.

### 4. Check every label against a located span and the label's definition

**What it would demonstrate.** How many labels disappear when a model must show its evidence, which makes over-assignment visible and gives a reviewer something to read.

**How it works.** CTI2TTP (1) splits the report into atomic behaviours, (2) proposes candidate labels broadly for each, (3) locates a text window for each behaviour with a deterministic step and not the LLM, and (4) keeps only candidates supported by both that window and the official label definition.

**Published evidence.** On a supplied taxonomy (MITRE ATT&CK) in long reports, removing verification drops precision from 82.15% to 55.55% and F1 by 23.0 points, removing span location costs 8.9 points of precision and 18.3 of recall, and removing decomposition drops recall from 95.50% to 76.30% ([CTI2TTP](https://arxiv.org/abs/2605.25836), Part, X). POINTERS, on UN resolutions, requires a quoted phrase per tag and placed first at 0.459, but the quote and the absence of pre-filtering are confounded, so it does not show that quoting raises accuracy ([POINTERS](https://aclanthology.org/2026.argmining-1.16/), Part). Span-centric medical coding raises micro-F1 from 36.8 to 43.9 with evidence reasoning and to 48.9 with spans, but those gains involve training on evidence annotations ([Zhang et al.](https://arxiv.org/abs/2603.15270), OLD, Part, X). A right label does not guarantee full evidence: eight models reach 97.3 to 99.4% concept recognition yet miss at least one target fact in 26.6 to 41.9% of correctly recognised groups ([MULTIFINGROUND](https://arxiv.org/abs/2610.19432), Abs, A). If an LLM judge is used to audit, hide the system's decision from it: discrimination collapses to ROC-AUC 0.12 to 0.87 when the judge sees the label, and a label-free protocol recovers 0.93 to 1.00, on entity alignment ([Ravideshik and Kejriwal](https://aclanthology.org/2026.emnlp-main.1717/), Abs, A).

**Strength.** Moderate: a large, clean ablation, but one domain and the authors' own benchmark. A paper isolating the effect of requiring quotes with everything else held fixed was not found, and neither was a user study of quoted evidence on taxonomy labels. A project constraint applies too: report text must not enter the repository and notebook outputs must stay small, so any quote showpiece has to display evidence without committing it, or use short agreed excerpts (E).

### 5. Test chunk-to-report rollup against report-level human labels

**What it would demonstrate.** Whether rolling chunk labels up to a report works, which published rule to trust, and what it costs. **This is the blog's strongest original angle**, because published tests were not found at report length or with a large taxonomy.

**How it works.** TEF (1) splits a post into sentences, (2) asks the LLM once per sentence for a single-token answer and keeps the probability for each category, (3) weights each sentence by 1 minus its entropy divided by log K, so an unsure sentence counts for almost nothing, (4) sums weight × log(p / (1 − p)) per category, clipping extremes with ε = 10⁻⁶ and log-odds at ±10, and (5) picks the highest score. No training is needed. The foil is the AI Risk Observatory's code rollup: a report "mentions" a type if any passage has it, and its substantiveness is the band held by the most passages, with ties going to the higher band.

**Published evidence.** On 5,474 Chinese and 2,884 English posts with five LLMs, TEF gains an average of 4.5 accuracy points over the best of direct prediction, majority vote and soft vote, at about three times the money on Qwen2.5-7B ($0.27-0.28 against $0.09, with 2.7M against 0.5M tokens). **"Segmentation alone does not ensure improvement"**: equal-weight voting does not consistently beat reading the whole post. Gains concentrate on smaller models, with Chinese culture accuracy on Qwen2.5-7B at 44.6 direct and 72.1 TEF, but 73.7 and 73.8 on Qwen3-14B ([TEF](https://arxiv.org/abs/2609.27165), Part, X?). The benchmark is partly machine-made, because a later stage adds LLM-generated samples and has an LLM edit ambiguous texts, and the researcher thinks the 28-point jump needs replicating. An older, very small test on 10 documents and 17 labels found any-chunk rollup beat whole-document prompting against human raters: GPT-4o accuracy 74.12 to 87.65 and recall 76.58 to 95.09, with precision about flat at 91.68 and 90.63 ([Rodriguez et al.](https://arxiv.org/abs/2504.00274), OLD, Part, X). The AI Risk Observatory covers 9,821 annual reports with 24,189 passages and shows agreement falling as the taxonomy gets finer (mean Jaccard 0.73, 0.41, 0.33). It also shows the any-passage rule inflating breadth: in 2025, 41.2% of reports mention AI risk and 4.3% have disclosure rated substantive. **It has no report-level gold, so its rollup itself is not tested** ([AI Risk Observatory](https://arxiv.org/abs/2610.02281), Part, X). Sampling also has support for coarse labels: 48 chunks of at most 128 tokens reach a best weighted F of 0.898 on 18 classes across 25 languages ([Cabrera-Diego](https://arxiv.org/abs/2512.24997), OLD, Part, X). Mean pooling of chunk embeddings is a cheap baseline that beats two hierarchical aggregators in one column on EURLEX, 0.6856 against 0.6318 and 0.6277 ([SATPool](https://aclanthology.org/2026.emnlp-main.2080/), Part, X?).

**Strength.** Weak to moderate for the rules themselves, since the tests use posts, 10 documents or 16 all-positive full texts ([Rahgozar and Mortezaagha](https://doi.org/10.1007/s10462-026-11599-2), Abs, A). The gap is the point: the demo would present an experiment, not a replication. Keep the counting in code, since an LLM reduce step can be swayed when jurors give evidence statements instead of YES/NO, on synthetic strengths ([Consensus Drift](https://aclanthology.org/2026.emnlp-main.1689/), Abs).

### Hold three further panels in reserve

A **two-tier cost panel** has the best cost evidence, but it says nothing about tiers. A third model adds nothing in 118 of 135 comparisons and at most 0.43 points ([Bouchard](https://arxiv.org/abs/2605.06350), Abs, A). Global routing creates language escalation gaps of up to 30.33 points, which equal-quota routing removes at little accuracy cost, though the abstract does not name the four languages ([Who Pays](https://arxiv.org/abs/2610.23249), Abs, A). A **parent-tier back-off curve** is a demo-friendly output, but hierarchical conformal work uses a BERT classifier ([HCC](https://arxiv.org/abs/2508.13288), OLD, Part, X) and conformal prediction over a label hierarchy was not found for LLM text tagging. A **gap report** has good parts, with 20 of 178 reviewed UN paragraphs showing genuine taxonomy gaps ([UZH overview](https://aclanthology.org/2026.argmining-1.10/), Part, X), a coverage score ([TaxCE](https://aclanthology.org/2026.emnlp-industry.76/), Part) and a proposal validator ([ReLTEx](https://arxiv.org/abs/2608.10970), Part, X), but an end-to-end loop on a supplied large taxonomy was not found.

## The comparison shows no technique wins on every axis

"Uses tiers" means the supplied hierarchy affects the prediction. Figures are the most useful published ones, from the setting named, and they are not comparable across rows.

| Technique | Uses supplied tiers? | Needs training labels? | Key published figure and setting | Main weakness | Strength | Role |
|---|---|---|---|---|---|---|
| Flat trained extreme classifier (XR-Transformer, Omikuji) | No | Yes | Full text F1* 0.398, 204,056 labels | Rare labels: cps-F1* 0.339 against 0.362 | Strong, library data | Slot 1 baseline |
| LLM-Ensemble, map after generating | No | No | Full text g-F1@5 0.601 | 998 ms per title, called infeasible for daily use | Strong, library data | Slot 1 |
| KI-FSPrompt, few-shot from similar documents | No | Yes | Titles F1* 0.453, 271 ms | Returns 3 to 10 terms; not graded or run on full text | Moderate | Slot 1, optional |
| Hard two-stage routing | Yes | Yes in the test | 0.332 against flat 0.3659, 82 labels | 30.2% cannot recover | Weak to moderate | Slot 3 contrast |
| Multi-step top-down prompt | Yes | No | Micro 58.01 against flat 74.51 | Loses micro, costs more time | Weak, single run | Slot 3 contrast |
| Tree in one prompt | Yes | No | Macro 62.48, micro 84.42 | Confounded with a reasoning model; 2,438 s | Weak to moderate | Slot 3 option |
| Whole-path prompt | Yes | No | Amazon ACC3 0.532 | Reverses by dataset; older paper | Moderate | Slot 3 option |
| Trained multi-branch tree walk | Yes | Yes | Macro 0.2562 against 0.0988 on 15,761 leaves | Micro ties at apparently equal data; 0.0012 zero-shot on a small model | Moderate | Slot 3 |
| Look-ahead and backtracking agent | Yes | No | Full code 43.75 against 40.00 | 408K tokens per case | Moderate, 560 records | Slot 3 |
| Anchor start | Yes | No | HS-Hard 50.54% to 58.29% | Very small gain on Dual-Use | Moderate, small sets | Slot 3 |
| Tier-pruned retrieval and rerank (MATCH) | Yes, pruning and roll-up | No | P@5 88.9, 113,892 Concepts, over two million articles | Self-evaluated; pruning may cut a real secondary topic (my reading of H's design note) | Moderate to strong | Slot 2 |
| Generated label cards (SLKR) | No | No | WOS-143 60.17 against 57.10; 29.04 to 55.83 on a smaller model | Single-label; flagship still higher | Moderate | Slot 2 |
| Confusion partners and rules | Tested on hierarchies | Yes, to find confusions | 98.1% recall at about 18 candidates | 7,100-10,100 input tokens per query | Moderate | Slot 2, optional |
| Reward-trained paths (TaxoAgent) | Yes | Yes, heavy | Micro 77.35 against best prior 74.86 | 5 to over 20 hours on 8 GPUs; 1.09 to 1.17 gold children | Moderate | Not shown |
| Semantic IDs (SemICD) | Replaces the tree | Yes | Micro-F1 up 1.34 to 8.00 points | Abstract and result lines only | Weak | Not shown |
| Agreement-gated whole-taxonomy correction | Yes | No | 0.7139 to 0.7352 with later steps; 394 to 233 to 119 topic-trap calls | Hand-written cues; leaderboard selection; not this step alone | Weak to moderate | Slot 3 story |
| After-the-fact repair of ancestors | Yes | No | Adds 0.38 and 0.55 points | A cheap guarantee, not an accuracy lever | Moderate, one ablation | Always on |
| Evidence verification | Via the supplied definition | No | Precision 82.15% against 55.55% | One domain, own benchmark | Moderate | Slot 4 |
| Two-tier cascade | No | Calibration set | LLM on 1.9% to about 40% of items in tests with far fewer classes | Confidence AUC as low as 0.510; language gaps | Moderate to strong, not on taxonomies | Reserve |
| Parent-tier conformal back-off | Yes | Calibration set | HCC preferred in about 57% of text cases | Not found on LLM tagging; prompt changes erode coverage | Weak for LLMs | Reserve |
| Certainty-weighted rollup (TEF) | No | No | +4.5 accuracy points over the best simple rule | Posts, partly synthetic | Weak to moderate | Slot 5 |
| Any-chunk rollup | No | No | Recall 76.58 to 95.09 on 10 documents | Inflates breadth: 41.2% against 4.3% | Weak | Slot 5 foil |
| Chunk sampling and mean pooling | No | Yes for the head | Weighted F 0.898 on 18 classes | Long tail unknown | Moderate, coarse labels | Slot 5 baseline |
| Batch, cache and in-prompt batching | No | No | 89% lower cost, AUC 0.609 against 0.608 | Metric is downstream ranking, not label accuracy | Moderate, one paper | Implementation |
| Gap report pieces | Yes | No | 20 of 178 paragraphs with gaps | Whole loop not found | Weak as a system | Reserve |

## Published benchmarks set the bar on the two test sets

Added 6 November 2026, after the experiments began, so this section reports on the demo's own test sets (B). **On a chronological split, BERT reaches mean R-Precision 48.4 on all MultiEURLEX labels, against 74.0 on a random split**, so the higher EuroVoc figures elsewhere are not comparable ([MultiEURLEX](https://arxiv.org/abs/2109.00904), Part, X). On HumSet the highest printed scores are [HumBERT](https://arxiv.org/abs/2305.16756)'s: 0.78, 0.70, 0.54, 0.71 and 0.51 for sectors, pillars 1D, subpillars 1D, pillars 2D and subpillars 2D (Full, X). Its table does not say how it averages, and it sits well above the [HumSet paper](https://arxiv.org/abs/2210.04573)'s macro scores for the same XLM-R model: 0.73, 0.53, 0.38, 0.63 and 0.40 (Part, X). The recent methods found report gains from under 1 point to about 5, and none reports a step change. Our own scores on the same measures are in B, from a scratch script that no notebook has yet reproduced.

## The 2025 shared tasks and three candidate ideas, checked on 9–10 November 2026

Added after notebook 10's test run. **The clearest recent head-to-head of subject taggers is the 2025 LLMs4Subjects
shared tasks, and Annif won them with classic extreme-label classifiers in an ensemble, plus an LLM writing synthetic
training records and reranking.** It placed first on the full-vocabulary track of SemEval-2025 Task 5 and first overall at
GermEval-2025 ([task overview](https://arxiv.org/abs/2504.07199), snippet only; [Annif at SemEval](https://arxiv.org/abs/2504.19675),
Full; [Annif at GermEval](https://arxiv.org/abs/2508.15877), snippet only). That recipe is close to notebook 10's stack, so
the stack is current practice, not a new method. No 2025–2026 result was found above BERT's 48.4 on MultiEURLEX's
chronological split, or above HumBERT's printed HumSet scores; the search was shallow.

Three ideas aimed at measured weak spots were checked against the literature by a subagent and against our development
data (B):

- **Synthetic examples for never-seen labels, written from the label's place in the tree.** The weak spot is small: 1.0%
  of the later development laws' gold labels (142 of 14,610) never appear in training. The nearest prior work does
  nearly this for medical codes, with small gains ([Vo et al. 2025](https://arxiv.org/abs/2511.14112), Full), and
  [TELEClass](https://arxiv.org/abs/2403.00165) (Full) writes documents for every path. Not pursued.
- **Choosing among sibling labels together.** Only 20% of the fine-tuned HumBERT's wrong labels on development excerpts
  share a parent with a gold label; 80% sit under another parent. Much of the idea exists already
  ([HiLight](https://arxiv.org/abs/2408.05786), snippet only). Not pursued.
- **Time-aware label priors.** On MultiEURLEX development laws, a label's share among the latest 5,000 training laws
  predicts its share in the later development laws better than its overall share (rank correlation 0.58 against 0.52),
  and a label's recent trend tends to continue (0.45 over the 2,739 labels used five times or more). No use of such
  priors in subject indexing or large-label classification was found. [ChronosLex](https://arxiv.org/abs/2405.14211)
  (Part) retrains over time on the same laws instead, and tag recommendation mixes how often and how recently a user
  applied a tag ([Kowald et al.](https://arxiv.org/abs/1406.7727), snippet only). Pursued in notebook 11.

**GLiNER2.5** (Fastino, Apache 2.0, August–November 2026) labels whole documents in one pass over the text and every
label's name, with label descriptions and rules between labels; its "Decide" checkpoints specialise in classification
([announcement](https://fastino.ai/blog/gliner2-5-span-free-information-extraction), Full; model cards). It reports no
many-label or tree results. Without fine-tuning it falls far below the fine-tuned encoder on HumSet development excerpts
(macro average precision 0.20 against 0.68 on 1D pillars, 400 excerpts); fine-tuned, it reaches 66.5 mean R-Precision on
HumSet and 53.4 on MultiEURLEX development items, against the fine-tuned encoder's 72.2 and 58.6 on the same items
(B). It is added as a source in notebook 11.

**Notebook 11's result, a labelled second attempt against notebook 10's unchanged pass marks (B).** Each addition was
kept only if it raised the combined method's development score by more than notebook 10's margin. On HumSet, GLiNER2.5
added 0.0002 mean macro F1 against a margin of 0.001, so nothing was kept and the test score is unchanged at 0.574, below
HumBERT's printed scores in every category. On MultiEURLEX with the newer laws, both were kept (development gains of 0.32
for GLiNER2.5 and a further 0.47 for the priors), and test mean R-Precision rose from 62.42 to 63.96 (gain +1.55, 95%
interval +1.25 to +1.79) on all labels and from 74.49 to 75.50 at level 3. With training laws only, where GLiNER2.5 was
not trained, the priors alone were kept and raised the test score from 60.04 to 61.05 (+1.01, +0.75 to +1.32). The two
GLiNER2.5 runs cost $5.18 on Hugging Face Jobs, a10g-small. These are results with our reduced combiner and one
GLiNER2.5 fine-tuning recipe, and say nothing about GLiNER2.5 used another way.

## Papers behind each technique, with read status and dates

"None found" means a title search found nothing, which is not proof that no preprint was posted. "Not checked" or "not looked up" means nobody looked. "Not stated" means the notes give no date.

### Benchmarks on HumSet and EuroVoc, and recent methods aimed at beating them

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| HumSet, Fekih et al. [arXiv 2210.04573](https://arxiv.org/abs/2210.04573), [2022.findings-emnlp.321](https://aclanthology.org/2022.findings-emnlp.321/) | Oct 2022, **OLD** | Findings of EMNLP 2022 | Part (B; appendix tables by summary) | X | Macro F1 per category, threshold per label; no agreement figure |
| HumBERT, Tamagnone et al. [arXiv 2305.16756](https://arxiv.org/abs/2305.16756) | v2 30 May 2023, **OLD** | IJCAI 2023 per a search result, not opened | Full (B) | X | Highest printed HumSet scores; averaging not stated |
| MultiEURLEX, Chalkidis et al. [arXiv 2109.00904](https://arxiv.org/abs/2109.00904), [2021.emnlp-main.559](https://aclanthology.org/2021.emnlp-main.559/) | v2 6 Sep 2021, **OLD** | EMNLP 2021 | Part, pages 1 to 8 (B) | X | Chronological 48.4 against random 74.0 |
| JEX, Steinberger et al. [arXiv 1309.5223](https://arxiv.org/abs/1309.5223) | Not checked, **OLD** | Not stated | Part, pages 1 to 6 (B) | X | English F1 0.5147 at rank 6; human agreement 74% and 84%, second-hand from a 2003 paper |
| PyEuroVoc, Avram et al. [arXiv 2108.01139](https://arxiv.org/abs/2108.01139) | 2 Aug 2021, **OLD** | RANLP 2021 | Full (B) | X | English BERT 78.95 on random splits |
| Shaheen et al. [arXiv 2010.12871](https://arxiv.org/abs/2010.12871) | Oct 2020, **OLD** | Not stated | Part; results by summary (B) | X for the abstract, F otherwise | Micro-F1 0.661 and 0.754, not chronological |
| EU Parliament [eurovoc_en model card](https://huggingface.co/scampion/eurovoc_en) | Not stated | Model card | Full (B) | X | Micro F1 0.7233; split not stated |
| Bocchi and Palmero Aprosio [CEUR Vol-3878](https://ceur-ws.org/Vol-3878/10_main_long.pdf) | Not checked | CLiC-it 2024, **OLD** | Full (B) | X | Filling 512 tokens beats reordering sentences |
| RAE-XMC, Wang et al. [arXiv 2502.10615](https://arxiv.org/abs/2502.10615) | 15 Feb 2025, **OLD** | Not stated | Part (B) | X | Label text and training documents in one memory; no legal data |
| UniDEC, Kharbanda et al. [arXiv 2405.03714](https://arxiv.org/abs/2405.03714), [DOI](https://doi.org/10.1145/3696410.3714704) | v2 3 Mar 2025, **OLD** | WWW 2025 | Part (B) | X; timings F | Joint dual encoder and classifier on one 48GB GPU |
| Annif at GermEval-2025, Suominen et al. [arXiv 2508.15877](https://arxiv.org/abs/2508.15877) | 21 Aug 2025, **OLD** | Not stated | Part (B) | X | First place; no LLM fine-tuned |
| SemEval-2025 Task 5, D'Souza et al. [arXiv 2504.07199](https://arxiv.org/abs/2504.07199) | v3 23 May 2025, **OLD** | Not stated | Part (B) | X, with one table's columns interleaved | A fine-tuned LLM placed second; not a controlled test |
| Dual-Encoders for Extreme Multi-label Classification, [Google Research](https://research.google/pubs/dual-encoders-for-extreme-multi-label-classification/) | Not checked | Not stated | Snip | S | Small dual encoders up to 2% better at million-label scale |

### Starting point: trained models against LLMs on a large vocabulary

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Kähler et al. [arXiv 2607.14882](https://arxiv.org/abs/2607.14882) | 16 Jul 2026, v2 17 Aug 2026 | Not confirmed; submitted to KONVENS 2026 | Full (H, plots as captions only); Part (E) | X; M's earlier figures passed through F | Flat scoring; favoured starting point |
| Kluge and Kähler [2025.semeval-1.148](https://aclanthology.org/2025.semeval-1.148/) | Not stated | 2025, **OLD** | Part (H); E says Sec only | X | Origin of LLM-Ensemble; chosen from 135 combinations |
| KI-FSPrompt, Kähler et al. [2025.konvens-2.42](https://aclanthology.org/2025.konvens-2.42/) | Not stated | Sep 2025, **OLD** | Part | X | Also the source of 20 LLM calls per document and 234 GPU hours for the SemEval system |
| Hauss et al. [DOI](https://doi.org/10.1145/3820755.3832808) | None found (M); "none checked" (H) | 18 Aug 2026, DocEng | Abs (OpenAlex; ACM returned 403) | No numbers | Flat extreme classifiers often competitive |
| DualMLC [arXiv 2609.12915](https://arxiv.org/abs/2609.12915) | 11 Sep 2026 | Preprint | Full (B) | X | Needs labels; no supplied tree; EURLex-4K P@1 88.82 against XR-Transformer 87.22 |
| TAGHN [arXiv 2610.20048](https://arxiv.org/abs/2610.20048), [DOI](https://doi.org/10.1145/3805696.3846044) | 14 Oct 2026 | 13 Oct 2026, JCDL per Crossref | Abs | A | Macro-F1 0.380 against 0.346; picks a narrower term |
| Usta [DOI](https://doi.org/10.1108/lht-10-2025-0269) | Not stated | 26 Aug 2026, Library Hi Tech | Abs | A | 2.4 million Spanish records; odd 97/3 blend |
| R3, Xia et al. [DOI](https://doi.org/10.1177/01655515261449401) | Not stated | 15 Jul 2026, J. Information Science | Abs | A | Candidates from similar documents, then LLM |
| EMMETT and IRENE [arXiv 2606.25237](https://arxiv.org/abs/2606.25237) | 23 Jun 2026 | Not stated | Abs | A | Up to 15 points of Recall@10 on zero-shot items |
| AstroConcepts [2026.nslp-1.1](https://aclanthology.org/2026.nslp-1.1/), [arXiv 2604.02156](https://arxiv.org/abs/2604.02156) | 2 Apr 2026, **PRE** | May 2026, NSLP | Part (M) | X | 2,367 concepts; tail F1 0.198 against 0.081 |
| Lynch and Bosco [DOI](https://doi.org/10.5465/amproc.2026.18478poster) | Not stated | Jul 2026, AoM Proceedings | Abs | A | Poster; zero-shot micro-F1 0.90 against SBERT 0.77 |
| BTZSC [arXiv 2603.11991](https://arxiv.org/abs/2603.11991) | 12 Mar 2026, **OLD** | Not stated | Abs (DataCite) | A | Rerankers macro-F1 0.72; small LLMs up to 0.67 |

### Top-down, flat and whole paths: head-to-heads and tier decline

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Tashkees-AI [2026.abjadnlp-1.20](https://aclanthology.org/2026.abjadnlp-1.20/) | None found | Mar 2026, **OLD**, just outside | Part | X | The one plain flat-against-two-stage test found |
| XplaiNLP [2026.nslp-1.29](https://aclanthology.org/2026.nslp-1.29/) | None found | May 2026, NSLP | Part | X | Times from CodeCarbon; single-run scores |
| Graph-Constrained Policy [arXiv 2607.11954](https://arxiv.org/abs/2607.11954) | 11 Jul 2026 | Preprint | Part (H); Abs (C) | X | Three-specialist cascade overflows a 32,000-token cap on 28 to 32% of notes |
| ToT-ICD [2026.findings-acl.191](https://aclanthology.org/2026.findings-acl.191/) | None found | Jul 2026, Findings of ACL | Part | X | Single runs per configuration |
| Constraint-Aware Search [arXiv 2607.10588](https://arxiv.org/abs/2607.10588) | 12 Jul 2026 | Preprint for Elsevier | Part (H); Abs (M, no numbers) | X | Mean of three runs |
| Yoshimura and Kashima [arXiv 2508.04219](https://arxiv.org/abs/2508.04219) | 6 Aug 2025, **OLD** | Not stated | Part | X | Clearest origin of the three-strategy comparison |
| Alangari et al. [DOI](https://doi.org/10.3390/electronics15142988) | Not checked | 8 Jul 2026, Electronics | Abs (Crossref, truncated) | A | Hierarchy-aware encoder 76.07 macro-F1; GPT-4o 5-shot 74.30 |
| He et al. [2026.emnlp-industry.121](https://aclanthology.org/2026.emnlp-industry.121/), [arXiv 2603.23172](https://arxiv.org/abs/2603.23172) | 24 Mar 2026, **PRE** | Oct 2026, EMNLP Industry | Part | X | Paired tier-and-language design; 30 short-text labels; no French |
| Nguyen et al. HTS [arXiv 2606.16987](https://arxiv.org/abs/2606.16987) | 15 Jun 2026 | Not stated | Part | X | 3,300 records; 40.97% exact match |
| MMM-Bench [arXiv 2605.10550](https://arxiv.org/abs/2605.10550) | 11 May 2026, v2 14 May | Not stated | Part | X | 5,990 documents, five levels |
| MAX-EVAL-11 [2026.bionlp-1.23](https://aclanthology.org/2026.bionlp-1.23/) | No arXiv title match | Jul 2026, BioNLP | Part (E); Abs (H) | X? | Exact match 0.000 to 0.048 on ICD-11 |
| AraGenre overview [2026.arabicnlp-sharedtasks.27](https://aclanthology.org/2026.arabicnlp-sharedtasks.27/), [arXiv 2609.27387](https://arxiv.org/abs/2609.27387) | 23 Sep 2026 | Oct 2026 | Part | X | Warns a hard broad-first constraint may propagate errors; cannot say whether constraints help |
| NAMAA [2026.arabicnlp-sharedtasks.33](https://aclanthology.org/2026.arabicnlp-sharedtasks.33/) | Not looked up | Oct 2026 | Abs | A | Broad 0.8913, specific 0.5114; bi-encoders 0.93 in development, as low as 0.22 on test |
| Phoenix [2026.arabicnlp-sharedtasks.31](https://aclanthology.org/2026.arabicnlp-sharedtasks.31/) | Not looked up | Oct 2026 | Abs | A | Margin-gated widening; test 0.5800 |
| Kelm et al. [arXiv 2607.07573](https://arxiv.org/abs/2607.07573) | 8 Jul 2026 | Not stated | Abs | A | Errors follow the tree; Pearson r > 0.92 across encoders |
| YNU-HPCC [2026.semeval-1.45](https://aclanthology.org/2026.semeval-1.45/) | Not looked up | Jul 2026, SemEval | Abs | A | Hierarchical prompting; 6th and 5th place |
| HiPP [2026.findings-acl.139](https://aclanthology.org/2026.findings-acl.139/), [arXiv 2605.13663](https://arxiv.org/abs/2605.13663) | 13 May 2026 | Jul 2026, Findings of ACL | Abs | No numbers | Predicts fine labels, then aggregates upwards |
| RHC [2026.findings-acl.541](https://aclanthology.org/2026.findings-acl.541/), [arXiv 2510.07167](https://arxiv.org/abs/2510.07167) | 8 Oct 2025, **PRE** | Jul 2026, Findings of ACL | Abs (M) | A | About +3% over fine-tuning on patents; needs training |
| SemICD [arXiv 2610.19301](https://arxiv.org/abs/2610.19301) | 13 Oct 2026 | Not stated | Abs plus result lines | X | Replaces the supplied tree for generation |
| MERIT [2026.emnlp-industry.218](https://aclanthology.org/2026.emnlp-industry.218/) | Not checked | Oct 2026, EMNLP Industry | Abs | No numbers beyond 250k+ categories | Names exposure bias in path generation |

### Recovery moves and tool-based walking

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Thakaa [2026.arabicnlp-sharedtasks.38](https://aclanthology.org/2026.arabicnlp-sharedtasks.38/) | Not looked up | Oct 2026 | Part (H, E); Abs (M) | X | 1,178 proposed moves, 460 applied; see disagreements |
| HCRE [2026.findings-acl.1985](https://aclanthology.org/2026.findings-acl.1985/), [arXiv 2604.07937](https://arxiv.org/abs/2604.07937) | 9 Apr 2026, **PRE** | Jul 2026, Findings of ACL | Abs | No numbers | Verifies each level; relation extraction |
| Understanding the Limits of Agentic ICD Coding [2026.emnlp-main.1583](https://aclanthology.org/2026.emnlp-main.1583/), [arXiv 2609.13806](https://arxiv.org/abs/2609.13806) | 12 Sep 2026 | Oct 2026, EMNLP | Part | X | "The bottleneck is selection"; rare micro-F1 0.560 against trained 0.227 |
| Sadkovskii et al. [DOI](https://doi.org/10.26907/1562-5419-2026-29-4-1212-1234), [English DOI](https://doi.org/10.3103/s0005105526700780) | Not stated | 13 Jul 2026; English Oct 2026 | Abs | A | Taxonomy expansion; agent works better as a checker than walking from the root |
| PaperRouter-Agent [arXiv 2607.11564](https://arxiv.org/abs/2607.11564) | 13 Jul 2026 | Not stated | Abs | A | Judges a node by the documents in it; formative study |
| MedDCR [2026.findings-acl.627](https://aclanthology.org/2026.findings-acl.627/) | Not looked up | Jul 2026, Findings of ACL | Abs | No numbers | Learns the workflow design |
| Deterministic Agentic Workflow [arXiv 2605.14857](https://arxiv.org/abs/2605.14857) | 14 May 2026 | Not stated | Abs | A | Top-1 64.2%, top-3 78.3% at six digits |

### Retrieval, pruning and label knowledge

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| MATCH [arXiv 2610.25948](https://arxiv.org/abs/2610.25948) | 19 Oct 2026 | Preprint | Part | X | Authors say transfer beyond MDPI is unevaluated |
| SLKR [arXiv 2610.16182](https://arxiv.org/abs/2610.16182) | 12 Oct 2026 | Under review | Part (H, C, differing tables) | X | Build can grow with the square of the label count |
| From Confusion to Clarity [2026.emnlp-industry.125](https://aclanthology.org/2026.emnlp-industry.125/), [arXiv 2609.01564](https://arxiv.org/abs/2609.01564) | 1 Sep 2026 | Oct 2026, EMNLP Industry | Part | X? | Prompt optimisers over a full list scored below zero-shot |
| UZH overview [2026.argmining-1.10](https://aclanthology.org/2026.argmining-1.10/) | Not looked up | Jul 2026, ArgMining | Part (M) | X | French resolutions; silver tags; models limited to 8B |
| POINTERS [2026.argmining-1.16](https://aclanthology.org/2026.argmining-1.16/) | Not looked up | Jul 2026 | Part | X | 126 codes here against 141 in the overview |
| LLM-INSTRUCT [arXiv 2607.20430](https://arxiv.org/abs/2607.20430), [2026.argmining-1.11](https://aclanthology.org/2026.argmining-1.11/) | 10 May 2026 | Jul 2026 | Abs via fetch tool (M) | F | Development tag micro-F1 35.83% to 40.08% |
| RESOLVENOW [2026.argmining-1.12](https://aclanthology.org/2026.argmining-1.12/) | Not looked up | Jul 2026 | Abs (M) | A | Retrieval "dropped relevant tags"; reverted to a full-taxonomy prompt |
| Foroutan et al. [2026.nslp-1.22](https://aclanthology.org/2026.nslp-1.22/) | Not looked up | May 2026, NSLP | Abs | A | Macro-F1 59.72%, second place |
| Zhao et al. Earth science [DOI](https://doi.org/10.3390/app16115230) | Not stated | 23 May 2026, Applied Sciences | Abs | A | +8.68 micro and +29.92 macro over BERT-BiLSTM on two levels |
| Achananuparp and Lim [DOI](https://doi.org/10.1609/icwsm.v20i1.42622) | Not checked | 25 May 2026, ICWSM | Abs | No numbers | Inference, retrieval, then taxonomy-guided rerank |
| RelaClass [DOI](https://doi.org/10.1109/ijcnn69854.2026.11730021) | Not stated | Jun 2026, IJCNN | Abs (OpenAlex) | No numbers | Class names only; hierarchy of codebooks |
| Mosaic [2026.emnlp-industry.33](https://aclanthology.org/2026.emnlp-industry.33/) | No arXiv title match | Oct 2026, EMNLP Industry | Abs plus introduction | A | Generated descriptions for 140 classes; 2-4 weeks to under 30 minutes |
| TEAMMix [arXiv 2608.11044](https://arxiv.org/abs/2608.11044), [DOI](https://doi.org/10.1109/cscwd68734.2026.11582679) | 11 Aug 2026; Crossref dates the paper 13 May 2026 | CSCWD 2026 | Abs (M) | No numbers | Pseudo-samples for rare labels |
| TaxMorph [2026.eacl-long.10](https://aclanthology.org/2026.eacl-long.10/), [arXiv 2601.18375](https://arxiv.org/abs/2601.18375) | 26 Jan 2026, **OLD** | Mar 2026, EACL | Abs | A | Up to 2.9 points of F1 by rewriting the tree; may only be a diagnostic for a supplied taxonomy |
| BioHiCL [2026.acl-short.62](https://aclanthology.org/2026.acl-short.62/), [arXiv 2604.15591](https://arxiv.org/abs/2604.15591) | Registered 20 Apr 2026 per DataCite, v1 not checked, **PRE** | Jul 2026, ACL Short | Abs | No numbers | MeSH hierarchy as graded positives |
| HiCo-EL [2026.findings-emnlp.103](https://aclanthology.org/2026.findings-emnlp.103/) | No preprint found | Oct 2026 | Abs (M) | Not used | Ontology structure in a bi-encoder |
| OntologyAligner [arXiv 2609.10055](https://arxiv.org/abs/2609.10055) | 9 Sep 2026 | Not stated | Abs (M) | Not used | Hierarchy-guided rerank |
| Sci-ZSEL [2026.findings-emnlp.1286](https://aclanthology.org/2026.findings-emnlp.1286/), [arXiv 2609.00228](https://arxiv.org/abs/2609.00228) | 31 Aug 2026 | Oct 2026 | Part (M) | X | Filtered LLM aliases; 72.91% to 90.32% accuracy |
| Kononykhina et al. [DOI](https://doi.org/10.31235/osf.io/g6wjy_v1) | 19 May 2026, SocArXiv | Not stated | Abs (Crossref) | A | How categories are written moves classification 11-21 points |
| Yilmaz et al. [DOI](https://doi.org/10.31235/osf.io/xv2gf_v2) | 25 Jul 2026, SocArXiv | v2 11 Sep 2026 | Abs, partly | A | 59.2% exact four digits; about 80% at coarser levels |
| RAPT [arXiv 2605.16535](https://arxiv.org/abs/2605.16535) | 15 May 2026 | Not stated | Abs | A | Per-document thresholds from similar cases; 0.87 macro-F1 |
| InfluMatch [arXiv 2607.05968](https://arxiv.org/abs/2607.05968) | Not stated | Not stated | Abs | A | 50 reranked to 10; only 11 queries |
| Student-Guided Teacher Distillation [arXiv 2610.02516](https://arxiv.org/abs/2610.02516) | 1 Oct 2026 | Not stated | Abs (M) | A | Coverage@16 of 91-100%, 200 examples |
| SelectED [2026.uncertainlp-main.9](https://aclanthology.org/2026.uncertainlp-main.9/) | No preprint found | Oct 2026 | Part (M) | X | k=20 uses 37.6% more tokens than k=2 with no accuracy gain |
| DuCEI [2026.findings-emnlp.1298](https://aclanthology.org/2026.findings-emnlp.1298/) | No preprint found | Oct 2026 | Abs (M) | A | Second stage on 27.8% |
| Hydropower [arXiv 2608.23241](https://arxiv.org/abs/2608.23241) | 24 Aug 2026 | Not stated | Abs via fetch tool (M) | F | 135 categories, 40 unseen; hybrid micro-F1 0.524 |
| SNOMED CT Concept Recommendation [arXiv 2609.17855](https://arxiv.org/abs/2609.17855) | 15 Sep 2026 | Not stated | Abs (M) | A | Recall@10 7.74% rare against 43.90% common |
| Odds-Shift Slippage [arXiv 2609.13810](https://arxiv.org/abs/2609.13810) | 12 Sep 2026 | Not stated | Abs, partly | A | MAP@7 0.808 to 0.117; tabular |
| HCMUS at AbjadMed [2026.abjadnlp-1.7](https://aclanthology.org/2026.abjadnlp-1.7/) | Not looked up | Mar 2026, **OLD** | Abs | A | Macro-F1 0.4902 |
| Shlkamy and Zaghouani [2026.arabicnlp-sharedtasks.30](https://aclanthology.org/2026.arabicnlp-sharedtasks.30/) | Not looked up | Oct 2026 | Abs | A | Definition-only retrieval 0.6125 in development; identical tier scores look odd |
| GHADAI [2026.arabicnlp-sharedtasks.34](https://aclanthology.org/2026.arabicnlp-sharedtasks.34/) | Not looked up | Oct 2026 | Abs | A | Option rotation; 0.4213 to 0.6050 |

### Choosing among many labels, consistency and repair

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Lost in Choice [2026.findings-emnlp.328](https://aclanthology.org/2026.findings-emnlp.328/) | Not found | Oct 2026, Findings of EMNLP | Abs plus lines (H); Abs plus Part (C); degradation table not extracted | A | Up to 1,000 options; accuracy collapses at late positions |
| dLLM-SetScore [arXiv 2608.14649](https://arxiv.org/abs/2608.14649) | 30 Jul 2026 | Not stated | Abs | A | First slot positive on 99.4% and 100%; diffusion models |
| LinkedIn semantic labelling [arXiv 2610.36383](https://arxiv.org/abs/2610.36383) | 26 Oct 2026 | Not stated | Abs plus Part | X | Reordering changes at most 0.01% of decisions; needs fine-tuning |
| OPINE [2026.findings-acl.1617](https://aclanthology.org/2026.findings-acl.1617/) | Not looked up | Jul 2026, Findings of ACL | Abs (H); Abs-cut (E) | A | Per-label scoring with a quantile rule; micro-F1 63.20 on 18 labels |
| TaxoAgent [2026.emnlp-main.1006](https://aclanthology.org/2026.emnlp-main.1006/) | None found | Oct 2026, EMNLP | Part (H, E) | X | Inconsistency 10.2% to 5.3%; out-of-scope F1 up to 3.68 points |
| Flügel et al. [DOI](https://doi.org/10.21203/rs.3.rs-9023090/v1) | 25 May 2026, Research Square | Not stated | Abs | A | Reconciles ensembles on ChEBI, 1,742 classes; not text |
| Hierarchical Utility Calibration [arXiv 2609.36532](https://arxiv.org/abs/2609.36532) | 29 Sep 2026 | Not stated | Abs | No numbers | Errors at different nodes can cancel |
| Profiling Hallucinations [2026.clinicalnlp-1.41](https://aclanthology.org/2026.clinicalnlp-1.41/) | No preprint found | May 2026 | Part (M) | Not used | Unconstrained models invent codes |

### Cascades and routing

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Task Cascades [arXiv 2601.05536](https://arxiv.org/abs/2601.05536), [DOI](https://doi.org/10.1145/3786702) | 9 Jan 2026, **OLD** | Feb 2026, SIGMOD | Part | X | 36% lower cost on eight tasks; changes how much of a document is read |
| BARGAIN [DOI](https://doi.org/10.1145/3769776) | Not established | Dec 2025, **OLD** | Part | X | Certified threshold for the cheap model |
| ScaleDoc [arXiv 2509.12610](https://arxiv.org/abs/2509.12610) | 16 Sep 2025, v2 3 Mar 2026, **OLD** | SIGMOD 2026 per PDF name, not confirmed | Part | X | Up to 85% fewer LLM calls |
| Chung et al. [arXiv 2603.15970](https://arxiv.org/abs/2603.15970) | 16 Mar 2026, **OLD** | Accepted to SIGMOD 2026 per the paper | Part | X | Vendor-authored; logistic regression on 200 examples as proxy |
| HIRA [arXiv 2608.21792](https://arxiv.org/abs/2608.21792), [DOI](https://doi.org/10.1145/3799682.3840098) | 22 Aug 2026 | CIKM 2026, 7-11 Nov 2026 | Part | X | OCR-noisy documents; macro-F1 0.6218 to 0.8548 on 80 classes |
| CARE [2026.emnlp-industry.263](https://aclanthology.org/2026.emnlp-industry.263/) | Not looked up | Oct 2026 | Abs plus Part | X | 1.9% of one call per document on WikiLeaks; few classes |
| SIFT [arXiv 2607.18358](https://arxiv.org/abs/2607.18358) | 20 Jul 2026 | Not stated | Abs | No benchmark numbers | Frozen golden set gates retraining |
| L2D-Clinical [2026.clinicalnlp-1.5](https://aclanthology.org/2026.clinicalnlp-1.5/) | Not looked up | May 2026 | Abs | A | Defers 7% and 16.8%; MIMIC truth is LLM consensus |
| Sun and Sang [2026.emnlp-industry.257](https://aclanthology.org/2026.emnlp-industry.257/) | Not looked up | Oct 2026 | Abs | A | Small-LLM rerank below retrieval |
| The Embedder's Dilemma [arXiv 2608.12875](https://arxiv.org/abs/2608.12875) | 13 Aug 2026 | COLM 2026, venue S only | Abs | A | LLM up to 1,431× cost for comparable quality |
| AutoRelAnnotator [arXiv 2606.25871](https://arxiv.org/abs/2606.25871) | 24 Jun 2026 | SIGIR 2026 workshop | Abs | A | Vendor-authored; 150M+ annotations |
| Abusaqer and Saquer [DOI](https://doi.org/10.1109/icmla69890.2026.00102) | Not stated | Oct 2026, ICMLA | Abs (OpenAlex) | A | $26.00 against $96.29 |
| Chinnam et al. [DOI](https://doi.org/10.1109/icact70625.2026.11748740) | Not stated | Sep 2026, ICACT | Abs (OpenAlex) | A | Conformal risk control; 25.3% escalated |
| Özaslan and Adak [DOI](https://doi.org/10.1109/yef-ece70590.2026.11614634) | Not stated | Jun 2026 | Abs (OpenAlex) | A | Cascade 0.9662; 5.2% escalated |
| MECH [2026.acl-long.1028](https://aclanthology.org/2026.acl-long.1028/) | Not looked up | ACL 2026 | Abs | A | 44.4% lower API cost |
| JEVDB [arXiv 2610.02046](https://arxiv.org/abs/2610.02046) | 1 Oct 2026 | Not stated | Abs | A | Removes 87.4% of pairs |
| Khatri et al. [arXiv 2610.36915](https://arxiv.org/abs/2610.36915) | 27 Oct 2026 | Not stated | Abs | A | Yes/no decomposition of a taxonomy lowered accuracy for every model |
| System1-Arena [arXiv 2610.21244](https://arxiv.org/abs/2610.21244) | 15 Oct 2026 | Not stated | Abs | A | 92k questions, 30 models |
| Bouchard [arXiv 2605.06350](https://arxiv.org/abs/2605.06350) | 7 May 2026, revised 5 Oct 2026 | Not stated | Abs | A | Two tiers usually enough |
| Who Pays [arXiv 2610.23249](https://arxiv.org/abs/2610.23249) | 16 Oct 2026 | Not stated | Abs | A | Confidence AUC 0.705, 0.612, 0.510 |
| UCCI [arXiv 2605.18796](https://arxiv.org/abs/2605.18796) | 11 May 2026 | Not stated | Abs | A | 31% saving at micro-F1 0.91; single author |
| Cornacchia et al. [2026.emnlp-industry.252](https://aclanthology.org/2026.emnlp-industry.252/) | Not looked up | Oct 2026 | Abs | A | Calibration makes thresholds deployable, not rankings better |
| Mansoori et al. [2026.emnlp-industry.187](https://aclanthology.org/2026.emnlp-industry.187/) | Not looked up | Oct 2026 | Abs | A | About 11 points more values accepted, from per-task reliability |
| Conformal Cascade [arXiv 2607.25018](https://arxiv.org/abs/2607.25018) | 27 Jul 2026 | Not stated | Abs | No numbers beyond 18 benchmarks | Accept when the set has a single answer |
| Circuits of LLM Classifiers [arXiv 2610.30076](https://arxiv.org/abs/2610.30076) | 21 Oct 2026 | Not stated | Abs plus Part (C); Abs (M) | X | 38-86% fewer calls |
| Peale et al. [arXiv 2605.07805](https://arxiv.org/abs/2605.07805) | 8 May 2026 | Not stated | Abs | No numbers | Route when uncertainty can be reduced |
| Yadav [arXiv 2610.33258](https://arxiv.org/abs/2610.33258) | 23 Oct 2026 | Not stated | Abs | A | AUROC 0.83 on a cheap model |
| Negi et al. [2026.uncertainlp-main.35](https://aclanthology.org/2026.uncertainlp-main.35/) | Not looked up | UncertaiNLP 2026 | Abs | No numbers | Agreement among LLMs predicts correctness |
| QUORUM [2026.emnlp-main.1172](https://aclanthology.org/2026.emnlp-main.1172/), [arXiv 2608.27974](https://arxiv.org/abs/2608.27974) | 28 Aug 2026 | Oct 2026, EMNLP | Abs | A | Quality up to +34.4%, cost -8.8% |
| Differential Reasoning Router [2026.emnlp-industry.166](https://aclanthology.org/2026.emnlp-industry.166/) | Not looked up | EMNLP Industry | Abs | A | More than 60% saving on reasoning tokens |
| Reiter et al. [2026.uncertainlp-main.20](https://aclanthology.org/2026.uncertainlp-main.20/), [arXiv 2609.35824](https://arxiv.org/abs/2609.35824) | 22 Sep 2026 | Oct 2026 | Abs | A | Design and model raise prevalence variance 76.7 and 110.6 times |
| Rajendran [arXiv 2608.06607](https://arxiv.org/abs/2608.06607) | 6 Aug 2026 | Not stated | Abs | A | Router does not transfer across datasets |
| Patel [DOI](https://doi.org/10.20944/preprints202610.2704.v1) | 27 Oct 2026, Preprints.org | Not stated | Abs | A | 57.8% tokens saved, F1 0.657 to 0.370; 16 syllabi |
| Kim et al. [arXiv 2606.08090](https://arxiv.org/abs/2606.08090) | 6 Jun 2026 | Not stated | Abs | A | Soft labels from LLM confidence |
| PROBE [arXiv 2611.01454](https://arxiv.org/abs/2611.01454) | 1 Nov 2026 | Acceptance not confirmed | Part | X | The rare-leaf problem in database terms (C's reading) |
| Selection Effect [2026.emnlp-main.1049](https://aclanthology.org/2026.emnlp-main.1049/) | Not looked up | Oct 2026 | Abs | No numbers | No compact LLM verifier improved the frontier behind an encoder gate |
| Rafe and Das [arXiv 2610.00346](https://arxiv.org/abs/2610.00346) | 29 Sep 2026 | Not stated | Abs-cut | A | 5 percent risk leaves 0.528 automated |
| Jev-as-a-Judge [arXiv 2610.20634](https://arxiv.org/abs/2610.20634) | 30 Sep 2026 | Not stated | Abs (M) | A | Tie at a 187-fold cost difference, yet 15.2% disagreement |

### Silver labels, active learning and prompt optimisation

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Bonhomme and Dulaunoy [arXiv 2607.25572](https://arxiv.org/abs/2607.25572) | 28 Jul 2026 | Not stated | Abs plus Part | X | At about 0.39 agreement no reliable gain; checkpoint selection moved recall@5 by up to 0.05 |
| SemDistill [DOI](https://doi.org/10.1145/3837124) | Not stated | 24 Sep 2026, Proc. ACM Manag. Data | Abs (Crossref) | A | Column annotation; 3800× cost cut |
| Kurniawan et al. [arXiv 2610.21254](https://arxiv.org/abs/2610.21254) | 15 Oct 2026 | Not stated | Abs plus Part | X | Students within 3 points of human-trained |
| HP at AraGenre [2026.arabicnlp-sharedtasks.29](https://aclanthology.org/2026.arabicnlp-sharedtasks.29/) | Not looked up | Oct 2026 | Abs | A | 140 examples, 0.7169; transductive |
| Mathur et al. [arXiv 2610.24260](https://arxiv.org/abs/2610.24260) | 16 Oct 2026 | Not stated | Abs | A | Ensemble 97.5% at 4.4% of cost |
| Rokon et al. [arXiv 2606.23911](https://arxiv.org/abs/2606.23911) | 22 Jun 2026 | SIGIR 2026 workshop | Abs | A | 240M+ examples, 89.1% agreement |
| CAPC-CG [2026.acl-long.42](https://aclanthology.org/2026.acl-long.42/) | Not looked up | ACL 2026 | Abs | A | 3.3 million policy paragraphs, κ=0.86, five-colour typology |
| A Cheap Lunch [2026.lrec-1.813](https://aclanthology.org/2026.lrec-1.813/) | Not looked up | May 2026, LREC | Abs | No numbers | Synthetic plus manual annotations |
| Kasliwal and Singh [arXiv 2610.09063](https://arxiv.org/abs/2610.09063) | 6 Oct 2026 | Not stated | Abs via fetch tool (M) | F | 112 topics; discriminative students lose 35% macro-F1 |
| Divide, Conquer, and Distill [2026.findings-emnlp.1447](https://aclanthology.org/2026.findings-emnlp.1447/) | Not looked up | Findings of EMNLP 2026 | Abs | No numbers | Tail items resemble mislabelled ones |
| DeCo [2026.findings-acl.1526](https://aclanthology.org/2026.findings-acl.1526/) | Not looked up | Findings of ACL 2026 | Abs | No numbers | As above |
| GRIP [2026.findings-emnlp.1901](https://aclanthology.org/2026.findings-emnlp.1901/) | Not looked up | Findings of EMNLP 2026 | Abs | A | +9.76 accuracy points on TREC |
| How Noisy is Automatic Annotation? [2026.wnut-1.10](https://aclanthology.org/2026.wnut-1.10/) | Not looked up | Oct 2026 | Abs | A | LLM errors lean to additions, over 51% |
| ESG-SASB Label Stability [DOI](https://doi.org/10.3390/informatics13070106) | Not stated | 3 Jul 2026 | Abs (Crossref) | A | 40.3% of sentences flagged ambiguous |
| Who Gets Annotated? [2026.nlp4pi-1.44](https://aclanthology.org/2026.nlp4pi-1.44/) | Not looked up | Oct 2026, NLP4PI | Abs | A | Closest domain match; critical share 3.4% to 26.8% |
| EvoPool [2026.findings-emnlp.743](https://aclanthology.org/2026.findings-emnlp.743/) | Not looked up | Oct 2026 | Abs | A | Code-written labellers; +0.268 on PubMed |
| EURO-5K [arXiv 2606.02971](https://arxiv.org/abs/2606.02971) | 1 Jun 2026 | Not stated | Abs | A | Converges at around 3K samples |
| Ceccherini et al. [arXiv 2608.18033](https://arxiv.org/abs/2608.18033) | 18 Aug 2026 | Not stated | Abs | A | 0.9 F1 with about 100 invoices |
| Lucca [arXiv 2610.32955](https://arxiv.org/abs/2610.32955) | 23 Oct 2026 | Not stated | Abs, partly | A | 231 emails; per-item accuracy may be the wrong metric |
| Rahman et al. [arXiv 2607.25946](https://arxiv.org/abs/2607.25946) | 28 Jul 2026 | Not stated | Abs | A | Over 124,000 conversations |
| Durmazkeser et al. [arXiv 2605.24981](https://arxiv.org/abs/2605.24981) | 24 May 2026 | Not stated | Abs | A | Annotation cost down up to 81.8% for model selection |
| Beyond Keywords [arXiv 2609.35856](https://arxiv.org/abs/2609.35856) | 25 Sep 2026 | Not stated | Abs | No numbers | Hierarchical policy uncertainty in public-sector news |
| Liu and Hou [2026.findings-eacl.125](https://aclanthology.org/2026.findings-eacl.125/) | Not looked up | Mar 2026, **OLD** | Abs | A | 12,277 segments |
| ERGO [arXiv 2607.20497](https://arxiv.org/abs/2607.20497) | 15 Jun 2026 per arXiv search record | Not stated | Abs | A | No prompt method dominates across 2 to 150 classes |
| Prompt Duel Optimizer [2026.findings-acl.490](https://aclanthology.org/2026.findings-acl.490/) | Not looked up | Findings of ACL 2026 | Abs | No numbers | Not tested on large-label classification |
| ClusterFewshot [arXiv 2609.25939](https://arxiv.org/abs/2609.25939) | 22 Sep 2026 | Not stated | Abs | No numbers | Few-shot selection |
| Susnjak [arXiv 2605.06937](https://arxiv.org/abs/2605.06937) | 7 May 2026 | Not stated | Abs | No numbers | DSPy with GEPA |
| Kim et al. guidelines [2026.acl-long.1760](https://aclanthology.org/2026.acl-long.1760/) | Not looked up | ACL 2026 | Abs | No numbers | Reusing written guidelines |

### Chunk to report

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| TEF [arXiv 2609.27165](https://arxiv.org/abs/2609.27165) | 22 Sep 2026 | Preprint; acceptance not confirmed | Part (Table 3 partly) | X? | Best or tied in 116 of 120 comparisons |
| Rodriguez et al. [arXiv 2504.00274](https://arxiv.org/abs/2504.00274) | 31 Mar 2025, **OLD** | Journal record not opened | Part | X | 10 documents, 170 judgments |
| Rahgozar and Mortezaagha [DOI](https://doi.org/10.1007/s10462-026-11599-2) | Not stated | 26 May 2026, AI Review | Abs (OpenAlex) | A | All-positive gold, so precision cannot be measured |
| Wu and Liu [DOI](https://doi.org/10.21203/rs.3.rs-10632286/v1) | 19 Oct 2026, Research Square | Not stated | Abs | A | Equal weighting 83.0 beaten only by BiGRU plus attention at 84.6 |
| SATPool [2026.emnlp-main.2080](https://aclanthology.org/2026.emnlp-main.2080/) | Not looked up | Oct 2026, EMNLP | Part | X? | English datasets only |
| Cabrera-Diego [arXiv 2512.24997](https://arxiv.org/abs/2512.24997) | 31 Dec 2025, **OLD** | Not stated | Part | X | 498 seconds per 100 files on CPU |
| CPE [2026.lrec-1.586](https://aclanthology.org/2026.lrec-1.586/) | Not looked up | May 2026, LREC | Part | X | About +2-4 macro-F1; specific figures not extracted |
| AI Risk Observatory [arXiv 2610.02281](https://arxiv.org/abs/2610.02281) | 1 Oct 2026 | Not stated | Part | X | Best template in shape; single annotator |
| Soria [DOI](https://doi.org/10.31235/osf.io/er6mz_v1) | 5 Jun 2026, SocArXiv | Not stated | Abs (Crossref, v1) | A | False positives 50% to 3% on ambiguous categories |
| Singer et al. [arXiv 2607.24622](https://arxiv.org/abs/2607.24622) | 27 Jul 2026 | Not stated | Abs | No numbers beyond 33 datasets | Majority voting preserves class proportions |
| Consensus Drift [2026.emnlp-main.1689](https://aclanthology.org/2026.emnlp-main.1689/) | Not looked up | Oct 2026, EMNLP | Abs | No numbers | Strong-sounding minority can override |
| GraphReduce [2026.emnlp-industry.91](https://aclanthology.org/2026.emnlp-industry.91/) | Not looked up | Oct 2026 | Abs | No numbers | Membership kept outside the model |
| When Chunks Disagree [DOI](https://doi.org/10.1109/eecct71142.2026.11746532) | Not stated | Sep 2026, EECCT | Abs (OpenAlex) | A | 58.33% to 81.82%; end-to-end gain not significant |
| Lost in a Single Vector [arXiv 2606.18781](https://arxiv.org/abs/2606.18781) | 17 Jun 2026 | Not stated | Abs | A | Retrieval; 30.0 to 90.0 on one model |
| Information Representation Fairness [2026.findings-acl.246](https://aclanthology.org/2026.findings-acl.246/) | Not looked up | Jul 2026 | Abs | No numbers | Early segments and English over-represented |
| Sarmin et al. [DOI](https://doi.org/10.1109/peccii70991.2026.11661871) | Not stated | Jun 2026 | Abs, partly | A | 50 classes, F1 0.9337 |
| Tekale and Masum [DOI](https://doi.org/10.1109/compsac69091.2026.00037) | Not stated | Jul 2026 | Abs (OpenAlex) | A | Summarising cuts cost 30-54% |
| HyTRAM [DOI](https://doi.org/10.1016/j.rineng.2026.112355) | Crossref record 7 Aug 2026 | Dec 2026 issue | Abs (OpenAlex) | A | Accident reports; 85.76% precision |
| Budget-Aware Routing [2026.findings-acl.2114](https://aclanthology.org/2026.findings-acl.2114/) | Not looked up | Jul 2026 | Abs plus Part | X | Selector matters more than segmentation; summarisation, not classification |
| Campos et al. [2026.propor-1.61](https://aclanthology.org/2026.propor-1.61.pdf) | Not looked up | PROPOR 2026, month not checked | Abs from PDF | X | Tokens down 64-88% |
| SumRank [2026.emnlp-main.1451](https://aclanthology.org/2026.emnlp-main.1451/) | Not looked up | EMNLP 2026 | Abs | No numbers | Task-trained summariser |
| Golub et al. [DOI](https://doi.org/10.1108/jd-06-2026-0363) | Not stated | 22 Sep 2026 | Abs (M) | A | Metadata F1 44% against full text 12%; 14 works of fiction |

### Serving: batching, caching and reuse

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| PromptPack [arXiv 2607.20528](https://arxiv.org/abs/2607.20528) | 10 Jul 2026 | AI4IR 2026 per page header | Part | X | Vendor-authored; 8,500-token shared prompt is 94% of input |
| RoBatch [arXiv 2605.28268](https://arxiv.org/abs/2605.28268) | 27 May 2026 | Not stated | Abs | No numbers | Chooses model and batch size |
| Cascaded Batch Prompting [2026.findings-emnlp.119](https://aclanthology.org/2026.findings-emnlp.119/), [arXiv 2608.27038](https://arxiv.org/abs/2608.27038) | 27 Aug 2026 | Oct 2026 | Abs | No numbers | Separates reasoning from assignment |
| Cache-Aware Prompt Compression [arXiv 2607.15516](https://arxiv.org/abs/2607.15516) | 16 Jul 2026 | Not stated | Abs | A | Mean 49% saving over cache only; single author |
| Silent and Costly [arXiv 2610.34817](https://arxiv.org/abs/2610.34817) | 26 Oct 2026 | Not stated | Abs | A | 120 pull requests; most caching errors fail silently |
| Token Reduction Is Not Cost Reduction [arXiv 2607.12161](https://arxiv.org/abs/2607.12161) | 13 Jul 2026 | Not stated | Abs | A | Report billed cost |
| Belenki et al. [DOI](https://doi.org/10.21203/rs.3.rs-10909832/v1) | 4 Sep 2026, Research Square | Not stated | Abs | A | About $1.0 per 1,000 on the batch API |
| OCTOPUS [arXiv 2610.32586](https://arxiv.org/abs/2610.32586) | 23 Oct 2026 | Not stated | Abs | A | Stores judgments as tags |
| Semantic Views [arXiv 2610.12768](https://arxiv.org/abs/2610.12768) | 8 Oct 2026 | Not stated | Abs | A | 725,000 recorded labels; saves 28-45% of calls |
| SemReuse [arXiv 2610.16035](https://arxiv.org/abs/2610.16035) | 12 Oct 2026 | Not stated | Abs | A | Up to 15.4× reductions |
| NousDB [arXiv 2610.28551](https://arxiv.org/abs/2610.28551) | 20 Oct 2026 | Not stated | Abs | A | Up to 86× speed-up |
| CaBSALLM [2026.acl-short.51](https://aclanthology.org/2026.acl-short.51/) | Not looked up | ACL 2026 Short | Abs | No numbers | Dynamic batching |

### Evaluation: tier credit, working without gold, evidence

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| LCSHBench [arXiv 2606.04382](https://arxiv.org/abs/2606.04382) | 3 Jun 2026 | Preprint | Part | X | 22,346 books, 15 languages; neural runs on a 2,002-record subset |
| Physics subject map [arXiv 2606.14043](https://arxiv.org/abs/2606.14043) | 12 Jun 2026 | Preprint | Part | X | Semantic F1 80.9%, discipline F1 77.6% |
| Semantic F1 [arXiv 2509.21633](https://arxiv.org/abs/2509.21633) | 25 Sep 2025, **OLD** | Not stated | Abs | No numbers | Credit through a label similarity matrix |
| Plaud et al. [arXiv 2410.01305](https://arxiv.org/abs/2410.01305) | 2 Oct 2024, **OLD** | CoNLL 2024 | Abs | No numbers | Metric and inference interact |
| High Scores, Hidden Failures [arXiv 2610.35179](https://arxiv.org/abs/2610.35179) | 26 Oct 2026 | Not stated | Abs | No numbers | Label-level disparities under high aggregates |
| When the Ruler is Broken [arXiv 2605.07293](https://arxiv.org/abs/2605.07293) | 8 May 2026 | Not stated | Abs | A | 0% against 76% from the scorer alone |
| Labbe et al. [2026.bionlp-1.5](https://aclanthology.org/2026.bionlp-1.5/) | Not looked up | Jul 2026 | Abs (M) | No numbers | Hierarchy-aware evaluation |
| Vasquez [arXiv 2606.13436](https://arxiv.org/abs/2606.13436) | 11 Jun 2026 | Preprint | Part | X | Level 1 0.64 to 0.37. Level 2 0.54 to 0.03 is confounded, so do not quote it without that caveat |
| Hossain et al. crash narratives [2026.law-main.18](https://aclanthology.org/2026.law-main.18/) | No arXiv title match | Jul 2026 | Abs | A | 49.4% discrepancy in administrative labels |
| van Weesep et al. [arXiv 2610.01616](https://arxiv.org/abs/2610.01616) | 1 Oct 2026 | Accepted to AIDaR at NeurIPS | Abs | A | Curator revised own labels; about 2 million assays |
| FrenchNews-7 [arXiv 2608.18097](https://arxiv.org/abs/2608.18097) | 8 Jun 2026, v2 15 Oct | Accepted at AACL-IJCNLP, record not checked | Abs | A | Publisher-derived labels in French |
| Włodarczyk and Zdunek [DOI](https://doi.org/10.1080/01639374.2026.2714375) | Not stated | 14 Aug 2026 | Abs (M) | A | LLMs weak at catching wrong descriptors |
| Ravideshik and Kejriwal [2026.emnlp-main.1717](https://aclanthology.org/2026.emnlp-main.1717/), [arXiv 2610.09554](https://arxiv.org/abs/2610.09554) | 7 Oct 2026 | Oct 2026, EMNLP | Abs | A | Hide the decision from the judge |
| Mukherjee et al. [2026.findings-emnlp.1696](https://aclanthology.org/2026.findings-emnlp.1696/), [arXiv 2606.03043](https://arxiv.org/abs/2606.03043) | 2 Jun 2026 | Oct 2026 | Abs-cut | A | 58 to 66% of human agreement |
| Han and Kim [arXiv 2610.13303](https://arxiv.org/abs/2610.13303) | 9 Oct 2026 | Not stated | Abs-cut | A | Consensus up 0.069, agreement with experts unchanged |
| Song and Zheng [2026.emnlp-main.1552](https://aclanthology.org/2026.emnlp-main.1552/) | Not looked up | Oct 2026 | Abs-cut | A | Agreement cut by 62% under independent rubrics |
| Popovic et al. [2026.woah-1.26](https://aclanthology.org/2026.woah-1.26/) | No arXiv title match | Oct 2026, WOAH | Part | X | Agreement is a per-tier property |
| Lin [DOI](https://doi.org/10.31235/osf.io/n3zhf_v1) | 25 Sep 2026, SocArXiv | Not stated | Abs-cut (Crossref) | A | Prompt flag wrong on 64.7% at κ=.971 |
| Nakamura et al. [arXiv 2609.22133](https://arxiv.org/abs/2609.22133) | 24 Aug 2026, updated 28 Oct | Preprint | Part | X | Where LLMs agree, experts agree 93%; where not, 60% |
| Systematic Failure Analysis [2026.emnlp-main.2037](https://aclanthology.org/2026.emnlp-main.2037/) | Not looked up | Oct 2026 | Abs (M) | A | 72 to 94% accuracy at 71 to 90% coverage on clean tasks |
| Piot et al. [2026.lrec-1.341](https://aclanthology.org/2026.lrec-1.341/), [arXiv 2512.09662](https://arxiv.org/abs/2512.09662) | 10 Dec 2025, **PRE** | May 2026, LREC | Part; table values not captured | No numbers | LLM labels preserve classifier ranking; simulated classifiers |
| Gao et al. [2026.emnlp-main.827](https://aclanthology.org/2026.emnlp-main.827/), [arXiv 2608.26638](https://arxiv.org/abs/2608.26638) | 27 Aug 2026 | Oct 2026, EMNLP | Part | X | Prediction-powered saving ratio; translation data |
| Divekar [arXiv 2606.05308](https://arxiv.org/abs/2606.05308) | 3 Jun 2026 | Accepted at GEM, ACL 2026 | Abs | A | 30 human annotations cut standard error 21% |
| Ryan et al. ATT&CK [arXiv 2606.18166](https://arxiv.org/abs/2606.18166) | 16 Jun 2026 | Not stated | Part | X | Best micro-F1 0.22 over 114 labels; 795 of 2,076 sentences are negatives |
| CTI2TTP [arXiv 2605.25836](https://arxiv.org/abs/2605.25836) | 25 May 2026, v2 13 Oct | Preprint | Part | X | Own benchmark |
| Zhang et al. spans [arXiv 2603.15270](https://arxiv.org/abs/2603.15270) | 16 Mar 2026, v3 14 Oct, **OLD** | Not stated | Part | X | May answer "No Evidence Found" |
| RULERS [2026.emnlp-main.566](https://aclanthology.org/2026.emnlp-main.566/), [arXiv 2601.08654](https://arxiv.org/abs/2601.08654) | 13 Jan 2026, **PRE** | Oct 2026, EMNLP | Abs | No numbers | Scoring, not tagging |
| ICDAGENT [2026.acl-long.643](https://aclanthology.org/2026.acl-long.643/) | Not looked up | Jul 2026 | Abs-cut | No numbers | Critic agent verifies |
| KREL [2026.findings-emnlp.1504](https://aclanthology.org/2026.findings-emnlp.1504/) | 21 Aug 2026 | Oct 2026 | Abs-cut | No numbers | Coding with rationales |
| Li et al. rationales [2026.eacl-long.232](https://aclanthology.org/2026.eacl-long.232/) | Not looked up | Mar 2026, **OLD** | Abs-cut | No numbers | Faithfulness and plausibility |
| CPTCoder [2026.acl-demo.60](https://aclanthology.org/2026.acl-demo.60/) | Not looked up | Jul 2026 | Abs | A | Micro-F1 0.61 and 0.51 |
| Brandl and Eberle [2026.trustnlp-main.44](https://aclanthology.org/2026.trustnlp-main.44/), [arXiv 2410.03296](https://arxiv.org/abs/2410.03296) | 4 Oct 2024, **PRE** | Jul 2026, TrustNLP | Abs | No numbers | Depends on length and task |
| Banerjee et al. [arXiv 2605.19274](https://arxiv.org/abs/2605.19274) | 19 May 2026 | Not stated | Abs | A | English-pivot explanations up to 5.7 times worse on comprehensiveness |
| MULTIFINGROUND [arXiv 2610.19432](https://arxiv.org/abs/2610.19432) | 14 Oct 2026 | Not stated | Abs | A | English, French, Greek |
| Believing without Seeing [2026.acl-long.673](https://aclanthology.org/2026.acl-long.673/) | Not looked up | ACL 2026 | Abs-cut | No numbers | Vision-language models; adjacent only |

### Abstention and parent-tier back-off

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| HCC [arXiv 2508.13288](https://arxiv.org/abs/2508.13288) | 18 Aug 2025, **OLD** | Not stated | Part | X | Prefers about 57% text, 71% images; BERT classifier |
| Mortier et al. [arXiv 2501.19038](https://arxiv.org/abs/2501.19038) | 31 Jan 2025, v3 10 Apr 2026, **OLD** | Not stated | Abs | No numbers | Internal nodes as prediction sets |
| HG-CRC [arXiv 2607.24562](https://arxiv.org/abs/2607.24562) | 27 Jul 2026 | Preprint | Part | X | The hierarchy is of groups, not labels; analogy only |
| Strong et al. [arXiv 2605.02734](https://arxiv.org/abs/2605.02734) | 4 May 2026 | Not stated | Abs (E); Meta (H) | No numbers | Per-label deferral can contradict the taxonomy; imaging |
| Zhu et al. [2026.emnlp-main.786](https://aclanthology.org/2026.emnlp-main.786/), [arXiv 2608.01460](https://arxiv.org/abs/2608.01460) | 2 Aug 2026 | Oct 2026, EMNLP | Abs | No numbers beyond study size | Prompt and temperature changes drive coverage below target |
| Kotte, conformal risk control [arXiv 2606.29054](https://arxiv.org/abs/2606.29054) | 27 Jun 2026, v2 7 Sep | Not stated | Abs | A | Target fails on 14 of 16 transfers |
| PASC [arXiv 2605.18812](https://arxiv.org/abs/2605.18812) | 12 May 2026 | Not stated | Abs (M) | A | 96.4% against 86.5% coverage |
| Zecevic et al. [2026.bionlp-1.16](https://aclanthology.org/2026.bionlp-1.16/) | Not looked up | Jul 2026 | Abs | A | Reports review burden beside coverage; 0.61 false-positive labels per report |
| BOLT [2026.findings-acl.667](https://aclanthology.org/2026.findings-acl.667/) | No arXiv title match | Jul 2026 | Abs | No numbers beyond study size | Most methods struggle with unseen classes |
| One Score, Two Decisions [arXiv 2608.14683](https://arxiv.org/abs/2608.14683) | 4 Aug 2026 | Not stated | Abs (M) | No numbers | Margin can remove the "no answer" signal |
| Henning et al. [arXiv 2609.26468](https://arxiv.org/abs/2609.26468) | 22 Sep 2026 | Not stated | Abs (M) | No numbers | Calibration binning misleads when most labels are negative |
| Fonseca et al. [2026.acl-long.2128](https://aclanthology.org/2026.acl-long.2128/) | No arXiv title match | Jul 2026 | Abs-cut | No numbers | Balanced Brier Score |
| de Andrade et al. [2026.uncertainlp-main.41](https://aclanthology.org/2026.uncertainlp-main.41/) | No arXiv title match | Oct 2026 | Abs | No numbers | Class-specific errors under low global error |
| Arana et al. [2026.acl-srw.72](https://aclanthology.org/2026.acl-srw.72/) | Not looked up | Jul 2026 | Abs-cut | No numbers | Repeat-run variation as a signal |
| Khallaf and Sharoff [2026.lrec-1.11](https://aclanthology.org/2026.lrec-1.11/) | Not looked up | May 2026 | Abs | A | Abstaining on 10% lifts macro-F1 0.81 to 0.85 |
| Conformal Labeling [arXiv 2510.14581](https://arxiv.org/abs/2510.14581) | 16 Oct 2025, v3 14 Feb 2026, **OLD** | Not stated | Part | X | 63.73% labelled at FDR 9.85% |
| Choo and Hong [2026.healing-1.14](https://aclanthology.org/2026.healing-1.14/) | Not looked up | Mar 2026, **OLD** | Abs (M) | A | Single-label defaults 96% and 54% fall to 19% and 0.0% |
| Zhao et al. LLMs as Annotators [2026.lrec-1.852](https://aclanthology.org/2026.lrec-1.852/) | Not looked up | May 2026 | Abs | No numbers | Over-assignment concentrates in particular labels |

### Taxonomy gaps and overlaps

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Reference-Free Evaluation [2026.findings-acl.1273](https://aclanthology.org/2026.findings-acl.1273/), [arXiv 2505.11470](https://arxiv.org/abs/2505.11470) | 16 May 2025, **PRE** | Jul 2026, Findings of ACL | Part (E); Abs (M); correlations in figures not captured | No numbers beyond α=0.001 | Scores a taxonomy before any tagging |
| TaxCE [2026.emnlp-industry.76](https://aclanthology.org/2026.emnlp-industry.76/), [arXiv 2608.30614](https://arxiv.org/abs/2608.30614) | 31 Aug 2026 | Oct 2026 | Part | X | Coverage threshold 0.65; granularity definition not read |
| ReLTEx [arXiv 2608.10970](https://arxiv.org/abs/2608.10970) | 11 Aug 2026 | Preprint | Part | X | Generates from the taxonomy, not from unplaced text |
| ConceptE [2026.findings-emnlp.737](https://aclanthology.org/2026.findings-emnlp.737/), [arXiv 2606.21048](https://arxiv.org/abs/2606.21048) | 19 Jun 2026 | Oct 2026 | Abs-cut | No numbers | Closest method from text to a proposed node |
| Liu et al. K-12 [arXiv 2607.28889](https://arxiv.org/abs/2607.28889) | 30 Jul 2026 | Not stated | Abs | A | Human coders added five codes the LLM phases missed |
| CoDA [2026.emnlp-demos.53](https://aclanthology.org/2026.emnlp-demos.53/), [arXiv 2610.24157](https://arxiv.org/abs/2610.24157) | 16 Oct 2026 | Oct 2026, Demos | Abs | A | Reference for a review interface |
| Luo et al. | Not stated | 2025, **OLD**, World Bank WP 11248 | Abs (M) | A | 36 aid areas not tracked in the OECD taxonomy; no link in the notes |
| Bossler et al. [DOI](https://doi.org/10.21203/rs.3.rs-9588669/v1) | 13 May 2026 | Not stated | Abs (M) | A | Vocabulary fragmentation |
| Raymond et al. [arXiv 2609.09381](https://arxiv.org/abs/2609.09381) | 8 Sep 2026 | Not stated | Abs (M) | No numbers | Target vocabulary lacks granularity |
| BEATS [arXiv 2606.04909](https://arxiv.org/abs/2606.04909) | 3 Jun 2026 | SIGIR 2026 Industry | Abs-cut | No numbers | Iterative review rounds |
| TLSA [2026.acl-long.869](https://aclanthology.org/2026.acl-long.869/) | Not looked up | Jul 2026 | Abs-cut | No numbers | Forced renaming of overlapping clusters |
| SC-Taxo [arXiv 2605.00620](https://arxiv.org/abs/2605.00620) | 1 May 2026 | Not stated | Abs-cut | No numbers | Consistency between levels |
| VITC [2026.acl-long.275](https://aclanthology.org/2026.acl-long.275/) | Not looked up | Jul 2026 | Abs-cut | No numbers | Low relevance |
| DeLFGCD [2026.eacl-long.358](https://aclanthology.org/2026.eacl-long.358/) | Not looked up | Mar 2026, **OLD** | Abs-cut | No numbers | Category discovery |
| SkiLLens [2026.eacl-industry.65](https://aclanthology.org/2026.eacl-industry.65/) | Not looked up | Mar 2026, **OLD** | Abs | No numbers | Multilingual novel-skill detection linked to ESCO |
| GRACE [arXiv 2609.04442](https://arxiv.org/abs/2609.04442) | 3 Sep 2026 | Not stated | Abs-cut | No numbers | Costed review rule |
| From DQI to C-DQI [arXiv 2610.27937](https://arxiv.org/abs/2610.27937) | 20 Oct 2026 | Not stated | Abs | No numbers | Inductive schemas gain inter-model agreement |
| Duraj et al. [arXiv 2601.00510](https://arxiv.org/abs/2601.00510) | 1 Jan 2026, **OLD** | Not stated | Abs | No numbers | Tree-search scoring can flag tree issues |

### Languages

| Paper | First posted | Published | Read | Numbers | Note |
|---|---|---|---|---|---|
| Henriksen [DOI](https://doi.org/10.5117/ccr2026.1.11.henr) | Not stated | 16 Sep 2026 per Crossref; 20 Aug 2026 per OpenAlex | Abs (OpenAlex; publisher 403) | A | Danish, Swedish, German; English translations beat Danish and Swedish setups |
| Thellmann et al. [2026.acl-long.1916](https://aclanthology.org/2026.acl-long.1916/), [arXiv 2605.24904](https://arxiv.org/abs/2605.24904) | 24 May 2026 | Jul 2026, ACL | Abs | No numbers | Translation errors drop accuracy |
| Al Ali et al. [arXiv 2608.03446](https://arxiv.org/abs/2608.03446) | 4 Aug 2026, v2 24 Aug | Accepted to EMNLP 2026 | Abs | No numbers beyond 27 scores | Alignment with English predicts performance |
| Ben Saad and Madiouni [DOI](https://doi.org/10.1109/acit69831.2026.11700134) | Not stated | Sep 2026, ACIT | Abs (OpenAlex) | A | Low-tier macro-F1 0.4632 to 0.6828 by translating; languages not stated |
| Mahmood et al. [arXiv 2608.22432](https://arxiv.org/abs/2608.22432) | 23 Aug 2026, v2 6 Sep | Not stated | Abs | A | Top model alternates across eight languages |
| Chung and Freienthal [2026.mme-main.8](https://aclanthology.org/2026.mme-main.8/) | 2 Feb 2026, **OLD** | Mar 2026 | Abs | No numbers | Rank inversions across Finno-Ugric languages |
| Beręsewicz et al. [arXiv 2411.03779](https://arxiv.org/abs/2411.03779) | 6 Nov 2024, **OLD** | Not stated | Abs-cut | Translation claim S only, unconfirmed | Narrows 1 digit to 6 digits |
| Nasuto et al. [arXiv 2508.06435](https://arxiv.org/abs/2508.06435) | 8 Aug 2025, **OLD** | Not stated | Abs | Translation claim S only, unconfirmed | Topic transfers across languages |
| BioELX [2026.findings-emnlp.1109](https://aclanthology.org/2026.findings-emnlp.1109/), [arXiv 2605.27380](https://arxiv.org/abs/2605.27380) | 9 Apr 2026, **PRE** | Oct 2026 | Part (M) | X | EN 91.2, ES 69.6; linking, not tiers |
| LongBEL [arXiv 2605.13451](https://arxiv.org/abs/2605.13451) | 13 May 2026 | No venue stated | Full via fetch tool (M) | F | FR 77.6, EN 80.0 |
| MDPO [2026.findings-emnlp.1873](https://aclanthology.org/2026.findings-emnlp.1873/), [arXiv 2609.07379](https://arxiv.org/abs/2609.07379) | 10 Sep 2026 | Oct 2026 | Part (M) | X | OCR-noisy historical text |
| MOSAIC radiology [2026.clinicalnlp-1.35](https://aclanthology.org/2026.clinicalnlp-1.35/) | Not looked up | May 2026 | Abs (M) | Not used | Multilingual reports |
| Volkov [2026.nlp4pi-1.7](https://aclanthology.org/2026.nlp4pi-1.7/) | No preprint found | Oct 2026 | Part (M) | X | Assignment disagreement en-es 33.0% |
| Fallatah [PDF](https://om.ontologymatching.org/2026/papers/om2026-STpaper3.pdf) | Not stated | OM-2026 | Part (M) | X | Full list with no real "none" path forces wrong picks |
| AIDA, Jones et al. [DOI](https://doi.org/10.4324/9781003799139-20) | None found | 18 Aug 2026, Routledge | Part (D) | X | 7,000+ UNDP evaluation reports; taxonomy tags in version 1; F1 0.85 belongs to the older version |

## Unread papers to obtain

**Nothing in this first table has been read, beyond what the status column says.** The second table lists abstract-only papers where the full text would change a decision.

| Paper | Identifier | Status | Why it may matter |
|---|---|---|---|
| *When Constraints Go Both Ways*, Di Palma et al., IJCNN Jun 2026 | [10.1109/ijcnn69854.2026.11730020](https://doi.org/10.1109/ijcnn69854.2026.11730020) | Meta, no abstract available | Likely the most direct work on the logical side of hierarchy (H) |
| *Evaluating robustness of hierarchical and multi-label requirements classification*, Nam Nguyen Van, Moataz Ahmed | [10.1016/j.infsof.2026.108212](https://doi.org/10.1016/j.infsof.2026.108212) | Meta | Robustness across hierarchy |
| *Top-Down Hyperbolic Case Retrieval for Few-Shot Hierarchical Ordinal Text Classification*, Wijayasekara et al., 1 Aug 2026 | [10.1007/978-3-032-33865-5_15](https://doi.org/10.1007/978-3-032-33865-5_15) | Meta; OpenAlex lists an accepted manuscript | Few-shot top-down |
| *A large language model-augmented hierarchical text classification approach for intelligent failure data management*, Ding et al. | [10.1016/j.engappai.2026.115750](https://doi.org/10.1016/j.engappai.2026.115750) | Meta | LLM-augmented hierarchy |
| *Relative contrastive pretraining of heterogeneous graph aware language model*, Fang et al. | [10.1016/j.knosys.2026.115711](https://doi.org/10.1016/j.knosys.2026.115711) | Meta; OpenAlex links an SSRN version | Zero-shot multi-label |
| *JCLDE* | [10.1016/j.knosys.2026.116155](https://doi.org/10.1016/j.knosys.2026.116155) | Title only | Hierarchical contrastive model |
| *L-SIMCL* | [10.1016/j.ipm.2026.104818](https://doi.org/10.1016/j.ipm.2026.104818) | Title only | Hierarchical contrastive model |
| *Generative LLM for adaptive taxonomy-informed multi-label classification in construction safety*, Shuang et al. | [10.1016/j.autcon.2026.107297](https://doi.org/10.1016/j.autcon.2026.107297) | Meta, no abstract available | Taxonomy-informed multi-label |
| *When to Trust: Margin-Calibrated Routing Between LLM-as-a-Judge and Human Experts*, Zhang et al. | [2026.emnlp-industry.201](https://aclanthology.org/2026.emnlp-industry.201/) | Meta, no abstract retrieved | Routing to humans |
| *Speak Beyond English*, Banerjee et al., SIGIR, 19 Jul 2026 per Crossref | [10.1145/3805712.3808528](https://doi.org/10.1145/3805712.3808528) | Meta | Multilingual prompts |
| *Conformal Prediction for Statistically Reliable LLM-as-a-Judge in Recommender Systems*, Amendola et al., CIKM, Crossref 6 Nov 2026 | [10.1145/3799682.3839966](https://doi.org/10.1145/3799682.3839966) | Meta | Judges and conformal sets |
| *Large Language Models Do Multi-Label Classification Differently*, EMNLP 2025 (**OLD**) | [2025.emnlp-main.126](https://aclanthology.org/2025.emnlp-main.126/) | Title confirmed only | Over-assignment |
| Argchestrators, Prompteam and HybridArguer, UZH task teams | [2026.argmining-1.13](https://aclanthology.org/2026.argmining-1.13/), [2026.argmining-1.14](https://aclanthology.org/2026.argmining-1.14/), [2026.argmining-1.17](https://aclanthology.org/2026.argmining-1.17/) | Titles and metadata only (M) | UN text; Prompteam was 5th on F1 and 1st on the judge |
| *Evaluation measures for hierarchical classification*, Kosmopoulos et al. (**OLD**) | [10.1007/s10618-014-0382-x](https://doi.org/10.1007/s10618-014-0382-x) | Crossref confirmed, not read | Origin of hierarchical measures |
| *CoPHE*, Falis et al., EMNLP 2021 (**OLD**) | [10.18653/v1/2021.emnlp-main.69](https://doi.org/10.18653/v1/2021.emnlp-main.69) | Crossref confirmed, not read | Count-preserving hierarchical metric |
| Kekäläinen and Järvelin, JASIST 2002 (**OLD**) | [10.1002/asi.10137](https://doi.org/10.1002/asi.10137) | Crossref confirmed, not read | Origin of the generalised precision Kähler uses |
| *Prediction-powered inference*, Angelopoulos et al., Science 2023 (**OLD**) | [10.1126/science.adi6000](https://doi.org/10.1126/science.adi6000) | Crossref confirmed, not read | Origin of the correction method |
| TELEClass, Zhang et al. (**OLD**) | [arXiv 2403.00165](https://arxiv.org/abs/2403.00165) | Title and first lines; v3 stamp 5 Feb 2025; v1 not checked | Top-down candidate search that newer work compares against |
| XR-Transformer origin, Zhang et al. 2021 (**OLD**) | No identifier in the notes | Not read; known through Kähler | Kähler is brief on the method |
| Unnamed single-pass LLM paper taking the top three classes per level | [PDF](https://payberah.github.io/files/download/papers/llm_classification.pdf) | Snip | Possible LLM beam search |
| Non-LLM study with beam size 5 | [arXiv 2005.02473](https://arxiv.org/abs/2005.02473) | Snip | Beam search baseline |
| *Beyond Majority Voting*, Ai et al. (**OLD**; October 2025 and ICML 2026 rest on S) | [arXiv 2510.01499](https://arxiv.org/abs/2510.01499) | Snip; ID from TEF's reference list | Weighted aggregation of models |
| *Operationalizing Document AI*, Fehlis et al., first posted 12 May 2026 | [arXiv 2605.18818](https://arxiv.org/abs/2605.18818) | Abstract read; its numbers are S only | Page classification before the LLM |
| *Evaluation of Embedding-Based and Generative Methods for LLM-Driven Document Classification*, April 2026 by search result (**OLD**) | [arXiv 2604.04997](https://arxiv.org/abs/2604.04997) | First page only, Snip | Embedding against generative |
| *Ex-ante Identification of Disruptive Technologies*, Ding and Lu | [10.21203/rs.3.rs-9585548/v1](https://doi.org/10.21203/rs.3.rs-9585548/v1) | Abs, partly | Rolls patent evidence up to themes |
| *Hierarchical Compositional Hypergraphs*, Gordji and Berahman | [10.21203/rs.3.rs-10829259/v1](https://doi.org/10.21203/rs.3.rs-10829259/v1) | Abs | Small gain on 20 Newsgroups |
| LLM Cache Bandits; Continuous Semantic Caching; Semantic Caching (INFOCOM); Prompt Context Caching Architecture | [2026.emnlp-main.1849](https://aclanthology.org/2026.emnlp-main.1849/), [10.1145/3842721.3849656](https://doi.org/10.1145/3842721.3849656), [10.1109/infocom59046.2026.11571467](https://doi.org/10.1109/infocom59046.2026.11571467), [10.17762/ijisae.v14i1s.8385](https://doi.org/10.17762/ijisae.v14i1s.8385) | Meta, except the last, which is Abs with no measured result in the part read | Boilerplate may repeat across reports |
| GenGate; CascadeDebate; Larch; Routing Should Pay for Itself; CRE-Router; Cost-Aware LLM Invocation Strategies | [2026.emnlp-main.2071](https://aclanthology.org/2026.emnlp-main.2071/), [2026.acl-industry.93](https://aclanthology.org/2026.acl-industry.93/), [arXiv 2606.07923](https://arxiv.org/abs/2606.07923), [arXiv 2609.37402](https://arxiv.org/abs/2609.37402), [arXiv 2606.27457](https://arxiv.org/abs/2606.27457), [10.1109/icarai70085.2026.11635457](https://doi.org/10.1109/icarai70085.2026.11635457) | Abs only, the last partly | Routing alternatives |

| Abstract-only paper worth reading in full | Identifier | The question it would settle |
|---|---|---|
| Hauss et al. (ACM returned 403) | [10.1145/3820755.3832808](https://doi.org/10.1145/3820755.3832808) | How large is the flat advantage on hierarchical data? The abstract gives no numbers |
| Who Pays | [arXiv 2610.23249](https://arxiv.org/abs/2610.23249) | Which four languages, and whether French and Spanish behave like English |
| Bouchard | [arXiv 2605.06350](https://arxiv.org/abs/2605.06350) | Whether two tiers hold for many-label tasks |
| RelaClass | [10.1109/ijcnn69854.2026.11730021](https://doi.org/10.1109/ijcnn69854.2026.11730021) | Numbers for class-name-only hierarchy methods |
| Alangari et al. (record truncated) | [10.3390/electronics15142988](https://doi.org/10.3390/electronics15142988) | Full tier comparison |
| Rahgozar and Mortezaagha | [10.1007/s10462-026-11599-2](https://doi.org/10.1007/s10462-026-11599-2) | Details of the closest in-window chunk-to-document design |
| Henriksen (publisher returned 403) | [10.5117/ccr2026.1.11.henr](https://doi.org/10.5117/ccr2026.1.11.henr) | Source language against translated setups |
| Deterministic Agentic Workflow | [arXiv 2605.14857](https://arxiv.org/abs/2605.14857) | Whether a fixed flow suits a rule-heavy taxonomy |
| HiPP | [arXiv 2605.13663](https://arxiv.org/abs/2605.13663) | Whether bottom-up aggregation helps on ambiguous taxonomies |

## What the notes did not find

These are search results, not claims about the literature. Each row means these searches did not surface the item. One lesson on sensitivity: TEF surfaced only when an unquoted query used different words, after phrase queries had missed it (C), so other papers may have been missed the same way.

| Not found | Nearest evidence |
|---|---|
| Any technique applied to UN evaluation reports, or to DAC or UNBIS tiers, within the window (H) | UZH on UNESCO resolutions; AIDA on UNDP evaluations (see disagreements) |
| Flat, top-down and whole-path compared with LLMs on one dataset with hundreds to thousands of multi-label leaves (H) | Yoshimura and Kashima, at small scale and older |
| Top-down against flat on long report chunks, or chunk-level against document-level labels (H) | Tashkees-AI; XplaiNLP; both on short text |
| LLM methods against trained extreme models on a tiered taxonomy of hundreds to low thousands of leaves in general text (H) | Most tests are ICD (tens of thousands) or small benchmarks (under 150) |
| A named LLM beam search over a supplied taxonomy with a reported comparison (H) | Omikuji's beam search in a trained tree; two snippets, unverified |
| Recall on secondary branches, the central question for multi-topic chunks (H) | Subset walk and leaf roll-up as designs only |
| Recall@K by label frequency on a large document taxonomy (C) | SNOMED CT Concept Recommendation; UZH across teams |
| A cascade on a taxonomy of hundreds to thousands of leaves at chunk level (C) | Tests mostly 2 to 150 classes |
| How noisy PDF parsing changes escalation rates or tagging quality, in a controlled test (C, E, H) | HIRA and RAPT use OCR text but do not vary noise |
| A classifier trained on LLM silver labels for hundreds or thousands of classes, evaluated against human gold (C) | Kasliwal and Singh at 112 topics (F) |
| Active learning at large taxonomy scale or on long reports (C) | Who Gets Annotated? on crisis text |
| Rollup against human document labels on long reports, with hundreds of labels, for multi-label output, or with map-reduce or section weights (C) | TEF on posts; Rodriguez et al. on 10 documents |
| Summary, first pages and all chunks compared for multi-label classification of long reports (C) | Task Cascades against an oracle LLM; HyTRAM without that comparison |
| A real bill on a corpus of millions of documents (C) | AutoRelAnnotator gives scale without dollars |
| In-prompt batching effects on chunk-level multi-label accuracy (C) | PromptPack, on downstream AUC |
| Hierarchical F1, tree distance, similarity-matrix credit and graded judgement compared on one dataset (E) | MAX-EVAL-11's 50-case pilot |
| Any paper confirmed as building on Kähler et al. (E, H) | Topic neighbours only; no citation data was available. Two later papers checked by text search do not cite it: TAGHN and [SHELF, arXiv 2609.03047](https://arxiv.org/abs/2609.03047) |
| LLM judge reliability on tier-level tagging of long policy documents; sample-size guidance for per-tier estimates (E) | Judge papers on free text, translation, entity pairs |
| The effect of requiring quotes with everything else fixed; a user study of quoted evidence on taxonomy labels (E) | CTI2TTP ablation; POINTERS, confounded |
| Conformal prediction over a label hierarchy applied to LLM text tagging (E) | HCC with BERT; HG-CRC on group hierarchies. Most hits use "hierarchical" in other senses, such as [grouped data, arXiv 2608.15500](https://arxiv.org/abs/2608.15500) |
| How often LLMs choose "none of the above" correctly across a large taxonomy; over-assignment measured across taxonomy size (E) | ATT&CK negatives; Choo and Hong on output format |
| End-to-end loop from unplaced text to reviewed new nodes, with precision for proposed nodes; a dedicated method for unused labels (E) | UZH, TaxCE, ConceptE, ReLTEx as pieces |
| A tier-based tagger tested across English, French and Spanish; translate-then-tag against source-language tagging on a taxonomy, with numbers (E) | He et al. for tiers; Henriksen for translated input |

## Researchers' proposals and estimates are untested

**Nothing in this section is a finding.** Each row is a researcher's own proposal or arithmetic, and the notes found no paper reporting it for this corpus.

| Untested hypothesis | Assumption or basis | Note |
|---|---|---|
| A three-way race on one chunk (flat with roll-up, tree walk, MATCH-style funnel) shows where methods disagree | Panels draw on papers tested on other corpora | H |
| A tier waterfall, marking how many final errors began at each tier, would be a real contribution | The measure is rarely reported | H |
| A scorecard with exact, parent and top-tier match per language would show most errors as fine-tier | Holds in AraGenre and LCSHBench | E |
| A metric-flip chart using publisher tags, pipeline LLM tags and the new system would be a strong opening | The sample supports it | E |
| Kähler's reversal is partly usefulness against rules and partly operating point; a paired re-rating at the same k would separate them | Shared documents | E |
| A flat extreme classifier trained on publisher tags is a fair baseline | Publisher tags exist for part of the corpus; Vasquez warns operational labels can flatter | H, E |
| K of about 15 to 40 from a hybrid retriever is a workable start | Evidence is from far smaller taxonomies, so measure recall by frequency and language first | C |
| Per-branch quotas or confusion partners protect recall cheaply | UZH's caps come from a different pipeline | C |
| Retrieve, prune and rerank on a sample, calling a tree walk only on low-margin cases | Selective calling is supported in entity linking, not this task | H |
| Two tiers suffice, with thresholds per branch or per language | Scope-aware routing and Who Pays were not tested together | C |
| Multilingual embeddings such as BGE-M3 suit French and Spanish | Evidence is Arabic text with English definitions | H |
| A deterministic check that each quote appears in the chunk catches fabricated evidence cheaply, but exact matching will reject true quotes broken by PDF parsing | Measure the failure rate first | E |
| Showing quotes in the source language with an English gloss is more faithful and useful | Banerjee et al., on other tasks | E |
| Accept a leaf above a calibrated threshold, otherwise emit the parent if the children's combined score clears, otherwise abstain | A plain cousin of HCC and HG-CRC | E |
| Repeat-run stability or per-label yes/no scores can be calibrated | Whether that is exchangeable enough for conformal validity is unknown | E |
| "None fits" plus negative chunks in the test set is the cheapest measure of over-assignment | Modelled on the ATT&CK dataset | E |
| Rollup experiment: all chunks with LLM labels, then any-chunk, at least n chunks, share above a threshold, TEF log-odds and calibrated noisy-OR; embedding mean; first pages or executive summary; random 20 and 48 chunks; a taxonomy-aware summary | Reported by label frequency, tier, language and cost. TEF per label is an extension | C |
| Any-chunk false positives grow with 146 chunks per report and thousands of labels; log-odds sums pile up small evidence on long reports | A normalising step or a threshold tuned on gold may be needed | C |
| Section weighting suits evaluation reports' predictable sections | Nothing evaluated | C |
| Silver labels only for labels above an agreement threshold, with gold kept out of training and selection | No evidence for what the threshold should be; the CVE failure at 0.39 is a data point, not a bound | C |
| Publisher and LLM tags as separate silver tracks against one blind reviewed sample | Method from Vasquez | E |
| Prediction-powered estimates work per tier and per language | Stratified sample; tests covered neither tiers nor large label sets | E |
| A gap report from abstained chunks, low nearest-leaf similarity and ConceptE-style cluster names, with ReLTEx-style validation | UZH's 11% is only a rough prior, from selected paragraphs of another corpus | E |
| A paired test of source, machine translation and English taxonomy on the same French and Spanish chunks | Leaf counts will be small, so sample sizes need planning | E |
| Stored chunk labels make re-tagging cheap; a NO on a branch can settle its children | Not tested on taxonomies | C |

| Estimate (not a finding) | Figure | Assumptions | Note |
|---|---|---|---|
| KI-FSPrompt, LLM-Ensemble and XR-Transformer sequentially over 1.9 million items | About 6 days, about 22 days, about 5.5 hours | Kähler's **titles** timings, so chunk time is understated; hardware differs by method | H, M |
| MATCH's rate over 1.9 million chunks | About 1,000 hours, or about 71 days at 14 hours a day | Chunks may be longer than abstracts; hardware not read | H |
| ToT-ICD token rate over 1.9 million chunks | About 7.8 × 10^11 tokens | Its inputs are full medical records | H |
| Strong LLM on every chunk | About $7,800 | SLKR's $4.111 per 1k; printed prices unchecked against provider lists | C |
| Same, at Abusaqer and Saquer's standalone Sonnet price | About $8,700 | $96.29 for 20,997 posts | C |
| Small LLM reader on every chunk | About $110 | SLKR's $0.0596 per 1k; BANKING77 queries are probably shorter, so perhaps 1.2 to 2 times higher (a guess) | C |
| Two tiers at 5%, 10% and 20% escalation | About $500, $900 and $1,700 | Escalation rates are assumed, not measured | C |
| Cascade at Abusaqer and Saquer's rate | About $2,400 | $26 for 20,997 posts | C |
| Mini model via batch API | About $1,900 | Belenki et al., about $1.0 per 1k | C |
| Silver labelling 20,000 chunks with a strong LLM | About $80 | $4.111 per 1k | C |
| Label-knowledge build for 2,000 labels | $90 to $2,300 once | $3.40 for 77 labels, scaled linearly or quadratically | C |
| Reading 5, 20 or 48 chunks per report | 65,000, 260,000 or at most 624,000 calls (3.4%, 13.7%, 32.8%) | 13,000 reports; whether quality holds is the open question | C |

All dollar rows exclude embedding and retrieval, and apply no batch or cache discount unless stated. Savings come from different datasets and cannot be multiplied together. The corpus figures of about 1.9 million chunks and about 13,000 reports come from the notes.

## The new notes correct the earlier research

**Chunk-then-aggregate.** The earlier research recorded no chunk-then-aggregate test against a document-level gold standard. **That statement no longer holds.** TEF compares direct prediction, majority vote, soft vote and weighted log-odds on 8,358 human-labelled posts, and Rodriguez et al. tests any-chunk rollup on 10 documents against human raters. What stays not found is the same test on long reports, with hundreds of labels, for multi-label output, or with section weights.

**Kähler's speed.** The earlier research printed 10.5 ms against 998 ms per record beside full-text scores, and from them about 22 days against about 5.5 hours. **Those timings are for titles only**, on differing hardware, so those figures understate chunk time. Kähler's numbers there also passed through a summarising fetch tool. They are now read from extracted text and the quoted rows match.

**Kähler's reach.** The earlier research called it the most useful single paper on the design choice. That holds for trained models against LLMs. It does not hold for tiers, which the paper does not use.

**Large vocabularies.** The earlier research said the largest measured success was 419 classes. That figure comes from ontology-guided extraction and should not be read across to tagging, where MATCH reports expert-reviewed precision against 113,892 Concepts and Kähler scores 204,056 entities.

**The UN resolutions result.** The earlier research linked POINTERS' 0.459 to quoting plus no pre-filter. The new notes say plainly that these are confounded, so neither can claim the gain.

**Deeper reads.** Constraint-Aware Search had no numbers in the portion first read and now has them. Thakaa, Reference-Free Evaluation, Circuits of LLM Classifiers and Graph-Constrained Policy were also read further, and the Thakaa reading narrows what its correction step can claim.

## Where the notes disagree

| Topic | What the notes say | Status here |
|---|---|---|
| What "FULL" means | H means the whole body. E uses it for named sections read directly, and says it did not read every line | All E entries are shown as Part |
| Kähler table numbers | H cites Tables 4, 5, 6; E cites Tables 3, 4, 5 for the same scores | Values match; table numbers unresolved |
| Kähler's formula | E flags g-Rec as rendered with fp where it expected fn, and a printed CI whose upper bound equals the point estimate (0.601) | Check the PDF before reproducing |
| Kluge and Kähler 2025 | H read it in part. E says it is known only through LCSHBench | H's read stands; E's fourth-on-F1, first-on-rating description stays unconfirmed |
| Hauss first-posted date | M's title search found no preprint; H says none checked | Both shown |
| Graph-Constrained Policy | C, from the abstract, reports 0.527 and +0.157 macro. H's table reading warns the micro headline compares unequal data | H's caution governs |
| Thakaa | H ties 0.7139 to 0.7352 to correction "after later steps" and calls 394 to 119 a tellable story. E says later post-processing means the gain is not this step alone, and the authors call agreement a conservatism device. C and M say the full-taxonomy prompt won | Use the topic-trap story, not the score gain |
| AraGenre size | E says 17 ranked entries; M says first of 18 teams | Unresolved |
| TEF's overhead | C's summary says about three times the token cost. Its detail gives 2.7M against 0.5M tokens and $0.27-0.28 against $0.09 on Qwen2.5-7B | Three times holds for money, not tokens |
| UZH vocabulary | The overview gives 141 tags; POINTERS describes 126 codes | Unresolved |
| "No UN application" | H says no in-window paper applies these patterns to UN, development or evaluation reports. UZH is UN text with a supplied taxonomy, and AIDA (D) reports taxonomy tags on UNDP evaluations, from its older version | H's statement is too broad for those two, but neither tests these tier patterns on evaluation chunks |
| Should you narrow candidates? | UZH and RESOLVENOW suggest harm; Foroutan, SLKR and Confusion to Clarity show benefit; XplaiNLP won with no narrowing described | Open; measure candidate recall |
| Is a small LLM reader worth it? | SLKR and Confusion to Clarity show gains. Sun and Sang found small-LLM rerank below plain retrieval. Selection Effect found no compact verifier improved the frontier | Open; differs by task and by what the reader is given |
| Separate or joint label questions | dLLM-SetScore and LinkedIn favour one label at a time. Khatri et al. found yes/no decomposition lowered accuracy for every model | Open; different tasks |
| In-prompt batching | PromptPack finds AUC unchanged and a worst degradation of 0.012. Reiter et al. find grouping six tweets lowered mean confidence by 660 basis points | Do not mix batched and single-prompt confidences (C's inference) |
| Silver labels | Tweets and HP show near-parity. Bonhomme and Dulaunoy find no gain at about 0.39 agreement and harm at about 1,000 items | Depends on measured agreement |
| Human or LLM primacy | Nakamura et al. argue human labels should not automatically take precedence. Mukherjee et al. and Han and Kim show consensus is not alignment | They differ in task and in what they measure |
| Prompt optimisers | Confusion to Clarity finds MIPROv2 and GEPA below zero-shot. ERGO finds exploration can win on many-class intent | Not general either way |
| Best demo | H picks a three-way race. C picks label cards, the cascade and the chunk-to-document test. E picks a tier-by-language scorecard plus a gap report | My five combine these; the notes found no paper measuring viewer appeal |

## Search coverage limits every "not found"

ACL Anthology event pages for ACL, EMNLP, EACL, LREC and their Findings and workshops worked, as did DataCite for arXiv DOIs, Crossref title queries and OpenAlex lookup by DOI. The arXiv API and advanced search returned 429 errors for all three researchers. H completed 15 of 56 planned advanced-search queries and stopped, C waited and eventually ran its planned queries, and E completed 42 phrase queries by API and about ten on the web page. Semantic Scholar returned 429 throughout for E, so no citation graph was available, including for Kähler. KONVENS 2026 and NAACL 2026 event pages returned 404, so GermEval 2026 was not checked. SIGIR, CIKM, KDD, SIGMOD, VLDB, ICML, NeurIPS, JCDL and TPDL 2026 were not enumerated directly and are covered only through arXiv and Crossref titles. AACL 2026 was only partly covered. Keyword filters on titles miss other wordings, and some figures were seen only as captions. No bot check was met or bypassed. The lists are verified but not exhaustive.

## Conclusion

The research shifts the demo's question from "should we go top-down?" to "at which tier does each method lose the chunk, and what does recovery cost?". Kähler gives the honest frame for comparing trained models with LLMs and for judging usefulness rather than agreement, but the tier chart has to be built, because reports of where errors begin in the tree are rare. The most interesting tension is that validity and accuracy come apart: enforcing the tree after the fact added only 0.38 points in the one ablation, while a separate evidence check moved precision by far more on its one test. That suggests spending effort on checking labels against text and treating tree validity as a cheap guarantee, as a hypothesis to test on your own chunks.

The strongest original contribution sits at the edge of what was found: a report-level human test of rollup rules on a tiered taxonomy across English, French and Spanish. A small, blind, stratified gold set is the prerequisite for all five techniques, and the literature cannot supply your corpus's labels.

In plain terms: I recommend showing five techniques, and presenting them as experiments on your reports rather than proven methods, because none was found tested on UN evaluation reports and Kähler says nothing about tiers. The cost figures and pipelines are researchers' untested estimates. What you need to decide is which of the five to build first, and whether you can fund human labelling of a small sample, since every result depends on it.
