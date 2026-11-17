# Published benchmarks on HumSet and EuroVoc, and recent ways to beat them

Researched 6 November 2026. Every figure below comes from the paper named. "Ours" figures come from a scratch script on the
repository's own models and are not yet in a notebook, so treat them as provisional until one prints them.

## HumSet: the bar is HumBERT, with a question over how it averages

[HumSet](https://arxiv.org/abs/2210.04573) (Fekih et al., Findings of EMNLP 2022). Read: HTML to the end of the references;
appendix tables only through a summary. It states "We report macro-averages of the metrics", with a threshold tuned per label on
the validation set and scores given per category. 148,621 entries split 80/10/10; 46 projects. It gives no inter-annotator figure
but warns of annotator subjectivity, and says 33 tags appear in under 2% of the data.

[HumBERT](https://arxiv.org/abs/2305.16756) (Tamagnone et al., arXiv v2 30 May 2023; a search result lists IJCAI 2023, which was
not opened). Read in full. HumBERT is XLM-RoBERTa further trained on about 2B tokens of humanitarian text, then fine-tuned on
HumSet with the last layer split by category. Input is capped at 200 tokens, for 3 epochs at batch size 8. The pretrained model is
public as `nlp-thedeep/humbert`.

| Category | HumSet paper, XLM-R | HumBERT paper, XLM-R Base | HumBERT, best | Ours, XR-Transformer, macro | Ours, XR-Transformer, micro |
|---|---|---|---|---|---|
| Sectors | 0.73 | 0.76 | 0.78 | 0.734 | 0.746 |
| Pillars 1D | 0.53 | 0.68 | 0.70 | 0.519 | 0.585 |
| Subpillars 1D | 0.38 | 0.49 | 0.54 | 0.440 | 0.510 |
| Pillars 2D | 0.63 | 0.68 | 0.71 | 0.617 | 0.655 |
| Subpillars 2D | 0.40 | 0.47 | 0.51 | 0.446 | 0.557 |

**Caution on comparing these.** HumBERT's Table 5 caption does not say macro or micro. Its XLM-R Base row is well above the HumSet
paper's XLM-R on the same data (0.68 against 0.53 on Pillars 1D), and its "Avg" column (0.73 for the best row) does not equal the
mean of that row (about 0.65), so the column may be something else. Its zero-shot table does say macro, and there the average does
match. So the safe bar is the printed HumBERT row, without assuming its averaging. Ours uses thresholds tuned per label for macro
F1 in both columns. Our pillar scores are rolled up from subpillars rather than trained directly.

### What HumBERT's own code shows (checked 11 November 2026)

HumBERT's paper is the same as IJCAI 2023's "Leveraging Domain Knowledge for Inclusive and Bias-aware Humanitarian
Response Entry Classification", and its code is public:
[the-deep-nlp/bias-aware-humanitarian-entry-classification](https://github.com/the-deep-nlp/bias-aware-humanitarian-entry-classification)
(last commit 30 May 2023), with a packaged trainer in
[the-deep-nlp/classifier-trainer](https://github.com/the-deep-nlp/classifier-trainer). Hugging Face holds only the
pretrained `nlp-thedeep/humbert`, `nlp-thedeep/sentenceHumbert` and the data; no fine-tuned HumSet classifier is
released. Read from the code:

- **Data.** `models_training.ipynb` calls `load_dataset('nlp-thedeep/humset')` with no version. Version 2.0.0 has been
  the default since 6 February 2023, so the run most likely used it. Its test set has 12,896 entries; version 1.0.0,
  which this repository uses, has 15,147. Only 1,252 entry ids are in both, and 78% of version 2's test excerpts
  (10,010 by id, 10,051 by exact text) are in version 1's training split. So HumBERT was very probably scored on a
  different test set from ours, and our models cannot be scored on its test set without leakage.
- **Scoring.** F1 per label at a threshold per label chosen on validation (the best smoothed F1 whose precision and
  recall differ by under 0.4), then averaged: a plain mean for sectors and pillars, and for subpillars a mean of
  each pillar's mean. "Avg" is the mean of the three tier-1 category scores (sectors, pillars 1D and 2D), which
  is why it does not equal the row's mean: HumBERT's best row gives (0.78 + 0.70 + 0.71) / 3 = 0.73.
  The test set is not filtered by project, although a docstring says it should be.
- **Training.** HumBERT with the last transformer layer copied once per category and a head on each; focal loss
  (gamma 1, label weights from label frequency, power 0.5); learning rate 3e-5; 3 epochs; batch 8; 200 tokens; one
  frozen layer. Ours: one shared encoder, plain binary cross-entropy, learning rate 2e-5 with 1e-3 for the head,
  batch 32, 256 tokens.

Their averaging would not explain the gap: applied to our combined method's own test rankings, with thresholds set
on the test items for both averages, the mean of pillar means is 0.038 lower than the flat mean on Subpillars 1D and
0.015 lower on Subpillars 2D (scratch script, not a notebook). What remains, untested: the different test set, and
their training recipe.

**Rerun (notebook 13, 12 November 2026).** The authors' packaged trainer, unchanged, with the paper
notebook's settings, on HumSet 2.0.0's own splits (129,268 train, 11,940 validation, 12,896 test), one run, one
A10G, 1.2 hours, $1.20. Its own table gives Sectors 0.779, Pillars 1D 0.688, Subpillars 1D 0.499, Pillars 2D
0.704, Subpillars 2D 0.485 (mean 0.631; tier-1 Avg 0.724 against 0.73 printed). Averaging every label once
instead gives 0.778, 0.685, 0.548, 0.705, 0.501 (mean 0.643), within 0.015 of every printed figure (0.78, 0.70,
0.54, 0.71, 0.51; mean 0.648). So whether it reproduces the paper depends on the averaging, which the paper
does not state: by the trainer's own averaging Subpillars 1D falls 0.041 short, outside the 0.03 set before
the run, and notebook 13 records it as not reproducing. Hypothesis, not tested: the
printed table averaged subpillars plainly, or the May 2023 code differed from the September packaging. Version
2.0.0's test set is not ours, so none of this says how HumBERT's recipe does on version 1.0.0.

**Rerun on version 1.0.0, our test set (notebook 13).** The same trainer and settings on version 1.0.0's splits
(117,435 train, 16,039 validation, 15,147 test), one run, 1.1 hours, $1.13. Averaging every label once: Sectors
0.734, Pillars 1D 0.532, Subpillars 1D 0.428, Pillars 2D 0.640, Subpillars 2D 0.463, mean 0.560. On the same
test set our combined method scores 0.742, 0.553, 0.453, 0.646, 0.475, mean 0.574, ahead in every group by 0.006
to 0.025, and our plainer fine-tuned HumBERT scores a mean of 0.561. So the gap to the printed 0.648 comes from
the test set, not the recipe. One run each, so the gaps have no interval.

**Every method on version 2.0.0 (notebook 15, 12 November 2026).** Every HumSet model retrained on version
2.0.0's splits, with its version 1.0.0 settings. On the full 12,896 test excerpts, averaging every label once,
our combined method scores Sectors 0.794, Pillars 1D 0.700, Subpillars 1D 0.579, Pillars 2D 0.724, Subpillars 2D
0.534, mean 0.666 (95% interval 0.659 to 0.673), against the printed 0.78, 0.70, 0.54, 0.71, 0.51, mean 0.648, and
against the rerun recipe's 0.643. It is ahead of the recipe in every group, and level with print on Pillars 1D
(0.6999). Why every score is higher on 2.0.0, measured: version 1.0.0 keeps each source document's excerpts in
one split (no test excerpt shares a document with training), while 96.9% of version 2.0.0's test excerpts come
from a document that also has excerpts in training. Identical text is rare in both (1.9% and 2.3%). Hypothesis,
not tested: neighbouring excerpts of the same report in training are what lift the similar-documents model
(exact-label F1 0.59 on the 1.0.0 sample, 0.67 on the 2.0.0 sample) and the others on 2.0.0.

**Every method by the paper's measure, on the samples (notebook 16, 12 November 2026).** The 1,500-excerpt
samples of both versions scored by HumBERT's measure too: macro F1 per published group, every label once, a
threshold per label chosen on the same 4,000 development excerpts the results tables tuned on; LLM methods,
which give no scores, count every label they return. Every method's saved table score was reproduced first. The
combined method is first on both (0.583 on 1.0.0, 0.636 on 2.0.0). The order below it changes: on 1.0.0,
fine-tuned HumBERT moves from eighth by exact-label F1 to second (0.574) and the rerun recipe from seventh to
third (0.570), while similar documents and Omikuji fall from third and fourth to eighth and ninth. On the 2.0.0
sample the rerun recipe scores 0.572, against 0.643 on the full test set with its own thresholds. Why that gap,
not measured. Hypotheses, not tested: its thresholds here come from 4,000 development excerpts rather than the
full 11,940, and three labels have no excerpt in the 2.0.0 sample, so score zero for every method.

A blog post, [Enhancing HumSet with AdapterFusion](https://www.auroria.io/humset-improving-humanitarian-crisis-response-adapter-fusion-transfomers-nlp-research/)
(Filzmoser, July 2023), tries adapters on XLM-R and XtremeDistil against the HumSet paper. Its numbers are only in
images, it does not compare with HumBERT, and it links no code.

## MultiEURLEX: the only directly comparable EuroVoc figures

[MultiEURLEX](https://arxiv.org/abs/2109.00904) (Chalkidis et al., EMNLP 2021). Read: pages 1 to 8. Splits are chronological:
train 1958 to 2010, development 2010 to 2012, test 2012 to 2016. Metric is mean R-Precision (mRP): precision at k, where k is the
document's number of gold labels. Table 2 gives 7,390 labels, 4,220 in training.

| Setting, English | Train | Development | Test |
|---|---|---|---|
| BERT, all labels, random split | 99.2 | 74.7 | 74.0 |
| BERT, all labels, chronological split | 96.7 | 58.7 | 48.4 |

The same model loses 25.6 points on test when tested on later laws. Level 3 (567 labels), chronological, English: native BERT
67.7, XLM-RoBERTa 67.4.

Ours on the same 5,000 English test laws (first 500 words; no fine-tuning except where stated): Omikuji mRP 47.1 for all labels and
65.0 for level 3 rolled up; embedding classifier 45.8 and 65.4; similar documents 42.4 and 60.0; label search 13.9 and 26.8. Our
level 3 has 545 labels against the paper's 567. XR-Transformer's score is pending.

**Our own drift test, development laws only, Omikuji.** Scored on the later 2,500 development laws (2011 to 2014), training on the
55,000 training laws gives mRP 52.70; adding the earlier 2,500 development laws (2010 to 2011) gives 56.66. Splitting the same
laws at random instead showed 58.31 to 69.83, so a random split overstates this gain about threefold. Test laws run from 2012 to
2016, and this has not been run on them. One run, one model, and no interval, from a scratch script.

## Other EuroVoc figures, on easier non-chronological splits

Not comparable with MultiEURLEX's test split, and listed so nobody compares them by accident.

| Source | Result | Setting |
|---|---|---|
| [JEX](https://arxiv.org/abs/1309.5223), Steinberger et al. Read: pages 1 to 6 | English P 0.4801, R 0.5547, F1 0.5147 at rank 6; dynamic rank 0.5227. Range F1 0.44 to 0.54 across 22 languages | Ten-fold cross-validation; 41,672 English documents; 4,229 categories present; a category needs 4 or more documents |
| [PyEuroVoc](https://arxiv.org/abs/2108.01139), Avram et al., RANLP 2021. Read in full | English JEX 50.15 at F1@6, BERT 78.95 | Five random stratified splits, first 512 tokens |
| [Shaheen et al.](https://arxiv.org/abs/2010.12871), 2020. Read about 57% verbatim, results via summary | Abstract: micro-F1 0.661 on JRC-Acquis, 0.754 on EURLEX57K. A summary also gives 0.758, unverified | Stratified splits and the corpus's own split; 5.07 labels per document on EURLEX57K |
| [EU Parliament model card](https://huggingface.co/scampion/eurovoc_en). Read in full | Micro F1 0.7233 at threshold 0.34 on Eurlex57k; 7,331 labels; trained on over 200,000 documents | Training corpus and split not stated, so test overlap cannot be ruled out |
| [Bocchi and Palmero Aprosio](https://ceur-ws.org/Vol-3878/10_main_long.pdf), CLiC-it 2024. Read in full | English macro F1 0.484 title plus text, 0.474 text alone, 0.432 title only; sentence reordering no better | 4,028 labels; documents 2010 to 2022; stratified; threshold 0.5. "The best results are obtained when all the 512 tokens" are filled |
| [DualMLC](https://arxiv.org/abs/2609.12915), Ye et al., Sep 2026. Read in full | EURLex-4K P@1/3/5: DualMLC 88.82, 75.80, 62.74; MatchXML 88.12, 75.00, 62.22; XR-Transformer 87.22, 74.39, 61.69 | 3,956 labels; 15,449 train; eight RTX 4090s |

**A human limit, second-hand.** JEX reports a manual evaluation from Pouliquen et al. (2003): the top 5 automatic descriptors were
judged good 75% (English) and 82% (Spanish) of the time even when no human had assigned them, and a second annotator agreed with
the first in 74% and 84% of cases, which the authors call an upper bound. The 2003 paper itself was not read.

## What recent work offers for beating them

| Approach | Source | Result | Limits |
|---|---|---|---|
| Fine-tune a domain-pretrained encoder | HumBERT | 0.02 to 0.05 above XLM-R Base per category | See the averaging caution |
| Late fusion of a LoRA-tuned LLM with a fine-tuned BERT | DualMLC | 0.5 to 0.8 points over MatchXML | Qwen2.5-7B, 8 GPUs, batch 2 per device, no unseen labels |
| Dual encoder with training documents and label text in one memory | [RAE-XMC](https://arxiv.org/abs/2502.10615), Wang et al., 2025. Read to the appendix | P@1 86.49 on LF-Wikipedia-500K against XR-Transformer 81.62, 0.6 to 7.4 hours on 8 A100s | No legal data; unseen labels only possible by design, not tested |
| Joint dual encoder and classifier training | [UniDEC](https://arxiv.org/abs/2405.03714), Kharbanda et al., WWW 2025. Read in part | P@1 57.41 against NGAME 54.69 on LF-AmazonTitles-1.3M; a single 48GB GPU | No legal data; backbone and timings only via a summary |
| Traditional classifiers plus LLM synthetic data and LLM ranking | [Annif at GermEval-2025](https://arxiv.org/abs/2508.15877), Suominen et al. Read in part | First place, nDCG@20 0.5697 against 0.4180; LLM ranking added 0.01 to 0.03; synthetic data helped Bonsai "a lot", XTransformer less | Title and abstract input; no LLM fine-tuned |
| Fine-tuned LLM against others | [SemEval-2025 Task 5](https://arxiv.org/abs/2504.07199), D'Souza et al. Read in part | Annif first, DUTIR831 (fine-tuned Qwen2.5-72B) second on all subjects; DNB-AI (few-shot ensemble, no fine-tuning) first on the qualitative track | A leaderboard, not a controlled test; the paper does not conclude that fine-tuning paid off |

None of these reports a gain anywhere near a step change. The largest published gains here run from under 1 point to about 5.
No paper reporting newer scores on MultiEURLEX's chronological split or on HumSet after 2023 was found, though the searches were
few and incomplete, so this means only that they did not surface one.
