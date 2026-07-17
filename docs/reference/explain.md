# Explainability

Why a dialect prediction was made. Explainers are registered under canonical
names in `tulip.explain.EXPLAINERS` (`top_tfidf`, `lime`, `shap`, `attention`,
`nearest_examples`, `dialect_evidence`) and looked up by name. Heavy dependencies
(lime, shap, torch) load lazily inside each `explain` call, so importing the
package stays cheap.

## Registry

::: tulip.explain.EXPLAINERS

::: tulip.explain.get_explainer

## Dialect evidence

`dialect_evidence` attributes one prediction to named phenomena: which marker
lexemes matched and which isoglosses fired. It composes over the marker lexicon
and the phonological rules, so the evidence is resource-defined, not a claim
about what the model attended to.

::: tulip.explain.dialect_evidence.DialectEvidenceExplainer

## Global dialect evidence

`dataset_evidence` rolls those per-sample findings up over a whole labelled
corpus. For each phenomenon it reports how often it occurs, which gold dialect
its carriers belong to, and how concentrated that link is by class-conditional
lift: the probability that a carrier has a class divided by the base rate of that
class. A high-lift isogloss is one that genuinely separates a dialect. Phenomena
carried by too few samples are flagged low-support and never headline. The
roll-up needs no fitted model and is byte-stable.

::: tulip.explain.dataset_evidence

::: tulip.explain.GlobalEvidenceReport

::: tulip.explain.PhenomenonFrequency

::: tulip.explain.FamilyEvidence

::: tulip.explain.ClassCount

## Contrastive analysis

Where `dataset_evidence` asks which phenomenon most identifies a class one versus
rest, `contrast_dialects` asks the pairwise question dialectology cares about:
given two dialects, which lexical markers, phonological isoglosses, and
morphological endings separate them, in which direction, and by how much. For each
feature it reports the document-occurrence rate in each group, a smoothed log-odds
ratio (the effect size and its sign), and a two-proportion z-test p-value,
Holm-corrected across the whole comparison. It is model-free and byte-stable, and
is exposed as `tulip contrast DATA DIALECT_A DIALECT_B` (use `--level family` to
contrast, for example, silesian against standard).

::: tulip.explain.contrast_dialects

::: tulip.explain.ContrastReport

::: tulip.explain.ContrastFeature
