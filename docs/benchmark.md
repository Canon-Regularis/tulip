# Computational Identification of Polish Dialect Variation: A Speaker-Disjoint Benchmark

> **Synthetic fixture, not real dialect accuracy.** These numbers come from a procedurally generated corpus that exercises the pipeline; they say nothing about real dialect classification. See the caption on the results table.

## Abstract

Polish dialect identification is a fine-grained classification problem grounded in phonology, morphology, lexicon, and geography. This report presents a reproducible, speaker-disjoint benchmark: models are trained and evaluated on frozen, content-fingerprinted splits in which no speaker appears in more than one partition, so a reported score reflects generalisation to unseen speakers rather than memorised voices. It documents the dataset, the label hierarchy, the geographic footprint, and the evaluation protocol, then reports each model's accuracy, macro-averaged F1, and calibration alongside paired significance tests against the majority-class floor.

## Label hierarchy

Labels are hierarchical across 5 levels (family, dialect, region, village, voivodeship); a family label auto-derives from a dialect label, and a corpus may carry labels at whichever levels it annotated. The regional dialects group into families as follows.

| Family | Regional dialects |
| :--- | ---: |
| Greater Polish | Greater Poland, Kociewie, Kujawy |
| Kashubian | Kashubia |
| Lesser Polish | Lesser Poland, Orawa, Podhale, Podolia (Mackowce), Spisz |
| Masovian | Kurpie, Masuria, Mazovia, Podlasie, Warmia |
| Silesian | Cieszyn Silesia, Silesia |

Families with no regional-dialect members in the taxonomy (used at family level only): Standard Polish.

## Dataset

Generate the corpus datasheet with `tulip card datasheet <build-dir> --spec <spec.yaml>` and embed it here; it documents provenance, splits and speaker counts, the class distribution at every level, the geographic footprint, and the demographic composition.

## Protocol

Every model is trained and scored on one identical, frozen split. The split is speaker-disjoint and label-stratified: whole speaker groups are assigned to train, validation, or test, so no speaker crosses partitions and a score measures generalisation to unseen speakers, not recall of a memorised voice. The split is content-fingerprinted (a per-split BLAKE2b digest recorded in ``split_lock.json``), so any silent change to the data is caught and the exact split behind a reported number can be reconstructed and verified.

Reported metrics are accuracy, macro-averaged F1 (which weights every dialect equally, so a strong score cannot come from the majority class alone), weighted F1, one-versus-rest ROC AUC where probabilities allow it, and top-label calibration (expected calibration error and Brier score). Models are ranked by macro F1, ties broken deterministically, and each is compared to the majority-class floor with a paired bootstrap confidence interval and an exact McNemar test, Holm-corrected across the comparison set.

## Results

**Synthetic fixture, not real dialect accuracy.** These scores come from a procedurally generated corpus that exercises the pipeline; they say nothing about real dialect classification. See docs/benchmark.md and benchmarks/results/real-text-leaderboard/ for the real benchmark.

| Experiment | Model | Accuracy | F1 (macro) | F1 (weighted) | ROC AUC | ECE | Brier | Train |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic-char-baseline | linear_svm | 0.8333 | 0.8313 | 0.8313 | 0.9520 | 0.1975 | 0.3142 | 576 |
| synthetic-char-baseline | naive_bayes | 0.8125 | 0.8109 | 0.8109 | 0.9674 | 0.0521 | 0.2637 | 576 |
| synthetic-char-baseline | logistic_regression | 0.7986 | 0.7972 | 0.7972 | 0.9505 | 0.3564 | 0.4540 | 576 |
| synthetic-lexical-baseline | linear_svm | 0.7986 | 0.7925 | 0.7925 | 0.9456 | 0.2001 | 0.3412 | 576 |
| synthetic-lexical-baseline | random_forest | 0.7431 | 0.7452 | 0.7452 | 0.9100 | 0.2931 | 0.4786 | 576 |
| synthetic-lexical-baseline | logistic_regression | 0.7431 | 0.7389 | 0.7389 | 0.9153 | 0.3520 | 0.5406 | 576 |
| synthetic-char-baseline | random_forest | 0.6806 | 0.6900 | 0.6900 | 0.9046 | 0.1439 | 0.4686 | 576 |
| synthetic-lexical-baseline | naive_bayes | 0.4444 | 0.3562 | 0.3562 | 0.8615 | 0.3128 | 0.8442 | 576 |
| synthetic-char-baseline | majority | 0.1667 | 0.0476 | 0.0476 | 0.5000 | 0.0000 | 0.8333 | 576 |
| synthetic-lexical-baseline | majority | 0.1667 | 0.0476 | 0.0476 | 0.5000 | 0.0000 | 0.8333 | 576 |

## Significance

# Significance: test (n=144)

Best by F1 (macro): **linear_svm**. Statistically tied with best (Holm-corrected McNemar, alpha=0.05): linear_svm, naive_bayes, logistic_regression.

Confidence intervals are 95% percentile bootstrap (2000 resamples, seed 0).

| Model | Accuracy | F1 (macro) | F1 (weighted) | Tied w/ best |
| :--- | ---: | ---: | ---: | ---: |
| linear_svm | 0.8333 [0.7708, 0.8958] | 0.8313 [0.7655, 0.8882] | 0.8313 [0.7688, 0.8916] | yes |
| naive_bayes | 0.8125 [0.7500, 0.8750] | 0.8109 [0.7421, 0.8706] | 0.8109 [0.7464, 0.8731] | yes |
| logistic_regression | 0.7986 [0.7361, 0.8611] | 0.7972 [0.7313, 0.8565] | 0.7972 [0.7324, 0.8594] | yes |
| random_forest | 0.6806 [0.6042, 0.7569] | 0.6900 [0.6194, 0.7540] | 0.6900 [0.6172, 0.7600] | no |
| majority | 0.1667 [0.1111, 0.2292] | 0.0476 [0.0333, 0.0621] | 0.0476 [0.0222, 0.0855] | no |

## Pairwise McNemar tests (discordant a/b)

| Comparison | Δ acc | Discordant | p | p (Holm) | sig. |
| :--- | ---: | ---: | ---: | ---: | ---: |
| linear_svm vs naive_bayes | 0.0208 | 7/4 | 0.5488 | 1.0000 | no |
| linear_svm vs logistic_regression | 0.0347 | 6/1 | 0.1250 | 0.3750 | no |
| linear_svm vs random_forest | 0.1528 | 27/5 | 0.0001 | 0.0007 | yes |
| linear_svm vs majority | 0.6667 | 99/3 | 0.0000 | 0.0000 | yes |
| naive_bayes vs logistic_regression | 0.0139 | 6/4 | 0.7539 | 1.0000 | no |
| naive_bayes vs random_forest | 0.1319 | 25/6 | 0.0009 | 0.0044 | yes |
| naive_bayes vs majority | 0.6458 | 96/3 | 0.0000 | 0.0000 | yes |
| logistic_regression vs random_forest | 0.1181 | 21/4 | 0.0009 | 0.0044 | yes |
| logistic_regression vs majority | 0.6319 | 94/3 | 0.0000 | 0.0000 | yes |
| random_forest vs majority | 0.5139 | 78/4 | 0.0000 | 0.0000 | yes |

# Significance: test (n=144)

Best by F1 (macro): **linear_svm**. Statistically tied with best (Holm-corrected McNemar, alpha=0.05): linear_svm, random_forest, logistic_regression.

Confidence intervals are 95% percentile bootstrap (2000 resamples, seed 0).

| Model | Accuracy | F1 (macro) | F1 (weighted) | Tied w/ best |
| :--- | ---: | ---: | ---: | ---: |
| linear_svm | 0.7986 [0.7361, 0.8611] | 0.7925 [0.7225, 0.8542] | 0.7925 [0.7226, 0.8588] | yes |
| random_forest | 0.7431 [0.6736, 0.8125] | 0.7452 [0.6679, 0.8097] | 0.7452 [0.6711, 0.8130] | yes |
| logistic_regression | 0.7431 [0.6734, 0.8125] | 0.7389 [0.6644, 0.8048] | 0.7389 [0.6647, 0.8075] | yes |
| naive_bayes | 0.4444 [0.3611, 0.5278] | 0.3562 [0.2978, 0.4113] | 0.3562 [0.2742, 0.4434] | no |
| majority | 0.1667 [0.1111, 0.2292] | 0.0476 [0.0333, 0.0621] | 0.0476 [0.0222, 0.0855] | no |

## Pairwise McNemar tests (discordant a/b)

| Comparison | Δ acc | Discordant | p | p (Holm) | sig. |
| :--- | ---: | ---: | ---: | ---: | ---: |
| linear_svm vs random_forest | 0.0556 | 12/4 | 0.0768 | 0.2304 | no |
| linear_svm vs logistic_regression | 0.0556 | 12/4 | 0.0768 | 0.2304 | no |
| linear_svm vs naive_bayes | 0.3542 | 55/4 | 0.0000 | 0.0000 | yes |
| linear_svm vs majority | 0.6319 | 94/3 | 0.0000 | 0.0000 | yes |
| random_forest vs logistic_regression | 0.0000 | 12/12 | 1.0000 | 1.0000 | no |
| random_forest vs naive_bayes | 0.2986 | 52/9 | 0.0000 | 0.0000 | yes |
| random_forest vs majority | 0.5764 | 89/6 | 0.0000 | 0.0000 | yes |
| logistic_regression vs naive_bayes | 0.2986 | 51/8 | 0.0000 | 0.0000 | yes |
| logistic_regression vs majority | 0.5764 | 90/7 | 0.0000 | 0.0000 | yes |
| naive_bayes vs majority | 0.2778 | 64/24 | 0.0000 | 0.0001 | yes |

## Demographic and geographic bias

Subgroup disparity is measured with `tulip analyze <predictions> --fairness`, which reports the best-versus-worst group gap over the geographic (region, voivodeship, family, dialect) and demographic (age band, gender) slices, with Holm-corrected two-proportion tests and low-support groups flagged. It runs on the per-sample predictions the board does not commit, so it is produced locally rather than embedded here.

## Limitations

The taxonomy is a discrete approximation of a dialect continuum: real dialect boundaries are gradients, and a single hard label per sample understates that. Coverage is uneven across regions and speakers, so per-class and per-subgroup results with low support are flagged and must not be read as headline findings. Text-based identification cannot capture the phonetic cues that live only in audio, and self-reported or surrogate speaker and demographic metadata is imperfect. The benchmark measures identification accuracy under these constraints; it makes no claim about the sociolinguistic reality of any speaker or community.
