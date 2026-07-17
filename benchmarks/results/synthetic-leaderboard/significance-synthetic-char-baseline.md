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
