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
