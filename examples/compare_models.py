"""Compare several classical models on one identical, frozen split, offline.

Every competitor trains and is scored on the *same* speaker-disjoint split, so
the macro-F1 numbers are directly comparable. Runs on the in-memory synthetic
corpus (core install only)::

    python examples/compare_models.py

For a reproducible, committed benchmark over many models, see
``tulip leaderboard benchmarks/suite.yaml`` and benchmarks/README.md.
"""

from __future__ import annotations

from tulip.config.schemas import SplitConfig
from tulip.data import SyntheticSpec, generate_corpus, speaker_disjoint_split
from tulip.pipeline import DialectClassifier, evaluate_samples

COMPETITORS = ("naive_bayes", "logistic_regression", "linear_svm", "random_forest")


def main() -> None:
    samples = generate_corpus(SyntheticSpec(n_speakers_per_dialect=8, samples_per_speaker=12))
    splits = speaker_disjoint_split(samples, SplitConfig(seed=42))

    print(f"{'model':<22} {'accuracy':>10} {'f1_macro':>10}")
    print("-" * 44)
    for model in COMPETITORS:
        classifier = DialectClassifier(
            model=model, features=["char_tfidf"], seed=42
        ).fit(splits.train)
        report = evaluate_samples(classifier, splits.test, name="test")
        print(f"{model:<22} {report.accuracy:>10.3f} {report.f1_macro:>10.3f}")


if __name__ == "__main__":
    main()
