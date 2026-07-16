"""Train a dialect classifier and predict, entirely offline.

Runs on the in-memory ``synthetic`` corpus, so a fresh checkout with only the
core install (no downloads, no heavy extras) can execute it end to end::

    python examples/train_and_predict.py

The synthetic corpus is a generated fixture, not real speech: it exists to show
the API and machinery, not to make a dialectology claim (see the caveats in
docs/datasets.md). Swap the generator for a real corpus config once you have
one assembled under ``data/raw/`` (see docs/datasets.md).
"""

from __future__ import annotations

from tulip.config.schemas import SplitConfig
from tulip.data import SyntheticSpec, generate_corpus, speaker_disjoint_split
from tulip.pipeline import DialectClassifier, evaluate_samples


def main() -> None:
    # 1. Build a small, learnable, speaker-diverse corpus in memory.
    samples = generate_corpus(SyntheticSpec(n_speakers_per_dialect=8, samples_per_speaker=12))

    # 2. Split it speaker-disjoint and label-stratified, so no speaker leaks
    #    across train/validation/test and a score cannot reward memorising voices.
    splits = speaker_disjoint_split(samples, SplitConfig(seed=42))
    print(
        f"train={len(splits.train)}  validation={len(splits.validation)}  "
        f"test={len(splits.test)}"
    )

    # 3. Compose features + a model into one classifier and fit it.
    classifier = DialectClassifier(
        model="logistic_regression", features=["char_tfidf"], seed=42
    ).fit(splits.train)

    # 4. Evaluate on the held-out test split.
    report = evaluate_samples(classifier, splits.test, name="test")
    print()
    print(report.to_markdown())

    # 5. Classify a fresh sentence; a Prediction carries the full distribution.
    prediction = classifier.predict("hej baca kaj se owce pasą na holi")
    print()
    print(f"predicted dialect: {prediction.label}")
    for candidate in prediction.probabilities[:3]:
        print(f"  {candidate.label:<12} {candidate.probability:.1%}")


if __name__ == "__main__":
    main()
