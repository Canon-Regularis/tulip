"""Pipeline analysis commands.

Covers learning-curve, active-loop, selftrain, crossval, transfer, robustness,
conformal, openset, acquire, distill, isogloss-diagnostics, evaluate.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from tulip.cli._context import _console, _emit_report, _parse_number_csv, _tulip_errors, app


@app.command("learning-curve")
@_tulip_errors
def learning_curve_command(
    config_path: Path = typer.Argument(..., help="Experiment config YAML."),
    fractions: str = typer.Option(
        "0.1,0.25,0.5,0.75,1.0",
        "--fractions",
        help="Comma-separated training fractions, each in (0, 1].",
    ),
    seed: int | None = typer.Option(
        None, "--seed", help="Subsampling seed (default: the config's seed)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    out: Path | None = typer.Option(None, "--out", help="Also write the report JSON here."),
) -> None:
    """Train on nested fractions of the training split and score each on test.

    The curve shows what more annotation would buy: every point trains on a
    stratified subset of the same training split and evaluates on the identical
    held-out test split.
    """
    from tulip.config import load_experiment_config
    from tulip.pipeline import learning_curve

    parsed = _parse_number_csv(fractions, float, name="fractions")
    report = learning_curve(load_experiment_config(config_path), fractions=parsed, seed=seed)
    _emit_report(report, json_output=json_output, out=out, saved_label="learning curve")


@app.command("active-loop")
@_tulip_errors
def active_loop_command(
    config_path: Path = typer.Argument(..., help="Experiment config YAML."),
    strategy: str = typer.Option(
        "entropy", "--strategy", help="Acquisition strategy name, or 'random' for the baseline."
    ),
    seed_size: int = typer.Option(20, "--seed-size", min=1, help="Samples labeled before round 1."),
    batch_size: int = typer.Option(20, "--batch-size", min=1, help="Samples labeled each round."),
    rounds: int = typer.Option(5, "--rounds", min=1, help="Maximum acquisition rounds."),
    seed: int | None = typer.Option(
        None, "--seed", help="Seed for the seed set and random draws (default: the config's seed)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    out: Path | None = typer.Option(None, "--out", help="Also write the report JSON here."),
) -> None:
    """Simulate acquire, label, retrain over the training split; score each round on test.

    Closes the active-learning loop `acquire` only opens: the training split is
    the pool, its gold labels are the oracle, and every round labels a batch the
    strategy chose. Run it once per strategy and against `--strategy random` to
    see whether the strategy beats labeling at random.
    """
    from tulip.config import load_experiment_config
    from tulip.pipeline import active_learning_loop

    report = active_learning_loop(
        load_experiment_config(config_path),
        strategy=strategy,
        seed_size=seed_size,
        batch_size=batch_size,
        rounds=rounds,
        seed=seed,
    )
    _emit_report(report, json_output=json_output, out=out, saved_label="active-loop curve")


@app.command()
@_tulip_errors
def selftrain(
    labeled: Path = typer.Argument(..., help="Labelled samples (split .jsonl or manifest)."),
    unlabeled: Path = typer.Argument(..., help="Unlabelled samples to pseudo-label."),
    model: str = typer.Option("logistic_regression", "--model", "-m", help="Model registry name."),
    feature: list[str] = typer.Option(
        [], "--feature", "-f", help="Feature registry name (repeatable)."
    ),
    raw: bool = typer.Option(
        False, "--raw", help="The model consumes raw text/audio itself (neural); pass no features."
    ),
    threshold: float = typer.Option(
        0.90, "--threshold", min=0.0, max=1.0, help="Minimum confidence to trust a pseudo-label."
    ),
    iters: int = typer.Option(3, "--iters", min=1, help="Maximum self-training rounds."),
    out: Path | None = typer.Option(None, "--out", help="Save the improved model here."),
) -> None:
    """Grow a classifier from a labelled seed set using confident pseudo-labels.

    This is what makes label-less corpora (e.g. `bigos`, which carries no
    dialect labels) contribute to training rather than sitting unused.
    """
    from tulip.core.exceptions import ConfigurationError
    from tulip.data import read_samples
    from tulip.pipeline.selftrain import SelfTrainConfig, self_train

    # A classical model handed raw strings dies deep inside sklearn with an
    # unreadable ValueError. The registry carries no "raw input" capability flag
    # to infer this from, so make the caller say which shape they meant.
    if raw and feature:
        raise ConfigurationError("--raw takes no --feature; drop one of them")
    if not raw and not feature:
        raise ConfigurationError(
            "no --feature given: classical models need at least one feature extractor "
            "(e.g. -f char_tfidf). Raw-input models (herbert, fasttext, wav2vec2, ...) "
            "take none; pass --raw to say so explicitly."
        )

    result = self_train(
        labeled=list(read_samples(labeled)),
        unlabeled=list(read_samples(unlabeled)),
        model=model,
        features=list(feature),
        config=SelfTrainConfig(confidence_threshold=threshold, max_iterations=iters),
    )

    table = Table(title="self-training")
    table.add_column("round", justify="right")
    table.add_column("pseudo-labels added", justify="right")
    for index, added in enumerate(result.n_pseudo_per_iteration, start=1):
        table.add_row(str(index), str(added))
    _console.print(table)
    _console.print(
        f"converged after {result.iterations} round(s); "
        f"{len(result.pseudo_samples)} pseudo-label(s) total"
    )
    if out is not None:
        _console.print(f"[green]model saved to {result.classifier.save(out)}[/green]")


@app.command()
@_tulip_errors
def crossval(
    config_path: Path = typer.Argument(..., help="Experiment config YAML."),
    k: int = typer.Option(5, "--k", min=2, help="Number of folds."),
    seeds: str = typer.Option("0", "--seeds", help="Comma-separated fold seeds (e.g. 0,1,2)."),
    jobs: int = typer.Option(
        1,
        "--jobs",
        "-j",
        help="Fold runs to execute in parallel (-1 for all cores); "
        "the aggregate is identical to the sequential run.",
    ),
) -> None:
    """Grouped, stratified K-fold cross-validation with multi-seed aggregation.

    Reports each metric's mean and 95% confidence interval across all folds, so a
    single lucky split cannot flatter the model. Folds are speaker-disjoint.
    """
    from tulip.config import load_experiment_config
    from tulip.pipeline import CVConfig, run_cross_validation

    config = load_experiment_config(config_path)
    seed_tuple = _parse_number_csv(seeds, int, name="seeds")
    report = run_cross_validation(config, CVConfig(k=k, seeds=seed_tuple), n_jobs=jobs)

    table = Table(title=f"cross-validation {config.model.name!r} ({report.target})")
    for column in ("metric", "mean", "std", "95% CI"):
        table.add_column(column)
    for metric in report.metrics:
        table.add_row(
            metric.metric,
            f"{metric.mean:.4f}",
            f"{metric.std:.4f}",
            f"[{metric.low:.4f}, {metric.high:.4f}]",
        )
    _console.print(table)
    _console.print(
        f"[dim]{len(report.folds)} fold runs ({k}-fold x {len(seed_tuple)} seed(s))[/dim]"
    )


@app.command()
@_tulip_errors
def transfer(
    config_path: Path = typer.Argument(..., help="Experiment config YAML (multi-corpus data)."),
    matrix: bool = typer.Option(
        False, "--matrix", help="Full train-by-test transfer matrix instead of leave-one-out."
    ),
) -> None:
    """Cross-corpus transfer: does the model learn dialect or corpus artifacts?

    Partitions the data by source corpus. By default runs leave-one-corpus-out
    (train on the rest, test on the held-out corpus). With ``--matrix`` fills the
    full train-corpus by test-corpus grid.
    """
    from tulip.config import load_experiment_config
    from tulip.evaluation import run_loco, transfer_matrix

    config = load_experiment_config(config_path)
    report = transfer_matrix(config) if matrix else run_loco(config)
    _console.print(report.to_markdown())


@app.command()
@_tulip_errors
def robustness(
    config_path: Path = typer.Argument(..., help="Experiment config YAML."),
    perturbation: list[str] | None = typer.Option(
        None,
        "--perturbation",
        "-p",
        help="Perturbation name, repeatable (default dialect_intensity_dial). "
        "Options: dialect_intensity_dial, standardize, asr_noise, typo_noise.",
    ),
    levels: str = typer.Option(
        "0,0.25,0.5,0.75,1.0", "--levels", help="Comma-separated intensity levels in [0, 1]."
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for the perturbation draws."),
    out: Path | None = typer.Option(
        None, "--out", help="Directory to write robustness-<name>.md and .json."
    ),
) -> None:
    """Score a model as its inputs are perturbed along a linguistic intensity axis.

    Trains once on the clean split, then re-scores the test split perturbed at
    each level. The grounded perturbations (dialect_intensity_dial, standardize)
    move text along the standard-to-dialect axis; asr_noise and typo_noise stress
    the surface channel.
    """
    from tulip._serialize import write_markdown
    from tulip.config import load_experiment_config
    from tulip.core.exceptions import ConfigurationError
    from tulip.robustness import PerturbationConfig, run_robustness

    level_tuple = _parse_number_csv(levels, float, name="levels")
    if not level_tuple or any(not 0.0 <= level <= 1.0 for level in level_tuple):
        raise ConfigurationError("--levels must be non-empty and within [0, 1]")
    names = perturbation or ["dialect_intensity_dial"]
    specs = [PerturbationConfig(name=name, levels=level_tuple, seed=seed) for name in names]

    config = load_experiment_config(config_path)
    report = run_robustness(config, perturbations=specs)
    _console.print(report.to_markdown())
    if out is not None:
        write_markdown(out / f"robustness-{config.name}.md", report.to_markdown())
        report.save(out / f"robustness-{config.name}.json")
        _console.print(f"[green]wrote robustness artifacts to {out}[/green]")


@app.command()
@_tulip_errors
def conformal(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    calibration: Path = typer.Argument(..., help="Held-out calibration samples."),
    test: Path = typer.Argument(..., help="Test samples to measure coverage on."),
    alpha: float = typer.Option(0.1, "--alpha", min=0.0, max=1.0, help="Miscoverage rate."),
    mondrian: bool = typer.Option(False, "--mondrian", help="Per-class (class-conditional) sets."),
) -> None:
    """Calibrate prediction sets and report their coverage.

    Fits split conformal on the calibration split, then measures empirical
    coverage and mean set size on the test split. Coverage should meet the
    ``1 - alpha`` target.
    """
    from tulip.data import read_samples
    from tulip.pipeline import ConformalClassifier, DialectClassifier

    classifier = DialectClassifier.load(model_path)
    conformal_classifier = ConformalClassifier(classifier, alpha=alpha, mondrian=mondrian)
    conformal_classifier.fit_conformal(list(read_samples(calibration)))
    report = conformal_classifier.evaluate_coverage(list(read_samples(test)))
    kind = "Mondrian" if mondrian else "marginal"
    _console.print(
        f"{kind} conformal (alpha={alpha}): coverage "
        f"[bold]{report.coverage:.3f}[/bold] (target {report.target_coverage:.2f}), "
        f"mean set size {report.mean_set_size:.2f} over {report.n_samples} samples"
    )


@app.command()
@_tulip_errors
def openset(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    calibration: Path = typer.Argument(..., help="Held-out calibration samples."),
    test: Path = typer.Argument(..., help="Test samples, possibly including unseen dialects."),
    alpha: float = typer.Option(0.1, "--alpha", min=0.0, max=1.0, help="Miscoverage rate."),
    mondrian: bool = typer.Option(False, "--mondrian", help="Per-class conformal thresholds."),
) -> None:
    """Flag inputs unlike any known dialect, and report open-set quality.

    Fits split conformal on the calibration split, then evaluates novelty
    detection on the test split. A test sample whose gold dialect was never
    trained on counts as truly novel, which is the deployment question of
    meeting a new region.
    """
    from tulip.data import read_samples
    from tulip.pipeline import ConformalClassifier, DialectClassifier, OpenSetClassifier

    classifier = DialectClassifier.load(model_path)
    conformal = ConformalClassifier(classifier, alpha=alpha, mondrian=mondrian)
    conformal.fit_conformal(list(read_samples(calibration)))
    report = OpenSetClassifier(conformal).evaluate(list(read_samples(test)))
    _console.print(report.to_markdown())


@app.command()
@_tulip_errors
def acquire(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    unlabeled: Path = typer.Argument(..., help="Unlabeled pool: split .jsonl or manifest."),
    strategy: str = typer.Option(
        "entropy",
        "--strategy",
        help="Acquisition strategy name; an unknown value lists the registered options.",
    ),
    budget: int | None = typer.Option(
        None, "--budget", min=1, help="Keep only the top-N candidates (default all)."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the ranking as JSON."),
) -> None:
    """Rank an unlabeled pool by which samples to label first.

    A model trained on a labeled seed set scores each unlabeled sample by an
    acquisition strategy, so a fixed annotation budget buys the most signal. The
    dialect-aware ``intensity_gated`` strategy keeps budget off standard Polish
    the model merely happens to be unsure about. Ranking only; labeling is a
    human step.
    """
    from tulip.core.exceptions import ConfigurationError, UnknownComponentError
    from tulip.data import read_samples
    from tulip.pipeline import STRATEGIES, DialectClassifier, rank_for_labeling

    classifier = DialectClassifier.load(model_path)
    try:
        candidates = rank_for_labeling(
            classifier, list(read_samples(unlabeled)), strategy=strategy, budget=budget
        )
    except UnknownComponentError as exc:
        # The valid set is derived from the registry, never a hardcoded list, so a
        # newly registered strategy is discoverable without editing the CLI.
        options = ", ".join(STRATEGIES.names())
        raise ConfigurationError(f"unknown strategy {strategy!r}; choose from: {options}") from exc
    if json_output:
        _console.print_json(data=[candidate.model_dump() for candidate in candidates])
        return
    table = Table(title=f"acquisition ranking ({strategy})")
    table.add_column("#", justify="right")
    table.add_column("sample")
    table.add_column("predicted")
    table.add_column("confidence", justify="right")
    table.add_column("score", justify="right")
    for rank, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(rank),
            candidate.sample_id,
            candidate.predicted_label,
            f"{candidate.confidence:.1%}",
            f"{candidate.score:.4f}",
        )
    _console.print(table)


@app.command()
@_tulip_errors
def evaluate(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    data: Path = typer.Argument(
        ..., help="Labelled samples: split .jsonl, manifest file, or manifest directory."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Evaluate a saved model on labelled samples."""
    from tulip.data import read_samples
    from tulip.pipeline import DialectClassifier, evaluate_samples

    classifier = DialectClassifier.load(model_path)
    report = evaluate_samples(classifier, list(read_samples(data)), name=str(data))
    if json_output:
        _console.print_json(report.model_dump_json())
    else:
        _console.print(report.to_markdown())


@app.command()
@_tulip_errors
def distill(
    teacher_path: Path = typer.Argument(..., help="Saved teacher model directory."),
    transfer: Path = typer.Argument(..., help="Pool the teacher labels (split .jsonl/manifest)."),
    test: Path = typer.Argument(..., help="Gold-labelled samples both models are scored on."),
    student: str = typer.Option(
        "logistic_regression", "--student", "-s", help="Small student model registry name."
    ),
    feature: list[str] = typer.Option(
        [], "--feature", "-f", help="Student feature registry name (repeatable)."
    ),
    min_confidence: float = typer.Option(
        0.0, "--min-confidence", min=0.0, max=1.0, help="Drop teacher labels below this confidence."
    ),
    workdir: Path | None = typer.Option(
        None, "--workdir", help="Save both models here to also measure on-disk size."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    out: Path | None = typer.Option(None, "--out", help="Also write the report JSON here."),
) -> None:
    """Distil a saved teacher into a small student and report accuracy vs cost.

    The teacher labels the transfer pool, the student trains on those labels,
    and both are scored on the test split. The report shows how much of the
    teacher's accuracy the student keeps and how much smaller and faster it is.
    """
    from tulip.data import read_samples
    from tulip.models import MODELS
    from tulip.pipeline import DialectClassifier, DistillationConfig
    from tulip.pipeline import distill as run_distill

    # A classical student needs at least one feature extractor; a raw-input model
    # (fasttext, herbert, ...) takes none. Default to char_tfidf for a classical
    # student given no --feature, so the bare command works out of the box.
    student_features = list(feature)
    if not student_features and not MODELS.metadata(student).get("raw_input", False):
        student_features = ["char_tfidf"]

    teacher = DialectClassifier.load(teacher_path)
    report = run_distill(
        teacher=teacher,
        transfer=list(read_samples(transfer)),
        test=list(read_samples(test)),
        student_model=student,
        features=student_features,
        config=DistillationConfig(min_teacher_confidence=min_confidence),
        workdir=workdir,
    )
    _emit_report(report, json_output=json_output, out=out, saved_label="distillation report")


@app.command("isogloss-diagnostics")
@_tulip_errors
def isogloss_diagnostics_command(
    model_path: Path = typer.Argument(..., help="Saved model directory."),
    data: Path = typer.Argument(..., help="Labelled samples: split .jsonl or manifest."),
    rules: Path | None = typer.Option(
        None, "--rules", help="Phonological rule file (default: the bundled set)."
    ),
    min_support: int = typer.Option(
        5, "--min-support", min=1, help="Group size below which a diagnostic is flagged."
    ),
    json_output: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
    out: Path | None = typer.Option(None, "--out", help="Also write the report JSON here."),
) -> None:
    """Check whether accuracy collapses when a dialect marker is absent.

    For each detectable isogloss, splits the samples of the dialects it signals
    by whether the reflex is present in the text and compares accuracy. A large
    positive gap means the model reads the surface marker, not the dialect.
    """
    from tulip.data import read_samples
    from tulip.pipeline import DialectClassifier, isogloss_diagnostics

    classifier = DialectClassifier.load(model_path)
    report = isogloss_diagnostics(
        classifier, list(read_samples(data)), rules_path=rules, min_support=min_support
    )
    _emit_report(report, json_output=json_output, out=out, saved_label="isogloss diagnostics")
