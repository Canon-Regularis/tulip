"""The structural contract shared by every classifier the pipeline layer exposes.

:class:`~tulip.pipeline.classifier.DialectClassifier` predicts at one fixed
label level from one fixed modality. Two of its siblings deliberately do not:

* :class:`~tulip.pipeline.hierarchical.HierarchicalDialectClassifier` returns a
  prediction whose ``level`` varies *per sample*: fine-grained when the model
  is confident, backed off to a coarser level when it is not.
* :class:`~tulip.pipeline.fusion.MultimodalClassifier` reads *both* ``text`` and
  ``audio_path`` from a sample and fuses two probability distributions.

Neither can be a subclass of ``DialectClassifier`` without breaking it.
``DialectClassifier.predict_batch`` guarantees that every returned
:class:`~tulip.core.types.Prediction` carries ``level == self.target``, and that
its input is a sequence of *raw* values (texts or audio paths) for a single
modality. A subclass that violated either postcondition would not be
substitutable for its base; the Liskov substitution principle is precisely the
rule being obeyed by *not* subclassing here.

So the three are related by a common protocol rather than a class hierarchy.
:class:`SamplePredictor` is stated over :class:`~tulip.core.types.Sample`, which
is the only input rich enough to describe all three: it carries every modality
and every label level at once. Consumers (evaluation, the CLI, visualisation)
depend on this abstraction instead of on any concrete classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from tulip.config.schemas import ComponentConfig
    from tulip.core.types import Prediction, Sample, TaskType
    from tulip.labels.taxonomy import LabelLevel
    from tulip.pipeline.classifier import LabelledBatch

__all__ = ["CalibratableClassifier", "ProbabilisticClassifier", "SamplePredictor"]


@runtime_checkable
class SamplePredictor(Protocol):
    """Turns :class:`Sample` records into :class:`Prediction` records.

    Implementations decide which of a sample's modalities they read and at which
    label level they answer; the protocol fixes only that one prediction is
    returned per input sample, in order.
    """

    def predict_samples(self, samples: Sequence[Sample]) -> list[Prediction]:
        """Classify each sample, returning one prediction per input, in order.

        Raises:
            DataError: if a sample lacks the input(s) the implementation needs.
        """
        ...


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    """The slice of a classifier that composition needs: aligned probabilities.

    A deliberately narrow structural type (DIP/ISP): a consumer such as
    :class:`~tulip.pipeline.fusion.MultimodalClassifier` needs a class
    vocabulary, a shared label level, a modality tag, and a probability matrix,
    and nothing else. :class:`~tulip.pipeline.classifier.DialectClassifier`
    satisfies it structurally, so no import of the concrete class is required to
    depend on this behaviour, which is what lets a test inject a cheap
    deterministic probability stub instead of training a real model.

    It lives here, beside :class:`SamplePredictor`, rather than in any one
    consumer, because it is a general pipeline abstraction, not fusion's alone.
    """

    classes_: tuple[str, ...]
    target: LabelLevel
    task: TaskType

    def predict_proba(self, raws: Sequence[Any]) -> np.ndarray:
        """Return the probability matrix for ``raws``, columns aligned to ``classes_``."""
        ...


@runtime_checkable
class CalibratableClassifier(ProbabilisticClassifier, Protocol):
    """The surface the uncertainty wrappers need from the classifier they wrap.

    :class:`~tulip.pipeline.calibrated.CalibratedClassifier`,
    :class:`~tulip.pipeline.conformal.ConformalClassifier`, and
    :class:`~tulip.pipeline.openset.OpenSetClassifier` compose over a base by
    reading its aligned probabilities (:class:`ProbabilisticClassifier`), its
    labelled-batch builder (samples to raw/label pairs), and its ``model_config``
    (which names the wrapped model in a report). Typing the wrappers against this
    abstraction rather than the concrete
    :class:`~tulip.pipeline.classifier.DialectClassifier` keeps them composable
    over any conforming classifier and injectable with a stub in tests, exactly
    as :class:`~tulip.pipeline.fusion.MultimodalClassifier` already depends on
    :class:`ProbabilisticClassifier`.
    """

    model_config: ComponentConfig

    def labelled_batch(self, samples: Sequence[Sample]) -> LabelledBatch:
        """Pair each usable sample's raw model input with its label at ``target``."""
        ...
