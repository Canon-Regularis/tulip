"""The structural contract shared by every classifier the pipeline layer exposes.

:class:`~tulip.pipeline.classifier.DialectClassifier` predicts at one fixed
label level from one fixed modality. Two of its siblings deliberately do not:

* :class:`~tulip.pipeline.hierarchical.HierarchicalDialectClassifier` returns a
  prediction whose ``level`` varies *per sample* -- fine-grained when the model
  is confident, backed off to a coarser level when it is not.
* :class:`~tulip.pipeline.fusion.MultimodalClassifier` reads *both* ``text`` and
  ``audio_path`` from a sample and fuses two probability distributions.

Neither can be a subclass of ``DialectClassifier`` without breaking it.
``DialectClassifier.predict_batch`` guarantees that every returned
:class:`~tulip.core.types.Prediction` carries ``level == self.target``, and that
its input is a sequence of *raw* values (texts or audio paths) for a single
modality. A subclass that violated either postcondition would not be
substitutable for its base -- the Liskov substitution principle is precisely the
rule being obeyed by *not* subclassing here.

So the three are related by a common protocol rather than a class hierarchy.
:class:`SamplePredictor` is stated over :class:`~tulip.core.types.Sample`, which
is the only input rich enough to describe all three: it carries every modality
and every label level at once. Consumers (evaluation, the CLI, visualisation)
depend on this abstraction instead of on any concrete classifier.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tulip.core.types import Prediction, Sample

__all__ = ["SamplePredictor"]


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
