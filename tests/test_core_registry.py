"""Tests for the generic component registry (tulip.core.registry)."""

from __future__ import annotations

import pytest

from tulip.core.exceptions import DuplicateComponentError, UnknownComponentError
from tulip.core.registry import Registry


@pytest.fixture
def registry() -> Registry[object]:
    reg: Registry[object] = Registry("widget")
    reg.add("alpha", dict, aliases=("first",), metadata={"training_aware": True})
    reg.add("beta_two", list)
    return reg


class TestLookup:
    def test_get_resolves_names_aliases_and_normalisation(self, registry: Registry[object]) -> None:
        assert registry.get("alpha") is dict
        assert registry.get("first") is dict  # alias
        assert registry.get(" Alpha ") is dict  # case/whitespace-insensitive
        assert registry.get("beta-two") is list  # dash normalised to underscore

    def test_unknown_name_raises_with_suggestions(self, registry: Registry[object]) -> None:
        with pytest.raises(UnknownComponentError, match="alpha"):
            registry.get("alpah")

    def test_names_and_container_protocol(self, registry: Registry[object]) -> None:
        assert registry.names() == ["alpha", "beta_two"]
        assert "first" in registry
        assert "gamma" not in registry
        assert len(registry) == 2

    def test_create_instantiates_with_params(self, registry: Registry[object]) -> None:
        assert registry.create("beta_two") == []
        assert registry.create("alpha", key="value") == {"key": "value"}


class TestRegistration:
    def test_duplicate_names_and_aliases_are_rejected(self, registry: Registry[object]) -> None:
        with pytest.raises(DuplicateComponentError):
            registry.add("alpha", set)
        with pytest.raises(DuplicateComponentError):
            registry.add("gamma", set, aliases=("beta_two",))

    def test_register_decorator(self) -> None:
        reg: Registry[object] = Registry("thing")

        @reg.register("decorated", metadata={"flag": 1})
        class Thing:
            pass

        assert reg.get("decorated") is Thing
        assert reg.metadata("decorated") == {"flag": 1}


class TestMetadata:
    def test_metadata_resolves_aliases_and_defaults_empty(self, registry: Registry[object]) -> None:
        assert registry.metadata("alpha") == {"training_aware": True}
        assert registry.metadata("first") == {"training_aware": True}  # via alias
        assert registry.metadata("beta_two") == {}

    def test_metadata_returns_a_defensive_copy(self, registry: Registry[object]) -> None:
        registry.metadata("alpha")["training_aware"] = False
        assert registry.metadata("alpha") == {"training_aware": True}

    def test_metadata_for_unknown_name_raises(self, registry: Registry[object]) -> None:
        with pytest.raises(UnknownComponentError):
            registry.metadata("gamma")

    def test_neural_models_declare_training_awareness(self) -> None:
        from tulip.models import MODELS

        assert MODELS.metadata("herbert") == {"training_aware": True}
        assert MODELS.metadata("wav2vec2") == {"training_aware": True}
        assert MODELS.metadata("logistic_regression") == {}
        assert MODELS.metadata("ecapa_tdnn") == {}  # frozen encoder + sklearn head
