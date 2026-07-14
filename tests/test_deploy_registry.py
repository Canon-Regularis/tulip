"""Tests for the content-addressed model registry (tulip.deploy.registry_store)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sklearn.linear_model import LogisticRegression

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.deploy import ModelRegistry, Stage, artifact_digest
from tulip.models.persistence import save_model

if TYPE_CHECKING:
    from pathlib import Path


def _save_model(path: Path, *, seed: int) -> Path:
    """Save a tiny fitted estimator as a persisted artifact (distinct per seed)."""
    features = [[0.0], [1.0], [float(seed)]]
    model = LogisticRegression().fit(features, ["a", "b", "a"])
    return save_model(model, path, metadata={"target": "dialect", "task": "text"})


@pytest.fixture
def model_a(tmp_path: Path) -> Path:
    return _save_model(tmp_path / "model_a", seed=1)


@pytest.fixture
def model_b(tmp_path: Path) -> Path:
    return _save_model(tmp_path / "model_b", seed=99)


class TestArtifactDigest:
    def test_is_deterministic(self, model_a: Path) -> None:
        assert artifact_digest(model_a) == artifact_digest(model_a)

    def test_changes_when_bytes_change(self, model_a: Path) -> None:
        before = artifact_digest(model_a)
        (model_a / "metadata.json").write_text("{}", encoding="utf-8")  # tamper
        assert artifact_digest(model_a) != before

    def test_missing_artifact_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DataError, match="cannot digest"):
            artifact_digest(tmp_path / "nope")


class TestAdd:
    def test_records_a_self_describing_entry(self, model_a: Path, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "reg")
        entry = registry.add(model_a, name="dialect", version="1")
        assert entry.name == "dialect" and entry.version == "1"
        assert entry.stage is Stage.STAGING
        assert entry.digest == artifact_digest(model_a)
        assert entry.target == "dialect" and entry.task == "text"
        assert set(entry.classes) == {"a", "b"}

    def test_is_content_addressed_and_dedupes(self, model_a: Path, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "reg")
        registry.add(model_a, name="dialect", version="1")
        registry.add(model_a, name="dialect", version="2")  # same bytes, new version
        artifact_dirs = list((tmp_path / "reg" / "artifacts").iterdir())
        assert len(artifact_dirs) == 1  # one stored artifact, two entries

    def test_reject_version_conflict_with_different_digest(
        self, model_a: Path, model_b: Path, tmp_path: Path
    ) -> None:
        registry = ModelRegistry(tmp_path / "reg")
        registry.add(model_a, name="dialect", version="1")
        with pytest.raises(ConfigurationError, match="already exists"):
            registry.add(model_b, name="dialect", version="1")

    def test_re_add_identical_is_idempotent(self, model_a: Path, tmp_path: Path) -> None:
        registry = ModelRegistry(tmp_path / "reg")
        first = registry.add(model_a, name="dialect", version="1")
        again = registry.add(model_a, name="dialect", version="1")
        assert first == again


class TestPromotionAndResolution:
    @pytest.fixture
    def registry(self, model_a: Path, model_b: Path, tmp_path: Path) -> ModelRegistry:
        registry = ModelRegistry(tmp_path / "reg")
        registry.add(model_a, name="dialect", version="1")
        registry.add(model_b, name="dialect", version="2")
        return registry

    def test_resolve_before_promotion_has_no_production(self, registry: ModelRegistry) -> None:
        with pytest.raises(DataError, match="production"):
            registry.resolve("dialect")

    def test_promote_and_resolve(self, registry: ModelRegistry) -> None:
        registry.promote("dialect", "1")
        assert registry.resolve("dialect").version == "1"
        assert registry.resolve("dialect@production").version == "1"
        assert registry.resolve("dialect@2").version == "2"  # exact version
        assert registry.resolve("dialect@staging").version == "2"  # latest staging

    def test_promoting_a_second_version_archives_the_first(self, registry: ModelRegistry) -> None:
        registry.promote("dialect", "1")
        registry.promote("dialect", "2")
        assert registry.resolve("dialect").version == "2"
        archived = {e.version for e in registry.versions_of("dialect") if e.stage is Stage.ARCHIVED}
        assert archived == {"1"}

    def test_rollback_restores_previous_production(self, registry: ModelRegistry) -> None:
        registry.promote("dialect", "1")
        registry.promote("dialect", "2")
        restored = registry.rollback("dialect")
        assert restored.version == "1"
        assert registry.resolve("dialect").version == "1"
        # v2 (the rolled-back version) is archived; there is only one production.
        production = [e for e in registry.versions_of("dialect") if e.stage is Stage.PRODUCTION]
        assert [e.version for e in production] == ["1"]

    def test_rollback_without_history_raises(self, registry: ModelRegistry) -> None:
        registry.promote("dialect", "1")
        with pytest.raises(DataError, match="roll back"):
            registry.rollback("dialect")

    def test_path_for_points_at_the_content_address(self, registry: ModelRegistry) -> None:
        entry = registry.resolve("dialect@1")
        path = registry.path_for(entry)
        assert path.name == entry.digest
        assert (path / "model.joblib").is_file()

    def test_resolve_unknown_name_raises(self, registry: ModelRegistry) -> None:
        with pytest.raises(DataError, match="no registered model"):
            registry.resolve("ghost")


class TestIndexDeterminism:
    def test_index_regenerates_byte_identically(self, model_a: Path, tmp_path: Path) -> None:
        # Same operation sequence into two registries -> byte-identical index.
        for root in ("reg_a", "reg_b"):
            registry = ModelRegistry(tmp_path / root)
            registry.add(model_a, name="dialect", version="1")
            registry.promote("dialect", "1")
        a = (tmp_path / "reg_a" / "registry.json").read_bytes()
        b = (tmp_path / "reg_b" / "registry.json").read_bytes()
        assert a == b

    def test_index_round_trips_after_reload(self, model_a: Path, tmp_path: Path) -> None:
        ModelRegistry(tmp_path / "reg").add(model_a, name="dialect", version="1")
        reopened = ModelRegistry(tmp_path / "reg")
        assert reopened.resolve("dialect@1").version == "1"
