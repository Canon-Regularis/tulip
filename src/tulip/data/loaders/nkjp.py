"""Loader for the National Corpus of Polish (NKJP) as standard-Polish negatives."""

from __future__ import annotations

import csv
import tarfile
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, ClassVar

from tulip.core.exceptions import ConfigurationError, DataError
from tulip.data.download import fetch_file
from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.manifest import ManifestColumns
from tulip.data.registry import DATASETS
from tulip.utils.logging import get_logger

_logger = get_logger(__name__)

#: Stable download URL of the manually annotated 1-million-word subcorpus
#: (GNU GPL; see http://clip.ipipan.waw.pl/NationalCorpusOfPolish).
NKJP_1M_URL = (
    "http://clip.ipipan.waw.pl/NationalCorpusOfPolish"
    "?action=AttachFile&do=get&target=NKJP-PodkorpusMilionowy-1.2.tar.gz"
)

_TEI_NS = "{http://www.tei-c.org/ns/1.0}"


@DATASETS.register("nkjp")
class NkjpLoader(ManifestBackedLoader):
    """NKJP (https://nkjp.pl/): standard-Polish negative examples.

    Tier-3 corpus used as the *negative* class for dialect-vs-standard
    tasks: every sample is labelled ``family="standard"`` and dialect-level
    manifest columns are deliberately ignored (NKJP text is general Polish
    regardless of the author's origin).

    ``tulip data download nkjp`` acquires it automatically: the NKJP-1M
    balanced subcorpus tarball (~163 MB, GNU GPL) is streamed, its TEI
    ``text.xml`` documents are parsed in place (no extraction to disk), and
    the paragraphs land in::

        data/raw/nkjp/
            manifest.csv

    Each source document becomes one surrogate speaker, so all text from one
    document stays in one split. Manual assembly with the same layout also
    works (columns: ``text`` required, plus ``id``/``speaker_id``).
    """

    dataset_name = "nkjp"

    auto_downloadable: ClassVar[bool] = True

    acquisition: ClassVar[str] = (
        "automatic: `tulip data download nkjp` fetches the NKJP-1M balanced "
        "subcorpus (~163 MB, GNU GPL) from clip.ipipan.waw.pl and parses its "
        "TEI documents into data/raw/nkjp/manifest.csv (see docs/datasets.md)"
    )

    columns: ClassVar[ManifestColumns] = ManifestColumns(
        family=None, dialect=None, region=None, village=None, voivodeship=None
    )
    label_defaults: ClassVar[dict[str, str]] = {"family": "standard"}

    def download(self, root: Path, **options: Any) -> None:
        """Fetch the NKJP-1M tarball and materialise ``manifest.csv``.

        The tarball is parsed member-by-member (``tar.extractfile``) rather
        than extracted, so nothing but the manifest is written and hostile
        archive paths are never materialised on disk.

        Args:
            root: Corpus directory (``data/raw/nkjp``).
            **options: ``limit`` caps the number of paragraphs; ``url``
                overrides the tarball source (``file://`` mirrors work);
                ``keep_archive=True`` retains the downloaded tarball.

        Raises:
            ConfigurationError: on unknown options.
            DataError: if the download fails, the archive is unreadable, or
                no paragraphs are extracted.
        """
        limit = options.pop("limit", None)
        url = options.pop("url", NKJP_1M_URL)
        keep_archive = options.pop("keep_archive", False)
        if options:
            raise ConfigurationError(
                f"nkjp download got unknown option(s): {', '.join(sorted(options))}"
            )

        root.mkdir(parents=True, exist_ok=True)
        manifest_path = root / "manifest.csv"
        with tempfile.TemporaryDirectory(prefix="tulip-nkjp-") as tmp:
            archive = fetch_file(url, Path(tmp) / "nkjp-1m.tar.gz", description="NKJP-1M")
            try:
                count = _write_manifest_from_archive(archive, manifest_path, limit=limit)
            except BaseException:
                manifest_path.unlink(missing_ok=True)  # never leave a partial manifest
                raise
            if keep_archive:
                archive.replace(root / archive.name)
        if count == 0:
            manifest_path.unlink(missing_ok=True)
            raise DataError(f"NKJP archive from {url} contained no parseable paragraphs")
        _logger.info("nkjp download complete: %d paragraphs -> %s", count, manifest_path)


def _write_manifest_from_archive(archive: Path, manifest_path: Path, *, limit: int | None) -> int:
    """Parse every ``text.xml`` in the tarball into manifest rows."""
    count = 0
    documents = 0
    try:
        with (
            tarfile.open(archive, "r:gz") as tar,
            manifest_path.open("w", encoding="utf-8", newline="") as handle,
        ):
            writer = csv.writer(handle)
            writer.writerow(["id", "text", "speaker_id"])
            for member in tar:
                if not member.isfile() or not member.name.endswith("/text.xml"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                document_id = Path(member.name).parent.name
                documents += 1
                for index, paragraph in enumerate(_iter_tei_paragraphs(extracted), start=1):
                    writer.writerow([f"nkjp-{document_id}-{index}", paragraph, document_id])
                    count += 1
                    if limit is not None and count >= limit:
                        return count
                if documents % 500 == 0:
                    _logger.info("nkjp download: %d documents, %d paragraphs", documents, count)
    except tarfile.TarError as exc:
        raise DataError(f"NKJP archive is not a readable tar.gz: {exc}") from exc
    return count


def _iter_tei_paragraphs(source: IO[bytes]) -> Iterator[str]:
    """Yield paragraph texts from one NKJP TEI ``text.xml`` document.

    NKJP-1M texts carry their content in ``<ab>`` (anonymous block) elements
    — some written documents use ``<p>`` — inside the TEI namespace; nested
    inline markup is flattened with ``itertext``. Undecodable or malformed
    documents are skipped with a warning rather than failing a 4000-document
    parse for one bad file.
    """
    try:
        tree = ET.parse(source)
    except ET.ParseError as exc:
        _logger.warning("skipping malformed NKJP document: %s", exc)
        return
    for tag in ("ab", "p"):
        for element in tree.iter(f"{_TEI_NS}{tag}"):
            text = " ".join("".join(element.itertext()).split())
            if text:
                yield text


__all__ = ["NKJP_1M_URL", "NkjpLoader"]
