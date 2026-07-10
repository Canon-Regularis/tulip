"""Corpus loaders; importing this package registers them all in DATASETS."""

from tulip.data.loaders._base import ManifestBackedLoader
from tulip.data.loaders.bigos import BigosLoader
from tulip.data.loaders.common_voice import CommonVoiceLoader
from tulip.data.loaders.dgp import DgpLoader
from tulip.data.loaders.dialektarium import DialektariumLoader
from tulip.data.loaders.korpus_spiski import KorpusSpiskiLoader
from tulip.data.loaders.mackowce import MackowceLoader
from tulip.data.loaders.manifest_loader import GenericManifestLoader
from tulip.data.loaders.nkjp import NkjpLoader
from tulip.data.loaders.spokes import SpokesLoader
from tulip.data.loaders.synthetic import SyntheticLoader

__all__ = [
    "BigosLoader",
    "CommonVoiceLoader",
    "DgpLoader",
    "DialektariumLoader",
    "GenericManifestLoader",
    "KorpusSpiskiLoader",
    "MackowceLoader",
    "ManifestBackedLoader",
    "NkjpLoader",
    "SpokesLoader",
    "SyntheticLoader",
]
