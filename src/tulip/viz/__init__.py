"""Visualisation for tulip: prediction maps, charts, and embedding-space plots.

Importing this package never requires an optional dependency; folium,
matplotlib, plotly, and umap-learn are imported lazily inside the plotting
functions via :func:`tulip.utils.optional.optional_import` (extras ``viz`` and
``umap``).
"""

from __future__ import annotations

from tulip.viz.charts import (
    confusion_matrix_heatmap,
    probability_bar_chart,
    reliability_diagram,
)
from tulip.viz.embedding_space import (
    PROJECTION_COLUMNS,
    plot_embedding_space,
    project_embeddings,
)
from tulip.viz.map import confidence_heatmap, prediction_map, save_map

__all__ = [
    "PROJECTION_COLUMNS",
    "confidence_heatmap",
    "confusion_matrix_heatmap",
    "plot_embedding_space",
    "prediction_map",
    "probability_bar_chart",
    "project_embeddings",
    "reliability_diagram",
    "save_map",
]
