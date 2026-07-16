# Examples

Runnable, self-contained examples. Every one uses the in-memory `synthetic`
corpus, so they run on a **core install** with no downloads and no heavy extras:

```bash
pip install -e .
python examples/train_and_predict.py
python examples/compare_models.py
```

| File | What it shows |
| :--- | :--- |
| `train_and_predict.py` | Generate a corpus, split it speaker-disjoint, train a classifier, evaluate on the held-out split, and classify a fresh sentence. |
| `compare_models.py` | Train four classical models on one identical frozen split and compare macro-F1. |
| `tutorial.ipynb` | The same walkthrough as a notebook, with commentary. Open it with `jupyter lab examples/tutorial.ipynb`. |

> The synthetic corpus is a generated fixture, not real speech. These scores
> measure whether the machinery works, not real dialect accuracy. Assemble a
> real corpus under `data/raw/` (see [docs/datasets.md](../docs/datasets.md)) and
> point a config at it for meaningful results, then run `tulip train <config>`.

## Where to go next

- `tulip --help` for the full command surface.
- `benchmarks/README.md` for the reproducible leaderboard.
- `docs/` (built with `mkdocs serve`) for the guide and API reference.
