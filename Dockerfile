# Clean-room environment for reproducing the tulip leaderboard from source.
#
# The image installs tulip and, by default, reproduces the committed synthetic
# leaderboard from scratch and diffs it against the committed board. Byte-exact
# matching is platform sensitive (BLAS builds differ across operating systems),
# so a mismatch here is informative rather than a failure of the pipeline: run
# the image on the platform that produced the committed board for an exact match.
# On any platform the run still proves the whole pipeline builds the dataset,
# trains, and renders the board end to end from source alone.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install from the packaging metadata and source. The synthetic leaderboard uses
# only classical models, so the base install (no torch or fasttext extras) is
# enough to regenerate it.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# The configs, the suite, and the committed board to reproduce and compare with.
COPY benchmarks ./benchmarks

# Reproduce the committed board in full isolation and diff it. Override this
# command to regenerate the board instead, for example:
#   docker run --rm -v "$PWD/out:/out" tulip \
#     tulip leaderboard benchmarks/suite.yaml --out /out
CMD ["tulip", "repro", "from-scratch", "benchmarks/suite.yaml"]
