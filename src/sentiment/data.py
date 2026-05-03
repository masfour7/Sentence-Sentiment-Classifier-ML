"""Dataset loading and preprocessing for the UCI Sentiment Labelled Sentences corpus."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SOURCES = {
    "amazon": "amazon_cells_labelled.txt",
    "imdb": "imdb_labelled.txt",
    "yelp": "yelp_labelled.txt",
}

_CLEAN_RE = re.compile(r"[^a-z0-9'\s]")
_WS_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Light, model-friendly cleaning: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = _CLEAN_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def load_dataset(sources: Iterable[str] | None = None, data_dir: Path | None = None) -> pd.DataFrame:
    """Load the labelled sentences dataset as a DataFrame with columns:
    `review`, `sentiment` (0/1), `source`, `clean`.
    """
    sources = list(sources) if sources else list(SOURCES.keys())
    data_dir = data_dir or DATA_DIR

    frames = []
    for src in sources:
        path = data_dir / SOURCES[src]
        df = pd.read_csv(
            path,
            delimiter="\t",
            header=None,
            names=["review", "sentiment"],
            quoting=3,
        )
        df["source"] = src
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["clean"] = out["review"].astype(str).map(clean_text)
    out = out[out["clean"].str.len() > 0].reset_index(drop=True)
    return out
