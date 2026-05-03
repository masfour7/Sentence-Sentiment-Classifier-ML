"""Train, persist, and run inference for multiple sentiment classifiers.

Mirrors the original notebook (Logistic Regression, KNN, SVM, Naive Bayes)
but uses TF-IDF features end-to-end inside scikit-learn pipelines so that a
single fitted artefact handles preprocessing + prediction.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .data import clean_text, load_dataset

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODEL_DISPLAY = {
    "logreg": "Logistic Regression",
    "linsvm": "Linear SVM",
    "knn": "K-Nearest Neighbors",
    "nb": "Multinomial Naive Bayes",
}


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )


def _build_pipelines() -> Dict[str, Pipeline]:
    return {
        "logreg": Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", LogisticRegression(max_iter=1000, C=4.0)),
        ]),
        "linsvm": Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", LinearSVC(C=1.0)),
        ]),
        "knn": Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", KNeighborsClassifier(n_neighbors=15, metric="cosine")),
        ]),
        "nb": Pipeline([
            ("tfidf", _vectorizer()),
            ("clf", MultinomialNB()),
        ]),
    }


@dataclass
class ModelMetrics:
    key: str
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    train_seconds: float
    confusion: List[List[int]]


def train_all_models(test_size: float = 0.2, random_state: int = 42) -> Dict[str, ModelMetrics]:
    """Train every model, persist artefacts to disk, and return a metrics summary."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"].tolist(),
        df["sentiment"].to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["sentiment"],
    )

    pipelines = _build_pipelines()
    results: Dict[str, ModelMetrics] = {}

    for key, pipe in pipelines.items():
        t0 = time.time()
        pipe.fit(X_train, y_train)
        train_seconds = time.time() - t0

        preds = pipe.predict(X_test)
        cm = confusion_matrix(y_test, preds).tolist()

        metrics = ModelMetrics(
            key=key,
            name=MODEL_DISPLAY[key],
            accuracy=float(accuracy_score(y_test, preds)),
            precision=float(precision_score(y_test, preds)),
            recall=float(recall_score(y_test, preds)),
            f1=float(f1_score(y_test, preds)),
            train_seconds=float(train_seconds),
            confusion=cm,
        )
        results[key] = metrics

        joblib.dump(pipe, MODELS_DIR / f"{key}.joblib")

    METRICS_PATH.write_text(
        json.dumps({k: asdict(v) for k, v in results.items()}, indent=2)
    )
    return results


def load_models() -> Dict[str, Pipeline]:
    """Load all persisted model pipelines from disk."""
    out: Dict[str, Pipeline] = {}
    for key in MODEL_DISPLAY:
        path = MODELS_DIR / f"{key}.joblib"
        if path.exists():
            out[key] = joblib.load(path)
    return out


def load_metrics() -> Dict[str, dict]:
    if METRICS_PATH.exists():
        return json.loads(METRICS_PATH.read_text())
    return {}


def _positive_proba(pipe: Pipeline, text: str) -> float:
    """Return a 0..1 score for the positive class in a single inference pass.
    Uses predict_proba where available, otherwise a sigmoid over decision_function.
    """
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        return float(pipe.predict_proba([text])[0][1])
    if hasattr(clf, "decision_function"):
        score = float(pipe.decision_function([text])[0])
        return float(1.0 / (1.0 + np.exp(-score)))
    return float(pipe.predict([text])[0])


def predict_with_all(text: str, models: Dict[str, Pipeline]) -> List[dict]:
    """Run a single piece of text through every model and return per-model results."""
    cleaned = clean_text(text)
    rows = []
    for key, pipe in models.items():
        conf_pos = _positive_proba(pipe, cleaned)
        label = 1 if conf_pos >= 0.5 else 0
        rows.append({
            "key": key,
            "name": MODEL_DISPLAY[key],
            "label": label,
            "sentiment": "Positive" if label == 1 else "Negative",
            "confidence_positive": conf_pos,
            "confidence": conf_pos if label == 1 else 1.0 - conf_pos,
        })
    return rows


def top_features(pipe: Pipeline, n: int = 10) -> dict | None:
    """Return the top-N positive and negative features for a linear model pipeline."""
    clf = pipe.named_steps["clf"]
    vec = pipe.named_steps["tfidf"]
    if not hasattr(clf, "coef_"):
        return None
    coefs = clf.coef_[0]
    names = np.array(vec.get_feature_names_out())
    top_pos_idx = np.argsort(coefs)[-n:][::-1]
    top_neg_idx = np.argsort(coefs)[:n]
    return {
        "positive": [(names[i], float(coefs[i])) for i in top_pos_idx],
        "negative": [(names[i], float(coefs[i])) for i in top_neg_idx],
    }
