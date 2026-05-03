# Sentiment Classifier — NLP Showcase

## Overview
A polished sentiment analysis project (originally CS4563 @ NYU Tandon). It classifies short reviews as positive/negative using four classical ML models trained on the UCI Sentiment Labelled Sentences corpus, and serves an interactive Streamlit demo.

## Architecture
- **`app.py`** — Streamlit web demo on port 5000 (live predictions, model comparison, dataset explorer, "how it works")
- **`train.py`** — CLI script to train all four models and persist them
- **`src/sentiment/data.py`** — Dataset loading + shared `clean_text` preprocessing
- **`src/sentiment/models.py`** — Pipeline definitions, training, inference (`predict_with_all`), feature explainability (`top_features`)
- **`data/raw/`** — UCI dataset (Amazon, IMDB, Yelp .txt files, 1k each)
- **`models/`** — Persisted joblib pipelines + `metrics.json`
- **`MLProject_trainedw2v.ipynb`** — Original Colab notebook (kept for reference)
- **`.streamlit/config.toml`** — Streamlit server config (port 5000, allow remote)

## Models
Four scikit-learn pipelines (TF-IDF unigrams+bigrams → classifier):
- Logistic Regression — best (~84% accuracy)
- Linear SVM — ~83%
- Multinomial Naive Bayes — ~82%
- K-Nearest Neighbors (cosine) — ~80%

## Running
- Workflow `Start application` runs `streamlit run app.py` on port 5000.
- Retrain anytime via `python train.py`.

## Deployment
Configured for `vm` (always-running) since Streamlit holds an in-memory session.

## Dependencies (pyproject.toml)
streamlit, scikit-learn, pandas, numpy, altair, joblib, matplotlib, gensim, jupyter/notebook (kept for the original notebook).
