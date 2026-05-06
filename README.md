# 🎯 Sentiment Classifier — NLP Showcase

> An end-to-end NLP project that compares four classical machine-learning models on real customer reviews, wrapped in an interactive web demo.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](#license)

This is a refreshed version of a project I originally built for **CS4563 at NYU Tandon**. It takes any short piece of text — a product review, a tweet, a comment — and predicts whether the sentiment is positive or negative, using four different classical ML models trained on 3,000 labelled reviews from Amazon, IMDB and Yelp.

---

## ✨ Highlights

- 🔮 **Live interactive demo** — type any sentence and watch all four models vote in real time, with confidence bars and a side-by-side comparison.
- 📊 **Model showdown** — accuracy, precision, recall and F1 for Logistic Regression, Linear SVM, K-Nearest Neighbors and Multinomial Naive Bayes, plus confusion matrices for each.
- 🔍 **Explainability** — see the top positive and negative words each linear model latched onto.
- 🗂 **Dataset explorer** — browse the corpus, filter by source and sentiment, inspect class balance.
- 🧱 **Clean, modular codebase** — refactored from the original Colab notebook into a proper Python package with reproducible training.

## 🚀 Run it

```bash
# 1. install dependencies (uses pyproject.toml)
pip install -e .

# 2. train the models (writes pipelines to ./models/)
python train.py

# 3. launch the demo
streamlit run app.py
```

The app starts on [http://localhost:8501](http://localhost:8501) by default (or port 5000 when run inside this repo's bundled `.streamlit/config.toml`).

## 📈 Results

Trained on an 80/20 stratified split of the UCI _Sentiment Labelled Sentences_ corpus (3,000 reviews):

| Model                     | Accuracy | Precision | Recall |   F1  |
| ------------------------- | :------: | :-------: | :----: | :---: |
| **Logistic Regression**   |   0.84   |    0.83   |  0.85  |  0.84 |
| Linear SVM                |   0.83   |    0.82   |  0.84  |  0.83 |
| Multinomial Naive Bayes   |   0.82   |    0.82   |  0.82  |  0.82 |
| K-Nearest Neighbors       |   0.80   |    0.80   |  0.79  |  0.79 |

## 🏗 How it works

```
 raw reviews ──► clean_text() ──► TF-IDF (1–2 grams) ──► [LogReg │ SVM │ KNN │ NB]
                                                              │
                                                              ▼
                                                  joblib artefacts in ./models/
                                                              │
                                                              ▼
                                                Streamlit live demo (app.py)
```

1. **Data** — UCI _Sentiment Labelled Sentences_, 1,000 reviews each from Amazon, IMDB and Yelp.
2. **Preprocessing** — lowercase, strip non-alphanumerics, collapse whitespace. Shared between training and inference so there's no train/serve skew.
3. **Features** — TF-IDF over unigrams + bigrams with `min_df=2` and sublinear term frequency. Replaces the original word2vec setup with something tiny enough to ship inside a web app.
4. **Models** — four classifiers, each wrapped in its own `sklearn.Pipeline` so vectoriser and classifier are saved as one artefact.
5. **Evaluation** — accuracy, precision, recall, F1 and confusion matrices on the held-out test set; metrics persisted as JSON for the demo.

## 📁 Project layout

```
.
├── app.py                       # Streamlit demo (live UI)
├── train.py                     # Train + persist all models
├── src/sentiment/
│   ├── data.py                  # Dataset loading & text cleaning
│   └── models.py                # Pipelines, training, inference helpers
├── data/raw/                    # UCI sentiment dataset (3 .txt files)
├── models/                      # Trained pipelines + metrics.json
├── MLProject_trainedw2v.ipynb   # Original NYU notebook (kept for reference)
└── pyproject.toml
```

## 🧠 What changed from the original notebook

The original notebook was built for Google Colab and mixed exploration, training and evaluation into one long file with hard-coded Drive paths. This refresh:

- Splits everything into a small, importable Python package.
- Replaces `word2vec` averaging with TF-IDF n-grams — same idea (turn text into vectors) but a fraction of the size and dependency footprint, with comparable or better accuracy on this corpus.
- Wraps every model in a `Pipeline` so the saved artefact handles preprocessing and prediction together.
- Adds a polished web demo so the project is something you can _try_, not just read.

## 👤 Author

**Mohammad Asfour** — CS student at NYU Tandon School of Engineering.
Built as a course project for CS4563 (Introduction to Machine Learning), later refactored into this standalone showcase.

## 📜 License

MIT — feel free to fork, learn from it, or build on top.
