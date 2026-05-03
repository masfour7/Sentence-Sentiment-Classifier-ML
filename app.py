"""Interactive Streamlit demo for the sentiment classifier showcase."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.sentiment.data import load_dataset
from src.sentiment.models import (
    MODEL_DISPLAY,
    load_metrics,
    load_models,
    predict_with_all,
    top_features,
)

st.set_page_config(
    page_title="Sentiment Classifier — ML Showcase",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#6366f1"
POS = "#10b981"
NEG = "#ef4444"

st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
        h1, h2, h3 { letter-spacing: -0.01em; }
        .hero {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
            padding: 2.2rem 2.4rem; border-radius: 18px; color: white;
            box-shadow: 0 10px 40px -15px rgba(99,102,241,0.55); margin-bottom: 1.6rem;
        }
        .hero h1 { color: white !important; margin: 0 0 0.4rem 0; font-size: 2.2rem; }
        .hero p { color: rgba(255,255,255,0.9); margin: 0; font-size: 1.05rem; }
        .pill {
            display: inline-block; padding: 4px 12px; margin: 4px 6px 0 0;
            border-radius: 999px; background: rgba(255,255,255,0.15);
            color: white; font-size: 0.78rem; backdrop-filter: blur(4px);
        }
        .verdict-card {
            border-radius: 14px; padding: 1.4rem 1.6rem; color: white;
            box-shadow: 0 8px 30px -10px rgba(0,0,0,0.25);
        }
        .verdict-pos { background: linear-gradient(135deg, #10b981, #059669); }
        .verdict-neg { background: linear-gradient(135deg, #ef4444, #b91c1c); }
        .verdict-card h2 { color: white !important; margin: 0; font-size: 1.6rem; }
        .verdict-card p { color: rgba(255,255,255,0.92); margin: 4px 0 0 0; }
        .metric-card {
            background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
            padding: 1rem 1.2rem; height: 100%;
        }
        .metric-card .label { color: #6b7280; font-size: 0.78rem; text-transform: uppercase;
            letter-spacing: 0.06em; }
        .metric-card .value { font-size: 1.6rem; font-weight: 600; color: #111827; }
        .footer { color: #6b7280; font-size: 0.85rem; text-align: center; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_models():
    return load_models()


@st.cache_resource(show_spinner=False)
def get_metrics():
    return load_metrics()


@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()


models = get_models()
metrics = get_metrics()

st.markdown(
    """
    <div class="hero">
        <h1>🎯 Sentiment Classifier</h1>
        <p>An end-to-end NLP project that compares four classical machine-learning models on real customer reviews from Amazon, IMDB, and Yelp.</p>
        <div>
            <span class="pill">scikit-learn</span>
            <span class="pill">TF-IDF + n-grams</span>
            <span class="pill">Logistic Regression · SVM · KNN · Naive Bayes</span>
            <span class="pill">3,000 labelled sentences</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### About")
    st.write(
        "This is a refreshed version of my CS4563 project at **NYU Tandon**. "
        "It classifies short reviews as positive or negative using classical NLP."
    )
    st.markdown("**Pipeline**")
    st.markdown(
        "- Clean & normalise text\n"
        "- Encode with TF-IDF (1–2 grams)\n"
        "- Train 4 classifiers\n"
        "- Compare & visualise results"
    )
    st.markdown("**Dataset**")
    st.write("UCI _Sentiment Labelled Sentences_ — 3,000 reviews from Amazon, IMDB, Yelp.")
    st.markdown("---")
    st.caption("Built with scikit-learn + Streamlit")

tab_demo, tab_perf, tab_explore, tab_about = st.tabs(
    ["🔮 Live Demo", "📊 Model Comparison", "🔍 Dataset Explorer", "📚 How it works"]
)

# ─────────────────────────────────────────────────────────────────────
# Live demo
# ─────────────────────────────────────────────────────────────────────
with tab_demo:
    st.subheader("Try it on any sentence")
    st.caption("Type a review or pick an example. All four models will weigh in.")

    examples = [
        "Battery life is fantastic and the camera blew me away.",
        "The food was cold and the waiter was rude.",
        "Honestly the best movie I've seen all year — pure magic.",
        "Cheap build quality, broke after two days.",
        "It's okay, nothing special but not bad either.",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        if col.button(ex[:32] + ("…" if len(ex) > 32 else ""), use_container_width=True):
            st.session_state["text_input"] = ex

    text = st.text_area(
        "Your text",
        value=st.session_state.get("text_input", "I absolutely love this product, it works perfectly!"),
        height=110,
        key="text_input",
    )

    if not models:
        st.error("No trained models found. Run `python train.py` first.")
        st.stop()

    if text.strip():
        rows = predict_with_all(text, models)
        df = pd.DataFrame(rows)

        avg_pos = float(df["confidence_positive"].mean())
        votes_pos = int((df["label"] == 1).sum())
        verdict_pos = avg_pos >= 0.5

        col_v, col_chart = st.columns([1, 1.4])
        with col_v:
            klass = "verdict-pos" if verdict_pos else "verdict-neg"
            verdict_text = "Positive sentiment" if verdict_pos else "Negative sentiment"
            emoji = "😊" if verdict_pos else "😞"
            score_pct = avg_pos * 100 if verdict_pos else (1 - avg_pos) * 100
            st.markdown(
                f"""
                <div class="verdict-card {klass}">
                    <h2>{emoji} {verdict_text}</h2>
                    <p><strong>{score_pct:.1f}%</strong> average confidence ·
                    <strong>{votes_pos}/{len(df)}</strong> models agree it's positive</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_chart:
            chart_df = df.assign(
                pos=lambda d: d["confidence_positive"] * 100,
                neg=lambda d: (1 - d["confidence_positive"]) * 100,
            )
            base = alt.Chart(chart_df).encode(
                y=alt.Y("name:N", title=None, sort=list(MODEL_DISPLAY.values())),
            )
            bars = base.mark_bar(size=22).encode(
                x=alt.X("pos:Q", title="Positive probability (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.condition(
                    alt.datum.pos >= 50,
                    alt.value(POS),
                    alt.value(NEG),
                ),
                tooltip=["name", alt.Tooltip("pos:Q", format=".1f", title="P(positive) %")],
            )
            rule = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(
                color="#9ca3af", strokeDash=[4, 4]
            ).encode(x="x:Q")
            st.altair_chart((bars + rule).properties(height=220), use_container_width=True)

        st.markdown("#### Per-model verdicts")
        view = df[["name", "sentiment", "confidence"]].rename(
            columns={"name": "Model", "sentiment": "Prediction", "confidence": "Confidence"}
        )
        st.dataframe(
            view,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                )
            },
        )

# ─────────────────────────────────────────────────────────────────────
# Model comparison
# ─────────────────────────────────────────────────────────────────────
with tab_perf:
    st.subheader("How the four models compare")
    if not metrics:
        st.warning("Metrics not available — run `python train.py`.")
    else:
        m_df = pd.DataFrame(metrics).T.reset_index(drop=True)
        m_df = m_df[["name", "accuracy", "precision", "recall", "f1", "train_seconds"]]
        best = m_df.loc[m_df["accuracy"].idxmax()]

        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in [
            (c1, "Best model", best["name"]),
            (c2, "Best accuracy", f"{best['accuracy']*100:.1f}%"),
            (c3, "Models trained", str(len(m_df))),
            (c4, "Total samples", f"{len(get_dataset()):,}"),
        ]:
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value">{value}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("#### Headline metrics")
        long_df = m_df.melt(
            id_vars="name",
            value_vars=["accuracy", "precision", "recall", "f1"],
            var_name="metric",
            value_name="score",
        )
        chart = (
            alt.Chart(long_df)
            .mark_bar()
            .encode(
                x=alt.X("metric:N", title=None),
                y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1]), title="Score"),
                color=alt.Color("metric:N", legend=None,
                                scale=alt.Scale(range=["#6366f1", "#8b5cf6", "#ec4899", "#f59e0b"])),
                column=alt.Column("name:N", title=None, header=alt.Header(labelFontSize=12)),
                tooltip=["name", "metric", alt.Tooltip("score:Q", format=".3f")],
            )
            .properties(width=120, height=240)
        )
        st.altair_chart(chart, use_container_width=False)

        st.markdown("#### Confusion matrices (test set)")
        cm_cols = st.columns(len(metrics))
        for col, (key, m) in zip(cm_cols, metrics.items()):
            cm = np.array(m["confusion"])
            cm_df = pd.DataFrame(
                [
                    {"actual": "Negative", "predicted": "Negative", "count": int(cm[0, 0])},
                    {"actual": "Negative", "predicted": "Positive", "count": int(cm[0, 1])},
                    {"actual": "Positive", "predicted": "Negative", "count": int(cm[1, 0])},
                    {"actual": "Positive", "predicted": "Positive", "count": int(cm[1, 1])},
                ]
            )
            heat = (
                alt.Chart(cm_df)
                .mark_rect()
                .encode(
                    x=alt.X("predicted:N", title="Predicted"),
                    y=alt.Y("actual:N", title="Actual"),
                    color=alt.Color("count:Q", scale=alt.Scale(scheme="purples"), legend=None),
                    tooltip=["actual", "predicted", "count"],
                )
            )
            text = (
                alt.Chart(cm_df)
                .mark_text(fontSize=14, fontWeight="bold")
                .encode(
                    x="predicted:N",
                    y="actual:N",
                    text="count:Q",
                    color=alt.condition("datum.count > 100", alt.value("white"), alt.value("#374151")),
                )
            )
            with col:
                st.markdown(f"**{m['name']}**")
                st.altair_chart((heat + text).properties(height=180), use_container_width=True)

        st.markdown("#### Words the linear models latched onto")
        for key in ("logreg", "linsvm"):
            pipe = models.get(key)
            if pipe is None:
                continue
            feats = top_features(pipe, n=10)
            if feats is None:
                continue
            st.markdown(f"**{MODEL_DISPLAY[key]}**")
            f_df = pd.DataFrame(
                [{"word": w, "weight": s, "polarity": "Positive"} for w, s in feats["positive"]] +
                [{"word": w, "weight": s, "polarity": "Negative"} for w, s in feats["negative"]]
            )
            chart_f = (
                alt.Chart(f_df)
                .mark_bar()
                .encode(
                    x=alt.X("weight:Q", title="Coefficient"),
                    y=alt.Y("word:N", sort="-x", title=None),
                    color=alt.Color(
                        "polarity:N",
                        scale=alt.Scale(domain=["Positive", "Negative"], range=[POS, NEG]),
                        legend=None,
                    ),
                    tooltip=["word", alt.Tooltip("weight:Q", format=".3f")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart_f, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# Dataset explorer
# ─────────────────────────────────────────────────────────────────────
with tab_explore:
    df = get_dataset()
    st.subheader("Dataset at a glance")
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="metric-card"><div class="label">Total samples</div>'
        f'<div class="value">{len(df):,}</div></div>', unsafe_allow_html=True)
    c2.markdown(
        f'<div class="metric-card"><div class="label">Sources</div>'
        f'<div class="value">{df["source"].nunique()}</div></div>', unsafe_allow_html=True)
    c3.markdown(
        f'<div class="metric-card"><div class="label">Class balance</div>'
        f'<div class="value">{df["sentiment"].mean()*100:.1f}% pos</div></div>',
        unsafe_allow_html=True)

    st.markdown("#### Source × sentiment breakdown")
    bd = (
        df.groupby(["source", "sentiment"]).size().reset_index(name="count")
        .assign(sentiment=lambda d: d["sentiment"].map({0: "Negative", 1: "Positive"}))
    )
    st.altair_chart(
        alt.Chart(bd).mark_bar().encode(
            x=alt.X("source:N", title=None),
            y=alt.Y("count:Q"),
            color=alt.Color(
                "sentiment:N",
                scale=alt.Scale(domain=["Negative", "Positive"], range=[NEG, POS]),
            ),
            tooltip=["source", "sentiment", "count"],
        ).properties(height=260),
        use_container_width=True,
    )

    st.markdown("#### Browse samples")
    src_filter = st.multiselect("Source", sorted(df["source"].unique()), default=sorted(df["source"].unique()))
    sent_filter = st.radio("Sentiment", ["All", "Positive", "Negative"], horizontal=True)
    sub = df[df["source"].isin(src_filter)]
    if sent_filter == "Positive":
        sub = sub[sub["sentiment"] == 1]
    elif sent_filter == "Negative":
        sub = sub[sub["sentiment"] == 0]
    st.dataframe(
        sub[["source", "sentiment", "review"]].sample(min(100, len(sub)), random_state=1),
        hide_index=True, use_container_width=True, height=380,
    )

# ─────────────────────────────────────────────────────────────────────
# About
# ─────────────────────────────────────────────────────────────────────
with tab_about:
    st.subheader("How it works")
    st.markdown(
        """
**1. Data.** 3,000 short reviews (1,000 each from Amazon, IMDB and Yelp) labelled
positive/negative — the UCI _Sentiment Labelled Sentences_ corpus.

**2. Preprocessing.** Lowercase, strip non-alphanumeric characters, collapse
whitespace. Done in a single `clean_text()` function so the demo and training
share the exact same pipeline.

**3. Features.** Each review is encoded with a TF-IDF vectorizer over unigrams
and bigrams (`min_df=2`, sublinear term frequency). This swaps in cleanly for
the original word2vec embeddings while staying tiny enough to ship inside a web
app.

**4. Models.** Four classical classifiers from the original notebook:
Logistic Regression, Linear SVM, K-Nearest Neighbors and Multinomial Naive
Bayes — each wrapped in its own scikit-learn `Pipeline` so vectoriser +
classifier are persisted as one artefact.

**5. Evaluation.** 80/20 stratified split, reporting accuracy, precision,
recall and F1 plus per-model confusion matrices.

**6. Showcase.** This Streamlit app loads the persisted pipelines and runs every
sentence you type through all four models in real time.
        """
    )

st.markdown(
    '<div class="footer">Originally built at NYU Tandon · refreshed and open-sourced</div>',
    unsafe_allow_html=True,
)
