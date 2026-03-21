"""
Streamlit application for Calgary Development Permit Approval Predictor.

Provides an interactive dashboard to explore development permit data,
predict approval probability for new permit descriptions, examine NLP
insights, and compare model performance.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure src/ is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_and_preprocess  # noqa: E402
from src.model import (  # noqa: E402
    CLASSIFIERS,
    FeatureBuilder,
    get_feature_importance,
    get_tfidf_importance,
    load_artifacts,
    save_artifacts,
    split_data,
    train_and_evaluate,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Calgary Development Permit Approval Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading development permit data...")
def get_data() -> pd.DataFrame:
    """Load and preprocess the development permits dataset."""
    return load_and_preprocess(use_cache=True)


@st.cache_resource(show_spinner="Training models (this may take a minute)...")
def get_trained_models(_df: pd.DataFrame):
    """Train all classifiers and return results dict + feature builder."""
    results = train_and_evaluate(_df)
    # Pick best model by AUC and persist
    best_name = max(results, key=lambda k: results[k]["metrics"]["auc_roc"])
    best = results[best_name]
    save_artifacts(best["model"], best["feature_builder"], "best_model")
    return results, best_name


@st.cache_resource(show_spinner="Loading saved model...")
def get_saved_model():
    """Attempt to load a previously saved model."""
    try:
        artifacts = load_artifacts("best_model")
        return artifacts["model"], artifacts["feature_builder"]
    except FileNotFoundError:
        return None, None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Permit Dashboard",
        "Approval Predictor",
        "NLP Insights",
        "Model Performance",
        "About",
    ],
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = get_data()

# ===================================================================
# PAGE: Permit Dashboard
# ===================================================================
if page == "Permit Dashboard":
    st.title("Calgary Development Permit Dashboard")
    st.markdown(
        "An overview of **188,000+** development permits issued by the "
        "City of Calgary, dating back to 1979."
    )

    # --- KPI row -------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    total_permits = len(df)
    approval_rate = df["approved"].mean() if "approved" in df.columns else 0
    n_communities = df["communityname"].nunique() if "communityname" in df.columns else 0
    year_range = ""
    if "applied_year" in df.columns:
        min_yr = int(df["applied_year"].min())
        max_yr = int(df["applied_year"].max())
        year_range = f"{min_yr} - {max_yr}"

    col1.metric("Total Permits", f"{total_permits:,}")
    col2.metric("Approval Rate", f"{approval_rate:.1%}")
    col3.metric("Communities", f"{n_communities:,}")
    col4.metric("Year Range", year_range)

    st.markdown("---")

    # --- Permits by status (pie) ----------------------------------------
    left, right = st.columns(2)
    with left:
        st.subheader("Permits by Current Status")
        if "statuscurrent" in df.columns:
            status_counts = (
                df["statuscurrent"]
                .value_counts()
                .head(10)
                .reset_index()
            )
            status_counts.columns = ["Status", "Count"]
            fig_pie = px.pie(
                status_counts,
                names="Status",
                values="Count",
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

    # --- Permits over time (line) ----------------------------------------
    with right:
        st.subheader("Permits Over Time")
        if "applied_year" in df.columns:
            yearly = (
                df.groupby("applied_year")
                .size()
                .reset_index(name="Count")
            )
            yearly = yearly[yearly["applied_year"] > 0]
            fig_line = px.line(
                yearly,
                x="applied_year",
                y="Count",
                labels={"applied_year": "Year", "Count": "Number of Permits"},
                markers=True,
            )
            fig_line.update_layout(margin=dict(t=30, b=10))
            st.plotly_chart(fig_line, use_container_width=True)

    # --- Top communities by volume --------------------------------------
    st.subheader("Top 20 Communities by Permit Volume")
    if "communityname" in df.columns:
        top_comm = (
            df["communityname"]
            .value_counts()
            .head(20)
            .reset_index()
        )
        top_comm.columns = ["Community", "Permits"]
        fig_bar = px.bar(
            top_comm,
            x="Permits",
            y="Community",
            orientation="h",
            color="Permits",
            color_continuous_scale="Tealgrn",
        )
        fig_bar.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10),
            height=550,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ===================================================================
# PAGE: Approval Predictor
# ===================================================================
elif page == "Approval Predictor":
    st.title("Development Permit Approval Predictor")
    st.markdown(
        "Enter details about a development permit application and the model "
        "will estimate the probability of approval."
    )

    # Try loading a saved model first
    model, fb = get_saved_model()

    if model is None:
        st.info(
            "No trained model found. Go to **Model Performance** to train "
            "models first, or one will be trained now."
        )
        results, best_name = get_trained_models(df)
        model = results[best_name]["model"]
        fb = results[best_name]["feature_builder"]

    # --- Input form ------------------------------------------------------
    with st.form("predict_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            categories = sorted(df["category"].dropna().unique().tolist())
            selected_category = st.selectbox("Permit Category", categories)

            districts = sorted(df["landusedistrict"].dropna().unique().tolist())
            selected_district = st.selectbox("Land Use District", districts)

            communities = sorted(df["communityname"].dropna().unique().tolist())
            selected_community = st.selectbox("Community", communities)

        with col_b:
            quadrants = sorted(df["quadrant"].dropna().unique().tolist())
            selected_quadrant = st.selectbox("Quadrant", quadrants)

            perm_disc_options = ["Permitted", "Discretionary"]
            if "permitteddiscretionary" in df.columns:
                perm_disc_options = sorted(
                    df["permitteddiscretionary"].dropna().unique().tolist()
                )
            selected_perm_disc = st.selectbox(
                "Permitted / Discretionary", perm_disc_options
            )

            description_input = st.text_area(
                "Permit Description",
                height=120,
                placeholder="e.g. New single detached house with secondary suite and detached garage...",
            )

        submitted = st.form_submit_button("Predict Approval")

    if submitted:
        from src.data_loader import clean_text

        input_row = pd.DataFrame(
            [
                {
                    "category": selected_category,
                    "landusedistrict": selected_district,
                    "communityname": selected_community,
                    "quadrant": selected_quadrant,
                    "permitteddiscretionary": selected_perm_disc,
                    "description_clean": clean_text(description_input),
                    "applied_year": 2025,
                    "applied_month": 1,
                    "applied_day_of_week": 2,
                }
            ]
        )

        try:
            X_input = fb.transform(input_row)
            proba = model.predict_proba(X_input)[0][1]
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            proba = None

        if proba is not None:
            st.markdown("---")
            st.subheader("Prediction Result")

            # Gauge chart
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=proba * 100,
                    number={"suffix": "%"},
                    title={"text": "Approval Probability"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#2ecc71" if proba >= 0.5 else "#e74c3c"},
                        "steps": [
                            {"range": [0, 40], "color": "#fadbd8"},
                            {"range": [40, 60], "color": "#fdebd0"},
                            {"range": [60, 100], "color": "#d5f5e3"},
                        ],
                        "threshold": {
                            "line": {"color": "black", "width": 4},
                            "thickness": 0.75,
                            "value": 50,
                        },
                    },
                )
            )
            fig_gauge.update_layout(height=350, margin=dict(t=60, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            verdict = "LIKELY APPROVED" if proba >= 0.5 else "LIKELY NOT APPROVED"
            colour = "green" if proba >= 0.5 else "red"
            st.markdown(
                f"### Verdict: :{colour}[{verdict}]"
            )

# ===================================================================
# PAGE: NLP Insights
# ===================================================================
elif page == "NLP Insights":
    st.title("NLP Insights")
    st.markdown(
        "Explore how the language in permit descriptions relates to "
        "approval outcomes."
    )

    # Ensure models are trained
    model, fb = get_saved_model()
    if model is None:
        results, best_name = get_trained_models(df)
        model = results[best_name]["model"]
        fb = results[best_name]["feature_builder"]

    # --- Top TF-IDF terms for approval vs refusal -----------------------
    st.subheader("Top Terms Associated with Approval vs Refusal")

    approval_terms, refusal_terms = get_tfidf_importance(model, fb, top_n=20)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**Terms Associated with Approval**")
        if not approval_terms.empty:
            fig_app = px.bar(
                approval_terms,
                x="weight",
                y="term",
                orientation="h",
                color="weight",
                color_continuous_scale="Greens",
                labels={"weight": "Weight", "term": "Term"},
            )
            fig_app.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(t=10, b=10),
                height=500,
                showlegend=False,
            )
            st.plotly_chart(fig_app, use_container_width=True)
        else:
            st.info("No approval-term data available.")

    with col_r:
        st.markdown("**Terms Associated with Refusal**")
        if not refusal_terms.empty:
            fig_ref = px.bar(
                refusal_terms,
                x="weight",
                y="term",
                orientation="h",
                color="weight",
                color_continuous_scale="Reds",
                labels={"weight": "Weight", "term": "Term"},
            )
            fig_ref.update_layout(
                yaxis=dict(autorange="reversed"),
                margin=dict(t=10, b=10),
                height=500,
                showlegend=False,
            )
            st.plotly_chart(fig_ref, use_container_width=True)
        else:
            st.info("No refusal-term data available.")

    # --- Word frequency analysis ----------------------------------------
    st.markdown("---")
    st.subheader("Word Frequency Analysis")

    if "description_clean" in df.columns and "approved" in df.columns:
        from sklearn.feature_extraction.text import CountVectorizer

        status_choice = st.radio(
            "Show word frequencies for:",
            ["Approved Permits", "Refused Permits", "All Permits"],
            horizontal=True,
        )
        if status_choice == "Approved Permits":
            subset = df[df["approved"] == 1]
        elif status_choice == "Refused Permits":
            subset = df[df["approved"] == 0]
        else:
            subset = df

        corpus = subset["description_clean"].fillna("").astype(str)
        cv = CountVectorizer(stop_words="english", max_features=30)
        word_matrix = cv.fit_transform(corpus)
        word_counts = pd.DataFrame(
            {
                "Word": cv.get_feature_names_out(),
                "Frequency": word_matrix.sum(axis=0).A1,
            }
        ).sort_values("Frequency", ascending=False)

        fig_wf = px.bar(
            word_counts,
            x="Frequency",
            y="Word",
            orientation="h",
            color="Frequency",
            color_continuous_scale="Viridis",
        )
        fig_wf.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10),
            height=550,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

    # --- TF-IDF feature importance overall ------------------------------
    st.markdown("---")
    st.subheader("TF-IDF Feature Importance")

    feature_names = fb.get_feature_names()
    imp_df = get_feature_importance(model, feature_names, top_n=30)
    tfidf_imp = imp_df[imp_df["feature"].str.startswith("tfidf__")].copy()
    tfidf_imp["feature"] = tfidf_imp["feature"].str.replace("tfidf__", "", regex=False)

    if not tfidf_imp.empty:
        fig_imp = px.bar(
            tfidf_imp,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Sunset",
            labels={"importance": "Importance", "feature": "TF-IDF Term"},
        )
        fig_imp.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10),
            height=500,
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No TF-IDF importance data to display.")

# ===================================================================
# PAGE: Model Performance
# ===================================================================
elif page == "Model Performance":
    st.title("Model Performance Comparison")
    st.markdown(
        "Train and compare multiple classifiers on development permit "
        "approval prediction."
    )

    if st.button("Train / Retrain All Models"):
        st.cache_resource.clear()

    results, best_name = get_trained_models(df)

    # --- Model comparison table -----------------------------------------
    st.subheader("Metrics Comparison")
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append(
            {
                "Model": name,
                "Accuracy": f"{m['accuracy']:.4f}",
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1 Score": f"{m['f1']:.4f}",
                "AUC-ROC": f"{m['auc_roc']:.4f}",
            }
        )
    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.success(f"Best model by AUC-ROC: **{best_name}**")

    # --- ROC curves ------------------------------------------------------
    st.subheader("ROC Curves")
    fig_roc = go.Figure()
    for name, res in results.items():
        m = res["metrics"]
        fig_roc.add_trace(
            go.Scatter(
                x=m["fpr"],
                y=m["tpr"],
                mode="lines",
                name=f"{name} (AUC={m['auc_roc']:.3f})",
            )
        )
    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="grey"),
            name="Random",
        )
    )
    fig_roc.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=500,
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # --- Confusion matrix heatmap ---------------------------------------
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox(
        "Select model for confusion matrix", list(results.keys())
    )
    cm = results[selected_model]["metrics"]["confusion_matrix"]
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Not Approved", "Approved"],
        y=["Not Approved", "Approved"],
        color_continuous_scale="Blues",
    )
    fig_cm.update_layout(height=400, margin=dict(t=30, b=10))
    st.plotly_chart(fig_cm, use_container_width=True)

    # --- Feature importance -----------------------------------------------
    st.subheader("Feature Importance")
    fb = results[selected_model]["feature_builder"]
    feature_names = fb.get_feature_names()
    imp_df = get_feature_importance(
        results[selected_model]["model"], feature_names, top_n=25
    )

    if not imp_df.empty:
        fig_fi = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            color="abs_importance",
            color_continuous_scale="Bluered",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig_fi.update_layout(
            yaxis=dict(autorange="reversed"),
            margin=dict(t=10, b=10),
            height=600,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

# ===================================================================
# PAGE: About
# ===================================================================
elif page == "About":
    st.title("About This Project")

    st.markdown(
        """
        ## Development Permit Approval Predictor

        **Problem Statement**

        Applying for a development permit in Calgary involves time,
        cost, and uncertainty.  Applicants -- homeowners, developers,
        and architects -- often have little insight into whether their
        application is likely to be approved.  This project builds a
        machine-learning model that estimates the probability of
        approval based on historical permit data and the text of the
        permit description.

        ---

        ### Dataset

        | Detail | Value |
        |--------|-------|
        | Source | [Calgary Open Data -- Development Permits](https://data.calgary.ca/Business-and-Financial-Services/Development-Permits/6933-unw5) |
        | Records | ~188,653 |
        | Columns | 40 |
        | Time span | 1979 -- present |

        Key columns include `permitnum`, `applieddate`, `statuscurrent`,
        `category`, `description`, `proposedusecode`, `landusedistrict`,
        `communityname`, `quadrant`, `latitude`, `longitude`, and more.

        ---

        ### Methodology

        1. **Data Collection** -- Fetched via the Socrata API (`sodapy`)
           and cached locally as CSV.
        2. **Preprocessing** -- Date parsing, text cleaning (lower-case,
           remove HTML/special characters), and binary target creation
           (*approved* vs *not approved*).
        3. **NLP Feature Engineering** -- TF-IDF vectorisation of cleaned
           descriptions (unigrams + bigrams, max 500 features, English
           stop-words removed).
        4. **Categorical & Numerical Features** -- Label-encoded permit
           category, land-use district, community, quadrant, and
           permitted/discretionary flag; plus temporal features (year,
           month, day of week).
        5. **Modelling** -- Four classifiers trained and compared:
           - Logistic Regression
           - Random Forest
           - Gradient Boosting
           - XGBoost
        6. **Evaluation** -- Accuracy, Precision, Recall, F1, and
           AUC-ROC on a stratified 80/20 split.

        ---

        ### NLP Pipeline Explanation

        The *description* field on each permit contains free-text
        written by the applicant (e.g., "New single detached house
        with secondary suite").  The NLP pipeline:

        * **Cleans** the text: lower-case, strip HTML tags, remove
          non-alphabetic characters, collapse whitespace.
        * **Vectorises** with scikit-learn's `TfidfVectorizer`:
          - Unigrams and bigrams (`ngram_range=(1,2)`).
          - Maximum 500 features to balance signal and dimensionality.
          - Minimum document frequency of 5 to remove very rare terms.
          - Maximum document frequency of 95 % to remove ubiquitous terms.
          - English stop-words removed.
        * The resulting sparse TF-IDF matrix is concatenated with the
          categorical and numerical features before being fed to the
          classifier.

        ---

        ### How to Run

        ```bash
        pip install -r requirements.txt
        streamlit run app.py
        ```

        ---

        *Built with Python, scikit-learn, XGBoost, Plotly, and Streamlit.*
        """
    )
