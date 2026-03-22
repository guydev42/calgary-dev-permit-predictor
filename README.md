<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=220&section=header&text=Development%20Permit%20Approval%20Predictor&fontSize=34&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=NLP%20%2B%20XGBoost%20classification%20on%20189K%2B%20Calgary%20permits&descSize=16&descAlignY=55&descColor=c8e0ff" width="100%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/XGBoost-AUC_0.93-blue?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-TF--IDF_+_ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Calgary_Open_Data-Socrata_API-orange?style=for-the-badge" />
</p>

---

## Table of contents

- [Overview](#overview)
- [Results](#results)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Quickstart](#quickstart)
- [Dataset](#dataset)
- [Tech stack](#tech-stack)
- [Methodology](#methodology)
- [Acknowledgements](#acknowledgements)

---

## Overview

> **Problem** -- Applying for a development permit in Calgary involves significant time, cost, and uncertainty. Applicants -- homeowners, developers, and architects -- have little insight into whether their application will be approved or refused.
>
> **Solution** -- This project uses 189K+ historical permits and NLP on free-text descriptions (TF-IDF vectorization with 500 features and bigrams) combined with categorical features to estimate the probability of approval using XGBoost.
>
> **Impact** -- Gives applicants a data-driven estimate of approval probability before submission, reducing wasted effort and enabling proactive application improvements.

---

## Results

| Model | AUC-ROC | Accuracy | F1 score |
|-------|---------|----------|----------|
| **XGBoost** | **~0.93** | ~0.87 | ~0.87 |
| Gradient Boosting | ~0.92 | ~0.86 | ~0.86 |
| Random Forest | ~0.91 | ~0.85 | ~0.85 |

---

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────────┐
│  Calgary Open   │────>│  189K+ permits   │────>│  TF-IDF (500)    │────>│  Classifier    │────>│  Streamlit      │
│  Data (Socrata) │     │  Text cleaning   │     │  + Categorical   │     │  training      │     │  dashboard      │
│  Dev permits    │     │  Binary target   │     │  Land-use dist   │     │  XGBoost       │     │  Approval pred  │
│                 │     │  75% baseline    │     │  Community/quad  │     │  LR / RF / GB  │     │  Text explorer  │
└─────────────────┘     └──────────────────┘     └──────────────────┘     └────────────────┘     └─────────────────┘
```

---

## Project structure

<details>
<summary>Click to expand</summary>

```
project_07_dev_permit_approval_predictor/
├── app.py                          # Streamlit dashboard
├── index.html                      # Static landing page
├── requirements.txt                # Python dependencies
├── README.md
├── data/
│   └── development_permits.csv     # Cached permit data
├── models/                         # Saved model artifacts (joblib)
├── notebooks/
│   └── 01_eda.ipynb                # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py              # Data fetching and preprocessing
    └── model.py                    # Feature engineering, training, evaluation
```

</details>

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/guydev42/dev-permit-approval-predictor.git
cd dev-permit-approval-predictor

# Install dependencies
pip install -r requirements.txt

# Fetch permit data from Calgary Open Data
python src/data_loader.py

# Launch the dashboard
streamlit run app.py
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Calgary Open Data -- Development Permits](https://data.calgary.ca/) |
| Records | 189,000+ |
| Access method | Socrata API (sodapy) |
| Key fields | Permit description (free text), land-use district, community, quadrant, decision |
| Target variable | Binary: approved vs. not approved (75% baseline approval rate) |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-189FDD?style=flat-square&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/NLTK-NLP-3776AB?style=flat-square" />
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/sodapy-Socrata_API-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/joblib-Model_Persistence-4B8BBE?style=flat-square" />
</p>

---

## Methodology

### Data ingestion and target creation

- Fetched 189,000+ development permits from Calgary Open Data via the Socrata API
- Created a binary target: approved vs. not approved (75% approval rate baseline)
- Cleaned and standardized permit descriptions for NLP processing

### NLP feature extraction

- Applied TF-IDF vectorization on cleaned permit descriptions with 500 features and bigrams
- Used NLTK for text preprocessing: lowercasing, stopword removal, and tokenization
- Combined NLP features with categorical encodings (land-use district, community, quadrant)

### Model training and evaluation

- Trained Logistic Regression, Random Forest, Gradient Boosting, and XGBoost classifiers
- Evaluated with AUC-ROC, accuracy, and F1 score
- XGBoost achieved the best AUC-ROC of ~0.93 with accuracy and F1 of ~0.87

### Feature importance analysis

- Identified top TF-IDF terms influencing approval decisions
- Land-use district and community emerged as strong categorical predictors
- Text features captured nuanced permit characteristics not available in structured fields

### Interactive dashboard

- Built a Streamlit dashboard with a permit approval predictor and text analysis explorer
- Users can input permit details and receive a probability of approval with explanation

---

## Acknowledgements

- [City of Calgary Open Data Portal](https://data.calgary.ca/) for providing development permit data
- [Socrata Open Data API](https://dev.socrata.com/) for programmatic data access

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:1e3a5f,100:2d8cf0&height=120&section=footer" width="100%" />
</p>
