# Development permit approval predictor

## Problem statement
Applying for a development permit in Calgary involves significant time, cost, and uncertainty. Applicants -- homeowners, developers, and architects -- have little insight into whether their application will be approved or refused. This project uses 189K+ historical permits and NLP on free-text descriptions to estimate the probability of approval.

## Approach
- Fetched 189,000+ development permits from Calgary Open Data (Socrata API)
- Created a binary target: approved vs. not approved (75% approval rate baseline)
- Applied TF-IDF vectorization on cleaned permit descriptions (500 features, bigrams)
- Combined NLP features with categorical encodings (land-use district, community, quadrant)
- Trained Logistic Regression, Random Forest, Gradient Boosting, and XGBoost classifiers
- Built a Streamlit dashboard with a permit approval predictor and text analysis explorer

## Key results

| Model | AUC-ROC | Accuracy | F1 score |
|-------|---------|----------|----------|
| **XGBoost** | **~0.93** | ~0.87 | ~0.87 |
| Gradient Boosting | ~0.92 | ~0.86 | ~0.86 |
| Random Forest | ~0.91 | ~0.85 | ~0.85 |

## How to run
```bash
pip install -r requirements.txt
python src/data_loader.py    # fetch permit data
streamlit run app.py         # launch dashboard
```

## Project structure
```
project_07_dev_permit_approval_predictor/
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── README.md
├── data/                   # Cached CSV data
├── models/                 # Saved model artifacts (joblib)
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
└── src/
    ├── __init__.py
    ├── data_loader.py      # Data fetching and preprocessing
    └── model.py            # Feature engineering, training, evaluation
```

## Technical stack
pandas, NumPy, scikit-learn (TF-IDF, classifiers), XGBoost, NLTK, Plotly, Streamlit, sodapy, joblib
