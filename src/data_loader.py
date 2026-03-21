"""
Data loader for Calgary Development Permits dataset.

Fetches development permit data from Calgary Open Data portal via the
Socrata API (sodapy), caches it locally, and preprocesses it for
modelling. The target variable is a binary indicator of whether a
permit was approved.

Dataset: Development Permits
Socrata ID: 6933-unw5
Records: ~188,653 | Columns: 40
"""

import os
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sodapy import Socrata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SOCRATA_DOMAIN = "data.calgary.ca"
DATASET_ID = "6933-unw5"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_FILE = DATA_DIR / "development_permits.csv"
RECORD_LIMIT = 200_000  # slightly above 188,653 to capture everything

# Statuses that count as "approved"
APPROVED_STATUSES = [
    "Approved",
    "Approved - Conditions",
    "Approved - Active",
    "Released",
]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------
def fetch_data(use_cache: bool = True) -> pd.DataFrame:
    """Fetch development-permit data from Calgary Open Data.

    Parameters
    ----------
    use_cache : bool
        If *True* and a local CSV already exists, load from disk instead
        of hitting the API.

    Returns
    -------
    pd.DataFrame
        Raw data with original column names (lower-cased).
    """
    if use_cache and CACHE_FILE.exists():
        logger.info("Loading cached data from %s", CACHE_FILE)
        return pd.read_csv(CACHE_FILE, low_memory=False)

    logger.info("Fetching data from Calgary Open Data (dataset %s)...", DATASET_ID)
    try:
        client = Socrata(SOCRATA_DOMAIN, None, timeout=60)
        results = client.get(DATASET_ID, limit=RECORD_LIMIT)
        client.close()
        df = pd.DataFrame.from_records(results)

        # Ensure data/ directory exists
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_FILE, index=False)
        logger.info("Fetched and cached %d records to %s", len(df), CACHE_FILE)
        return df
    except Exception as exc:
        logger.error("Failed to fetch data from Socrata API: %s", exc)
        if CACHE_FILE.exists():
            logger.warning("Falling back to cached data.")
            return pd.read_csv(CACHE_FILE, low_memory=False)
        raise


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Clean a single description string for downstream NLP.

    Steps:
    - Lower-case
    - Remove HTML tags
    - Remove special characters / digits
    - Collapse whitespace
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)          # non-alpha
    text = re.sub(r"\s+", " ", text).strip()        # whitespace
    return text


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the raw dataframe for modelling.

    Operations
    ----------
    1. Parse *applieddate* and extract temporal features.
    2. Create the binary target *approved*.
    3. Clean the *description* field.
    4. Derive permit-category, land-use-district, and community features.
    5. Drop rows where essential columns are entirely missing.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering / modelling.
    """
    df = df.copy()

    # --- Standardise column names --------------------------------------
    df.columns = [c.strip().lower() for c in df.columns]

    # --- Date features --------------------------------------------------
    if "applieddate" in df.columns:
        df["applieddate"] = pd.to_datetime(df["applieddate"], errors="coerce")
        df["applied_year"] = df["applieddate"].dt.year
        df["applied_month"] = df["applieddate"].dt.month
        df["applied_day_of_week"] = df["applieddate"].dt.dayofweek  # 0=Mon
    else:
        logger.warning("'applieddate' column not found; skipping date features.")

    # --- Binary target --------------------------------------------------
    if "statuscurrent" in df.columns:
        df["statuscurrent"] = df["statuscurrent"].astype(str).str.strip()
        df["approved"] = (
            df["statuscurrent"]
            .str.lower()
            .apply(lambda s: int(any(a.lower() in s for a in APPROVED_STATUSES)))
        )
    else:
        logger.warning("'statuscurrent' column not found; cannot create target.")

    # --- Clean description text -----------------------------------------
    if "description" in df.columns:
        df["description_clean"] = df["description"].apply(clean_text)
    else:
        df["description_clean"] = ""

    # --- Permit category ------------------------------------------------
    if "category" in df.columns:
        df["category"] = df["category"].astype(str).str.strip()
    else:
        df["category"] = "Unknown"

    # --- Land use district ----------------------------------------------
    if "landusedistrict" in df.columns:
        df["landusedistrict"] = df["landusedistrict"].astype(str).str.strip()
    else:
        df["landusedistrict"] = "Unknown"

    if "landusedistrictdescription" in df.columns:
        df["landusedistrictdescription"] = (
            df["landusedistrictdescription"].astype(str).str.strip()
        )

    # --- Community ------------------------------------------------------
    if "communityname" in df.columns:
        df["communityname"] = df["communityname"].astype(str).str.strip()
    else:
        df["communityname"] = "Unknown"

    # --- Quadrant -------------------------------------------------------
    if "quadrant" in df.columns:
        df["quadrant"] = df["quadrant"].astype(str).str.strip().str.upper()
    else:
        df["quadrant"] = "Unknown"

    # --- Permitted vs discretionary -------------------------------------
    if "permitteddiscretionary" in df.columns:
        df["permitteddiscretionary"] = (
            df["permitteddiscretionary"].astype(str).str.strip()
        )

    # --- Latitude / Longitude -------------------------------------------
    for col in ("latitude", "longitude"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Drop rows missing critical columns ----------------------------
    essential = [c for c in ["statuscurrent", "category"] if c in df.columns]
    if essential:
        df.dropna(subset=essential, inplace=True)

    df.reset_index(drop=True, inplace=True)
    logger.info("Preprocessed dataframe: %d rows, %d columns", *df.shape)
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------
def load_and_preprocess(use_cache: bool = True) -> pd.DataFrame:
    """One-call convenience: fetch then preprocess.

    Parameters
    ----------
    use_cache : bool
        Passed through to :func:`fetch_data`.

    Returns
    -------
    pd.DataFrame
    """
    raw = fetch_data(use_cache=use_cache)
    return preprocess(raw)


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_and_preprocess()
    print(df.head())
    print(f"\nShape: {df.shape}")
    if "approved" in df.columns:
        print(f"Approval rate: {df['approved'].mean():.2%}")
