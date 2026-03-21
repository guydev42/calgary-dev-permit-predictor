"""Download development permits data from Calgary Open Data."""
import pandas as pd
import os

URL = "https://data.calgary.ca/resource/6933-unw5.csv?$limit=200000"
OUT = os.path.join(os.path.dirname(__file__), "development_permits.csv")

if os.path.exists(OUT):
    print(f"Data already exists: {OUT}")
else:
    print("Downloading development permits data...")
    df = pd.read_csv(URL)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} rows to {OUT}")
