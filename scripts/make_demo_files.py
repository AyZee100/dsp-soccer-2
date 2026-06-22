"""
Generate the 5 demo files required for the defense.

Output goes to ./demo_data (NOT raw_data). During the defense, copy a file into
data/raw_data to demo the ingestion + prediction pipelines from a clean state:

  1. demo_all_errors.csv   -> every row invalid (High criticality, all rows -> bad_data)
  2. demo_clean_1/2/3.csv  -> only valid rows (clean ingestion + prediction)
  3. demo_mixed.csv        -> mix of good and bad rows (demonstrates data splitting)
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_data" / "input_data.csv"
OUT = ROOT / "demo_data"
COLS = ["home_team_api_id", "away_team_api_id", "home_team_goal",
        "away_team_goal", "B365H", "B365D", "B365A"]


def main() -> None:
    df = pd.read_csv(SOURCE).dropna(subset=COLS)[COLS].reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # 3 clean files (30 valid rows each, non-overlapping slices).
    for i in range(3):
        clean = df.iloc[i * 30:(i + 1) * 30].copy()
        clean.to_csv(OUT / f"demo_clean_{i + 1}.csv", index=False)

    # 1 all-errors file: null B365H on every row -> completeness error on all rows.
    err = df.iloc[100:130].copy()
    err["B365H"] = None
    err.to_csv(OUT / "demo_all_errors.csv", index=False)

    # 1 mixed file: 20 valid rows + 10 corrupted rows (mix of categories).
    mixed = df.iloc[200:230].copy().reset_index(drop=True)
    mixed["away_team_goal"] = mixed["away_team_goal"].astype("object")
    mixed.loc[0:2, "B365H"] = None                 # completeness
    mixed.loc[3:4, "home_team_goal"] = -5          # validity
    mixed.loc[5:6, "away_team_goal"] = "ERROR"     # type
    mixed.loc[7:9, "B365A"] = 0.0                  # validity (non-positive odds)
    mixed.to_csv(OUT / "demo_mixed.csv", index=False)

    print(f"Wrote 5 demo files to {OUT}")


if __name__ == "__main__":
    main()
