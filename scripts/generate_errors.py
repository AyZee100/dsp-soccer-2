from pathlib import Path
import random

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = ROOT_DIR / "data" / "raw_data"


def inject_defense_errors(folder_path: Path) -> None:
    files = sorted([f for f in folder_path.iterdir() if f.suffix == ".csv"])
    for i, file_path in enumerate(files):
        df = pd.read_csv(file_path)
        # We intentionally inject a string value "ERROR" into `away_team_goal` to simulate
        # a type error. Cast to object once to avoid pandas FutureWarning spam.
        if "away_team_goal" in df.columns:
            df["away_team_goal"] = df["away_team_goal"].astype("object")

        # Schema error in one file
        if i == 5 and "B365A" in df.columns:
            df = df.drop(columns=["B365A"])

        for index in df.index:
            prob = random.random()
            if prob < 0.05:
                df.at[index, "B365H"] = None
            elif prob < 0.10:
                df.at[index, "home_team_goal"] = -5
            elif prob < 0.15:
                df.at[index, "home_team_api_id"] = 999999
            elif prob < 0.20:
                df.at[index, "away_team_goal"] = "ERROR"
            elif prob < 0.25:
                df.at[index, "B365H"] = 0.0
                if "B365A" in df.columns:
                    df.at[index, "B365A"] = 0.0

        # Duplicate rows
        df = pd.concat([df, df.head(2)], ignore_index=True)
        df.to_csv(file_path, index=False)

    print("Injected errors in raw_data files.")


if __name__ == "__main__":
    inject_defense_errors(RAW_DATA_DIR)
