import pandas as pd


def compute_criticality(missing_columns: bool, invalid_rows: int, row_count: int) -> str:
    if missing_columns or (row_count > 0 and invalid_rows / row_count > 0.5):
        return "High"
    if row_count > 0 and invalid_rows / row_count >= 0.1:
        return "Medium"
    if invalid_rows > 0:
        return "Low"
    return "None"


def split_valid_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    invalid_mask = (
        df["B365H"].isna()
        | (df["home_team_goal"] < 0)
        | (~df["away_team_goal"].apply(lambda x: str(x).lstrip("-").isdigit()))
    )
    return df.loc[~invalid_mask], df.loc[invalid_mask]


def test_compute_criticality_high():
    assert compute_criticality(True, 0, 10) == "High"
    assert compute_criticality(False, 6, 10) == "High"


def test_compute_criticality_medium_low_none():
    assert compute_criticality(False, 2, 10) == "Medium"
    assert compute_criticality(False, 1, 20) == "Low"
    assert compute_criticality(False, 0, 20) == "None"


def test_split_valid_invalid():
    df = pd.DataFrame(
        {
            "B365H": [1.5, None, 2.0],
            "home_team_goal": [1, -1, 0],
            "away_team_goal": [2, 1, "ERROR"],
        }
    )
    good_df, bad_df = split_valid_invalid(df)
    assert len(good_df) == 1
    assert len(bad_df) == 2
