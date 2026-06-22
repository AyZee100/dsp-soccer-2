import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "model_service" / "model.pkl"

# Features order:
# [home_team_id, away_team_id, home_odds, draw_odds, away_odds]
X = np.array(
    [
        [9987, 10000, 1.8, 3.2, 4.1],
        [10000, 9987, 2.3, 3.0, 2.8],
        [9993, 8342, 1.5, 3.6, 5.4],
        [8342, 9993, 2.1, 3.1, 3.3],
    ]
)
y = np.array([1, 0, 1, 0])

model = RandomForestClassifier(random_state=42).fit(X, y)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"Model created: {MODEL_PATH}")
