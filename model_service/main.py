"""
Model service API: ORM schema, Pydantic contracts, sklearn predictor loaded once in lifespan,
prediction persistence, and read endpoints for the Streamlit app (no direct DB access there).
"""
from contextlib import asynccontextmanager
from datetime import datetime
import json
import os
import pickle
from typing import Generator, List, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is required")

# MLflow model registry configuration (Defense 2).
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "soccer_model")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

# Model feature order (must match training in dags/ml_utils.py).
FEATURES = ["home_team_id", "away_team_id", "home_odds", "draw_odds", "away_odds"]

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionRecord(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(32), nullable=False, default="webapp")
    model_version = Column(String(64), nullable=False, default="local_v1")
    input_data = Column(Text, nullable=False)
    prediction = Column(Float, nullable=False)
    proba_home_win = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class IngestionStat(Base):
    __tablename__ = "ingestion_stats"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    status = Column(String(32), nullable=False)
    row_count = Column(Integer, nullable=False, default=0)
    error_count = Column(Integer, nullable=False, default=0)
    criticality = Column(String(32), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class TrainingStat(Base):
    __tablename__ = "training_stats"
    id = Column(Integer, primary_key=True, index=True)
    model_version = Column(String(64), nullable=False, default="local_v1")
    # Minimal schema for Defense 1; populated in Defense 2 when you implement training + drift monitoring.
    training_rows = Column(Integer, nullable=False, default=0)
    drift_score = Column(Float, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class FeatureInput(BaseModel):
    home_team_id: int = Field(..., alias="home_team_id")
    away_team_id: int
    home_odds: float
    draw_odds: float
    away_odds: float

    model_config = {"populate_by_name": True}


class PredictRequest(BaseModel):
    features: List[FeatureInput]
    source: str = Field(default="webapp")


class PredictionRow(BaseModel):
    id: int
    source: str
    model_version: str
    features: dict
    prediction: float
    proba_home_win: Optional[float] = None
    created_at: datetime


class PredictResponse(BaseModel):
    predictions: List[PredictionRow]


ml_models: dict = {}


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _load_local_fallback() -> None:
    """
    Resilience only: if no champion is registered in MLflow yet (e.g. before the
    first training run), build a tiny local model so /predict still responds and
    the container stays healthy. Marked clearly as a non-MLflow fallback.
    """
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    X = np.array(
        [
            [9987, 10000, 1.8, 3.2, 4.1],
            [10000, 9987, 2.3, 3.0, 2.8],
            [9993, 8342, 1.5, 3.6, 5.4],
            [8342, 9993, 2.1, 3.1, 3.3],
        ]
    )
    y = np.array([1, 0, 1, 0])
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                loaded_model = pickle.load(f)
            loaded_model.predict([X[0].tolist()])
            ml_models["predictor"] = loaded_model
            ml_models["version"] = "local_fallback"
            ml_models["source"] = "local"
            return
        except Exception as exc:
            print(f"Incompatible/corrupted model.pkl, rebuilding local fallback: {exc}")

    model = RandomForestClassifier(random_state=42).fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    ml_models["predictor"] = model
    ml_models["version"] = "local_fallback"
    ml_models["source"] = "local"


def _load_model() -> dict:
    """
    Load the production model from the MLflow registry via alias:
        models:/<MODEL_NAME>@<MODEL_ALIAS>   (e.g. models:/soccer_model@champion)
    Falls back to a local model if the registry has no champion yet.
    """
    try:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")

        ml_models["predictor"] = model
        ml_models["version"] = f"{MODEL_NAME}_v{mv.version}"
        ml_models["source"] = "mlflow"
        print(f"Loaded champion model {ml_models['version']} from MLflow.")
        return {"loaded": True, "version": ml_models["version"], "source": "mlflow"}
    except Exception as exc:
        print(f"Could not load champion from MLflow ({exc}); using local fallback.")
        _load_local_fallback()
        return {"loaded": True, "version": ml_models["version"], "source": "local"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    # Backward-compatible migration: add probability column if missing.
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE predictions ADD COLUMN IF NOT EXISTS proba_home_win DOUBLE PRECISION;"))
    _load_model()
    yield
    ml_models.clear()


http_app = FastAPI(lifespan=lifespan)


@http_app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": "predictor" in ml_models,
        "model_version": ml_models.get("version"),
        "model_source": ml_models.get("source"),
    }


@http_app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, db: Session = Depends(get_db)):
    if "predictor" not in ml_models:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not payload.features:
        raise HTTPException(status_code=400, detail="features list cannot be empty")

    model = ml_models["predictor"]
    model_version = ml_models.get("version", "unknown")
    rows: List[PredictionRow] = []
    for item in payload.features:
        values = [
            item.home_team_id,
            item.away_team_id,
            item.home_odds,
            item.draw_odds,
            item.away_odds,
        ]
        # Pass a named DataFrame so the model sees the same feature names it trained on.
        X_row = pd.DataFrame([values], columns=FEATURES)
        pred = float(np.asarray(model.predict(X_row))[0])
        proba_home_win: Optional[float] = None
        try:
            if hasattr(model, "predict_proba"):
                proba_home_win = float(model.predict_proba(X_row)[0][1])
        except Exception:
            proba_home_win = None
        feature_dict = item.model_dump(by_alias=True)

        record = PredictionRecord(
            source=payload.source,
            model_version=model_version,
            input_data=json.dumps(feature_dict),
            prediction=pred,
            proba_home_win=proba_home_win,
        )
        db.add(record)
        db.flush()
        if record.created_at is None:
            # Defensive fallback for legacy schemas/migrations.
            record.created_at = datetime.utcnow()
        rows.append(
            PredictionRow(
                id=record.id,
                source=record.source,
                model_version=record.model_version,
                features=feature_dict,
                prediction=pred,
                proba_home_win=record.proba_home_win,
                created_at=record.created_at,
            )
        )

    db.commit()
    return PredictResponse(predictions=rows)


@http_app.get("/past-predictions", response_model=List[PredictionRow])
def past_predictions(
    limit: int = Query(default=100, ge=1, le=1000),
    source: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(PredictionRecord).order_by(PredictionRecord.created_at.desc())
    if source and source != "all":
        query = query.filter(PredictionRecord.source == source)

    records = query.limit(limit).all()
    response: List[PredictionRow] = []
    for rec in records:
        response.append(
            PredictionRow(
                id=rec.id,
                source=rec.source,
                model_version=rec.model_version,
                features=json.loads(rec.input_data),
                prediction=rec.prediction,
                proba_home_win=rec.proba_home_win,
                created_at=rec.created_at,
            )
        )
    return response


@http_app.post("/reload-model")
def reload_model():
    """
    Reload the production model from the MLflow registry (models:/<name>@champion).
    Called by the training DAG's notify_api_reload task after a promotion so the API
    serves the new champion without a container rebuild.
    """
    ml_models.clear()
    result = _load_model()
    return {"status": "reloaded", "model_loaded": "predictor" in ml_models, **result}


@http_app.get("/ingestion-stats")
def get_ingestion_stats(limit: int = Query(default=100, ge=1, le=1000), db: Session = Depends(get_db)):
    rows = (
        db.query(IngestionStat)
        .order_by(IngestionStat.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": row.id,
            "file_name": row.file_name,
            "status": row.status,
            "row_count": row.row_count,
            "error_count": row.error_count,
            "criticality": row.criticality,
            "created_at": row.created_at.isoformat(),
        }
        for row in rows
    ]
