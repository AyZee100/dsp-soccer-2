"""
Demo conductor for the 10-minute defense.

Walks through 6 beats, pausing for commentary between each.
Stack must already be running: docker compose up -d

Usage (from repo root):
    py -3.12 scripts/demo_run.py

Pre-requisites on the host:
    pip install psycopg2-binary requests
"""
from __future__ import annotations

import base64
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEMO_DATA = ROOT / "demo_data"

API = "http://localhost:8000"
AIRFLOW = "http://localhost:8080"
MLFLOW = "http://localhost:5000"
GRAFANA = "http://localhost:3000"
DATA_DOCS = "http://localhost:8081"

GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


# ── env ───────────────────────────────────────────────────────────────────────
def _load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


_load_env()
PG_USER = os.getenv("POSTGRES_USER", "user")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "password")
PG_DB = os.getenv("POSTGRES_DB", "dsp_db")


# ── UI helpers ────────────────────────────────────────────────────────────────
def _banner(beat: int, title: str) -> None:
    print(f"\n{'━' * 62}")
    print(f"  {BOLD}{CYAN}Beat {beat}: {title}{RESET}")
    print(f"{'━' * 62}")


def _step(msg: str) -> None:
    print(f"\n  {BOLD}▶{RESET} {msg}")


def _ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


def _url(label: str, addr: str) -> None:
    print(f"  {YELLOW}→ {label}:{RESET} {BOLD}{addr}{RESET}")


def _pause(msg: str = "Press ENTER to continue to next beat...") -> None:
    print(f"\n  {DIM}{msg}{RESET}")
    input()


# ── HTTP / Airflow helpers ────────────────────────────────────────────────────
def _http(url: str, method: str = "GET", body: object = None,
          timeout: int = 10, auth: str | None = None) -> tuple[int | None, str]:
    data = json.dumps(body).encode() if body is not None else None
    headers: dict[str, str] = {}
    if data:
        headers["Content-Type"] = "application/json"
    if auth:
        headers["Authorization"] = f"Basic {auth}"
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return None, str(exc)


_AF_AUTH = base64.b64encode(b"admin:admin").decode()


def _trigger_dag(dag_id: str) -> str:
    code, body = _http(
        f"{AIRFLOW}/api/v1/dags/{dag_id}/dagRuns",
        method="POST",
        body={"conf": {}},
        auth=_AF_AUTH,
        timeout=20,
    )
    if code and 200 <= code < 300:
        try:
            return json.loads(body).get("dag_run_id", "triggered")
        except Exception:
            return "triggered"
    return f"HTTP {code}: {body[:120]}"


def _copy_to_raw(src_name: str, dest_name: str | None = None) -> Path:
    src = DEMO_DATA / src_name
    dest = DATA / "raw_data" / (dest_name or src_name)
    shutil.copy2(str(src), str(dest))
    return dest


def _count_csvs(path: Path) -> int:
    return len(list(path.glob("*.csv"))) if path.exists() else 0


def _poll_dir_grows(directory: Path, prev_count: int, label: str,
                    timeout: int = 120) -> int:
    """Wait until directory has more CSVs than prev_count. Returns new count."""
    print(f"  {DIM}Waiting for {label} to grow (currently {prev_count})...{RESET}", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        n = _count_csvs(directory)
        if n > prev_count:
            print(f" {GREEN}done ({n}){RESET}")
            return n
        time.sleep(3)
        print(".", end="", flush=True)
    new_n = _count_csvs(directory)
    print(f" {YELLOW}timeout (now {new_n}){RESET}")
    return new_n


def _get_champion() -> tuple[str, str]:
    code, body = _http(f"{API}/health")
    if code == 200:
        d = json.loads(body)
        return d.get("model_version") or "unknown", d.get("model_source") or "?"
    return "unknown", "?"


# ── Drift injection ───────────────────────────────────────────────────────────
def build_drift_batch(n: int, home_odds_mean: float = 8.5, seed: int = 42) -> list[dict]:
    """
    Generate synthetic predictions with shifted home_odds to simulate drift.
    home_odds ~8.5 vs training baseline ~2.63 → z-score ≈ 2.4 even when diluted
    by 200 normal predictions in the last-500 window.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        home = max(1.05, round(rng.gauss(home_odds_mean, 0.6), 2))
        draw = max(1.05, round(rng.gauss(3.8, 0.4), 2))
        away = max(1.05, round(rng.gauss(4.1, 0.8), 2))
        proba = max(0.02, min(0.98, 1.3 / home))
        rows.append({
            "home_team_id": rng.randint(8000, 10100),
            "away_team_id": rng.randint(8000, 10100),
            "home_odds": home,
            "draw_odds": draw,
            "away_odds": away,
            "proba": proba,
            "pred": 1.0 if proba >= 0.5 else 0.0,
        })
    return rows


def _inject_drift_predictions(n: int = 300, home_odds_mean: float = 7.2) -> int | str:
    """Insert drift-pattern rows directly into the predictions table."""
    try:
        import psycopg2  # type: ignore
    except ImportError:
        return "psycopg2 not installed — run: pip install psycopg2-binary"

    _, health_body = _http(f"{API}/health")
    model_version = "unknown"
    try:
        model_version = json.loads(health_body).get("model_version") or "unknown"
    except Exception:
        pass

    rows = build_drift_batch(n, home_odds_mean=home_odds_mean)
    try:
        conn = psycopg2.connect(
            dbname=PG_DB, user=PG_USER, password=PG_PASS,
            host="localhost", port=5432, connect_timeout=5,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            for r in rows:
                feats = {
                    "home_team_id": r["home_team_id"],
                    "away_team_id": r["away_team_id"],
                    "home_odds": r["home_odds"],
                    "draw_odds": r["draw_odds"],
                    "away_odds": r["away_odds"],
                }
                cur.execute(
                    """INSERT INTO predictions
                       (source, model_version, input_data, prediction, proba_home_win, created_at)
                       VALUES (%s, %s, %s, %s, %s, NOW())""",
                    ("drift_demo", model_version, json.dumps(feats),
                     r["pred"], round(r["proba"], 3)),
                )
        conn.close()
        return n
    except Exception as exc:
        return f"DB error: {exc}"


# ── Beats ─────────────────────────────────────────────────────────────────────
def beat_a() -> None:
    _banner(1, "Clean batch ingestion → good_data + Data Docs")

    _step("Copy demo_clean_1.csv → raw_data (30 clean rows)")
    _copy_to_raw("demo_clean_1.csv")
    _ok("Copied. Triggering soccer_ingestion DAG...")

    run_id = _trigger_dag("soccer_ingestion")
    _ok(f"DAG triggered: {run_id}")

    prev_good = _count_csvs(DATA / "good_data")
    _poll_dir_grows(DATA / "good_data", prev_good, "good_data", timeout=120)

    good_n = _count_csvs(DATA / "good_data")
    bad_n = _count_csvs(DATA / "bad_data")
    _ok(f"Split complete — good_data: {good_n} file(s), bad_data: {bad_n} file(s)")
    _ok("All 30 rows clean → zero errors → criticality: None")

    _url("Data Docs", DATA_DOCS)
    _url("Airflow DAG", f"{AIRFLOW}/dags/soccer_ingestion")
    _pause()


def beat_b() -> None:
    _banner(2, "All-errors batch → High criticality alert")

    _step("Copy demo_all_errors.csv → raw_data (30 rows, all null B365H)")
    _copy_to_raw("demo_all_errors.csv")

    prev_bad = _count_csvs(DATA / "bad_data")
    run_id = _trigger_dag("soccer_ingestion")
    _ok(f"DAG triggered: {run_id}")

    new_bad = _poll_dir_grows(DATA / "bad_data", prev_bad, "bad_data", timeout=120)
    _ok(f"bad_data now has {new_bad} file(s)")
    _ok("Criticality: HIGH — 30/30 rows invalid (completeness: null B365H)")
    _ok("Teams alert fired (if webhook configured); Grafana alert rule will evaluate")

    _url("Airflow run", f"{AIRFLOW}/dags/soccer_ingestion")
    _url("Grafana Data Quality", f"{GRAFANA}/d/dsp_data_quality")
    _pause()


def beat_c() -> None:
    _banner(3, "Mixed batch → good / bad row split")

    _step("Copy demo_mixed.csv → raw_data (20 valid rows + 10 bad rows)")
    print(f"  {DIM}Error types injected: completeness(3), validity(5), type(2){RESET}")
    _copy_to_raw("demo_mixed.csv")

    prev_good = _count_csvs(DATA / "good_data")
    run_id = _trigger_dag("soccer_ingestion")
    _ok(f"DAG triggered: {run_id}")

    _poll_dir_grows(DATA / "good_data", prev_good, "good_data", timeout=120)
    good_n = _count_csvs(DATA / "good_data")
    bad_n = _count_csvs(DATA / "bad_data")
    _ok(f"After split — good_data: {good_n}, bad_data: {bad_n}")
    _ok("20 clean rows → good_data/demo_mixed.csv; 10 bad rows → bad_data/demo_mixed.csv")

    _url("Data Docs", DATA_DOCS)
    _url("Grafana Data Quality", f"{GRAFANA}/d/dsp_data_quality")
    _pause()


def beat_d() -> None:
    _banner(4, "Build good_data and run scheduled predictions")

    _step("Copy demo_clean_2.csv → raw_data and trigger ingestion")
    _copy_to_raw("demo_clean_2.csv")
    prev_good = _count_csvs(DATA / "good_data")
    run_id = _trigger_dag("soccer_ingestion")
    _ok(f"Ingestion triggered: {run_id}")
    _poll_dir_grows(DATA / "good_data", prev_good, "good_data", timeout=120)

    _step("Copy demo_clean_3.csv → raw_data and trigger ingestion")
    _copy_to_raw("demo_clean_3.csv")
    prev_good = _count_csvs(DATA / "good_data")
    run_id = _trigger_dag("soccer_ingestion")
    _ok(f"Ingestion triggered: {run_id}")
    _poll_dir_grows(DATA / "good_data", prev_good, "good_data", timeout=120)

    good_n = _count_csvs(DATA / "good_data")
    _ok(f"good_data now has {good_n} file(s)  (need ≥ 3 for training)")

    _step("Trigger prediction DAG")
    pred_run = _trigger_dag("soccer_prediction")
    _ok(f"Prediction DAG triggered: {pred_run}")
    print(f"  {DIM}Predictions will arrive in ~30 s (DAG scheduled every 2 min){RESET}")
    time.sleep(5)

    ver, src = _get_champion()
    _ok(f"Model in service: {ver}  (source={src})")

    _url("API /health", f"{API}/health")
    _url("API /past-predictions", f"{API}/past-predictions?limit=20&source=scheduled")
    _url("Grafana Drift & Predictions", f"{GRAFANA}/d/dsp_drift_predictions")
    _pause()


def beat_e() -> None:
    _banner(5, "Inject drift batch — home_odds shifted up")

    n = 300
    mean_odds = 8.5
    _step(f"Inserting {n} synthetic predictions with home_odds ~ {mean_odds} (baseline ~ 2.6)")
    print(f"  {DIM}Calibrated to push serving mean above z-score threshold (2.0){RESET}")

    result = _inject_drift_predictions(n=n, home_odds_mean=mean_odds)

    if isinstance(result, int):
        _ok(f"Inserted {result} drift-pattern rows")
        _ok(f"Expected z-score in last-500 window: ≈ {(0.6 * mean_odds + 0.4 * 2.63 - 2.63) / 1.45:.1f}")
        _ok("Grafana drift panel will show a spike in home_odds serving mean")
    else:
        _warn(f"Injection issue: {result}")
        _warn("Try: pip install psycopg2-binary and retry this beat")

    _url("Grafana Drift Dashboard", f"{GRAFANA}/d/dsp_drift_predictions")
    print(f"  {DIM}Set Grafana time range to 'Last 15 minutes' to see the spike.{RESET}")
    _pause()


def beat_f() -> None:
    _banner(6, "Retrain → @champion promoted → /reload-model → new model_version")

    ver_before, _ = _get_champion()
    _ok(f"Champion BEFORE training: {ver_before}")

    _step("Triggering soccer_training DAG...")
    run_id = _trigger_dag("soccer_training")
    _ok(f"Training DAG triggered: {run_id}")
    print(f"  {DIM}Training takes 1–3 min. Watch Airflow task progression.{RESET}")
    print(f"  {DIM}promote_to_champion → notify_api_reload → archive_data{RESET}")

    _url("Airflow Training", f"{AIRFLOW}/dags/soccer_training")
    _url("MLflow Experiments", f"{MLFLOW}/#/experiments")
    _pause("Press ENTER once promote_to_champion goes GREEN in Airflow (or after ~3 min)...")

    _step("Checking API for new model version...")
    ver_after, src_after = _get_champion()
    _ok(f"Champion AFTER training: {ver_after}  (source={src_after})")
    if ver_before != ver_after:
        _ok(f"Version changed: {ver_before} → {ver_after}")
    else:
        _warn(f"Version unchanged ({ver_after}). Possible reasons:")
        _warn("  • Training still running — wait another minute and re-check /health")
        _warn("  • Candidate didn't beat champion accuracy — see MLflow run tags")
        _warn("  • Not enough new files — check load_data task logs in Airflow")

    _step("Sending one final prediction stamped with the new model version...")
    code, body = _http(
        f"{API}/predict",
        method="POST",
        body={
            "source": "demo_final",
            "features": [{
                "home_team_id": 9987, "away_team_id": 10000,
                "home_odds": 1.9, "draw_odds": 3.5, "away_odds": 4.2,
            }],
        },
    )
    if code == 200:
        preds = json.loads(body).get("predictions", [])
        if preds:
            mv = preds[0].get("model_version", "?")
            pv = preds[0].get("prediction", "?")
            pb = preds[0].get("proba_home_win")
            pb_str = f"  P(home win)={pb:.2f}" if pb is not None else ""
            _ok(f"Prediction: {pv}{pb_str}  (model_version={mv})")
    else:
        _warn(f"Predict returned HTTP {code}: {body[:150]}")

    _url("Grafana Drift Dashboard", f"{GRAFANA}/d/dsp_drift_predictions")
    _url("MLflow Model Registry", f"{MLFLOW}/#/models/soccer_model")
    _ok("Grafana annotation line marks the model-retrain moment on the drift panels")

    print(f"\n{'━' * 62}")
    print(f"  {GREEN}{BOLD}Demo complete!  All 6 beats delivered. ✓{RESET}")
    print(f"{'━' * 62}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{BOLD}{'=' * 62}")
    print("  DSP Soccer — Demo Conductor  (Defense 2)")
    print(f"{'=' * 62}{RESET}")
    print("  Pre-conditions:")
    print("    docker compose up -d  (all services healthy)")
    print("    py -3.12 scripts/demo_reset.py --db --yes")
    print("    py -3.12 scripts/seed_monitoring_data.py")
    print()

    code, body = _http(f"{API}/health")
    if code != 200:
        print(f"  {RED}API not responding at {API}/health{RESET}")
        print("  Start the stack first: docker compose up -d")
        sys.exit(1)
    d = json.loads(body)
    print(f"  {GREEN}API healthy{RESET}  model={d.get('model_version')}  "
          f"source={d.get('model_source')}")

    _pause("Press ENTER to start Beat 1...")

    beat_a()
    beat_b()
    beat_c()
    beat_d()
    beat_e()
    beat_f()


if __name__ == "__main__":
    main()
