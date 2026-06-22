"""
Reset the environment to a clean demo state.

Actions (always):
  - Empty data/raw_data, good_data, bad_data, archived_data
  - Delete .training_seen and .training_cache
  - Delete .prediction_state

With --db flag:
  - Truncate predictions, ingestion_errors, ingestion_stats,
    training_runs, training_feature_stats, model_annotations

Usage (from repo root):
    py -3.12 scripts/demo_reset.py           # file dirs only
    py -3.12 scripts/demo_reset.py --db      # + DB tables
    py -3.12 scripts/demo_reset.py --db --yes  # non-interactive
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

_DB_TABLES = [
    "predictions",
    "ingestion_errors",
    "ingestion_stats",
    "training_runs",
    "training_feature_stats",
    "model_annotations",
]

_ENSURE_ANNOTATIONS = """
CREATE TABLE IF NOT EXISTS model_annotations (
    id             SERIAL PRIMARY KEY,
    annotation_text TEXT NOT NULL,
    tags           TEXT DEFAULT 'model-retrain',
    model_version  TEXT,
    created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


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


def _clear_dir(path: Path, label: str) -> int:
    path.mkdir(parents=True, exist_ok=True)
    removed = 0
    for item in list(path.iterdir()):
        try:
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)
            removed += 1
        except Exception as exc:
            print(f"  {YELLOW}warn{RESET}  could not remove {item.name}: {exc}")
    colour = GREEN if removed else DIM
    print(f"  {colour}cleared{RESET}  {label}/ ({removed} item(s))")
    return removed


def _remove_file(path: Path, label: str) -> None:
    if path.exists():
        path.unlink()
        print(f"  {GREEN}removed{RESET}  {label}")
    else:
        print(f"  {DIM}(skip){RESET}   {label} — not found")


def _truncate_db() -> None:
    _load_env()
    pg_user = os.getenv("POSTGRES_USER", "user")
    pg_pass = os.getenv("POSTGRES_PASSWORD", "password")
    pg_db = os.getenv("POSTGRES_DB", "dsp_db")

    try:
        import psycopg2  # type: ignore
    except ImportError:
        print(f"  {RED}psycopg2 not installed — skipping DB truncation.{RESET}")
        print("  Run: pip install psycopg2-binary")
        return

    try:
        conn = psycopg2.connect(
            dbname=pg_db, user=pg_user, password=pg_pass,
            host="localhost", port=5432, connect_timeout=5,
        )
        conn.autocommit = True
        with conn.cursor() as cur:
            # Ensure model_annotations table exists before trying to truncate it.
            cur.execute(_ENSURE_ANNOTATIONS)
            for table in _DB_TABLES:
                try:
                    cur.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY;")
                    print(f"  {GREEN}truncated{RESET} {table}")
                except Exception as exc:
                    print(f"  {YELLOW}(skip){RESET}   {table}: {exc}")
        conn.close()
    except Exception as exc:
        print(f"  {RED}DB error (is Postgres running?): {exc}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset the dsp-soccer demo environment.")
    parser.add_argument("--db", action="store_true", help="Also truncate DB tables")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    print(f"\n{BOLD}Demo Reset — dsp-soccer{RESET}")

    if not args.yes:
        scope = "file directories" + (" + DB tables" if args.db else "")
        ans = input(f"  This will wipe {scope}. Continue? [y/N] ").strip().lower()
        if ans != "y":
            print("  Aborted.")
            sys.exit(0)

    print(f"\n{BOLD}[Clearing data directories]{RESET}")
    _clear_dir(DATA / "raw_data", "raw_data")
    _clear_dir(DATA / "good_data", "good_data")
    _clear_dir(DATA / "bad_data", "bad_data")
    _clear_dir(DATA / "archived_data", "archived_data")
    _clear_dir(DATA / ".training_cache", ".training_cache")

    print(f"\n{BOLD}[Removing state files]{RESET}")
    _remove_file(DATA / ".training_seen", ".training_seen")
    _remove_file(DATA / ".prediction_state", ".prediction_state")

    if args.db:
        print(f"\n{BOLD}[Truncating DB tables]{RESET}")
        _truncate_db()

    print(f"\n{GREEN}{BOLD}Reset complete.{RESET}")
    print("  Next: py -3.12 scripts/seed_monitoring_data.py  (establish baseline)")
    print("  Then: py -3.12 scripts/healthcheck.py           (verify services)\n")


if __name__ == "__main__":
    main()
