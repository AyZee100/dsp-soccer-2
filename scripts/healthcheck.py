"""
Poll all 7 Docker-compose services and print a colour-coded status table.

Usage (from repo root):
    py -3.12 scripts/healthcheck.py

Exit 0 = all healthy.  Exit 1 = at least one service is DOWN.
Reads DB credentials from .env (falls back to docker-compose defaults).
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


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


# ── HTTP helper ───────────────────────────────────────────────────────────────
def _http_get(url: str, timeout: int = 5) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except Exception as exc:
        return None, str(exc)


# ── Individual checks ─────────────────────────────────────────────────────────
def check_api() -> tuple[bool, str]:
    code, body = _http_get("http://localhost:8000/health")
    if code == 200:
        try:
            d = json.loads(body)
            ver = d.get("model_version") or "?"
            src = d.get("model_source") or "?"
            return True, f"version={ver}  source={src}"
        except Exception:
            return True, "ok"
    return False, f"HTTP {code}"


def check_airflow() -> tuple[bool, str]:
    code, body = _http_get("http://localhost:8080/health", timeout=8)
    if code == 200:
        try:
            d = json.loads(body)
            meta = d.get("metadatabase", {}).get("status", "?")
            sched = d.get("scheduler", {}).get("status", "?")
            ok = meta == "healthy"
            return ok, f"meta={meta}  scheduler={sched}"
        except Exception:
            return True, "ok"
    return False, f"HTTP {code}"


def check_mlflow() -> tuple[bool, str]:
    code, _ = _http_get("http://localhost:5000/health")
    return code == 200, f"HTTP {code}"


def check_grafana() -> tuple[bool, str]:
    code, body = _http_get("http://localhost:3000/api/health")
    if code == 200:
        try:
            d = json.loads(body)
            db = d.get("database", "?")
            return db == "ok", f"database={db}"
        except Exception:
            return True, "ok"
    return False, f"HTTP {code}"


def check_nginx() -> tuple[bool, str]:
    code, _ = _http_get("http://localhost:8081")
    ok = code is not None and code < 400
    return ok, f"HTTP {code}"


def check_postgres() -> tuple[bool, str]:
    try:
        import psycopg2  # type: ignore
        conn = psycopg2.connect(
            dbname=PG_DB, user=PG_USER, password=PG_PASS,
            host="localhost", port=5432, connect_timeout=5,
        )
        conn.close()
        return True, f"connected  db={PG_DB}"
    except ImportError:
        # psycopg2 not on host: try raw socket to confirm port is open
        import socket
        try:
            s = socket.create_connection(("localhost", 5432), timeout=5)
            s.close()
            return True, "port open (psycopg2 not installed)"
        except Exception as exc:
            return False, str(exc)
    except Exception as exc:
        return False, str(exc)


def check_streamlit() -> tuple[bool, str]:
    code, _ = _http_get("http://localhost:8501", timeout=8)
    ok = code is not None and code < 400
    return ok, f"HTTP {code}"


# ── Table printer ─────────────────────────────────────────────────────────────
CHECKS = [
    ("API /health", check_api),
    ("Airflow", check_airflow),
    ("MLflow", check_mlflow),
    ("Grafana", check_grafana),
    ("nginx / Data Docs", check_nginx),
    ("PostgreSQL", check_postgres),
    ("Streamlit", check_streamlit),
]


def format_row(name: str, ok: bool, detail: str) -> str:
    dot = f"{GREEN}●{RESET}" if ok else f"{RED}●{RESET}"
    word = f"{GREEN}UP  {RESET}" if ok else f"{RED}DOWN{RESET}"
    return f"  {dot} {word}  {name:<22} {detail}"


def main() -> None:
    print(f"\n{BOLD}Service Health Check — dsp-soccer{RESET}")
    print("─" * 65)

    results: list[tuple[str, bool, str]] = []
    for name, fn in CHECKS:
        ok, detail = fn()
        results.append((name, ok, detail))

    for name, ok, detail in results:
        print(format_row(name, ok, detail))

    print("─" * 65)
    all_ok = all(ok for _, ok, _ in results)
    if all_ok:
        print(f"  {GREEN}{BOLD}All 7 services healthy ✓{RESET}\n")
        sys.exit(0)
    else:
        down = sum(1 for _, ok, _ in results if not ok)
        print(f"  {RED}{BOLD}{down} service(s) DOWN — fix before the demo!{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
