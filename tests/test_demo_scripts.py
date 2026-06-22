"""
Tests for pure-Python logic in the new demo scripts.
No Docker / DB / network required — all fixtures are in-process.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Make scripts/ importable even when pytest runs from the repo root.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))


# ── healthcheck helpers ────────────────────────────────────────────────────────
from healthcheck import format_row  # noqa: E402


def test_format_row_ok():
    row = format_row("API /health", True, "version=v1 source=mlflow")
    assert "UP" in row
    assert "API /health" in row
    assert "version=v1" in row


def test_format_row_down():
    row = format_row("PostgreSQL", False, "connection refused")
    assert "DOWN" in row
    assert "PostgreSQL" in row
    assert "connection refused" in row


# ── demo_run: drift batch generation ─────────────────────────────────────────
from demo_run import build_drift_batch  # noqa: E402


def test_build_drift_batch_count():
    rows = build_drift_batch(50)
    assert len(rows) == 50


def test_build_drift_batch_odds_range():
    rows = build_drift_batch(200, home_odds_mean=8.5, seed=1)
    home_odds = [r["home_odds"] for r in rows]
    # All odds must be ≥ 1.05 (enforced by build_drift_batch).
    assert all(o >= 1.05 for o in home_odds)
    # Mean should be near the requested centre (within 1.5 of target).
    mean = sum(home_odds) / len(home_odds)
    assert abs(mean - 8.5) < 1.5, f"Expected mean ~8.5, got {mean:.2f}"


def test_build_drift_batch_z_score_sufficient():
    """
    Verify the injected batch pushes the last-500 window z-score above 2.0.
    Window model: 300 injected (most recent) + 200 existing at baseline mean.
    """
    baseline_mean = 2.63
    baseline_std = 1.45
    drift_threshold = 2.0

    n_inject = 300
    n_existing = 200  # last-500 window: 300 injected + 200 old
    rows = build_drift_batch(n_inject, home_odds_mean=8.5, seed=42)
    inject_mean = sum(r["home_odds"] for r in rows) / len(rows)

    # Combined mean across the full last-500 window
    combined = (n_inject * inject_mean + n_existing * baseline_mean) / (n_inject + n_existing)
    z = abs(combined - baseline_mean) / baseline_std
    assert z > drift_threshold, (
        f"Drift batch z={z:.2f} does not exceed threshold {drift_threshold}. "
        f"Increase home_odds_mean or injection count."
    )


def test_build_drift_batch_reproducible():
    a = build_drift_batch(10, seed=99)
    b = build_drift_batch(10, seed=99)
    assert [r["home_odds"] for r in a] == [r["home_odds"] for r in b]


# ── demo_reset: _load_env parsing ─────────────────────────────────────────────
def test_load_env_parses_key_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_KEY_DEMO=hello_world\n# comment\nBAD LINE\n")

    import demo_reset as dr_mod

    # Patch ROOT so _load_env finds our tmp .env
    monkeypatch.setattr(dr_mod, "ROOT", tmp_path)
    monkeypatch.delenv("TEST_KEY_DEMO", raising=False)

    dr_mod._load_env()
    assert os.environ.get("TEST_KEY_DEMO") == "hello_world"


def test_load_env_skips_missing_file(tmp_path, monkeypatch):
    import demo_reset as dr_mod
    monkeypatch.setattr(dr_mod, "ROOT", tmp_path)
    # No .env file; _load_env should not raise.
    dr_mod._load_env()


# ── healthcheck: _load_env independent copy ───────────────────────────────────
def test_healthcheck_load_env_parses(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HC_TEST_VAR=42\n")

    import healthcheck as hc_mod
    monkeypatch.setattr(hc_mod, "ROOT", tmp_path)
    monkeypatch.delenv("HC_TEST_VAR", raising=False)

    hc_mod._load_env()
    assert os.environ.get("HC_TEST_VAR") == "42"
