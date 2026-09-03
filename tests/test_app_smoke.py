"""
Smoke tests for the Streamlit app shell (Phase: production hardening).

These run the app headlessly with ``streamlit.testing.v1.AppTest`` so that a
broken import, a crashing top-level render, or a widget wiring error is caught
in CI without needing a browser or network access.

The heavy download/model pipeline only runs after the user clicks
"Build Portfolio" (and would hit live Yahoo Finance), so this smoke test covers
the initial render only - i.e. the app shell must load with no exceptions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"

pytestmark = pytest.mark.skipif(
    not APP_PATH.exists(), reason="app.py not found"
)


def test_app_shell_renders_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    assert not at.exception, (
        f"App shell raised an exception: {[e.exception for e in at.exception]}"
    )


def test_app_shell_shows_build_prompt_before_click():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    info_text = "\n".join(i.value for i in at.info)
    assert "Build Portfolio" in info_text, (
        "App should prompt the user to click Build Portfolio before running."
    )


def test_app_has_primary_build_button_in_sidebar():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    buttons = [b.label for b in at.button]
    assert "Build Portfolio" in buttons, (
        f"Expected 'Build Portfolio' button, got: {buttons}"
    )


def test_app_sidebar_offers_force_refresh_toggle():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP_PATH), default_timeout=60)
    at.run()

    checkboxes = [c.label for c in at.checkbox]
    assert any("Force fresh data download" in c for c in checkboxes), (
        f"Expected force-refresh checkbox, got: {checkboxes}"
    )