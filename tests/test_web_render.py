"""Tests for HTML rendering, including escaping of user input (XSS guard)."""

from triage_buddy.adapters.web.app import render_page
from triage_buddy.adapters.web.service import run_triage


def test_empty_form_renders_inputs():
    page = render_page(provider="groq")
    assert "<form" in page
    assert 'name="description"' in page
    assert "groq" in page


def test_result_is_rendered():
    _, result = run_triage(description="severe chest pain", provider="mock")
    page = render_page(result=result)
    assert "EMERGENCY" in page
    assert "911" in page


def test_user_input_is_escaped():
    payload = {"description": '<script>alert(1)</script>'}
    page = render_page(form=payload)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_error_is_rendered_and_escaped():
    page = render_page(error="<bad>", form={"description": "x"})
    assert "Error" in page
    assert "<bad>" not in page
    assert "&lt;bad&gt;" in page
