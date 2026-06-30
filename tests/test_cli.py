from triage_buddy.adapters.cli.app import main


def test_cli_emergency_path(capsys):
    code = main(["I", "have", "severe", "chest", "pain"])
    out = capsys.readouterr().out
    assert code == 0
    assert "EMERGENCY" in out
    assert "911" in out


def test_cli_mild_path(capsys):
    code = main(["mild", "runny", "nose"])
    out = capsys.readouterr().out
    assert code == 0
    assert "LOW" in out
    assert "not a medical diagnosis" in out


def test_cli_rejects_empty_input_noninteractive(capsys, monkeypatch):
    # No positional args and stdin is not a tty -> read empty stdin -> error.
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "")
    code = main([])
    assert code == 2
    assert "No symptoms" in capsys.readouterr().err


def test_cli_unknown_provider_reports_error(capsys):
    code = main(["--provider", "nope", "headache"])
    assert code == 3
    assert "Provider error" in capsys.readouterr().err
