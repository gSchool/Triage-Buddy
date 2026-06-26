from triage_buddy.config import load_dotenv


def test_missing_file_is_not_an_error(tmp_path):
    assert load_dotenv(tmp_path / "nope.env") == {}


def test_parses_pairs_comments_quotes_and_export(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FOO", raising=False)
    env = tmp_path / ".env"
    env.write_text(
        "\n".join(
            [
                "# a comment",
                "",
                "GROQ_API_KEY=abc123",
                'export FOO="bar baz"',
                "EMPTY=",
                "not a pair",
            ]
        )
    )
    parsed = load_dotenv(env)
    assert parsed["GROQ_API_KEY"] == "abc123"
    assert parsed["FOO"] == "bar baz"
    assert parsed["EMPTY"] == ""
    assert "not a pair" not in parsed


def test_existing_env_var_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "from-environment")
    env = tmp_path / ".env"
    env.write_text("GROQ_API_KEY=from-file")
    load_dotenv(env)
    import os

    assert os.environ["GROQ_API_KEY"] == "from-environment"
