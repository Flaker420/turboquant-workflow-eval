from pathlib import Path

from turboquant_workflow_eval.download import (
    check_cache_status,
    discover_model_configs,
    format_summary_table,
)


def test_discover_model_configs(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "alpha.yaml").write_text(
        "model_name: org/alpha\nlanguage_model_only: false\n", encoding="utf-8"
    )
    (model_dir / "beta.yaml").write_text(
        "model_name: org/beta\nlanguage_model_only: true\n", encoding="utf-8"
    )
    # Non-yaml files should be ignored
    (model_dir / "readme.txt").write_text("ignore me", encoding="utf-8")

    configs = discover_model_configs(model_dir)
    assert len(configs) == 2
    assert configs[0]["model_name"] == "org/alpha"
    assert configs[1]["model_name"] == "org/beta"
    assert "_config_path" in configs[0]


def test_discover_model_configs_empty(tmp_path: Path) -> None:
    model_dir = tmp_path / "empty"
    model_dir.mkdir()
    configs = discover_model_configs(model_dir)
    assert configs == []


def test_check_cache_status_missing(tmp_path: Path) -> None:
    result = check_cache_status("org/nonexistent", cache_dir=str(tmp_path))
    assert result == {"model_cached": False, "tokenizer_cached": False}


def test_check_cache_status_tokenizer_only(tmp_path: Path) -> None:
    snap_dir = tmp_path / "hub" / "models--org--mymodel" / "snapshots" / "abc123"
    snap_dir.mkdir(parents=True)
    (snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    result = check_cache_status("org/mymodel", cache_dir=str(tmp_path))
    assert result["model_cached"] is False
    assert result["tokenizer_cached"] is True


def test_check_cache_status_full(tmp_path: Path) -> None:
    snap_dir = tmp_path / "hub" / "models--org--mymodel" / "snapshots" / "abc123"
    snap_dir.mkdir(parents=True)
    (snap_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (snap_dir / "model.safetensors.index.json").write_text("{}", encoding="utf-8")

    result = check_cache_status("org/mymodel", cache_dir=str(tmp_path))
    assert result["model_cached"] is True
    assert result["tokenizer_cached"] is True


def test_format_summary_table_empty() -> None:
    assert format_summary_table([]) == "No models processed."


def test_format_summary_table_mixed() -> None:
    results = [
        {"model_name": "org/alpha", "status": "ok", "downloaded": "model_and_tokenizer", "error": None},
        {"model_name": "org/beta", "status": "cached", "downloaded": None, "error": None},
        {"model_name": "org/gamma", "status": "failed", "downloaded": None, "error": "connection timeout"},
    ]
    table = format_summary_table(results)
    assert "org/alpha" in table
    assert "ok" in table
    assert "cached" in table
    assert "failed" in table
    assert "already in cache" in table
    assert "connection timeout" in table
