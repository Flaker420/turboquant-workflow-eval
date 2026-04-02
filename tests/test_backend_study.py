from pathlib import Path

from qwen35_turboquant_workflow_study.config import load_yaml


def test_backend_study_config_exists():
    cfg = load_yaml(Path('configs/backend_studies/default.yaml'))
    assert cfg['name'] == 'qwen35_backend_workflow_study'
    assert 'prompt_pack' in cfg
    assert 'backend_configs' in cfg
