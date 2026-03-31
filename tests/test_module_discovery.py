import torch.nn as nn

from qwen35_turboquant_workflow_study.module_discovery import discover_attention_blocks


class FakeAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.k_proj = nn.Linear(8, 8)
        self.v_proj = nn.Linear(8, 8)


class FakeDeltaNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value_gate = nn.Linear(8, 8)


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                FakeDeltaNet(),
                FakeAttention(),
                FakeDeltaNet(),
                FakeAttention(),
            ]
        )


def test_discover_attention_blocks_filters_non_attention() -> None:
    model = FakeBackbone()
    blocks = discover_attention_blocks(model, expected_count=2)
    assert len(blocks) == 2
    assert blocks[0].module_path == "layers.1"
    assert blocks[1].module_path == "layers.3"
    assert blocks[0].q_proj_path == "layers.1.q_proj"
    assert blocks[0].k_proj_path == "layers.1.k_proj"
    assert blocks[0].v_proj_path == "layers.1.v_proj"
