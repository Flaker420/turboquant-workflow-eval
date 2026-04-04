from __future__ import annotations

import math

import torch

from turboquant_workflow_eval.stats import OnlineScalarStats, ProjectionTensorStats, take_first_tensor


class TestOnlineScalarStats:
    def test_empty(self) -> None:
        s = OnlineScalarStats()
        assert s.count == 0
        assert s.mean is None
        assert s.std is None
        assert s.minimum is None
        assert s.maximum is None

    def test_single_value(self) -> None:
        s = OnlineScalarStats()
        s.update(torch.tensor([5.0]))
        assert s.count == 1
        assert s.mean == pytest.approx(5.0)
        assert s.std == pytest.approx(0.0)
        assert s.minimum == pytest.approx(5.0)
        assert s.maximum == pytest.approx(5.0)

    def test_known_dataset(self) -> None:
        s = OnlineScalarStats()
        data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        s.update(torch.tensor(data))
        assert s.count == 8
        expected_mean = sum(data) / len(data)
        assert s.mean == pytest.approx(expected_mean)
        expected_var = sum((x - expected_mean) ** 2 for x in data) / len(data)
        assert s.std == pytest.approx(math.sqrt(expected_var))
        assert s.minimum == pytest.approx(2.0)
        assert s.maximum == pytest.approx(9.0)

    def test_incremental_updates(self) -> None:
        s = OnlineScalarStats()
        s.update(torch.tensor([1.0, 2.0]))
        s.update(torch.tensor([3.0, 4.0]))
        assert s.count == 4
        assert s.mean == pytest.approx(2.5)
        assert s.minimum == pytest.approx(1.0)
        assert s.maximum == pytest.approx(4.0)

    def test_empty_tensor_ignored(self) -> None:
        s = OnlineScalarStats()
        s.update(torch.tensor([]))
        assert s.count == 0

    def test_to_dict(self) -> None:
        s = OnlineScalarStats()
        s.update(torch.tensor([3.0, 6.0]))
        d = s.to_dict()
        assert set(d.keys()) == {"count", "mean", "std", "minimum", "maximum"}
        assert d["count"] == 2
        assert d["mean"] == pytest.approx(4.5)

    def test_scalar_input(self) -> None:
        s = OnlineScalarStats()
        s.update(5.0)
        assert s.count == 1
        assert s.mean == pytest.approx(5.0)


class TestProjectionTensorStats:
    def test_update_from_output(self) -> None:
        ps = ProjectionTensorStats()
        tensor = torch.randn(2, 4, 8)
        ps.update_from_output(tensor)
        assert ps.vector_norm.count > 0
        assert ps.abs_value.count > 0
        assert ps.channel_rms_mean.count > 0
        assert ps.channel_rms_max.count > 0
        assert len(ps.shape_examples) == 1
        assert ps.shape_examples[0] == [2, 4, 8]

    def test_shape_examples_capped_at_5(self) -> None:
        ps = ProjectionTensorStats()
        for i in range(7):
            ps.update_from_output(torch.randn(1, i + 1, 4))
        assert len(ps.shape_examples) == 5

    def test_to_dict_structure(self) -> None:
        ps = ProjectionTensorStats()
        ps.update_from_output(torch.randn(2, 4))
        d = ps.to_dict()
        assert set(d.keys()) == {"vector_norm", "abs_value", "channel_rms_mean", "channel_rms_max", "shape_examples"}

    def test_empty_tensor_ignored(self) -> None:
        ps = ProjectionTensorStats()
        ps.update_from_output(torch.tensor([]))
        assert ps.vector_norm.count == 0


class TestTakeFirstTensor:
    def test_tensor_input(self) -> None:
        t = torch.tensor([1.0])
        assert take_first_tensor(t) is t

    def test_tuple_input(self) -> None:
        t = torch.tensor([1.0])
        assert take_first_tensor((t, "not a tensor")) is t

    def test_dict_input(self) -> None:
        t = torch.tensor([1.0])
        assert take_first_tensor({"key": t}) is t

    def test_none_for_non_tensor(self) -> None:
        assert take_first_tensor("string") is None
        assert take_first_tensor(42) is None
        assert take_first_tensor(None) is None

    def test_empty_containers(self) -> None:
        assert take_first_tensor(()) is None
        assert take_first_tensor([]) is None
        assert take_first_tensor({}) is None


import pytest
