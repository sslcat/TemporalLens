import numpy as np
import torch
from fastapi.testclient import TestClient
from research.stage_metrics import aggregate, stream_stats
from temporallens import api
from temporallens.two_stage import intervals, labeled_intervals, fusion_input


def test_proposals_merge_short_gaps_then_filter_short_spans():
    trace = np.array([0, 0.7, 0.7, 0, 0.7, 0.7, 0, 0, 0.9, 0])
    assert intervals(trace, 0.6, 1, 4) == [(1, 6)]
    assert intervals(trace, 0.6, 0, 4) == []
    assert intervals(np.zeros(20), 0.6, 3, 6) == []


def test_label_abstention_keeps_proposal_and_preserves_bounds():
    logits = torch.zeros(8, 5)
    logits[4:, 2] = 4
    spans = [(0, 4), (4, 8)]
    result = labeled_intervals(spans, "flow", 0.6, None, logits, {})
    assert len(result) == 1 and result[0][:3] == (2, 4, 8)
    assert spans == [(0, 4), (4, 8)]


def test_fusion_preserves_order_of_interval_halves():
    x = torch.arange(30).reshape(6, 5).float()
    y = fusion_input(x, x)
    assert y.shape == (20,)
    torch.testing.assert_close(
        fusion_input(x.flip(0), x.flip(0)), torch.cat((y[10:], y[:10]))
    )


def test_class_blind_candidate_can_be_class_aware_false_alarm():
    truth = [(1, 4, 16)]
    row = aggregate([stream_stats(truth, [(2, 4, 16)], 32, np.eye(4, dtype=int))])
    assert row["oracle_macro_f1"] == 1
    assert row["counts"][-1] == [0, 1, 1] and row["endpoint_mae_frames"] is None
    candidate = aggregate([stream_stats(truth, [(1, 4, 16)], 32)])
    assert candidate["recall"][-1] == 1


def test_unavailable_two_stage_http_status(monkeypatch):
    monkeypatch.setenv("TEMPORALLENS_DEVICE", "cpu")
    monkeypatch.setattr(api, "two_stage_ready", lambda: False)
    with TestClient(api.app) as client:
        assert client.get("/api/status").json()["two_stage_ready"] is False
        assert client.post("/api/analyze?mode=two-stage").status_code == 503
