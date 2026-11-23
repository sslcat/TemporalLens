import numpy as np

from research.ipn_metrics import match, source_score, summarize


def test_known_interval_and_continuous_detection_are_scored_independently():
    dense = np.array([0] * 4 + [1] * 12 + [0] * 16)
    probabilities = np.tile([0.90, 0.04, 0.02, 0.02, 0.02], (8, 1))
    probabilities[1:4] = [0.37, 0.60, 0.01, 0.01, 0.01]
    report = summarize([source_score(probabilities, dense, 0.65, 2)])
    assert report["accuracy"] == 1
    assert report["segment_f1"] == [0, 0, 0]
    assert report["matches"] == 0 and report["misses"] == 1
    assert report["exact"] == 0 and report["boundary_mae_frames"] is None


def test_segment_matching_uses_each_same_class_interval_once():
    refs = [(1, 0, 12), (2, 20, 32)]
    predicted = [(1, 0, 11), (1, 1, 12), (3, 20, 32)]
    pairs = match(refs, predicted, 0.5)
    assert len(pairs) == 1 and pairs[0][0] == 0 and pairs[0][1] in (0, 1)
