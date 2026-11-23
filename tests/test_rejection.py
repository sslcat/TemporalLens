import numpy as np
import pytest

from research.rejection_data import load_development, proposal_labels, verify_timeline
from research.rejection_metrics import (
    apply_rule,
    calibrate,
    confusion,
    pr_curve,
    reliability,
    score,
)
from research.rejection_study import leave_person_out, passes


def test_development_loader_rejects_any_other_split_before_cache_access():
    with pytest.raises(ValueError, match="only train and val"):
        load_development("test")


def test_candidate_label_is_one_to_one_and_partial_overlap_is_not_background_evidence():
    ref = [(3, 0, 20)]
    spans = [(0, 10), (10, 20), (25, 35)]
    mapping = proposal_labels(ref, spans)
    assert len(mapping) == 1 and set(mapping.values()) == {0} and 2 not in mapping
    verify_timeline([(0, 20, "G10"), (20, 35, "D0X")], np.array([3] * 20 + [0] * 15))
    with pytest.raises(ValueError):
        verify_timeline(
            [(0, 20, "G10"), (23, 35, "D0X")], np.array([3] * 20 + [0] * 15)
        )


def test_wrong_label_counts_as_fp_and_fn_and_confusion_has_rejection_column():
    y = np.array([1, 2, 0, 3])
    pred = np.array([1, 3, 2, 0])
    cm = confusion(y, pred)
    result = score(cm)
    assert [result[k] for k in ("tp", "fp", "fn")] == [1, 2, 2]
    assert cm[3, 0] == cm[0, 2] == 1


def test_explicit_background_rule_and_temperature_keep_target_ranking():
    logits = np.array([[9, 2, 1, 0, 0], [0, 5, 2, 1, 0]], float)
    cfg = dict(temperature=0.7, cutoff=0.0)
    raw = apply_rule("temperature", cfg, logits, None, 0.5)[0]
    joint = apply_rule("explicit_other", cfg, logits, None, 0.5)[0]
    assert raw.tolist() == [1, 1] and joint.tolist() == [0, 1]


def test_per_class_threshold_fallback_and_group_exclusion():
    people = np.array(["a", "a", "b", "c", "d", "d"])
    for person, fit, scored in leave_person_out(people):
        assert person not in people[fit] and set(people[scored]) == {person}
    x = np.array([[0, 3, 0, 0, 0], [0, 1, 3, 0, 0]], float)
    cfg = calibrate("per_class", x, np.array([1, 0]), None, 0.5)
    assert cfg["per_class"][1:] == [cfg["cutoff"]] * 3


def test_ranking_ties_are_scored_together_and_confidence_one_is_in_last_bin():
    top = np.array([1, 1, 2])
    curve = pr_curve(
        top, np.array([0.9, 0.9, 0.2]), np.ones(3, bool), np.array([1, 0, 2])
    )
    assert len(curve["points"]) == 2 and curve["points"][0]["precision"] == 0.5
    r = reliability(np.array([0.0, 1.0]), np.array([False, True]))
    assert r["ece"] == 0 and r["brier"] == 0


def test_improvement_from_one_person_does_not_clear_the_stopping_rule():
    old = dict(
        candidate=dict(tp=2, f1=0.2),
        people={"a": {"tp": 1}, "b": {"tp": 1}},
        stream={"false_per_min": 0.2},
    )
    new = dict(
        candidate=dict(tp=4, f1=0.4),
        people={"a": {"tp": 3}, "b": {"tp": 1}},
        stream={"false_per_min": 0.2},
    )
    assert not passes(new, old)["two_people_gain"]
