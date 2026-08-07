from mindlab_monitor.config import StabilityWeights
from mindlab_monitor.session.stability import breath_regularity_score, compute_stability_score


def test_breath_regularity_perfect_for_constant_bpm():
    assert breath_regularity_score([12.0, 12.0, 12.0]) == 100.0


def test_breath_regularity_drops_for_variable_bpm():
    steady = breath_regularity_score([12.0, 12.0, 12.0])
    variable = breath_regularity_score([6.0, 20.0, 8.0, 22.0])
    assert variable < steady


def test_breath_regularity_none_with_fewer_than_two_estimates():
    assert breath_regularity_score([]) is None
    assert breath_regularity_score([12.0]) is None


def test_stability_score_blends_all_components():
    weights = StabilityWeights(eyes=0.4, movement=0.4, breath=0.2)
    score = compute_stability_score(eye_score=80.0, movement_score=60.0, breath_score=100.0, weights=weights)
    assert score == 80.0 * 0.4 + 60.0 * 0.4 + 100.0 * 0.2


def test_missing_component_redistributes_weight():
    weights = StabilityWeights(eyes=0.4, movement=0.4, breath=0.2)
    # No breath estimate yet — its weight should drop out, not silently
    # shrink the overall score.
    score = compute_stability_score(eye_score=80.0, movement_score=60.0, breath_score=None, weights=weights)
    expected = (80.0 * 0.4 + 60.0 * 0.4) / 0.8
    assert abs(score - expected) < 1e-9


def test_no_components_returns_zero():
    weights = StabilityWeights()
    assert compute_stability_score(None, None, None, weights) == 0.0
