from mindlab_monitor.config import EyeSettings
from mindlab_monitor.vision.eyes import (
    EyeSample,
    compute_eye_stability,
    eye_aspect_ratio,
    eyes_open_from_ear,
)

# [outer, top1, top2, inner, bottom1, bottom2]
_OPEN_EYE = ((0.0, 0.5), (0.3, 0.0), (0.7, 0.0), (1.0, 0.5), (0.7, 1.0), (0.3, 1.0))
_CLOSED_EYE = ((0.0, 0.5), (0.3, 0.48), (0.7, 0.48), (1.0, 0.5), (0.7, 0.52), (0.3, 0.52))


def test_ear_open_is_larger_than_closed():
    assert eye_aspect_ratio(_OPEN_EYE) > eye_aspect_ratio(_CLOSED_EYE)


def test_eyes_open_from_ear_threshold():
    threshold = 0.21
    assert eyes_open_from_ear(eye_aspect_ratio(_OPEN_EYE), threshold) is True
    assert eyes_open_from_ear(eye_aspect_ratio(_CLOSED_EYE), threshold) is False


def _samples(pattern: list[tuple[float, bool]]) -> list[EyeSample]:
    return [EyeSample(timestamp=t, eyes_open=o) for t, o in pattern]


def test_sustained_closed_scores_perfectly_with_closed_target():
    settings = EyeSettings(target_state="closed", settle_seconds=0.0, min_dwell_seconds=1.0)
    samples = _samples([(t, False) for t in range(0, 60)])
    result = compute_eye_stability(samples, settings)
    assert result is not None
    assert result.deviation_ratio == 0.0
    assert result.transition_count == 0
    assert result.score == 100.0


def test_sustained_open_is_penalized_against_closed_target():
    settings = EyeSettings(
        target_state="closed",
        settle_seconds=0.0,
        min_dwell_seconds=1.0,
        deviation_penalty_cap=60.0,
        transition_penalty_cap=40.0,
    )
    samples = _samples([(t, True) for t in range(0, 60)])
    result = compute_eye_stability(samples, settings)
    assert result is not None
    assert result.deviation_ratio == 1.0
    assert result.score == 40.0  # 100 - full deviation cap, no transitions


def test_brief_blinks_are_ignored_via_min_dwell():
    settings = EyeSettings(target_state="closed", settle_seconds=0.0, min_dwell_seconds=1.0)
    # Eyes closed throughout, with a single-sample "blink" (open for 0.1s) that
    # should be absorbed rather than counted as deviation or a transition.
    pattern = [(float(t), False) for t in range(0, 30)]
    pattern.append((30.1, True))  # 0.1s blink
    pattern.append((30.2, False))
    pattern += [(float(t), False) for t in range(31, 60)]
    samples = _samples(pattern)
    result = compute_eye_stability(samples, settings)
    assert result is not None
    assert result.transition_count == 0
    assert result.deviation_ratio == 0.0


def test_sustained_transitions_are_penalized():
    settings = EyeSettings(
        target_state="closed",
        settle_seconds=0.0,
        min_dwell_seconds=1.0,
        reference_transitions_per_min=5.0,
        transition_penalty_cap=40.0,
    )
    # Alternate closed/open every 2 seconds for 60s -> well above 5 sustained
    # transitions/min, so the transition penalty should hit its cap.
    pattern = []
    state = False
    t = 0.0
    while t < 60.0:
        pattern.append((t, state))
        t += 2.0
        state = not state
    samples = _samples(pattern)
    result = compute_eye_stability(samples, settings)
    assert result is not None
    assert result.transitions_per_min >= 5.0
    assert result.score <= 60.0  # transition penalty capped at 40 off the top


def test_settle_window_is_excluded_from_scoring():
    settings = EyeSettings(target_state="closed", settle_seconds=20.0, min_dwell_seconds=1.0)
    # Open (deviating) only during the first 15s settle-in period, closed after.
    pattern = [(float(t), True) for t in range(0, 15)] + [(float(t), False) for t in range(15, 60)]
    samples = _samples(pattern)
    result = compute_eye_stability(samples, settings)
    assert result is not None
    assert result.deviation_ratio == 0.0
    assert result.score == 100.0


def test_target_state_none_disables_scoring():
    settings = EyeSettings(target_state="none")
    samples = _samples([(float(t), t % 2 == 0) for t in range(0, 30)])
    assert compute_eye_stability(samples, settings) is None
