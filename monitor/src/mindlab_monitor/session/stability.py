from __future__ import annotations

import statistics

from mindlab_monitor.config import StabilityWeights


def breath_regularity_score(bpm_estimates: list[float]) -> float | None:
    """Higher when successive breath-rate estimates within a session are
    consistent (steady breathing), lower when they swing around a lot."""
    if len(bpm_estimates) < 2:
        return None
    mean_bpm = statistics.mean(bpm_estimates)
    if mean_bpm <= 0:
        return None
    coefficient_of_variation = statistics.pstdev(bpm_estimates) / mean_bpm
    return max(0.0, 100.0 - coefficient_of_variation * 200.0)


def compute_stability_score(
    eye_score: float | None,
    movement_score: float | None,
    breath_score: float | None,
    weights: StabilityWeights,
) -> float:
    """Weighted blend of whichever component scores are available. A missing
    component (e.g. eyes excluded via target_eye_state="none", or breath not
    yet resolved) drops out and its weight is redistributed across the rest."""
    components: list[tuple[float, float]] = []
    if eye_score is not None:
        components.append((eye_score, weights.eyes))
    if movement_score is not None:
        components.append((movement_score, weights.movement))
    if breath_score is not None:
        components.append((breath_score, weights.breath))

    if not components:
        return 0.0

    total_weight = sum(w for _, w in components)
    if total_weight <= 0:
        return 0.0
    return sum(score * w for score, w in components) / total_weight
