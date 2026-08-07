from pathlib import Path

from mindlab_monitor.session.history_store import HistoryStore
from mindlab_monitor.session.models import SessionSample, SessionSummary


def _summary(session_id: str, started_at: float) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        started_at=started_at,
        duration_seconds=600.0,
        sample_count=2,
        eye_score=90.0,
        movement_score=85.0,
        breath_score=70.0,
        stability_score=84.0,
        avg_breath_bpm=13.5,
        eyes_deviation_ratio=0.05,
        eyes_transitions_per_min=1.2,
    )


def _samples() -> list[SessionSample]:
    return [
        SessionSample(timestamp=0.0, ear=0.3, eyes_open=True, stillness_score=95.0, breath_bpm=None),
        SessionSample(timestamp=1.0, ear=0.15, eyes_open=False, stillness_score=90.0, breath_bpm=13.0),
    ]


def test_save_and_list_summaries_roundtrip(tmp_path: Path):
    store = HistoryStore(tmp_path)
    summary = _summary("session-a", started_at=1000.0)
    store.save_session(summary, _samples())

    listed = store.list_summaries()
    assert len(listed) == 1
    assert listed[0] == summary


def test_load_samples_roundtrip(tmp_path: Path):
    store = HistoryStore(tmp_path)
    summary = _summary("session-b", started_at=2000.0)
    samples = _samples()
    store.save_session(summary, samples)

    loaded = store.load_samples("session-b")
    assert loaded == samples


def test_list_summaries_orders_most_recent_first(tmp_path: Path):
    store = HistoryStore(tmp_path)
    store.save_session(_summary("older", started_at=1000.0), _samples())
    store.save_session(_summary("newer", started_at=2000.0), _samples())

    listed = store.list_summaries()
    assert [s.session_id for s in listed] == ["newer", "older"]


def test_load_samples_missing_session_returns_empty(tmp_path: Path):
    store = HistoryStore(tmp_path)
    assert store.load_samples("does-not-exist") == []
