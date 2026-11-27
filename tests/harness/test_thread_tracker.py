from datetime import UTC, datetime, timedelta

import pytest

from kairos.harness.thread_tracker import ThreadTracker, format_thread_board
from kairos.models import NormalizedMessage

_START = datetime(2026, 11, 27, tzinfo=UTC)


def _message(msg_id: str, seconds: int, *, reply_to: str | None = None) -> NormalizedMessage:
    at = _START + timedelta(seconds=seconds)
    return NormalizedMessage(
        bot_id="bot",
        group_id=1,
        msg_id=msg_id,
        last_msg_id=msg_id,
        sender_qq=int(msg_id),
        sender_alias=f"U{msg_id}",
        content="聊天",
        reply_to_msg_id=reply_to,
        is_bot=False,
        is_at_bot=False,
        is_reply_to_bot=False,
        is_peer_bot=False,
        created_at=at,
        latest_at=at,
    )


def test_reply_parent_takes_precedence_over_recent_thread() -> None:
    tracker = ThreadTracker()
    tracker.observe(_message("1", 0))
    tracker.observe(_message("2", 300))
    tracker.observe(_message("3", 601))
    tracker.observe(_message("4", 602, reply_to="2"))
    tracker.observe(_message("4", 602, reply_to="2"))

    older, newer = tracker.active_threads(_START + timedelta(seconds=602))
    assert (older.root_msg_id, older.latest_msg_id, older.message_count) == ("1", "4", 3)
    assert (newer.root_msg_id, newer.message_count) == ("3", 1)
    assert older.participants == ["U1", "U2", "U4"]
    assert "m1 → m4" in format_thread_board([older, newer])
    assert tracker.active_threads(_START, limit=1)[0].thread_id == older.thread_id


def test_heat_halves_after_ten_minutes_and_snapshots_are_independent() -> None:
    tracker = ThreadTracker()
    tracker.observe(_message("1", 0))
    tracker.observe(_message("2", 1))
    (warm,) = tracker.active_threads(_START + timedelta(seconds=1))
    (cool,) = tracker.active_threads(_START + timedelta(seconds=601))
    assert warm.heat == 2
    assert cool.heat == pytest.approx(warm.heat / 2)
    cool.participants.clear()
    assert tracker.active_threads(_START)[0].participants == ["U1", "U2"]


def test_threads_and_reply_lookup_are_bounded() -> None:
    tracker = ThreadTracker(max_threads=2, max_tracked_messages=3)
    tracker.observe(_message("1", 0))
    tracker.observe(_message("2", 301))
    tracker.observe(_message("3", 602))
    tracker.observe(_message("4", 603, reply_to="1"))
    tracker.observe(_message("5", 604))
    assert len(tracker._message_threads) == 3
    assert [item.root_msg_id for item in tracker.active_threads(_START)] == ["3", "2"]
    assert ThreadTracker().active_threads(_START) == []
