import pytest

from kairos.models import StaleAction
from kairos.runtime.stale_check import compute_typing_delay, evaluate_stale


@pytest.mark.parametrize(
    ("advance", "action", "reply_to"),
    [
        (-1, StaleAction.SEND_DIRECT, None),
        (0, StaleAction.SEND_DIRECT, None),
        (1, StaleAction.SEND_DIRECT, None),
        (2, StaleAction.SEND_WITH_REPLY, "42"),
        (5, StaleAction.SEND_WITH_REPLY, "42"),
        (6, StaleAction.DROP_STALE, None),
    ],
)
def test_evaluate_stale(advance: int, action: StaleAction, reply_to: str | None) -> None:
    decision = evaluate_stale(10, 10 + advance, "42")

    assert decision.action is action
    assert decision.delta == max(0, advance)
    assert decision.reply_to_msg_id == reply_to


def test_evaluate_stale_uses_configured_thresholds() -> None:
    assert evaluate_stale(0, 1, "42", 0, 1).action is StaleAction.SEND_WITH_REPLY
    assert evaluate_stale(0, 2, "42", 0, 1).action is StaleAction.DROP_STALE


def test_typing_delay_interpolates_and_clamps() -> None:
    assert compute_typing_delay("") == 0.6
    assert compute_typing_delay("字" * 10) == pytest.approx(1.05)
    assert compute_typing_delay("字" * 100) == 1.5
    assert compute_typing_delay("字" * 10, min_sec=1.0, max_sec=3.0) == 2.0
