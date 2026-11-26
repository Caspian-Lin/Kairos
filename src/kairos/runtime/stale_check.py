"""Choose how to deliver a reply after the conversation has advanced."""

from kairos.models import StaleAction, StaleDecision


def evaluate_stale(
    trigger_seq: int,
    current_seq: int,
    target_msg_id: str | None,
    direct_max_delta: int = 1,
    reply_max_delta: int = 5,
) -> StaleDecision:
    delta = max(0, current_seq - trigger_seq)
    if delta <= direct_max_delta:
        action = StaleAction.SEND_DIRECT
    elif delta <= reply_max_delta:
        action = StaleAction.SEND_WITH_REPLY
    else:
        action = StaleAction.DROP_STALE

    return StaleDecision(
        action=action,
        delta=delta,
        reply_to_msg_id=target_msg_id if action is StaleAction.SEND_WITH_REPLY else None,
    )


def compute_typing_delay(text: str, min_sec: float = 0.6, max_sec: float = 1.5) -> float:
    """Interpolate the delay up to its maximum at twenty characters."""
    progress = min(len(text) / 20, 1.0)
    return min_sec + (max_sec - min_sec) * progress
