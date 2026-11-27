from datetime import UTC, datetime, timedelta

import pytest

from kairos.config import BotProfile
from kairos.harness.compactor import estimate_tokens
from kairos.harness.context_builder import ContextBuilder, format_relative_time
from kairos.harness.postprocess import process_llm_output
from kairos.models import (
    ActionType,
    ContextSummary,
    GateAction,
    GateDecision,
    MemoryNote,
    NormalizedMessage,
    ThreadSummary,
    UserImpression,
    WarmMemory,
)

NOW = datetime(2026, 11, 27, 12, tzinfo=UTC)


@pytest.fixture
def profile() -> BotProfile:
    return BotProfile(
        bot_id="companion",
        self_id=100,
        nickname="小满",
        persona_prompt="我叫小满，讲话短一点。",
        chat_model_key="chat",
    )


def _message(
    msg_id: int,
    content: str,
    age: float,
    *,
    own: bool = False,
    alias: str = "U1:阿明",
    reply_to: str | None = None,
) -> NormalizedMessage:
    timestamp = NOW - timedelta(seconds=age)
    return NormalizedMessage(
        bot_id="companion",
        group_id=200,
        msg_id=str(msg_id),
        last_msg_id=str(msg_id),
        sender_qq=100 if own else 101,
        sender_alias=alias,
        content=content,
        reply_to_msg_id=reply_to,
        is_bot=own,
        is_at_bot=False,
        is_reply_to_bot=False,
        is_peer_bot=False,
        created_at=timestamp,
        latest_at=timestamp,
    )


def test_group_history_keeps_group_and_assistant_turns_separate(profile: BotProfile) -> None:
    first_user = _message(1, "晚上吃啥", 13)
    bot = _message(3, "我想吃粉", 8, own=True, alias="小满")
    peer_bot = _message(2, "还没定", 10, alias="U2:咪咪").model_copy(update={"is_peer_bot": True})
    last_user = _message(4, "那就米粉", 2, reply_to="3")

    turns = ContextBuilder.build_messages(
        profile, [last_user, bot, first_user, peer_bot], ["U1:阿明", "U2:咪咪"], now=NOW
    )

    assert [turn.role for turn in turns] == ["system", "user", "assistant", "user"]
    assert profile.persona_prompt in turns[0].content
    assert "U1:阿明、U2:咪咪" in turns[0].content
    for marker in ("[mID]", "[-Xs]", "[U1:昵称]", "<|act:silent|>", "第一人称"):
        assert marker in turns[0].content
    assert "<|reason|>" not in turns[0].content
    assert turns[1].content == "[m1] [-13s] [U1:阿明]: 晚上吃啥\n[m2] [-10s] [U2:咪咪]: 还没定"
    assert turns[2].content == "我想吃粉"
    assert turns[3].content == "[m4] [-2s] [U1:阿明] (回复 m3): 那就米粉"


def test_leading_assistant_is_omitted_and_consecutive_replies_share_a_turn(
    profile: BotProfile,
) -> None:
    history = [
        _message(1, "孤立回复", 32, own=True),
        _message(2, "另一条孤立回复", 31, own=True),
        _message(3, "在吗", 30),
        _message(4, "在呀", 20, own=True),
        _message(5, "我马上来了", 10, own=True),
        _message(6, "好", 9),
    ]

    turns = ContextBuilder.build_messages(profile, history, [], now=NOW)

    assert [turn.role for turn in turns] == ["system", "user", "assistant", "user"]
    assert turns[2].content == "在呀\n我马上来了"
    assert all("孤立回复" not in turn.content for turn in turns)


def test_only_leading_assistant_leaves_system_turn(profile: BotProfile) -> None:
    turns = ContextBuilder.build_messages(profile, [_message(1, "我在", 2, own=True)], [], now=NOW)

    assert [turn.role for turn in turns] == ["system"]


@pytest.mark.parametrize(
    ("age", "expected"),
    [
        (-1.0, "[-0s]"),
        (0.0, "[-0s]"),
        (59.9, "[-59s]"),
        (60.0, "[-1m]"),
        (3599.9, "[-59m]"),
        (3600.0, "[-1h]"),
        (86399.9, "[-23h]"),
        (86400.0, "[-1d]"),
        (172800.0, "[-2d]"),
    ],
)
def test_relative_time_rendering(age: float, expected: str) -> None:
    assert format_relative_time(age) == expected


def test_user_block_has_quiet_break_only_after_five_minutes(profile: BotProfile) -> None:
    history = [
        _message(1, "早啊", 601),
        _message(2, "醒了吗", 301),
        _message(3, "有人吗", 0),
    ]

    turns = ContextBuilder.build_messages(profile, history, [], now=NOW)

    assert turns[1].content.splitlines() == [
        "[m1] [-10m] [U1:阿明]: 早啊",
        "[m2] [-5m] [U1:阿明]: 醒了吗",
        "--- [静默 5 分钟] ---",
        "[m3] [-0s] [U1:阿明]: 有人吗",
    ]


def test_quiet_break_after_assistant_starts_next_user_turn(profile: BotProfile) -> None:
    history = [
        _message(1, "明天一起吃饭？", 4301),
        _message(2, "好", 4300, own=True),
        _message(3, "刚忙完", 0),
    ]

    turns = ContextBuilder.build_messages(profile, history, [], now=NOW)

    assert turns[2].content == "好"
    assert turns[3].content.splitlines()[0] == "--- [静默 1 小时 11 分钟] ---"


def test_assistant_reply_never_carries_quiet_break_marker(profile: BotProfile) -> None:
    history = [_message(1, "在吗", 301), _message(2, "刚回来", 0, own=True)]

    turns = ContextBuilder.build_messages(profile, history, [], now=NOW)

    assert turns[1].content.endswith("--- [静默 5 分钟] ---")
    assert turns[2].content == "刚回来"


def test_quiet_break_counts_from_end_of_a_burst(profile: BotProfile) -> None:
    burst = _message(1, "早啊 / 有人没", 309).model_copy(
        update={"last_msg_id": "2", "latest_at": NOW - timedelta(seconds=300)}
    )

    turns = ContextBuilder.build_messages(profile, [burst, _message(3, "在呢", 0)], [], now=NOW)

    assert turns[1].content == ("[m1] [-5m] [U1:阿明]: 早啊 / 有人没\n[m3] [-0s] [U1:阿明]: 在呢")


def test_naive_utc_and_aware_times_share_timeline(profile: BotProfile) -> None:
    older = _message(1, "先发的", 3)
    older = older.model_copy(
        update={
            "created_at": older.created_at.replace(tzinfo=None),
            "latest_at": older.latest_at.replace(tzinfo=None),
        }
    )

    turns = ContextBuilder.build_messages(profile, [older, _message(2, "后来", -2)], [], now=NOW)

    assert turns[1].content == "[m1] [-3s] [U1:阿明]: 先发的\n[m2] [-0s] [U1:阿明]: 后来"


@pytest.mark.parametrize(
    "raw_text",
    ["<|act:silent|>", "   <|act:silent|>\n"],
)
def test_standalone_veto_marker(raw_text: str) -> None:
    result = process_llm_output(raw_text, [])

    assert result.vetoed is True
    assert result.repeated_dropped is False
    assert result.content == ""


def test_a_mentioned_marker_in_reply_is_ordinary_text() -> None:
    text = "我看到 <|act:silent|> 这串字符。"
    result = process_llm_output(text, [])
    assert not result.vetoed
    assert result.content == text


def test_empty_output_is_distinct_from_silence() -> None:
    result = process_llm_output(" \n ", [])
    assert not result.vetoed
    assert result.content == ""


def test_reply_body_preserves_paragraphs_and_literal_delimiters() -> None:
    text = "[周末约] 我有两个想法。\n\n第一段 || 第二段。"
    assert process_llm_output(f" \n{text}\n ", []).content == text


def test_bigram_repeat_detection_compares_latest_two_replies() -> None:
    repeated = process_llm_output("今天一起吃拉面吧", ["明天约球", "今天一起吃拉面"])
    old_only_match = process_llm_output("今天一起吃拉面吧", ["今天一起吃拉面", "好天气", "几点见"])

    assert repeated.vetoed is False
    assert repeated.repeated_dropped is True
    assert repeated.content == ""
    assert old_only_match.repeated_dropped is False
    assert old_only_match.content == "今天一起吃拉面吧"


def test_jaccard_threshold_is_strict_and_short_replies_compare_characters() -> None:
    exact_threshold = process_llm_output("abcdefgh", ["abcdefghijk"])
    repeated_ack = process_llm_output("嗯", ["嗯"])
    different_ack = process_llm_output("好", ["嗯"])

    assert exact_threshold.repeated_dropped is False
    assert exact_threshold.content == "abcdefgh"
    assert repeated_ack.repeated_dropped is True
    assert different_ack.content == "好"


def test_context_injects_scoped_memory_and_gate_suggestion(profile: BotProfile) -> None:
    prior = ContextSummary(
        bot_id=profile.bot_id,
        group_id=200,
        summary_text="开始聊做饭",
        covered_until_msg_id="10",
        covered_until_at=NOW - timedelta(minutes=10),
        token_estimate=5,
        created_at=NOW,
    )
    recent = prior.model_copy(
        update={"covered_until_at": NOW, "covered_until_msg_id": "11", "summary_text": "约好吃面"}
    )
    memory = WarmMemory(
        summaries=[recent, prior],
        impressions=[
            UserImpression(
                bot_id=profile.bot_id,
                group_id=200,
                user_qq=101,
                nickname="阿明",
                impression_summary="喜欢米粉",
                updated_at=NOW,
            )
        ],
        notes=[
            MemoryNote(
                bot_id=profile.bot_id,
                scope_group_id=200,
                entity_qqs=[101],
                summary_text="上周在成都吃过米粉",
                keywords=["米粉"],
                created_at=NOW,
            )
        ],
        threads=[
            ThreadSummary(
                thread_id="10",
                root_msg_id="10",
                latest_msg_id="11",
                participants=["阿明"],
                message_count=2,
                heat=2,
                last_active_at=NOW,
            )
        ],
    )
    prompt = ContextBuilder.build_messages(
        profile,
        [_message(12, "去哪吃？", 0)],
        [],
        now=NOW,
        warm_memory=memory,
        gate_decision=GateDecision(
            action=GateAction.PASS, candidate_target="12", reason="群友问地点"
        ),
        enable_react=True,
    )[0].content

    assert prompt.index("开始聊做饭") < prompt.index("约好吃面")
    for fragment in (
        "历史对话摘要",
        "[m10]",
        "群友画像",
        "阿明：喜欢米粉",
        "相关事件笔记",
        "上周在成都吃过米粉",
        "活跃线程板",
        "m10 → m11",
        "初筛建议目标 m12",
        "群友问地点",
        "仅供参考",
        "<|act:react:mID:emoji_id|>",
    ):
        assert fragment in prompt


def test_token_budget_cuts_early_messages_inside_a_user_block(profile: BotProfile) -> None:
    history = [
        _message(1, "过期话题" * 180, 10),
        _message(2, "一个旧的回复", 9, own=True),
        _message(3, "也过期了" * 180, 8),
        _message(4, "现在去吃饭？", 7),
        _message(5, "好啊", 6, own=True),
    ]
    turns = ContextBuilder.build_messages(profile, history, [], now=NOW, token_budget=512)
    assert [turn.role for turn in turns] == ["system", "user", "assistant"]
    assert "现在去吃饭？" in turns[1].content and "也过期了" not in turns[1].content
    assert turns[2].content == "好啊"
    assert sum(4 + estimate_tokens(turn.content) for turn in turns) <= 512


def test_token_budget_fits_an_oversized_latest_user_turn(profile: BotProfile) -> None:
    turns = ContextBuilder.build_messages(
        profile,
        [_message(9, "新话题" * 1000, 1)],
        [],
        now=NOW,
        token_budget=512,
    )
    assert [turn.role for turn in turns] == ["system", "user"]
    assert "[m9]" in turns[1].content
    assert sum(4 + estimate_tokens(turn.content) for turn in turns) <= 512


@pytest.mark.parametrize("target", ["m123", "123"])
def test_standalone_reaction_is_an_independent_action(target: str) -> None:
    result = process_llm_output(f" <|act:react:{target}:76|>\n", [])
    assert result.action.action is ActionType.REACT
    assert result.action.target_msg_id == "123"
    assert result.action.emoji_id == "76"
    assert result.content == "" and not result.vetoed and not result.repeated_dropped


@pytest.mark.parametrize("marker", ["<|act:react:mABC:76|>", "<|act:react:m123:|>"])
def test_invalid_standalone_reaction_is_discarded(marker: str) -> None:
    result = process_llm_output(marker, [])
    assert result.action.action is ActionType.SILENT
    assert result.action.target_msg_id is None and not result.vetoed


def test_literal_reaction_text_and_m1_opt_out_remain_ordinary_replies() -> None:
    marker = "<|act:react:123:76|>"
    text = f"提到了 {marker} 这个格式。"
    assert process_llm_output(text, []).action.content == text
    assert process_llm_output(marker, [], enable_react=False).content == marker
