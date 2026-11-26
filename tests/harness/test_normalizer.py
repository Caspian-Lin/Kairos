from collections import deque
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from kairos.harness.normalizer import (
    MessageNormalizer,
    UidAliasTable,
    build_roster,
    clean_nickname,
    coalesce_message,
)
from kairos.models import MessageSegment, NormalizedMessage, RawGroupEvent

NOW = datetime(2026, 1, 1, tzinfo=UTC)
BOT_QQ = 90001
PEER_QQ = 90002
SENDER_QQ = 70001


def make_event(
    *,
    segments: list[MessageSegment] | None = None,
    raw_message: str = "",
    sender_qq: int = SENDER_QQ,
) -> RawGroupEvent:
    return RawGroupEvent(
        self_id=BOT_QQ,
        group_id=80001,
        msg_id="88",
        sender_qq=sender_qq,
        sender_nickname="原昵称",
        sender_card="【✨小明】",
        segments=segments or [],
        raw_message=raw_message,
        timestamp=NOW,
    )


def message(**overrides: Any) -> NormalizedMessage:
    fields = {
        "bot_id": "lin",
        "group_id": 80001,
        "msg_id": "1",
        "last_msg_id": "1",
        "sender_qq": SENDER_QQ,
        "sender_alias": "U1:小明",
        "content": "前半句",
        "reply_to_msg_id": None,
        "is_bot": False,
        "is_at_bot": False,
        "is_reply_to_bot": False,
        "is_peer_bot": False,
        "created_at": NOW,
        "latest_at": NOW,
    }
    fields.update(overrides)
    return NormalizedMessage.model_validate(fields)


@pytest.mark.parametrize(
    ("card", "nickname", "max_len", "expected"),
    [
        ("【✨阿\u200b明\x00★】", "别名", 6, "阿明"),
        ("  ", " ꧁ 火星\t 少年 ꧂ ", 6, "火星 少年"),
        ("超长群昵称一二三四五", "别名", 6, "超长群昵称一"),
        ("★[]✨💫\u200d", "别名", 6, "群友"),
        ("", "阿明阿红", 3, "阿明阿"),
    ],
)
def test_clean_nickname(card: str, nickname: str, max_len: int, expected: str) -> None:
    assert clean_nickname(card, nickname, max_len) == expected


def test_aliases_remain_stable_until_lru_eviction() -> None:
    aliases = UidAliasTable(max_size=2)
    assert aliases.get_or_assign(1, "", "小明") == "U1:小明"
    assert aliases.get_or_assign(2, "", "小红") == "U2:小红"
    assert aliases.get_or_assign(1, "新群名片") == "U1:小明"
    assert aliases.get_or_assign(3, "", "小绿") == "U3:小绿"
    assert len(aliases) == 2
    assert aliases.lookup(2) is None
    assert aliases.lookup(1) == "U1:小明"
    assert aliases.get_or_assign(2, "", "小红") == "U4:小红"
    assert len(aliases) == 2
    assert aliases.lookup(3) is None


def test_warmup_aliases_resume_the_id_sequence() -> None:
    aliases = UidAliasTable(max_size=2)
    aliases.restore_alias(10001, "U8:小明")
    aliases.restore_alias(10002, "U6:小红")
    assert aliases.get_or_assign(10001, "别名") == "U8:小明"
    assert aliases.get_or_assign(10003, "", "小绿") == "U9:小绿"
    assert aliases.lookup(10002) is None
    assert len(aliases) == 2


def test_roster_is_deduplicated_by_qq_in_recency_order_and_excludes_bots() -> None:
    messages = deque(
        [
            message(sender_qq=10001, sender_alias="U1:旧名"),
            message(sender_qq=10002, sender_alias="U2:小红"),
            message(sender_qq=BOT_QQ, sender_alias="机器人", is_bot=True),
            message(sender_qq=10001, sender_alias="U8:小明"),
            message(sender_qq=PEER_QQ, sender_alias="U9:同伴", is_peer_bot=True),
        ]
    )
    assert build_roster(messages) == ["U8:小明", "U2:小红"]


def test_normalize_array_mentions_and_reply() -> None:
    aliases = UidAliasTable()
    aliases.get_or_assign(70002, "", "小红")
    normalizer = MessageNormalizer("lin", BOT_QQ, aliases)
    event = make_event(
        raw_message="不用重复渲染",
        segments=[
            MessageSegment(type="reply", data={"id": 33}),
            MessageSegment(type="text", data={"text": "  打招呼\t\n"}),
            MessageSegment(type="at", data={"qq": str(BOT_QQ)}),
            MessageSegment(type="at", data={"qq": 70002}),
            MessageSegment(type="at", data={"qq": "all"}),
            MessageSegment(type="at", data={"qq": "70003"}),
        ],
    )

    result = normalizer.normalize(event, {"33"}, {BOT_QQ, PEER_QQ})

    assert result is not None
    assert result.bot_id == "lin"
    assert result.group_id == event.group_id
    assert result.msg_id == result.last_msg_id == "88"
    assert result.created_at == result.latest_at == NOW
    assert result.sender_alias == "U2:小明"
    assert result.content == "打招呼 @我@[U1:小红]@全体成员@[U?]"
    assert result.reply_to_msg_id == "33"
    assert result.is_at_bot and result.is_reply_to_bot


@pytest.mark.parametrize(
    ("kind", "placeholder"),
    [
        ("image", "[图片]"),
        ("face", "[表情]"),
        ("mface", "[表情]"),
        ("record", "[语音]"),
        ("video", "[视频]"),
        ("forward", "[合并转发]"),
        ("json", "[卡片消息]"),
        ("xml", "[卡片消息]"),
    ],
)
def test_normalize_media_renders_only_the_placeholder(kind: str, placeholder: str) -> None:
    normalizer = MessageNormalizer("lin", BOT_QQ, UidAliasTable())
    event = make_event(
        segments=[
            MessageSegment(
                type=kind,
                data={"url": "https://example.test/secret", "file": "base64://c2VjcmV0"},
            )
        ]
    )
    result = normalizer.normalize(event, set(), {BOT_QQ})
    assert result is not None
    assert result.content == placeholder


def test_normalize_cq_string_unescapes_values_and_drops_media_payload() -> None:
    event = make_event(
        raw_message=(
            "&#91;你好&#93; &amp; 主播  "
            f"[CQ:at,qq={BOT_QQ}]"
            "[CQ:reply,id=msg&#44;33]"
            "[CQ:image,file=base64://abc123,url=https://example.test/a?x&#44;y]"
            "[CQ:face,id=7][CQ:record,file=voice][CQ:forward,id=forward-id]"
            "[CQ:unsupported,file=private_data]"
        )
    )

    result = MessageNormalizer("lin", BOT_QQ, UidAliasTable()).normalize(
        event, {"msg,33"}, {BOT_QQ}
    )

    assert result is not None
    assert result.content == "[你好] & 主播 @我[图片][表情][语音][合并转发]"
    assert result.reply_to_msg_id == "msg,33"
    assert result.is_at_bot and result.is_reply_to_bot


def test_normalize_truncated_cq_media_omits_data() -> None:
    event = make_event(raw_message="看看[CQ:image,url=https://example.test/private")
    result = MessageNormalizer("lin", BOT_QQ, UidAliasTable()).normalize(event, set(), set())
    assert result is not None
    assert result.content == "看看[图片]"


def test_normalize_ignores_reply_only_and_other_nonsemantic_input() -> None:
    aliases = UidAliasTable()
    event = make_event(
        segments=[
            MessageSegment(type="reply", data={"id": 33}),
            MessageSegment(type="text", data={"text": " \t "}),
            MessageSegment(type="unknown", data={"text": "禁止透传"}),
        ]
    )
    assert MessageNormalizer("lin", BOT_QQ, aliases).normalize(event, {"33"}, {BOT_QQ}) is None
    assert len(aliases) == 0


@pytest.mark.parametrize(
    ("sender", "expected_flags"), [(BOT_QQ, (True, False)), (PEER_QQ, (False, True))]
)
def test_normalize_distinguishes_own_messages_from_hosted_peers(
    sender: int, expected_flags: tuple[bool, bool]
) -> None:
    event = make_event(sender_qq=sender, raw_message="正文[CQ:reply,id=42]")
    result = MessageNormalizer("lin", BOT_QQ, UidAliasTable()).normalize(
        event, {"33"}, {BOT_QQ, PEER_QQ}
    )
    assert result is not None
    assert (result.is_bot, result.is_peer_bot) == expected_flags
    assert not result.is_reply_to_bot


def test_coalescing_keeps_first_and_last_ids_flags_and_times() -> None:
    last = message(
        last_msg_id="2",
        latest_at=NOW + timedelta(seconds=8),
        reply_to_msg_id="55",
        is_reply_to_bot=True,
    )
    incoming = message(
        msg_id="3",
        last_msg_id="3",
        content="后半句",
        created_at=NOW + timedelta(seconds=18),
        latest_at=NOW + timedelta(seconds=18),
        is_at_bot=True,
    )
    history = deque([last])

    combined, merged = coalesce_message(history, incoming)

    assert merged
    assert combined.content == "前半句 / 后半句"
    assert combined.msg_id == last.msg_id == "1"
    assert combined.last_msg_id == "3"
    assert combined.created_at == NOW
    assert combined.latest_at == incoming.latest_at
    assert combined.reply_to_msg_id == "55"
    assert combined.is_at_bot and combined.is_reply_to_bot
    assert history[-1] is last
    assert last.last_msg_id == "2"


def test_coalescing_accepts_same_reply_target() -> None:
    first = message(reply_to_msg_id="55", is_at_bot=True)
    second = message(msg_id="2", last_msg_id="2", reply_to_msg_id="55", is_reply_to_bot=True)

    combined, merged = coalesce_message([first], second)

    assert merged
    assert combined.is_at_bot and combined.is_reply_to_bot


@pytest.mark.parametrize(
    ("last_changes", "incoming_changes"),
    [
        ({}, {"sender_qq": 80001}),
        ({}, {"created_at": NOW + timedelta(seconds=10.001)}),
        ({}, {"created_at": NOW - timedelta(seconds=1)}),
        ({"reply_to_msg_id": "31"}, {"reply_to_msg_id": "32"}),
        ({}, {"reply_to_msg_id": "32"}),
        ({"is_bot": True}, {}),
        ({}, {"is_bot": True}),
        ({"is_peer_bot": True}, {}),
        ({}, {"is_peer_bot": True}),
        ({}, {"bot_id": "alternative-bot"}),
        ({}, {"group_id": 80002}),
    ],
)
def test_coalescing_declines_incompatible_tail(
    last_changes: dict[str, Any], incoming_changes: dict[str, Any]
) -> None:
    incoming = message(**incoming_changes)
    assert coalesce_message([message(**last_changes)], incoming) == (incoming, False)


def test_coalescing_never_merges_across_another_participant() -> None:
    incoming = message(msg_id="3", last_msg_id="3")
    history = [message(), message(msg_id="2", last_msg_id="2", sender_qq=70002)]
    assert coalesce_message(history, incoming) == (incoming, False)
    assert coalesce_message([], incoming) == (incoming, False)
