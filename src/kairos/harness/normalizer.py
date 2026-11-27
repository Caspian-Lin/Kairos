"""Convert OneBot group messages into stable speaker labels and readable text."""

import re
import unicodedata
from collections import OrderedDict
from collections.abc import Collection, Sequence

from kairos.models import MessageSegment, NormalizedMessage, RawGroupEvent

_NICKNAME_DECORATIONS = frozenset("【】[]「」『』〖〗《》〈〉〔〕（）(){}꧁꧂༺༻✨★☆✦✧")
_CQ_CODE = re.compile(r"\[CQ:([^\]]*)(?:\]|\Z)")
_SAVED_ALIAS = re.compile(r"U([1-9][0-9]*):(.+)")
_MEDIA_TEXT = {
    "image": "[图片]",
    "face": "[表情]",
    "mface": "[表情]",
    "record": "[语音]",
    "video": "[视频]",
    "forward": "[合并转发]",
    "json": "[卡片消息]",
    "xml": "[卡片消息]",
}


def clean_nickname(card: str, nickname: str, max_len: int = 6) -> str:
    """Choose a display name and remove decoration and invisible characters."""
    name = card.strip() or nickname
    visible = "".join(
        " " if char.isspace() else char
        for char in name
        if char not in _NICKNAME_DECORATIONS
        and char not in "\u034f\ufe0e\ufe0f"
        and (char.isspace() or not unicodedata.category(char).startswith("C"))
    )
    short_name = " ".join(visible.split())[:max_len].rstrip()
    return short_name if any(char.isalnum() for char in short_name) else "群友"


class UidAliasTable:
    """Bounded per-group LRU of speaker aliases with increasing short IDs."""

    def __init__(self, max_size: int = 200) -> None:
        if max_size < 1:
            raise ValueError("max_size must be positive")
        self._max_size = max_size
        self._next_id = 1
        self._aliases: OrderedDict[int, str] = OrderedDict()

    def __len__(self) -> int:
        return len(self._aliases)

    def lookup(self, qq: int) -> str | None:
        alias = self._aliases.get(qq)
        if alias is not None:
            self._aliases.move_to_end(qq)
        return alias

    def get_or_assign(self, qq: int, card: str = "", nickname: str = "") -> str:
        alias = self.lookup(qq)
        if alias is not None:
            return alias

        alias = f"U{self._next_id}:{clean_nickname(card, nickname)}"
        self._next_id += 1
        self._remember(qq, alias)
        return alias

    def restore_alias(self, qq: int, sender_alias: str) -> None:
        """Restore a previously normalized speaker alias during chronological warmup."""
        match = _SAVED_ALIAS.fullmatch(sender_alias)
        if match is None:
            return
        self._next_id = max(self._next_id, int(match[1]) + 1)
        self._remember(qq, sender_alias)

    def _remember(self, qq: int, alias: str) -> None:
        self._aliases[qq] = alias
        self._aliases.move_to_end(qq)
        if len(self._aliases) > self._max_size:
            self._aliases.popitem(last=False)


def build_roster(messages: Sequence[NormalizedMessage]) -> list[str]:
    """Return active human aliases with the most recent speaker first."""
    seen: set[int] = set()
    roster: list[str] = []
    for message in reversed(messages):
        if message.is_bot or message.is_peer_bot or message.sender_qq in seen:
            continue
        seen.add(message.sender_qq)
        roster.append(message.sender_alias)
    return roster


def _unescape_cq(value: str, *, parameter: bool = False) -> str:
    if parameter:
        value = value.replace("&#44;", ",")
    return value.replace("&#91;", "[").replace("&#93;", "]").replace("&amp;", "&")


def _parse_cq_message(raw_message: str) -> list[MessageSegment]:
    segments: list[MessageSegment] = []
    end = 0
    for match in _CQ_CODE.finditer(raw_message):
        if match.start() > end:
            text = _unescape_cq(raw_message[end : match.start()])
            segments.append(MessageSegment(type="text", data={"text": text}))

        kind, _, raw_parameters = match[1].partition(",")
        data: dict[str, str] = {}
        for parameter in raw_parameters.split(","):
            key, equals, value = parameter.partition("=")
            if equals:
                data[key] = _unescape_cq(value, parameter=True)
        segments.append(MessageSegment(type=kind, data=data))
        end = match.end()

    if end < len(raw_message):
        text = _unescape_cq(raw_message[end:])
        segments.append(MessageSegment(type="text", data={"text": text}))
    return segments


def _as_qq(value: object) -> int | None:
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_message_id(value: object) -> str | None:
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        return None
    return str(value).strip() or None


class MessageNormalizer:
    """Normalize events for one bot in one group."""

    def __init__(
        self, bot_id: str, bot_self_id: int, alias_table: UidAliasTable, *, bot_name: str = ""
    ) -> None:
        self._bot_id = bot_id
        self._bot_self_id = bot_self_id
        self._alias_table = alias_table
        self._bot_name = bot_name.strip().casefold()

    def normalize(
        self,
        event: RawGroupEvent,
        bot_msg_ids: Collection[str],
        hosted_bot_ids: Collection[int],
    ) -> NormalizedMessage | None:
        segments = event.segments or _parse_cq_message(event.raw_message)
        parts: list[str] = []
        plain_text: list[str] = []
        is_at_bot = False
        reply_to_msg_id: str | None = None

        for segment in segments:
            if segment.type != "text":
                plain_text.append(" ")
            match segment.type:
                case "text":
                    text = segment.data.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        plain_text.append(text)
                case "at":
                    target = segment.data.get("qq")
                    if target == "all":
                        parts.append("@全体成员")
                        continue
                    qq = _as_qq(target)
                    if qq == self._bot_self_id:
                        is_at_bot = True
                        parts.append("@我")
                    else:
                        alias = self._alias_table.lookup(qq) if qq is not None else None
                        parts.append(f"@[{alias or 'U?'}]")
                case "reply":
                    if reply_to_msg_id is None:
                        reply_to_msg_id = _as_message_id(segment.data.get("id"))
                case _:
                    parts.append(_MEDIA_TEXT.get(segment.type, ""))

        content = " ".join("".join(parts).split())
        if not content:
            return None

        is_bot = event.sender_qq == self._bot_self_id
        return NormalizedMessage(
            bot_id=self._bot_id,
            group_id=event.group_id,
            msg_id=event.msg_id,
            last_msg_id=event.msg_id,
            sender_qq=event.sender_qq,
            sender_alias=self._alias_table.get_or_assign(
                event.sender_qq, event.sender_card, event.sender_nickname
            ),
            content=content,
            reply_to_msg_id=reply_to_msg_id,
            is_bot=is_bot,
            is_at_bot=is_at_bot,
            is_reply_to_bot=reply_to_msg_id is not None and reply_to_msg_id in bot_msg_ids,
            mentions_bot_name=bool(
                self._bot_name and self._bot_name in "".join(plain_text).casefold()
            ),
            is_peer_bot=not is_bot and event.sender_qq in hosted_bot_ids,
            created_at=event.timestamp,
            latest_at=event.timestamp,
        )


def coalesce_message(
    messages: Sequence[NormalizedMessage],
    incoming: NormalizedMessage,
    window_seconds: float = 10.0,
) -> tuple[NormalizedMessage, bool]:
    """Return the merged tail or the new item without mutating either input."""
    if not messages:
        return incoming, False

    last = messages[-1]
    gap_seconds = (incoming.created_at - last.latest_at).total_seconds()
    if (
        last.is_bot
        or incoming.is_bot
        or last.is_peer_bot
        or incoming.is_peer_bot
        or (last.bot_id, last.group_id, last.sender_qq)
        != (incoming.bot_id, incoming.group_id, incoming.sender_qq)
        or not 0 <= gap_seconds <= window_seconds
        or (
            incoming.reply_to_msg_id is not None
            and incoming.reply_to_msg_id != last.reply_to_msg_id
        )
    ):
        return incoming, False

    return last.model_copy(
        update={
            "content": f"{last.content} / {incoming.content}",
            "last_msg_id": incoming.msg_id,
            "latest_at": incoming.latest_at,
            "is_at_bot": last.is_at_bot or incoming.is_at_bot,
            "is_reply_to_bot": last.is_reply_to_bot or incoming.is_reply_to_bot,
            "mentions_bot_name": last.mentions_bot_name or incoming.mentions_bot_name,
        }
    ), True
