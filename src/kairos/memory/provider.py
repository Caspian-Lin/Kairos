"""Recall group memory using structured impressions, keywords, and summaries."""

import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Sequence

import structlog

from kairos.models import NormalizedMessage, WarmMemory
from kairos.protocols import MemoryProvider, MemoryStore, SummaryStore

_log = structlog.get_logger()
_WORDS = re.compile(r"[a-z][a-z0-9_-]{1,31}|[\u3400-\u9fff]+")
_EN_STOP = frozenset(
    [
        "about",
        "and",
        "are",
        "can",
        "could",
        "did",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "just",
        "like",
        "more",
        "not",
        "our",
        "out",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "too",
        "was",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)
_ZH_STOP = re.compile(
    r"今天|昨天|刚才|现在|大家|这个|那个|我们|你们|[我你他她它的了是在和有也都就很呢吗吧啊呀]"
)


def extract_keywords(messages: Sequence[NormalizedMessage], max_keywords: int = 8) -> list[str]:
    if max_keywords <= 0:
        return []
    counts: Counter[str] = Counter()
    for message in reversed(messages[-16:]):
        for word in _WORDS.findall(message.content[:256].casefold()):
            if word.isascii():
                if word not in _EN_STOP:
                    counts[word] += 1
                continue
            for phrase in _ZH_STOP.split(word):
                if 2 <= len(phrase) <= 8:
                    counts[phrase] += 1
                if len(phrase) > 2:
                    counts.update(phrase[index : index + 2] for index in range(len(phrase) - 1))
    return [word for word, _ in counts.most_common(max_keywords)]


async def _load[Item](
    source: str, query: Awaitable[list[Item]] | None, bot_id: str, group_id: int
) -> list[Item]:
    if query is None:
        return []
    try:
        return await query
    except Exception as exc:  # noqa: BLE001 - A failed source must not hide other memory.
        _log.warning(
            "memory_recall_source_failed",
            source=source,
            bot_id=bot_id,
            group_id=group_id,
            error_type=type(exc).__name__,
        )
        return []


class PgMemoryProvider(MemoryProvider):
    def __init__(
        self,
        memory_store: MemoryStore | None,
        summary_store: SummaryStore | None,
        *,
        max_impressions_recalled: int = 10,
        max_notes_recalled: int = 5,
        max_persisted_summaries: int = 3,
    ) -> None:
        self._memory_store = memory_store
        self._summary_store = summary_store
        self._max_impressions = max_impressions_recalled
        self._max_notes = max_notes_recalled
        self._max_summaries = max_persisted_summaries

    async def recall(
        self,
        bot_id: str,
        group_id: int,
        active_user_qqs: Sequence[int],
        recent_messages: Sequence[NormalizedMessage],
    ) -> WarmMemory:
        user_ids = list(dict.fromkeys(active_user_qqs))[: self._max_impressions]
        keywords = extract_keywords(recent_messages) if self._max_notes else []
        query_text = " ".join(message.content[:120] for message in recent_messages[-4:]).strip()
        impressions_query = (
            self._memory_store.load_user_impressions(bot_id, group_id, user_ids)
            if self._memory_store is not None and user_ids
            else None
        )
        notes_query = (
            self._memory_store.recall_memory_notes(
                bot_id, group_id, keywords, query_text, self._max_notes
            )
            if self._memory_store is not None and self._max_notes and (keywords or query_text)
            else None
        )
        summaries_query = (
            self._summary_store.load_recent_summaries(bot_id, group_id, self._max_summaries)
            if self._summary_store is not None and self._max_summaries
            else None
        )
        impressions, notes, summaries = await asyncio.gather(
            _load("impressions", impressions_query, bot_id, group_id),
            _load("notes", notes_query, bot_id, group_id),
            _load("summaries", summaries_query, bot_id, group_id),
        )
        return WarmMemory(
            impressions=impressions[: self._max_impressions],
            notes=notes[: self._max_notes],
            summaries=summaries[-self._max_summaries :] if self._max_summaries else [],
        )
