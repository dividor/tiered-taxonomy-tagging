"""Ask a chat model for JSON, saving every reply so that a rerun does not ask or pay twice.

Replies live in a SQLite file in the data folder, because they would soon pass the repository's file size
limit. Only
transport failures are retried. A reply that is not a JSON object is recorded as a failure and never asked
again, so
that malformed replies count against a method rather than disappearing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import anthropic
import openai
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from tiered_taxonomy.models import ModelSpec, make_chat_model

# A provider can refuse with "too many requests" for many minutes at a time, so a call waits up to about half
# an hour in all before giving up: pauses double from 2 seconds and stop growing at one minute.
TRANSPORT_ATTEMPTS = 35
MAX_TRANSPORT_WAIT = 60
# Failures worth retrying. A refused key or exhausted credit is none of these, so it stops the run at once.
TRANSPORT_ERRORS = (
    ConnectionError,
    TimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
    openai.InternalServerError,
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
)
_SCHEMA = """
CREATE TABLE IF NOT EXISTS replies (
    key TEXT PRIMARY KEY, model TEXT, content TEXT, served_by TEXT,
    input_tokens INTEGER, output_tokens INTEGER, seconds REAL
)
"""


def transport_wait(attempt: int) -> float:
    """Return the seconds to wait after failed attempt `attempt`, counted from 0: doubling from 2 seconds, at
    most a minute."""
    return float(min(2 ** (attempt + 1), MAX_TRANSPORT_WAIT))


@dataclass(frozen=True)
class Reply:
    """A model's reply. `parsed` is None when the reply was not a JSON object."""

    content: str
    parsed: dict[str, Any] | None
    input_tokens: int
    output_tokens: int
    seconds: float
    from_cache: bool
    served_by: str = ""


def parse_json_object(content: str | None) -> dict[str, Any] | None:
    """Return the JSON object in a reply, removing whitespace and one code fence, or None if it is not one."""
    if not content or not content.strip():
        return None
    text = content.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def extract_json_object(content: str | None) -> dict[str, Any] | None:
    """Return the first JSON object in a reply with text around it, as models with no JSON mode write."""
    if not content or "{" not in content:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(content[content.index("{") :])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def request_key(model: str, system: str, user: str, max_tokens: int, temperature: float | None) -> str:
    """Return the hash that identifies a request."""
    body = json.dumps([model, system, user, max_tokens, temperature], ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


class ReplyStore:
    """A thread-safe store of model replies."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute(_SCHEMA)

    def get(self, key: str) -> Reply | None:
        """Return a saved reply, or None."""
        with self._lock:
            row = self._connection.execute(
                "SELECT content, served_by, input_tokens, output_tokens, seconds FROM replies WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        content, served_by, input_tokens, output_tokens, seconds = row
        return Reply(
            content, parse_json_object(content), input_tokens, output_tokens, seconds, True, served_by
        )

    def put(self, key: str, model: str, reply: Reply) -> None:
        """Save a reply."""
        row = (
            key,
            model,
            reply.content,
            reply.served_by,
            reply.input_tokens,
            reply.output_tokens,
            reply.seconds,
        )
        with self._lock, self._connection:
            self._connection.execute("INSERT OR REPLACE INTO replies VALUES (?, ?, ?, ?, ?, ?, ?)", row)

    def close(self) -> None:
        """Close the store."""
        with self._lock:
            self._connection.close()


def _usage(message: AIMessage) -> tuple[int, int]:
    usage = message.usage_metadata or {"input_tokens": 0, "output_tokens": 0}
    return int(usage["input_tokens"]), int(usage["output_tokens"])


def _text(message: AIMessage) -> str:
    """Return a reply's text. Claude replies with a list of blocks, thinking before text; only text counts."""
    if isinstance(message.content, str):
        return message.content
    blocks = [block for block in message.content if isinstance(block, dict)]
    if len(blocks) != len(message.content) or any("type" not in block for block in blocks):
        return json.dumps(message.content)
    return "".join(str(block.get("text", "")) for block in blocks if block["type"] == "text")


def _served_by(message: AIMessage) -> str:
    return str(message.response_metadata.get("model_name", ""))


def _unusable_reply(
    error: openai.LengthFinishReasonError | openai.ContentFilterFinishReasonError,
) -> AIMessage:
    usage = error.completion.usage if error.completion else None
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    return AIMessage(
        content="",
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )


class Client:
    """Asks one model through LangChain, reusing saved replies and counting tokens and calls."""

    def __init__(
        self,
        spec: ModelSpec,
        store: ReplyStore,
        chat_model: BaseChatModel | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.spec = spec
        self._store = store
        self._chat_model = chat_model if chat_model is not None else make_chat_model(spec)
        self._sleep = sleep
        self._lock = threading.Lock()
        self.counts: Counter[str] = Counter()
        self._given_model = chat_model

    def limited(self, max_tokens: int) -> Client:
        """Return a client for the same model with a different reply limit, sharing the saved replies. The
        limit is part of each request, so it changes which saved reply a request finds."""
        if max_tokens == self.spec.max_tokens:
            return self
        return Client(replace(self.spec, max_tokens=max_tokens), self._store, self._given_model, self._sleep)

    def _invoke(self, system: str, user: str) -> AIMessage:
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        for attempt in range(TRANSPORT_ATTEMPTS):
            try:
                reply = self._chat_model.invoke(messages)
                if not isinstance(reply, AIMessage):
                    raise TypeError(f"expected an AIMessage, got {type(reply).__name__}")
                return reply
            except (openai.LengthFinishReasonError, openai.ContentFilterFinishReasonError) as error:
                # The model stopped early, so there is no usable reply. It counts as malformed, like any
                # other.
                return _unusable_reply(error)
            except TRANSPORT_ERRORS:
                if attempt == TRANSPORT_ATTEMPTS - 1:
                    raise
                self._sleep(transport_wait(attempt))
        raise RuntimeError("transport retries ran out")

    def ask(self, system: str, user: str) -> Reply:
        """Return a model's reply to a request, from the store when it has been asked before."""
        key = request_key(self.spec.model, system, user, self.spec.max_tokens, self.spec.temperature)
        saved = self._store.get(key)
        if saved is not None:
            read = self._read_prose_reply(saved)
            self._count(read)
            return read
        started = time.monotonic()
        message = self._invoke(system, user)
        input_tokens, output_tokens = _usage(message)
        content = _text(message)
        reply = Reply(
            content,
            parse_json_object(content),
            input_tokens,
            output_tokens,
            time.monotonic() - started,
            False,
            _served_by(message),
        )
        self._store.put(key, self.spec.model, reply)
        read = self._read_prose_reply(reply)
        self._count(read)
        return read

    def _read_prose_reply(self, reply: Reply) -> Reply:
        if self.spec.json_mode or reply.parsed is not None:
            return reply
        return replace(reply, parsed=extract_json_object(reply.content))

    def _count(self, reply: Reply) -> None:
        with self._lock:
            self.counts["calls"] += 1
            self.counts["cached"] += int(reply.from_cache)
            self.counts["malformed"] += int(reply.parsed is None)
            self.counts["input_tokens"] += reply.input_tokens
            self.counts["output_tokens"] += reply.output_tokens

    def cost(self) -> float:
        """Return the estimated cost in US dollars of the tokens counted so far, including saved replies."""
        return self.spec.cost(self.counts["input_tokens"], self.counts["output_tokens"])


def run_all[ITEM, RESULT](
    work: Callable[[ITEM], RESULT], items: Iterable[ITEM], workers: int = 8
) -> list[RESULT]:
    """Apply a function to items on a thread pool, keeping their order. The first error stops the run."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(work, list(items)))


def make_client(spec: ModelSpec, store_path: Path) -> Client:
    """Return a client for a model with replies saved at a path."""
    return Client(spec, ReplyStore(store_path))
