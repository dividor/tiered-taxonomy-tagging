"""Tests for the reply layer, using LangChain's fake chat model so nothing is sent or paid for."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion

from tiered_taxonomy.llm import (
    Client,
    ReplyStore,
    extract_json_object,
    parse_json_object,
    request_key,
    run_all,
)
from tiered_taxonomy.models import (
    ModelSpec,
    batches,
    hf_token,
    make_embeddings,
    spec_from_config,
)

SPEC = ModelSpec("fake", "fake/model", input_price=1.0, output_price=2.0)


def fake_client(tmp_path: Path, *replies: str) -> tuple[Client, GenericFakeChatModel]:
    model = GenericFakeChatModel(messages=iter(AIMessage(content=reply) for reply in replies))
    return Client(SPEC, ReplyStore(tmp_path / "replies.sqlite"), model), model


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ("not json", None),
        ("[1, 2]", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_json_object(content: str | None, expected: dict[str, Any] | None) -> None:
    assert parse_json_object(content) == expected


def test_a_saved_reply_is_reused_and_not_asked_again(tmp_path: Path) -> None:
    client, _ = fake_client(tmp_path, '{"n": 1}')
    first = client.ask("system", "user")
    second = client.ask("system", "user")
    assert first.parsed == {"n": 1}
    assert not first.from_cache
    assert second.from_cache
    assert second.parsed == {"n": 1}
    assert client.counts["calls"] == 2
    assert client.counts["cached"] == 1


def test_a_malformed_reply_is_counted_and_not_asked_again(tmp_path: Path) -> None:
    client, _ = fake_client(tmp_path, "nonsense")
    assert client.ask("s", "u").parsed is None
    assert client.ask("s", "u").parsed is None
    assert client.counts["malformed"] == 2


def test_cost_uses_the_configured_prices() -> None:
    assert SPEC.cost(1_000_000, 1_000_000) == pytest.approx(3.0)


def test_request_key_depends_on_every_part_of_the_request() -> None:
    base = request_key("m", "s", "u", 10, 0.0)
    assert base == request_key("m", "s", "u", 10, 0.0)
    assert base != request_key("m2", "s", "u", 10, 0.0)
    assert base != request_key("m", "s", "u2", 10, 0.0)


class Flaky:
    """A chat model that fails with a connection error before answering."""

    def __init__(self, failures: int) -> None:
        self.failures = failures

    def invoke(self, _messages: object) -> AIMessage:
        if self.failures:
            self.failures -= 1
            raise ConnectionError("down")
        return AIMessage(content='{"ok": true}')


def test_transport_errors_are_retried_then_raised(tmp_path: Path) -> None:
    waits: list[float] = []
    store = ReplyStore(tmp_path / "r.sqlite")
    flaky: Any = Flaky(2)
    client = Client(SPEC, store, flaky, sleep=waits.append)
    assert client.ask("s", "u").parsed == {"ok": True}
    assert waits == [2, 4]
    broken: Any = Flaky(10)
    failing = Client(SPEC, store, broken, sleep=waits.append)
    with pytest.raises(ConnectionError):
        failing.ask("s", "another request")


class CutOff:
    """A chat model whose reply stops at the token limit, which the OpenAI client raises as an error."""

    def invoke(self, _messages: object) -> AIMessage:
        completion = ChatCompletion(
            id="x",
            choices=[],
            created=0,
            model="m",
            object="chat.completion",
            usage=CompletionUsage(prompt_tokens=7, completion_tokens=11, total_tokens=18),
        )
        raise openai.LengthFinishReasonError(completion=completion)


def test_a_reply_cut_off_at_the_limit_counts_as_malformed_and_keeps_its_tokens(tmp_path: Path) -> None:
    model: Any = CutOff()
    client = Client(SPEC, ReplyStore(tmp_path / "r.sqlite"), model)
    reply = client.ask("s", "u")
    assert reply.parsed is None
    assert reply.output_tokens == 11
    assert client.counts["malformed"] == 1
    assert client.ask("s", "u").from_cache


def test_run_all_keeps_order() -> None:
    assert run_all(lambda x: x * 2, [1, 2, 3], workers=2) == [2, 4, 6]


def test_hf_token_prefers_the_variable_then_the_login_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    login = tmp_path / "token"
    login.write_text("from-file\n")
    monkeypatch.setenv("HF_TOKEN", "from-env")
    assert hf_token(login) == "from-env"
    monkeypatch.delenv("HF_TOKEN")
    assert hf_token(login) == "from-file"
    with pytest.raises(RuntimeError, match="no Hugging Face token"):
        hf_token(tmp_path / "missing")


def test_unknown_embeddings_provider_raises() -> None:
    with pytest.raises(ValueError, match="unknown embeddings provider"):
        make_embeddings("nobody", "x")


def test_spec_from_config_and_batches() -> None:
    assert spec_from_config({"provider": "p", "model": "m"}).model == "m"
    assert spec_from_config({"provider": "p", "model": "m"}).json_mode
    assert not spec_from_config({"provider": "p", "model": "m", "json_mode": False}).json_mode
    assert batches(["a", "b", "c"], 2) == [["a", "b"], ["c"]]


class Canned:
    """A chat model that always gives the same reply."""

    def __init__(self, content: str) -> None:
        self.content = content

    def invoke(self, _messages: object) -> AIMessage:
        return AIMessage(content=self.content)


def test_extract_json_object_finds_it_among_prose() -> None:
    assert extract_json_object('Here it is:\n```\n{"a": 1}\n```\nI hope that helps.') == {"a": 1}
    assert extract_json_object('{"a": [1, 2') is None
    assert extract_json_object("no braces") is None
    assert extract_json_object(None) is None


def test_prose_around_json_is_only_read_for_models_with_no_json_mode(tmp_path: Path) -> None:
    prose = 'Sure: {"a": 1} done'
    for json_mode, expected in ((True, None), (False, {"a": 1})):
        spec = ModelSpec("fake", f"m{json_mode}", json_mode=json_mode)
        model: Any = Canned(prose)
        client = Client(spec, ReplyStore(tmp_path / "r.sqlite"), model)
        assert client.ask("s", "u").parsed == expected
        assert client.ask("s", "u").parsed == expected


class Blocks:
    """A chat model that replies as Claude does: a list of blocks, a thinking block before the text."""

    def invoke(self, _messages: object) -> AIMessage:
        return AIMessage(
            content=[{"type": "thinking", "thinking": ""}, {"type": "text", "text": '{"a": 1}'}],
            usage_metadata={"input_tokens": 3, "output_tokens": 5, "total_tokens": 8},
        )


def test_a_reply_in_blocks_is_read_from_its_text_blocks_only(tmp_path: Path) -> None:
    model: Any = Blocks()
    reply = Client(SPEC, ReplyStore(tmp_path / "r.sqlite"), model).ask("s", "u")
    assert reply.content == '{"a": 1}'
    assert reply.parsed == {"a": 1}
    assert (reply.input_tokens, reply.output_tokens) == (3, 5)


class Overloaded:
    """A chat model whose provider is busy for a set number of calls, as Anthropic's API reports it."""

    def __init__(self, failures: int) -> None:
        self.failures = failures

    def invoke(self, _messages: object) -> AIMessage:
        if self.failures:
            self.failures -= 1
            raise anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.org"))
        return AIMessage(content='{"ok": true}')


def test_anthropic_transport_errors_are_retried(tmp_path: Path) -> None:
    waits: list[float] = []
    model: Any = Overloaded(1)
    client = Client(SPEC, ReplyStore(tmp_path / "r.sqlite"), model, sleep=waits.append)
    assert client.ask("s", "u").parsed == {"ok": True}
    assert waits == [2]


def test_a_limited_client_shares_saved_replies_but_asks_a_different_request(tmp_path: Path) -> None:
    store = ReplyStore(tmp_path / "r.sqlite")
    client = Client(SPEC, store, GenericFakeChatModel(messages=iter([AIMessage(content='{"ok": true}')] * 3)))
    assert client.limited(SPEC.max_tokens) is client
    limited = client.limited(4000)
    assert limited.spec.max_tokens == 4000 and limited.spec.model == SPEC.model
    first = limited.ask("s", "u")
    again = limited.ask("s", "u")
    assert first.parsed == again.parsed == {"ok": True}
    assert (limited.counts["calls"], limited.counts["cached"]) == (2, 1)
    client.ask("s", "u")
    assert client.counts["cached"] == 0
