"""Chat and embedding models behind LangChain, so a model is swapped by editing configs/experiment.yaml.

Hugging Face's router speaks the OpenAI API, so it goes through LangChain's OpenAI integration with a
different base URL. Claude deployed on Microsoft Foundry speaks the Anthropic API, so it goes through
LangChain's Anthropic integration with the deployment's base URL. Any other LangChain provider name in the
configuration works without changing the calling code, with that provider's own key in the environment.

The Hugging Face token comes from `HF_TOKEN` or the local Hugging Face login file. Foundry's key and base URL
come from `ANTHROPIC_FOUNDRY_API_KEY` and `ANTHROPIC_FOUNDRY_BASE_URL`. None of them is written to a file or
printed.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_huggingface import HuggingFaceEndpointEmbeddings

HF_ROUTER_URL = "https://router.huggingface.co/v1"
HF_ROUTER = "huggingface_router"
HF_INFERENCE = "huggingface_inference"
ANTHROPIC_FOUNDRY = "anthropic_foundry"
FOUNDRY_SETTINGS = ("ANTHROPIC_FOUNDRY_BASE_URL", "ANTHROPIC_FOUNDRY_API_KEY")


@dataclass(frozen=True)
class ModelSpec:
    """One chat model, with its price in US dollars per million tokens for cost estimates."""

    provider: str
    model: str
    input_price: float = 0.0
    output_price: float = 0.0
    max_tokens: int = 2000
    # None leaves the provider's default, for models that refuse a temperature.
    temperature: float | None = 0.0
    json_mode: bool = True

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return the estimated cost in US dollars of a number of tokens."""
        return (input_tokens * self.input_price + output_tokens * self.output_price) / 1_000_000


def hf_token(login_file: Path | None = None) -> str:
    """Return `HF_TOKEN` if set, otherwise the local Hugging Face login. Raises if there is neither."""
    token = os.environ.get("HF_TOKEN")
    path = login_file or Path.home() / ".cache" / "huggingface" / "token"
    if not token and path.exists():
        token = path.read_text().strip()
    if not token:
        raise RuntimeError("no Hugging Face token: log in with `hf auth login` or set HF_TOKEN")
    return token


def foundry_settings() -> dict[str, str]:
    """Return the base URL and key of a Claude deployment on Microsoft Foundry. Raises if either is unset."""
    missing = [name for name in FOUNDRY_SETTINGS if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"set {' and '.join(missing)} to use a Claude deployment on Microsoft Foundry")
    return {"base_url": os.environ[FOUNDRY_SETTINGS[0]], "api_key": os.environ[FOUNDRY_SETTINGS[1]]}


def make_chat_model(spec: ModelSpec) -> BaseChatModel:
    """Return a LangChain chat model, asking for JSON where the model's provider supports it."""
    settings: dict[str, Any] = {"max_tokens": spec.max_tokens, "timeout": 180, "max_retries": 0}
    if spec.temperature is not None:
        settings["temperature"] = spec.temperature
    if spec.provider == HF_ROUTER:
        settings.update(model_provider="openai", base_url=HF_ROUTER_URL, api_key=hf_token())
    elif spec.provider == ANTHROPIC_FOUNDRY:
        if spec.json_mode:
            raise ValueError("the Anthropic API has no JSON mode: set json_mode to false for this model")
        settings.update(model_provider="anthropic", **foundry_settings())
    else:
        settings.update(model_provider=spec.provider)
    model = init_chat_model(spec.model, **settings)
    return model.bind(response_format={"type": "json_object"}) if spec.json_mode else model


def make_embeddings(provider: str, model: str) -> Embeddings:
    """Return a LangChain embeddings model. Hugging Face's hosted inference is the only provider set up so
    far."""
    if provider != HF_INFERENCE:
        raise ValueError(f"unknown embeddings provider: {provider}")
    return HuggingFaceEndpointEmbeddings(
        model=model, provider="hf-inference", huggingfacehub_api_token=hf_token()
    )


def spec_from_config(config: Mapping[str, Any]) -> ModelSpec:
    """Build a ModelSpec from one entry of the configuration's `models` section."""
    return ModelSpec(**config)


def batches(items: Sequence[str], size: int) -> list[Sequence[str]]:
    """Split items into consecutive batches of at most `size`."""
    return [items[start : start + size] for start in range(0, len(items), size)]
