"""LLM backends behind one interface (Sec. III-D, Sec. IV).

The template-wording evaluation used two hosted models, LLM-Large and LLM-Mini,
through :class:`OpenAIBackend`. The journal adds one Claude model
(:class:`AnthropicBackend`) and open-weight models served locally by
vLLM, which speaks the OpenAI protocol and so reuses
:class:`OpenAIBackend` with a ``base_url``. Everything is resolved from
``configs/llm.yaml`` so model names live in configuration, not code.

Paid calls go through :class:`MeteredBackend`, which prices every call
from the tokens the API reports and refuses to start a call once the
ledger has reached the budget in ``configs/llm.yaml``.

:class:`ScriptedBackend` is the offline counterpart. It exists so the
pipeline, its tests, and anyone reproducing the artifact can run without
credentials or spend -- **and it is never a stand-in for a reported
number**: every result records ``backend`` and ``model``, and a run that
used the scripted backend says so in its own JSON.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


class BackendError(RuntimeError):
    """The backend could not return a completion."""


@dataclass
class Completion:
    """One model response plus what it cost to get it."""

    text: str
    model: str
    backend: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMBackend:
    """Interface the agent depends on."""

    name = "base"
    model = "none"

    def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> Completion:
        raise NotImplementedError


class OpenAIBackend(LLMBackend):
    """Hosted models through the OpenAI API.

    Decoding is deterministic where the model allows it: ``temperature``
    and ``seed`` come from the config. Note that a fixed seed on a hosted
    model is a request, not a guarantee -- which is why the evaluation
    reports variation across repeats rather than claiming determinism.
    """

    name = "openai"

    def __init__(self, model: str, *, temperature: float | None = 0.0, seed: int | None = 0,
                 api_key: str | None = None, timeout: float = 120.0,
                 base_url: str | None = None, extra_body: dict | None = None,
                 name: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise BackendError(
                "the 'llm' extra is not installed: pip install -e '.[llm]'"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise BackendError("OPENAI_API_KEY is not set")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.extra_body = extra_body
        if name:
            self.name = name
        self._client = OpenAI(api_key=key, timeout=timeout, base_url=base_url)

    def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> Completion:
        t0 = time.time()
        kwargs = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            message = str(exc)
            # some models reject sampling controls; retry without them
            # rather than failing the run over a parameter we do not need
            if "temperature" in message or "seed" in message:
                kwargs.pop("temperature", None)
                kwargs.pop("seed", None)
                try:
                    resp = self._client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    raise BackendError(str(exc2)) from exc2
            else:
                raise BackendError(message) from exc

        usage = getattr(resp, "usage", None)
        return Completion(
            text=(resp.choices[0].message.content or "").strip(),
            model=self.model,
            backend=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_s=time.time() - t0,
        )


class AnthropicBackend(LLMBackend):
    """Claude models through the Anthropic Messages API.

    Current Claude models reject sampling controls (``temperature``), so
    there is no deterministic-decoding request to make; the evaluation
    reports variation across repeats instead. ``thinking`` is passed
    through as given (``"disabled"`` keeps the reply to the answer alone,
    which is what the parse and code prompts ask for); ``None`` omits it.
    """

    name = "anthropic"

    def __init__(self, model: str, *, thinking: str | None = "disabled",
                 api_key: str | None = None, timeout: float = 120.0):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise BackendError(
                "the 'llm' extra is not installed: pip install -e '.[llm]'"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise BackendError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.thinking = thinking
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key, timeout=timeout)

    def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> Completion:
        t0 = time.time()
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self.thinking:
            kwargs["thinking"] = {"type": self.thinking}
        try:
            resp = self._client.messages.create(**kwargs)
        except self._anthropic.APIStatusError as exc:
            raise BackendError(f"{exc.status_code}: {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise BackendError(f"connection failed: {exc}") from exc
        if resp.stop_reason == "refusal":
            raise BackendError(f"refusal: {resp.stop_details}")

        usage = resp.usage
        text = "".join(b.text for b in resp.content if b.type == "text")
        return Completion(
            text=text.strip(),
            model=self.model,
            backend=self.name,
            prompt_tokens=(usage.input_tokens or 0)
            + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
            completion_tokens=usage.output_tokens or 0,
            latency_s=time.time() - t0,
        )


class SpendLedger:
    """Running total of paid-API spend, kept in a JSON file.

    One ledger is shared by every run and every model, so the budget is a
    project total, not a per-run one. Prices are USD per million tokens.
    """

    def __init__(self, path: str | Path, cap_usd: float):
        self.path = Path(path)
        self.cap_usd = float(cap_usd)

    def _read(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {"total_usd": 0.0, "by_model": {}}

    @property
    def spent_usd(self) -> float:
        return float(self._read()["total_usd"])

    def check(self) -> None:
        spent = self.spent_usd
        if spent >= self.cap_usd:
            raise BackendError(f"LLM budget exhausted: ${spent:.2f} of "
                               f"${self.cap_usd:.2f} spent ({self.path})")

    def record(self, model: str, prompt_tokens: int, completion_tokens: int,
               price_in: float, price_out: float) -> float:
        import fcntl

        cost = (prompt_tokens * price_in + completion_tokens * price_out) / 1e6
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Parallel batch processes share this file: hold an exclusive lock
        # across read-modify-write so no booking is lost.
        with open(self.path.with_suffix(".lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            return self._record_locked(model, prompt_tokens, completion_tokens, cost)

    def _record_locked(self, model, prompt_tokens, completion_tokens, cost) -> float:
        data = self._read()
        m = data["by_model"].setdefault(
            model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usd": 0.0})
        m["calls"] += 1
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["usd"] = round(m["usd"] + cost, 6)
        data["total_usd"] = round(data["total_usd"] + cost, 6)
        tmp = self.path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(data, indent=1))
        tmp.replace(self.path)
        return cost


class MeteredBackend(LLMBackend):
    """Wraps a paid backend: checks the ledger before a call, books it after."""

    def __init__(self, inner: LLMBackend, ledger: SpendLedger,
                 price_in: float, price_out: float):
        self.inner = inner
        self.ledger = ledger
        self.price_in = price_in
        self.price_out = price_out
        self.name = inner.name
        self.model = inner.model

    def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> Completion:
        self.ledger.check()
        reply = self.inner.complete(system, user, max_tokens=max_tokens)
        self.ledger.record(self.model, reply.prompt_tokens, reply.completion_tokens,
                           self.price_in, self.price_out)
        return reply


@dataclass
class ScriptedBackend(LLMBackend):
    """Deterministic offline backend.

    ``responses`` maps a substring of the prompt to the text to return, so
    a test can script a malformed parse, an unresolvable reference, or code
    that reaches for the filesystem, and assert the safeguard fires.
    Anything unmatched falls back to ``default``.
    """

    responses: dict[str, str] = field(default_factory=dict)
    default: str = ""
    name: str = "scripted"
    model: str = "scripted"
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, system: str, user: str, *, max_tokens: int = 1200) -> Completion:
        self.calls.append((system, user))
        for needle, reply in self.responses.items():
            if needle in user:
                return Completion(text=reply, model=self.model, backend=self.name)
        return Completion(text=self.default, model=self.model, backend=self.name)


def load_backend(name: str | None = None, config_path: str | Path | None = None,
                 *, offline: bool = False, **kwargs) -> LLMBackend:
    """Resolve a backend from ``configs/llm.yaml``.

    ``offline`` forces the scripted backend, which is what runs when there
    is no API key -- explicitly, rather than by silently degrading.
    """
    import yaml

    root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path or root / "configs" / "llm.yaml")
    config = yaml.safe_load(config_path.read_text())
    name = name or config.get("default_backend", "llm-large")
    spec = config["backends"].get(name)
    if spec is None:
        raise BackendError(f"unknown backend {name!r}; "
                           f"have {sorted(config['backends'])}")
    provider = spec["provider"]
    key_var = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider)
    if offline or (key_var and not os.environ.get(key_var)):
        return ScriptedBackend(default="", model=f"{spec['model']} (offline)")
    if provider == "openai":
        backend = OpenAIBackend(spec["model"], temperature=spec.get("temperature", 0.0),
                                seed=spec.get("seed", 0), **kwargs)
    elif provider == "anthropic":
        backend = AnthropicBackend(spec["model"], thinking=spec.get("thinking", "disabled"),
                                   **kwargs)
    elif provider == "vllm":
        # a local server: free, no key; the OpenAI client wants a non-empty one
        return OpenAIBackend(spec["model"], temperature=spec.get("temperature", 0.0),
                             seed=spec.get("seed", 0), api_key="local",
                             base_url=spec.get("base_url", "http://localhost:8000/v1"),
                             extra_body=spec.get("extra_body"), name="vllm",
                             timeout=float(spec.get("timeout", 120.0)), **kwargs)
    else:
        raise BackendError(f"unsupported provider {provider!r}")
    budget = config.get("budget")
    if budget and "price" not in spec:
        raise BackendError(f"backend {name!r} is paid but has no 'price' in "
                           f"{config_path.name}; refusing to run it unmetered")
    if budget:
        ledger = SpendLedger(root / budget["ledger"], budget["cap_usd"])
        return MeteredBackend(backend, ledger, *spec["price"])
    return backend
