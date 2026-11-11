"""The paid-API budget: every call is booked, and the cap is enforced."""

import pytest

from bodynav.llm.backends import (BackendError, Completion, LLMBackend, MeteredBackend,
                                OpenAIBackend, ScriptedBackend, SpendLedger, load_backend)


class FixedCost(LLMBackend):
    name, model = "fake", "fake-model"

    def complete(self, system, user, *, max_tokens=1200):
        return Completion(text="{}", model=self.model, backend=self.name,
                          prompt_tokens=1000, completion_tokens=500)


def test_ledger_books_each_call(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=1.0)
    backend = MeteredBackend(FixedCost(), ledger, price_in=2.0, price_out=10.0)
    backend.complete("s", "u")
    backend.complete("s", "u")
    # (1000 * 2 + 500 * 10) / 1e6 = $0.007 per call
    assert ledger.spent_usd == pytest.approx(0.014)
    assert ledger._read()["by_model"]["fake-model"]["calls"] == 2


def test_cap_refuses_the_next_call(tmp_path):
    ledger = SpendLedger(tmp_path / "spend.json", cap_usd=0.01)
    backend = MeteredBackend(FixedCost(), ledger, price_in=2.0, price_out=10.0)
    backend.complete("s", "u")                 # $0.007, under the cap
    backend.complete("s", "u")                 # $0.014, crosses it
    with pytest.raises(BackendError, match="budget exhausted"):
        backend.complete("s", "u")


def test_ledger_is_shared_across_instances(tmp_path):
    path = tmp_path / "spend.json"
    MeteredBackend(FixedCost(), SpendLedger(path, 1.0), 2.0, 10.0).complete("s", "u")
    assert SpendLedger(path, 1.0).spent_usd == pytest.approx(0.007)


def _config(tmp_path, body: str):
    p = tmp_path / "llm.yaml"
    p.write_text("budget: {cap_usd: 1.0, ledger: spend.json}\nbackends:\n" + body)
    return p


def test_paid_backend_without_price_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    cfg = _config(tmp_path, "  m: {provider: openai, model: gpt-x}\n")
    with pytest.raises(BackendError, match="no 'price'"):
        load_backend("m", cfg)


def test_paid_backend_is_metered(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    cfg = _config(tmp_path, "  m: {provider: openai, model: gpt-x, price: [1, 2]}\n")
    assert isinstance(load_backend("m", cfg), MeteredBackend)


def test_local_vllm_backend_is_free_and_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = _config(tmp_path, "  q: {provider: vllm, model: some/model, "
                            "base_url: 'http://localhost:9/v1'}\n")
    backend = load_backend("q", cfg)
    assert isinstance(backend, OpenAIBackend) and backend.name == "vllm"


def test_missing_anthropic_key_falls_back_to_scripted(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg = _config(tmp_path, "  c: {provider: anthropic, model: claude-x, price: [1, 2]}\n")
    assert isinstance(load_backend("c", cfg), ScriptedBackend)


def test_parallel_bookings_are_not_lost(tmp_path):
    import multiprocessing as mp
    path = tmp_path / "spend.json"
    procs = [mp.Process(target=_book_many, args=(path, 25)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    data = SpendLedger(path, 100.0)._read()
    assert data["by_model"]["fake-model"]["calls"] == 100
    assert data["total_usd"] == pytest.approx(100 * 0.007)


def _book_many(path, n):
    ledger = SpendLedger(path, 100.0)
    for _ in range(n):
        ledger.record("fake-model", 1000, 500, 2.0, 10.0)
