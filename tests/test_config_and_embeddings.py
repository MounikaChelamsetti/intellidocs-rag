import pytest

from src.config import Settings, load_settings
from src.embeddings import CachedEmbedder, GeminiEmbedder
from src.errors import ConfigError, GeminiAuthError, GeminiRateLimitError, GeminiServiceError
from src.gemini_utils import call_with_retries, translate_error
from src.llm import GeminiLLM
from src.offline import HashingEmbedder
from google.genai import errors as genai_errors


# ------------------------------------------------------------------ configuration
def test_api_key_from_env_and_never_in_repr():
    s = load_settings(env={"GEMINI_API_KEY": "secret-123"}, load_dotenv_file=False)
    assert s.require_api_key() == "secret-123" and "secret-123" not in repr(s)


def test_api_key_from_streamlit_secrets_fallback():
    s = load_settings(env={}, secrets={"GEMINI_API_KEY": "from-secrets"}, load_dotenv_file=False)
    assert s.gemini_api_key == "from-secrets"


def test_missing_key_raises_config_error_only_when_required():
    s = load_settings(env={}, load_dotenv_file=False)
    assert s.gemini_api_key is None
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        s.require_api_key()


@pytest.mark.parametrize("env", [{"CHUNK_SIZE_WORDS": "abc"}, {"CHUNK_SIZE_WORDS": "50", "CHUNK_OVERLAP_WORDS": "50"},
                                 {"TOP_K": "0"}, {"MIN_RELEVANCE": "7"}])
def test_invalid_settings_rejected(env):
    with pytest.raises(ConfigError):
        load_settings(env=env, load_dotenv_file=False)


def test_defaults_match_spec():
    s = Settings()
    assert (s.chunk_size_words, s.chunk_overlap_words, s.top_k) == (400, 60, 5)


# ------------------------------------------------------------------ error translation / retries
def test_error_translation():
    assert isinstance(translate_error(genai_errors.ClientError(429, {"error": {"message": "quota"}})), GeminiRateLimitError)
    assert isinstance(translate_error(genai_errors.ClientError(403, {"error": {"message": "denied"}})), GeminiAuthError)
    assert isinstance(translate_error(genai_errors.ClientError(400, {"error": {"message": "API key not valid"}})), GeminiAuthError)
    assert isinstance(translate_error(genai_errors.ServerError(503, {"error": {"message": "x"}})), GeminiServiceError)
    assert isinstance(translate_error(ConnectionError("dns")), GeminiServiceError)


def test_retries_then_success_uses_backoff():
    sleeps, attempts = [], {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise genai_errors.ClientError(429, {"error": {"message": "slow down"}})
        return "ok"

    assert call_with_retries(flaky, 4, 1.0, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_rate_limit_exhausts_retries_and_auth_is_not_retried():
    sleeps = []
    with pytest.raises(GeminiRateLimitError):
        call_with_retries(lambda: (_ for _ in ()).throw(genai_errors.ClientError(429, {})), 2, 0.01, sleep=sleeps.append)
    assert len(sleeps) == 2
    sleeps.clear()
    with pytest.raises(GeminiAuthError):
        call_with_retries(lambda: (_ for _ in ()).throw(genai_errors.ClientError(401, {})), 3, 0.01, sleep=sleeps.append)
    assert sleeps == []


# ------------------------------------------------------------------ Gemini wrappers with a fake client
class FakeEmb:
    def __init__(self, values):
        self.values = values


class FakeModels:
    def __init__(self):
        self.embed_calls = []

    def embed_content(self, model, contents, config):
        self.embed_calls.append((model, list(contents), config.task_type, config.output_dimensionality))
        return type("R", (), {"embeddings": [FakeEmb([3.0, 4.0]) for _ in contents]})()

    def generate_content(self, model, contents, config):
        return type("R", (), {"text": "  hello [1]  "})()


class FakeClient:
    def __init__(self):
        self.models = FakeModels()


def test_gemini_embedder_batches_normalises_and_sets_task_types():
    client = FakeClient()
    emb = GeminiEmbedder(Settings(embed_batch_size=2, embedding_dim=2), client=client)
    vecs = emb.embed_documents(["a", "b", "c"])
    assert [len(c[1]) for c in client.models.embed_calls] == [2, 1]
    assert vecs[0] == pytest.approx([0.6, 0.8])
    assert client.models.embed_calls[0][2] == "RETRIEVAL_DOCUMENT"
    emb.embed_query("q")
    assert client.models.embed_calls[-1][2] == "RETRIEVAL_QUERY" and client.models.embed_calls[-1][3] == 2


def test_gemini_llm_returns_stripped_text_and_rejects_empty():
    client = FakeClient()
    assert GeminiLLM(Settings(), client=client).generate("sys", "user") == "hello [1]"
    client.models.generate_content = lambda **kw: type("R", (), {"text": None})()
    with pytest.raises(GeminiServiceError, match="no answer"):
        GeminiLLM(Settings(max_retries=0), client=client).generate("sys", "user")


def test_cached_embedder_avoids_repeat_calls(tmp_path):
    inner = HashingEmbedder()
    count = {"n": 0}
    orig = inner.embed_documents
    inner.embed_documents = lambda t: count.__setitem__("n", count["n"] + len(t)) or orig(t)
    path = tmp_path / "cache.json"
    c1 = CachedEmbedder(inner, path)
    c1.embed_documents(["alpha beta", "gamma delta"])
    c2 = CachedEmbedder(inner, path)  # new instance reads the disk cache
    c2.embed_documents(["alpha beta", "gamma delta", "new text"])
    assert count["n"] == 3 and c2.hits == 2 and c2.misses == 1
