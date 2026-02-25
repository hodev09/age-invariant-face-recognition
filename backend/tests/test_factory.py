"""Tests for the AI provider factory."""

import os
from unittest.mock import patch

from ai_providers.factory import get_provider
from ai_providers.insightface_provider import LocalInsightFaceProvider
from ai_providers.openai_provider import OpenAIProvider


class TestGetProvider:
    """Unit tests for get_provider() environment-based selection."""

    def test_returns_local_provider_when_use_openai_not_set(self):
        env = {k: v for k, v in os.environ.items() if k not in ("USE_OPENAI", "OPENAI_API_KEY")}
        with patch.dict(os.environ, env, clear=True):
            provider = get_provider()
            assert isinstance(provider, LocalInsightFaceProvider)

    def test_returns_local_provider_when_use_openai_false(self):
        env = {k: v for k, v in os.environ.items() if k not in ("USE_OPENAI", "OPENAI_API_KEY")}
        env["USE_OPENAI"] = "false"
        with patch.dict(os.environ, env, clear=True):
            provider = get_provider()
            assert isinstance(provider, LocalInsightFaceProvider)

    def test_returns_openai_provider_when_enabled_and_key_set(self):
        env = {k: v for k, v in os.environ.items() if k not in ("USE_OPENAI", "OPENAI_API_KEY")}
        env["USE_OPENAI"] = "true"
        env["OPENAI_API_KEY"] = "sk-test-key"
        with patch.dict(os.environ, env, clear=True):
            provider = get_provider()
            assert isinstance(provider, OpenAIProvider)

    def test_falls_back_to_local_when_openai_enabled_but_no_key(self):
        env = {k: v for k, v in os.environ.items() if k not in ("USE_OPENAI", "OPENAI_API_KEY")}
        env["USE_OPENAI"] = "true"
        with patch.dict(os.environ, env, clear=True):
            provider = get_provider()
            assert isinstance(provider, LocalInsightFaceProvider)

    def test_use_openai_case_insensitive(self):
        env = {k: v for k, v in os.environ.items() if k not in ("USE_OPENAI", "OPENAI_API_KEY")}
        env["USE_OPENAI"] = "True"
        env["OPENAI_API_KEY"] = "sk-test-key"
        with patch.dict(os.environ, env, clear=True):
            provider = get_provider()
            assert isinstance(provider, OpenAIProvider)

    def test_returns_local_provider_when_use_openai_empty_string(self):
        env = {k: v for k, v in os.environ.items() if k not in ("USE_OPENAI", "OPENAI_API_KEY")}
        env["USE_OPENAI"] = ""
        with patch.dict(os.environ, env, clear=True):
            provider = get_provider()
            assert isinstance(provider, LocalInsightFaceProvider)
