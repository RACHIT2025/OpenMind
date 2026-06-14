"""
Unit tests for the API server.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Note: These tests require httpx for async testing
# pip install httpx


class TestAPIModels:
    """Test API Pydantic models."""

    def test_chat_completion_request(self):
        from inference.api_server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            temperature=0.5,
            max_tokens=100,
        )
        assert req.model == "openmind-125m"
        assert len(req.messages) == 1
        assert req.temperature == 0.5

    def test_chat_completion_request_defaults(self):
        from inference.api_server import ChatCompletionRequest, ChatMessage

        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hi")]
        )
        assert req.temperature == 0.7
        assert req.top_p == 0.9
        assert req.max_tokens == 512
        assert req.stream is False

    def test_chat_message_defaults(self):
        from inference.api_server import ChatMessage

        msg = ChatMessage(content="test")
        assert msg.role == "user"
        assert msg.content == "test"


class TestAPIEndpoints:
    """Test API endpoints (requires running server or mocked)."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        try:
            from httpx import AsyncClient, ASGITransport
            from inference.api_server import app
            return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
        except ImportError:
            pytest.skip("httpx not installed")


    @pytest.mark.asyncio
    async def test_health_endpoint(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_models_endpoint(self, client):
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

    @pytest.mark.asyncio
    async def test_chat_completions_no_model(self, client):
        response = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hello"}]
        })
        # Should return 503 when no model is loaded
        assert response.status_code == 503


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
