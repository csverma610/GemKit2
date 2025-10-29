"""
Unit tests for gemini_text_client.py

This module contains comprehensive unit tests for the GeminiClient class,
testing text generation and structured output generation (including Pydantic models).
"""

import pytest
import os
from unittest.mock import Mock, patch
from pydantic import BaseModel
from typing import List
from google.genai import types
from gemini_text_client import GeminiClient, ModelInput

# Sample Pydantic models for testing
class Recipe(BaseModel):
    """Sample Pydantic model for recipe."""
    name: str
    ingredients: List[str]
    prep_time: int


class Person(BaseModel):
    """Sample Pydantic model for person."""
    name: str
    age: int
    email: str


class TestGeminiClientInit:
    """Test cases for GeminiClient initialization."""

    def test_init_with_api_key(self):
        """Test initialization when GEMINI_API_KEY is set."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_text_client.genai.Client') as mock_client:
                client = GeminiClient()

                mock_client.assert_called_once_with(api_key='test-api-key')
                assert client.model_name == 'gemini-2.5-flash'

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_text_client.genai.Client'):
                client = GeminiClient(config=Mock(model_name='gemini-2.5-pro'))

                assert client.model_name == 'gemini-2.5-pro'

    def test_init_without_api_key_fails(self):
        """Test initialization fails when no API key and fallback fails."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError) as exc_info:
                GeminiClient()

            assert "API key must be provided" in str(exc_info.value)


class TestGenerateText:
    """Test cases for generate_text method."""

    @pytest.fixture
    def mock_client(self):
        """Create a GeminiClient with mocked API client."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_text_client.genai.Client'):
                client = GeminiClient()

                # Mock the API response
                mock_response = Mock()
                mock_response.text = "This is a generated response"
                client.client.models.generate_content = Mock(return_value=mock_response)

                yield client

    def test_generate_text_user_prompt_only(self, mock_client):
        """Test generate_text with only user prompt."""
        model_input = ModelInput(user_prompt="What is AI?")
        response = mock_client.generate_text(model_input)

        assert response == "This is a generated response"

        # Verify API call
        call_args = mock_client.client.models.generate_content.call_args
        assert call_args[1]['model'] == 'gemini-2.5-flash'
        assert call_args[1]['config'] is None

        # Check contents
        contents = call_args[1]['contents']
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "What is AI?"

    def test_generate_text_with_system_prompt(self, mock_client):
        """Test generate_text with system prompt."""
        model_input = ModelInput(
            user_prompt="Explain quantum computing",
            sys_prompt="You are a physics professor"
        )
        response = mock_client.generate_text(model_input)

        assert response == "This is a generated response"

        call_args = mock_client.client.models.generate_content.call_args

        # Verify system instruction in config
        config = call_args[1]['config']
        assert config is not None
        assert isinstance(config, types.GenerateContentConfig)
        assert config.system_instruction == "You are a physics professor"

    def test_generate_text_with_assistant_prompt(self, mock_client):
        """Test generate_text with assistant prompt (few-shot)."""
        model_input = ModelInput(
            user_prompt="What is deep learning?",
            assist_prompt="I am an expert in neural networks and AI."
        )
        response = mock_client.generate_text(model_input)

        assert response == "This is a generated response"

        call_args = mock_client.client.models.generate_content.call_args
        contents = call_args[1]['contents']

        # Should have 2 messages: assistant (model) then user
        assert len(contents) == 2
        assert contents[0].role == "model"
        assert contents[0].parts[0].text == "I am an expert in neural networks and AI."
        assert contents[1].role == "user"
        assert contents[1].parts[0].text == "What is deep learning?"

    def test_generate_text_all_prompts(self, mock_client):
        """Test generate_text with all prompt types."""
        model_input = ModelInput(
            user_prompt="How does backpropagation work?",
            assist_prompt="I specialize in training neural networks",
            sys_prompt="You are a machine learning educator"
        )
        response = mock_client.generate_text(model_input)

        assert response == "This is a generated response"

        call_args = mock_client.client.models.generate_content.call_args

        # Config should have system instruction
        assert call_args[1]['config'] is not None

        # Contents should have both model and user
        contents = call_args[1]['contents']
        assert len(contents) == 2
        assert contents[0].role == "model"
        assert contents[1].role == "user"

    def test_generate_text_empty_optional_params(self, mock_client):
        """Test generate_text with empty optional parameters."""
        model_input = ModelInput(
            user_prompt="Test query",
            assist_prompt="",
            sys_prompt=""
        )
        response = mock_client.generate_text(model_input)

        assert response == "This is a generated response"

        call_args = mock_client.client.models.generate_content.call_args

        # Config should be None for empty system prompt
        assert call_args[1]['config'] is None

        # Only user message in contents
        contents = call_args[1]['contents']
        assert len(contents) == 1
        assert contents[0].role == "user"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
