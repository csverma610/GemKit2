"""
Unit tests for gemini_client.py

This module contains comprehensive unit tests for the GeminiClient class,
testing text generation and structured output generation (including Pydantic models).
"""

import pytest
import os
from unittest.mock import Mock, patch
from pydantic import BaseModel
from typing import List
from google.genai import types
from gemini_text_client import GeminiClient

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
            with patch('gemini_client.genai.Client') as mock_client:
                client = GeminiClient()

                mock_client.assert_called_once_with(api_key='test-api-key')
                assert client.model_name == GeminiClient.DEFAULT_MODEL

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_client.genai.Client'):
                client = GeminiClient(model_name='gemini-2.5-flash-lite')

                assert client.model_name == 'gemini-2.5-flash-lite'

    def test_init_without_api_key_fallback(self):
        """Test initialization without API key attempts fallback."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('gemini_client.genai.Client') as mock_client:
                client = GeminiClient()

                mock_client.assert_called_once_with()

    def test_init_without_api_key_fails(self):
        """Test initialization fails when no API key and fallback fails."""
        with patch.dict(os.environ, {}, clear=True):
            with patch('gemini_client.genai.Client', side_effect=Exception("Auth error")):
                with pytest.raises(ValueError) as exc_info:
                    GeminiClient()

                assert "GEMINI_API_KEY" in str(exc_info.value)


class TestGenerateText:
    """Test cases for generate_text method."""

    @pytest.fixture
    def mock_client(self):
        """Create a GeminiClient with mocked API client."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_client.genai.Client'):
                client = GeminiClient()

                # Mock the API response
                mock_response = Mock()
                mock_response.text = "This is a generated response"
                client.client.models.generate_content = Mock(return_value=mock_response)

                yield client

    def test_generate_text_user_prompt_only(self, mock_client):
        """Test generate_text with only user prompt."""
        response = mock_client.generate_text(user_prompt="What is AI?")

        assert response == "This is a generated response"

        # Verify API call
        call_args = mock_client.client.models.generate_content.call_args
        assert call_args[1]['model'] == GeminiClient.DEFAULT_MODEL
        assert call_args[1]['config'] is None

        # Check contents
        contents = call_args[1]['contents']
        assert len(contents) == 1
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "What is AI?"

    def test_generate_text_with_system_prompt(self, mock_client):
        """Test generate_text with system prompt."""
        response = mock_client.generate_text(
            user_prompt="Explain quantum computing",
            sys_prompt="You are a physics professor"
        )

        assert response == "This is a generated response"

        call_args = mock_client.client.models.generate_content.call_args

        # Verify system instruction in config
        config = call_args[1]['config']
        assert config is not None
        assert isinstance(config, types.GenerateContentConfig)

    def test_generate_text_with_assistant_prompt(self, mock_client):
        """Test generate_text with assistant prompt (few-shot)."""
        response = mock_client.generate_text(
            user_prompt="What is deep learning?",
            assist_prompt="I am an expert in neural networks and AI."
        )

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
        response = mock_client.generate_text(
            user_prompt="How does backpropagation work?",
            assist_prompt="I specialize in training neural networks",
            sys_prompt="You are a machine learning educator"
        )

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
        response = mock_client.generate_text(
            user_prompt="Test query",
            assist_prompt="",
            sys_prompt=""
        )

        assert response == "This is a generated response"

        call_args = mock_client.client.models.generate_content.call_args

        # Config should be None for empty system prompt
        assert call_args[1]['config'] is None

        # Only user message in contents
        contents = call_args[1]['contents']
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_generate_text_custom_model(self):
        """Test generate_text uses custom model name."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_client.genai.Client'):
                client = GeminiClient(model_name='gemini-flash-latest')

                mock_response = Mock()
                mock_response.text = "response"
                client.client.models.generate_content = Mock(return_value=mock_response)

                client.generate_text("test")

                call_args = client.client.models.generate_content.call_args
                assert call_args[1]['model'] == 'gemini-flash-latest'


class TestGenerateStructured:
    """Test cases for generate_structured method with dict schemas and Pydantic models."""

    @pytest.fixture
    def mock_client(self):
        """Create a GeminiClient with mocked API client."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_client.genai.Client'):
                client = GeminiClient()

                # Mock the API response
                mock_response = Mock()
                mock_response.parsed = {"name": "Test", "value": 123}
                client.client.models.generate_content = Mock(return_value=mock_response)

                yield client

    def test_generate_structured_dict_schema(self, mock_client):
        """Test generate_structured with dictionary schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }

        response = mock_client.generate_structured(
            user_prompt="Generate a person",
            response_schema=schema
        )

        assert response == {"name": "Test", "value": 123}

        # Verify API call
        call_args = mock_client.client.models.generate_content.call_args
        config = call_args[1]['config']
        assert config is not None

    def test_generate_structured_pydantic_model(self, mock_client):
        """Test generate_structured with Pydantic model."""
        # Mock response with Recipe data
        mock_client.client.models.generate_content.return_value.parsed = Recipe(
            name="Pasta Carbonara",
            ingredients=["pasta", "eggs", "bacon", "parmesan"],
            prep_time=30
        )

        response = mock_client.generate_structured(
            user_prompt="Generate a pasta recipe",
            response_schema=Recipe
        )

        # Verify it's a Recipe instance
        assert isinstance(response, Recipe)
        assert response.name == "Pasta Carbonara"
        assert "pasta" in response.ingredients
        assert response.prep_time == 30

    def test_generate_structured_list_of_pydantic(self, mock_client):
        """Test generate_structured with list of Pydantic models."""
        # Mock response with list of Person objects
        mock_client.client.models.generate_content.return_value.parsed = [
            Person(name="Alice", age=30, email="alice@example.com"),
            Person(name="Bob", age=25, email="bob@example.com")
        ]

        response = mock_client.generate_structured(
            user_prompt="Generate 2 people",
            response_schema=List[Person]
        )

        # Verify it's a list of Person instances
        assert isinstance(response, list)
        assert len(response) == 2
        assert all(isinstance(p, Person) for p in response)
        assert response[0].name == "Alice"
        assert response[1].email == "bob@example.com"

    def test_generate_structured_with_system_prompt(self, mock_client):
        """Test generate_structured with system prompt."""
        schema = {"type": "object"}

        response = mock_client.generate_structured(
            user_prompt="Generate data",
            response_schema=schema,
            sys_prompt="Be precise and accurate"
        )

        assert response == {"name": "Test", "value": 123}

        call_args = mock_client.client.models.generate_content.call_args
        config = call_args[1]['config']
        assert config is not None

    def test_generate_structured_with_assistant_prompt(self, mock_client):
        """Test generate_structured with assistant prompt."""
        schema = {"type": "array"}

        response = mock_client.generate_structured(
            user_prompt="List programming languages",
            response_schema=schema,
            assist_prompt="I understand you need a JSON array"
        )

        call_args = mock_client.client.models.generate_content.call_args
        contents = call_args[1]['contents']

        # Should have both model and user messages
        assert len(contents) == 2
        assert contents[0].role == "model"
        assert contents[0].parts[0].text == "I understand you need a JSON array"
        assert contents[1].role == "user"

    def test_generate_structured_all_params_pydantic(self, mock_client):
        """Test generate_structured with all parameters and Pydantic model."""
        mock_client.client.models.generate_content.return_value.parsed = Person(
            name="Charlie",
            age=35,
            email="charlie@test.com"
        )

        response = mock_client.generate_structured(
            user_prompt="Create a user profile",
            response_schema=Person,
            assist_prompt="I will create structured user data",
            sys_prompt="Output valid Person objects"
        )

        # Verify Person instance
        assert isinstance(response, Person)
        assert response.name == "Charlie"
        assert response.age == 35

        # Verify API call structure
        call_args = mock_client.client.models.generate_content.call_args
        assert call_args[1]['config'] is not None
        assert len(call_args[1]['contents']) == 2

    def test_generate_structured_array_schema(self, mock_client):
        """Test generate_structured with array schema."""
        mock_client.client.models.generate_content.return_value.parsed = [
            "Python", "JavaScript", "Rust"
        ]

        schema = {
            "type": "array",
            "items": {"type": "string"}
        }

        response = mock_client.generate_structured(
            user_prompt="List 3 programming languages",
            response_schema=schema
        )

        assert isinstance(response, list)
        assert len(response) == 3
        assert "Python" in response

    def test_generate_structured_custom_model(self):
        """Test generate_structured uses custom model name."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_client.genai.Client'):
                client = GeminiClient(model_name='gemini-2.5-flash-lite')

                mock_response = Mock()
                mock_response.parsed = {}
                client.client.models.generate_content = Mock(return_value=mock_response)

                client.generate_structured("test", {"type": "object"})

                call_args = client.client.models.generate_content.call_args
                assert call_args[1]['model'] == 'gemini-2.5-flash-lite'


class TestIntegration:
    """Integration tests for GeminiClient."""

    def test_multiple_text_generations(self):
        """Test multiple generate_text calls."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_client.genai.Client'):
                client = GeminiClient()

                responses = [Mock(text=f"response{i}") for i in range(3)]
                client.client.models.generate_content = Mock(side_effect=responses)

                result1 = client.generate_text("first")
                result2 = client.generate_text("second")
                result3 = client.generate_text("third")

                assert result1 == "response0"
                assert result2 == "response1"
                assert result3 == "response2"
                assert client.client.models.generate_content.call_count == 3

    def test_mixed_text_and_structured_calls(self):
        """Test mixing generate_text and generate_structured calls."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
            with patch('gemini_client.genai.Client'):
                client = GeminiClient()

                text_response = Mock(text="text response")
                structured_response = Mock(parsed={"key": "value"})

                client.client.models.generate_content = Mock(
                    side_effect=[text_response, structured_response]
                )

                result1 = client.generate_text("generate text")
                result2 = client.generate_structured("generate json", {"type": "object"})

                assert result1 == "text response"
                assert result2 == {"key": "value"}
                assert client.client.models.generate_content.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
