
import pytest
import os
from unittest.mock import Mock, patch

import google.generativeai as genai
from google.generativeai.errors import APIError

from gemini_text_client import GeminiClient, ModelInput, ModelConfig

@pytest.fixture
def mock_gemini_client():
    """Fixture to create a GeminiClient with a mocked genai.Client."""
    with patch.dict(os.environ, {'GEMINI_API_KEY': 'test-api-key'}):
        # Patch the genai.Client inside the module where it's used
        with patch('gemini_text_client.genai.Client') as mock_genai_client_constructor:
            # Mock the instance returned by the constructor
            mock_genai_instance = Mock()
            mock_genai_client_constructor.return_value = mock_genai_instance
            
            client = GeminiClient(config=ModelConfig())
            yield client

def test_generate_text_handles_api_error(mock_gemini_client):
    """Verify that generate_text catches, logs (implicitly), and re-raises APIError."""
    mock_gemini_client.client.models.generate_content.side_effect = APIError("An API error occurred")

    with pytest.raises(APIError, match="An API error occurred"):
        mock_gemini_client.generate_text(ModelInput(user_prompt="This prompt will trigger an API error"))

def test_generate_json_handles_api_error_and_recovers(mock_gemini_client):
    """Verify that generate_json attempts self-correction after an APIError on the first pass."""
    # Mock the sequence of responses from the API
    # 1. The first call (for structured output) raises an APIError.
    # 2. The second call (for self-correction) returns a successful text response with valid JSON.
    correction_response = Mock()
    correction_response.text = '{"status": "corrected"}'
    
    mock_gemini_client.client.models.generate_content.side_effect = [
        APIError("Initial API error during structured generation"),
        correction_response
    ]

    model_input = ModelInput(
        user_prompt="some prompt for json",
        response_schema={"type": "object", "properties": {"status": {"type": "string"}}}
    )

    # The generate_json method should catch the APIError and proceed to self-correction.
    result = mock_gemini_client.generate_json(model_input)

    # Assert that the result is from the successful self-correction call
    assert result == {"status": "corrected"}
    # Verify that the API was called twice (initial attempt + correction)
    assert mock_gemini_client.client.models.generate_content.call_count == 2
