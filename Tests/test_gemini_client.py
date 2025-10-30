import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import os

from pydantic import BaseModel

from gemkit.gemini_client import GeminiClient, ModelConfig, ModelInput


class TestGeminiClient(unittest.TestCase):
    """
    Unit tests for the GeminiClient class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        This involves patching the `genai.Client` to avoid actual API calls.
        """
        self.mock_client = MagicMock()
        self.patcher = patch('gemkit.gemini_client.genai.Client', return_value=self.mock_client)
        self.mock_genai_client = self.patcher.start()
        os.environ['GEMINI_API_KEY'] = 'test_api_key'

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        self.patcher.stop()
        os.environ.pop('GEMINI_API_KEY', None)

    def test_initialization_success(self):
        """
        Test successful initialization of GeminiClient.
        """
        client = GeminiClient()
        self.assertIsNotNone(client)
        self.mock_genai_client.assert_called_once_with(api_key='test_api_key')

    def test_initialization_no_api_key(self):
        """
        Test initialization failure when no API key is provided.
        """
        del os.environ['GEMINI_API_KEY']
        with self.assertRaises(ValueError):
            GeminiClient()

    def test_generate_content_text(self):
        """
        Test basic text generation.
        """
        # Configure the mock to return a response with a 'text' attribute
        mock_response = MagicMock()
        mock_response.text = "Hello, world!"
        self.mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        model_input = ModelInput(user_prompt="Say hello")
        result = client.generate_content(model_input)

        self.assertEqual(result, "Hello, world!")
        self.mock_client.models.generate_content.assert_called_once()

    def test_generate_content_streaming(self):
        """
        Test streaming text generation.
        """
        # Configure the mock to return a stream of chunks
        mock_chunk1 = MagicMock()
        mock_chunk1.text = "Hello, "
        mock_chunk2 = MagicMock()
        mock_chunk2.text = "world!"
        self.mock_client.models.generate_content_stream.return_value = [mock_chunk1, mock_chunk2]

        client = GeminiClient()
        model_input = ModelInput(user_prompt="Say hello")
        stream = client.generate_content(model_input, stream=True)

        result = "".join(chunk for chunk in stream)
        self.assertEqual(result, "Hello, world!")
        self.mock_client.models.generate_content_stream.assert_called_once()

    def test_generate_json_success(self):
        """
        Test successful JSON generation with a Pydantic schema.
        """
        class MySchema(BaseModel):
            message: str

        # Configure the mock to return a response with a 'parsed' attribute
        mock_response = MagicMock()
        mock_response.parsed = MySchema(message="Hello, world!")
        self.mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        model_input = ModelInput(user_prompt="Say hello", response_schema=MySchema)
        result = client.generate_content(model_input)

        self.assertIsInstance(result, MySchema)
        self.assertEqual(result.message, "Hello, world!")

    def test_image_handling(self):
        """
        Test that the client correctly handles image inputs.
        """
        # Create a dummy image file
        image_path = Path("test_image.png")
        with open(image_path, "wb") as f:
            f.write(b"dummy image data")

        mock_response = MagicMock()
        mock_response.text = "Image processed"
        self.mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        model_input = ModelInput(user_prompt="What is in this image?", images=[image_path])
        result = client.generate_content(model_input)

        self.assertEqual(result, "Image processed")
        
        # Clean up the dummy image file
        os.remove(image_path)


if __name__ == '__main__':
    unittest.main()