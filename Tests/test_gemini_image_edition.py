import unittest
import argparse
from unittest.mock import patch, MagicMock, mock_open
import os
from io import BytesIO
from PIL import Image

# Add the current directory to the path to allow importing the script
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gemini_image_edition import GeminiImageEditor, parse_arguments

class TestGeminiImageEditor(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        # Create a dummy image in memory to be used by mocks
        self.dummy_image_bytes = BytesIO()
        Image.new('RGB', (10, 10)).save(self.dummy_image_bytes, format='PNG')
        self.dummy_image_bytes.seek(0)
        self.dummy_pil_image = Image.open(self.dummy_image_bytes)

        # Mock environment variable for API key
        self.patcher = patch.dict(os.environ, {"GEMINI_API_KEY": "test_api_key"})
        self.patcher.start()

    def tearDown(self):
        """Clean up after each test."""
        self.patcher.stop()

    @patch('gemini_image_edition.genai.Client')
    def test_init_success(self, mock_client):
        """Test successful initialization of GeminiImageEditor."""
        editor = GeminiImageEditor()
        self.assertIsNotNone(editor.client)
        mock_client.assert_called_once_with(api_key="test_api_key")

    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_api_key(self):
        """Test initialization failure when API key is not set."""
        with self.assertRaises(ValueError) as context:
            GeminiImageEditor()
        self.assertIn("API key must be set", str(context.exception))

    @patch('gemini_image_edition.ImageSource')
    @patch('gemini_image_edition.genai.Client')
    def test_edit_image_success(self, mock_client, mock_image_source):
        """Test the main success path of the edit_image method."""
        # --- Mock API Response ---
        mock_response = MagicMock()
        mock_part = MagicMock()
        mock_part.text = "This is the generated text."
        # Create mock inline data with the dummy image bytes
        mock_inline_data = MagicMock()
        mock_inline_data.data = self.dummy_image_bytes.getvalue()
        mock_part.inline_data = mock_inline_data
        
        # The response should have two parts: one text, one image
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [
            MagicMock(text="Generated text", inline_data=None),
            MagicMock(text=None, inline_data=mock_inline_data)
        ]
        
        # Configure the mock client to return the mock response
        mock_genai_instance = mock_client.return_value
        mock_genai_instance.generate_content.return_value = mock_response

        # --- Mock ImageSource ---
        mock_image_source_instance = mock_image_source.return_value
        mock_image_source_instance.get_image.return_value = MagicMock(data=self.dummy_pil_image)

        # --- Test Execution ---
        editor = GeminiImageEditor()
        
        # Mock the open function to avoid actual file writing
        with patch('builtins.open', mock_open()) as mocked_file:
            result = editor.edit_image(
                source="dummy_path.png",
                prompt="test prompt",
                output_filename="output.png"
            )

        # --- Assertions ---
        self.assertIsNotNone(result)
        self.assertEqual(result['text'], "Generated text")
        self.assertEqual(result['filepath'], "output.png")
        mock_genai_instance.generate_content.assert_called_once()
        mock_image_source.assert_called_with("dummy_path.png")

    @patch('gemini_image_edition.ImageSource')
    @patch('gemini_image_edition.genai.Client')
    def test_edit_image_source_not_found(self, mock_client, mock_image_source):
        """Test edit_image handling of FileNotFoundError from ImageSource."""
        # Configure ImageSource mock to raise an error
        mock_image_source.side_effect = FileNotFoundError("File not found")

        editor = GeminiImageEditor()
        result = editor.edit_image(
            source="nonexistent.png",
            prompt="test prompt"
        )

        self.assertIn("Error: Input source not found", result['text'])
        self.assertIsNone(result['filepath'])

    @patch('gemini_image_edition.genai.Client')
    def test_process_response_no_candidates(self, mock_client):
        """Test _process_response when the API returns no candidates."""
        editor = GeminiImageEditor()
        mock_response = MagicMock()
        mock_response.candidates = [] # No candidates

        result = editor._process_response(mock_response, "output.png")
        self.assertIn("model returned no content", result['text'])
        self.assertIsNone(result['filepath'])

class TestArguments(unittest.TestCase):

    @patch('argparse.ArgumentParser.parse_args')
    def test_parse_arguments(self, mock_parse_args):
        """Test if argument parser is set up correctly."""
        mock_parse_args.return_value = argparse.Namespace(
            image_source='test.jpg',
            prompt='a test prompt',
            output='out.png',
            model='test-model'
        )
        
        args = parse_arguments()

        self.assertEqual(args.image_source, 'test.jpg')
        self.assertEqual(args.prompt, 'a test prompt')
        self.assertEqual(args.output, 'out.png')
        self.assertEqual(args.model, 'test-model')

if __name__ == '__main__':
    unittest.main()
