import unittest
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from gemini_object_visibility import ObjectVisibilityChecker
from google.genai.errors import APIError

class TestObjectVisibilityChecker(unittest.TestCase):
    """
    Tests for the ObjectVisibilityChecker class.
    """

    @patch('gemini_object_visibility.genai.GenerativeModel')
    def setUp(self, mock_generative_model):
        """
        Set up a mock Gemini model and a dummy image for testing.
        """
        self.mock_model = mock_generative_model.return_value
        self.checker = ObjectVisibilityChecker()
        self.test_image = Image.new('RGB', (100, 100), color='red')

    def test_initialization(self):
        """
        Test that the class initializes correctly.
        """
        self.assertIsNotNone(self.checker)

    @patch('gemini_object_visibility.ObjectVisibilityChecker._pil_to_bytes')
    def test_is_fully_visible_yes(self, mock_pil_to_bytes):
        """
        Test the is_fully_visible method for a 'yes' response.
        """
        mock_pil_to_bytes.return_value = b'dummy_image_bytes'
        self.mock_model.generate_content.return_value.text = 'yes'
        self.assertTrue(self.checker.is_fully_visible(self.test_image))

    @patch('gemini_object_visibility.ObjectVisibilityChecker._pil_to_bytes')
    def test_is_fully_visible_no(self, mock_pil_to_bytes):
        """
        Test the is_fully_visible method for a 'no' response.
        """
        mock_pil_to_bytes.return_value = b'dummy_image_bytes'
        self.mock_model.generate_content.return_value.text = 'no'
        self.assertFalse(self.checker.is_fully_visible(self.test_image))

    @patch('gemini_object_visibility.ObjectVisibilityChecker._pil_to_bytes')
    def test_is_fully_visible_api_error(self, mock_pil_to_bytes):
        """
        Test that the is_fully_visible method returns False on an APIError.
        """
        mock_pil_to_bytes.return_value = b'dummy_image_bytes'
        self.mock_model.generate_content.side_effect = APIError(
            "Test API Error", response_json={'error': {'message': 'Test API Error'}}
        )
        self.assertFalse(self.checker.is_fully_visible(self.test_image))

    @patch('gemini_object_visibility.ObjectVisibilityChecker._pil_to_bytes')
    def test_is_fully_visible_unexpected_error(self, mock_pil_to_bytes):
        """
        Test that the is_fully_visible method returns False on an unexpected error.
        """
        mock_pil_to_bytes.return_value = b'dummy_image_bytes'
        self.mock_model.generate_content.side_effect = Exception("Test Unexpected Error")
        self.assertFalse(self.checker.is_fully_visible(self.test_image))

if __name__ == '__main__':
    unittest.main()
