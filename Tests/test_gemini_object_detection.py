import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np

from gemini_object_detection import GeminiObjectDetection

class TestGeminiObjectDetection(unittest.TestCase):

    def setUp(self):
        self.api_key = "test_api_key"
        self.model_name = "gemini-1.5-flash"
        self.detector = GeminiObjectDetection(api_key=self.api_key, model_name=self.model_name)

    def test_normalize_to_absolute_coordinates(self):
        bbox_normalized = [100, 200, 300, 400]
        img_width = 1000
        img_height = 800
        expected_bbox_absolute = [100, 160, 300, 320]
        
        absolute_coords = self.detector.normalize_to_absolute_coordinates(bbox_normalized, img_width, img_height)
        
        self.assertEqual(absolute_coords, expected_bbox_absolute)

    @patch('gemini_object_detection.ImageSource')
    def test_detect_objects(self, mock_image_source):
        # Create a mock image
        mock_image = Image.fromarray(np.uint8(np.zeros((100, 100, 3))))
        
        # Configure the mock ImageSource
        mock_image_source.return_value.get_image.return_value.data = mock_image
        
        # Mock the Gemini API client
        self.detector.client = MagicMock()
        
        # Define the mock API response
        mock_response = MagicMock()
        mock_response.text = '{"detected_objects": [{"label": "cat", "bounding_box": [100, 200, 300, 400]}]}'
        self.detector.client.models.generate_content.return_value = mock_response
        
        # Call the method to be tested
        result = self.detector.detect_objects("dummy_path.jpg", save_annotated_image=False)
        
        # Assertions
        self.assertIn("detected_objects", result)
        self.assertEqual(len(result["detected_objects"]), 1)
        self.assertEqual(result["detected_objects"][0]["label"], "cat")
        
        # Check if absolute coordinates are calculated correctly
        expected_absolute_coords = self.detector.normalize_to_absolute_coordinates([100, 200, 300, 400], 100, 100)
        self.assertEqual(result["detected_objects"][0]["bounding_box"], expected_absolute_coords)

if __name__ == '__main__':
    unittest.main()
