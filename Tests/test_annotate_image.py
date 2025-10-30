import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import os
import shutil

from gemkit.annotate_image import AnnotateImage, BoundingBox

class TestAnnotateImage(unittest.TestCase):
    """
    Unit tests for the AnnotateImage class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        This involves creating a dummy image and an output directory.
        """
        self.output_dir = "test_annotated_images"
        os.makedirs(self.output_dir, exist_ok=True)
        self.image = Image.new('RGB', (100, 100), color = 'red')
        self.image_path = os.path.join(self.output_dir, "test_image.png")
        self.image.save(self.image_path)

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        shutil.rmtree(self.output_dir)

    def test_initialization(self):
        """
        Test that AnnotateImage can be initialized with a path or a PIL image.
        """
        annotator_from_path = AnnotateImage(image_path=self.image_path)
        self.assertIsNotNone(annotator_from_path)

        annotator_from_image = AnnotateImage(image=self.image)
        self.assertIsNotNone(annotator_from_image)

        with self.assertRaises(ValueError):
            AnnotateImage()

    def test_draw_boxes(self):
        """
        Test drawing bounding boxes on an image.
        """
        annotator = AnnotateImage(image=self.image)
        boxes = [
            BoundingBox(x_min=10, y_min=10, x_max=30, y_max=30, label="cat"),
            {'x_min': 40, 'y_min': 40, 'x_max': 60, 'y_max': 60, 'label': 'dog'}
        ]
        
        # To test the drawing, we can check if the image has been modified.
        # A simple way is to compare the image data before and after.
        original_image_data = self.image.tobytes()
        annotator.draw_boxes(boxes)
        modified_image_data = annotator.get_image().tobytes()

        self.assertNotEqual(original_image_data, modified_image_data)

    def test_save_image(self):
        """
        Test saving the annotated image.
        """
        annotator = AnnotateImage(image=self.image)
        output_path = annotator.save()
        self.assertTrue(os.path.exists(output_path))

    def test_from_gemini_detection(self):
        """
        Test creating an AnnotateImage instance from Gemini detection results.
        """
        detection_results = {
            'detected_objects': [
                {
                    'bounding_box_absolute': [10, 10, 30, 30],
                    'label': 'cat',
                    'confidence': 0.9
                }
            ]
        }
        annotator = AnnotateImage.from_gemini_detection(self.image_path, detection_results)
        self.assertIsNotNone(annotator)

        # Check if the image was modified
        original_image = Image.open(self.image_path)
        self.assertNotEqual(original_image.tobytes(), annotator.get_image().tobytes())

if __name__ == '__main__':
    unittest.main()
