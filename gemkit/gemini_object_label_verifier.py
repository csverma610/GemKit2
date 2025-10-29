import argparse
import io
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VerifierConfig:
    """Configuration for the Gemini Object Label Verifier."""
    model_name: str = "gemini-2.5-flash"


class GeminiObjectLabelVerifier:
    """
    Verifies object detection labels by querying sub-images with the Gemini API.
    """

    def __init__(self, config: Optional[VerifierConfig] = None):
        """
        Initialize the Gemini client and set up the model.

        Args:
            config: Optional configuration object. If None, default settings are used.
        """
        self.config = config or VerifierConfig()
        self.client = self._create_client()

    def _create_client(self) -> Optional[genai.Client]:
        """Initialize and return the Gemini client."""
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set.")
            raise ValueError("API key is missing.")
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            return None

    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Loads an image from the specified path."""
        try:
            image = Image.open(image_path).convert("RGB")
            logger.info(f"Successfully loaded image from {image_path}")
            return image
        except FileNotFoundError:
            logger.error(f"Image file not found at {image_path}")
            return None
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def _load_detection_data(self, json_path: str) -> Optional[Dict[str, Any]]:
        """Loads object detection data from a JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            if 'detected_objects' not in data:
                logger.warning(f"JSON file {json_path} is missing 'detected_objects' key.")
                return None
            logger.info(f"Successfully loaded detection data from {json_path}")
            return data
        except FileNotFoundError:
            logger.error(f"JSON file not found at {json_path}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {json_path}")
            return None

    def _pil_to_bytes(self, image: Image.Image) -> bytes:
        """Converts a PIL image to bytes."""
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

    def verify_single_object(self, sub_image: Image.Image, label: str) -> Tuple[str, bool]:
        """
        Queries the Gemini model to verify if the label exists in the sub-image.

        Returns:
            A tuple containing the model's text response and a boolean indicating
            if the verification was successful.
        """
        if not self.client:
            raise ConnectionError("Gemini client is not initialized.")

        prompt = f"Is there a {label} in this image? Please answer with only 'yes' or 'no'."
        image_bytes = self._pil_to_bytes(sub_image)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/png')

        try:
            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=[prompt, image_part]
            )
            model_response = response.text.strip().lower()
            is_verified = 'yes' in model_response
            return model_response, is_verified
        except APIError as e:
            logger.error(f"Gemini API error during verification for label '{label}': {e}")
            return f"API Error: {e}", False
        except Exception as e:
            logger.error(f"An unexpected error occurred during verification for label '{label}': {e}")
            return f"Unexpected Error: {e}", False

    def verify_labels(
        self,
        image_path: str,
        detection_json_path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Crops objects from an image based on detection data and verifies each label.
        """
        original_image = self._load_image(image_path)
        if not original_image:
            return None

        detection_data = self._load_detection_data(detection_json_path)
        if not detection_data:
            return None

        verification_results = []
        detected_objects = detection_data.get('detected_objects', [])

        if not detected_objects:
            logger.info("No objects to verify.")
            return {
                "original_image": image_path,
                "detection_json": detection_json_path,
                "verification_results": []
            }

        for obj in detected_objects:
            label = obj.get('label')
            bbox = obj.get('bounding_box')

            if not label or not bbox or len(bbox) != 4:
                logger.warning(f"Skipping invalid object entry: {obj}")
                continue

            logger.info(f"Verifying label '{label}' with bounding box {bbox}...")
            sub_image = original_image.crop(bbox)

            model_response, is_verified = self.verify_single_object(sub_image, label)

            verification_results.append({
                "label": label,
                "bounding_box": bbox,
                "model_response": model_response,
                "is_verified": is_verified
            })

        return {
            "original_image": image_path,
            "detection_json": detection_json_path,
            "verification_results": verification_results
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify object detection labels using sub-images and the Gemini API."
    )
    parser.add_argument(
        "-i", "--image_path",
        type=str,
        required=True,
        help="The local file path to the original image."
    )
    parser.add_argument(
        "-j", "--detection_json_path",
        type=str,
        required=True,
        help="Path to the JSON file containing detection results from gemini_object_detection.py."
    )

    args = parser.parse_args()

    verifier = GeminiObjectLabelVerifier()
    results = verifier.verify_labels(
        image_path=args.image_path,
        detection_json_path=args.detection_json_path
    )

    if results:
        print("\n--- Gemini Object Label Verification Result ---")
        print(json.dumps(results, indent=2))
        print("---------------------------------------------")
