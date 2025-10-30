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
from pydantic import BaseModel, Field

from annotate_image import AnnotateImage, BoundingBox
from image_source import ImageSource, ImageSourceError, OutputType, SourceConfig
from object_detection_postprocessor import DetectionProcessor, DetectionProcessorConfig

# Get logger (configuration done in __main__ if script is run directly)
logger = logging.getLogger(__name__)

@dataclass
class DetectionConfig:
    """
    Configuration for the Gemini Object Detection model.

    Attributes:
        model_name (str): The name of the Gemini model to use.
        default_prompt (str): The default prompt to use for object detection.
    """
    model_name: str = "gemini-2.5-flash"
    default_prompt: str = (
        "Identify all prominent objects in this image. "
        "For each object, provide its 'label', a 'confidence' score (a float between 0 and 1), "
        "and a precise, tight-fitting normalized 'bounding_box'. "
        "Each bounding box must tightly enclose the entire object with no extra padding or background. "
        "Provide the output as a JSON object that strictly adheres to the provided schema."
    )

class Object(BaseModel):
    """
    A Pydantic model representing a single detected object.
    """
    label: str = Field(..., description="Name of the detected object")
    confidence: Optional[float] = Field(None, description="Detection confidence between 0 and 1")
    bounding_box: List[int] = Field(..., description=f"Bounding box [x_min, y_min, x_max, y_max] in 0-1000 range")

class DetectedObjects(BaseModel):
    """
    A Pydantic model representing the root JSON object for the detection response.
    """
    annotated_image_description: str = Field(..., description="A description summarizing the scene and object locations.")
    detected_objects: List[Object] = Field(..., description="List of detected objects with bounding boxes")


class GeminiObjectDetection:
    """
    Detects objects in an image using the Google Gemini API.

    This class provides a high-level interface for sending an image to the Gemini
    API and receiving a list of detected objects with their bounding boxes and labels.
    """
    # Bounding box normalization constant, internal to the class
    _BBOX_NORMALIZATION_RANGE = 1000

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initializes the GeminiObjectDetection instance.

        Args:
            config (Optional[DetectionConfig], optional): A configuration object. If not provided,
                                                          default settings are used.
        """
        self.config = config or DetectionConfig()
        self.client = self._create_client()

    def _create_client(self) -> Optional[genai.Client]:
        """Initialize and return the Gemini client."""
        api_key = os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            logger.warning("No API key provided. Please set the GEMINI_API_KEY environment variable or pass it to the constructor.")
            return None
        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            logger.error(f"Error initializing Gemini client: {e}")
            return None


    @staticmethod
    def normalize_to_absolute_coordinates(
        bbox_normalized: List[float],
        img_width: int,
        img_height: int
    ) -> List[int]:
        """
        Converts normalized bounding box coordinates to absolute pixel coordinates.

        Args:
            bbox_normalized (List[float]): A list of four floats representing the
                                           normalized bounding box `[x_min, y_min, x_max, y_max]`.
            img_width (int): The width of the image in pixels.
            img_height (int): The height of the image in pixels.

        Returns:
            List[int]: A list of four integers representing the absolute bounding
                       box `[x_min, y_min, x_max, y_max]`.
        """
        if len(bbox_normalized) != 4:
            raise ValueError("Bounding box must have exactly 4 values [x_min, y_min, x_max, y_max]")

        x_min, y_min, x_max, y_max = map(float, bbox_normalized)

        # Validate that coordinates are within expected range
        for i, val in enumerate([x_min, y_min, x_max, y_max]):
            if not (0 <= val <= GeminiObjectDetection._BBOX_NORMALIZATION_RANGE):
                logger.warning(
                    f"Bounding box coordinate [{i}] value {val} outside expected range "
                    f"[0, {GeminiObjectDetection._BBOX_NORMALIZATION_RANGE}]"
                )

        # Convert from 0-1000 range to pixel coordinates
        normalization_range = float(GeminiObjectDetection._BBOX_NORMALIZATION_RANGE)
        x1 = int((x_min / normalization_range) * img_width)
        y1 = int((y_min / normalization_range) * img_height)
        x2 = int((x_max / normalization_range) * img_width)
        y2 = int((y_max / normalization_range) * img_height)

        # Ensure coordinates are within image bounds and ordered correctly
        final_x_min = max(0, min(x1, x2))
        final_y_min = max(0, min(y1, y2))
        final_x_max = min(img_width, max(x1, x2))
        final_y_max = min(img_height, max(y1, y2))

        return [final_x_min, final_y_min, final_x_max, final_y_max]

    @staticmethod
    def _calculate_bbox_area(bbox: List[int]) -> int:
        """Calculates the area of a bounding box."""
        if len(bbox) != 4:
            return 0
        x_min, y_min, x_max, y_max = bbox
        return (x_max - x_min) * (y_max - y_min)

    def _add_absolute_coordinates(
        self,
        result: Dict[str, Any],
        pil_image: Image.Image
    ) -> None:
        """Convert normalized bounding boxes to absolute coordinates in-place."""
        img_width, img_height = pil_image.size

        for obj in result.get('detected_objects', []):
            if 'bounding_box' in obj:
                bbox_norm = obj['bounding_box']
                # Validate bounding box has exactly 4 elements
                if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
                    logger.warning(
                        f"Invalid bounding box format for object '{obj.get('label', 'unknown')}': {bbox_norm}. "
                        f"Expected list of 4 values. Skipping this object."
                    )
                    continue
                try:
                    obj['bounding_box'] = self.normalize_to_absolute_coordinates(
                        bbox_norm, img_width, img_height
                    )
                except ValueError as e:
                    logger.warning(f"Failed to normalize bounding box for object '{obj.get('label', 'unknown')}': {e}")

    def _build_prompt(
        self,
        prompt: Optional[str],
        objects_to_detect: Optional[List[str]]
    ) -> str:
        """
        Build the appropriate detection prompt based on input parameters.

        Args:
            prompt: Custom prompt text (takes highest precedence)
            objects_to_detect: List of specific objects to detect

        Returns:
            str: The constructed prompt for object detection
        """
        if prompt:
            return prompt

        if objects_to_detect:
            objects_str = ", ".join(f'"{obj}"' for obj in objects_to_detect)
            # SIMPLIFIED TARGETED PROMPT: Focuses the model on the requested objects and JSON format
            return (
                f"Analyze the image and for each of the requested objects: {objects_str}, provide a precise normalized bounding box. "
                f"The bounding box must tightly enclose all visible parts of the object. "
                f"Return ONLY the requested objects in the exact JSON format with 'label', 'confidence' (float from 0 to 1), and 'bounding_box' fields."
            )

        return self.config.default_prompt

    def _create_payload(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Create the detection request for the Gemini API.
        """
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

        full_prompt = (
            f"{prompt} Provide the output as a single JSON object that strictly adheres to the provided schema. "
            f"The bounding box coordinates (x_min, y_min, x_max, y_max) must be normalized "
            f"to a range of 0 to {self._BBOX_NORMALIZATION_RANGE}, where (0,0) is the top-left corner and ({self._BBOX_NORMALIZATION_RANGE}, {self._BBOX_NORMALIZATION_RANGE}) is the bottom-right. "
            "The 'annotated_image_description' should summarize where the objects are located."
        )

        return {
            "contents": [full_prompt, image_part],
            "config": types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DetectedObjects,
            )
        }

    def _process_response(
        self,
        response: Any,
        pil_image: Image.Image
    ) -> Dict[str, Any]:
        """
        Process the API response, add absolute coordinates, and sort objects.
        
        Args:
            response: The raw response from the Gemini API
            pil_image: The PIL Image object used for detection
            
        Returns:
            Dict containing the processed detection results
        """
        # Validate response has text attribute
        if not hasattr(response, 'text') or not response.text:
            return {"error": "Empty or invalid response from API"}

        try:
            result = json.loads(response.text)

            if 'detected_objects' in result:
                if result['detected_objects']:
                    self._add_absolute_coordinates(result, pil_image)

                    # Sort by bounding box area, largest first
                    result['detected_objects'].sort(
                        key=lambda obj: self._calculate_bbox_area(obj.get('bounding_box', [])),
                        reverse=True
                    )
                    logger.info(f"Successfully detected {len(result['detected_objects'])} object(s)")
                else:
                    logger.info("No objects detected in the image")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            return {"error": f"Failed to decode JSON response: {response.text}"}

    def _pil_to_image_part(self, pil_image: Image.Image) -> types.Part:
        """
        Convert a PIL image to a Gemini API `types.Part` object.
        """
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return types.Part.from_bytes(data=img_byte_arr, mime_type='image/png')

    def _load_and_prepare_image(self, image_source: str) -> Tuple[Optional[Image.Image], Optional[types.Part]]:
        """Load an image using ImageSource and prepare it for the API."""
        try:
            source_config = SourceConfig(output_type=OutputType.PIL)
            source = ImageSource(image_source, config=source_config)
            image_result = source.get_image()
            pil_image = image_result.data
            return pil_image, self._pil_to_image_part(pil_image)
        except ImageSourceError as e:
            logger.error(f"Error loading image source '{image_source}': {e}")
            return None, None

    def detect_objects(
        self,
        image_source: str,
        prompt: Optional[str] = None,
        objects_to_detect: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """
        Detects objects in an image.

        This method takes an image source (file path or URL), an optional prompt,
        and an optional list of objects to detect. It returns a dictionary
        containing the detection results and the PIL Image object.

        Args:
            image_source (str): The path or URL to the image.
            prompt (Optional[str], optional): A custom prompt to guide the detection.
            objects_to_detect (Optional[List[str]], optional): A list of specific objects
                                                               to detect.

        Returns:
            Tuple[Dict[str, Any], Optional[Image.Image]]: A tuple containing a dictionary
                                                          with the detection results and
                                                          the PIL Image object.
        """
        if not self.client:
            return {"error": "Gemini client is not initialized"}, None

        try:
            pil_image, image_part = self._load_and_prepare_image(image_source)
            if not pil_image or not image_part:
                return {"error": f"Failed to load image: {image_source}"}, None

            prompt = self._build_prompt(prompt, objects_to_detect)
            payload = self._create_payload(image_part.inline_data.data, image_part.inline_data.mime_type, prompt)

            logger.info(f"Calling Gemini model for object detection: {self.config.model_name}")
            response = self.client.models.generate_content(
                model=self.config.model_name,
                **payload
            )

            # Process the JSON response
            result = self._process_response(response=response, pil_image=pil_image)

            return result, pil_image

        except APIError as e:
            logger.error(f"Gemini API Error: {e}")
            return {"error": f"Gemini API Error (SDK): {e}"}, None
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}, None



if __name__ == "__main__":
    # Configure logging for CLI usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 1. Setup argument parser
    parser = argparse.ArgumentParser(
        description="Run Gemini Object Detection on a local image file or URL."
    )
    parser.add_argument(
        "-i", "--image_source",
        type=str,
        required=True,
        help="The local file path or URL of the image (e.g., /path/to/photo.jpg, http://example.com/image.png)."
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        default=None,
        help="Optional custom string prompt for object detection (e.g., 'Detect only cars and people'). Takes precedence over --objects."
    )
    parser.add_argument(
        "-d", "--detect",
        type=str,
        default=None,
        help="Optional comma-separated list of objects to detect (e.g., 'cat,dog,tree'). If provided, it overrides the default prompt."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="annotated_images",
        help="The directory where annotated images will be saved."
    )
    parser.add_argument(
        "--one-object-per-image",
        action="store_true",
        help="If set, save a separate annotated image for each detected object."
    )

    args = parser.parse_args()

    # 2. Parse objects_to_detect list
    objects_list = [obj.strip() for obj in args.detect.split(',')] if args.detect else None

    # 3. Initialize the object detection and processor classes
    detector = GeminiObjectDetection()
    processor_config = DetectionProcessorConfig(
        output_dir=args.output_dir,
        one_object_per_image=args.one_object_per_image
    )
    processor = DetectionProcessor(config=processor_config)

    # 4. Call the detection method
    result, pil_image = detector.detect_objects(
        image_source=args.image_source,
        prompt=args.prompt,
        objects_to_detect=objects_list
    )

    # 5. Process and save the annotated image if detection was successful
    if result and 'error' not in result and pil_image:
        annotated_image_paths = processor.process_and_save(
            pil_image=pil_image,
            detection_result=result,
            image_source=args.image_source
        )
        if annotated_image_paths:
            result['annotated_image_paths'] = annotated_image_paths

    # 6. Print the result
    if isinstance(result, dict):
        # Pretty print the JSON output
        print(json.dumps(result, indent=2))
    else:
        # Print error messages
        print(result)
