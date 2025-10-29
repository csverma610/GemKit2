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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError
)

from annotate_image import AnnotateImage, BoundingBox
from image_source import ImageSource, ImageSourceError, OutputType, SourceConfig
from object_detection_postprocessor import DetectionProcessor, DetectionProcessorConfig

# Get logger (configuration done in __main__ if script is run directly)
logger = logging.getLogger(__name__)


# Custom Exceptions
class GeminiDetectionError(Exception):
    """Base exception for Gemini object detection errors."""
    pass


class ConfigurationError(GeminiDetectionError):
    """Raised when there's a configuration error."""
    pass


class APIClientError(GeminiDetectionError):
    """Raised when API client initialization fails."""
    pass


class ImageProcessingError(GeminiDetectionError):
    """Raised when image processing fails."""
    pass


class DetectionAPIError(GeminiDetectionError):
    """Raised when detection API call fails."""
    pass

@dataclass
class DetectionConfig:
    """
    Configuration for the Gemini Object Detection.

    Attributes:
        model_name: Gemini model to use for detection
        default_prompt: Default prompt for object detection
        enable_tiling: Enable tiling approach for small object detection
        tile_size: Size of each square tile in pixels (must be > 0)
        overlap_ratio: Overlap between tiles (must be 0.0 to 1.0)
        iou_threshold: IoU threshold for deduplication (must be 0.0 to 1.0)
        max_batch_size_mb: Maximum batch size in MB (must be > 0)
        max_retries: Maximum number of retry attempts for API calls
        timeout_seconds: Timeout for API calls in seconds
    """
    model_name: str = "gemini-2.5-flash"
    default_prompt: str = (
        "Identify all prominent objects in this image. "
        "For each object, provide its 'label', a 'confidence' score (a float between 0 and 1), "
        "and a precise, tight-fitting normalized 'bounding_box'. "
        "Each bounding box must tightly enclose the entire object with no extra padding or background. "
        "Provide the output as a JSON object that strictly adheres to the provided schema."
    )
    # Tiling configuration for small object detection
    enable_tiling: bool = False
    tile_size: int = 640  # Size of each tile in pixels (square tiles)
    overlap_ratio: float = 0.2  # Overlap between tiles (0.0 to 1.0)
    iou_threshold: float = 0.5  # IoU threshold for deduplication
    max_batch_size_mb: float = 10.0  # Maximum batch size in MB for processing tiles
    max_retries: int = 3  # Maximum number of retry attempts
    timeout_seconds: int = 120  # Timeout for API calls in seconds

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {self.tile_size}")

        if not 0.0 <= self.overlap_ratio < 1.0:
            raise ValueError(f"overlap_ratio must be in [0.0, 1.0), got {self.overlap_ratio}")

        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0.0, 1.0], got {self.iou_threshold}")

        if self.max_batch_size_mb <= 0:
            raise ValueError(f"max_batch_size_mb must be positive, got {self.max_batch_size_mb}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")

        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")

class Object(BaseModel):
    """Represents a single detected object."""
    label: str = Field(..., description="Name of the detected object")
    confidence: Optional[float] = Field(None, description="Detection confidence between 0 and 1")
    bounding_box: List[int] = Field(..., description=f"Bounding box [x_min, y_min, x_max, y_max] in 0-1000 range")

class DetectedObjects(BaseModel):
    """The root JSON object for the detection response."""
    annotated_image_description: str = Field(..., description="A description summarizing the scene and object locations.")
    detected_objects: List[Object] = Field(..., description="List of detected objects with bounding boxes")


class GeminiObjectDetection:
    # Bounding box normalization constant, internal to the class
    _BBOX_NORMALIZATION_RANGE = 1000

    def __init__(self, config: Optional[DetectionConfig] = None):
        """
        Initialize the Gemini client and set up the model.

        Args:
            config: Optional configuration object. If None, default settings are used.
        """
        self.config = config or DetectionConfig()
        self.client = self._create_client()

    def _create_client(self) -> genai.Client:
        """
        Initialize and return the Gemini client.

        Returns:
            Initialized Gemini client

        Raises:
            APIClientError: If client initialization fails
        """
        api_key = os.getenv('GEMINI_API_KEY', '')
        if not api_key:
            raise APIClientError(
                "No API key provided. Please set the GEMINI_API_KEY environment variable."
            )
        try:
            client = genai.Client(api_key=api_key)
            logger.info(f"Successfully initialized Gemini client with model: {self.config.model_name}")
            return client
        except Exception as e:
            raise APIClientError(f"Failed to initialize Gemini client: {e}") from e

    def _retry_api_call(self, func, *args, **kwargs):
        """
        Execute API call with retry logic using tenacity library.

        Args:
            func: Function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Result from the function call

        Raises:
            DetectionAPIError: If all retries fail
        """
        from tenacity import Retrying

        # Configure retry strategy using tenacity
        retryer = Retrying(
            stop=stop_after_attempt(self.config.max_retries + 1),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type(APIError),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True
        )

        try:
            # Execute the function with retry logic
            return retryer(func, *args, **kwargs)
        except APIError as e:
            # All retries exhausted
            logger.error(f"API call failed after {self.config.max_retries + 1} attempts")
            raise DetectionAPIError(f"API call failed after {self.config.max_retries + 1} attempts") from e
        except RetryError as e:
            # This shouldn't happen with reraise=True, but handle it just in case
            logger.error(f"Retry error: {e}")
            raise DetectionAPIError(f"API call failed: {e}") from e
        except Exception as e:
            # Non-retryable error
            raise DetectionAPIError(f"API call failed with non-retryable error: {e}") from e


    @staticmethod
    def normalize_to_absolute_coordinates(
        bbox_normalized: List[float],
        img_width: int,
        img_height: int
    ) -> List[int]:
        """
        Convert normalized bounding box coordinates (0-1000 range) to absolute pixel coordinates.
        This corrected logic scales the coordinates directly by the image dimensions.
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

    @staticmethod
    def _calculate_image_size_mb(pil_image: Image.Image) -> float:
        """
        Calculate the approximate size of a PIL image in megabytes.

        Args:
            pil_image: PIL Image object

        Returns:
            Approximate size in MB
        """
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format='PNG')
        size_bytes = img_byte_arr.tell()
        size_mb = size_bytes / (1024 * 1024)
        return size_mb

    def _tile_image(self, pil_image: Image.Image) -> List[Tuple[Image.Image, int, int]]:
        """
        Tile the image into overlapping subimages.

        Args:
            pil_image: The original PIL Image to tile

        Returns:
            List of tuples containing (tile_image, x_offset, y_offset)
            where x_offset and y_offset are the top-left coordinates of the tile
            in the original image coordinate system.
        """
        img_width, img_height = pil_image.size
        tile_size = self.config.tile_size
        overlap_ratio = self.config.overlap_ratio

        # Calculate stride (step size) based on overlap
        stride = int(tile_size * (1 - overlap_ratio))

        if stride <= 0:
            logger.warning(f"Invalid stride {stride}. Using tile_size as stride.")
            stride = tile_size

        tiles = []

        # Generate tiles with overlap
        y = 0
        while y < img_height:
            x = 0
            while x < img_width:
                # Calculate tile boundaries
                x_end = min(x + tile_size, img_width)
                y_end = min(y + tile_size, img_height)

                # Crop the tile from the original image
                tile = pil_image.crop((x, y, x_end, y_end))
                tiles.append((tile, x, y))

                # Move to next column
                x += stride
                if x >= img_width:
                    break

            # Move to next row
            y += stride
            if y >= img_height:
                break

        logger.info(f"Created {len(tiles)} tiles from image of size {img_width}x{img_height}")
        return tiles

    def _get_image_size_safe(self, img: Image.Image, idx: int) -> Optional[float]:
        """
        Get image size in MB with error handling.

        Args:
            img: PIL Image
            idx: Image index for logging

        Returns:
            Size in MB, or None if calculation fails
        """
        try:
            return self._calculate_image_size_mb(img)
        except Exception as e:
            logger.warning(f"Failed to calculate size for image {idx}: {e}. Skipping.")
            return None

    def _should_start_new_batch(self, current_size_mb: float, img_size_mb: float) -> bool:
        """
        Check if a new batch should be started.

        Args:
            current_size_mb: Current batch size in MB
            img_size_mb: Size of image to add in MB

        Returns:
            True if new batch should be started
        """
        return current_size_mb > 0 and (current_size_mb + img_size_mb) > self.config.max_batch_size_mb

    def _finalize_current_batch(
        self,
        batches: List[List[Tuple[Image.Image, Optional[int], Optional[int]]]],
        current_batch: List[Tuple[Image.Image, Optional[int], Optional[int]]]
    ) -> Tuple[List[Tuple[Image.Image, Optional[int], Optional[int]]], float]:
        """
        Finalize current batch and add to batches list.

        Args:
            batches: List of all batches
            current_batch: Current batch to finalize

        Returns:
            Tuple of (new empty batch, size of 0.0)
        """
        if current_batch:
            batches.append(current_batch)
        return [], 0.0

    def _handle_oversized_image(
        self,
        batches: List[List[Tuple[Image.Image, Optional[int], Optional[int]]]],
        current_batch: List[Tuple[Image.Image, Optional[int], Optional[int]]],
        img: Image.Image,
        x_offset: Optional[int],
        y_offset: Optional[int],
        img_size_mb: float,
        idx: int
    ) -> Tuple[List[Tuple[Image.Image, Optional[int], Optional[int]]], float]:
        """
        Handle image that exceeds max batch size.

        Args:
            batches: List of all batches
            current_batch: Current batch being built
            img: The oversized image
            x_offset: X offset (None for original)
            y_offset: Y offset (None for original)
            img_size_mb: Size of image in MB
            idx: Image index for logging

        Returns:
            Tuple of (new empty batch, size of 0.0)
        """
        logger.warning(
            f"Image {idx} size ({img_size_mb:.2f} MB) exceeds max_batch_size_mb "
            f"({self.config.max_batch_size_mb} MB). Creating separate batch."
        )
        # Finalize current batch if it exists
        if current_batch:
            batches.append(current_batch)
        # Create separate batch for oversized image
        batches.append([(img, x_offset, y_offset)])
        return [], 0.0

    def _batch_images(
        self,
        original_image: Image.Image,
        tiles: List[Tuple[Image.Image, int, int]]
    ) -> List[List[Tuple[Image.Image, Optional[int], Optional[int]]]]:
        """
        Batch the original image and tiles based on maximum batch size in MB.
        Each batch will be sent in a single API call.

        Args:
            original_image: The original full image
            tiles: List of (tile_image, x_offset, y_offset) tuples

        Returns:
            List of batches, where each batch is a list of (image, x_offset, y_offset) tuples.
            For the original image, offsets are None.

        Raises:
            ImageProcessingError: If batching fails
        """
        if not original_image:
            raise ImageProcessingError("Original image cannot be None")

        try:
            all_images = [(original_image, None, None)] + tiles

            if not all_images:
                logger.warning("No images to batch")
                return []

            batches = []
            current_batch = []
            current_batch_size_mb = 0.0

            for idx, (img, x_offset, y_offset) in enumerate(all_images):
                # Get image size safely
                img_size_mb = self._get_image_size_safe(img, idx)
                if img_size_mb is None:
                    continue

                # Handle oversized images
                if img_size_mb > self.config.max_batch_size_mb:
                    current_batch, current_batch_size_mb = self._handle_oversized_image(
                        batches, current_batch, img, x_offset, y_offset, img_size_mb, idx
                    )
                    continue

                # Start new batch if adding this image would exceed limit
                if self._should_start_new_batch(current_batch_size_mb, img_size_mb):
                    current_batch, current_batch_size_mb = self._finalize_current_batch(batches, current_batch)

                # Add image to current batch
                current_batch.append((img, x_offset, y_offset))
                current_batch_size_mb += img_size_mb

            # Finalize last batch
            self._finalize_current_batch(batches, current_batch)

            total_images = len(all_images)
            logger.info(
                f"Created {len(batches)} batch(es) from {total_images} images "
                f"(1 original + {len(tiles)} tiles, max batch size: {self.config.max_batch_size_mb} MB)"
            )
            return batches

        except Exception as e:
            raise ImageProcessingError(f"Failed to batch images: {e}") from e

    @staticmethod
    def _is_box_completely_inside(bbox: List[int], tile_width: int, tile_height: int) -> bool:
        """
        Check if a bounding box is completely inside a tile.

        Args:
            bbox: Bounding box in absolute coordinates [x_min, y_min, x_max, y_max]
            tile_width: Width of the tile
            tile_height: Height of the tile

        Returns:
            True if the bounding box is completely inside the tile boundaries
        """
        if len(bbox) != 4:
            return False

        x_min, y_min, x_max, y_max = bbox

        # Check if all coordinates are within tile boundaries
        return (x_min >= 0 and y_min >= 0 and
                x_max <= tile_width and y_max <= tile_height)

    @staticmethod
    def _transform_bbox_to_original(bbox: List[int], x_offset: int, y_offset: int) -> List[int]:
        """
        Transform bounding box coordinates from tile-local to original image coordinates.

        Args:
            bbox: Bounding box in tile coordinates [x_min, y_min, x_max, y_max]
            x_offset: X offset of the tile in the original image
            y_offset: Y offset of the tile in the original image

        Returns:
            Bounding box in original image coordinates
        """
        if len(bbox) != 4:
            return bbox

        x_min, y_min, x_max, y_max = bbox
        return [x_min + x_offset, y_min + y_offset, x_max + x_offset, y_max + y_offset]

    @staticmethod
    def _calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            bbox1: First bounding box [x_min, y_min, x_max, y_max]
            bbox2: Second bounding box [x_min, y_min, x_max, y_max]

        Returns:
            IoU value between 0 and 1
        """
        if len(bbox1) != 4 or len(bbox2) != 4:
            return 0.0

        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection area
        x_intersection_min = max(x1_min, x2_min)
        y_intersection_min = max(y1_min, y2_min)
        x_intersection_max = min(x1_max, x2_max)
        y_intersection_max = min(y1_max, y2_max)

        # Check if there is an intersection
        if x_intersection_max <= x_intersection_min or y_intersection_max <= y_intersection_min:
            return 0.0

        intersection_area = (x_intersection_max - x_intersection_min) * (y_intersection_max - y_intersection_min)

        # Calculate union area
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - intersection_area

        # Avoid division by zero
        if union_area == 0:
            return 0.0

        return intersection_area / union_area

    def _deduplicate_detections(self, all_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate detections using IoU threshold and keep highest confidence.

        Args:
            all_detections: List of detection objects with 'label', 'confidence', and 'bounding_box'

        Returns:
            Deduplicated list of detections
        """
        if not all_detections:
            return []

        # Sort by confidence (highest first)
        sorted_detections = sorted(
            all_detections,
            key=lambda x: x.get('confidence', 0.0),
            reverse=True
        )

        kept_detections = []

        for detection in sorted_detections:
            bbox = detection.get('bounding_box')
            if not bbox or len(bbox) != 4:
                continue

            # Check if this detection overlaps significantly with any kept detection
            is_duplicate = False
            for kept in kept_detections:
                kept_bbox = kept.get('bounding_box')
                if not kept_bbox:
                    continue

                # Calculate IoU
                iou = self._calculate_iou(bbox, kept_bbox)

                # If IoU is above threshold and labels match, consider it a duplicate
                if iou >= self.config.iou_threshold and detection.get('label') == kept.get('label'):
                    is_duplicate = True
                    logger.debug(
                        f"Removing duplicate detection: {detection.get('label')} "
                        f"(IoU={iou:.2f}, conf={detection.get('confidence', 0):.2f})"
                    )
                    break

            if not is_duplicate:
                kept_detections.append(detection)

        logger.info(f"Deduplication: {len(all_detections)} -> {len(kept_detections)} detections")
        return kept_detections

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

    def _detect_objects_single(
        self,
        pil_image: Image.Image,
        prompt: str
    ) -> Dict[str, Any]:
        """
        Detect objects in a single image (helper method for tiling).

        Args:
            pil_image: PIL Image to process
            prompt: Detection prompt

        Returns:
            Dict with detection results or error information
        """
        try:
            image_part = self._pil_to_image_part(pil_image)
            payload = self._create_payload(
                image_part.inline_data.data,
                image_part.inline_data.mime_type,
                prompt
            )

            response = self.client.models.generate_content(
                model=self.config.model_name,
                **payload
            )

            return self._process_response(response=response, pil_image=pil_image)

        except APIError as e:
            logger.error(f"Gemini API Error: {e}")
            return {"error": f"Gemini API Error (SDK): {e}"}
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return {"error": f"An unexpected error occurred: {e}"}

    def _detect_objects_batch(
        self,
        images_with_metadata: List[Tuple[Image.Image, Optional[int], Optional[int]]],
        prompt: str
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in multiple images with a single API call (with retry logic).

        Args:
            images_with_metadata: List of (image, x_offset, y_offset) tuples
            prompt: Detection prompt

        Returns:
            List of detection results, one dict per image in the same order

        Raises:
            DetectionAPIError: If detection fails after retries
            ImageProcessingError: If image processing fails
        """
        if not images_with_metadata:
            logger.warning("Empty image list provided for batch detection")
            return []

        num_images = len(images_with_metadata)
        logger.info(f"Starting batch detection for {num_images} image(s)")

        try:
            # Create image parts for all images
            image_parts = []
            for idx, (img, _, _) in enumerate(images_with_metadata):
                if img is None:
                    raise ImageProcessingError(f"Image at index {idx} is None")
                try:
                    image_parts.append(self._pil_to_image_part(img))
                except Exception as e:
                    raise ImageProcessingError(f"Failed to convert image {idx} to part: {e}") from e

            # Build prompt for multi-image analysis
            multi_image_prompt = (
                f"You are analyzing {num_images} images. "
                f"For each image (numbered 0 to {num_images - 1}), "
                f"{prompt} "
                f"Return a JSON array where each element corresponds to one image's results in order. "
                f"Each element should have 'image_index', 'annotated_image_description', and 'detected_objects'."
            )

            # Create contents array with prompt and all images
            contents = [multi_image_prompt] + image_parts

            # Create payload with multiple images
            payload = {
                "contents": contents,
                "config": types.GenerateContentConfig(
                    response_mime_type="application/json",
                )
            }

            # Execute API call with retry logic
            def api_call():
                return self.client.models.generate_content(
                    model=self.config.model_name,
                    **payload
                )

            logger.info(f"Sending batch API call with {num_images} image(s)")
            response = self._retry_api_call(api_call)

            # Parse response
            if not hasattr(response, 'text') or not response.text:
                raise DetectionAPIError("Empty or invalid response from API")

            try:
                results_array = json.loads(response.text)
                if not isinstance(results_array, list):
                    logger.warning("API returned non-array response, wrapping in array")
                    results_array = [results_array]

                # Validate response length
                if len(results_array) != num_images:
                    logger.warning(
                        f"Expected {num_images} results, got {len(results_array)}. "
                        f"Padding or truncating as needed."
                    )

                # Process each image's results
                processed_results = []
                for idx, (img, _, _) in enumerate(images_with_metadata):
                    if idx < len(results_array):
                        result = results_array[idx]
                        # Validate result structure
                        if not isinstance(result, dict):
                            logger.warning(f"Invalid result format for image {idx}")
                            processed_results.append({"detected_objects": []})
                            continue

                        # Add absolute coordinates
                        if 'detected_objects' in result and isinstance(result['detected_objects'], list):
                            temp_result = {'detected_objects': result.get('detected_objects', [])}
                            try:
                                self._add_absolute_coordinates(temp_result, img)
                                result['detected_objects'] = temp_result['detected_objects']
                            except Exception as e:
                                logger.warning(f"Failed to add absolute coordinates for image {idx}: {e}")

                        processed_results.append(result)
                    else:
                        logger.warning(f"No result for image {idx}, using empty result")
                        processed_results.append({"detected_objects": []})

                logger.info(f"Successfully processed batch detection for {num_images} image(s)")
                return processed_results

            except json.JSONDecodeError as e:
                raise DetectionAPIError(f"Failed to decode JSON response: {e}") from e

        except (DetectionAPIError, ImageProcessingError):
            # Re-raise these specific exceptions
            raise
        except Exception as e:
            raise DetectionAPIError(f"Unexpected error during batch detection: {e}") from e

    def detect_objects_with_tiling(
        self,
        image_source: str,
        prompt: Optional[str] = None,
        objects_to_detect: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """
        Detect objects using tiling approach for small object detection.

        This production-grade method:
        1. Validates inputs and loads the image
        2. Tiles the image into overlapping subimages
        3. Batches original + tiles based on payload size limit
        4. Sends each batch in a SINGLE API call (multiple images per call) with retry logic
        5. Filters detections that are completely inside tiles
        6. Transforms tile coordinates to original image coordinates
        7. Deduplicates using IoU and keeps highest confidence
        8. Handles all errors gracefully with detailed logging

        Args:
            image_source: Path or URL to the image
            prompt: Optional custom detection prompt
            objects_to_detect: Optional list of specific objects to detect

        Returns:
            Tuple containing:
                - Dict with detection results or error information
                - PIL Image object if successful, None if error occurred

        Raises:
            Exceptions are caught and returned as error dict for backward compatibility
        """
        # Validate inputs
        if not image_source or not isinstance(image_source, str):
            error_msg = "image_source must be a non-empty string"
            logger.error(error_msg)
            return {"error": error_msg}, None

        try:
            # Load and validate image
            pil_image, _ = self._load_and_prepare_image(image_source)
            if not pil_image:
                error_msg = f"Failed to load image from source: {image_source}"
                logger.error(error_msg)
                return {"error": error_msg}, None

            # Get image dimensions for logging
            img_width, img_height = pil_image.size
            logger.info(f"Loaded image: {img_width}x{img_height} from {image_source}")

            # Build detection prompt
            prompt_text = self._build_prompt(prompt, objects_to_detect)

            # 1. Tile the image
            try:
                tiles = self._tile_image(pil_image)
                if not tiles:
                    logger.warning("No tiles created, image may be smaller than tile_size")
            except Exception as e:
                error_msg = f"Failed to tile image: {e}"
                logger.error(error_msg, exc_info=True)
                return {"error": error_msg}, None

            # 2. Batch original image + all tiles based on size limit
            try:
                batches = self._batch_images(pil_image, tiles)
                if not batches:
                    error_msg = "No batches created from images"
                    logger.error(error_msg)
                    return {"error": error_msg}, None
            except ImageProcessingError as e:
                error_msg = f"Failed to batch images: {e}"
                logger.error(error_msg, exc_info=True)
                return {"error": error_msg}, None

            all_detections = []
            total_api_calls = 0
            failed_batches = 0

            # 3. Process each batch with a SINGLE API call
            for batch_idx, batch in enumerate(batches):
                logger.info(
                    f"Processing batch {batch_idx+1}/{len(batches)} with {len(batch)} image(s) "
                    f"(1 API call for this batch)"
                )

                try:
                    # Send all images in this batch in ONE API call
                    batch_results = self._detect_objects_batch(batch, prompt_text)
                    total_api_calls += 1

                    # Process results for each image in the batch
                    for img_idx, (img, x_offset, y_offset) in enumerate(batch):
                        if img_idx >= len(batch_results):
                            logger.warning(f"Missing result for image {img_idx} in batch {batch_idx+1}")
                            continue

                        result = batch_results[img_idx]
                        if not isinstance(result, dict):
                            logger.warning(f"Invalid result type for image {img_idx}: {type(result)}")
                            continue

                        detections = result.get('detected_objects', [])
                        if not isinstance(detections, list):
                            logger.warning(f"Invalid detections type: {type(detections)}")
                            continue

                        # If this is the original image (offsets are None)
                        if x_offset is None and y_offset is None:
                            logger.info(f"Original image: {len(detections)} detections")
                            all_detections.extend(detections)
                        else:
                            # This is a tile - filter and transform coordinates
                            tile_width, tile_height = img.size
                            valid_detections = 0

                            for detection in detections:
                                if not isinstance(detection, dict):
                                    continue

                                bbox = detection.get('bounding_box')
                                if not bbox or not isinstance(bbox, list) or len(bbox) != 4:
                                    continue

                                # Check if box is completely inside tile
                                if self._is_box_completely_inside(bbox, tile_width, tile_height):
                                    # Transform coordinates to original image space
                                    try:
                                        original_bbox = self._transform_bbox_to_original(bbox, x_offset, y_offset)
                                        detection['bounding_box'] = original_bbox
                                        all_detections.append(detection)
                                        valid_detections += 1
                                    except Exception as e:
                                        logger.warning(f"Failed to transform bbox: {e}")

                            logger.debug(f"Tile at ({x_offset}, {y_offset}): added {valid_detections}/{len(detections)} detections")

                except (DetectionAPIError, ImageProcessingError) as e:
                    failed_batches += 1
                    logger.error(f"Batch {batch_idx+1} failed: {e}", exc_info=True)
                    # Continue processing remaining batches
                except Exception as e:
                    failed_batches += 1
                    logger.error(f"Unexpected error in batch {batch_idx+1}: {e}", exc_info=True)

            # Check if all batches failed
            if failed_batches == len(batches):
                error_msg = f"All {len(batches)} batches failed during detection"
                logger.error(error_msg)
                return {"error": error_msg}, None

            # 4. Deduplicate detections
            logger.info(f"Total detections before deduplication: {len(all_detections)}")
            try:
                deduplicated_detections = self._deduplicate_detections(all_detections)
            except Exception as e:
                logger.warning(f"Deduplication failed, using all detections: {e}")
                deduplicated_detections = all_detections

            # 5. Sort by area (largest first)
            try:
                deduplicated_detections.sort(
                    key=lambda obj: self._calculate_bbox_area(obj.get('bounding_box', [])),
                    reverse=True
                )
            except Exception as e:
                logger.warning(f"Sorting failed: {e}")

            # Build final result
            result = {
                'annotated_image_description': (
                    f'Detected objects using tiled approach with {total_api_calls} API call(s). '
                    f'Failed batches: {failed_batches}/{len(batches)}'
                ),
                'detected_objects': deduplicated_detections,
                'tiling_stats': {
                    'num_tiles': len(tiles),
                    'num_batches': len(batches),
                    'num_api_calls': total_api_calls,
                    'failed_batches': failed_batches,
                    'tile_size': self.config.tile_size,
                    'overlap_ratio': self.config.overlap_ratio,
                    'max_batch_size_mb': self.config.max_batch_size_mb,
                    'total_detections_before_dedup': len(all_detections),
                    'final_detections': len(deduplicated_detections)
                }
            }

            logger.info(
                f"Tiling detection completed: {len(deduplicated_detections)} final detections "
                f"from {total_api_calls} API call(s)"
            )
            return result, pil_image

        except Exception as e:
            error_msg = f"Unexpected error in tiling detection: {e}"
            logger.error(error_msg, exc_info=True)
            return {"error": error_msg}, None

    def detect_objects(
        self,
        image_source: str,
        prompt: Optional[str] = None,
        objects_to_detect: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], Optional[Image.Image]]:
        """
        Detect objects in an image using the Gemini API.

        If tiling is enabled in config, uses the tiling approach for small object detection.
        Otherwise, performs standard detection on the full image.

        Returns:
            Tuple containing:
                - Dict with detection results or error information (never None)
                - PIL Image object if successful, None if error occurred
        """
        # Use tiling approach if enabled
        if self.config.enable_tiling:
            logger.info("Tiling enabled - using tiled detection approach")
            return self.detect_objects_with_tiling(image_source, prompt, objects_to_detect)

        # Standard detection (original implementation)
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
    parser.add_argument(
        "--enable-tiling",
        action="store_true",
        help="Enable tiling approach for small object detection. The image will be divided into overlapping tiles."
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Size of each tile in pixels (default: 640). Only used when --enable-tiling is set."
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.2,
        help="Overlap ratio between tiles, from 0.0 to 1.0 (default: 0.2). Only used when --enable-tiling is set."
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for deduplication, from 0.0 to 1.0 (default: 0.5). Only used when --enable-tiling is set."
    )
    parser.add_argument(
        "--max-batch-size-mb",
        type=float,
        default=10.0,
        help="Maximum batch size in MB for processing tiles (default: 10.0). Only used when --enable-tiling is set."
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for API calls (default: 3)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Timeout for API calls in seconds (default: 120)."
    )

    args = parser.parse_args()

    try:
        # 2. Parse objects_to_detect list
        objects_list = [obj.strip() for obj in args.detect.split(',')] if args.detect else None

        # 3. Initialize the object detection and processor classes with validation
        detection_config = DetectionConfig(
            enable_tiling=args.enable_tiling,
            tile_size=args.tile_size,
            overlap_ratio=args.overlap_ratio,
            iou_threshold=args.iou_threshold,
            max_batch_size_mb=args.max_batch_size_mb,
            max_retries=args.max_retries,
            timeout_seconds=args.timeout
        )
        detector = GeminiObjectDetection(config=detection_config)
    except (ValueError, ConfigurationError, APIClientError) as e:
        logger.error(f"Configuration/initialization error: {e}")
        print(json.dumps({"error": str(e)}, indent=2))
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected initialization error: {e}", exc_info=True)
        print(json.dumps({"error": f"Unexpected error: {e}"}, indent=2))
        exit(1)
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
