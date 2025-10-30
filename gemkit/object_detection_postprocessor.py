import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image

from annotate_image import AnnotateImage, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class DetectionProcessorConfig:
    """
    Configuration for the DetectionProcessor.

    Attributes:
        output_dir (str): The directory to save the annotated images.
        annotated_image_prefix (str): The prefix for the annotated image filenames.
        one_object_per_image (bool): If True, save a separate image for each detected object.
    """
    output_dir: str = "annotated_images"
    annotated_image_prefix: str = "detected"
    one_object_per_image: bool = False


class DetectionProcessor:
    """
    Processes the results of an object detection, including annotating
    and saving the images.
    """

    def __init__(self, config: Optional[DetectionProcessorConfig] = None):
        """
        Initializes the DetectionProcessor.

        Args:
            config (Optional[DetectionProcessorConfig], optional): A configuration object.
                                                                    If not provided, default
                                                                    settings are used.
        """
        self.config = config or DetectionProcessorConfig()

    def _prepare_bounding_boxes(self, objects: List[Dict[str, Any]]) -> List[BoundingBox]:
        """Convert detection data into BoundingBox objects."""
        boxes = []
        for i, obj in enumerate(objects):
            if 'bounding_box' in obj:
                bbox_abs = obj['bounding_box']
                logger.debug(f"Object {i+1} - {obj.get('label', 'unknown')}:")
                logger.debug(f"  Absolute bbox: {bbox_abs}")

                box = BoundingBox(
                    x_min=bbox_abs[0],
                    y_min=bbox_abs[1],
                    x_max=bbox_abs[2],
                    y_max=bbox_abs[3],
                    label=obj.get('label', ''),
                    confidence=obj.get('confidence', 0.0)
                )
                boxes.append(box)
        return boxes

    def _generate_output_path(
        self,
        image_source: str,
        obj_info: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate the output path for the annotated image.
        If obj_info is provided, create a unique name for single-object images.
        """
        if image_source.startswith(('http://', 'https://')):
            base_name = os.path.splitext(os.path.basename(image_source.split('?')[0]))[0]
        else:
            base_name = os.path.splitext(os.path.basename(image_source))[0]

        if obj_info:
            label = obj_info.get('label', 'object')
            index = obj_info.get('index', 0)
            output_filename = f"{self.config.annotated_image_prefix}_{base_name}_{label}_{index}_annotated.png"
        else:
            output_filename = f"{self.config.annotated_image_prefix}_{base_name}_annotated.png"

        return os.path.join(self.config.output_dir, output_filename)

    def process_and_save(
        self,
        pil_image: Image.Image,
        detection_result: Dict[str, Any],
        image_source: str,
    ) -> List[str]:
        """
        Draws bounding boxes on an image and saves the result.

        This method can either draw all bounding boxes on a single image or create
        a separate image for each detected object, depending on the configuration.

        Args:
            pil_image (Image.Image): The original image.
            detection_result (Dict[str, Any]): The object detection results.
            image_source (str): The original source of the image (for naming the output file).

        Returns:
            List[str]: A list of paths to the saved annotated images.
        """
        saved_paths = []
        objects = detection_result.get('detected_objects', [])
        if not objects:
            logger.warning("No detected objects to process.")
            return []

        try:
            os.makedirs(self.config.output_dir, exist_ok=True)

            if self.config.one_object_per_image:
                for i, obj in enumerate(objects):
                    if 'bounding_box' not in obj:
                        continue

                    box = self._prepare_bounding_boxes([obj])[0]
                    image_copy = pil_image.copy()
                    annotator = AnnotateImage(image=image_copy, output_dir=self.config.output_dir)
                    annotator.draw_boxes([box])

                    obj_info = {'label': obj.get('label', 'object'), 'index': i}
                    output_path = self._generate_output_path(image_source, obj_info)
                    saved_path = annotator.save(output_path)
                    saved_paths.append(saved_path)
                    logger.info(f"Annotated image for '{obj.get('label')}' saved to: {saved_path}")

            else:
                boxes = self._prepare_bounding_boxes(objects)
                if not boxes:
                    logger.warning("No valid bounding boxes to draw")
                    return []

                annotator = AnnotateImage(image=pil_image.copy(), output_dir=self.config.output_dir)
                annotator.draw_boxes(boxes)
                output_path = self._generate_output_path(image_source)
                saved_path = annotator.save(output_path)
                saved_paths.append(saved_path)
                logger.info(f"Annotated image saved to: {saved_path}")

            return saved_paths

        except Exception as e:
            logger.error(f"Error drawing bounding boxes: {e}", exc_info=True)
            return []
