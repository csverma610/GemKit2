"""
A class for annotating images with bounding boxes, labels, and other visual elements.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image, ImageDraw, ImageFont
import random
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates and metadata."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    label: str = ""
    confidence: float = 0.0
    color: Optional[Tuple[int, int, int]] = None


class AnnotateImage:
    """
    A class for annotating images with bounding boxes and text.

    Features:
    - Draw multiple bounding boxes with customizable styles
    - Add labels with confidence scores
    - Support for different fonts and text sizes
    - Automatic color assignment for different labels
    - Save annotated images with timestamps
    """

    # Default color palette for bounding boxes (RGB)
    DEFAULT_COLORS = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (0, 255, 255),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 128, 0),  # Orange
        (128, 0, 255),  # Purple
    ]

    # Drawing configuration constants
    DEFAULT_LINE_WIDTH = 2
    DEFAULT_BOX_PADDING = 1
    DEFAULT_FONT_SIZE = 20

    def __init__(
        self,
        image_path: Optional[str] = None,
        image: Optional[Image.Image] = None,
        output_dir: str = "annotated_images"
    ) -> None:
        """
        Initialize the AnnotateImage class.

        Args:
            image_path: Path to the input image file
            image: PIL Image object (alternative to image_path)
            output_dir: Directory to save annotated images
        """
        if image_path:
            self.image = Image.open(image_path).convert('RGB')
        elif image:
            self.image = image.convert('RGB')
        else:
            raise ValueError("Either image_path or image must be provided")

        self.draw = ImageDraw.Draw(self.image)
        self.output_dir = output_dir
        self.font = self._get_default_font()
        self.label_colors: Dict[str, Tuple[int, int, int]] = {}

    @staticmethod
    def _get_default_font(size: int = DEFAULT_FONT_SIZE) -> ImageFont.FreeTypeFont:
        """Get the default font with fallback to PIL's default font."""
        # Try multiple common font names for cross-platform compatibility
        font_names = [
            "Arial.ttf",
            "DejaVuSans.ttf",
            "FreeSans.ttf",
            "Helvetica.ttf",
            "/System/Library/Fonts/Helvetica.ttc",  # macOS
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
            "C:\\Windows\\Fonts\\arial.ttf",  # Windows
        ]

        for font_name in font_names:
            try:
                return ImageFont.truetype(font_name, size)
            except (IOError, OSError):
                continue

        # Fall back to default font if no system font is found
        logger.warning("Could not load any system fonts, using default PIL font")
        return ImageFont.load_default()

    def _get_color_for_label(self, label: str) -> Tuple[int, int, int]:
        """Get a consistent color for a given label."""
        if label not in self.label_colors:
            # If we've used all colors, start reusing them
            color_index = len(self.label_colors) % len(self.DEFAULT_COLORS)
            self.label_colors[label] = self.DEFAULT_COLORS[color_index]
        return self.label_colors[label]

    def draw_boxes(
        self,
        boxes: List[Union[BoundingBox, Dict]],
        draw_labels: bool = True,
        line_width: int = DEFAULT_LINE_WIDTH,
        font_size: Optional[int] = None
    ) -> 'AnnotateImage':
        """
        Draw tight bounding boxes on the image with improved visualization.

        Args:
            boxes: List of BoundingBox objects or dictionaries with box data
            draw_labels: Whether to draw labels on the boxes
            line_width: Width of the box border
            font_size: Font size for labels (uses default if None)

        Returns:
            self for method chaining
        """
        font = self._get_default_font(font_size) if font_size else self.font

        # Get image dimensions for validation
        img_width, img_height = self.image.size
        logger.debug(f"Image size: {img_width}x{img_height}")

        for i, box in enumerate(boxes, 1):
            if isinstance(box, dict):
                box = BoundingBox(**box)

            # Use provided color or get one based on label
            color = box.color or self._get_color_for_label(box.label)

            # Ensure coordinates are within image bounds and properly ordered
            x1 = max(0, min(box.x_min, img_width - 1))
            y1 = max(0, min(box.y_min, img_height - 1))
            x2 = max(0, min(box.x_max, img_width - 1))
            y2 = max(0, min(box.y_max, img_height - 1))

            # Ensure min < max for both x and y
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            # Calculate box dimensions
            box_width = x2 - x1
            box_height = y2 - y1

            logger.debug(f"Box {i} ({box.label}): ({x1}, {y1}) - ({x2}, {y2}), "
                         f"size: {box_width}x{box_height}")

            # Skip if box is too small
            if box_width < 2 or box_height < 2:
                logger.warning(f"Skipping box {i} - too small: {box_width}x{box_height}")
                continue

            # Draw a tight box with a slight offset for better visibility
            box_padding = self.DEFAULT_BOX_PADDING
            self.draw.rectangle(
                [x1 - box_padding, y1 - box_padding,
                 x2 + box_padding, y2 + box_padding],
                outline=color,
                width=line_width,
                fill=None  # No fill
            )

            # Add a subtle inner highlight for better visibility
            if box_width > 10 and box_height > 10:  # Only for reasonably sized boxes
                self.draw.rectangle(
                    [x1 + 1, y1 + 1, x2 - 1, y2 - 1],
                    outline=tuple(min(255, c + 50) for c in color),  # Lighter inner line
                    width=1,
                    fill=None
                )

            if draw_labels and box.label:
                # Update the box coordinates to include the padding for label positioning
                box.x_min = x1 - box_padding
                box.y_min = y1 - box_padding
                box.x_max = x2 + box_padding
                box.y_max = y2 + box_padding
                self._draw_label(box, color, font)

        return self

    def _draw_label(
        self,
        box: BoundingBox,
        color: Tuple[int, int, int],
        font: ImageFont.FreeTypeFont
    ) -> None:
        """Draw a clean label with semi-transparent background for the bounding box."""
        # Prepare label text
        label_text = box.label
        if box.confidence > 0:
            label_text = f"{box.label} {box.confidence:.2f}"

        # Get image dimensions for bounds checking
        img_width, img_height = self.image.size

        # Calculate text size
        text_bbox = self.draw.textbbox((0, 0), label_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Calculate label position (above the box)
        padding = 2
        text_padding = 4

        # Position label at top of the box
        label_x = box.x_min
        label_y = box.y_min - text_height - 2 * padding

        # If label would go above image, position it inside the box at the top
        if label_y < 0:
            label_y = box.y_min + padding

        # Ensure label stays within image bounds
        label_x = max(padding, min(label_x, img_width - text_width - 2 * text_padding))
        label_y = max(padding, min(label_y, img_height - text_height - 2 * padding))

        # Draw background rectangle for better readability
        bg_rect = [
            max(0, label_x - text_padding),
            max(0, label_y),
            min(img_width, label_x + text_width + text_padding),
            min(img_height, label_y + text_height + padding)
        ]

        # Draw filled background
        self.draw.rectangle(
            bg_rect,
            fill=color,
            outline=color,
            width=1
        )

        # Calculate text position (centered in the background)
        text_x = label_x
        text_y = label_y

        # Draw text shadow for depth (slightly offset)
        shadow_offset = 1
        self.draw.text(
            (text_x + shadow_offset, text_y + shadow_offset),
            label_text,
            fill=(0, 0, 0),  # Black shadow
            font=font
        )

        # Draw main text
        self.draw.text(
            (text_x, text_y),
            label_text,
            fill='white',
            font=font,
            stroke_width=1,
            stroke_fill='black'  # Outline for better visibility
        )

    def save(self, output_path: Optional[str] = None) -> str:
        """
        Save the annotated image.

        Args:
            output_path: Full path to save the image. If None, generates a filename.

        Returns:
            Path to the saved image
        """
        if output_path is None:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            # Generate a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"annotated_{timestamp}.jpg")

        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the image
        self.image.save(output_path)
        return output_path

    def show(self) -> None:
        """Display the annotated image (for Jupyter notebooks)."""
        try:
            self.image.show()
        except Exception as e:
            logger.error(f"Could not display image: {e}")

    def get_image(self) -> Image.Image:
        """Get the annotated PIL Image object."""
        return self.image

    @classmethod
    def from_gemini_detection(
        cls,
        image_path: str,
        detection_results: Dict,
        output_dir: str = "annotated_images"
    ) -> 'AnnotateImage':
        """
        Create an AnnotateImage instance from Gemini detection results.

        Args:
            image_path: Path to the original image
            detection_results: Detection results from GeminiObjectDetection
            output_dir: Directory to save annotated images

        Returns:
            Annotated image instance
        """
        annotator = cls(image_path=image_path, output_dir=output_dir)

        if 'detected_objects' not in detection_results:
            return annotator

        boxes = []
        for obj in detection_results['detected_objects']:
            if 'bounding_box_absolute' in obj:
                box = BoundingBox(
                    x_min=obj['bounding_box_absolute'][0],
                    y_min=obj['bounding_box_absolute'][1],
                    x_max=obj['bounding_box_absolute'][2],
                    y_max=obj['bounding_box_absolute'][3],
                    label=obj.get('label', ''),
                    confidence=obj.get('confidence', 0.0)
                )
                boxes.append(box)

        return annotator.draw_boxes(boxes)
