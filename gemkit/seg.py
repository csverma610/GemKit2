"""
Gemini Image Segmentation Module

Provides image segmentation capabilities using Google's Gemini API.
Generates segmentation masks with bounding boxes and visual overlays.
"""

import base64
import os
import json
import argparse
import io
from typing import Optional, List, Dict, Union, Any
import numpy as np
from google import genai
from google.genai import types
from google.genai.errors import APIError
from PIL import Image, ImageDraw

class GeminiImageSegmentation:
    """
    Performs image segmentation using the Google Gemini API.

    This class takes an image and an optional prompt, and then uses the Gemini
    API to generate segmentation masks for the objects in the image. The masks,
    along with their bounding boxes and labels, are saved to an output directory.
    """

    def __init__(self, model_name: str = 'gemini-2.5-flash') -> None:
        """
        Initializes the GeminiImageSegmentation instance.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to 'gemini-2.5-flash'.
        """
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError(
                "API key required. Provide via api_key parameter or GEMINI_API_KEY environment variable."
            )
        self.model_name = model_name
        self.client = genai.Client(api_key=self.api_key)

    def _build_prompt(self, objects_to_segment: Optional[List[str]] = None) -> str:
        """
        Build segmentation prompt based on target objects.
        
        Args:
            objects_to_segment: Specific objects to segment, or None for all objects.
            
        Returns:
            Formatted prompt string for the model.
        """
        base_instruction = (
            "Output a JSON list of segmentation masks where each entry contains "
            "the 2D bounding box in 'box_2d', the segmentation mask in 'mask', "
            "and the text label in 'label'."
        )
        
        if objects_to_segment:
            objects = ', '.join(objects_to_segment)
            return f"Segment these objects: {objects}. {base_instruction}"
        return f"Segment all prominent objects in the image. {base_instruction}"

    def _parse_json_response(self, text: str) -> str:
        """
        Extract JSON from response, handling markdown fencing.
        
        Args:
            text: Raw response text from model.
            
        Returns:
            Cleaned JSON string.
        """
        lines = text.strip().splitlines()
        
        # Remove markdown code fences if present
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                text = "\n".join(lines[i+1:]).split("```")[0]
                break
        
        text = text.strip()
        
        # Extract complete JSON array/object if there's trailing content
        if text.startswith('['):
            depth = 0
            for i, char in enumerate(text):
                if char == '[': depth += 1
                elif char == ']': 
                    depth -= 1
                    if depth == 0: return text[:i+1]
        elif text.startswith('{'):
            depth = 0
            for i, char in enumerate(text):
                if char == '{': depth += 1
                elif char == '}': 
                    depth -= 1
                    if depth == 0: return text[:i+1]
        
        return text

    def _decode_mask(self, mask_str: str) -> Optional[Image.Image]:
        """
        Decode base64 mask string to PIL Image.
        
        Args:
            mask_str: Base64-encoded PNG mask with data URI prefix.
            
        Returns:
            PIL Image object or None if decoding fails.
        """
        if not mask_str.startswith("data:image/png;base64,"):
            return None
        
        png_data = base64.b64decode(mask_str.removeprefix("data:image/png;base64,"))
        return Image.open(io.BytesIO(png_data))

    def _save_mask_outputs(
        self, image: Image.Image, mask_data: Dict[str, Any], index: int, output_dir: str
    ) -> Optional[Dict[str, Any]]:
        """
        Save individual mask and overlay visualization.
        
        Args:
            image: Original PIL Image.
            mask_data: Dictionary containing box_2d, mask, and label.
            index: Mask index for unique naming.
            output_dir: Directory to save outputs.
            
        Returns:
            Dictionary with mask metadata or None if processing fails.
        """
        label = mask_data.get("label", f"object_{index}")
        safe_label = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in label)
        
        # Get bounding box
        box = mask_data["box_2d"]
        h, w = image.size[1], image.size[0]
        y0, x0 = int(box[0] / 1000 * h), int(box[1] / 1000 * w)
        y1, x1 = int(box[2] / 1000 * h), int(box[3] / 1000 * w)
        
        if y0 >= y1 or x0 >= x1:
            return None
        
        # Decode and resize mask
        mask = self._decode_mask(mask_data.get("mask", ""))
        if not mask:
            return None
        
        mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
        mask_array = np.array(mask)
        
        # Save mask
        mask_path = os.path.join(output_dir, f"{safe_label}_{index}_mask.png")
        mask.save(mask_path)
        
        # Create overlay
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        
        for y in range(y0, y1):
            for x in range(x0, x1):
                if mask_array[y - y0, x - x0] > 128:
                    overlay_draw.point((x, y), fill=(255, 255, 255, 200))
        
        overlay_path = os.path.join(output_dir, f"{safe_label}_{index}_overlay.png")
        Image.alpha_composite(image.convert('RGBA'), overlay).save(overlay_path)
        
        return {
            "label": label,
            "bounding_box": [x0, y0, x1, y1],
            "mask_file": os.path.basename(mask_path),
            "overlay_file": os.path.basename(overlay_path)
        }

    def segment_image(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        objects_to_segment: Optional[List[str]] = None,
        output_dir: str = "segmentation_outputs",
    ) -> Dict[str, Any]:
        """
        Performs image segmentation on a given image.

        Args:
            image_path (str): The path to the image file.
            prompt (Optional[str], optional): A custom prompt to guide the segmentation.
                                    If provided, `objects_to_segment` is ignored.
            objects_to_segment (list[str], optional): A list of specific objects to segment.
            output_dir (str, optional): The directory to save the output files.
                                        Defaults to "segmentation_outputs".

        Returns:
            dict: A dictionary containing the segmentation results, or an
                  error message if the operation fails.
        """
        if not self.client:
            return {"error": "Gemini client not initialized."}
        
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        try:
            # Load and resize image
            image = Image.open(image_path)
            image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            print(f"Loaded image: {image.size}")
            
            # Generate prompt and call API
            final_prompt = prompt or self._build_prompt(objects_to_segment)
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            
            print(f"Calling {self.model_name}...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[final_prompt, image],
                config=config
            )
            
            # Parse response
            items = json.loads(self._parse_json_response(response.text))
            if not isinstance(items, list):
                return {"error": "Invalid response format from model"}
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Process each mask
            results = {"total_masks": 0, "output_directory": output_dir, "masks": []}
            
            for i, item in enumerate(items):
                try:
                    mask_result = self._save_mask_outputs(image, item, i, output_dir)
                    if mask_result:
                        results["masks"].append(mask_result)
                        results["total_masks"] += 1
                        print(f"Saved: {mask_result['label']}")
                except Exception as e:
                    print(f"Skipped mask {i}: {e}")
            
            return results
            
        except APIError as e:
            return {"error": f"API Error: {e}"}
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}


def main() -> None:
    """CLI entry point for Gemini image segmentation."""
    parser = argparse.ArgumentParser(description="Gemini Image Segmentation")
    parser.add_argument("-i", "--image_path", required=True, help="Path to image file")
    parser.add_argument("--prompt", help="Custom segmentation prompt")
    parser.add_argument("--objects", help="Comma-separated objects to segment")
    parser.add_argument("--output-dir", default="segmentation_outputs", 
                       help="Output directory (default: segmentation_outputs)")
    
    args = parser.parse_args()
    
    # Parse objects list
    objects_list = [obj.strip() for obj in args.objects.split(',')] if args.objects else None
    
    # Run segmentation
    segmenter = GeminiImageSegmentation()
    result = segmenter.segment_image(
        image_path=args.image_path,
        prompt=args.prompt,
        objects_to_segment=objects_list,
        output_dir=args.output_dir
    )
    
    print("\n" + "="*50)
    print("SEGMENTATION RESULTS")
    print("="*50)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
