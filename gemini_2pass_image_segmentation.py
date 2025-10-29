"""
Two-Pass Gemini Image Segmentation Module

First pass: Detect all prominent objects in the image using single-pass algorithm.
Second pass: Refine segmentation for each detected object individually using single-pass algorithm.
"""

import os
import json
import argparse
from typing import Optional, List, Dict, Any, Tuple
from PIL import Image
import numpy as np
import base64
import io

from google.genai import types
from gemini_image_segmentation import GeminiImageSegmentation

class GeminiTwoPassImageSegmentation:
    """Two-pass segmentation: coarse detection followed by refined per-object segmentation."""
    
    def __init__(self, model_name: str = 'gemini-2.5-flash') -> None:
        """
        Initialize using the single-pass segmentation engine.
        
        Args:
            model_name: Gemini model to use for segmentation.
        """
        # Use single-pass algorithm as the core engine
        self.segmenter = GeminiImageSegmentation(model_name=model_name)
        self.model_name = model_name

    def _extract_bounding_box(self, box_2d: List[float], image_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Convert normalized bounding box to pixel coordinates.
        
        Args:
            box_2d: Normalized coordinates [y0, x0, y1, x1] in range [0, 1000].
            image_size: (width, height) of the image.
            
        Returns:
            Tuple of (x0, y0, x1, y1) in pixel coordinates.
        """
        w, h = image_size
        y0 = int(box_2d[0] / 1000 * h)
        x0 = int(box_2d[1] / 1000 * w)
        y1 = int(box_2d[2] / 1000 * h)
        x1 = int(box_2d[3] / 1000 * w)
        return x0, y0, x1, y1

    def _first_pass_segment(
        self, 
        image_path: str, 
        objects_to_segment: Optional[List[str]] = None,
        output_dir: str = "segmentation_outputs"
    ) -> Tuple[Dict[str, Any], Tuple[int, int]]:
        """
        First pass: Detect all prominent objects using single-pass algorithm.
        
        Args:
            image_path: Path to the image file.
            objects_to_segment: Optional list of specific objects to detect.
            output_dir: Directory for temporary outputs.
            
        Returns:
            Tuple of (result dictionary from single-pass segmentation, resized image dimensions)
        """
        print(f"[Pass 1] Detecting objects using single-pass algorithm...")
        
        # Load original image to get dimensions
        original_image = Image.open(image_path)
        original_size = original_image.size
        
        # The single-pass algorithm will resize, so we need to track the resized dimensions
        resized_image = original_image.copy()
        resized_image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        resized_size = resized_image.size
        
        # Use single-pass segmentation for first pass
        pass1_output = os.path.join(output_dir, "pass1")
        result = self.segmenter.segment_image(
            image_path=image_path,
            objects_to_segment=objects_to_segment,
            output_dir=pass1_output
        )
        
        if isinstance(result, str):  # Error occurred
            return {"error": result}, resized_size
        
        print(f"[Pass 1] Detected {result.get('total_masks', 0)} objects")
        print(f"[Pass 1] Original size: {original_size}, Resized size: {resized_size}")
        
        return result, resized_size

    def _second_pass_refine_batch(
        self, 
        original_image_path: str,
        first_pass_result: Dict[str, Any],
        resized_dimensions: Tuple[int, int],
        output_dir: str = "segmentation_outputs",
        padding: int = 20
    ) -> Dict[str, Any]:
        """
        Second pass: Refine segmentation for all detected objects in a single batched API call.
        Uses original image for cropping, not the resized version.
        
        Args:
            original_image_path: Path to the original image.
            first_pass_result: Results from first pass segmentation.
            resized_dimensions: (width, height) of the resized image used in first pass.
            output_dir: Directory for outputs.
            padding: Pixels to add around bounding box for context.
            
        Returns:
            Dictionary with refined segmentation results.
        """
        masks = first_pass_result.get('masks', [])
        if not masks:
            return {"error": "No masks from first pass to refine"}
        
        # Load ORIGINAL image (not resized)
        original_image = Image.open(original_image_path)
        original_size = original_image.size
        resized_width, resized_height = resized_dimensions
        
        # Calculate scale factors to convert bounding boxes from resized to original coordinates
        scale_x = original_size[0] / resized_width
        scale_y = original_size[1] / resized_height
        
        print(f"[Pass 2] Original image size: {original_size}")
        print(f"[Pass 2] Resized image size: {resized_dimensions}")
        print(f"[Pass 2] Scale factors: x={scale_x:.3f}, y={scale_y:.3f}")
        
        refined_results = {
            "total_refined": 0,
            "output_directory": output_dir,
            "refined_masks": []
        }
        
        pass2_output = os.path.join(output_dir, "pass2")
        os.makedirs(pass2_output, exist_ok=True)
        
        print(f"[Pass 2] Preparing batch refinement for {len(masks)} objects...")
        
        # Prepare all crops from ORIGINAL image
        crop_data = []
        for i, mask_info in enumerate(masks):
            try:
                label = mask_info.get('label', f'object_{i}')
                bbox = mask_info.get('bounding_box')
                
                if not bbox:
                    continue
                
                # Bounding box is in resized image coordinates, convert to original
                x0_resized, y0_resized, x1_resized, y1_resized = bbox
                
                x0 = int(x0_resized * scale_x)
                y0 = int(y0_resized * scale_y)
                x1 = int(x1_resized * scale_x)
                y1 = int(y1_resized * scale_y)
                
                # Add padding and clip to original image bounds
                x0 = max(0, x0 - padding)
                y0 = max(0, y0 - padding)
                x1 = min(original_image.width, x1 + padding)
                y1 = min(original_image.height, y1 + padding)
                
                if x0 >= x1 or y0 >= y1:
                    continue
                
                # Crop from ORIGINAL image
                subimage = original_image.crop((x0, y0, x1, y1))
                
                # Resize crop for API efficiency (optional, but helps with token limits)
                max_crop_size = 512
                if subimage.width > max_crop_size or subimage.height > max_crop_size:
                    subimage.thumbnail([max_crop_size, max_crop_size], Image.Resampling.LANCZOS)
                
                crop_data.append({
                    'index': i,
                    'label': label,
                    'subimage': subimage,
                    'offset': (x0, y0),
                    'original_bbox': (x0, y0, x1, y1),
                    'size': (x1 - x0, y1 - y0)
                })
                
                print(f"[Pass 2] Crop {i}: '{label}' at original coords ({x0},{y0})-({x1},{y1}), size {x1-x0}x{y1-y0}")
                
            except Exception as e:
                print(f"[Pass 2] Error preparing crop {i}: {e}")
                continue
        
        if not crop_data:
            return {"error": "No valid crops to refine"}
        
        print(f"[Pass 2] Sending {len(crop_data)} crops in single batched API call...")
        
        # Build batched prompt with all crops
        prompt_parts = [
            "You are given multiple cropped images, each containing a specific object. "
            "For each image, provide the segmentation mask for the main object. "
            "Output a JSON list where each entry corresponds to one input image (in order) "
            "and contains the 2D bounding box in key 'box_2d', the segmentation mask in key 'mask', "
            "and the text label in key 'label'.\n\n"
        ]
        
        for i, crop in enumerate(crop_data):
            prompt_parts.append(f"Image {i+1}: Segment the '{crop['label']}'.\n")
        
        prompt = "".join(prompt_parts)
        
        # Prepare content list: prompt + all images
        contents = [prompt]
        for crop in crop_data:
            contents.append(crop['subimage'])
        
        try:
            # Single batched API call with all subimages
            
            config = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
            
            response = self.segmenter.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Parse response
            items = json.loads(self.segmenter.parse_json(response.text))
            
            if not isinstance(items, list):
                return {"error": "Expected a list of segmentation masks from the model"}
            
            print(f"[Pass 2] Received {len(items)} refined masks from batched call")
            
            # Process each refined mask
            for i, (crop_info, item) in enumerate(zip(crop_data, items)):
                try:
                    
                    label = crop_info['label']
                    x_offset, y_offset = crop_info['offset']
                    orig_x0, orig_y0, orig_x1, orig_y1 = crop_info['original_bbox']
                    
                    # Get bounding box from refined result
                    box = item.get("box_2d", [])
                    if not box:
                        continue
                    
                    crop_width, crop_height = crop_info['size']
                    
                    # Convert to pixel coordinates in crop space
                    y0 = int(box[0] / 1000 * crop_height)
                    x0 = int(box[1] / 1000 * crop_width)
                    y1 = int(box[2] / 1000 * crop_height)
                    x1 = int(box[3] / 1000 * crop_width)
                    
                    if y0 >= y1 or x0 >= x1:
                        continue
                    
                    # Process mask
                    png_str = item.get("mask", "")
                    if not png_str.startswith("data:image/png;base64,"):
                        continue
                    
                    png_str = png_str.removeprefix("data:image/png;base64,")
                    mask_data = base64.b64decode(png_str)
                    mask = Image.open(io.BytesIO(mask_data))
                    mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
                    
                    # Save refined mask
                    safe_label = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in label)
                    mask_filename = f"{safe_label}_{crop_info['index']}_refined_mask.png"
                    mask_path = os.path.join(pass2_output, mask_filename)
                    mask.save(mask_path)
                    
                    # Create overlay on ORIGINAL image
                    mask_array = np.array(mask)
                    overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
                    
                    from PIL import ImageDraw
                    overlay_draw = ImageDraw.Draw(overlay)
                    
                    for y in range(y1 - y0):
                        for x in range(x1 - x0):
                            if mask_array[y, x] > 128:
                                orig_x = x_offset + x0 + x
                                orig_y = y_offset + y0 + y
                                overlay_draw.point((orig_x, orig_y), fill=(255, 255, 255, 200))
                    
                    overlay_filename = f"{safe_label}_{crop_info['index']}_refined_overlay.png"
                    overlay_path = os.path.join(pass2_output, overlay_filename)
                    Image.alpha_composite(original_image.convert('RGBA'), overlay).save(overlay_path)
                    
                    refined_results['refined_masks'].append({
                        "label": label,
                        "bounding_box_original": [
                            x_offset + x0,
                            y_offset + y0,
                            x_offset + x1,
                            y_offset + y1
                        ],
                        "crop_offset": [x_offset, y_offset],
                        "mask_file": mask_filename,
                        "overlay_file": overlay_filename
                    })
                    
                    refined_results['total_refined'] += 1
                    print(f"[Pass 2] Successfully refined '{label}' at original coords")
                    
                except Exception as e:
                    print(f"[Pass 2] Error processing refined mask {i}: {e}")
                    continue
            
            return refined_results
            
        except Exception as e:
            return {"error": f"Batched API call failed: {e}"}

    def segment_two_pass(
        self,
        image_path: str,
        objects_to_segment: Optional[List[str]] = None,
        output_dir: str = "segmentation_outputs",
        enable_refinement: bool = True,
        padding: int = 20
    ) -> Dict[str, Any]:
        """
        Perform two-pass segmentation using single-pass algorithm for both passes.
        
        Args:
            image_path: Path to image file.
            objects_to_segment: Optional list of specific objects to detect.
            output_dir: Output directory for results.
            enable_refinement: If True, performs second pass refinement.
            padding: Pixels to add around crops in second pass.
            
        Returns:
            Dictionary containing:
                - pass1_detections: Number of objects detected in first pass
                - pass2_refined: Number of objects refined in second pass
                - output_directory: Path to output directory
                - pass1_result: Full result from first pass
                - pass2_result: Full result from second pass (if enabled)
                - error: Error message if operation failed
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}"}
        
        try:
            # Pass 1: Coarse detection using single-pass algorithm
            pass1_result, resized_dims = self._first_pass_segment(
                image_path=image_path,
                objects_to_segment=objects_to_segment,
                output_dir=output_dir
            )
            
            if "error" in pass1_result:
                return pass1_result
            
            num_detections = pass1_result.get('total_masks', 0)
            
            if num_detections == 0:
                return {"error": "No objects detected in first pass"}
            
            # Pass 2: Refined segmentation (optional)
            if enable_refinement:
                pass2_result = self._second_pass_refine_batch(
                    original_image_path=image_path,
                    first_pass_result=pass1_result,
                    resized_dimensions=resized_dims,
                    output_dir=output_dir,
                    padding=padding
                )
                
                return {
                    "pass1_detections": num_detections,
                    "pass2_refined": pass2_result.get('total_refined', 0),
                    "output_directory": output_dir,
                    "pass1_result": pass1_result,
                    "pass2_result": pass2_result
                }
            else:
                return {
                    "pass1_detections": num_detections,
                    "output_directory": output_dir,
                    "pass1_result": pass1_result
                }
            
        except Exception as e:
            return {"error": f"Unexpected error: {e}"}


def main() -> None:
    """CLI entry point for two-pass Gemini image segmentation."""
    parser = argparse.ArgumentParser(description="Two-Pass Gemini Image Segmentation")
    parser.add_argument("-i", "--image_path", required=True, help="Path to image file")
    parser.add_argument("--objects", help="Comma-separated objects to segment")
    parser.add_argument("--output-dir", default="segmentation_outputs", 
                       help="Output directory (default: segmentation_outputs)")
    parser.add_argument("--no-refinement", action="store_true",
                       help="Skip second pass refinement")
    parser.add_argument("--padding", type=int, default=20,
                       help="Padding around crops in second pass (default: 20)")
    
    args = parser.parse_args()
    
    # Parse objects list
    objects_list = [obj.strip() for obj in args.objects.split(',')] if args.objects else None
    
    # Run two-pass segmentation
    segmenter = GeminiTwoPassImageSegmentation()
    result = segmenter.segment_two_pass(
        image_path=args.image_path,
        objects_to_segment=objects_list,
        output_dir=args.output_dir,
        enable_refinement=not args.no_refinement,
        padding=args.padding
    )
    
    print("\n" + "="*60)
    print("TWO-PASS SEGMENTATION RESULTS")
    print("="*60)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
