import argparse
import base64
import io
import json
import os

from genai.errors import APIError
from google import genai
from PIL import Image

from image_source import ImageSource


class GeminiImageSegmentation:
    """
    A class to handle image segmentation using the Google GenAI SDK (Gemini model).
    Generates segmentation masks with bounding boxes and labels.
    """
    def __init__(self, model_name='gemini-1.5-flash'):
        """Initializes the Gemini client and sets the model name."""
        self.model_name = model_name
        self.client = None
        try:
            self.client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")

    def segment_image(self, image: Image.Image, prompt: str = None, objects_to_segment: list[str] = None, output_dir: str = "segmentation_outputs") -> dict | str:
        """
        Orchestrates the image segmentation process.
        """
        if not self.client:
            return "Error: Gemini client is not initialized."

        try:
            image.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
            print(f"Image loaded successfully. Size: {image.size}")

            final_prompt = self._create_prompt(prompt, objects_to_segment)
            
            print(f"Calling Gemini model: {self.model_name}...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[final_prompt, image]
            )
            
            items = self.parse_json(response.text)
            if not isinstance(items, list):
                return "Error: Expected a list of segmentation masks from the model."

            os.makedirs(output_dir, exist_ok=True)
            results = {"output_directory": output_dir, "masks": []}
            
            for i, item in enumerate(items):
                mask_data = self._process_item(item, image, output_dir, i)
                if mask_data:
                    results["masks"].append(mask_data)
            
            return results
            
        except APIError as e:
            return f"Gemini API Error (SDK): {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def _create_prompt(self, prompt: str = None, objects_to_segment: list[str] = None) -> str:
        """Creates the segmentation prompt based on user input."""
        if prompt:
            return prompt
        if objects_to_segment:
            object_list_str = ', '.join(objects_to_segment)
            return (
                f"Give the segmentation masks for the following objects: {object_list_str}. "
                "Output a JSON list of segmentation masks where each entry contains the 2D "
                "bounding box in the key 'box_2d', the segmentation mask in key 'mask', and "
                "the text label in the key 'label'. Use descriptive labels."
            )
        return (
            "Give the segmentation masks for all prominent objects in the image. "
            "Output a JSON list of segmentation masks where each entry contains the 2D "
            "bounding box in the key 'box_2d', the segmentation mask in key 'mask', and "
            "the text label in the key 'label'. Use descriptive labels."
        )

    def _extract_and_validate_data(self, item: dict, image_size: tuple[int, int], index: int) -> tuple | None:
        """Extracts and validates data from a single response item."""
        box = item.get("box_2d")
        png_str = item.get("mask")
        label = item.get("label", f"object_{index}")

        if not box or not png_str or not png_str.startswith("data:image/png;base64,"):
            print(f"Skipping item {index} ('{label}'): missing or invalid data.")
            return None

        im_width, im_height = image_size
        y0, x0, y1, x1 = [int(c / 1000 * s) for c, s in zip(box, [im_height, im_width, im_height, im_width])]
        
        if y0 >= y1 or x0 >= x1:
            print(f"Skipping item {index} ('{label}'): invalid bounding box.")
            return None
            
        return x0, y0, x1, y1, png_str, label

    def _create_and_save_images(self, validated_data: tuple, original_image: Image.Image, output_dir: str, index: int) -> dict:
        """Creates and saves the mask and overlay images, returning metadata."""
        x0, y0, x1, y1, png_str, label = validated_data

        # Decode and resize the mask
        mask_data = base64.b64decode(png_str.removeprefix("data:image/png;base64,"))
        mask = Image.open(io.BytesIO(mask_data)).resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)

        # Prepare file paths
        safe_label = "".join(c for c in label if c.isalnum() or c in (' ', '_')).replace(' ', '_')
        mask_filename = f"{safe_label}_{index}_mask.png"
        overlay_filename = f"{safe_label}_{index}_overlay.png"
        mask_path = os.path.join(output_dir, mask_filename)
        overlay_path = os.path.join(output_dir, overlay_filename)

        # Save the mask file
        mask.save(mask_path)

        # Create and save the overlay image
        overlay = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
        colored_mask = Image.new('RGBA', mask.size, (255, 255, 255, 180))
        overlay.paste(colored_mask, (x0, y0), mask)
        
        composite = Image.alpha_composite(original_image.convert('RGBA'), overlay)
        composite.convert("RGB").save(overlay_path)
        
        print(f"Saved mask and overlay for '{label}'")
        return {
            "label": label,
            "bounding_box": [x0, y0, x1, y1],
            "mask_file": mask_filename,
            "overlay_file": overlay_filename
        }

    def _process_item(self, item: dict, original_image: Image.Image, output_dir: str, index: int) -> dict | None:
        """Orchestrates the processing of a single item from the model's response."""
        try:
            validated_data = self._extract_and_validate_data(item, original_image.size, index)
            if not validated_data:
                return None
            
            return self._create_and_save_images(validated_data, original_image, output_dir, index)
        except Exception as e:
            print(f"Error processing item {index}: {e}")
            return None

    @staticmethod
    def parse_json(json_output: str) -> list | None:
        """
        Parses a JSON list from a string, removing markdown fencing if present.
        """
        if "```json" in json_output:
            json_output = json_output.split("```json")[1].split("```")[0]
        
        try:
            return json.loads(json_output)
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from the response.")
            return None

def main():
    """Parses command-line arguments and runs the segmentation process."""
    parser = argparse.ArgumentParser(description="Run Gemini Image Segmentation on a local image file or URL.")
    parser.add_argument("-s", "--source", required=True, type=str, help="The image source (local file path or URL).")
    parser.add_argument("--prompt", type=str, help="Custom prompt for segmentation. Overrides --objects.")
    parser.add_argument("--objects", type=str, help="Comma-separated list of objects to segment (e.g., 'cat,dog').")
    parser.add_argument("--output-dir", default="segmentation_outputs", help="Directory to save outputs.")
    args = parser.parse_args()

    if not os.getenv('GEMINI_API_KEY'):
        print("Error: The 'GEMINI_API_KEY' environment variable is not set.")
        return

    try:
        image_loader = ImageSource(args.source)
        pil_image = image_loader.get_image().data
    except Exception as e:
        print(f"Error loading image source: {e}")
        return

    objects_list = [obj.strip() for obj in args.objects.split(',')] if args.objects else None
    
    segmenter = GeminiImageSegmentation()
    result = segmenter.segment_image(
        image=pil_image, 
        prompt=args.prompt, 
        objects_to_segment=objects_list,
        output_dir=args.output_dir
    )

    print("\n--- Gemini Segmentation Result ---")
    if isinstance(result, dict):
        print(json.dumps(result, indent=2))
    else:
        print(result)
    print("----------------------------------")

if __name__ == "__main__":
    main()
