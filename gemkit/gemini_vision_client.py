import os
import sys
import argparse
from io import BytesIO
from PIL import Image

from google import genai
from image_source import ImageSource, SourceConfig, OutputType, ImageResult

class GeminiVision:
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name = DEFAULT_MODEL):
        self.model_name = model_name
        self._create_client()

    def _create_client(self):
        """
        Initializes the Gemini client.
        The API key is read exclusively from the GEMINI_API_KEY environment variable.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("API key must be set in the GEMINI_API_KEY environment variable.")
            
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            # We don't raise here as the client check in edit_image will handle it.
            print(f"Error initializing Gemini client: {e}")
            self.client = None

    def _prepare_image(self, input_source: str) -> Image.Image:
        """
        Loads an image from various sources using ImageSource utility.

        Supports:
            - Image files (.jpg, .png, .bmp, .gif, .webp, .tiff)
            - URLs (automatic download)
            - Camera/Webcam (camera index as int)
            - Screenshots (use "screenshot" string)
            - Video files (extracts first frame)
            - Base64 strings
            - PIL Images
            - Numpy arrays

        Args:
            input_source: The image source (file path, URL, camera index, etc.)

        Returns:
            A PIL Image object.

        Raises:
            FileNotFoundError: If the input file is not found.
            ImageSourceError: If the source is invalid or unsupported.
        """
        config = SourceConfig(output_type=OutputType.PIL)
        source = ImageSource(input_source, config)
        result = source.get_image()
        source.close()
        return result.data

    def _ask_model(self, prompt: str, image: Image.Image):
        """
        Calls the Gemini API to generate content (edited image and text) based on
        the provided prompt and image.
        
        Args:
            prompt: The text instruction for the image edit.
            image: The PIL Image object to be edited.
            
        Returns:
            The raw response object from the Gemini API.
        """
        # 2. Call the Gemini API for image generation/editing
        return self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image]
        )

    def generate_text(self, input_source: str, prompt: str) -> str:
        """
        Loads an image from various sources, sends it with a text prompt to the
        Gemini model, and returns the generated text.

        Args:
            input_source: The image source (file path, URL, camera index, screenshot, etc.)
            prompt: The text instruction for the image analysis.

        Returns:
            A string containing the response text from the model.
        """
        if not self.client:
            return "Client not initialized due to previous error."

        try:
            # 1. Load the input image using ImageSource
            image = self._prepare_image(input_source)

            # 2. Call the Gemini API for image analysis
            response = self._ask_model(prompt, image)
            return response.text

        except FileNotFoundError as e:
            # Explicitly catch FileNotFoundError for clearer messages
            return f"Error: Input file not found - {e}"
        except Exception as e:
            # Catch all other exceptions (e.g., ImageSource errors, connection issues)
            return f"An unexpected error occurred: {e}"

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Process images using Google Gemini AI.',
        epilog='Examples:\n'
               '  %(prog)s -i photo.jpg -p "Describe the image"\n'
               '  %(prog)s -i https://example.com/image.jpg -p "What is in this image?"\n'
               '  %(prog)s -i screenshot -p "Analyze this screenshot"\n'
               '  %(prog)s -i 0 -p "Describe what the camera sees" (use camera)\n',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('-i', '--image', required=True,
                       help='Input image source: file path, URL, "screenshot", or camera index (0, 1, etc.)')
    parser.add_argument('-p', '--prompt', default='Describe the image',
                        help='Text prompt for image processing (default: "Describe the image")')
    parser.add_argument('--model', default=GeminiVision.DEFAULT_MODEL,
                       help=f'Model to use (default: {GeminiVision.DEFAULT_MODEL})')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    vlm = GeminiVision(model_name=args.model)

    # Handle camera index (convert string to int if it's a digit)
    image_source = args.image
    if args.image.isdigit():
        image_source = int(args.image)

    text = vlm.generate_text(input_source=image_source, prompt=args.prompt)
    print(text)

        
