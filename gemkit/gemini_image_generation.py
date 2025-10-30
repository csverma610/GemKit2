import argparse
import os
from io import BytesIO

from PIL import Image
from google import genai
from google.genai import types

class GeminiImageGenerator:
    """
    Generates images from text prompts using the Google Gemini API.

    This class provides a high-level interface for sending a text prompt to the
    Gemini API and saving the generated image to a file.
    """
    DEFAULT_MODEL = "gemini-2.5-flash-image"

    def __init__(self, model_name=DEFAULT_MODEL):
        """
        Initializes the GeminiImageGenerator.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-2.5-flash-image".
        """
        self.model_name = model_name
        self._create_client()

    def generate_image(self, prompt: str, output_filename: str = "new_image.png") -> dict:
        """
        Generates an image based on a text prompt.

        Args:
            prompt (str): The text prompt describing the image to generate.
            output_filename (str, optional): The path to save the generated image.
                                             Defaults to "new_image.png".

        Returns:
            dict: A dictionary containing the response text from the model and the
                  filepath where the generated image was saved.
        """
        if not self.client:
            return {"text": "Client not initialized due to previous error.", "filepath": None}

        try:
            # 1. Call the Gemini API for image generation/editing using the new function
            response = self._generate_response(prompt)

            # 2. Process the response and return the result
            return self._process_response(response, output_filename)

        except Exception as e:
            # Catch all exceptions (e.g., API errors, connection issues)
            return {"text": f"An unexpected error occurred: {e}", "filepath": None}

    def _create_client(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key must be set in the GEMINI_API_KEY environment variable.")

        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            # We don't raise here as the client check in generate_image will handle it.
            print(f"Error initializing Gemini client: {e}")
            self.client = None

    def _generate_response(self, prompt: str):
        """
        Calls the Gemini API to generate content (image and text) based on
        the provided prompt.

        Args:
            prompt: The text instruction for image generation.

        Returns:
            The raw response object from the Gemini API.
        """
        # Define the image generation configuration, including the desired aspect ratio
        image_config = types.ImageConfig(
            aspect_ratio="16:9",  # Supported values include "1:1", "3:4", "4:3", "9:16", "16:9", "21:9", etc.
        )

        # Define the overall content generation configuration
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            image_config=image_config,
        )
        
        return self.client.models.generate_content(
            model=self.model_name,
            contents=prompt, 
            config=generate_content_config,
        )

    def _process_response(self, response, output_filename: str) -> dict:
        """
        Processes the API response candidate to extract generated text and image data,
        saving the image to the specified output file.
        """
        generated_text = None
        image_saved = False

        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            # Handle case where response is valid but contains no candidates/parts (e.g., blocked content)
            return {
                "text": "The model returned no content. This may be due to safety filters or an empty response.",
                "filepath": None
            }

        #  Process the response candidates
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                generated_text = part.text

            elif part.inline_data is not None:
                generated_image = Image.open(BytesIO(part.inline_data.data))
                generated_image.save(output_filename)
                image_saved = True

        return {
            "text": generated_text,
            "filepath": output_filename if image_saved else None
        }

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Generate images using Gemini AI. Requires GEMINI_API_KEY environment variable to be set.'
    )
    
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        required=True,
        help='Text prompt describing the image to generate (required)'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='generated_image.png',
        help='Output file path (default: generated_image.png)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=GeminiImageGenerator.DEFAULT_MODEL,
        help=f'Model to use (default: {GeminiImageGenerator.DEFAULT_MODEL})'
    )
    
    return parser.parse_args()


def main():
    """Main function to handle command line execution."""
    args = parse_arguments()
    
    try:
        print(f"Generating image with prompt: {args.prompt}")
        print(f"Using model: {args.model}")
        
        generator = GeminiImageGenerator(model_name=args.model)
        result = generator.generate_image(
            prompt=args.prompt,
            output_filename=args.output
        )

        print("\n--- Generation Results ---")
        if result.get("filepath"):
            print(f"✅ Image successfully generated and saved to: {result['filepath']}")
        if result.get("text"):
            print(f"\nModel Response:\n{result['text']}")
        if not result.get("filepath"):
            print(f"❌ Image generation failed. Reason: {result.get('text', 'Unknown error')}")
            
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Fatal Error during execution: {e}")


if __name__ == "__main__":
    main()
