import argparse
import os
from io import BytesIO

from google import genai
from PIL import Image

from image_source import ImageSource

class GeminiImageEditor:
    """
    Edits images using the Google Gemini API.

    This class provides a high-level interface for loading an image, sending it to
    the Gemini API with a text prompt, and saving the edited image.
    """
    DEFAULT_MODEL = "gemini-2.5-flash-image"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initializes the GeminiImageEditor.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-2.5-flash-image".
        """
        self.model_name = model_name
        self._create_client()

    def edit_image(self, source: str, prompt: str, output_filename: str = "edited_image.png") -> dict:
        """
        Loads an image, sends it to the Gemini model for editing, and saves the result.

        Args:
            source (str): The path or URL to the input image.
            prompt (str): The text prompt describing the desired edit.
            output_filename (str, optional): The path to save the edited image.
                                             Defaults to "edited_image.png".

        Returns:
            dict: A dictionary containing the response text from the model and the
                  filepath where the edited image was saved.
        """
        if not self.client:
            return {"text": "Client not initialized due to previous error.", "filepath": None}
        
        try:
            image = self._load_image(source)
            response = self._generate_response(prompt, image)
            return self._process_response(response, output_filename)

        except FileNotFoundError:
            return {"text": f"Error: Input source not found at {source}", "filepath": None}
        except Exception as e:
            return {"text": f"An unexpected error occurred while loading or editing the image: {e}", "filepath": None}

    def _create_client(self):
        """
        Initializes the Gemini client with the API key from environment variables.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("API key must be set in the GEMINI_API_KEY environment variable.")
            
        try:
            self.client = genai.Client(api_key=api_key)
        except Exception as e:
            print(f"Error initializing Gemini client: {e}")
            self.client = None

    def _load_image(self, source: str) -> Image.Image:
        """Loads an image from a given source using ImageSource."""
        print(f"Loading image from '{source}'...")
        image_loader = ImageSource(source)
        return image_loader.get_image().data

    def _generate_response(self, prompt: str, image: Image.Image):
        """
        Calls the Gemini API to generate content based on the provided prompt and image.
        """
        print(f"Sending request to Gemini model '{self.model_name}'...")
        return self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, image]
        )

    def _process_response(self, response, output_filename: str) -> dict:
        """
        Processes the API response to extract text and save the generated image.
        """
        if not (response.candidates and response.candidates[0].content and response.candidates[0].content.parts):
            return {
                "text": "The model returned no content. This may be due to safety filters.",
                "filepath": None
            }
            
        generated_text = None
        generated_image_path = None
        
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                generated_text = part.text
            
            elif part.inline_data is not None:
                generated_image = Image.open(BytesIO(part.inline_data.data))
                generated_image.save(output_filename)
                generated_image_path = output_filename
        
        return {
            "text": generated_text,
            "filepath": generated_image_path
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Edit images using Gemini AI. Requires GEMINI_API_KEY environment variable to be set.'
    )
    
    parser.add_argument(
        '-i', '--image_source',
        required=True,
        type=str,
        help='Path or URL to the input image'
    )
    
    parser.add_argument(
        '-p', '--prompt',
        required=True,
        type=str,
        help='Text prompt describing the desired image edit'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='edited_image.png',
        help='Output file path (default: edited_image.png)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=GeminiImageEditor.DEFAULT_MODEL,
        help=f'Model to use (default: {GeminiImageEditor.DEFAULT_MODEL})'
    )
    
    return parser.parse_args()


def main():
    """Main function to handle command line execution."""
    args = parse_arguments()
    
    try:
        editor = GeminiImageEditor(model_name=args.model)
        
        result = editor.edit_image(
            source=args.image_source,
            prompt=args.prompt,
            output_filename=args.output
        )

        print("\n--- Summary of Operation ---")
        if result.get("filepath"):
            print(f"Image successfully saved to: {result['filepath']}")
            if result.get("text"):
                print(f"Model Response Text:\n{result['text']}")
        else:
            error_message = result.get("text", "An unknown error occurred.")
            print(f"Image editing failed. Reason: {error_message}")
            
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Fatal Error during execution: {e}")


if __name__ == "__main__":
    main()
