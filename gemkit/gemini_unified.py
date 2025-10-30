"""
Unified Gemini API Client
--------------------------
Combines text, structured output, and vision capabilities into a single interface.

This module provides a unified interface to Google's Gemini API, supporting:
- Text generation
- Structured output with schemas
- Vision/multimodal (image + text) processing
- LMDB storage for conversations
- System and assistant prompts

Example usage:
    # Text generation
    python gemini_unified.py -q "What is AI?"

    # Vision task
    python gemini_unified.py -q "Describe this image" -i photo.jpg

    # Structured output
    python gemini_unified.py -q "List 3 colors" -r '{"type": "array", "items": {"type": "string"}}'

    # Save to LMDB
    python gemini_unified.py -q "Hello" --save-lmdb
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Optional, Any, Union
from PIL import Image

from google import genai
from google.genai import types

# Import from existing modules
from gemini_client import GeminiClient
from gemini_vision import GeminiVision
from image_source import ImageSource, SourceConfig, OutputType
from lmdb_storage import LMDBStorage


@dataclass
class UnifiedModelInput:
    """
    A dataclass for structuring input to the UnifiedGeminiClient.

    This class consolidates all possible inputs for both text-only and multimodal
    (vision) tasks, including prompts, model configuration, and image sources.

    Attributes:
        user_prompt (str): The primary text prompt from the user.
        model_name (str): The name of the Gemini model to use.
        sys_prompt (str): Optional system-level instructions to guide the model's behavior.
        assist_prompt (str): Optional assistant message for providing few-shot examples or context.
        response_schema (Optional[Any]): A Pydantic model or a JSON schema dictionary for structured output.
        image_source (Optional[Union[str, int, Image.Image]]): The source of the image for vision tasks.
                                                               This can be a file path, URL, "screenshot",
                                                               a camera index (as an integer), or a PIL Image object.
    """
    user_prompt: str
    model_name: str = 'gemini-2.5-flash'
    sys_prompt: str = ""
    assist_prompt: str = ""
    response_schema: Optional[Any] = None
    image_source: Optional[Union[str, int, Image.Image]] = None


class UnifiedGeminiClient:
    """
    A unified client for interacting with the Google Gemini API, supporting
    text, structured output, and vision tasks.

    This class provides a single, consistent interface for various types of
    interactions with the Gemini model, automatically handling the differences
    between text-only and multimodal requests.
    """

    MODELS_NAME = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite']
    DEFAULT_MODEL = 'gemini-2.5-flash'

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initializes the UnifiedGeminiClient.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to 'gemini-2.5-flash'.
        """
        self.model_name = model_name
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Gemini API client with proper error handling."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                self.client = genai.Client()
            except Exception as e:
                raise ValueError(f"GEMINI_API_KEY environment variable not found or client initialization failed: {e}")
        else:
            self.client = genai.Client(api_key=api_key)

    def _prepare_image(self, image_source: Union[str, int, Image.Image]) -> Image.Image:
        """
        Load an image from various sources using ImageSource utility.

        Supports:
            - Image files (.jpg, .png, .bmp, .gif, .webp, .tiff)
            - URLs (automatic download)
            - Camera/Webcam (camera index as int)
            - Screenshots (use "screenshot" string)
            - Video files (extracts first frame)
            - Base64 strings
            - PIL Images (returned as-is)
            - Numpy arrays

        Args:
            image_source: The image source (file path, URL, camera index, etc.)

        Returns:
            A PIL Image object.

        Raises:
            FileNotFoundError: If the input file is not found.
            ImageSourceError: If the source is invalid or unsupported.
        """
        # If already a PIL Image, return it
        if isinstance(image_source, Image.Image):
            return image_source

        config = SourceConfig(output_type=OutputType.PIL)
        source = ImageSource(image_source, config)
        result = source.get_image()
        source.close()
        return result.data

    def _build_api_payload(
        self,
        model_input: UnifiedModelInput
    ) -> tuple[list[types.Content], Optional[types.GenerateContentConfig]]:
        """
        Helper function to construct the contents list and configuration.

        Handles both text-only and vision (multimodal) modes.

        Args:
            model_input: UnifiedModelInput containing all prompt and configuration settings.

        Returns:
            tuple: (list of contents for API call, configuration object)
        """
        contents = []

        # 1. Handle System Instruction (Config)
        config = None
        if model_input.sys_prompt:
            config = types.GenerateContentConfig(system_instruction=model_input.sys_prompt)

        # 2. Handle Assistant Prompt (Contents role="model")
        if model_input.assist_prompt:
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(text=model_input.assist_prompt)]
                )
            )

        # 3. Handle User Prompt with optional image (Contents role="user")
        user_parts = []

        # Add image first if provided (for vision tasks)
        if model_input.image_source is not None:
            image = self._prepare_image(model_input.image_source)
            user_parts.append(image)  # Gemini API accepts PIL Images directly

        # Add text prompt
        user_parts.append(types.Part(text=model_input.user_prompt))

        contents.append(
            types.Content(
                role="user",
                parts=user_parts
            )
        )

        return contents, config

    def generate_text(self, model_input: UnifiedModelInput) -> str:
        """
        Generates a text response from the Gemini model.

        This method can handle both text-only and multimodal (text and image) inputs.

        Args:
            model_input (UnifiedModelInput): The input for the model, including the prompt
                                             and an optional image source.

        Returns:
            str: The text response generated by the model.
        """
        contents, config = self._build_api_payload(model_input)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )

        return response.text

    def generate_structured(self, model_input: UnifiedModelInput):
        """
        Generates a structured (JSON) response from the Gemini model.

        This method can handle both text-only and multimodal inputs, and it
        formats the output according to the provided `response_schema`.

        Args:
            model_input (UnifiedModelInput): The input for the model, including the prompt,
                                             an optional image source, and a `response_schema`.

        Returns:
            A Pydantic model instance or a dictionary representing the structured response.
        """
        contents, base_config = self._build_api_payload(model_input)

        # Overlay structured output settings onto the base config
        config_params = {
            'response_schema': model_input.response_schema,
            'response_mime_type': 'application/json'
        }

        # Merge system instruction from base_config if it exists
        if base_config and base_config.system_instruction:
            config_params['system_instruction'] = base_config.system_instruction

        config = types.GenerateContentConfig(**config_params)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config
        )

        return response.parsed

    def generate(self, model_input: UnifiedModelInput) -> Union[str, dict, list]:
        """
        A smart generation method that automatically chooses between text and
        structured output based on the presence of a `response_schema`.

        Args:
            model_input (UnifiedModelInput): The input for the model.

        Returns:
            Union[str, dict, list]: The generated response, which can be a string
                                    or a structured JSON object.
        """
        if model_input.response_schema:
            return self.generate_structured(model_input)
        else:
            return self.generate_text(model_input)


def parse_arguments():
    """Parse command-line arguments for the unified Gemini client."""
    parser = argparse.ArgumentParser(
        description="Unified CLI tool to query the Gemini API with text and vision support. Set GEMINI_API_KEY environment variable.",
        epilog="""
Examples:
  # Text generation
  %(prog)s -q "What is artificial intelligence?"

  # Vision task with local image
  %(prog)s -q "Describe this image" -i photo.jpg

  # Vision task with URL
  %(prog)s -q "What's in this image?" -i https://example.com/image.jpg

  # Vision task with screenshot
  %(prog)s -q "Analyze this screenshot" -i screenshot

  # Vision task with camera
  %(prog)s -q "What do you see?" -i 0

  # Structured output (text-only)
  %(prog)s -q "List 3 primary colors" -r '{"type": "array", "items": {"type": "string"}}'

  # Structured output (vision + structured)
  %(prog)s -q "Extract text from image" -i doc.jpg -r '{"type": "object", "properties": {"text": {"type": "string"}}}'

  # Save to LMDB
  %(prog)s -q "Hello, Gemini!" --save-lmdb

  # All features combined
  %(prog)s -q "Describe colors" -i photo.jpg -s "You are a color expert" --save-lmdb
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-q", "--question",
        type=str,
        required=True,
        help="The question or prompt to send to the Gemini model."
    )

    parser.add_argument(
        "-i", "--image",
        type=str,
        default=None,
        help='Image source for vision tasks: file path, URL, "screenshot", or camera index (0, 1, etc.)'
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default=UnifiedGeminiClient.DEFAULT_MODEL,
        choices=UnifiedGeminiClient.MODELS_NAME,
        help=f"The Gemini model to use. Available: {', '.join(UnifiedGeminiClient.MODELS_NAME)}"
    )

    parser.add_argument(
        "-s", "--system-prompt",
        type=str,
        default="",
        dest="sys_prompt",
        help="Optional system prompt to set the model's persona or behavior."
    )

    parser.add_argument(
        "-a", "--assistant-prompt",
        type=str,
        default="",
        dest="assist_prompt",
        help="Optional assistant message (role='model') for few-shot examples or context."
    )

    parser.add_argument(
        "-r", "--response-schema",
        type=str,
        default=None,
        dest="response_schema",
        help="Optional JSON string or file path containing the response schema for structured output."
    )

    parser.add_argument(
        "--save-lmdb",
        action="store_true",
        help="Save the conversation (question and response) to LMDB storage."
    )

    parser.add_argument(
        "--lmdb-path",
        type=str,
        default="geminiqa.lmdb",
        help="Path to the LMDB database file. Default: geminiqa.lmdb"
    )

    return parser.parse_args()


def main():
    """Main entry point for the unified Gemini CLI."""
    args = parse_arguments()

    # Parse response schema if provided
    response_schema = None
    if args.response_schema:
        try:
            # Try to parse as JSON string first
            response_schema = json.loads(args.response_schema)
        except json.JSONDecodeError:
            # If not valid JSON, try to read as file path
            try:
                with open(args.response_schema, 'r') as f:
                    response_schema = json.load(f)
            except FileNotFoundError:
                print(f"\n[ERROR]: Could not parse response schema as JSON or find file: {args.response_schema}")
                exit(1)
            except json.JSONDecodeError:
                print(f"\n[ERROR]: File {args.response_schema} does not contain valid JSON")
                exit(1)

    # Handle camera index (convert string to int if it's a digit)
    image_source = args.image
    if args.image and args.image.isdigit():
        image_source = int(args.image)

    # Create UnifiedModelInput object
    model_input = UnifiedModelInput(
        user_prompt=args.question,
        model_name=args.model,
        sys_prompt=args.sys_prompt,
        assist_prompt=args.assist_prompt,
        response_schema=response_schema,
        image_source=image_source
    )

    # Create client and generate response
    try:
        client = UnifiedGeminiClient(model_input.model_name)

        # Display configuration
        print(f"\nModel: {model_input.model_name}")
        if args.sys_prompt:
            print(f"System Prompt: '{args.sys_prompt}'")
        if args.assist_prompt:
            print(f"Assistant Prompt: '{args.assist_prompt}'")
        if response_schema:
            print(f"Response Schema: {json.dumps(response_schema, indent=2)}")
        if image_source:
            print(f"Image Source: {args.image}")
        print(f"User Query: {args.question}\n")

        # Generate response
        start_time = time.time()

        if model_input.response_schema:
            result = client.generate_structured(model_input)
            text = json.dumps(result, indent=2)
            print("\n--- Gemini Structured Response (JSON) ---")
        else:
            text = client.generate_text(model_input)
            print("\n--- Gemini Response ---")

        end_time = time.time()

        print(text)
        print(f"\n--- Generation Time: {end_time - start_time:.2f} seconds ---\n")

        # Store to LMDB if requested
        if args.save_lmdb:
            try:
                storage = LMDBStorage(args.lmdb_path)
                # Include image info in the key if present
                storage_key = f"{args.question} [image: {args.image}]" if image_source else args.question
                storage.put(storage_key, text)
                print(f"[INFO]: Conversation saved to LMDB at '{args.lmdb_path}'")
            except Exception as storage_error:
                print(f"[WARNING]: Failed to save to LMDB: {storage_error}")

    except ValueError as ve:
        print(f"\n[CRITICAL CONFIG ERROR]: {ve}")
    except Exception as e:
        print(f"\n[RUNTIME ERROR]: An unexpected error occurred: {e}")
    finally:
        print("\nExecution finished.")


if __name__ == "__main__":
    main()
