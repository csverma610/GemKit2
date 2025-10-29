import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path to import gemini_client
sys.path.insert(0, str(Path(__file__).parent.parent))

from gemini_client import GeminiClient, ModelConfig, ModelInput

def main():
    """
    A simple command-line interface for the GeminiClient.
    """
    parser = argparse.ArgumentParser(description="A simple CLI for GeminiClient.")
    parser.add_argument("-p", "--prompt", type=str, required=True, help="The user's text prompt.")
    parser.add_argument("-i", "--images", type=str, nargs='*', help="A list of image file paths.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output.")
    parser.add_argument("-m", "--model-name", type=str, default="gemini-2.5-flash", help="The Gemini model to use.")

    args = parser.parse_args()

    # Create ModelConfig and GeminiClient
    model_config = ModelConfig(model_name=args.model_name)
    gemini_client = GeminiClient(config=model_config)

    # Create ModelInput
    model_input = ModelInput(
        user_prompt=args.prompt,
        images=args.images
    )

    # Generate content
    try:
        if args.stream:
            response_stream = gemini_client.generate_content(model_input, stream=True)
            for chunk in response_stream:
                print(chunk, end="", flush=True)
            print()
        else:
            response = gemini_client.generate_content(model_input, stream=False)
            print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
