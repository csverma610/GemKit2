import argparse
import logging
import os
from typing import List

from google import genai
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from image_source import ImageSource, SourceConfig, OutputType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic Models ---

class MatchFeature(BaseModel):
    """Represents a single visual feature and its match status."""
    description: str
    matching: bool
    confidence: float

class FeaturesResult(BaseModel):
    """A Pydantic model for the final identification result, containing a list of features."""
    features: List[MatchFeature]


class ObjectIdentifier:
    """
    Identifies an object in an image by generating, checking, and scoring 
    visual features in a single, efficient API call.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        if not model_name:
            raise ValueError("A valid Gemini model name must be provided.")
        self.client = self._create_client()
        self.model_name = model_name

    def _create_client(self) -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        return genai.Client(api_key=api_key)

    def _generate_prompt(self, object_name: str) -> str:
        """
        Generates the prompt for object identification.

        Args:
            object_name: The name of the object to identify.

        Returns:
            A formatted prompt string.
        """
        return (
            f"Your task is to identify a '{object_name}' in the provided image. "
            f"Perform the following steps in your reasoning process:\n"
            f"1. First, internally generate a list of 5 distinct visual features to identify a '{object_name}'.\n"
            f"2. Second, examine the image and determine which of those 5 features are present.\n"
            f"3. Finally, respond with a JSON object that strictly follows this Pydantic schema: "
            f"class MatchFeature(BaseModel): description: str; matching: bool; confidence: float; "
            f"class FeaturesResult(BaseModel): features: List[MatchFeature].\n"
            f"Set 'matching' to true if the feature is present, false otherwise. "
            f"Set 'confidence' to a value between 0.0 and 1.0 indicating how confident you are in the match (1.0 = very confident, 0.0 = not present)."
        )

    def _generate_payload(self, prompt: str, image_path: str) -> list:
        """
        Generates the payload for the API call by loading the image.
        Retries up to 3 times on network/IO errors with exponential backoff.

        Args:
            prompt: The prompt string to send.
            image_path: The path or URL to the image file.

        Returns:
            A list containing the prompt and loaded image.

        Raises:
            ValueError: If the image fails to load.
        """
        config = SourceConfig(output_type=OutputType.PIL)
        image_source = ImageSource(image_path, config)
        image = image_source.get_image().data

        # Validate image was loaded successfully
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        return [prompt, image]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _call_api(self, payload: list) -> FeaturesResult:
        """
        Calls the Gemini API with retry logic for network errors.
        Retries up to 3 times on connection/timeout errors with exponential backoff.

        Args:
            payload: The payload containing prompt and image.

        Returns:
            A FeaturesResult object containing the API response.

        Raises:
            Exception: If the API call fails after all retries.
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=payload,
            config={
                "response_mime_type": "application/json",
                "response_schema": FeaturesResult,
            },
        )
        return response.parsed

    def identify_object(self, image_path: str, object_name: str) -> FeaturesResult:
        """
        Identifies an object in an image using a single, comprehensive API call.

        Args:
            image_path: The path or URL to the image file.
            object_name: The name of the object to identify.

        Returns:
            A FeaturesResult object containing the detailed analysis.

        Raises:
            ValueError: If object_name or image_path is empty or invalid.
        """
        # Validate inputs
        if not object_name or not object_name.strip():
            raise ValueError("object_name cannot be empty")
        object_name = object_name.strip()[:100]  # Limit length to prevent abuse

        if not image_path or not image_path.strip():
            raise ValueError("image_path cannot be empty")
        image_path = image_path.strip()

        try:
            logger.info(f"Identifying '{object_name}' in the image with a single API call...")

            prompt = self._generate_prompt(object_name)
            payload = self._generate_payload(prompt, image_path)
            return self._call_api(payload)

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            raise
        except OSError as e:
            logger.error(f"File system error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during identification: {e}", exc_info=True)
            return FeaturesResult(features=[])

def main():
    parser = argparse.ArgumentParser(description="Identify an object in an image using a single-call, feature-based analysis.")
    parser.add_argument("-i", "--image_path", required=True, help="Path or URL to the image file.")
    parser.add_argument("-o", "--object_name", required=True, help="The name of the object to identify.")
    parser.add_argument("-m", "--model_name", default="gemini-2.5-flash", help="Name of the Gemini model to use.")
    args = parser.parse_args()

    try:
        identifier = ObjectIdentifier(model_name=args.model_name)
        result = identifier.identify_object(args.image_path, args.object_name)
        
        print("\n--- Object Identification Report ---")
        if not result.features:
            print("Identification process failed or returned no results.")
        else:
            # Extract the generated features for the report
            generated_features = [f.description for f in result.features]
            matched_count = sum(1 for r in result.features if r.matching)
            # Calculate match score: only count confidence for matched features
            overall_confidence = (sum(f.confidence for f in result.features if f.matching) / len(result.features)) * 100 if result.features else 0
            
            print(f"Object to Identify: {args.object_name}")
            print(f"Generated & Matched Features: {', '.join(generated_features)}")
            print("\n--- Feature Analysis ---")
            for res in result.features:
                status = "MATCHED" if res.matching else "NOT MATCHED"
                print(f"  - Feature: '{res.description}' -> {status} (confidence: {res.confidence:.2f})")
            print("------------------------")
            print(f"\nOverall Confidence Score: {overall_confidence:.2f}%")

        print("------------------------------------")
            
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
