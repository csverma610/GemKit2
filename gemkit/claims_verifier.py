import argparse
import json
import logging
import os
import sys
import time
from enum import Enum
from typing import Optional

from google import genai
from google.api_core import exceptions as google_exceptions
from pydantic import BaseModel, Field, field_validator
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


# Configure logging
logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """
    Enumeration of possible verification statuses for a claim.
    """
    TRUE = "True"
    FALSE = "False"
    PARTIALLY_TRUE = "Partially True"
    UNVERIFIABLE = "Unverifiable"


class ClaimVerificationResult(BaseModel):
    """
    A Pydantic model representing the structured result of a claim verification.

    Attributes:
        summary (str): A brief overview of the claim being verified.
        verification_status (VerificationStatus): The verification status of the claim.
        evidence (str): Supporting evidence or reasoning for the verification status.
        sources (str): Relevant sources or context that support the evidence.
        truthfulness_score (float): A score between 0.0 (certainly false) and 1.0 (certainly true).
    """
    summary: str = Field(description="A brief overview of the claim", min_length=1, max_length=1000)
    verification_status: VerificationStatus = Field(description="The verification status of the claim")
    evidence: str = Field(description="Supporting evidence or reasoning for the verification", min_length=1)
    sources: str = Field(description="Relevant sources or context if available", min_length=1)
    truthfulness_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Score between 0 (certainly false) and 1 (certainly true)"
    )

    @field_validator('summary', 'evidence', 'sources')
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure fields are not empty or just whitespace."""
        if not v or not v.strip():
            raise ValueError("Field cannot be empty or whitespace only")
        return v.strip()


class ClaimVerifier:
    """
    Verifies claims using the Google Gemini API, with features like retry logic,
    input validation, and structured output.

    This class can be used to verify a single claim or a batch of claims, with an
    option to enable Google Search grounding for more accurate fact-checking.
    """

    # Constants
    MAX_CLAIM_LENGTH = 10000
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT = 60  # seconds

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        timeout: int = REQUEST_TIMEOUT
    ):
        """
        Initializes the ClaimVerifier.

        Args:
            model (str, optional): The name of the Gemini model to use.
            api_key (Optional[str], optional): The Google API key. If not provided, it will be
                                               read from the GOOGLE_API_KEY environment variable.
            timeout (int, optional): The timeout for API requests in seconds.

        Raises:
            ValueError: If the API key is not provided and cannot be found in the environment.
        """
        self.model = model
        self.timeout = timeout

        # Validate API key
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            error_msg = (
                "API key not found. Please provide it via the api_key parameter "
                "or set the GOOGLE_API_KEY environment variable."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(f"Initializing ClaimVerifier with model: {model}")

        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info("Successfully initialized Gemini client")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise

    @staticmethod
    def validate_claim(claim: str) -> str:
        """
        Validates and sanitizes a claim.

        Args:
            claim (str): The claim to validate.

        Returns:
            str: The sanitized claim.

        Raises:
            ValueError: If the claim is empty, whitespace-only, or exceeds the maximum length.
        """
        if not claim or not claim.strip():
            raise ValueError("Claim cannot be empty or whitespace only")

        claim = claim.strip()

        if len(claim) > ClaimVerifier.MAX_CLAIM_LENGTH:
            raise ValueError(
                f"Claim exceeds maximum length of {ClaimVerifier.MAX_CLAIM_LENGTH} characters"
            )

        return claim

    @retry(
        retry=retry_if_exception_type((
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    def _make_api_call(self, prompt: str, enable_grounding: bool) -> str:
        """
        Make API call with retry logic.

        Args:
            prompt: The prompt to send
            enable_grounding: Whether to enable grounding

        Returns:
            Response text from the API

        Raises:
            Various Google API exceptions
        """
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": ClaimVerificationResult
        }

        # Add Google Search grounding if enabled
        if enable_grounding:
            generation_config["tools"] = [{"google_search": {}}]

        api_config = {
            "model": self.model,
            "contents": prompt,
            "config": generation_config
        }

        logger.debug(f"Making API call with grounding={enable_grounding}")
        response = self.client.models.generate_content(**api_config)
        return response.text

    def verify_claim(self, claim: str, enable_grounding: bool = False) -> Optional[ClaimVerificationResult]:
        """
        Verifies a single claim using the Gemini model.

        This method sends a claim to the Gemini API and returns a structured verification
        result. It includes retry logic for transient API errors.

        Args:
            claim (str): The claim to be verified.
            enable_grounding (bool, optional): If True, enables Google Search grounding to
                                               improve fact-checking accuracy. Defaults to False.

        Returns:
            Optional[ClaimVerificationResult]: A Pydantic model containing the verification
                                               results, or None if an unrecoverable error occurs.

        Raises:
            ValueError: If the claim is invalid.
            RuntimeError: If the API call fails after all retries or if the response is invalid.
        """
        # Validate claim
        try:
            claim = self.validate_claim(claim)
        except ValueError as e:
            logger.error(f"Claim validation failed: {e}")
            raise

        logger.info(f"Verifying claim (grounding={enable_grounding}): {claim[:100]}...")

        prompt = f"""Verify the given claim: "{claim}"

Provide a comprehensive verification analysis."""

        try:
            start_time = time.time()
            response_text = self._make_api_call(prompt, enable_grounding)
            elapsed = time.time() - start_time

            logger.info(f"API call completed in {elapsed:.2f}s")

            # Parse the JSON response into Pydantic model
            result = ClaimVerificationResult.model_validate_json(response_text)
            logger.info(f"Verification complete: status={result.verification_status.value}, score={result.truthfulness_score}")
            return result

        except google_exceptions.PermissionDenied as e:
            logger.error(f"Permission denied - check API key: {e}")
            raise ValueError(f"API authentication failed: {e}")

        except google_exceptions.InvalidArgument as e:
            logger.error(f"Invalid API request: {e}")
            raise ValueError(f"Invalid request: {e}")

        except google_exceptions.ResourceExhausted as e:
            logger.error(f"API quota exceeded: {e}")
            raise RuntimeError(f"API quota exceeded. Please try again later: {e}")

        except (google_exceptions.ServiceUnavailable, google_exceptions.DeadlineExceeded) as e:
            logger.error(f"API service unavailable after retries: {e}")
            raise RuntimeError(f"Service temporarily unavailable: {e}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response as JSON: {e}")
            raise RuntimeError(f"Invalid response format from API: {e}")

        except Exception as e:
            logger.error(f"Unexpected error during verification: {e}", exc_info=True)
            raise RuntimeError(f"Verification failed: {e}")

    def verify_claims_batch(
        self,
        claims: list[str],
        enable_grounding: bool = False,
        rate_limit_delay: float = 0.5
    ) -> dict[str, ClaimVerificationResult]:
        """
        Verifies a batch of claims with a specified delay between requests.

        Args:
            claims (list[str]): A list of claims to be verified.
            enable_grounding (bool, optional): If True, enables Google Search grounding.
                                               Defaults to False.
            rate_limit_delay (float, optional): The delay in seconds between each API request
                                                to avoid rate limiting. Defaults to 0.5.

        Returns:
            dict[str, ClaimVerificationResult]: A dictionary mapping each claim to its
                                                verification result. Claims that failed
                                                verification are omitted.
        """
        logger.info(f"Starting batch verification of {len(claims)} claims")
        results = {}
        failed_claims = []

        for i, claim in enumerate(claims, 1):
            try:
                logger.info(f"Processing claim {i}/{len(claims)}")
                result = self.verify_claim(claim, enable_grounding)
                if result:
                    results[claim] = result

                # Rate limiting - sleep between requests (except after last one)
                if i < len(claims) and rate_limit_delay > 0:
                    logger.debug(f"Rate limiting: sleeping {rate_limit_delay}s")
                    time.sleep(rate_limit_delay)

            except Exception as e:
                logger.error(f"Failed to verify claim {i}: {claim[:100]}... Error: {e}")
                failed_claims.append(claim)
                # Continue processing other claims

        if failed_claims:
            logger.warning(f"Failed to verify {len(failed_claims)} out of {len(claims)} claims")

        logger.info(f"Batch verification complete: {len(results)}/{len(claims)} successful")
        return results


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs to
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to setup file logging: {e}")


def main():
    """Command-line interface for the ClaimVerifier."""
    parser = argparse.ArgumentParser(
        description="Verify claims using Google's Gemini AI with production-grade features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single claim:
    python claims_verifier.py "The Earth is flat"
    python claims_verifier.py "Python was created in 1991" --grounding
    python claims_verifier.py "The sky is blue" --json

  Multiple claims from file:
    python claims_verifier.py --file claims.txt
    python claims_verifier.py --file claims.txt --grounding --json

  With logging:
    python claims_verifier.py "claim" --log-level DEBUG
    python claims_verifier.py --file claims.txt --log-file verification.log

File format: One claim per line (empty lines are ignored)
        """
    )

    # Mutually exclusive group for claim input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-c", "--claim",
        nargs="?",
        help="The claim to verify"
    )
    input_group.add_argument(
        "-f", "--file",
        metavar="FILE",
        help="File containing claims (one per line)"
    )

    # Optional arguments
    parser.add_argument(
        "-g", "--grounding",
        action="store_true",
        help="Enable Google Search grounding for fact-checking"
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    parser.add_argument(
        "-m", "--model",
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)"
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set logging level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        help="Write logs to specified file"
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.5,
        help="Delay between batch requests in seconds (default: 0.5)"
    )

    parser.add_argument(
        "--api-key",
        help="Google API key (can also use GOOGLE_API_KEY env var)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)

    # Initialize verifier
    try:
        verifier = ClaimVerifier(model=args.model, api_key=args.api_key)
    except ValueError as e:
        logger.error(f"Initialization failed: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Process file input
    if args.file:
        try:
            # Validate file size
            file_size = os.path.getsize(args.file)
            if file_size > ClaimVerifier.MAX_FILE_SIZE_BYTES:
                error_msg = f"File size ({file_size} bytes) exceeds maximum ({ClaimVerifier.MAX_FILE_SIZE_BYTES} bytes)"
                logger.error(error_msg)
                print(f"Error: {error_msg}", file=sys.stderr)
                return 1

            # Read claims from file
            with open(args.file, 'r', encoding='utf-8') as f:
                claims = [line.strip() for line in f if line.strip()]

            if not claims:
                logger.error(f"No claims found in {args.file}")
                print(f"Error: No claims found in {args.file}", file=sys.stderr)
                return 1

            logger.info(f"Processing {len(claims)} claim(s) from {args.file}")
            if not args.json:
                print(f"Processing {len(claims)} claim(s) from {args.file}...\n")

            # Process batch with rate limiting
            results = verifier.verify_claims_batch(
                claims,
                enable_grounding=args.grounding,
                rate_limit_delay=args.rate_limit
            )

            if args.json:
                # Output as JSON array
                output = [
                    {
                        "claim": claim,
                        "result": result.model_dump()
                    }
                    for claim, result in results.items()
                ]
                print(json.dumps(output, indent=2))
            else:
                # Human-readable output
                for i, (claim, result) in enumerate(results.items(), 1):
                    print(f"{'='*80}")
                    print(f"Claim {i}: {claim}")
                    print(f"{'='*80}")
                    print(f"Summary: {result.summary}")
                    print(f"Verification Status: {result.verification_status.value}")
                    print(f"Evidence: {result.evidence}")
                    print(f"Sources: {result.sources}")
                    print(f"Truthfulness Score: {result.truthfulness_score}")
                    print()

        except FileNotFoundError:
            logger.error(f"File not found: {args.file}")
            print(f"Error: File '{args.file}' not found", file=sys.stderr)
            return 1
        except PermissionError:
            logger.error(f"Permission denied reading file: {args.file}")
            print(f"Error: Permission denied reading '{args.file}'", file=sys.stderr)
            return 1
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            print(f"Error processing file: {e}", file=sys.stderr)
            return 1

    # Process single claim
    else:
        try:
            result = verifier.verify_claim(args.claim, enable_grounding=args.grounding)

            if result:
                if args.json:
                    print(result.model_dump_json(indent=2))
                else:
                    print(f"Summary: {result.summary}")
                    print(f"Verification Status: {result.verification_status.value}")
                    print(f"Evidence: {result.evidence}")
                    print(f"Sources: {result.sources}")
                    print(f"Truthfulness Score: {result.truthfulness_score}")
            else:
                logger.error("Verification returned no result")
                return 1

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except RuntimeError as e:
            logger.error(f"Runtime error: {e}")
            print(f"Error: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            print(f"Error: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
