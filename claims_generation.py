import argparse
import logging
import os
import re
import time

from dataclasses import dataclass
from typing import Optional
from logging.handlers import RotatingFileHandler
from google import genai

# Custom Exceptions
class ClaimsGenerationError(Exception):
    """Base exception for claims generation errors."""
    pass


class FileSizeLimitError(ClaimsGenerationError):
    """Raised when file size exceeds the limit."""
    pass


class APIError(ClaimsGenerationError):
    """Raised when API request fails."""
    pass


class InvalidResponseError(ClaimsGenerationError):
    """Raised when API response is invalid or empty."""
    pass


# Configuration
@dataclass
class ClaimsConfig:
    """Configuration settings for claims generation."""
    model: str = "gemini-2.5-flash"
    max_claims: Optional[int] = None
    max_file_size_mb: int = 10
    max_retries: int = 3
    chunk_size_chars: int = 8000
    retry_delay_seconds: int = 2
    log_file: str = "claims_generation.log"
    log_level: str = "INFO"
    log_file_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5
    chunk_overlap_chars: int = 200  # Overlap between chunks to avoid missing context
    similarity_length_threshold: int = 10  # Character length difference threshold for similarity check
    similarity_word_overlap_ratio: float = 0.85  # Word overlap ratio threshold (85%)


class ClaimsGenerator:
    """
    A class to generate verifiable claims from text using Google's Gemini API.
    """

    def __init__(self, config: ClaimsConfig = None):
        """
        Initialize the ClaimsGenerator with configuration.

        Args:
            config: Configuration object. If not provided, uses default ClaimsConfig values.
        """
        self.config = config if config is not None else ClaimsConfig()
        self.client = genai.Client()
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initialized ClaimsGenerator with model={self.config.model}, "
            f"max_claims={self.config.max_claims}, max_file_size_mb={self.config.max_file_size_mb}, "
            f"max_retries={self.config.max_retries}, chunk_size_chars={self.config.chunk_size_chars}"
        )

    def read_file(self, filename: str) -> str:
        """
        Read content from a file with size validation.

        Args:
            filename: Path to the file to read

        Returns:
            The content of the file as a string

        Raises:
            FileNotFoundError: If the file doesn't exist
            FileSizeLimitError: If file size exceeds the limit
            ClaimsGenerationError: If file has encoding errors
        """
        self.logger.info(f"Reading file: {filename}")

        try:
            # Check file size before reading
            file_size_bytes = os.path.getsize(filename)
            file_size_mb = file_size_bytes / (1024 * 1024)

            self.logger.debug(f"File size: {file_size_mb:.2f}MB")

            if file_size_mb > self.config.max_file_size_mb:
                error_msg = (
                    f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed "
                    f"size ({self.config.max_file_size_mb}MB)"
                )
                self.logger.error(error_msg)
                raise FileSizeLimitError(error_msg)

            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                self.logger.debug(f"Successfully read {len(content)} characters from {filename}")
                return content

        except FileNotFoundError:
            self.logger.error(f"File not found: {filename}")
            raise
        except FileSizeLimitError:
            raise
        except UnicodeDecodeError as encoding_error:
            error_msg = f"File encoding error for {filename}: {encoding_error}"
            self.logger.error(error_msg)
            raise ClaimsGenerationError(error_msg)
        except Exception as file_error:
            self.logger.error(f"Error reading file {filename}: {file_error}")
            raise

    def _split_text_into_chunks(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks for processing.

        Args:
            text: The text to split

        Returns:
            List of text chunks
        """
        if len(text) <= self.config.chunk_size_chars:
            self.logger.debug("Text fits in single chunk")
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.config.chunk_size_chars

            # If this is not the last chunk, try to find a sentence boundary
            if end < text_length:
                # Look for sentence endings within the last 200 characters
                search_start = max(start, end - 200)
                for sentence_delimiter in ['. ', '.\n', '! ', '?\n', '? ']:
                    last_delimiter_pos = text.rfind(sentence_delimiter, search_start, end)
                    if last_delimiter_pos != -1:
                        end = last_delimiter_pos + 1
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                self.logger.debug(f"Created chunk {len(chunks)}: {len(chunk)} chars")

            # Move start position with overlap
            start = end - self.config.chunk_overlap_chars if end < text_length else text_length

        self.logger.info(f"Split text into {len(chunks)} chunks")
        return chunks

    def _deduplicate_claims(self, all_claims: list[str]) -> list[str]:
        """
        Remove duplicate and highly similar claims.

        Args:
            all_claims: List of all claims from different chunks

        Returns:
            List of unique claims
        """
        if not all_claims:
            return []

        unique_claims = []
        seen_normalized_claims = set()

        for claim in all_claims:
            claim_text = claim.strip()
            if not claim_text:
                continue

            # Normalize the claim for comparison (lowercase, remove extra spaces)
            normalized_claim = ' '.join(claim_text.lower().split())

            # Skip if we've seen this exact claim
            if normalized_claim in seen_normalized_claims:
                self.logger.debug(f"Skipping duplicate claim: {claim_text[:50]}...")
                continue

            # Check for high similarity with existing claims
            is_duplicate = False
            for existing_normalized_claim in seen_normalized_claims:
                # If claims are very similar in length and content
                if abs(len(normalized_claim) - len(existing_normalized_claim)) < self.config.similarity_length_threshold:
                    # Simple similarity check: shared word ratio
                    current_claim_words = set(normalized_claim.split())
                    existing_claim_words = set(existing_normalized_claim.split())
                    if len(current_claim_words) > 0 and len(existing_claim_words) > 0:
                        word_overlap_ratio = len(current_claim_words & existing_claim_words) / max(len(current_claim_words), len(existing_claim_words))
                        if word_overlap_ratio > self.config.similarity_word_overlap_ratio:
                            is_duplicate = True
                            self.logger.debug(f"Skipping similar claim: {claim_text[:50]}...")
                            break

            if not is_duplicate:
                unique_claims.append(claim_text)
                seen_normalized_claims.add(normalized_claim)

        self.logger.info(f"Deduplicated {len(all_claims)} claims to {len(unique_claims)} unique claims")
        return unique_claims

    def _parse_claims_from_response(self, api_response_text: str) -> list[str]:
        """
        Parse individual claims from the API response.

        Args:
            api_response_text: The raw response text containing numbered claims

        Returns:
            List of individual claims
        """
        claims = []
        lines = api_response_text.strip().split('\n')

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Remove common numbering patterns: "1. ", "1) ", "1.", etc.
            cleaned_claim = re.sub(r'^\d+[\.\)]\s*', '', line)
            if cleaned_claim:
                claims.append(cleaned_claim)

        return claims

    def _sanitize_input(self, text: str) -> str:
        """
        Sanitize input text by removing problematic characters.

        Args:
            text: The input text to sanitize

        Returns:
            Sanitized text with null bytes removed
        """
        # Remove any null bytes that could cause issues with text processing
        text = text.replace('\x00', '')

        return text

    def _make_api_call_with_retry(self, prompt: str) -> str:
        """
        Make an API call with retry logic.

        Args:
            prompt: The prompt to send to the API

        Returns:
            The API response text

        Raises:
            APIError: If all retry attempts fail
        """
        last_exception = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.debug(f"API call attempt {attempt}/{self.config.max_retries}")
                start_time = time.time()

                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=prompt
                )

                elapsed_time = time.time() - start_time
                self.logger.info(f"API call successful in {elapsed_time:.2f}s")

                return response.text

            except Exception as api_error:
                last_exception = api_error
                self.logger.warning(f"API call attempt {attempt} failed: {api_error}")

                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay_seconds * attempt  # Exponential backoff
                    self.logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"All {self.config.max_retries} API call attempts failed")

        # All retries failed
        raise APIError(f"API call failed after {self.config.max_retries} attempts: {last_exception}")

    def _generate_claims_for_chunk(self, chunk: str, chunk_number: int, total_chunks: int) -> list[str]:
        """
        Generate claims for a single chunk of text.

        Args:
            chunk: The text chunk to process
            chunk_number: The chunk number (1-indexed)
            total_chunks: Total number of chunks

        Returns:
            List of claims extracted from this chunk

        Raises:
            APIError: If API calls fail
            InvalidResponseError: If API response is invalid
        """
        self.logger.info(f"Processing chunk {chunk_number}/{total_chunks} ({len(chunk)} characters)")

        # Sanitize input
        chunk = self._sanitize_input(chunk)

        prompt = (
            f"Given the following text, provide an enumerated list of verifiable claims. "
            f"Each claim must be:\n"
            f"1. Unique - no duplicate or highly similar claims\n"
            f"2. Diverse - covering different aspects of the text\n"
            f"3. Specific and verifiable\n"
            f"4. Independent - each claim should stand on its own\n\n"
            f"Do not include any other text besides the numbered list itself.\n\n{chunk}"
        )

        self.logger.debug(f"Sending request to model: {self.config.model}")

        # Make API call with retry logic
        api_response_text = self._make_api_call_with_retry(prompt)

        # Validate response
        if not api_response_text or not api_response_text.strip():
            error_msg = f"Received empty response from API for chunk {chunk_number}"
            self.logger.error(error_msg)
            raise InvalidResponseError(error_msg)

        # Parse claims from response
        claims = self._parse_claims_from_response(api_response_text)
        self.logger.info(f"Extracted {len(claims)} claims from chunk {chunk_number}/{total_chunks}")

        return claims

    def generate_text(self, text: str) -> list[str]:
        """
        Generate verifiable claims from the given text.
        Large texts are split into chunks and processed separately.

        Args:
            text: The input text to extract claims from

        Returns:
            List of verifiable claims

        Raises:
            APIError: If API calls fail
            InvalidResponseError: If API response is invalid or no claims generated
        """
        self.logger.info(f"Generating claims from text ({len(text)} characters)")

        # Sanitize input
        text = self._sanitize_input(text)

        # Split text into chunks
        chunks = self._split_text_into_chunks(text)

        # Generate claims for each chunk
        all_claims = []
        for chunk_idx, chunk in enumerate(chunks, 1):
            chunk_claims = self._generate_claims_for_chunk(chunk, chunk_idx, len(chunks))
            all_claims.extend(chunk_claims)

        self.logger.info(f"Generated {len(all_claims)} total claims from {len(chunks)} chunks")

        # Deduplicate claims
        unique_claims = self._deduplicate_claims(all_claims)

        # Apply max_claims limit if specified
        if self.config.max_claims and len(unique_claims) > self.config.max_claims:
            self.logger.info(f"Limiting output from {len(unique_claims)} to {self.config.max_claims} claims")
            unique_claims = unique_claims[:self.config.max_claims]

        # Validate we have claims
        if not unique_claims:
            error_msg = "No claims were generated"
            self.logger.error(error_msg)
            raise InvalidResponseError(error_msg)

        self.logger.info(f"Successfully generated {len(unique_claims)} unique claims")

        return unique_claims

    def generate_claims_from_file(self, filename: str) -> list[str]:
        """
        Read a file and generate verifiable claims from its content.

        Args:
            filename: Path to the file to process

        Returns:
            List of verifiable claims

        Raises:
            FileNotFoundError: If the file doesn't exist
            FileSizeLimitError: If file size exceeds the limit
            ClaimsGenerationError: If file has encoding errors
            APIError: If API calls fail
            InvalidResponseError: If API response is invalid or no claims generated
        """
        content = self.read_file(filename)
        return self.generate_text(content)

    @staticmethod
    def format_claims_as_numbered_list(claims: list[str]) -> str:
        """
        Format a list of claims as a numbered list string.

        Args:
            claims: List of claims to format

        Returns:
            Formatted string with numbered claims
        """
        return '\n'.join(f"{num}. {claim}" for num, claim in enumerate(claims, 1))


def setup_logging(config: ClaimsConfig):
    """
    Configure logging to write to both file and console with rotation.

    Args:
        config: Configuration object containing log settings
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, config.log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Get root logger
    logger = logging.getLogger()

    # Clear any existing handlers to prevent accumulation
    if logger.handlers:
        logger.handlers.clear()

    logger.setLevel(numeric_level)

    # Rotating file handler (10MB per file, keep 5 backups)
    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.log_file_max_bytes,
        backupCount=config.log_file_backup_count
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)

    # Console handler (only for WARNING and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate verifiable claims from text using Google's Gemini API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s document.txt                    # Generate all claims
  %(prog)s document.txt --max-claims 5     # Generate at most 5 claims
  %(prog)s document.txt -m 10 --model gemini-2.5-pro  # Use different model
  %(prog)s document.txt --log-level DEBUG  # Enable debug logging
  %(prog)s document.txt --chunk-size 5000  # Use smaller chunks for processing
        """
    )

    parser.add_argument(
        'filename',
        help='Path to the text file to process'
    )

    parser.add_argument(
        '-m', '--max-claims',
        type=int,
        default=None,
        metavar='N',
        help='Maximum number of claims to generate (default: no limit)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='gemini-2.5-flash',
        help='Gemini model to use (default: gemini-2.5-flash)'
    )

    parser.add_argument(
        '--log-file',
        type=str,
        default='claims_generation.log',
        help='Path to log file (default: claims_generation.log)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    parser.add_argument(
        '--max-file-size',
        type=int,
        default=10,
        metavar='MB',
        help='Maximum file size in MB (default: 10)'
    )

    parser.add_argument(
        '--max-retries',
        type=int,
        default=3,
        metavar='N',
        help='Maximum number of API retry attempts (default: 3)'
    )

    parser.add_argument(
        '--chunk-size',
        type=int,
        default=8000,
        metavar='CHARS',
        help='Size of each text chunk in characters (default: 8000)'
    )

    args = parser.parse_args()

    # Validate max_claims
    if args.max_claims is not None and args.max_claims <= 0:
        parser.error("max_claims must be a positive integer")

    # Create config from command-line arguments
    config = ClaimsConfig(
        model=args.model,
        max_claims=args.max_claims,
        max_file_size_mb=args.max_file_size,
        max_retries=args.max_retries,
        chunk_size_chars=args.chunk_size,
        log_file=args.log_file,
        log_level=args.log_level
    )

    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)

    logger.info(f"Starting claims generation for file: {args.filename}")

    # Create the claims generator and process the file
    try:
        generator = ClaimsGenerator(config=config)
        claims = generator.generate_claims_from_file(args.filename)
        formatted_output = ClaimsGenerator.format_claims_as_numbered_list(claims)
        print(formatted_output)
        logger.info("Successfully completed claims generation")
    except FileNotFoundError:
        logger.error(f"File not found: {args.filename}")
        print(f"Error: The file '{args.filename}' was not found.")
        parser.exit(1)
    except FileSizeLimitError as size_error:
        logger.error(f"File size limit exceeded: {size_error}")
        print(f"Error: {size_error}")
        parser.exit(1)
    except APIError as api_error:
        logger.error(f"API error: {api_error}")
        print(f"Error: API request failed. {api_error}")
        parser.exit(1)
    except InvalidResponseError as response_error:
        logger.error(f"Invalid API response: {response_error}")
        print(f"Error: {response_error}")
        parser.exit(1)
    except ClaimsGenerationError as claims_error:
        logger.error(f"Claims generation error: {claims_error}")
        print(f"Error: {claims_error}")
        parser.exit(1)
    except Exception as unexpected_error:
        logger.exception("An unexpected error occurred during processing")
        print(f"An unexpected error occurred: {unexpected_error}")
        parser.exit(1)


if __name__ == "__main__":
    main()
