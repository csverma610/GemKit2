"""Base class for PDF document analysis using Google Gemini API.

This module provides a shared foundation for PDF analysis tools, eliminating
duplication between different analysis approaches (proofreading, interactive chat, etc).
"""

import logging
import os
import pathlib
from typing import Optional

from google import genai


def _get_logger(name: str):
    """Get or create a logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class PDFAnalysisClient:
    """Base class for PDF document analysis with Gemini API.

    Provides shared functionality for:
    - PDF file validation and upload
    - Gemini API client management
    - Response validation and error handling
    - Context manager support for cleanup

    Subclasses should implement specific analysis methods.
    """

    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """Initialize PDF analysis client.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)

        Raises:
            ValueError: If GEMINI_API_KEY environment variable not set
            ConnectionError: If Gemini client initialization fails
        """
        self.model_name = model_name
        self.uploaded_file = None
        self.logger = _get_logger(self.__class__.__name__)
        self.client = self._create_client()
        self.logger.info(f"✓ {self.__class__.__name__} initialized with model: {self.model_name}")

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and clean up resources."""
        self.delete_pdf()

    def _create_client(self):
        """Create and return Gemini API client.

        Raises:
            ValueError: If GEMINI_API_KEY not set
            ConnectionError: If client initialization fails
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = (
                "GEMINI_API_KEY environment variable not set. "
                "Set it with: export GEMINI_API_KEY='your-api-key'"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        try:
            self.logger.info(f"Connecting to Gemini API with model: {self.model_name}")
            client = genai.Client(api_key=api_key)
            self.logger.info("✓ Gemini client initialized successfully")
            return client
        except Exception as e:
            error_msg = f"Failed to initialize Gemini client: {e}"
            self.logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def _validate_pdf_file(self, pdf_file: str) -> pathlib.Path:
        """Validate PDF file exists and has correct extension.

        Args:
            pdf_file: Path to PDF file

        Returns:
            pathlib.Path: Validated PDF file path

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        pdf_path = pathlib.Path(pdf_file)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_file}")

        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF. Got: {pdf_path.suffix}")

        return pdf_path

    def load_pdf(self, pdf_file: str):
        """Load and upload a PDF file to Gemini.

        Args:
            pdf_file: Path to the PDF file to upload

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        if self.uploaded_file:
            self.delete_pdf()

        pdf_path = self._validate_pdf_file(pdf_file)
        self.logger.info(f"Uploading PDF: {pdf_path.name}...")

        self.uploaded_file = self.client.files.upload(
            file=pdf_path,
            config=dict(mime_type='application/pdf')
        )
        self.logger.info(f"✓ PDF uploaded: {pdf_path.name}")

    def delete_pdf(self):
        """Delete the uploaded PDF file from Gemini."""
        if self.uploaded_file:
            try:
                self.client.files.delete(name=self.uploaded_file.name)
                self.logger.info(f"✓ PDF file deleted: {self.uploaded_file.name}")
            finally:
                self.uploaded_file = None

    def _validate_prompt_feedback(self, response):
        """Validate prompt feedback for safety issues.

        Args:
            response: API response object

        Raises:
            RuntimeError: If response was blocked by API
        """
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            self.logger.error(f"Response blocked by API: {block_reason}")
            raise RuntimeError(f"API blocked the response: {block_reason}")

    def _validate_response_content(self, response):
        """Validate response has actual content.

        Args:
            response: API response object

        Raises:
            RuntimeError: If no content was generated
        """
        if not response.candidates or response.text is None:
            error_msg = (
                "The API did not generate a response. "
                "This may be due to safety filters or content policy violations."
            )
            self.logger.error("No response content generated by the API")
            raise RuntimeError(error_msg)

    def _validate_api_response(self, response):
        """Orchestrate validation of API response.

        Args:
            response: API response object

        Raises:
            RuntimeError: If response validation fails
        """
        self._validate_prompt_feedback(response)
        self._validate_response_content(response)

    def _check_pdf_loaded(self):
        """Verify that a PDF has been loaded.

        Raises:
            RuntimeError: If no PDF is currently loaded
        """
        if not self.uploaded_file:
            raise RuntimeError("No PDF loaded. Use load_pdf() first.")
