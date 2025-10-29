import argparse
import json
import logging
import os
import pathlib
from typing import Optional, List, Type
from datetime import datetime

from google import genai
from pydantic import BaseModel, Field, validator

# Configure logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('gemini_pdf.log'),
        logging.StreamHandler()
    ]
)


class GeminiPDFBase:
    """Base class for Gemini PDF operations with shared functionality."""
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the Gemini PDF base.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
        self.model_name = model_name
        self.uploaded_file = None
        self.pdf_filename = None
        self.client = self._create_client()

    def __enter__(self):
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager and clean up resources."""
        self._delete_pdf()

    def _validate_pdf(self, pdf_file: str):
        """
        Validate that the PDF file exists and has correct extension.

        Args:
            pdf_file: Path to the PDF file to validate

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        pdf_path = pathlib.Path(pdf_file)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_file}")

        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF with .pdf extension. Got: {pdf_path.suffix}")

    def _create_client(self):
        """Create and return a Gemini API client."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")

        try:
            return genai.Client(api_key=api_key)
        except Exception as e:
            logging.error(f"Error initializing Gemini client: {e}")
            raise ConnectionError("Failed to initialize Gemini client.") from e

    def load_pdf(self, pdf_file: str):
        """
        Load and upload a PDF file to Gemini.

        Args:
            pdf_file: Path to the PDF file to upload

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not a PDF
        """
        if self.uploaded_file:
            self._delete_pdf()

        self._validate_pdf(pdf_file)
        pdf_path = pathlib.Path(pdf_file)

        logging.info(f"Uploading PDF: {pdf_path.name}...")
        self.uploaded_file = self.client.files.upload(
            file=pdf_path,
            config=dict(mime_type='application/pdf')
        )
        logging.info(f"✓ PDF uploaded successfully: {pdf_path.name}\n")
        self.pdf_filename = pdf_path.name

    def _delete_pdf(self):
        """Delete the uploaded PDF file from Gemini."""
        if self.uploaded_file:
            try:
                self.client.files.delete(name=self.uploaded_file.name)
                logging.info(f"✓ PDF file deleted: {self.uploaded_file.name}")
            finally:
                self.uploaded_file = None

    def generate_text(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> str | BaseModel:
        """
        Send a message and get a response.

        Args:
            prompt: User's message/question
            response_schema: Optional Pydantic BaseModel class for structured output

        Returns:
            If response_schema provided: Pydantic model instance
            If no schema: Plain text string response
        """
        if not self.uploaded_file:
            raise ValueError("No PDF loaded. Use load_pdf() first.")

        config = None
        if response_schema:
            config = {
                "response_mime_type": "application/json",
                "response_schema": response_schema
            }

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.uploaded_file, prompt],
            config=config
        )

        if response_schema:
            # If parsed is None, try to parse the text response manually
            if response.parsed is not None:
                return response.parsed
            else:
                # Fall back to manual parsing from text
                from pydantic import ValidationError
                try:
                    data = json.loads(response.text)
                    return response_schema(**data)
                except (json.JSONDecodeError, ValidationError) as e:
                    logging.error(f"Failed to parse response: {e}\nResponse text: {response.text}")
                    raise
        return response.text


class MCQOption(BaseModel):
    """Model for a single MCQ option."""
    option_id: str = Field(..., description="Option identifier (A, B, C, D, etc.)")
    text: str = Field(..., description="Option text content")


class MultipleChoiceQuestion(BaseModel):
    """Model for a single Multiple-Choice Question with validation."""
    id: str = Field(..., description="Unique question identifier")
    question_text: str = Field(..., description="The question content")
    options: List[MCQOption] = Field(..., description="List of options")
    correct_answer: Optional[str] = Field(None, description="Correct answer option ID (e.g., 'A')")
    image_path: Optional[str] = Field(None, description="Path to associated image file if any")

    @validator('options')
    def validate_options(cls, v):
        """Ensure at least 2 options are provided."""
        if len(v) < 2:
            raise ValueError("At least 2 options are required")
        return v

    @validator('correct_answer')
    def validate_correct_answer(cls, v, values):
        """Ensure correct answer matches one of the options."""
        if v is None:
            return v

        if 'options' in values:
            valid_ids = [opt.option_id for opt in values['options']]
            if v not in valid_ids:
                raise ValueError(f"Correct answer '{v}' must match one of the option IDs: {valid_ids}")
        return v


class MCQExtractionResult(BaseModel):
    """Model for the complete extraction result."""
    pdf_filename: str = Field(..., description="Name of the source PDF")
    extraction_timestamp: str = Field(..., description="When extraction was performed")
    total_questions: int = Field(..., description="Total number of questions extracted")
    questions: List[MultipleChoiceQuestion] = Field(..., description="List of extracted questions")


class GeminiMCQExtractor(GeminiPDFBase):
    """MCQ extractor that inherits shared PDF operations from GeminiPDFBase."""

    def __init__(self, model_name: str = GeminiPDFBase.DEFAULT_MODEL, output_dir: str = "mcq_output"):
        """
        Initialize the MCQ extractor.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
            output_dir: Directory to save extracted MCQs and images
        """
        super().__init__(model_name)
        self.output_dir = pathlib.Path(output_dir)
        self._setup_output_directory()

    def _setup_output_directory(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = self.output_dir / "images"
        self.images_dir.mkdir(exist_ok=True)
        logging.info(f"✓ Output directory configured: {self.output_dir}")

    def extract_mcqs(self) -> MCQExtractionResult:
        """
        Extract all MCQs from the loaded PDF using Gemini.

        Returns:
            MCQExtractionResult: Structured data of extracted MCQs

        Raises:
            ValueError: If no PDF is loaded
            json.JSONDecodeError: If response cannot be parsed
        """
        if not self.uploaded_file:
            raise ValueError("No PDF loaded. Use load_pdf() first.")

        logging.info("Extracting MCQs from PDF...")

        prompt = """
Please extract all Multiple-Choice Questions from this PDF. For each question, provide:
1. id: The question number/identifier as it appears in the PDF (e.g., "6", "7", "8")
2. question_text: The complete question text (preserving all details and line breaks)
3. options: A list of options with:
   - option_id: The option identifier (A, B, C, D, E, etc.)
   - text: The option text without the parentheses
4. correct_answer: The correct option ID if available in the PDF (e.g., "A"), otherwise null
5. image_path: null (we'll handle images separately)

Important:
- Extract ALL questions from the PDF
- Preserve the exact wording of questions and options
- Use the numeric ID as shown in the PDF (not "Q1" format)
- If a question has an associated image, note it but set image_path to null
- Format the response as a JSON array of questions

Return ONLY valid JSON array without any markdown formatting or explanation.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.uploaded_file, prompt]
            )

            # Parse the JSON response
            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            questions_data = json.loads(response_text)

            # Validate and create MCQs
            validated_questions = []
            validation_errors = []

            for idx, q_data in enumerate(questions_data, 1):
                try:
                    mcq = MultipleChoiceQuestion(**q_data)
                    validated_questions.append(mcq)
                except Exception as e:
                    validation_errors.append(f"Question {idx}: {str(e)}")
                    logging.warning(f"Validation error for question {idx}: {e}")

            if validation_errors:
                logging.warning(f"Validation errors found:\n" + "\n".join(validation_errors))

            # Create result
            result = MCQExtractionResult(
                pdf_filename=self.pdf_filename,
                extraction_timestamp=datetime.now().isoformat(),
                total_questions=len(validated_questions),
                questions=validated_questions
            )

            logging.info(f"✓ Successfully extracted {len(validated_questions)} MCQs")
            return result

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse MCQ extraction response: {e}")
            raise
        except Exception as e:
            logging.error(f"Error extracting MCQs: {e}")
            raise

    def save_results(self, result: MCQExtractionResult, filename: Optional[str] = None) -> pathlib.Path:
        """
        Save extracted MCQs to JSON file.

        Args:
            result: MCQExtractionResult object to save
            filename: Custom filename (default: mcq_extraction_<timestamp>.json)

        Returns:
            Path to the saved JSON file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mcq_extraction_{timestamp}.json"

        output_file = self.output_dir / filename

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.model_dump_json(indent=2))

        logging.info(f"✓ MCQ data saved to: {output_file}")
        return output_file

    def extract_images(self) -> dict:
        """
        Extract and save images from the PDF using Gemini.

        Returns:
            Dictionary mapping question IDs to their image paths
        """
        if not self.uploaded_file:
            raise ValueError("No PDF loaded. Use load_pdf() first.")

        logging.info("Extracting images from PDF...")

        prompt = """
Extract all images from this PDF. For each image found, provide:
- image_id: A unique identifier (IMG1, IMG2, etc.)
- page_number: The page where the image appears
- description: Brief description of what the image contains
- associated_question: Which question (Q1, Q2, etc.) this image is associated with, if any

Return ONLY valid JSON array without markdown formatting.
"""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.uploaded_file, prompt]
            )

            response_text = response.text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            images_data = json.loads(response_text)
            image_mapping = {}

            logging.info(f"✓ Found {len(images_data)} images in PDF")

            # Note: Actual image extraction and saving would require additional
            # implementation with the Gemini Files API or other image extraction methods
            # For now, we log the metadata
            for img in images_data:
                image_mapping[img.get('image_id')] = {
                    'page': img.get('page_number'),
                    'description': img.get('description'),
                    'associated_question': img.get('associated_question')
                }
                logging.info(f"  - Image: {img.get('image_id')} (Page {img.get('page_number')})")

            return image_mapping

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse image extraction response: {e}")
            return {}
        except Exception as e:
            logging.error(f"Error extracting images: {e}")
            return {}


class GeminiPDFChat(GeminiPDFBase):
    """PDF chat session that inherits shared PDF operations from GeminiPDFBase."""

    def __init__(self, model_name: str = GeminiPDFBase.DEFAULT_MODEL):
        """
        Initialize the PDF chat session.

        Args:
            model_name: Gemini model to use (default: gemini-2.5-flash)
        """
        super().__init__(model_name)


def main():
    """Main function to handle command line execution."""
    parser = argparse.ArgumentParser(
        description='Extract Multiple-Choice Questions from PDF using Gemini API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python gemini_mcq_extractor.py -i questions.pdf

  # With custom output directory
  python gemini_mcq_extractor.py -i questions.pdf -o ./results

  # With custom model
  python gemini_mcq_extractor.py -i questions.pdf --model gemini-1.5-pro

  # Save to custom filename
  python gemini_mcq_extractor.py -i questions.pdf -o ./results -f my_questions.json
        """
    )
    parser.add_argument('-i', '--pdf', type=str, required=True,
                      help='Path to the PDF file containing MCQs')
    parser.add_argument('-o', '--output', type=str, default='mcq_output',
                      help='Output directory for extracted MCQs and images (default: mcq_output)')
    parser.add_argument('--model', type=str, default=GeminiMCQExtractor.DEFAULT_MODEL,
                      help=f'Gemini model to use (default: {GeminiMCQExtractor.DEFAULT_MODEL})')
    parser.add_argument('-f', '--filename', type=str, default=None,
                      help='Custom filename for MCQ JSON output (default: auto-generated)')

    args = parser.parse_args()

    try:
        with GeminiMCQExtractor(model_name=args.model, output_dir=args.output) as extractor:
            extractor.load_pdf(args.pdf)

            # Extract MCQs
            result = extractor.extract_mcqs()

            # Save results
            output_file = extractor.save_results(result, args.filename)

            # Extract images
            images = extractor.extract_images()

            print("\n" + "=" * 60)
            print("MCQ Extraction Complete")
            print("=" * 60)
            print(f"Total Questions Extracted: {result.total_questions}")
            print(f"Output Directory: {args.output}")
            print(f"MCQ Data File: {output_file}")
            print(f"Images Found: {len(images)}")
            print("=" * 60 + "\n")

    except (FileNotFoundError, ValueError, ConnectionError) as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
