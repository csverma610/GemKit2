import argparse
import json
import logging
from typing import Optional, Type

from pydantic import BaseModel, ValidationError
from gemini_pdf_base import PDFAnalysisClient

# Configure logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('pdf_chat.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GeminiPDFChat(PDFAnalysisClient):
    """
    An interactive chat interface for analyzing PDF documents.

    This class extends `PDFAnalysisClient` to provide a chat-like interface for
    asking questions about a loaded PDF. It supports both plain text and structured
    (Pydantic model) responses.
    """

    def generate_text(self, prompt: str, response_schema: Optional[Type[BaseModel]] = None) -> str | BaseModel:
        """
        Sends a prompt to the Gemini model and returns the response.

        Args:
            prompt (str): The user's question or prompt.
            response_schema (Optional[Type[BaseModel]], optional): A Pydantic model to use for
                                                                    structured output.

        Returns:
            str | BaseModel: If a `response_schema` is provided, a Pydantic model instance
                             is returned. Otherwise, a plain text string is returned.

        Raises:
            RuntimeError: If no PDF is loaded, the response is blocked by the API,
                          or the response cannot be parsed.
        """
        self._check_pdf_loaded()

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

        self._validate_api_response(response)

        if response_schema:
            # If parsed response available, use it
            if response.parsed is not None:
                return response.parsed
            # Otherwise, manually parse JSON text
            try:
                data = json.loads(response.text)
                return response_schema(**data)
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse response: {e}\nResponse text: {response.text}")
                raise

        return response.text

def main():
    """Main function to handle command line execution."""
    parser = argparse.ArgumentParser(
        description='Chat with PDF using Gemini API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python gemini_pdf_chat.py -i document.pdf

  # With initial question
  python gemini_pdf_chat.py -i document.pdf -q "Summarize this"

  # With custom model
  python gemini_pdf_chat.py -i document.pdf --model gemini-2.5-flash

Note: For paper reviews with structured output, use cli_paper_reviewer.py instead.
        """
    )
    parser.add_argument('-i', '--pdf', type=str, required=True,
                      help='Path to the PDF file to analyze')
    parser.add_argument('--model', type=str, default=GeminiPDFChat.DEFAULT_MODEL,
                      help=f'Gemini model to use (default: {GeminiPDFChat.DEFAULT_MODEL})')
    parser.add_argument('-q', '--question', type=str,
                      help='Optional question to ask about the PDF')

    args = parser.parse_args()

    try:
        with GeminiPDFChat(model_name=args.model) as pdfchat:
            pdfchat.load_pdf(args.pdf)

            if args.question:
                print(f"\nQuestion: {args.question}")
                response = pdfchat.generate_text(args.question)
                _print_response(response)

            interactive_mode(pdfchat)

    except (FileNotFoundError, ValueError, ConnectionError, RuntimeError) as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")

def _print_response(response: str | BaseModel):
    """Print response, formatting as JSON if it's a Pydantic model."""
    if isinstance(response, BaseModel):
        print(f"\nAssistant: {response.model_dump_json(indent=2)}\n")
    else:
        print(f"\nAssistant: {response}\n")


def interactive_mode(chat: GeminiPDFChat):
    """
    Start an interactive chat session.

    Args:
        chat: GeminiPDFChat instance
    """
    if not chat.uploaded_file:
        logging.error("Cannot start interactive mode, no PDF loaded.")
        return

    print("=" * 60)
    print("PDF Chat Session Started")
    print("=" * 60)
    print("Commands:")
    print("  - Type your question to chat with the PDF")
    print("  - Type 'exit' or 'quit' to end the session")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!")
                break

            response = chat.generate_text(user_input)
            _print_response(response)

        except KeyboardInterrupt:
            print("\n\nSession interrupted. Goodbye!")
            break
        except Exception as e:
            logging.error(f"Error: {e}\n")

if __name__ == "__main__":
    main()
