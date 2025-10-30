"""
PDF Paper Reviewer - Generates structured academic paper reviews using Gemini API.

This script allows you to:
- Generate structured academic paper reviews with comprehensive feedback
"""

import argparse
import logging
import os

from gemkit.paper_reviewer import ComprehensivePaperReview, PaperReviewer


# Configure logging
log_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('cli_paper_reviewer.log'),
        logging.StreamHandler()
    ]
)


def review_pdf_paper(pdf_file: str, model_name: str = "gemini-2.5-flash") -> ComprehensivePaperReview:
    """
    Reviews a PDF paper and returns a structured review.

    This function initializes a `PaperReviewer` with the specified model, and then
    calls the `review` method to generate a `ComprehensivePaperReview`.

    Args:
        pdf_file (str): The path to the PDF file of the paper to be reviewed.
        model_name (str, optional): The name of the Gemini model to use for the review.
                                    Defaults to "gemini-2.5-flash".

    Returns:
        ComprehensivePaperReview: A Pydantic model containing the structured review.

    Raises:
        FileNotFoundError: If the specified `pdf_file` does not exist.
    """
    if not os.path.exists(pdf_file):
        raise FileNotFoundError(f"PDF file not found: {pdf_file}")

    reviewer = PaperReviewer(model_name=model_name)
    return reviewer.review(pdf_file)


def main():
    """
    The main entry point for the command-line paper reviewer.

    This function parses command-line arguments, calls the `review_pdf_paper` function
    to generate a review, and then prints the review to the console. If an output
    file is specified, the review is also saved as a JSON file.
    """
    parser = argparse.ArgumentParser(
        description='PDF Paper Reviewer - Generate structured academic paper reviews with Gemini API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Review a PDF paper
  python cli_paper_reviewer.py -i paper.pdf

  # Review with custom model
  python cli_paper_reviewer.py -i paper.pdf --model gemini-1.5-pro

  # Save review to file
  python cli_paper_reviewer.py -i paper.pdf -o review.json
        """
    )

    parser.add_argument('-i', '--input', type=str, required=True, dest='pdf',
                        help='Path to the PDF paper to review')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash',
                        help='Gemini model to use (default: gemini-2.5-flash)')
    parser.add_argument('-o', '--output', type=str,
                        help='Output file to save the review (JSON format)')

    args = parser.parse_args()

    try:
        logging.info(f"Reviewing paper: {args.pdf}")
        review = review_pdf_paper(args.pdf, model_name=args.model)

        # Print review
        print("\n" + "=" * 80)
        print("PAPER REVIEW")
        print("=" * 80)
        print(review.model_dump_json(indent=2))

        # Save to file if requested
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with open(args.output, 'w') as f:
                f.write(review.model_dump_json(indent=2))
            logging.info(f"Review saved to {args.output}")

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
    except ValueError as e:
        logging.error(f"Validation error: {e}")
    except ConnectionError as e:
        logging.error(f"Connection error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
