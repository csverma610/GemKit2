
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, List, Optional

from google import genai
from pydantic import BaseModel, Field, ValidationError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Configure logging to file only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gemini_paper_analysis.log',
    filemode='a'
)
logger = logging.getLogger(__name__)


# ====================================================================
# PYDANTIC MODELS
# (Custom printing methods removed, now plain Pydantic models)
# ====================================================================

class MainIdea(BaseModel):
    """Main research idea or contribution"""
    title: str = Field(
        description="Concise title summarizing the core idea (max 100 characters)"
    )
    description: str = Field(
        description="Comprehensive explanation of the main idea, including problem statement, proposed solution, and significance"
    )


class Novelty(BaseModel):
    """Novelty and innovation aspects"""
    novel_aspects: List[str] = Field(
        description="List of specific novel contributions, innovations, or unique approaches introduced in this work"
    )
    comparison_with_prior_work: str = Field(
        description="Detailed comparison explaining how this work advances beyond existing research, highlighting key differences and improvements"
    )
    innovation_score: int = Field(
        ge=1,
        le=10,
        description="Innovation rating from 1 (incremental improvement) to 10 (groundbreaking breakthrough)"
    )


class LiteratureReview(BaseModel):
    """Literature review quality assessment"""
    coverage: str = Field(
        description="Assessment of breadth and depth of literature coverage, including whether key works and recent developments are adequately referenced"
    )
    relevance: str = Field(
        description="Evaluation of how well cited works relate to the research problem, methodology, and findings"
    )
    recency: str = Field(
        description="Analysis of citation timeline, noting whether recent literature (last 2-5 years) is adequately represented"
    )
    quality_score: int = Field(
        ge=1,
        le=10,
        description="Literature review quality from 1 (insufficient) to 10 (comprehensive and excellent)"
    )
    gaps_identified: List[str] = Field(
        description="Research gaps or limitations in existing literature that this paper aims to address"
    )


class Methodology(BaseModel):
    """Research methodology"""
    approach: str = Field(
        description="Overall research approach (e.g., experimental, theoretical, empirical, computational) and its appropriateness for the research questions"
    )
    techniques: List[str] = Field(
        description="Specific methods, algorithms, statistical techniques, or experimental procedures employed in the study"
    )
    datasets: List[str] = Field(
        description="Datasets used for experiments, including names, sizes, sources, and characteristics"
    )
    experimental_design: str = Field(
        description="Detailed description of experimental setup, controls, variables, protocols, and validation strategies"
    )


class ResultAnalysis(BaseModel):
    """Analysis of results"""
    key_findings: List[str] = Field(
        description="Primary findings and discoveries from the research, stated clearly and specifically"
    )
    metrics_used: List[str] = Field(
        description="Evaluation metrics and measurements used to assess performance or validate hypotheses"
    )
    performance_summary: str = Field(
        description="Summary of quantitative and qualitative performance results, including key numbers and trends"
    )
    comparison_with_baselines: str = Field(
        description="Detailed comparison of results against baseline methods, prior work, or control conditions, including relative improvements"
    )
    statistical_significance: str = Field(
        description="Analysis of statistical significance, confidence intervals, p-values, and reliability of results"
    )
    limitations: List[str] = Field(
        description="Acknowledged limitations, constraints, potential biases, or threats to validity in the methodology or results"
    )


class Conclusion(BaseModel):
    """Conclusion and future work"""
    summary: str = Field(
        description="Concise summary of main conclusions drawn from the research findings"
    )
    contributions: List[str] = Field(
        description="Specific contributions to the field, including theoretical, methodological, or practical advances"
    )
    future_work: List[str] = Field(
        description="Suggested directions for future research, open questions, or potential extensions of this work"
    )
    impact: str = Field(
        description="Assessment of potential impact on the field, practical applications, and broader implications"
    )


class StrengthsWeaknesses(BaseModel):
    """Strengths and weaknesses assessment"""
    strengths: List[str] = Field(
        description="Major strengths of the paper including methodological rigor, clarity, novelty, and significant contributions"
    )
    weaknesses: List[str] = Field(
        description="Major weaknesses, concerns, or areas needing improvement including methodological issues, unclear explanations, or insufficient evidence"
    )
    recommendations: List[str] = Field(
        description="Specific, actionable recommendations for improving the paper's quality, clarity, or impact"
    )


class PaperAnalysis(BaseModel):
    """Complete research paper analysis (Top-level model)"""
    title: str = Field(
        description="Full title of the research paper as it appears in the document"
    )
    authors: str = Field(
        description="Complete list of authors with affiliations if available"
    )
    abstract_summary: str = Field(
        description="Concise summary of the abstract capturing the paper's purpose, methods, and key findings"
    )
    main_ideas: List[MainIdea] = Field(
        description="List of main research ideas, hypotheses, or proposals presented in the paper"
    )
    novelty: Novelty = Field(
        description="Assessment of novelty, innovation, and unique contributions of this work"
    )
    literature_review: LiteratureReview = Field(
        description="Quality assessment of the literature review and positioning of the work"
    )
    methodology: Methodology = Field(
        description="Description and evaluation of research methodology and experimental design"
    )
    results_analysis: ResultAnalysis = Field(
        description="Analysis of research results, findings, and their significance"
    )
    conclusion: Conclusion = Field(
        description="Summary of conclusions, contributions, and future research directions"
    )
    strengths_weaknesses: StrengthsWeaknesses = Field(
        description="Critical assessment of paper strengths, weaknesses, and improvement recommendations"
    )
    overall_assessment: str = Field(
        description="Comprehensive overall assessment of paper quality, contribution, and suitability for publication"
    )
    overall_score: int = Field(
        ge=1,
        le=10,
        description="Overall quality score from 1 (poor quality, major issues) to 10 (exceptional quality, significant contribution)"
    )


# ====================================================================
# ANALYZER CLASS
# ====================================================================

class ResearchPaperAnalyzer:
    """Analyzer for research papers using Gemini API"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gemini-2.0-flash',
        max_retries: int = 3
    ):
        """Initialize with Gemini API key using the standard genai namespace.

        Args:
            api_key: Gemini API key (if None, reads from GEMINI_API_KEY env var)
            model_name: Model to use (default: gemini-2.0-flash-exp)
            max_retries: Maximum number of retry attempts (default: 3)

        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError(
                "API key is required. Provide it as a parameter or set GEMINI_API_KEY environment variable."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.uploaded_files = []  # Track uploaded files for cleanup
        self.analysis: Optional[PaperAnalysis] = None # Added attribute to store analysis result
        logger.info(f"Initialized ResearchPaperAnalyzer with model: {model_name} using standard client via genai.")

    def upload_pdf(self, pdf_path: str) -> Any:
        """Upload PDF to Gemini with error handling

        Args:
            pdf_path: Path to PDF file

        Returns:
            Uploaded file object

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ValueError: If file is not a PDF
            Exception: For upload failures
        """
        path = self._validate_pdf_path(pdf_path)
        file_size_mb = path.stat().st_size / (1024 * 1024)

        logger.info(f"Uploading PDF: {path} ({file_size_mb:.2f}MB)")

        try:
            pdf_file = self._upload_with_retry(str(path))
            self.uploaded_files.append(pdf_file)
            logger.info(f"PDF uploaded successfully: {pdf_file.name}")
            return pdf_file
        except Exception as e:
            logger.error(f"Failed to upload PDF: {e}")
            raise

    def cleanup_uploaded_files(self):
        """Delete uploaded files from Gemini storage using the modern client."""
        for file in self.uploaded_files:
            try:
                # Using client method (self.client.files.delete)
                self.client.files.delete(name=file.name)
                logger.info(f"Deleted uploaded file: {file.name}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file.name}: {e}")
        self.uploaded_files.clear()

    # Private methods

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _upload_with_retry(self, pdf_path: str) -> Any:
        """Upload PDF with automatic retry logic using the modern client."""
        # Using client method (self.client.files.upload)
        return self.client.files.upload(file=pdf_path)

    def _validate_pdf_path(self, pdf_path: str) -> Path:
        """Validate PDF file path

        Args:
            pdf_path: Path to PDF file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is not a file or not a PDF
        """
        # Remove quotes if user wrapped path in quotes
        pdf_path = pdf_path.strip('"').strip("'")

        path = Path(pdf_path)

        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Validate it's a file
        if not path.is_file():
            raise ValueError(f"Path is not a file: {pdf_path}")

        # Validate it's a PDF
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF, got: {path.suffix}")

        # Check file size (Gemini has limits)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > 200:  # 200MB limit for Gemini
            raise ValueError(f"PDF file too large: {file_size_mb:.2f}MB (max 200MB)")

        return path

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def _call_llm(self, pdf_file: Any, prompt: str) -> dict:
        """Generate analysis with automatic retry logic using the modern client and JSON mode.

        Args:
            pdf_file: Uploaded PDF file object
            prompt: Analysis prompt

        Returns:
            Parsed JSON dict
        """
        # Using client method (self.client.models.generate_content)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[pdf_file, prompt],
            config={
                "response_mime_type": "application/json",
                "response_schema": PaperAnalysis,
            }
        )
        return json.loads(response.text)

    def analyze_paper(self, pdf_path: str) -> PaperAnalysis:
        """Analyze research paper comprehensively using structured JSON output

        Args:
            pdf_path: Path to PDF file

        Returns:
            PaperAnalysis object

        Raises:
            Exception: For analysis failures
        """
        try:
            # Upload PDF
            pdf_file = self.upload_pdf(pdf_path)

            # Create detailed analysis prompt
            prompt = """
You are an expert research paper reviewer. Analyze this research paper comprehensively and provide a detailed assessment.

Please analyze the following aspects:

1. **Title, Authors, and Abstract**: Extract the paper title, authors, and summarize the abstract.

2. **Main Ideas and Proposals**: What are the core ideas, hypotheses, or proposals in this paper? What problem does it solve?

3. **Novelty and Innovation**:
    - What are the novel contributions?
    - How does this work differ from prior research?
    - Rate the innovation level (1-10)

4. **Literature Review Quality**:
    - Is the literature review comprehensive?
    - Are the citations relevant and recent?
    - What research gaps are identified?
    - Rate the literature review quality (1-10)

5. **Methodology**:
    - What methodological approach is used?
    - What specific techniques, algorithms, or methods are employed?
    - What datasets are used?
    - Describe the experimental design

6. **Results Analysis**:
    - What are the key findings?
    - What evaluation metrics are used?
    - How do results compare to baselines?
    - Is there statistical significance?
    - What limitations are present?

7. **Conclusion and Future Work**:
    - What are the main conclusions?
    - What are the claimed contributions?
    - What future work is suggested?
    - What is the potential impact?

8. **Strengths and Weaknesses**:
    - List major strengths of the paper
    - List major weaknesses or concerns
    - Provide recommendations for improvement

9. **Overall Assessment**:
    - Provide an overall assessment of the paper quality
    - Give an overall score (1-10)

Be thorough, critical, and constructive in your analysis.
"""

            # Generate analysis with structured JSON output
            logger.info("Analyzing paper with Gemini API using JSON mode...")

            analysis_data = self._call_llm(pdf_file, prompt)

            # Validate with Pydantic
            logger.info("Validating analysis structure...")
            try:
                paper_analysis = PaperAnalysis(**analysis_data)
                self.analysis = paper_analysis 
                logger.info("Analysis completed successfully")
                return paper_analysis
            except ValidationError as e:
                logger.error(f"Pydantic validation failed: {e}")
                logger.debug(f"Response data: {json.dumps(analysis_data, indent=2)[:500]}...")
                raise ValueError(f"Response doesn't match expected schema: {e}")

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def save(self, output_path: str):
        """Save structured analysis data to a JSON file using the json library.
        
        It retrieves the PaperAnalysis object from the instance attribute self.analysis.

        Args:
            output_path: Path to save JSON file
        """
        if self.analysis is None:
            logger.error("No analysis data available to save.")
            raise ValueError("No analysis data available to save. Run analyze_paper first.")

        try:
            # Use Pydantic's model_dump() for clean JSON serialization
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.analysis.model_dump(), f, indent=2, ensure_ascii=False)
            logger.info(f"Analysis saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            raise


def get_output_path(pdf_path: str) -> str:
    """Generate output path for analysis JSON

    Args:
        pdf_path: Path to input PDF

    Returns:
        Path for output JSON file
    """
    path = Path(pdf_path)
    # Replace extension with _analysis.json
    return str(path.parent / f"{path.stem}_analysis.json")


def main():
    """Main execution function, using argparse for client interface."""
    parser = argparse.ArgumentParser(
        description="Analyze a research paper PDF using the Gemini API and output a structured report."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the research paper PDF file."
    )
    args = parser.parse_args()
    pdf_path = args.pdf_path

    try:
        # Initialize analyzer (API key auto-detected from environment)
        analyzer = ResearchPaperAnalyzer()

        # Analyze paper
        print("\nStarting comprehensive paper analysis...\n")
        analysis = analyzer.analyze_paper(pdf_path)

        # Determine output path (only JSON output remains)
        output_json_path = get_output_path(pdf_path)

        # Save analysis to JSON file using the save method
        # The 'analysis' object is now stored internally by the analyzer
        print(f"Saving structured JSON analysis to: {output_json_path}...")
        analyzer.save(output_json_path)

        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to JSON: {output_json_path}")

    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except ValueError as e:
        print(f"\nError: {e}")
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        logger.error(f"Analysis failed: {e}", exc_info=True)
    finally:
        # Clean up uploaded files
        try:
            if 'analyzer' in locals():
                analyzer.cleanup_uploaded_files()
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


if __name__ == "__main__":
    main()

