from enum import Enum
from typing import List, Optional, Literal, Type
import os

from pydantic import BaseModel, Field, field_validator

from gemkit.pydantic_prompt_generator import PydanticPromptGenerator, PromptStyle

class ReviewCategory(str, Enum):
    """Categories for paper review aspects"""
    EXCELLENT = "excellent"
    GOOD = "good"
    SATISFACTORY = "satisfactory"
    NEEDS_IMPROVEMENT = "needs_improvement"
    POOR = "poor"

class PaperMetadata(BaseModel):
    """Metadata about the paper being reviewed"""
    title: str = Field(description="Title of the paper")
    authors: List[str] = Field(description="List of authors")
    abstract: str = Field(description="Paper abstract")
    keywords: Optional[List[str]] = Field(default=None, description="Keywords or tags")
    field_of_study: str = Field(description="Primary field or domain")
    paper_type: Literal["research", "survey", "position", "technical_report"] = Field(
        description="Type of paper"
    )

class MethodologyReview(BaseModel):
    """Review of the paper's methodology"""
    rating: ReviewCategory = Field(description="Overall methodology rating (excellent/good/satisfactory/needs_improvement/poor)")
    soundness: str = Field(min_length=20, max_length=1000, description="Assessment of scientific soundness and validity of the methodology")
    experimental_design: str = Field(min_length=20, max_length=1000, description="Evaluation of experimental design quality, controls, and setup appropriateness")
    statistical_rigor: str = Field(min_length=20, max_length=1000, description="Assessment of statistical methods, power analysis, and significance testing rigor")
    reproducibility: str = Field(min_length=20, max_length=1000, description="Evaluation of reproducibility: clarity of methods, availability of code/data, sufficient detail for replication")
    strengths: List[str] = Field(
        min_items=2,
        description="Methodological strengths (2+ items, each 1-2 sentences). List concrete strengths like 'well-designed controls', 'comprehensive datasets', 'rigorous validation approach'."
    )
    weaknesses: List[str] = Field(
        min_items=1,
        description="Methodological weaknesses (1+ items, each 1-2 sentences). Identify specific weaknesses like 'limited sample size', 'missing ablation studies', 'unclear hyperparameter choices'."
    )

class NoveltyAndContribution(BaseModel):
    """Assessment of paper's novelty and contribution"""
    rating: ReviewCategory = Field(description="Overall novelty rating (excellent/good/satisfactory/needs_improvement/poor)")
    originality: str = Field(min_length=20, max_length=1000, description="Assessment of how original and novel the work is compared to existing literature")
    significance: str = Field(min_length=20, max_length=1000, description="Evaluation of the significance and potential impact of the contribution to the field")
    incremental_vs_breakthrough: str = Field(min_length=20, max_length=1000, description="Analysis of whether the work is incremental improvement or breakthrough advance")
    comparison_to_prior_work: str = Field(min_length=20, max_length=1000, description="How well the paper compares itself to related work and positions its contributions")

class WritingAndPresentation(BaseModel):
    """Review of writing quality and presentation"""
    rating: ReviewCategory = Field(description="Overall writing quality rating (excellent/good/satisfactory/needs_improvement/poor)")
    clarity: str = Field(min_length=20, max_length=1000, description="Assessment of writing clarity: Is the paper easy to understand? Are concepts explained well? Are paragraphs coherent?")
    organization: str = Field(min_length=20, max_length=1000, description="Evaluation of paper organization and flow: Does structure support narrative? Are sections logically sequenced?")
    grammar_and_style: str = Field(min_length=20, max_length=1000, description="Assessment of grammar, punctuation, and writing style. Are there typos, grammatical errors, or awkward phrasing?")
    suggestions: List[str] = Field(
        min_items=1,
        description="Improvement suggestions (1-8 items, each 1-2 sentences). Provide specific, actionable recommendations like 'improve figure 2 label clarity', 'restructure section 3 for better flow', 'define abbreviations in text'."
    )

class BaseVisualElementReview(BaseModel):
    """Base class for reviewing individual visual elements - eliminates duplication"""
    element_id: str
    caption: Optional[str] = None
    element_type: str
    rating: ReviewCategory

    # Generic assessments applicable across element types
    clarity: str
    relevance: str
    caption_quality: str

    # Specific domain assessments (optional based on element type)
    technical_assessment: Optional[str] = None
    format_assessment: Optional[str] = None

    issues: List[str] = Field(
        default_factory=list,
        description="Specific issues found with this element (1-2 sentences each)"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Improvement suggestions (1-2 sentences each)"
    )


class VisualElementsCollectionReview(BaseModel):
    """Review of all visual elements of one type - replaces Table/Figure/Diagram/Image/EquationsReview"""
    element_type: str = Field(description="Type of elements reviewed (table/figure/diagram/image/equation)")
    elements: List[BaseVisualElementReview] = Field(
        default_factory=list,
        description="Individual reviews of each element of this type"
    )
    overall_rating: Optional[ReviewCategory] = Field(
        default=None,
        description="Overall quality rating for all elements of this type (optional if no elements)"
    )

    consistency_assessment: Optional[str] = Field(
        default=None,
        description="Consistency assessment across all elements of this type (formatting, styling, conventions, field standards)"
    )

    global_suggestions: List[str] = Field(
        default_factory=list,
        description="Global improvement suggestions affecting multiple or all elements of this type"
    )

# ============================================================================
# Visual Element Review Classes - Individual Elements
# ============================================================================

class TableReview(BaseVisualElementReview):
    """Review of a single table"""
    table_id: str = Field(description="Table identifier (e.g., 'Table 1', 'Table 2a')")
    element_id: str = Field(description="Element identifier (inherited from base)")
    element_type: str = Field(default="table", description="Type of element")


class FigureReview(BaseVisualElementReview):
    """Review of a single figure"""
    figure_id: str = Field(description="Figure identifier (e.g., 'Figure 1', 'Figure 3b')")
    chart_type: Optional[str] = Field(default=None, description="Type of chart/visualization")
    element_id: str = Field(description="Element identifier (inherited from base)")
    element_type: str = Field(default="figure", description="Type of element")


class DiagramReview(BaseVisualElementReview):
    """Review of a single diagram"""
    diagram_id: str = Field(description="Diagram identifier (e.g., 'Figure 2', 'Algorithm 1')")
    diagram_type: Optional[str] = Field(default=None, description="Type of diagram")
    element_id: str = Field(description="Element identifier (inherited from base)")
    element_type: str = Field(default="diagram", description="Type of element")


class ImageReview(BaseVisualElementReview):
    """Review of a single image"""
    image_id: str = Field(description="Image identifier (e.g., 'Figure 6', 'Photo 2')")
    image_type: Optional[str] = Field(default=None, description="Type of image")
    element_id: str = Field(description="Element identifier (inherited from base)")
    element_type: str = Field(default="image", description="Type of element")


class EquationReview(BaseVisualElementReview):
    """Review of a single equation"""
    equation_id: str = Field(description="Equation identifier (e.g., 'Equation 1', 'Eq. 3.2')")
    context_section: Optional[str] = Field(default=None, description="Section where equation appears")
    element_id: str = Field(description="Element identifier (inherited from base)")
    element_type: str = Field(default="equation", description="Type of element")


# ============================================================================
# Visual Element Collection Classes - Collections of Element Reviews
# ============================================================================

class TablesReview(VisualElementsCollectionReview):
    """Review of all tables"""
    tables: List[TableReview] = Field(
        default_factory=list,
        description="Individual reviews of each table in the paper"
    )
    general_consistency: Optional[str] = Field(default=None, description="Consistency assessment across tables")
    element_type: str = Field(default="table", description="Type of elements")


class FiguresReview(VisualElementsCollectionReview):
    """Review of all figures"""
    figures: List[FigureReview] = Field(
        default_factory=list,
        description="Individual reviews of each figure in the paper"
    )
    general_consistency: Optional[str] = Field(default=None, description="Consistency assessment across figures")
    element_type: str = Field(default="figure", description="Type of elements")


class DiagramsReview(VisualElementsCollectionReview):
    """Review of all diagrams"""
    diagrams: List[DiagramReview] = Field(
        default_factory=list,
        description="Individual reviews of each diagram in the paper"
    )
    general_consistency: Optional[str] = Field(default=None, description="Consistency assessment across diagrams")
    element_type: str = Field(default="diagram", description="Type of elements")


class ImagesReview(VisualElementsCollectionReview):
    """Review of all images"""
    images: List[ImageReview] = Field(
        default_factory=list,
        description="Individual reviews of each image in the paper"
    )
    general_consistency: Optional[str] = Field(default=None, description="Consistency assessment across images")
    element_type: str = Field(default="image", description="Type of elements")


class EquationsReview(VisualElementsCollectionReview):
    """Review of all equations"""
    equations: List[EquationReview] = Field(
        default_factory=list,
        description="Individual reviews of each significant equation in the paper"
    )
    notation_consistency: Optional[str] = Field(default=None, description="Consistency of mathematical notation")
    mathematical_rigor: Optional[str] = Field(default=None, description="Assessment of mathematical rigor")
    element_type: str = Field(default="equation", description="Type of elements")


class VisualElementsReview(BaseModel):
    """Comprehensive review of all visual elements - SIMPLIFIED VERSION"""
    overall_rating: ReviewCategory = Field(
        description="Overall visual elements rating (excellent/good/satisfactory/needs_improvement/poor)"
    )

    # All element types in a single unified collection
    all_elements: Optional[List[VisualElementsCollectionReview]] = Field(
        default=None,
        description="Reviews organized by element type (table, figure, diagram, image, equation) - unified structure"
    )


    # Cross-element assessments
    visual_consistency: str = Field(
        description="Cross-visual consistency: color schemes, font sizes, style, notation conventions, and adherence to field standards"
    )
    caption_consistency: str = Field(
        description="Caption consistency: Are captions formatted consistently in style, structure, detail level, and notation?"
    )

    global_issues: List[str] = Field(
        default_factory=list,
        description="Global visual issues affecting multiple element types"
    )
    global_suggestions: List[str] = Field(
        default_factory=list,
        description="Global improvement suggestions for visual presentation across the entire paper"
    )

class LiteratureReview(BaseModel):
    """Assessment of literature review and citations"""
    rating: ReviewCategory = Field(description="Literature review quality rating (excellent/good/satisfactory/needs_improvement/poor)")
    comprehensiveness: str = Field(description="Assessment of literature review comprehensiveness and coverage")
    gaps_identified: List[str] = Field(
        default_factory=list,
        description="Identified gaps in literature coverage (0-5 items)"
    )
    citation_accuracy: str = Field(description="Assessment of citation accuracy and appropriateness")

class ResultsAndAnalysis(BaseModel):
    """Review of results and their analysis"""
    rating: ReviewCategory = Field(description="Results quality rating (excellent/good/satisfactory/needs_improvement/poor)")
    result_quality: str = Field(min_length=20, max_length=1000, description="Assessment of result quality: Are results clear, well-presented, and support the claims?")
    interpretation: str = Field(min_length=20, max_length=1000, description="Evaluation of how well results are interpreted. Are conclusions drawn appropriately? Any over-interpretation?")
    limitations_assessment: str = Field(min_length=20, max_length=1000, description="Assessment of how well the paper discusses limitations and potential impact on conclusions")
    limitations_discussed: bool = Field(
        description="Whether paper explicitly addresses limitations. True if limitations section/discussion exists; False if missing or inadequate."
    )

class EthicalConsiderations(BaseModel):
    """Ethical aspects of the research"""
    ethical_concerns: Optional[str] = None
    data_privacy: Optional[str] = None
    bias_assessment: Optional[str] = None
    reproducibility_ethics: Optional[str] = None

class IntroductionQuality(BaseModel):
    """Assessment of introduction effectiveness and focus"""
    focus_rating: ReviewCategory = Field(
        description="How well introduction stays focused on relevant background (excellent/good/satisfactory/needs_improvement/poor)"
    )
    unnecessary_content: bool = Field(
        description="True if introduction contains unnecessary background, 'grandmotherly stories', or excessive historical context; False if content is all relevant and necessary"
    )
    relevance_analysis: str = Field(min_length=20, max_length=1000, description="Detailed analysis of introduction content relevance to the paper's research problem and objectives")
    issues: List[str] = Field(
        default_factory=list,
        description="Specific issues found in introduction. Examples: 'Paragraph 2: extensive history of field not directly relevant to this work', 'Overly long motivation section (3 pages) that could be condensed', 'Too much background on foundational concepts already well-known in the field', 'Missing connection between background and the specific research problem addressed'"
    )

class ClaimsAccuracy(BaseModel):
    """Assessment of claim accuracy and evidence support"""
    over_claiming: bool = Field(
        description="True if paper makes unsupported or exaggerated claims; False if claims are well-supported and measured"
    )
    evidence_support: ReviewCategory = Field(
        description="How well claims are supported by evidence/results (excellent/good/satisfactory/needs_improvement/poor)"
    )
    factual_accuracy: str = Field(min_length=20, max_length=1000, description="Assessment of factual accuracy and whether claims align with presented evidence")
    claim_strength: List[str] = Field(
        default_factory=list,
        description="Specific claims analyzed for accuracy. Format: 'Well-supported: [claim] ([evidence location])' or 'Oversold: [claim] ([why it's exaggerated])'. Examples: 'Well-supported: 10.6% improvement on agent benchmark (Table 1)', 'Oversold: claims superiority without mentioning baseline differences'"
    )

class AIGenerationDetection(BaseModel):
    """Assessment of whether the paper appears to be AI-generated"""
    is_likely_ai_generated: bool = Field(
        description="True if paper shows strong indicators of AI generation; False if appears human-authored; uncertain cases should be marked False with caveats in analysis"
    )
    authenticity_score: Literal["authentic", "mixed_signals", "likely_ai", "highly_likely_ai"] = Field(
        description="Authenticity assessment: 'authentic' (human-authored), 'mixed_signals' (unclear), 'likely_ai' (multiple AI indicators), 'highly_likely_ai' (strong AI generation signals)"
    )
    analysis: str
    indicators: List[str] = Field(
        default_factory=list,
        description="Specific indicators observed (if any). Examples: 'Generic transitions between sections', 'Repetitive phrasing patterns', 'Over-use of hedging language', 'Inconsistent technical depth', 'Clichéd examples', 'Natural voice and novel insights', 'Sophisticated argumentation', 'Unexpected turns in reasoning'"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the authenticity assessment. 'high' = clear signals, 'medium' = mixed indicators, 'low' = insufficient evidence to determine"
    )

class SpecificIssues(BaseModel):
    """Specific issues or concerns"""
    major_issues: List[str] = Field(
        default_factory=list,
        description="Major concerns (0-10 items, each 1-2 sentences). Critical issues affecting publication: fundamental flaws, wrong conclusions, missing experiments, serious limitations. Format: 'Issue: [description] Impact: [consequence]'"
    )
    minor_issues: List[str] = Field(
        default_factory=list,
        description="Minor concerns (0-15 items, each 1 sentence). Non-critical issues: typos, small improvements, clarifications needed. Example: 'Figure 3 caption lacks units' or 'Section 2.1 could be clearer'"
    )
    questions_for_authors: List[str] = Field(
        default_factory=list,
        description="Questions for authors (0-10 items). Specific, answerable questions about methodology, results, or claims. Each should be clear and help validate the work. Example: 'How were hyperparameters selected?' or 'What is the statistical significance of Table 2 differences?'"
    )

class OverallAssessment(BaseModel):
    """Overall strengths and weaknesses"""
    strengths: List[str] = Field(
        min_items=3,
        description="Key strengths of the paper (3-6 items, each 1-2 sentences). Highlight the most important positive aspects. Be specific and avoid generic praise. Example: 'Novel application of transformer architecture to time-series forecasting' vs 'Good paper'"
    )
    weaknesses: List[str] = Field(
        min_items=3,
        description="Key weaknesses of the paper (3-6 items, each 1-2 sentences). Identify the most significant limitations affecting publication decision. Be constructive and specific. Example: 'Limited to English language datasets' vs 'Limited scope'"
    )

class ExecutiveSummary(BaseModel):
    """Executive summary of the review"""
    summary: str

class DetailedFeedback(BaseModel):
    """Detailed feedback section"""
    comments_for_authors: str = Field(
        min_length=200,
        description="Detailed comments for authors (200-2000 chars). Comprehensive, constructive feedback addressing all review aspects. Be specific with examples. Structure: key strengths → key concerns → specific suggestions. Tone: professional, respectful, educational."
    )
    confidential_comments_to_editor: Optional[str] = Field(
        default=None,
        description="Confidential editor comments (20-500 chars, optional). Private feedback for editor only: publication recommendation rationale, strategic concerns, or handling suggestions. Not shared with authors."
    )


class RecommendationDetails(BaseModel):
    """Detailed recommendation breakdown"""
    decision: Literal[
        "strong_accept",
        "accept",
        "weak_accept",
        "borderline",
        "weak_reject",
        "reject",
        "strong_reject"
    ] = Field(
        description="Publication recommendation. Choices: 'strong_accept' (excellent, accept immediately), 'accept' (good, minor revisions), 'weak_accept' (acceptable with concerns), 'borderline' (cannot decide), 'weak_reject' (flawed, major revisions needed), 'reject' (significant problems), 'strong_reject' (fundamental issues)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in the review. 'high' = confident in assessment, 'medium' = reasonably confident, 'low' = uncertain, external expert opinion recommended"
    )
    justification: str = Field(
        min_length=100,
        description="Justification for decision (100-700 chars). Brief explanation of why this recommendation was chosen. Link to strengths, weaknesses, and significance. Explain how it aligns with publication standards."
    )
    conditions_for_acceptance: Optional[List[str]] = Field(
        default=None,
        description="Conditions for acceptance (0-5 items, optional). For 'weak_accept' or 'borderline' decisions, specify what authors must address. Example: 'Provide ablation studies', 'Compare to baseline X', 'Clarify Section 3 methodology'"
    )

class ComprehensivePaperReview(BaseModel):
    """Complete structured paper review with optimized review flow.

    Review phases in order:
    1. Authentication & Structure (AI detection, metadata)
    2. Core Content Review (introduction → methodology → results → novelty → claims → literature)
    3. Presentation Quality (writing, visual elements)
    4. Broader Context (ethical considerations)
    5. Synthesis & Decision (overall assessment, issues, feedback, recommendation)
    6. Summary (executive summary)
    """

    # PHASE 1: AUTHENTICATION & STRUCTURE
    ai_generation: AIGenerationDetection = Field(
        description="Assessment of whether paper appears to be AI-generated. Evaluate first as it affects credibility of all reviews."
    )
    metadata: PaperMetadata = Field(
        description="Paper metadata (title, authors, abstract, keywords, field, type)"
    )

    # PHASE 2: CORE CONTENT REVIEW
    introduction_quality: IntroductionQuality = Field(
        description="Assessment of introduction effectiveness: focus, relevance, and unnecessary content. Review first to understand paper's foundation."
    )
    methodology: MethodologyReview = Field(
        description="Methodology review: soundness, experimental design, statistical rigor, and reproducibility"
    )
    results: ResultsAndAnalysis = Field(
        description="Results analysis: quality of findings, interpretation, and limitations discussion"
    )
    novelty: NoveltyAndContribution = Field(
        description="Novelty assessment: originality, significance, and positioning relative to prior work"
    )
    claims_accuracy: ClaimsAccuracy = Field(
        description="Assessment of claim accuracy and evidence support. Validate against results presented."
    )
    literature: LiteratureReview = Field(
        description="Literature review assessment: comprehensiveness, coverage gaps, and citation accuracy"
    )

    # PHASE 3: PRESENTATION QUALITY
    writing: WritingAndPresentation = Field(
        description="Writing quality review: clarity, organization, grammar, and style suggestions"
    )
    visual_elements: VisualElementsReview = Field(
        description="Comprehensive review of tables, figures, diagrams, and images: clarity, relevance, consistency"
    )

    # PHASE 4: BROADER CONTEXT
    ethical_considerations: Optional[EthicalConsiderations] = Field(
        default=None,
        description="Ethical considerations (optional - include if paper involves sensitive data, bias, or ethics issues)"
    )

    # PHASE 5: SYNTHESIS & DECISION
    overall_assessment: OverallAssessment = Field(
        description="Key strengths and weaknesses synthesis. Identify most important positive and negative aspects."
    )
    specific_issues: SpecificIssues = Field(
        description="Specific issues identified: major concerns, minor issues, and questions for authors"
    )
    detailed_feedback: DetailedFeedback = Field(
        description="Detailed feedback for authors: comprehensive comments and editor confidential notes"
    )
    recommendation: RecommendationDetails = Field(
        description="Final publication recommendation with confidence level and conditions for acceptance"
    )

    # PHASE 6: SUMMARY
    executive_summary: ExecutiveSummary = Field(
        description="Brief executive summary of the entire review. Synthesize key points from all phases."
    )



# System prompt for academic paper review with critical instructions
REVIEW_SYSTEM_PROMPT = """You are an expert academic peer reviewer with deep knowledge across multiple disciplines.
Your task is to provide a comprehensive, constructive, and rigorous review of the submitted paper.

**CRITICAL INSTRUCTIONS:**

1. **Specific Evidence:** ALWAYS cite specific sections, figures, tables, or equations from the paper. NEVER provide generic assessments.
2. **Ratings:** Excellent (no major flaws, advances field), Good (sound with minor issues), Satisfactory (acceptable with limits), Needs Improvement (significant flaws), Poor (fundamental problems).
3. **Specificity:** "Uses novel attention mechanism for temporal dependencies" (good) vs "Well-written paper" (bad).
4. **Questions:** Ask specific, answerable questions that validate the work. Avoid yes/no questions.
5. **Issues:** Major = affects validity or has fundamental flaws. Minor = typos, formatting, small clarifications.
6. **Consistency:** Recommendations should align with ratings (Excellent/Good → Accept; Poor/Needs Improvement → Reject).
7. **Visuals:** Evaluate tables, figures, diagrams, images for clarity, relevance, consistency, and proper labeling."""


class PaperReviewer:
    """
    A class for generating comprehensive, structured reviews of academic papers
    using the Gemini API.

    This class uses a detailed Pydantic model, `ComprehensivePaperReview`, to
    ensure that the generated review is well-structured and covers all the
    key aspects of a high-quality academic review.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", prompt_style: PromptStyle = PromptStyle.DETAILED):
        """
        Initializes the PaperReviewer.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
            prompt_style (PromptStyle, optional): The style of the prompt to generate.
        """
        self.model_name = model_name
        self.prompt_style = prompt_style
        self._schema_prompt = self._generate_schema_prompt()

    def _generate_schema_prompt(self) -> str:
        """
        Generate schema prompt using PydanticPromptGenerator.

        Returns:
            Schema prompt string
        """
        generator = PydanticPromptGenerator(
            ComprehensivePaperReview,
            style=self.prompt_style,
            include_examples=True
        )
        return generator.generate_prompt()

    def review(self, pdf_file: str) -> ComprehensivePaperReview:
        """
        Reviews a PDF paper and returns a structured review.

        This method uploads the PDF, sends a detailed prompt to the Gemini API,
        and returns a `ComprehensivePaperReview` object with the structured
        review.

        Args:
            pdf_file (str): The path to the PDF file of the paper to be reviewed.

        Returns:
            ComprehensivePaperReview: A Pydantic model containing the structured review.
        """
        # Validate file exists
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")

        # Validate file is a PDF
        if not pdf_file.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF file (got {pdf_file})")

        # Import here to avoid circular imports
        from gemkit.gemini_pdf_chat import GeminiPDFChat

        try:
            with GeminiPDFChat(model_name=self.model_name) as chat:
                chat.load_pdf(pdf_file)

                prompt = f"""{REVIEW_SYSTEM_PROMPT}

{self._schema_prompt}

Please review the provided PDF paper."""

                review = chat.generate_text(prompt, response_schema=ComprehensivePaperReview)
                return review
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Failed to load PDF: {str(e)}") from e
        except ValueError as e:
            raise ValueError(f"Invalid schema or data: {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Review generation failed: {str(e)}") from e
