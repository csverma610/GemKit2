import argparse
import json
import logging
import os
import pathlib
from enum import Enum
from typing import List, Optional, Dict

from google import genai
from pydantic import BaseModel, Field, ValidationError
from pydantic_prompt_generator import PydanticPromptGenerator, PromptStyle

# Configure logging with enhanced formatting
log_format = "%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler('pdf_proofreader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_schema_prompt(model: BaseModel, instructions: str = "") -> str:
    """
    Generate a hybrid prompt combining auto-generated schema with custom instructions.

    Args:
        model: Pydantic model to generate schema from
        instructions: Custom instructions to include before schema

    Returns:
        Combined prompt with instructions and structured schema
    """
    try:
        generator = PydanticPromptGenerator(
            model,
            style=PromptStyle.DETAILED,
            include_examples=True,
            validate_schema=True
        )
        schema_prompt = generator.generate_prompt()

        if instructions:
            return f"{instructions}\n\n{schema_prompt}"
        return schema_prompt
    except Exception as e:
        logger.warning(f"Failed to generate schema prompt for {model.__name__}: {e}")
        # Fallback: return instructions if schema generation fails
        return instructions


# Priority tiers for issues
class IssuePriority(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"

class Issue(BaseModel):
    description: str = Field(description="Description of the issue")
    correction: str = Field(description="Single, justified correction to apply")
    rationale: str = Field(description="Why this correction matters (journal conventions, clarity, etc.)")
    priority: IssuePriority = Field(default=IssuePriority.MEDIUM, description="Impact priority: Critical/High/Medium/Low")
    before_example: Optional[str] = Field(default=None, description="Example of problematic text")
    after_example: Optional[str] = Field(default=None, description="Example of corrected text")
    location: Optional[str] = Field(default=None, description="Specific location (section, paragraph, page)")

class TitleAnalysis(BaseModel):
    title_present: bool
    title_text: str
    word_count: int
    clarity_score: int = Field(description="0-100")
    has_clickbait_language: bool = Field(default=False, description="Whether clickbait-style language was detected")
    clickbait_words: List[str] = Field(default_factory=list, description="List of words flagged as clickbait")
    issues: List[Issue]
    overall_assessment: str
    specific_feedback: str

class AbstractAnalysis(BaseModel):
    abstract_present: bool = Field(description="Whether an abstract section was found")
    word_count: int = Field(description="Word count of the abstract")
    clarity_score: int = Field(description="Clarity of abstract (0-100)")
    completeness_score: int = Field(description="Does it contain key information (0-100)")
    issues: List[Issue] = Field(description="Issues found with corrections")
    overall_assessment: str = Field(description="Assessment: Excellent/Good/Needs improvement/Poor")
    specific_feedback: str = Field(description="Detailed feedback")

class IntroductionAnalysis(BaseModel):
    section_present: bool
    word_count: int
    clarity_score: int
    logical_flow: int
    issues: List[Issue]
    overall_assessment: str
    specific_feedback: str


class LiteratureReviewAnalysis(BaseModel):
    section_present: bool
    word_count: int
    comprehensiveness: int
    identifies_research_gaps: bool
    critical_analysis_level: int
    citation_quality: int
    number_of_sources: int
    issues: List[Issue]
    overall_assessment: str
    specific_feedback: str


class CoreMethodsAnalysis(BaseModel):
    section_present: bool
    word_count: int
    clarity_score: int
    is_reproducible: bool
    logical_flow: int
    technical_soundness: int
    issues: List[Issue]
    missing_details: List[Issue]
    overall_assessment: str
    specific_feedback: str


class ResultsAnalysis(BaseModel):
    section_present: bool
    word_count: int
    presents_key_findings: bool
    figures_tables_quality: int
    results_interpretation: bool
    clarity_score: int
    completeness_score: int
    issues: List[Issue]
    missing_results: List[Issue]
    overall_assessment: str
    specific_feedback: str


class ConclusionsAnalysis(BaseModel):
    section_present: bool
    word_count: int
    addresses_research_questions: bool
    discusses_implications: bool
    acknowledges_limitations: bool
    suggests_future_work: bool
    avoids_overstatement: bool
    clarity_score: int
    issues: List[Issue]
    missing_elements: List[Issue]
    overall_assessment: str
    specific_feedback: str


class PublishabilityAssessment(BaseModel):
    overall_score: int
    verdict: str
    key_strengths: List[str]
    critical_issues: List[str]
    recommendations: List[str]


class TopFix(BaseModel):
    rank: int = Field(description="Priority rank (1-5)")
    title: str = Field(description="Short title of fix (e.g., 'Formalize Title Notation')")
    category: str = Field(description="Category: Title/Abstract/Methods/Notation/Grammar/Formatting")
    description: str = Field(description="Why this matters (1-2 sentences)")
    concrete_action: str = Field(description="Exact action to take")


class CohesionAnalysis(BaseModel):
    intro_to_methods: Optional[str] = Field(default=None, description="Suggested linking sentence between intro and methods")
    methods_to_results: Optional[str] = Field(default=None, description="Suggested linking sentence between methods and results")
    results_to_conclusion: Optional[str] = Field(default=None, description="Suggested linking sentence between results and conclusion")
    missing_transitions: List[str] = Field(default_factory=list, description="Sections that lack clear transitions")


class ProofreadingReport(BaseModel):
    document_title: str = Field(description="Title of document")
    total_pages: int = Field(description="Total pages")

    # Section-by-section analysis
    title_analysis: TitleAnalysis = Field(description="Title analysis")
    abstract_analysis: AbstractAnalysis = Field(description="Abstract analysis")
    introduction_analysis: IntroductionAnalysis = Field(description="Introduction analysis")
    literature_review_analysis: LiteratureReviewAnalysis = Field(description="Literature review analysis")
    core_methods_analysis: CoreMethodsAnalysis = Field(description="Methods analysis")
    results_analysis: ResultsAnalysis = Field(description="Results analysis")
    conclusions_analysis: ConclusionsAnalysis = Field(description="Conclusions analysis")

    # Top 5 prioritized fixes for maximum impact
    top_fixes: List[TopFix] = Field(description="Top 5 highest-impact fixes ranked by priority")

    # Cross-section cohesion analysis
    cohesion_analysis: CohesionAnalysis = Field(description="Analysis of transitions between major sections")

    # General issues across all sections (consolidated for clarity)
    grammar_issues: List[Issue] = Field(description="Grammar issues with corrections")
    flow_issues: List[Issue] = Field(description="Flow issues with corrections")
    style_issues: List[Issue] = Field(description="Style issues with corrections")
    word_choice_issues: List[Issue] = Field(description="Word choice issues with corrections")
    continuity_issues: List[Issue] = Field(description="Continuity issues with corrections")
    consistency_issues: List[Issue] = Field(description="Consistency issues with corrections")
    formatting_issues: List[Issue] = Field(description="Formatting issues with corrections")
    citation_issues: List[Issue] = Field(description="Citation issues with corrections")

    publishability: PublishabilityAssessment = Field(description="Publishability assessment")
    summary: str = Field(description="Executive summary")


class ProofreadingPrompts:
    """Centralized, journal-agnostic proofreading prompts with focus on strategic guidance."""

    MAIN_PROMPT = """You are an expert academic editor and proofreader specializing in publication-ready feedback.

CRITICAL INSTRUCTIONS:
- PRIORITIZE IMPACT: Rank all issues by strategic importance (Critical → High → Medium → Low)
- SINGLE CORRECTIONS: Provide ONE best correction per issue with rationale (not alternatives)
- ACTIONABLE FEEDBACK: Every correction must be concrete and immediately implementable
- MICRO-EXPLANATIONS: Define technical/domain-specific terms in parentheses
- STYLE ELEVATION: Go beyond grammatical correctness to stylistic improvement

ANALYSIS SCOPE (by Priority):

TIER 1 - CRITICAL IMPACT:
1. TITLE CLARITY & NOTATION: Is the title formal, specific, and free of vagueness? Consistent notation?
2. ABSTRACT PRECISION: Does it contain problem, method, results? Every word necessary? Reproducibility checklist present?
3. METHODS REPRODUCIBILITY: Can another researcher replicate this? All parameters, hyperparameters, datasets specified?
4. NOTATION & CONSISTENCY: Greek letters, symbols, abbreviations used consistently throughout
5. RESEARCH QUESTIONS: Clearly stated? Do results address them?

TIER 2 - HIGH IMPACT:
- Introduction → Results logical flow and linking sentences
- Literature review positioning (research gap identification)
- Grammar & mechanics (subject-verb agreement, tense consistency, parallelism)
- Academic tone (no informal phrasing, passive voice where appropriate)

TIER 3 - MEDIUM IMPACT:
- Word choice precision (vague → specific terminology)
- Formatting consistency (fonts, captions, heading hierarchy)
- Flow and paragraph transitions
- Continuity across sections

TIER 4 - LOW IMPACT:
- Citation format compliance
- Minor punctuation refinements
- Layout aesthetics

PROVIDE SPECIFIC FEEDBACK:
For EACH issue found, structure as:
{
  "description": "Clear problem statement with location",
  "correction": "Single, concrete correction (the exact replacement text)",
  "rationale": "Why this matters (journal conventions, clarity, reproducibility)",
  "priority": "Critical/High/Medium/Low",
  "before_example": "Problematic text (optional)",
  "after_example": "Corrected text (optional)",
  "location": "Section, paragraph, or page reference"
}

SPECIAL INSTRUCTIONS:
- Title/Abstract: Formalize all notation; remove vagueness
- Methods: Highlight reproducibility gaps explicitly
- Grammar: Show before→after pairs for ambiguous fixes
- Flow: Suggest specific linking sentences between Introduction→Methods→Results→Conclusions
- Technical Terms: Add parenthetical definitions (e.g., "micro-level F1 score (token-level precision metric)")
- Style: Replace mechanically correct fixes with stylistically superior ones (rhythm, academic tone)

CONSOLIDATION:
- Mark duplicates across Grammar/Flow/Style sections with "See Grammar #2"
- Focus on unique, non-redundant feedback

FINAL OUTPUT: Provide top 5 highest-impact fixes before detailed analysis."""

    JOURNAL_PROMPTS = {
        "apa": """Add to main analysis: Ensure APA 7th edition compliance:
- Running head and page numbers
- Section headings (Level 1-3 hierarchy)
- In-text citations: (Author, Year) or Author (Year)
- Reference list: Hanging indent, alphabetical order""",

        "ieee": """Add to main analysis: Ensure IEEE style compliance:
- Numbered references: [1], [2], etc.
- In-text citation format: Use numbered brackets
- Reference format: Authors, "Title," Journal, vol., pp., year
- No DOI required but include if available""",

        "nature": """Add to main analysis: Ensure Nature journal style:
- Methods section should be concise and reproducible
- Figures/tables: Caption below for figures, above for tables
- References: Numbered, authors listed (up to 5, then et al.)
- Supplementary info references""",

        "generic": "Apply broad academic writing standards without journal-specific formatting."
    }

    SECTION_PROMPTS = {
        "grammar": """Analyze this paper for grammatical errors, punctuation issues, and spelling mistakes.
Focus on: subject-verb agreement, tense consistency, parallelism, misplaced modifiers.
For EACH error, provide:
1. Exact location with example text in quotes
2. Explanation of why it's incorrect
3. SINGLE BEST CORRECTION with rationale
4. Before → After example (optional)""",

        "flow": """Evaluate logical flow, paragraph transitions, and section organization.
For EACH flow issue:
1. Exact section/paragraph location
2. Specific problem description
3. CONCRETE FIX: Suggest a linking sentence or reorganization (provide exact text)
4. Explain why the fix improves readability""",

        "academic_tone": """Assess academic tone, formality, and professional writing style.
Replace informal phrasing, conversational tone, contractions with formal equivalents.
For EACH tone issue:
1. Location and problematic text in quotes
2. Why it violates academic conventions
3. SINGLE BEST REWRITE using proper academic language (exact replacement)
4. Explain stylistic improvement""",

        "clarity": """Evaluate clarity of explanations, definitions, and technical arguments.
For EACH clarity issue:
1. Exact location with context
2. What specifically makes it confusing
3. EXACT REWORDING that clarifies (full replacement text)
4. Why the fix improves understanding""",

        "structure": """Analyze overall paper structure, section organization, and logical progression.
For EACH structural issue:
1. Exact section reference
2. What is structurally wrong
3. CONCRETE REORGANIZATION: 'Move paragraph X after paragraph Y' with rationale
4. Explain how reorganization improves flow""",

        "word_choice": """Analyze word choices and vocabulary; target precision and conciseness.
Replace vague language, redundancies, imprecise terms with specific alternatives.
For EACH word choice issue:
1. Exact location and problematic word/phrase in quotes
2. Why it is imprecise or vague
3. EXACT REPLACEMENT WORD/PHRASE (not suggestions, but the actual word)
4. Explain improvement in precision or tone""",

        "continuity": """Evaluate continuity, narrative flow, and consistency across sections.
For EACH continuity issue:
1. All exact locations being compared
2. Specific inconsistency or broken flow
3. EXACT CHANGES NEEDED at each location (provide specific revisions)
4. Explain how changes restore coherence""",

        "formatting": """Check formatting, layout, font consistency, captions, and heading hierarchy.
For EACH formatting issue:
1. Exact location (page, section, figure/table number)
2. Current formatting problem
3. PRECISE STEP-BY-STEP INSTRUCTIONS to fix (e.g., 'change font to Times New Roman 12pt')
4. Explain why consistency matters""",

        "citations": """Perform BIDIRECTIONAL citation checking:
1. Verify all in-text citations match the reference list
2. Verify ALL references in the list are actually cited in the paper
For EACH citation issue:
1. Exact location (section, paragraph, or reference number)
2. Specific problem
3. EXACT ACTION REQUIRED (e.g., 'remove reference [5] from bibliography', 'change [Smith 2020] to [Smith et al. 2020]')
4. Explain importance (bibliography hygiene, consistency)""",

        "consistency": """Check for ALL consistency issues: spelling variations (color vs colour), capitalization,
abbreviation use, unit notation, and terminology.
For EACH inconsistency:
1. All exact locations where the variation appears
2. The different versions found
3. Which standard version to use
4. SPECIFIC CHANGES at each location (e.g., 'line 45: change color to colour', 'page 3: change Fig. to Figure')"""
    }


class GeminiClient:
    """A client to interact with the Gemini API for file uploads and content generation."""
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.client = self._create_client()
        self.uploaded_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.delete_pdf()

    def _get_api_key(self) -> str:
        """Extract and validate API key from environment."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            error_msg = (
                "GEMINI_API_KEY environment variable not set. "
                "Set it with: export GEMINI_API_KEY='your-api-key'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        return api_key

    def _create_client(self):
        """Create and return a Gemini API client."""
        api_key = self._get_api_key()
        try:
            logger.info(f"Connecting to Gemini API with model: {self.model_name}")
            client = genai.Client(api_key=api_key)
            logger.info("✓ Gemini client initialized successfully")
            return client
        except Exception as e:
            error_msg = f"Failed to initialize Gemini client: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e

    def _validate_pdf_file(self, pdf_file: str) -> pathlib.Path:
        """Validate PDF file exists and has correct extension."""
        pdf_path = pathlib.Path(pdf_file)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_file}")
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File must be a PDF. Got: {pdf_path.suffix}")
        return pdf_path

    def upload_pdf(self, pdf_file: str):
        """Upload a PDF file to Gemini."""
        if self.uploaded_file:
            self.delete_pdf()

        pdf_path = self._validate_pdf_file(pdf_file)
        logging.info(f"Uploading PDF: {pdf_path.name}...")
        self.uploaded_file = self.client.files.upload(
            file=pdf_path,
            config=dict(mime_type='application/pdf')
        )
        logging.info(f"✓ PDF uploaded: {pdf_path.name}")

    def delete_pdf(self):
        """Delete the uploaded PDF file from Gemini."""
        if self.uploaded_file:
            try:
                self.client.files.delete(name=self.uploaded_file.name)
                logging.info(f"✓ PDF file deleted: {self.uploaded_file.name}")
            finally:
                self.uploaded_file = None

    def _build_response_config(self, response_model=None) -> dict:
        """Build response configuration for JSON schema if provided."""
        if not response_model:
            return {}
        return {
            "response_mime_type": "application/json",
            "response_schema": response_model
        }

    def generate_content(self, prompt: str, response_model=None):
        """Generate content using the Gemini API."""
        if not self.uploaded_file:
            raise RuntimeError("No PDF loaded. Use upload_pdf() first.")

        config = self._build_response_config(response_model)
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[self.uploaded_file, prompt],
            config=config
        )
        self._validate_api_response(response)
        return response

    def _validate_prompt_feedback(self, response):
        """Validate prompt feedback for safety issues."""
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            block_reason = response.prompt_feedback.block_reason
            logging.error(f"Response blocked by API: {block_reason}")
            raise RuntimeError(f"API blocked the response: {block_reason}")

    def _validate_response_content(self, response):
        """Validate response has actual content."""
        if not response.candidates or response.text is None:
            logging.error("No response content generated by the API")
            raise RuntimeError("The API did not generate a response. This may be due to safety filters or content policy violations.")

    def _validate_api_response(self, response):
        """Orchestrate validation of API response."""
        self._validate_prompt_feedback(response)
        self._validate_response_content(response)


class GeminiPDFProofreader:
    DEFAULT_MODEL = "gemini-2.5-flash"

    def __init__(self, model_name: str = DEFAULT_MODEL, journal: str = "generic"):
        if journal not in ProofreadingPrompts.JOURNAL_PROMPTS:
            raise ValueError(f"Unknown journal style '{journal}'. Supported: {', '.join(ProofreadingPrompts.JOURNAL_PROMPTS.keys())}")

        self.model_name = model_name
        self.journal = journal
        self.client = GeminiClient(model_name=self.model_name)
        logger.info(f"Initialized proofreader with model={model_name}, journal={journal}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.__exit__(exc_type, exc_val, exc_tb)

    def load_pdf(self, pdf_file: str):
        self.client.upload_pdf(pdf_file)

    def _check_pdf_loaded(self):
        """Check if a PDF file is loaded."""
        if not self.client.uploaded_file:
            raise RuntimeError("No PDF loaded. Use load_pdf() first.")

    def _analyze_section_json(self, section_name: str, prompt: str, response_model) -> Optional[BaseModel]:
        """
        Make a focused API call for a specific section with JSON response.
        """
        try:
            logger.info(f"Analyzing {section_name}...")
            response = self.client.generate_content(prompt, response_model)

            if response.parsed is not None:
                logger.info(f"✓ {section_name} parsed successfully")
                return response.parsed
            else:
                try:
                    data = json.loads(response.text)
                    logger.info(f"✓ {section_name} parsed from JSON text")
                    return response_model(**data)
                except (json.JSONDecodeError, ValidationError) as e:
                    logger.warning(f"Could not parse {section_name}: {e}")
                    return None
        except Exception as e:
            logger.warning(f"Error analyzing {section_name}: {e}")
            return None

    def _analyze_section(
        self,
        section_name: str,
        response_model: type,
        instructions: str,
        default_values: dict,
        journal_prompt: str
    ) -> BaseModel:
        """Generic method to analyze any section with specified model and defaults."""
        prompt = generate_schema_prompt(response_model, instructions)
        prompt += f"\n\n{journal_prompt}"

        result = self._analyze_section_json(section_name, prompt, response_model)
        return result or response_model(**default_values)

    def _analyze_title(self, journal_prompt: str) -> TitleAnalysis:
        """Analyzes the title of the paper."""
        instructions = """Analyze the TITLE of this paper. Provide:
- Whether a title is present
- The title text
- Word count
- Clarity score (0-100)
- Whether it contains clickbait language (e.g., sensational, emotionally charged, or overly simplistic words like 'shocking', 'secret', 'good/bad')
- A list of any words flagged as clickbait
- Issues found with corrections (prioritized)
- Overall assessment
- Specific feedback"""

        defaults = {
            "title_present": False, "title_text": "", "word_count": 0,
            "clarity_score": 0, "has_clickbait_language": False,
            "clickbait_words": [], "issues": [],
            "overall_assessment": "Unable to analyze", "specific_feedback": ""
        }
        return self._analyze_section("Title", TitleAnalysis, instructions, defaults, journal_prompt)

    def _analyze_abstract(self, journal_prompt: str) -> AbstractAnalysis:
        """Analyzes the abstract of the paper."""
        instructions = """Analyze the ABSTRACT of this paper. Provide:
- Whether an abstract is present
- Word count
- Clarity score (0-100)
- Completeness score (0-100)
- Issues found with corrections
- Overall assessment
- Specific feedback"""

        defaults = {
            "abstract_present": False, "word_count": 0, "clarity_score": 0,
            "completeness_score": 0, "issues": [],
            "overall_assessment": "Unable to analyze", "specific_feedback": ""
        }
        return self._analyze_section("Abstract", AbstractAnalysis, instructions, defaults, journal_prompt)

    def _analyze_introduction(self, journal_prompt: str) -> IntroductionAnalysis:
        """Analyzes the introduction of the paper."""
        instructions = """Analyze the INTRODUCTION section. Provide:
- Whether section is present
- Word count
- Clarity score (0-100)
- Logical flow score (0-100)
- Issues found with corrections
- Overall assessment
- Specific feedback"""

        defaults = {
            "section_present": False, "word_count": 0, "clarity_score": 0,
            "logical_flow": 0, "issues": [],
            "overall_assessment": "Not found", "specific_feedback": ""
        }
        return self._analyze_section("Introduction", IntroductionAnalysis, instructions, defaults, journal_prompt)

    def _analyze_literature_review(self, journal_prompt: str) -> LiteratureReviewAnalysis:
        """Analyzes the literature review of the paper."""
        instructions = """Analyze the LITERATURE REVIEW section. Provide:
- Whether section is present
- Word count
- Comprehensiveness score (0-100)
- Whether it identifies research gaps
- Critical analysis level (0-100)
- Citation quality score (0-100)
- Number of sources
- Issues found with corrections
- Overall assessment
- Specific feedback"""

        defaults = {
            "section_present": False, "word_count": 0, "comprehensiveness": 0,
            "identifies_research_gaps": False, "critical_analysis_level": 0,
            "citation_quality": 0, "number_of_sources": 0, "issues": [],
            "overall_assessment": "Not found", "specific_feedback": ""
        }
        return self._analyze_section("Literature Review", LiteratureReviewAnalysis, instructions, defaults, journal_prompt)

    def _analyze_methods(self, journal_prompt: str) -> CoreMethodsAnalysis:
        """Analyzes the methods section of the paper."""
        instructions = """Analyze the METHODS section. Provide:
- Whether section is present
- Word count
- Clarity score (0-100)
- Is reproducible (bool)
- Logical flow score (0-100)
- Technical soundness score (0-100)
- Issues found with corrections
- Missing details
- Overall assessment
- Specific feedback

CRITICAL: Flag ANY missing parameters, hyperparameters, datasets, or implementation details that would prevent replication."""

        defaults = {
            "section_present": False, "word_count": 0, "clarity_score": 0,
            "is_reproducible": False, "logical_flow": 0, "technical_soundness": 0,
            "issues": [], "missing_details": [],
            "overall_assessment": "Not found", "specific_feedback": ""
        }
        return self._analyze_section("Methods", CoreMethodsAnalysis, instructions, defaults, journal_prompt)

    def _analyze_results(self, journal_prompt: str) -> ResultsAnalysis:
        """Analyzes the results section of the paper."""
        instructions = """Analyze the RESULTS section. Provide:
- Whether section is present
- Word count
- Presents key findings (bool)
- Figures/tables quality (0-100)
- Results interpretation (bool)
- Clarity score (0-100)
- Completeness score (0-100)
- Issues found with corrections
- Missing results
- Overall assessment
- Specific feedback"""

        defaults = {
            "section_present": False, "word_count": 0, "presents_key_findings": False,
            "figures_tables_quality": 0, "results_interpretation": False,
            "clarity_score": 0, "completeness_score": 0, "issues": [],
            "missing_results": [], "overall_assessment": "Not found",
            "specific_feedback": ""
        }
        return self._analyze_section("Results", ResultsAnalysis, instructions, defaults, journal_prompt)

    def _analyze_conclusions(self, journal_prompt: str) -> ConclusionsAnalysis:
        """Analyzes the conclusions section of the paper."""
        instructions = """Analyze the CONCLUSIONS section. Provide:
- Whether section is present
- Word count
- Addresses research questions (bool)
- Discusses implications (bool)
- Acknowledges limitations (bool)
- Suggests future work (bool)
- Avoids overstatement (bool)
- Clarity score (0-100)
- Issues found with corrections
- Missing elements
- Overall assessment
- Specific feedback"""

        defaults = {
            "section_present": False, "word_count": 0,
            "addresses_research_questions": False, "discusses_implications": False,
            "acknowledges_limitations": False, "suggests_future_work": False,
            "avoids_overstatement": False, "clarity_score": 0, "issues": [],
            "missing_elements": [], "overall_assessment": "Not found",
            "specific_feedback": ""
        }
        return self._analyze_section("Conclusions", ConclusionsAnalysis, instructions, defaults, journal_prompt)

    def _analyze_general_issues(self, journal_prompt: str) -> dict:
        """Analyzes general issues like grammar, flow, style, etc."""
        instructions = """Analyze this paper for all types of issues:
- Grammar issues
- Flow issues
- Style/tone issues
- Word choice issues
- Continuity issues
- Consistency issues
- Formatting issues
- Citation issues

For each issue in each category, provide a structured issue object with description, correction, rationale, priority, and optional examples/location."""

        issue_schema = generate_schema_prompt(Issue, "")

        general_prompt = f"""{instructions}

Expected output structure:
{{
  "grammar_issues": [array of issue objects],
  "flow_issues": [array of issue objects],
  "style_issues": [array of issue objects],
  "word_choice_issues": [array of issue objects],
  "continuity_issues": [array of issue objects],
  "consistency_issues": [array of issue objects],
  "formatting_issues": [array of issue objects],
  "citation_issues": [array of issue objects]
}}

Where each issue object follows this structure:
{issue_schema}

{journal_prompt}"""
        return self._analyze_section_json("General Issues", general_prompt, dict) or {}

    def _analyze_top_fixes(self, journal_prompt: str) -> list:
        """Analyzes the top 5 fixes for the paper."""
        instructions = """Based on your analysis of this paper, identify the TOP 5 HIGHEST-IMPACT fixes ranked by strategic importance.
For each fix, provide a structured object with rank (1-5), title, category, description of why it matters, and concrete action steps."""

        topfix_schema = generate_schema_prompt(TopFix, "")

        top_fixes_prompt = f"""{instructions}

Return as JSON array where each element follows this structure:
{topfix_schema}"""
        return self._analyze_section_json("Top Fixes", top_fixes_prompt, list) or []

    def _analyze_cohesion(self, journal_prompt: str) -> CohesionAnalysis:
        """Analyzes the cohesion of the paper."""
        instructions = """Analyze the cohesion and transitions between major sections:
- Intro to Methods: Suggest a linking sentence if needed
- Methods to Results: Suggest a linking sentence if needed
- Results to Conclusion: Suggest a linking sentence if needed
- Identify any sections with poor transitions"""

        cohesion_prompt = generate_schema_prompt(CohesionAnalysis, instructions)

        cohesion_analysis = self._analyze_section_json("Cohesion", cohesion_prompt, CohesionAnalysis)
        return cohesion_analysis or CohesionAnalysis()

    def _analyze_publishability(self, journal_prompt: str) -> PublishabilityAssessment:
        """Analyzes the publishability of the paper."""
        instructions = """Assess the publishability of this paper. Provide:
- Overall score (0-100) based on journal-readiness
- Verdict with clear recommendation (e.g., "Ready for submission", "Needs major revisions", etc.)
- Key strengths that support publication
- Critical issues that must be addressed before submission
- Key recommendations for improvement"""

        publishability_prompt = generate_schema_prompt(PublishabilityAssessment, instructions)

        publishability = self._analyze_section_json("Publishability", publishability_prompt, PublishabilityAssessment)
        return publishability or PublishabilityAssessment(
            overall_score=0, verdict="Unable to assess",
            key_strengths=[], critical_issues=[], recommendations=[]
        )

    def _get_document_metadata(self) -> (str, int):
        """Gets the document title and page count."""
        metadata_prompt = "What is the title and approximate page count of this PDF?"
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.client.uploaded_file, metadata_prompt]
            )
            self._validate_api_response(response)
            metadata_text = response.text
            document_title = "Academic Paper"
            total_pages = 10
            if "title" in metadata_text.lower():
                lines = metadata_text.split('\n')
                for line in lines:
                    if "title" in line.lower():
                        document_title = line.split(':')[-1].strip() if ':' in line else "Academic Paper"
                        break
            if "page" in metadata_text.lower():
                import re
                matches = re.findall(r'\d+', metadata_text)
                if matches:
                    total_pages = int(matches[0])
            return document_title, total_pages
        except Exception as e:
            logger.warning(f"Could not extract metadata: {e}")
            return "Academic Paper", 10

    def _normalize_issues(self, issues_list: list) -> list:
        """Convert raw issues (dict or Issue) to Issue objects."""
        return [
            Issue(**issue) if isinstance(issue, dict) else issue
            for issue in issues_list
        ]

    def _run_section_analyses(self, journal_prompt: str) -> dict:
        """Run all section-specific analyses."""
        return {
            "title": self._analyze_title(journal_prompt),
            "abstract": self._analyze_abstract(journal_prompt),
            "introduction": self._analyze_introduction(journal_prompt),
            "literature_review": self._analyze_literature_review(journal_prompt),
            "methods": self._analyze_methods(journal_prompt),
            "results": self._analyze_results(journal_prompt),
            "conclusions": self._analyze_conclusions(journal_prompt),
        }

    def _extract_general_issues(self, general_response: dict) -> dict:
        """Extract and normalize all general issue categories."""
        issue_categories = [
            "grammar_issues", "flow_issues", "style_issues", "word_choice_issues",
            "continuity_issues", "consistency_issues", "formatting_issues", "citation_issues"
        ]
        return {
            cat.replace("_issues", ""): self._normalize_issues(general_response.get(cat, []))
            for cat in issue_categories
        }

    def _run_additional_analyses(self, journal_prompt: str) -> tuple:
        """Run top fixes, cohesion, publishability, and metadata analyses."""
        top_fixes_response = self._analyze_top_fixes(journal_prompt)
        top_fixes = [TopFix(**fix) if isinstance(fix, dict) else fix for fix in top_fixes_response]

        cohesion = self._analyze_cohesion(journal_prompt)
        publishability = self._analyze_publishability(journal_prompt)
        document_title, total_pages = self._get_document_metadata()

        return top_fixes, cohesion, publishability, document_title, total_pages

    def _build_proofreading_report(
        self,
        section_analyses: dict,
        general_issues: dict,
        top_fixes: list,
        cohesion: CohesionAnalysis,
        publishability: PublishabilityAssessment,
        document_title: str,
        total_pages: int
    ) -> ProofreadingReport:
        """Assemble final proofreading report."""
        summary = f"This {total_pages}-page paper has been analyzed across all major sections. Key areas of focus: title clarity, abstract completeness, methodology reproducibility, results presentation, and overall publishability. See top 5 fixes section for prioritized action items."

        return ProofreadingReport(
            document_title=document_title,
            total_pages=total_pages,
            title_analysis=section_analyses["title"],
            abstract_analysis=section_analyses["abstract"],
            introduction_analysis=section_analyses["introduction"],
            literature_review_analysis=section_analyses["literature_review"],
            core_methods_analysis=section_analyses["methods"],
            results_analysis=section_analyses["results"],
            conclusions_analysis=section_analyses["conclusions"],
            top_fixes=top_fixes,
            cohesion_analysis=cohesion,
            grammar_issues=general_issues["grammar"],
            flow_issues=general_issues["flow"],
            style_issues=general_issues["style"],
            word_choice_issues=general_issues["word_choice"],
            continuity_issues=general_issues["continuity"],
            consistency_issues=general_issues["consistency"],
            formatting_issues=general_issues["formatting"],
            citation_issues=general_issues["citation"],
            publishability=publishability,
            summary=summary
        )

    def proofread(self) -> ProofreadingReport:
        """
        Perform comprehensive proofreading on the uploaded PDF with prioritized feedback.
        Orchestrates multiple focused API calls to analyze all sections.

        Returns:
            ProofreadingReport: Structured proofreading results with impact-ranked issues

        Raises:
            RuntimeError: If no PDF is loaded or API returns an error
        """
        self._check_pdf_loaded()
        logger.info("Starting comprehensive proofreading analysis...")
        logger.info(f"Journal style: {self.journal}")

        journal_prompt = ProofreadingPrompts.JOURNAL_PROMPTS.get(
            self.journal, ProofreadingPrompts.JOURNAL_PROMPTS["generic"]
        )

        try:
            section_analyses = self._run_section_analyses(journal_prompt)
            general_response = self._analyze_general_issues(journal_prompt)
            general_issues = self._extract_general_issues(general_response)
            top_fixes, cohesion, publishability, doc_title, pages = self._run_additional_analyses(journal_prompt)

            report = self._build_proofreading_report(
                section_analyses, general_issues, top_fixes, cohesion,
                publishability, doc_title, pages
            )

            logger.info("✓ Comprehensive proofreading analysis completed successfully")
            return report

        except ConnectionError as e:
            logger.error(f"Connection error during proofreading: {e}")
            raise RuntimeError("Connection error. Check your API key and network connectivity.") from e
        except Exception as e:
            logger.error(f"Unexpected error during proofreading: {e}", exc_info=True)
            raise RuntimeError(f"Proofreading failed: {e}") from e

    def proofread_section(self, section_focus: str) -> str:
        """
        Perform targeted proofreading on a specific aspect of the paper.
        """
        section_lower = section_focus.lower()
        if section_lower not in ProofreadingPrompts.SECTION_PROMPTS:
            supported = ', '.join(ProofreadingPrompts.SECTION_PROMPTS.keys())
            raise ValueError(f"Unknown focus area '{section_focus}'. Supported: {supported}")

        logger.info(f"Analyzing paper for: {section_focus}...")
        prompt = ProofreadingPrompts.SECTION_PROMPTS[section_lower]

        try:
            logger.info(f"Sending focused analysis request (focus={section_focus})...")
            response = self.client.generate_content(prompt)
            logger.info(f"✓ Focused analysis completed")
            return response.text
        except (ValueError, ConnectionError, RuntimeError) as e:
            logger.error(f"Error during section analysis: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during section analysis: {e}", exc_info=True)
            raise RuntimeError(f"Section analysis failed: {e}") from e


def _format_priority_tag(priority: IssuePriority) -> str:
    """Format priority with visual indicator."""
    tags = {
        IssuePriority.CRITICAL: "[🔴 CRITICAL]",
        IssuePriority.HIGH: "[🟠 HIGH]",
        IssuePriority.MEDIUM: "[🟡 MEDIUM]",
        IssuePriority.LOW: "[🟢 LOW]"
    }
    return tags.get(priority, "[?]")


def _print_issue_with_details(issue: Issue, issue_num: int = None):
    """Print a single issue with all details."""
    num_prefix = f"{issue_num}. " if issue_num else ""
    print(f"\n  {num_prefix}{_format_priority_tag(issue.priority)}")
    if issue.location:
        print(f"     Location: {issue.location}")
    print(f"     Problem: {issue.description}")
    if issue.before_example:
        print(f'     Before: "{issue.before_example}"')
    print(f"     Fix: {issue.correction}")
    if issue.after_example:
        print(f'     After: "{issue.after_example}"')
    print(f"     Why: {issue.rationale}")


def _print_header(report: ProofreadingReport):
    """Prints the header of the report."""
    print("\n" + "=" * 80)
    print("PROOFREADING REPORT")
    print("=" * 80)
    print(f"\nDocument: {report.document_title}")
    print(f"Pages: {report.total_pages}")

def _print_summary(report: ProofreadingReport):
    """Prints the summary of the report."""
    print(f"\n{report.summary}")

def _print_top_fixes(report: ProofreadingReport):
    """Prints the top 5 highest-impact fixes."""
    print("\n" + "=" * 80)
    print("TOP 5 HIGHEST-IMPACT FIXES (Focus Here First)")
    print("=" * 80)
    if report.top_fixes:
        for fix in report.top_fixes:
            print(f"\n#{fix.rank} [{fix.category.upper()}]")
            print(f"   Title: {fix.title}")
            print(f"   Why It Matters: {fix.description}")
            print(f"   Action: {fix.concrete_action}")
    else:
        print("\n   ✓ No critical issues identified!")

def _print_cohesion_analysis(report: ProofreadingReport):
    """Prints the cohesion analysis."""
    print("\n" + "=" * 80)
    print("SECTION TRANSITIONS & COHESION")
    print("=" * 80)
    if report.cohesion_analysis:
        cohesion = report.cohesion_analysis
        if cohesion.intro_to_methods:
            print(f"\nIntroduction → Methods:")
            print(f"   Suggested transition: {cohesion.intro_to_methods}")
        if cohesion.methods_to_results:
            print(f"\nMethods → Results:")
            print(f"   Suggested transition: {cohesion.methods_to_results}")
        if cohesion.results_to_conclusion:
            print(f"\nResults → Conclusion:")
            print(f"   Suggested transition: {cohesion.results_to_conclusion}")
        if cohesion.missing_transitions:
            print(f"\nMissing Transitions:")
            for section in cohesion.missing_transitions:
                print(f"   • {section}")

def _print_publishability_assessment(report: ProofreadingReport):
    """Prints the publishability assessment."""
    print("\n" + "=" * 80)
    print("PUBLISHABILITY ASSESSMENT")
    print("=" * 80)
    print(f"\nScore: {report.publishability.overall_score}/100")
    print(f"Verdict: {report.publishability.verdict}")

    print("\n✓ Strengths:")
    for strength in report.publishability.key_strengths:
        print(f"  • {strength}")

    if report.publishability.critical_issues:
        print("\n✗ Critical Issues to Address:")
        for issue in report.publishability.critical_issues:
            print(f"  • {issue}")

    if report.publishability.recommendations:
        print("\n📋 Key Recommendations:")
        for i, rec in enumerate(report.publishability.recommendations, 1):
            print(f"  {i}. {rec}")

def _print_section_analysis(report: ProofreadingReport):
    """Prints the detailed section-by-section analysis."""
    print("\n" + "=" * 80)
    print("DETAILED SECTION ANALYSIS")
    print("=" * 80)

    # TITLE ANALYSIS
    if report.title_analysis.title_present:
        print("\n[TITLE]")
        print(f"  Text: {report.title_analysis.title_text}")
        print(f"  Clarity: {report.title_analysis.clarity_score}/100 | Words: {report.title_analysis.word_count}")
        if report.title_analysis.has_clickbait_language:
            print(f"  [!] Clickbait Language Detected: {', '.join(report.title_analysis.clickbait_words)}")
        print(f"  Assessment: {report.title_analysis.overall_assessment}")
        print(f"  Feedback: {report.title_analysis.specific_feedback}")
        if report.title_analysis.issues:
            for issue in sorted(report.title_analysis.issues, key=lambda x: (x.priority.value if hasattr(x.priority, 'value') else str(x.priority)), reverse=True):
                _print_issue_with_details(issue)

    # ABSTRACT ANALYSIS
    if report.abstract_analysis.abstract_present:
        print("\n[ABSTRACT]")
        print(f"  Clarity: {report.abstract_analysis.clarity_score}/100 | Completeness: {report.abstract_analysis.completeness_score}/100 | Words: {report.abstract_analysis.word_count}")
        print(f"  Assessment: {report.abstract_analysis.overall_assessment}")
        print(f"  Feedback: {report.abstract_analysis.specific_feedback}")
        if report.abstract_analysis.issues:
            for issue in sorted(report.abstract_analysis.issues, key=lambda x: (x.priority.value if hasattr(x.priority, 'value') else str(x.priority)), reverse=True):
                _print_issue_with_details(issue)
    else:
        print("\n[ABSTRACT] ✗ CRITICAL: No abstract found. Academic papers MUST have an abstract section.")

    # INTRODUCTION, METHODS, RESULTS, CONCLUSIONS (compact view)
    print("\n[INTRODUCTION]")
    if report.introduction_analysis.section_present:
        print(f"  Clarity: {report.introduction_analysis.clarity_score}/100 | Flow: {report.introduction_analysis.logical_flow}/100 | Words: {report.introduction_analysis.word_count}")
        print(f"  Assessment: {report.introduction_analysis.overall_assessment}")
        if report.introduction_analysis.issues:
            print(f"  Issues: {len(report.introduction_analysis.issues)}")
    else:
        print("  ✗ Not found")

    print("\n[METHODS]")
    if report.core_methods_analysis.section_present:
        print(f"  Clarity: {report.core_methods_analysis.clarity_score}/100 | Reproducibility: {report.core_methods_analysis.is_reproducible} | Words: {report.core_methods_analysis.word_count}")
        print(f"  Assessment: {report.core_methods_analysis.overall_assessment}")
        if report.core_methods_analysis.issues:
            print(f"  Issues: {len(report.core_methods_analysis.issues)}")
    else:
        print("  ✗ Not found")

    print("\n[RESULTS]")
    if report.results_analysis.section_present:
        print(f"  Clarity: {report.results_analysis.clarity_score}/100 | Completeness: {report.results_analysis.completeness_score}/100 | Words: {report.results_analysis.word_count}")
        print(f"  Assessment: {report.results_analysis.overall_assessment}")
        if report.results_analysis.issues:
            print(f"  Issues: {len(report.results_analysis.issues)}")
    else:
        print("  ✗ Not found")

    print("\n[CONCLUSIONS]")
    if report.conclusions_analysis.section_present:
        print(f"  Clarity: {report.conclusions_analysis.clarity_score}/100 | Words: {report.conclusions_analysis.word_count}")
        print(f"  Assessment: {report.conclusions_analysis.overall_assessment}")
        if report.conclusions_analysis.issues:
            print(f"  Issues: {len(report.conclusions_analysis.issues)}")
    else:
        print("  ✗ Not found")

def _print_issue_categories(report: ProofreadingReport):
    """Prints the consolidated issues by category."""
    print("\n" + "=" * 80)
    print("ALL ISSUES BY CATEGORY (Sorted by Priority)")
    print("=" * 80)

    issue_categories = [
        ("CRITICAL PRIORITY ISSUES", [issue for all_issues in [report.grammar_issues, report.flow_issues, report.style_issues, report.word_choice_issues, report.continuity_issues, report.consistency_issues, report.formatting_issues, report.citation_issues] for issue in all_issues if issue.priority == IssuePriority.CRITICAL]),
        ("HIGH PRIORITY ISSUES", [issue for all_issues in [report.grammar_issues, report.flow_issues, report.style_issues, report.word_choice_issues, report.continuity_issues, report.consistency_issues, report.formatting_issues, report.citation_issues] for issue in all_issues if issue.priority == IssuePriority.HIGH]),
        ("MEDIUM PRIORITY ISSUES", [issue for all_issues in [report.grammar_issues, report.flow_issues, report.style_issues, report.word_choice_issues, report.continuity_issues, report.consistency_issues, report.formatting_issues, report.citation_issues] for issue in all_issues if issue.priority == IssuePriority.MEDIUM]),
        ("LOW PRIORITY ISSUES", [issue for all_issues in [report.grammar_issues, report.flow_issues, report.style_issues, report.word_choice_issues, report.continuity_issues, report.consistency_issues, report.formatting_issues, report.citation_issues] for issue in all_issues if issue.priority == IssuePriority.LOW]),
    ]

    for category_name, issues in issue_categories:
        if issues:
            print(f"\n{category_name} ({len(issues)} found)")
            print("-" * 80)
            for i, issue in enumerate(issues, 1):
                _print_issue_with_details(issue, i)

    print("\n" + "=" * 80 + "\n")


def _print_report(report: ProofreadingReport):
    """Pretty print the proofreading report with prioritized feedback."""
    _print_header(report)
    _print_summary(report)
    _print_top_fixes(report)
    _print_cohesion_analysis(report)
    _print_publishability_assessment(report)
    _print_section_analysis(report)
    _print_issue_categories(report)


def main():
    """Main function to handle command line execution."""
    
    # Define focus areas
    section_choices = [
        'title', 'abstract', 'introduction', 'literature_review', 'methods', 
        'results', 'conclusions'
    ]
    issue_choices = [
        'grammar', 'flow', 'academic_tone', 'clarity', 'structure', 
        'word_choice', 'continuity', 'formatting', 'citations', 'consistency', 'cohesion'
    ]
    all_focus_choices = section_choices + issue_choices

    parser = argparse.ArgumentParser(
        description='Proofread PDF papers for journal submission using Gemini API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Full proofreading analysis (generic academic style)
  python gemini_pdf_proofreader.py -i paper.pdf

  # For Nature journal format
  python gemini_pdf_proofreader.py -i paper.pdf --journal nature

  # Focus on a specific aspect (e.g., grammar)
  python gemini_pdf_proofreader.py -i paper.pdf --focus grammar

  # Focus on a specific section (e.g., abstract)
  python gemini_pdf_proofreader.py -i paper.pdf --focus abstract

  # With custom model
  python gemini_pdf_proofreader.py -i paper.pdf --model gemini-2.5-flash

  # Export report to JSON
  python gemini_pdf_proofreader.py -i paper.pdf -o report.json

Supported journals: apa, ieee, nature, generic
Supported focus areas: {', '.join(all_focus_choices)}
        """
    )
    parser.add_argument('-i', '--pdf', type=str, required=True,
                      help='Path to the PDF file to proofread')
    parser.add_argument('--model', type=str, default=GeminiPDFProofreader.DEFAULT_MODEL,
                      help=f'Gemini model to use (default: {GeminiPDFProofreader.DEFAULT_MODEL})')
    parser.add_argument('--journal', type=str, default='generic',
                      choices=['apa', 'ieee', 'nature', 'generic'],
                      help='Target journal style (default: generic academic style)')
    parser.add_argument('--focus', type=str, choices=all_focus_choices,
                      help='Focus on a specific aspect or section instead of full analysis')
    parser.add_argument('-o', '--output', type=str,
                      help='Optional: Save JSON report to file')

    args = parser.parse_args()

    try:
        logger.info(f"Starting PDF proofreader for: {args.pdf}")
        with GeminiPDFProofreader(model_name=args.model, journal=args.journal) as proofreader:
            proofreader.load_pdf(args.pdf)

            if args.focus:
                print(f"\nAnalyzing {args.focus}...\n")
                
                journal_prompt = ProofreadingPrompts.JOURNAL_PROMPTS.get(proofreader.journal, ProofreadingPrompts.JOURNAL_PROMPTS["generic"])

                section_analyzers = {
                    "title": proofreader._analyze_title,
                    "abstract": proofreader._analyze_abstract,
                    "introduction": proofreader._analyze_introduction,
                    "literature_review": proofreader._analyze_literature_review,
                    "methods": proofreader._analyze_methods,
                    "results": proofreader._analyze_results,
                    "conclusions": proofreader._analyze_conclusions,
                }

                if args.focus in section_analyzers:
                    result = section_analyzers[args.focus](journal_prompt)
                    if isinstance(result, BaseModel):
                        print(result.model_dump_json(indent=2))
                    elif isinstance(result, list):
                        print(json.dumps([item.model_dump() for item in result], indent=2))
                    else:
                        print(result)
                elif args.focus == 'cohesion':
                    result = proofreader._analyze_cohesion(journal_prompt)
                    print(result.model_dump_json(indent=2))
                else:  # It's an issue-based focus
                    result = proofreader.proofread_section(args.focus)
                    print(result)

            else:
                print(f"\nRunning comprehensive proofreading analysis ({args.journal} style)...\n")
                report = proofreader.proofread()
                _print_report(report)

                if args.output:
                    with open(args.output, 'w') as f:
                        json.dump(report.model_dump(), f, indent=2)
                    logger.info(f"✓ Report saved to {args.output}")
                    print(f"\n✓ JSON report saved: {args.output}")

    except (FileNotFoundError, ValueError, ConnectionError, RuntimeError) as e:
        logger.error(f"Error: {e}")
        print(f"ERROR: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"ERROR: An unexpected error occurred. Check the log file for details.")


if __name__ == "__main__":
    main()
