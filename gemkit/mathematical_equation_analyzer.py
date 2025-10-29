"""
Mathematical Equation Analyzer - Simple Version
Understand and analyze mathematical equations in science using Gemini API.
Uses gemini_client.py and Pydantic BaseModels.
"""

import sys
from pydantic import BaseModel, Field
from typing import List, Literal
from gemini_client import GeminiClient, ModelConfig, ModelInput

# Audience types
AudienceType = Literal["high_school", "graduate", "professional", "researcher"]


# ============================================================================
# PYDANTIC MODELS - Ordered by Increasing Learning Steps
# ============================================================================

# === LEARNING STEP 1: Understanding What the Equation IS ===

class Assumption(BaseModel):
    """An assumption made in the mathematical equation"""
    name: str = Field(..., description="Name or title of the assumption")
    description: str = Field(..., description="Description of the assumption")
    impact: str = Field(..., description="Impact or consequence of this assumption")


class Overview(BaseModel):
    """Overview of a mathematical equation"""
    equation_name: str = Field(..., description="Name of the equation")
    latex_formula: str = Field(..., description="LaTeX formula of the equation")
    subject: str = Field(..., description="Subject or field (Physics, Chemistry, Mathematics, etc.)")
    variables: str = Field(..., description="Explanation of all variables used in the equation")
    assumptions: List[Assumption] = Field(..., description="List of assumptions made in the equation")


# === LEARNING STEP 2: Understanding the Origins ===

class InventorBiography(BaseModel):
    """Biography of the inventor or discoverer of a mathematical equation"""
    name: str = Field(..., description="Full name of the inventor or discoverer")
    birth_year: str = Field(..., description="Birth year and location")
    death_year: str = Field(..., description="Death year and location, or 'Living' if still alive")
    nationality: str = Field(..., description="Nationality and cultural background")
    education: str = Field(..., description="Educational background and institutions attended")
    career: str = Field(..., description="Professional career, positions held, and major milestones")
    key_contributions: str = Field(..., description="Key mathematical, scientific, or philosophical contributions beyond this equation")
    personal_life: str = Field(..., description="Interesting aspects of personal life, family, or personality")
    legacy: str = Field(..., description="Legacy and impact on mathematics, science, and society")


# === LEARNING STEP 3: Building Intuitive Understanding ===

class DeepIntuition(BaseModel):
    """A deep intuition or insight that led to the development of a mathematical equation"""
    title: str = Field(..., description="Title or name of the intuitive insight")
    core_insight: str = Field(..., description="The core intuitive idea or conceptual breakthrough")
    conceptual_motivation: str = Field(..., description="The conceptual or philosophical motivation behind this intuition")
    thought_process: str = Field(..., description="The logical reasoning and thought process that led to this insight")
    connection_to_equation: str = Field(..., description="How this intuition directly connects to and manifests in the final equation")
    why_necessary: str = Field(..., description="Why this intuition was necessary or fundamental for discovering this equation")


class Analogy(BaseModel):
    """An analogy to explain the core ideas of a mathematical equation"""
    title: str = Field(..., description="Title or name of the analogy")
    analogy_description: str = Field(..., description="The analogy itself - a relatable, everyday example")
    core_idea: str = Field(..., description="What core idea or concept does this analogy explain")
    connection: str = Field(..., description="How the analogy connects to the actual equation and its behavior")


class VisualPhysicalExample(BaseModel):
    """A visual or physical example of a mathematical equation"""
    title: str = Field(..., description="Title of the example")
    example_type: str = Field(..., description="Type of example: 'visual' or 'physical'")
    description: str = Field(..., description="Description of the visual or physical example")
    visualization_details: str = Field(..., description="For visual examples: description of what to plot/visualize. For physical examples: description of the physical setup or real-world scenario")
    key_observations: str = Field(..., description="Key observations or patterns to look for in the visualization or physical example")
    interpretation: str = Field(..., description="How to interpret the results and what they reveal about the equation's behavior")


# === LEARNING STEP 4: Understanding the Mathematics ===

class DerivationStep(BaseModel):
    """A step in deriving a mathematical equation"""
    step_number: int = Field(..., description="Step number in the derivation")
    description: str = Field(..., description="Description of this derivation step")
    equation: str = Field(..., description="The equation or expression at this step (in LaTeX)")
    explanation: str = Field(..., description="Explanation of why this step is valid or how it follows from the previous step")


class Derivation(BaseModel):
    """The derivation of a mathematical equation"""
    introduction: str = Field(..., description="Brief introduction to the derivation approach")
    steps: List[DerivationStep] = Field(..., description="Step-by-step derivation of the equation")
    conclusion: str = Field(..., description="Summary of how the final equation is obtained")


# === LEARNING STEP 5: Solving the Equation ===

class AnalyticSolution(BaseModel):
    """An analytical solution to a mathematical equation"""
    name: str = Field(..., description="Name or type of the analytical solution")
    solution: str = Field(..., description="The analytical solution formula or expression")
    conditions: str = Field(..., description="Conditions under which this solution is valid")
    description: str = Field(..., description="Description of when and why this solution applies")


class NumericalSolution(BaseModel):
    """A numerical method to solve a mathematical equation"""
    name: str = Field(..., description="Name of the numerical method")
    description: str = Field(..., description="How the numerical method works")
    advantages: str = Field(..., description="Advantages of this numerical method")
    applications: str = Field(..., description="Common applications of this numerical method")


class Solution(BaseModel):
    """Solutions to a mathematical equation"""
    analytic_solutions: List[AnalyticSolution] = Field(..., description="List of analytical solutions to the equation")
    numerical_solutions: List[NumericalSolution] = Field(..., description="List of numerical methods to solve the equation")


# === LEARNING STEP 6: Applications & Real-World Relevance ===

class TheoreticalApplication(BaseModel):
    """A theoretical application of a mathematical equation"""
    name: str = Field(..., description="Name of the theoretical application or concept")
    description: str = Field(..., description="How the equation is applied theoretically")
    theoretical_framework: str = Field(..., description="The theoretical framework or domain it contributes to")
    mathematical_significance: str = Field(..., description="Why this application is significant in pure mathematics or theoretical science")
    insights_provided: str = Field(..., description="What new insights or understanding this application provides")


class AppliedScienceApplication(BaseModel):
    """An applied science application of a mathematical equation"""
    name: str = Field(..., description="Name of the applied application or technology")
    description: str = Field(..., description="How the equation is applied in practice")
    field: str = Field(..., description="Field, industry, or technology where it is applied")
    practical_impact: str = Field(..., description="The practical impact or benefit of this application")
    implementation_details: str = Field(..., description="How it is implemented in real-world systems or devices")


class ModernRelevance(BaseModel):
    """Modern relevance and contemporary applications of a mathematical equation"""
    area: str = Field(..., description="Field, technology, or research area where this equation is currently relevant")
    description: str = Field(..., description="Description of how the equation is relevant in this modern context")
    current_applications: str = Field(..., description="Specific current applications or use cases in this area")
    recent_developments: str = Field(..., description="Recent developments, breakthroughs, or trends related to this equation in this field")
    future_potential: str = Field(..., description="Potential future applications or emerging opportunities for this equation")


# === VALIDATION: Proving the Equation Works ===

class Validation(BaseModel):
    """Validation and verification of a mathematical equation through observation and experiments"""
    name: str = Field(..., description="Name or type of validation approach")
    description: str = Field(..., description="Description of the validation method")
    observations: str = Field(..., description="Key observations that validate or support this equation in real-world contexts")
    experiments: str = Field(..., description="Experimental evidence, procedures, or results that validate this equation")
    accuracy: str = Field(..., description="Level of accuracy or agreement between theory and experimental results")


# === LEARNING STEP 7: Advanced Understanding & Extensions ===

class Extension(BaseModel):
    """A modern extension or variation of a mathematical equation"""
    name: str = Field(..., description="Name of the extension")
    description: str = Field(..., description="Description of how this extends or modifies the original equation")
    latex_formula: str = Field(..., description="LaTeX formula of the extended equation")
    motivations: str = Field(..., description="Why this extension was developed and what problems it addresses")
    applications: str = Field(..., description="Modern applications and use cases of this extension")


class Generalization(BaseModel):
    """A generalized form or broader version of a mathematical equation"""
    name: str = Field(..., description="Name of the generalized form")
    description: str = Field(..., description="How this generalizes the original equation")
    latex_formula: str = Field(..., description="LaTeX formula of the generalized equation")
    special_cases: str = Field(..., description="Special cases of this generalization that reduce back to the original equation")
    conditions: str = Field(..., description="General conditions under which this generalized form applies")


class OpenProblem(BaseModel):
    """An open problem or unsolved question related to a mathematical equation"""
    name: str = Field(..., description="Name or title of the open problem")
    description: str = Field(..., description="Description of the open problem")
    significance: str = Field(..., description="Why this problem is important or what impact solving it would have")
    current_approaches: str = Field(..., description="Current approaches or research directions toward solving this problem")
    related_research: str = Field(..., description="Related research areas and recent developments")


# === LEARNING STEP 8: Additional Resources & Context ===

class InterestingFact(BaseModel):
    """An interesting fact about a mathematical equation"""
    title: str = Field(..., description="Title or headline of the interesting fact")
    description: str = Field(..., description="Description of the interesting fact")


class SeeAlso(BaseModel):
    """A related topic or resource to guide readers to additional information"""
    name: str = Field(..., description="Name of the related topic, equation, or resource")
    description: str = Field(..., description="Brief description of how this relates to the current equation")
    relationship: str = Field(..., description="Type of relationship (e.g., 'generalization', 'special case', 'related concept', 'prerequisite')")


class Reference(BaseModel):
    """A reference or citation related to a mathematical equation"""
    title: str = Field(..., description="Title of the reference")
    authors: str = Field(..., description="Authors or creators of the reference")
    year: str = Field(..., description="Publication year")
    publication: str = Field(..., description="Where it was published (journal, book, website, conference, etc.)")
    reference_type: str = Field(..., description="Type of reference: 'paper', 'book', 'textbook', 'website', 'dissertation', 'article', etc.")
    description: str = Field(..., description="Brief description of what this reference covers and its relevance to the equation")
    url: str = Field(default="", description="URL or DOI if available")


class EquationAnalysis(BaseModel):
    """Analysis of a mathematical equation"""
    overview: Overview = Field(..., description="Overview of the equation")
    history: str = Field(..., description="Historical background and development of the equation")
    inventor_biography: InventorBiography = Field(..., description="Biography of the inventor or discoverer of the equation")
    deep_intuitions: List[DeepIntuition] = Field(..., description="List of deep intuitions and insights that led to the development of the equation")
    analogies: List[Analogy] = Field(..., description="List of analogies to explain the core ideas of the equation")
    visual_physical_examples: List[VisualPhysicalExample] = Field(..., description="List of visual and physical examples of the equation")
    modern_relevance: List[ModernRelevance] = Field(..., description="List of modern relevance and contemporary applications of the equation")
    derivation: Derivation = Field(..., description="Step-by-step derivation of the equation")
    solutions: Solution = Field(..., description="Solutions to the equation")
    theoretical_applications: List[TheoreticalApplication] = Field(..., description="List of theoretical applications of the equation")
    applied_science_applications: List[AppliedScienceApplication] = Field(..., description="List of applied science applications of the equation")
    interesting_facts: List[InterestingFact] = Field(..., description="List of interesting facts about the equation")
    extensions: List[Extension] = Field(..., description="List of modern extensions or variations of the equation")
    generalizations: List[Generalization] = Field(..., description="List of generalized forms of the equation")
    open_problems: List[OpenProblem] = Field(..., description="List of open problems or unsolved questions related to the equation")
    validations: List[Validation] = Field(..., description="List of validation and verification approaches with observations and experiments")
    references: List[Reference] = Field(..., description="List of references, citations, and sources related to the equation")
    see_also: List[SeeAlso] = Field(..., description="List of related topics, equations, or resources")


# ============================================================================
# AUDIENCE-SPECIFIC CONFIGURATION
# ============================================================================

def get_system_prompt(audience: AudienceType) -> str:
    """Get system prompt tailored to specific audience."""

    base_instruction = "You are a mathematician, physicist, and expert science communicator with deep expertise in making complex ideas intuitive."

    audience_configs = {
        "high_school": {
            "description": "high school students (ages 14-18) with basic math and science knowledge",
            "instruction": f"""
{base_instruction}
YOUR AUDIENCE: {audience_configs['high_school']['description']}

Guidelines:
1. Use ONLY simple language. Avoid technical jargon. If you must use a term, explain it immediately in everyday words.
2. Start with concrete, relatable examples from daily life before introducing abstractions.
3. Use simple analogies and visual descriptions extensively.
4. Focus on the "big picture" - why this equation matters, not all the nitty-gritty math details.
5. Keep mathematical complexity to algebra and basic calculus level.
6. Explain variables as "quantities we measure" before using mathematical notation.
7. Use emojis and storytelling to make content engaging and memorable.
8. Avoid discussing advanced extensions, open problems, or cutting-edge research.
9. Focus on: Overview, History (simple), Inventor biography, Intuitions, Analogies, Examples, Basic Applications, Simple Solutions.
10. Every reader should feel excited about mathematics, not intimidated.
"""
        },
        "graduate": {
            "description": "graduate students with strong mathematical and scientific background",
            "instruction": f"""
{base_instruction}
YOUR AUDIENCE: {audience_configs['graduate']['description']}

Guidelines:
1. Assume solid understanding of calculus, linear algebra, and scientific fundamentals.
2. Provide rigorous mathematical derivations with clear logical steps.
3. Connect to broader mathematical frameworks and theoretical significance.
4. Include advanced applications, generalizations, and extensions.
5. Discuss modern research directions and contemporary developments.
6. Balance intuition with mathematical rigor - explain both the "why" and the formal "how".
7. Include discussion of limitations, assumptions, and when equations break down.
8. Focus on: Complete overview, Deep history, Rigorous derivation, Theoretical applications, Modern extensions, Generalizations, Open problems.
9. Reference advanced concepts and related mathematical structures.
10. Prepare students for research and advanced work in their field.
"""
        },
        "professional": {
            "description": "professionals applying these equations in industry, engineering, or applied science",
            "instruction": f"""
{base_instruction}
YOUR AUDIENCE: {audience_configs['professional']['description']}

Guidelines:
1. Assume comfort with mathematics but focus on practical application over theory.
2. Emphasize real-world use cases, implementation strategies, and practical limitations.
3. Discuss computational efficiency, numerical stability, and practical considerations.
4. Include specific technologies, software, and tools where these equations are used.
5. Cover validation, accuracy expectations, and when to use which method.
6. Focus on time-to-impact and immediate utility.
7. Include cost-benefit analysis of different solution approaches.
8. Focus on: Overview, Practical applications, Numerical solutions, Implementation details, Real-world validation, Modern industrial use.
9. Minimize pure theoretical content unless it has practical implications.
10. Help professionals make informed decisions about when and how to use these equations.
"""
        },
        "researcher": {
            "description": "researchers and academics at the frontier of knowledge in this field",
            "instruction": f"""
{base_instruction}
YOUR AUDIENCE: {audience_configs['researcher']['description']}

Guidelines:
1. Assume expert-level mathematical sophistication and deep domain knowledge.
2. Focus on cutting-edge developments, recent breakthroughs, and open frontiers.
3. Discuss unsolved problems, research gaps, and opportunities for new contributions.
4. Connect to current literature and recent publications.
5. Include advanced generalizations, variations, and theoretical extensions.
6. Discuss counterintuitive aspects, edge cases, and boundary conditions.
7. Include discussion of numerical methods, computational challenges, and optimization.
8. Focus on: Advanced theory, Modern extensions, Open problems, Research frontiers, Recent developments, Comparative advantages.
9. Reference specific papers, researchers, and ongoing research programs.
10. Enable researchers to identify promising research directions and gaps in knowledge.
"""
        }
    }

    return audience_configs[audience]["instruction"]


def get_user_prompt(equation_name: str, audience: AudienceType) -> str:
    """Get user prompt tailored to specific audience."""

    base_prompt = f"Provide comprehensive information for this equation: {equation_name}"

    audience_prompts = {
        "high_school": f"""
{base_prompt}

SPECIAL FOCUS FOR HIGH SCHOOL STUDENTS
========================================
This is for bright high school students (14-18 years old) with basic math knowledge. Make it exciting and accessible!

INCLUDE:
1. Simple equation overview with everyday examples
2. Simple history (who discovered it, when, why they needed it)
3. Inventor biography (make them feel like real people with interesting lives)
4. 2-3 simple, relatable analogies from everyday life
5. 2-3 visual examples they can actually visualize or sketch
6. Simple, concrete applications they can relate to
7. Basic way to solve or use the equation
8. 2-3 fun, surprising facts about the equation

AVOID:
- Advanced mathematical notation and proofs
- Overly technical terminology
- Discussion of extensions or generalizations
- Open problems or cutting-edge research
- Numerical methods or computational details

Make every explanation intuitive. A 10th grader should be able to understand the essence of this equation.
""",
        "graduate": f"""
{base_prompt}

SPECIAL FOCUS FOR GRADUATE STUDENTS
====================================
Target advanced students who are ready to engage with sophisticated mathematics and potentially contribute to this field.

INCLUDE:
1. Complete rigorous overview with formal definitions
2. Detailed history and development timeline
3. Full inventor/discoverer biography with their other contributions
4. 3-4 deep intuitions that motivated development
5. Multiple powerful analogies at different abstraction levels
6. 2-4 visual and physical examples with technical detail
7. Complete step-by-step rigorous mathematical derivation
8. Both analytical and numerical solution methods
9. 2-3 theoretical applications with mathematical significance
10. 2-3 modern extensions and variations with LaTeX formulas
11. 2-3 generalizations showing broader mathematical frameworks
12. 2-3 open problems and research directions
13. 5-8 key references including recent papers
14. Discussion of limitations, assumptions, and edge cases

EMPHASIZE:
- Mathematical rigor and formal proofs
- Connections to broader mathematical structures
- Recent developments and research
- Potential for future contributions
- Comparative advantages vs. alternative approaches
""",
        "professional": f"""
{base_prompt}

SPECIAL FOCUS FOR PROFESSIONALS
================================
Target working professionals who need to apply these equations effectively in real-world contexts.

INCLUDE:
1. Clear, practical overview of what the equation is used for
2. Brief, relevant history (focus on practical development)
3. Concise inventor information (focus on practical contributions)
4. 2-3 practical, real-world analogies from industry/applied science
5. 2-4 examples showing actual industrial/professional applications
6. Modern relevance with specific current applications (2024-2025)
7. Practical derivation (explain logic, not just proofs)
8. Emphasis on numerical/computational solutions with implementation details
9. 2-3 real-world applied science applications with specific technologies
10. Software/tools where this equation is used
11. Validation and accuracy expectations in real-world conditions
12. Cost-benefit analysis of different solution approaches
13. Key references focused on practical guides and implementation resources
14. Common pitfalls and how to avoid them

EMPHASIZE:
- Practical utility and ROI
- Implementation considerations
- Numerical stability and computational efficiency
- Real industrial use cases
- Tools and software integration
- Time-to-implementation
""",
        "researcher": f"""
{base_prompt}

SPECIAL FOCUS FOR RESEARCHERS
==============================
Target researchers at the frontier of knowledge who may contribute new understanding or extensions to this equation.

INCLUDE:
1. Advanced, rigorous overview suitable for publication
2. Comprehensive historical development including recent evolution
3. Detailed inventor/discoverer biography with full research program
4. 3-4 profound intuitions with philosophical implications
5. Multiple analogies spanning different mathematical perspectives
6. Advanced visual/physical examples with analytical depth
7. Complete rigorous mathematical derivation with all details
8. Advanced analytical solutions and their domains of validity
9. Sophisticated numerical methods, convergence analysis, optimization
10. 2-3 deep theoretical applications with novel insights
11. 2-3 modern extensions with recent developments (2023-2025)
12. 2-3 generalizations to broader mathematical frameworks
13. 2-3 open problems with discussion of research approaches and potential impact
14. 7-8 key recent references including cutting-edge papers
15. Discussion of numerical stability, convergence, computational complexity
16. Unsolved variants and frontier questions
17. Comparative analysis with alternative formulations

EMPHASIZE:
- Cutting-edge research and recent breakthroughs
- Open problems and research opportunities
- Advanced mathematical frameworks
- Computational and theoretical challenges
- Current research landscape
- Opportunities for novel contributions
- Connection to other active research areas
"""
    }

    return audience_prompts[audience]


# ============================================================================
# SIMPLE FUNCTION - AUDIENCE AWARE
# ============================================================================

def analyze_equation(equation_name: str, audience: AudienceType = "graduate") -> EquationAnalysis:
    """
    Get LaTeX formula and analysis of a mathematical equation by name, tailored to audience.

    Args:
        equation_name (str): The name of the equation (e.g., "Poisson Equation")
        audience (AudienceType): Target audience - "high_school", "graduate", "professional", or "researcher"

    Returns:
        EquationAnalysis: Structured analysis with LaTeX formula tailored to audience
    """

    # Initialize the Gemini client
    config = ModelConfig(model_name='gemini-2.5-flash', temperature=0.3)
    client = GeminiClient(config=config)

    # Get audience-specific prompts
    system_prompt = get_system_prompt(audience)
    user_prompt = get_user_prompt(equation_name, audience)

    # Prepare input with response schema
    model_input = ModelInput(
        user_prompt=user_prompt,
        sys_prompt=system_prompt,
        response_schema=EquationAnalysis
    )

    # Generate structured response
    response = client.generate_content(model_input)

    return response


# ============================================================================
# MAIN - SIMPLE EXAMPLE
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("MATHEMATICAL EQUATION ANALYZER - AUDIENCE-AWARE")
    print("=" * 70)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python mathematical_equation_analyzer.py <equation_name> [audience]")
        print("Audience options: high_school, graduate, professional, researcher")
        print("Example: python mathematical_equation_analyzer.py 'Poisson Equation' graduate")
        sys.exit(1)

    equation = sys.argv[1]
    audience = sys.argv[2].lower() if len(sys.argv) > 2 else "graduate"

    # Validate audience
    valid_audiences = ["high_school", "graduate", "professional", "researcher"]
    if audience not in valid_audiences:
        print(f"Invalid audience: {audience}")
        print(f"Valid options: {', '.join(valid_audiences)}")
        sys.exit(1)

    print(f"\nAnalyzing for audience: {audience.replace('_', ' ').title()}")
    print("-" * 70)

    try:
        result = analyze_equation(equation, audience)

        print(f"\nEquation Name: {result.overview.equation_name}")
        print(f"LaTeX Formula: {result.overview.latex_formula}")
        print(f"Subject: {result.overview.subject}")
        print(f"Variables: {result.overview.variables}")
        print(f"History: {result.history}")

        print(f"\nBiography of the Inventor/Discoverer:")
        bio = result.inventor_biography
        print(f"  Name: {bio.name}")
        print(f"  Birth: {bio.birth_year}")
        print(f"  Death: {bio.death_year}")
        print(f"  Nationality: {bio.nationality}")
        print(f"  Education: {bio.education}")
        print(f"  Career: {bio.career}")
        print(f"  Key Contributions: {bio.key_contributions}")
        print(f"  Personal Life: {bio.personal_life}")
        print(f"  Legacy: {bio.legacy}")

        print(f"\nDeep Intuitions That Led to Development:")
        for i, intuition in enumerate(result.deep_intuitions, 1):
            print(f"  {i}. {intuition.title}")
            print(f"     Core Insight: {intuition.core_insight}")
            print(f"     Conceptual Motivation: {intuition.conceptual_motivation}")
            print(f"     Thought Process: {intuition.thought_process}")
            print(f"     Connection to Equation: {intuition.connection_to_equation}")
            print(f"     Why Necessary: {intuition.why_necessary}")

        print(f"\nAnalogies to Explain Core Ideas:")
        for i, analogy in enumerate(result.analogies, 1):
            print(f"  {i}. {analogy.title}")
            print(f"     Analogy: {analogy.analogy_description}")
            print(f"     Core Idea: {analogy.core_idea}")
            print(f"     Connection: {analogy.connection}")

        print(f"\nVisual and Physical Examples:")
        for i, example in enumerate(result.visual_physical_examples, 1):
            print(f"  {i}. {example.title}")
            print(f"     Type: {example.example_type.capitalize()}")
            print(f"     Description: {example.description}")
            print(f"     Details: {example.visualization_details}")
            print(f"     Key Observations: {example.key_observations}")
            print(f"     Interpretation: {example.interpretation}")

        print(f"\nModern Relevance and Contemporary Applications:")
        for i, relevance in enumerate(result.modern_relevance, 1):
            print(f"  {i}. {relevance.area}")
            print(f"     Description: {relevance.description}")
            print(f"     Current Applications: {relevance.current_applications}")
            print(f"     Recent Developments: {relevance.recent_developments}")
            print(f"     Future Potential: {relevance.future_potential}")

        print(f"\nDerivation:")
        print(f"  Introduction: {result.derivation.introduction}")
        for step in result.derivation.steps:
            print(f"\n  Step {step.step_number}: {step.description}")
            print(f"     Equation: {step.equation}")
            print(f"     Explanation: {step.explanation}")
        print(f"\n  Conclusion: {result.derivation.conclusion}")

        print(f"\nAssumptions:")
        for i, assumption in enumerate(result.overview.assumptions, 1):
            print(f"  {i}. {assumption.name}")
            print(f"     Description: {assumption.description}")
            print(f"     Impact: {assumption.impact}")

        print(f"\nTheoretical Applications:")
        for i, app in enumerate(result.theoretical_applications, 1):
            print(f"  {i}. {app.name}")
            print(f"     Description: {app.description}")
            print(f"     Theoretical Framework: {app.theoretical_framework}")
            print(f"     Mathematical Significance: {app.mathematical_significance}")
            print(f"     Insights Provided: {app.insights_provided}")

        print(f"\nApplied Science Applications:")
        for i, app in enumerate(result.applied_science_applications, 1):
            print(f"  {i}. {app.name}")
            print(f"     Description: {app.description}")
            print(f"     Field: {app.field}")
            print(f"     Practical Impact: {app.practical_impact}")
            print(f"     Implementation Details: {app.implementation_details}")

        print(f"\nAnalytic Solutions:")
        for i, sol in enumerate(result.solutions.analytic_solutions, 1):
            print(f"  {i}. {sol.name}")
            print(f"     Solution: {sol.solution}")
            print(f"     Conditions: {sol.conditions}")
            print(f"     Description: {sol.description}")

        print(f"\nNumerical Solutions:")
        for i, num in enumerate(result.solutions.numerical_solutions, 1):
            print(f"  {i}. {num.name}")
            print(f"     Description: {num.description}")
            print(f"     Advantages: {num.advantages}")
            print(f"     Applications: {num.applications}")

        print(f"\nInteresting Facts:")
        for i, fact in enumerate(result.interesting_facts, 1):
            print(f"  {i}. {fact.title}")
            print(f"     {fact.description}")

        print(f"\nModern Extensions:")
        for i, ext in enumerate(result.extensions, 1):
            print(f"  {i}. {ext.name}")
            print(f"     Description: {ext.description}")
            print(f"     Formula: {ext.latex_formula}")
            print(f"     Motivations: {ext.motivations}")
            print(f"     Applications: {ext.applications}")

        print(f"\nGeneralizations:")
        for i, gen in enumerate(result.generalizations, 1):
            print(f"  {i}. {gen.name}")
            print(f"     Description: {gen.description}")
            print(f"     Formula: {gen.latex_formula}")
            print(f"     Special Cases: {gen.special_cases}")
            print(f"     Conditions: {gen.conditions}")

        print(f"\nOpen Problems:")
        for i, problem in enumerate(result.open_problems, 1):
            print(f"  {i}. {problem.name}")
            print(f"     Description: {problem.description}")
            print(f"     Significance: {problem.significance}")
            print(f"     Current Approaches: {problem.current_approaches}")
            print(f"     Related Research: {problem.related_research}")

        print(f"\nValidation & Verification:")
        for i, validation in enumerate(result.validations, 1):
            print(f"  {i}. {validation.name}")
            print(f"     Description: {validation.description}")
            print(f"     Observations: {validation.observations}")
            print(f"     Experiments: {validation.experiments}")
            print(f"     Accuracy: {validation.accuracy}")

        print(f"\nReferences & Citations:")
        for i, ref in enumerate(result.references, 1):
            print(f"  {i}. {ref.title}")
            print(f"     Authors: {ref.authors}")
            print(f"     Year: {ref.year}")
            print(f"     Publication: {ref.publication}")
            print(f"     Type: {ref.reference_type}")
            print(f"     Description: {ref.description}")
            if ref.url:
                print(f"     URL/DOI: {ref.url}")

        print(f"\nSee Also:")
        for i, related in enumerate(result.see_also, 1):
            print(f"  {i}. {related.name}")
            print(f"     Relationship: {related.relationship}")
            print(f"     Description: {related.description}")

    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 70)
