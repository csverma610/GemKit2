"""
Mathematical Equation Story Generator
Creates engaging, narrative-driven explanations of mathematical equations for absolute beginners.
Generates flowing, professional narratives like those found in popular science magazines.
Uses Gemini API to make complex mathematics intuitive and memorable through compelling storytelling.
"""

import sys
from pydantic import BaseModel, Field
from typing import List
from gemini_client import GeminiClient, ModelConfig, ModelInput


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class MathematicalEquationStory(BaseModel):
    """A coherent narrative story explaining a mathematical equation"""
    equation_name: str = Field(..., description="Name of the equation being explained")
    latex_formula: str = Field(..., description="The LaTeX representation of the equation")
    title: str = Field(..., description="An engaging title for the story")
    subtitle: str = Field(..., description="A compelling subtitle or subheading")

    # The complete narrative - a flowing story, not fragmented sections
    story: str = Field(..., description="The complete narrative story as flowing prose, written like a professional science article")

    # Supporting materials
    vocabulary_notes: str = Field(..., description="Brief explanations of technical terms used in the story")
    discussion_questions: List[str] = Field(..., description="Thought-provoking questions for readers to reflect on (3-5 questions)")


# ============================================================================
# STORY GENERATION FUNCTION
# ============================================================================

def generate_equation_story(equation_name: str) -> MathematicalEquationStory:
    """
    Generates a narrative-driven explanation of a mathematical equation.

    This function creates a detailed prompt that instructs the Gemini model to
    write a compelling story about the specified equation, in the style of a
    popular science magazine. The story is designed to be accessible to a
    general audience and to convey the beauty and importance of the mathematics.

    Args:
        equation_name (str): The name of the equation to be explained (e.g.,
                             "Pythagorean Theorem", "E=mc²").

    Returns:
        MathematicalEquationStory: A Pydantic model containing the generated
                                   story and supporting materials.
    """

    # Initialize the Gemini client
    config = ModelConfig(model_name='gemini-2.5-flash', temperature=0.7)
    client = GeminiClient(config=config)

    # Create the story generation prompt
    user_prompt = f"""
Write a compelling narrative essay about {equation_name}, written like an article you'd find in Scientific American, Cosmos magazine, or a popular science publication.

CORE PRINCIPLES:
============================================================================

1. NARRATIVE FLOW: Write as a single, coherent story with natural transitions—NOT as sections or labeled parts. The story should flow like a professional essay.

2. ACCESSIBILITY: Make this accessible to intelligent high school students with no specialized math background. Use clear, elegant language.

3. ENGAGEMENT: Draw readers in with genuine intellectual interest. Show why this equation matters and why it's beautiful.

4. ACCURACY: Be mathematically accurate about core concepts, though you can simplify details for clarity.

5. STRUCTURE (But integrated seamlessly):
   - Hook readers with intrigue or a compelling question in the opening
   - Introduce the human or historical context: Who needed this? Why?
   - Build understanding through concrete examples and observations
   - Show how the equation emerges naturally from these observations
   - Explain what the equation really means and why it has this form
   - Connect to real-world applications and implications
   - Leave readers with a sense of wonder about the power of mathematics

THE STORY SHOULD:
============================================================================
- Feel like you're reading journalism or an essay, not a textbook
- Use vivid details and relatable contexts to ground abstract concepts
- Build intellectual momentum—each idea builds on the last
- Include moments that make readers think "Oh! That's why!"
- Show the elegance and beauty of the mathematics
- Make readers feel smart for understanding something profound
- Be substantial enough to fully explore the equation (700-1200 words)

SPECIFIC GUIDANCE FOR {equation_name}:
============================================================================
Research and explore:
1. Where did this equation come from? Who discovered it and why?
2. What real problem does it solve?
3. What makes this equation elegant or surprising?
4. How is it actually used in the modern world?
5. What misconceptions do people have about it?

Then weave these elements into a flowing narrative that brings the equation to life.

TONE & STYLE:
============================================================================
- Professional but warm and engaging
- Conversational without being casual
- Use specific examples and concrete details
- Build a sense of discovery as readers progress
- Celebrate the ingenuity of mathematical thinking
- Make the subject matter feel important and relevant

Write this as a complete essay that could be published in a science magazine.
The reader should finish feeling they understand something profound about mathematics
and the world."""

    # System prompt guiding the storyteller
    system_prompt = """You are a science writer and storyteller for major publications like Scientific American, Cosmos, and The Atlantic. You have the rare ability to make complex mathematical ideas feel accessible, beautiful, and profoundly relevant without dumbing them down.

Your approach:
- You write flowing essays, not explanations with section headers
- You use narrative momentum to carry readers through complex ideas
- You find the human story behind each equation—the curiosity, the problem, the "aha" moment
- You use vivid examples and clear language to make abstractions concrete
- You respect your readers' intelligence while never assuming specialized knowledge
- You show why mathematics matters to real people and real problems
- You celebrate intellectual discovery

Your strength is making readers think: "I didn't know that. That's beautiful. That matters."

Write engaging, flowing prose that reads like published science journalism."""

    # Prepare input with response schema
    model_input = ModelInput(
        user_prompt=user_prompt,
        sys_prompt=system_prompt,
        response_schema=MathematicalEquationStory
    )

    # Generate structured response
    response = client.generate_content(model_input)

    return response


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def display_story(story: MathematicalEquationStory):
    """Display the story as a coherent, published article."""

    # Header with title and metadata
    print("\n" + "=" * 80)
    print(f"\n{story.title}")
    print(f"\n{story.subtitle}")
    print(f"\n{'—' * 80}")
    print(f"Equation: {story.equation_name}")
    print(f"Formula: {story.latex_formula}")
    print("\n" + "=" * 80 + "\n")

    # The main narrative - displayed as continuous prose
    print(story.story)

    # Supporting materials
    print("\n" + "=" * 80)
    print("\nVOCABULARY & CONCEPTS")
    print("=" * 80 + "\n")
    print(story.vocabulary_notes)

    print("\n" + "=" * 80)
    print("\nDISCUSSION QUESTIONS")
    print("=" * 80 + "\n")
    for i, question in enumerate(story.discussion_questions, 1):
        print(f"{i}. {question}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    print("=" * 80)
    print("MATHEMATICAL EQUATION STORY GENERATOR")
    print("Science Writing About Mathematics")
    print("=" * 80)

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("\nUsage: python mathematical_equation_story.py <equation_name>")
        print("\nExamples:")
        print("  python mathematical_equation_story.py 'Pythagorean Theorem'")
        print("  python mathematical_equation_story.py 'E=mc²'")
        print("  python mathematical_equation_story.py 'Newton\\'s Law of Motion'")
        sys.exit(1)

    equation_name = sys.argv[1]

    print(f"\nGenerating story for: {equation_name}")
    print("(Crafting a compelling narrative...)\n")

    try:
        story = generate_equation_story(equation_name)
        display_story(story)

    except Exception as e:
        print(f"\nError generating story: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("\nStory generated successfully.")
    print("=" * 80)
