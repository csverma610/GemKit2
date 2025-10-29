from google import genai
from pydantic import BaseModel, Field
import json
import random
import string # Imported for generating option IDs (A, B, C, ...) from a list

# --- Pydantic Structured Data Schemas for Analysis Output ---

class AnswerOption(BaseModel):
    """Schema for a single answer choice analyzed by the model."""
    id: str = Field(description="A single letter identifier for the choice (e.g., 'A', 'B', 'C', 'D').")
    text: str = Field(description="The text of the answer choice.")
    # The justification field is where the model explains why this choice is correct or incorrect.
    justification: str = Field(description="A brief, precise explanation detailing why this choice is the correct answer or an incorrect distractor.")
    # Individual confidence score for this specific choice
    choice_confidence_score: float = Field(description="The model's confidence (0.0 to 1.0) that this specific choice is the correct answer to the question.")

class MCQResponse(BaseModel):
    """The complete structured output report for analyzing and answering a single MCQ. Supports multiple correct answers."""
    # UPDATED: Now a list of IDs to support single or multiple correct answers.
    correct_answer_ids: list[str] = Field(description="The list of IDs (e.g., ['A', 'C']) corresponding to the choices the model determined to be correct.")
    explanation: str = Field(description="A concise summary of the core concept and why the correct answer(s) are the solution.")
    analyzed_choices: list[AnswerOption] = Field(description="The list of all options, each with a detailed justification and individual confidence score.")
    # Overall certainty score, updated to range from -1.0 to 1.0
    confidence_score: float = Field(description="The model's overall certainty in the chosen correct_answer_id(s), expressed as a float between -1.0 (definitely wrong) and +1.0 (definitely correct), with 0.0 representing uncertainty.")


class MCQAnalyzer:
    """
    A class to interact with the Gemini API to analyze a given multiple-choice
    question and return a structured answer with justifications and confidence scores.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """Initializes the Gemini client and configuration."""
        self.model_name = model_name
        self.SYSTEM_INSTRUCTION = (
            "You are an expert academic tutor. Your task is to analyze a single provided multiple "
            "choice question, provide a precise justification for *every* single option, and state your overall certainty in the range of -1.0 to 1.0. Crucially, you must also "
            "provide the probability (0.0 to 1.0) that *each individual choice* is the correct answer. "
            "You must adhere strictly to the provided JSON schema."
        )

        try:
            self.client = genai.Client()
        except Exception as e:
            self.client = None
            print(f"Error initializing Gemini Client: {e}")
            print("Client is not initialized. API calls will be skipped.")

    def analyze_mcq(self, question: str, choices: dict[str, str] | list[str], is_multiple_select: bool = False) -> MCQResponse | None:
        """
        Analyzes a single MCQ by sending it to the model for structured answering.
        
        Args:
            question: The text of the question.
            choices: A dictionary mapping the choice ID (str) to the choice text (str)
                     OR a list of strings (the choice texts).
            is_multiple_select: Set to True if the question allows for multiple correct answers.
            
        Returns:
            A MCQResponse object with the structured answer, or None if the call fails.
        """
        if not self.client:
            return None

        # --- LOGIC TO HANDLE LIST INPUT AND CONVERT TO DICTIONARY ---
        final_choices_dict = {}
        if isinstance(choices, list):
            # Convert list of strings to dictionary using A, B, C, ... as keys
            if not choices:
                 print("Error: Choices list is empty.")
                 return None
            for i, text in enumerate(choices):
                # Use string.ascii_uppercase to generate IDs (A, B, C, ...)
                if i < len(string.ascii_uppercase):
                    choice_id = string.ascii_uppercase[i]
                    final_choices_dict[choice_id] = text
                else:
                    print("Warning: Too many choices provided (max 26 supported). Skipping extras.")
                    break
        elif isinstance(choices, dict):
            final_choices_dict = choices
        else:
            print("Error: Choices must be a dictionary (ID: Text) or a list of strings.")
            return None
        # -----------------------------------------------------------

        # 1. Format the input question and choices into a clear text prompt for the model
        # Sort choices by key (A, B, C, ...) for clean printing in the prompt
        sorted_choices = sorted(final_choices_dict.items())
        choices_text = "\n".join([f"[{id}] {text}" for id, text in sorted_choices])
        
        # New prompt adjustment based on question type
        answer_instruction = "This is a single-choice question, so you must identify ONLY ONE correct answer and return a list with a single ID."
        if is_multiple_select:
             answer_instruction = "This is a multiple-select question, so you must identify ALL correct answers and return a list of all correct IDs."
             
        user_prompt = (
            f"Analyze and answer the following question. {answer_instruction} "
            f"Provide justifications and individual confidence scores for all options.\n\n"
            f"Question: {question}\n\nChoices:\n{choices_text}"
        )

        print(f"Sending prompt to model for analysis...")

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_schema": MCQResponse, # Requesting a single structured object
                },
                system_instruction=self.SYSTEM_INSTRUCTION
            )

            # Access the raw JSON text response
            self.json_text = response.text
            # Access the response as an instantiated Pydantic object
            return response.parsed

        except genai.errors.APIError as e:
            print(f"\nAn API error occurred: {e}")
        except Exception as e:
            print(f"\nAn unexpected error occurred during analysis: {e}")

        return None

    def display_results(self, analysis: MCQResponse):
        """Prints the raw JSON and the structured, validated analysis."""

        print("\n--- Raw JSON Response Text ---")
        print(getattr(self, 'json_text', 'No raw JSON available.'))
        print("----------------------------\n")

        if not analysis:
            print("No structured analysis was returned.")
            return

        print("--- Structured Question Analysis ---")
        print(f"Question Analysis Complete.")
        # Display the overall certainty score using the new scale
        print(f"-> Overall Certainty Score: {analysis.confidence_score:.2f} (Scale: -1.0=Wrong, 1.0=Correct, 0.0=Uncertain)")
        # UPDATED: Display list of correct IDs
        correct_ids_str = ', '.join(analysis.correct_answer_ids)
        print(f"-> Correct Answer ID(s): {correct_ids_str}") 
        print(f"-> Overall Explanation: {analysis.explanation}")

        print("\n--- Detailed Choice Justifications and Individual Scores ---")
        for choice in analysis.analyzed_choices:
            # UPDATED: Check if choice ID is in the list of correct IDs
            is_correct = "CORRECT" if choice.id in analysis.correct_answer_ids else "INCORRECT"
            # Format and display individual confidence score (still 0.0 to 1.0 probability)
            choice_conf_percent = f"{choice.choice_confidence_score * 100:.1f}%" 
            print(f"   [{choice.id}] {choice.text} ({is_correct})")
            print(f"      - Probability of being Correct: {choice_conf_percent}") 
            print(f"      - Justification: {choice.justification}")
        print("------------------------------------")


# --- Example Usage ---
if __name__ == "__main__":
    # 2. Instantiate the analyzer
    analyzer = MCQAnalyzer()

    # ====================================================================
    # EXAMPLE 1: Single Choice - Dictionary Input
    # ====================================================================
    sample_question_dict = "Which pillar of Object-Oriented Programming (OOP) allows a child class to inherit methods and properties from a parent class?"
    sample_choices_dict = {
        "A": "Encapsulation",
        "B": "Polymorphism",
        "C": "Inheritance",
        "D": "Abstraction",
    }

    print("\n\n--- Analyzing MCQ with Dictionary Input (Single Choice) ---")
    analysis_result_dict = analyzer.analyze_mcq(sample_question_dict, sample_choices_dict, is_multiple_select=False)
    if analysis_result_dict is not None:
        analyzer.display_results(analysis_result_dict)
        
    # ====================================================================
    # EXAMPLE 4: Multiple Select - List Input (New Scenario)
    # ====================================================================
    sample_question_multi = "Which of the following are considered primary colors in the subtractive (CMYK) color model used for printing?"
    sample_choices_multi = [
        "Red",
        "Yellow",
        "Cyan",
        "Green",
        "Magenta",
        "Blue"
    ]

    print("\n\n--- Analyzing MCQ with List Input (Multiple Select) ---")
    # Pass is_multiple_select=True to instruct the model to look for all correct answers
    analysis_result_multi = analyzer.analyze_mcq(sample_question_multi, sample_choices_multi, is_multiple_select=True)

    # 4. Display the results for the multiple select input
    if analysis_result_multi is not None:
        analyzer.display_results(analysis_result_multi)

