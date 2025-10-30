import sys
from typing import Dict, Any

from gemini_client import GeminiClient

class PromptGenerator:
    """
    Generates various evaluation prompts for the PromptEvaluator.

    This class centralizes the logic for creating different types of prompts
    used to evaluate a model's response, ensuring consistency and ease of
    maintenance.
    """
    
    @staticmethod
    def get_criteria_instructions() -> Dict[str, str]:
        """Get the mapping of evaluation criteria to their instructions."""
        return {
            "comprehensive": """
Evaluate the response comprehensively across all aspects:
- Content accuracy and relevance
- Format and structure adherence
- Completeness of requirements
- Tone and style matching
- Specific instruction following
""",
            "format": """
Focus specifically on format and structure:
- Does it follow the requested format?
- Are structural elements correct?
- Is the organization as requested?
""",
            "content": """
Focus on content quality and relevance:
- Is the content relevant to the prompt?
- Does it address all key points?
- Is the information accurate?
""",
            "requirements": """
Focus on specific requirement fulfillment:
- Are all explicit requirements met?
- Are constraints respected?
- Are specifications followed?
"""
        }
    
    @classmethod
    def build_evaluation_prompt(
        cls, 
        prompt: str, 
        response: str, 
        criteria: str
    ) -> str:
        """
        Builds a detailed prompt for a comprehensive evaluation.

        Args:
            prompt (str): The original prompt that was given to the model.
            response (str): The model's response to be evaluated.
            criteria (str): The type of evaluation criteria to use ('comprehensive',
                            'format', 'content', or 'requirements').

        Returns:
            str: The complete evaluation prompt.
        """
        criteria_instructions = cls.get_criteria_instructions()
        criteria_instruction = criteria_instructions.get(
            criteria, criteria_instructions["comprehensive"]
        )
        
        evaluation_prompt = f"""
You are an expert evaluator. Analyze whether the given response properly follows the original prompt instructions.

ORIGINAL PROMPT:
"{prompt}"

RESPONSE TO EVALUATE:
"{response}"

EVALUATION CRITERIA:
{criteria_instruction}

Please provide your evaluation in the following format:

ADHERENCE SCORE: [X/10]
(10 = Perfect adherence, 1 = Poor adherence)

DETAILED ANALYSIS:
✅ What the response did well:
- [List specific strengths]

❌ What the response missed or did poorly:
- [List specific issues]

📋 INSTRUCTION COMPLIANCE:
- [Check each major instruction and mark as ✅ Followed or ❌ Not Followed]

💡 RECOMMENDATIONS:
- [Specific suggestions for improvement]

OVERALL ASSESSMENT:
[Brief summary of whether the response successfully follows the prompt]
"""
        
        return evaluation_prompt
    
    @staticmethod
    def build_quick_check_prompt(prompt: str, response: str) -> str:
        """
        Builds a prompt for a quick, binary evaluation.

        Args:
            prompt (str): The original prompt.
            response (str): The model's response.

        Returns:
            str: The quick evaluation prompt.
        """
        return f"""
Analyze if this response follows the given prompt instructions. Provide a simple evaluation.

PROMPT: "{prompt}"
RESPONSE: "{response}"

Answer in this exact format:
FOLLOWS INSTRUCTIONS: [YES/NO]
CONFIDENCE: [High/Medium/Low]
REASON: [Brief 1-2 sentence explanation]
"""
    
    @staticmethod
    def build_scoring_prompt(prompt: str, response: str) -> str:
        """
        Builds a prompt for a numerical score evaluation.

        Args:
            prompt (str): The original prompt.
            response (str): The model's response.

        Returns:
            str: The scoring evaluation prompt.
        """
        return f"""
Rate how well this response follows the given prompt instructions on a scale of 1-10.

PROMPT: "{prompt}"
RESPONSE: "{response}"

Provide only:
SCORE: [X/10]
JUSTIFICATION: [One sentence explaining the score]
"""


class PromptEvaluator:
    """
    Evaluates how well a model's response adheres to the instructions in a given prompt.

    This class uses the Gemini API to perform the evaluation, providing different
    levels of detail, from a quick binary check to a comprehensive analysis.
    """
    
    def __init__(self):
        """
        Initializes the PromptEvaluator.
        """
        self.prompt_generator = PromptGenerator()
        
        try:
            self.client = GeminiClient()
        except Exception as e:
            raise Exception(f"Failed to initialize client: {e}")
    
    def evaluate_response(
        self, 
        prompt: str, 
        response: str,
        evaluation_criteria: str = "comprehensive"
    ) -> str:
        """
        Performs a comprehensive evaluation of a model's response.

        Args:
            prompt (str): The original prompt given to the model.
            response (str): The model's response to be evaluated.
            evaluation_criteria (str, optional): The criteria to use for the evaluation.
                                                 Can be 'comprehensive', 'format', 'content',
                                                 or 'requirements'. Defaults to "comprehensive".

        Returns:
            str: A detailed evaluation, including a score, analysis, and recommendations.
        """
        self._validate_inputs(prompt, response)
        
        evaluation_prompt = self.prompt_generator.build_evaluation_prompt(
            prompt, response, evaluation_criteria
        )
        
        return self._make_api_call(evaluation_prompt, "Evaluation failed")
    
    def quick_check(self, prompt: str, response: str) -> str:
        """
        Performs a quick, binary check of a model's response.

        Args:
            prompt (str): The original prompt.
            response (str): The model's response.

        Returns:
            str: A simple "YES/NO" evaluation with a brief explanation.
        """
        self._validate_inputs(prompt, response)
        
        quick_prompt = self.prompt_generator.build_quick_check_prompt(
            prompt, response
        )
        
        return self._make_api_call(quick_prompt, "Quick check failed")
    
    def score_only(self, prompt: str, response: str) -> str:
        """
        Provides a numerical score for a model's response.

        Args:
            prompt (str): The original prompt.
            response (str): The model's response.

        Returns:
            str: A numerical score (1-10) with a brief justification.
        """
        self._validate_inputs(prompt, response)
        
        scoring_prompt = self.prompt_generator.build_scoring_prompt(
            prompt, response
        )
        
        return self._make_api_call(scoring_prompt, "Scoring failed")
    
    def _validate_inputs(self, prompt: str, response: str) -> None:
        """
        Validate that prompt and response are not empty.
        
        Args:
            prompt (str): The prompt to validate
            response (str): The response to validate
            
        Raises:
            ValueError: If either input is empty
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if not response or not response.strip():
            raise ValueError("Response cannot be empty")
    
    def _make_api_call(self, prompt: str, error_message: str) -> str:
        """
        Make an API call to the model.
        
        Args:
            prompt (str): The prompt to send
            error_message (str): Error message if call fails
            
        Returns:
            str: The model's response
            
        Raises:
            Exception: If API call fails
        """
        try:
            text  = self.client.generate_text(prompt)
            return text
            
        except Exception as e:
            raise Exception(f"{error_message}: {e}")


def main():
    """
    Main function to run from command line.
    Usage: python prompt_evaluator.py "prompt" "response" [evaluation_type]
    """
    if len(sys.argv) < 3:
        print("Usage: python prompt_evaluator.py \"prompt\" \"response\" [evaluation_type]")
        print("Evaluation types: comprehensive, format, content, requirements, quick, score")
        sys.exit(1)
    
    prompt = sys.argv[1]
    response = sys.argv[2]
    evaluation_type = sys.argv[3] if len(sys.argv) > 3 else "comprehensive"
    
    try:
        evaluator = PromptEvaluator()
        
        if evaluation_type == "quick":
            result = evaluator.quick_check(prompt, response)
        elif evaluation_type == "score":
            result = evaluator.score_only(prompt, response)
        else:
            result = evaluator.evaluate_response(prompt, response, evaluation_type)
            
        print(result)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
