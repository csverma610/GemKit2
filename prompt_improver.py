import sys
from typing import Optional, Dict, Any
from google import genai

from gemini_client import GeminiClient

class PromptImprover:
    """
    A class to improve prompts using Google's Generative AI API.
    
    This class provides methods to enhance prompt quality with customizable
    improvement strategies and robust error handling.
    """
    
    def __init__(self):
        """
        Initialize the PromptImprover.
        """
        self.client = GeminiClient()
    
    def generate_text(
        self, 
        original_prompt: str, 
        improvement_strategy: str = "general",
        return_function: bool = True,
        additional_context: Optional[str] = None
    ) -> str:
        """
        Improve the quality of a given prompt.
        
        Args:
            original_prompt (str): The original prompt to improve
            improvement_strategy (str): Strategy for improvement 
                ('general', 'clarity', 'specificity', 'creativity')
            return_function (bool): Whether to return a Python function format
            additional_context (Optional[str]): Additional context for improvement
            
        Returns:
            str: The improved prompt or function
            
        Raises:
            ValueError: If the original prompt is empty
            Exception: If API call fails
        """
        if not original_prompt or not original_prompt.strip():
            raise ValueError("Original prompt cannot be empty")
        
        prompt = self._build_prompt(
            original_prompt, improvement_strategy, return_function, additional_context
        )
        
        try:
            text = self.client.generate_text(prompt)
            return text
            
        except Exception as e:
            raise Exception(f"Prompt improvement failed: {e}")
    
    def _build_prompt(
        self, 
        original_prompt: str, 
        strategy: str, 
        return_function: bool,
        additional_context: Optional[str]
    ) -> str:
        """Build the improvement instruction prompt."""
        
        strategy_instructions = {
            "general": "Make it clearer, more specific, and more effective",
            "clarity": "Focus on making the prompt clearer and easier to understand",
            "specificity": "Make the prompt more specific and detailed",
            "creativity": "Enhance the creative aspects and inspire better responses"
        }
        
        strategy_instruction = strategy_instructions.get(
            strategy, strategy_instructions["general"]
        )
        
        base_prompt = f"""
Improve the quality of the following prompt: "{original_prompt}"

Improvement focus: {strategy_instruction}

Requirements:
1. Make the prompt more effective and engaging
2. Ensure it's clear and actionable
3. Add relevant context where helpful
4. Maintain the original intent
"""
        
        if additional_context:
            base_prompt += f"\n5. Consider this additional context: {additional_context}"
        
        if return_function:
            base_prompt += """

Please provide your response as a Python function in the following format:

```python
def get_improved_prompt():
    \"\"\"
    Returns the improved version of the original prompt.
    \"\"\"
    return '''[Your improved prompt here]'''
```
"""
        return base_prompt
    
def main():
    """
    Main function to run from command line.
    Usage: python prompt_improver.py "your prompt here" [strategy]
    """
    if len(sys.argv) < 2:
        print("Usage: python prompt_improver.py \"your prompt here\" [strategy]")
        print("Strategies: general, clarity, specificity, creativity")
        sys.exit(1)
    
    original_prompt = sys.argv[1]
    strategy = sys.argv[2] if len(sys.argv) > 2 else "general"
    
    try:
        improver = PromptImprover()
        improved_prompt = improver.generate_text(
            original_prompt, 
            improvement_strategy=strategy
        )
        print(improved_prompt)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
