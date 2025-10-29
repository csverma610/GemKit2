import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod

# For Google AI SDK
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    GOOGLE_AI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")

# For OpenAI compatible APIs
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not installed. Install with: pip install openai")

@dataclass
class ComparisonResult:
    """Structured result of model comparison"""
    original_query: str
    model_a_name: str
    model_b_name: str
    evaluation: str
    metadata: Dict[str, Any] = None

class JudgeModel(ABC):
    """Abstract base class for judge models"""
    
    @abstractmethod
    def evaluate(self, prompt: str, temperature: float = 0.1) -> str:
        pass

class GeminiJudge(JudgeModel):
    """Gemini judge using Google AI Python SDK"""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        if not GOOGLE_AI_AVAILABLE:
            raise ImportError("google-generativeai package required. Install with: pip install google-generativeai")
        
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
    def evaluate(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate evaluation using Gemini"""
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings={
                    genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                }
            )
            
            return response.text
            
        except Exception as e:
            return f"Error generating evaluation: {str(e)}"

class OpenAICompatibleJudge(JudgeModel):
    """Judge using OpenAI compatible API (works with OpenAI, Anthropic, local models, etc.)"""
    
    def __init__(self, 
                 api_key: str,
                 base_url: Optional[str] = None,
                 model_name: str = "gpt-4",
                 organization: Optional[str] = None):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required. Install with: pip install openai")
            
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization
        )
    
    def evaluate(self, prompt: str, temperature: float = 0.1) -> str:
        """Generate evaluation using OpenAI compatible API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert AI model evaluator. Provide thorough, objective, and detailed analysis of model outputs."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=4000,
                top_p=0.95
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating evaluation: {str(e)}"

class ModelOutputComparator:
    """Main comparator class that can use different judge models"""
    
    def __init__(self, judge_model: JudgeModel):
        """
        Initialize with a judge model instance
        
        Args:
            judge_model: Instance of a JudgeModel (GeminiJudge or OpenAICompatibleJudge)
        """
        self.judge = judge_model
    
    def create_comparison_prompt(self, 
                               original_query: str,
                               response_a: str, 
                               response_b: str,
                               model_a_name: str = "Model A",
                               model_b_name: str = "Model B",
                               criteria: Optional[str] = None,
                               output_format: str = "detailed") -> str:
        """
        Create a structured prompt for the judge model to compare outputs.
        
        Args:
            original_query: The original question/prompt given to both models
            response_a: First model's response
            response_b: Second model's response  
            model_a_name: Name/identifier for first model
            model_b_name: Name/identifier for second model
            criteria: Specific criteria for comparison (optional)
            output_format: Format of evaluation ('detailed', 'concise', 'structured')
        
        Returns:
            Formatted comparison prompt
        """
        
        default_criteria = """
        Please evaluate both responses based on:
        1. **Accuracy**: Correctness of information and facts
        2. **Completeness**: How thoroughly the response addresses the query
        3. **Clarity**: How clear and understandable the response is
        4. **Helpfulness**: Practical value and usefulness to the user
        5. **Coherence**: Logical flow and organization of ideas
        6. **Relevance**: How well the response stays on topic
        """
        
        evaluation_criteria = criteria if criteria else default_criteria
        
        format_instructions = {
            "detailed": """
**Instructions:**
Please provide your evaluation in the following format:

1. **Executive Summary**: Brief overview comparing both responses
2. **Detailed Analysis**: 
   - Strengths and weaknesses of each response
   - How well each addresses the original query
   - Quality comparison across the evaluation criteria
3. **Scoring**: Rate each response on a scale of 1-10 for each criterion, with brief justification
4. **Winner**: Which response is better overall and why (or if it's a tie)
5. **Recommendations**: Specific suggestions for how each response could be improved

Be objective, thorough, and provide concrete examples from the responses to support your evaluation.""",
            
            "concise": """
**Instructions:**
Provide a concise evaluation with:
1. **Winner**: Which response is better and why (2-3 sentences)
2. **Key Differences**: Main strengths/weaknesses of each (3-4 bullet points)
3. **Overall Scores**: Rate each response 1-10 overall with brief reasoning""",
            
            "structured": """
**Instructions:**
Provide evaluation in JSON format:
```json
{
  "summary": "Brief comparison overview",
  "scores": {
    "model_a": {"accuracy": X, "completeness": X, "clarity": X, "helpfulness": X, "overall": X},
    "model_b": {"accuracy": X, "completeness": X, "clarity": X, "helpfulness": X, "overall": X}
  },
  "winner": "model_a/model_b/tie",
  "reasoning": "Why this model won",
  "strengths": {
    "model_a": ["strength1", "strength2"],
    "model_b": ["strength1", "strength2"]
  },
  "weaknesses": {
    "model_a": ["weakness1", "weakness2"], 
    "model_b": ["weakness1", "weakness2"]
  }
}
```"""
        }
        
        instructions = format_instructions.get(output_format, format_instructions["detailed"])
        
        prompt = f"""You are an expert AI model evaluator. Please compare the following two responses to the same query and provide a comprehensive analysis.

**Original Query:**
{original_query}

**{model_a_name} Response:**
{response_a}

**{model_b_name} Response:**
{response_b}

**Evaluation Criteria:**
{evaluation_criteria}

{instructions}

Focus on being objective, specific, and providing actionable insights. Use concrete examples from the responses to support your analysis."""

        return prompt

    def compare_models(self, 
                      original_query: str,
                      response_a: str, 
                      response_b: str,
                      model_a_name: str = "Model A",
                      model_b_name: str = "Model B", 
                      criteria: Optional[str] = None,
                      output_format: str = "detailed",
                      temperature: float = 0.1) -> ComparisonResult:
        """
        Complete workflow to compare two model outputs.
        
        Returns:
            ComparisonResult object with evaluation details
        """
        # Create comparison prompt
        prompt = self.create_comparison_prompt(
            original_query, response_a, response_b, 
            model_a_name, model_b_name, criteria, output_format
        )
        
        # Get evaluation from judge model
        evaluation = self.judge.evaluate(prompt, temperature)
        
        # Return structured result
        return ComparisonResult(
            original_query=original_query,
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            evaluation=evaluation,
            metadata={
                "criteria": criteria,
                "output_format": output_format,
                "temperature": temperature
            }
        )

    def save_results(self, results: List[ComparisonResult], filename: str = "comparison_results.json"):
        """Save comparison results to a JSON file"""
        data = []
        for result in results:
            data.append({
                "original_query": result.original_query,
                "model_a_name": result.model_a_name,
                "model_b_name": result.model_b_name,
                "evaluation": result.evaluation,
                "metadata": result.metadata
            })
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {filename}")

# Factory functions for easy setup
def create_gemini_comparator(api_key: str, model_name: str = "gemini-1.5-pro") -> ModelOutputComparator:
    """Create comparator using Gemini as judge"""
    judge = GeminiJudge(api_key, model_name)
    return ModelOutputComparator(judge)

def create_openai_comparator(api_key: str, 
                           model_name: str = "gpt-4",
                           base_url: Optional[str] = None) -> ModelOutputComparator:
    """Create comparator using OpenAI compatible API as judge"""
    judge = OpenAICompatibleJudge(api_key, base_url, model_name)
    return ModelOutputComparator(judge)

def create_anthropic_comparator(api_key: str, model_name: str = "claude-3-5-sonnet-20241022") -> ModelOutputComparator:
    """Create comparator using Anthropic API as judge"""
    judge = OpenAICompatibleJudge(
        api_key=api_key,
        base_url="https://api.anthropic.com/v1",
        model_name=model_name
    )
    return ModelOutputComparator(judge)

def create_local_model_comparator(base_url: str, 
                                 model_name: str,
                                 api_key: str = "dummy") -> ModelOutputComparator:
    """Create comparator using local model (like Ollama, vLLM, etc.) as judge"""
    judge = OpenAICompatibleJudge(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name
    )
    return ModelOutputComparator(judge)

def example_usage():
    """Comprehensive example showing different judge models"""
    
    # Sample data
    original_query = "Explain the difference between supervised and unsupervised machine learning."
    
    response_a = """Supervised learning uses labeled training data where the correct answers are provided. The model learns to map inputs to outputs based on these examples. Common tasks include classification (predicting categories) and regression (predicting numbers). Examples: email spam detection, house price prediction.

Unsupervised learning works with data that has no labels or correct answers. The model finds hidden patterns and structures in the data. Common tasks include clustering (grouping similar items) and dimensionality reduction. Examples: customer segmentation, data compression."""
    
    response_b = """The main difference is that supervised learning has a teacher (labeled data) while unsupervised learning doesn't. In supervised learning, you show the algorithm examples with correct answers, like showing it photos labeled 'cat' or 'dog'. In unsupervised learning, you just give it data and let it find patterns on its own, like giving it a bunch of photos and letting it figure out there are different types of animals."""
    
    print("=== GEMINI JUDGE EXAMPLE ===")
    try:
        # Using Gemini as judge
        gemini_comparator = create_gemini_comparator("your-gemini-api-key")
        result = gemini_comparator.compare_models(
            original_query=original_query,
            response_a=response_a,
            response_b=response_b,
            model_a_name="Technical Model",
            model_b_name="Simple Model",
            output_format="detailed"
        )
        print(result.evaluation)
    except Exception as e:
        print(f"Gemini example failed: {e}")
    
    print("\n=== OPENAI JUDGE EXAMPLE ===")
    try:
        # Using OpenAI as judge
        openai_comparator = create_openai_comparator("your-openai-api-key", "gpt-4")
        result = openai_comparator.compare_models(
            original_query=original_query,
            response_a=response_a,
            response_b=response_b,
            model_a_name="Technical Model", 
            model_b_name="Simple Model",
            output_format="concise"
        )
        print(result.evaluation)
    except Exception as e:
        print(f"OpenAI example failed: {e}")

if __name__ == "__main__":
    print("Model Output Comparison Tool with Multiple Judge Options!")
    print("\nAvailable judge models:")
    print("1. Gemini (Google AI SDK)")
    print("2. OpenAI models (GPT-4, etc.)")
    print("3. Anthropic (Claude)")
    print("4. Local models (Ollama, vLLM, etc.)")
    print("\nInstall required packages:")
    print("pip install google-generativeai openai")
    
    # Uncomment to run examples
    # example_usage()
