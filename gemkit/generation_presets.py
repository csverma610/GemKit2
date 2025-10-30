"""
Generation parameter presets for different use cases.

Use these presets to easily configure deterministic vs creative output.
"""

# Deterministic Output - Highly consistent, minimal creativity
DETERMINISTIC = {
    'temperature': 0.0,
    'top_p': 0.1,
    'top_k': 1,
}

# Precise Output - Very focused, low randomness
PRECISE = {
    'temperature': 0.3,
    'top_p': 0.5,
    'top_k': 20,
}

# Balanced Output - Good middle ground (Gemini's default behavior is similar)
BALANCED = {
    'temperature': 1.0,
    'top_p': 0.95,
    'top_k': 40,
}

# Creative Output - More diverse and creative responses
CREATIVE = {
    'temperature': 1.5,
    'top_p': 0.95,
    'top_k': 64,
}

# Very Creative Output - Maximum creativity and diversity
VERY_CREATIVE = {
    'temperature': 2.0,
    'top_p': 1.0,
    'top_k': None,  # No top-k restriction
}

# Code Generation - Optimized for generating code
CODE_GENERATION = {
    'temperature': 0.2,
    'top_p': 0.8,
    'top_k': 40,
}

# Storytelling - Optimized for creative writing
STORYTELLING = {
    'temperature': 1.4,
    'top_p': 0.95,
    'top_k': 50,
}

# Data Extraction - For extracting structured data from text
DATA_EXTRACTION = {
    'temperature': 0.0,
    'top_p': 0.1,
    'top_k': 1,
}

# Summary - For summarization tasks
SUMMARY = {
    'temperature': 0.4,
    'top_p': 0.9,
    'top_k': 40,
}

# Q&A - For question answering
QA = {
    'temperature': 0.2,
    'top_p': 0.8,
    'top_k': 40,
}


def get_preset(preset_name: str) -> dict:
    """
    Retrieves a generation preset by its name.

    This function provides a convenient way to access predefined sets of
    generation parameters for different use cases, such as deterministic
    output, creative writing, or code generation.

    Args:
        preset_name (str): The name of the preset to retrieve. This is
                           case-insensitive.

    Returns:
        dict: A dictionary containing the generation parameters for the
              specified preset.

    Raises:
        ValueError: If the specified `preset_name` is not recognized.
    """
    presets = {
        'deterministic': DETERMINISTIC,
        'precise': PRECISE,
        'balanced': BALANCED,
        'creative': CREATIVE,
        'very_creative': VERY_CREATIVE,
        'code_generation': CODE_GENERATION,
        'code': CODE_GENERATION,
        'storytelling': STORYTELLING,
        'story': STORYTELLING,
        'data_extraction': DATA_EXTRACTION,
        'extract': DATA_EXTRACTION,
        'summary': SUMMARY,
        'summarize': SUMMARY,
        'qa': QA,
        'question': QA,
    }

    preset_name_lower = preset_name.lower()
    if preset_name_lower not in presets:
        available = ', '.join(sorted(set(presets.keys())))
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Available presets: {available}"
        )

    return presets[preset_name_lower].copy()


# Usage examples in docstring
__doc__ += """

## Usage Examples

### In Python Code:

```python
from gemini_text_client import GeminiClient, ModelInput
from generation_presets import DETERMINISTIC, CREATIVE, get_preset

# Method 1: Using preset constants directly
model_input = ModelInput(
    user_prompt="What is the capital of France?",
    **DETERMINISTIC
)

# Method 2: Using get_preset function
creative_params = get_preset('creative')
model_input = ModelInput(
    user_prompt="Write a short story about AI",
    **creative_params
)

# Method 3: Custom parameters
model_input = ModelInput(
    user_prompt="Generate code for...",
    temperature=0.1,
    top_p=0.5,
    top_k=10
)

client = GeminiClient()
response = client.generate_text(model_input)
```

### From Command Line:

```bash
# Deterministic output
python gemini_text_client.py -q "What is 2+2?" -t 0.0 --top-p 0.1 --top-k 1

# Creative output
python gemini_text_client.py -q "Write a poem" -t 1.5 --top-p 0.95 --top-k 64

# Balanced (default - no need to specify)
python gemini_text_client.py -q "Explain quantum computing"

# Code generation
python gemini_text_client.py -q "Write a Python function to sort a list" -t 0.2 --top-p 0.8
```

## Parameter Guide

### Temperature (0.0 - 2.0)
- **0.0**: Most deterministic, always picks the most likely token
- **0.0-0.3**: Very focused, minimal randomness (good for factual/code)
- **0.3-0.7**: Slightly creative but still focused
- **0.7-1.0**: Balanced creativity and coherence
- **1.0-1.5**: More creative and diverse
- **1.5-2.0**: Maximum creativity (may be less coherent)

### Top-P / Nucleus Sampling (0.0 - 1.0)
- **0.1-0.3**: Very focused, considers only most likely tokens
- **0.5-0.8**: Moderate diversity
- **0.8-0.95**: Good balance
- **0.95-1.0**: Maximum diversity

### Top-K
- **1-10**: Extremely focused
- **20-40**: Moderate focus (good for most tasks)
- **40-64**: More diverse
- **None/0**: No restriction (use with caution)

## Recommended Combinations by Use Case

| Use Case | Temperature | Top-P | Top-K | Preset |
|----------|-------------|-------|-------|--------|
| Code Generation | 0.2 | 0.8 | 40 | CODE_GENERATION |
| Math/Logic | 0.0 | 0.1 | 1 | DETERMINISTIC |
| Data Extraction | 0.0 | 0.1 | 1 | DATA_EXTRACTION |
| Question Answering | 0.2 | 0.8 | 40 | QA |
| Summarization | 0.4 | 0.9 | 40 | SUMMARY |
| Creative Writing | 1.4 | 0.95 | 50 | STORYTELLING |
| Brainstorming | 1.5 | 0.95 | 64 | CREATIVE |
| General Chat | 1.0 | 0.95 | 40 | BALANCED |
"""
