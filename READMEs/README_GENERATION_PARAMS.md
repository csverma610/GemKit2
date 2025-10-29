# Gemini API: Deterministic vs Creative Output Guide

## Quick Start

### Python Code

```python
from gemini_text_client import GeminiClient, ModelInput
from generation_presets import DETERMINISTIC, CREATIVE

client = GeminiClient()

# Deterministic output (consistent, factual)
model_input = ModelInput(
    user_prompt="What is the capital of France?",
    **DETERMINISTIC
)
response = client.generate_text(model_input)

# Creative output (varied, imaginative)
model_input = ModelInput(
    user_prompt="Write a short poem about the ocean",
    **CREATIVE
)
response = client.generate_text(model_input)
```

### Command Line

```bash
# Deterministic
python gemini_text_client.py -q "What is 2+2?" -t 0.0 --top-p 0.1 --top-k 1

# Creative
python gemini_text_client.py -q "Write a story" -t 1.5 --top-p 0.95 --top-k 64

# Balanced (default)
python gemini_text_client.py -q "Explain quantum computing"
```

## Generation Parameters Explained

### Temperature (`-t`, `--temperature`)

Controls randomness in token selection. **Most important parameter.**

| Value | Behavior | Use Case |
|-------|----------|----------|
| **0.0** | Deterministic, always picks most likely token | Math, code, data extraction |
| **0.0-0.3** | Very focused, minimal creativity | Technical writing, Q&A |
| **0.3-0.7** | Moderate creativity | Explanations, tutorials |
| **0.7-1.0** | Balanced | General chat, summaries |
| **1.0-1.5** | Creative | Brainstorming, creative writing |
| **1.5-2.0** | Very creative | Poetry, fiction (may be incoherent) |

**Example:**
```bash
# Deterministic math
python gemini_text_client.py -q "Calculate 15 * 23" -t 0.0

# Creative writing
python gemini_text_client.py -q "Write a haiku" -t 1.5
```

### Top-P / Nucleus Sampling (`--top-p`)

Controls diversity by considering only tokens with cumulative probability ≥ top_p.

| Value | Behavior | Use Case |
|-------|----------|----------|
| **0.1-0.3** | Very focused, considers only most likely tokens | Critical accuracy tasks |
| **0.5-0.8** | Moderate diversity | Most practical tasks |
| **0.8-0.95** | Good balance of diversity and coherence | Default recommended |
| **0.95-1.0** | Maximum diversity | Creative tasks |

**Example:**
```bash
python gemini_text_client.py -q "Summarize this text" --top-p 0.9
```

### Top-K (`--top-k`)

Limits token selection to the K most likely tokens.

| Value | Behavior | Use Case |
|-------|----------|----------|
| **1** | Only the most likely token | Deterministic output |
| **10-20** | Very focused | Code, data extraction |
| **20-40** | Moderate focus | General use |
| **40-64** | More diverse | Creative tasks |
| **None** | No restriction | Maximum creativity |

**Example:**
```bash
python gemini_text_client.py -q "Generate code" --top-k 20
```

### Max Output Tokens (`--max-tokens`)

Limits the response length.

```bash
python gemini_text_client.py -q "Brief summary" --max-tokens 100
```

## Preset Configurations

Use the `generation_presets.py` module for common scenarios:

### Available Presets

| Preset | Temperature | Top-P | Top-K | Best For |
|--------|-------------|-------|-------|----------|
| `DETERMINISTIC` | 0.0 | 0.1 | 1 | Math, logic, data extraction |
| `PRECISE` | 0.3 | 0.5 | 20 | Code generation, technical writing |
| `BALANCED` | 1.0 | 0.95 | 40 | General chat, Q&A |
| `CREATIVE` | 1.5 | 0.95 | 64 | Brainstorming, creative writing |
| `VERY_CREATIVE` | 2.0 | 1.0 | None | Fiction, poetry |
| `CODE_GENERATION` | 0.2 | 0.8 | 40 | Programming tasks |
| `STORYTELLING` | 1.4 | 0.95 | 50 | Creative writing |
| `DATA_EXTRACTION` | 0.0 | 0.1 | 1 | Extracting structured data |
| `SUMMARY` | 0.4 | 0.9 | 40 | Summarization |
| `QA` | 0.2 | 0.8 | 40 | Question answering |

### Using Presets in Python

```python
from generation_presets import get_preset, DETERMINISTIC, CODE_GENERATION

# Method 1: Direct import
model_input = ModelInput(
    user_prompt="Your question",
    **DETERMINISTIC
)

# Method 2: Using get_preset()
params = get_preset('code_generation')
model_input = ModelInput(
    user_prompt="Write a function",
    **params
)
```

## Use Case Examples

### 1. Code Generation (Deterministic)

```bash
python gemini_text_client.py \
    -q "Write a Python function to check if a number is prime" \
    -t 0.2 --top-p 0.8 --top-k 40
```

Or in Python:
```python
from generation_presets import CODE_GENERATION

model_input = ModelInput(
    user_prompt="Write a function to check prime numbers",
    **CODE_GENERATION
)
```

### 2. Creative Writing (Creative)

```bash
python gemini_text_client.py \
    -q "Write a short story about time travel" \
    -t 1.5 --top-p 0.95 --top-k 64
```

Or in Python:
```python
from generation_presets import STORYTELLING

model_input = ModelInput(
    user_prompt="Write a story about time travel",
    **STORYTELLING
)
```

### 3. Data Extraction (Very Deterministic)

```bash
python gemini_text_client.py \
    -q "Extract email from: Contact john@example.com for info" \
    -t 0.0 --top-p 0.1 --top-k 1
```

Or in Python:
```python
from generation_presets import DATA_EXTRACTION

model_input = ModelInput(
    user_prompt="Extract email from: Contact john@example.com",
    **DATA_EXTRACTION
)
```

### 4. Question Answering (Precise)

```bash
python gemini_text_client.py \
    -q "What is photosynthesis?" \
    -t 0.2 --top-p 0.8 --top-k 40
```

Or in Python:
```python
from generation_presets import QA

model_input = ModelInput(
    user_prompt="What is photosynthesis?",
    **QA
)
```

## Testing Determinism

Run the same prompt multiple times to verify determinism:

```bash
# Should give identical results each time
for i in {1..3}; do
    echo "Run $i:"
    python gemini_text_client.py -q "What is 2+2?" -t 0.0 --top-p 0.1 --top-k 1
    echo ""
done
```

Or use the example script:
```bash
python example_deterministic_creative.py
```

## Best Practices

### For Deterministic Output:
- **Temperature:** 0.0 - 0.3
- **Top-P:** 0.1 - 0.5
- **Top-K:** 1 - 20
- **Use Cases:** Math, code, data extraction, factual Q&A

### For Creative Output:
- **Temperature:** 1.0 - 2.0
- **Top-P:** 0.9 - 1.0
- **Top-K:** 40 - 64 (or None)
- **Use Cases:** Writing, brainstorming, art, poetry

### For Balanced Output:
- **Temperature:** 0.7 - 1.0
- **Top-P:** 0.8 - 0.95
- **Top-K:** 40
- **Use Cases:** General chat, summaries, explanations

## Common Mistakes

❌ **Don't:**
- Use high temperature for code generation
- Use temperature=0.0 for creative writing
- Mix extreme settings (e.g., temp=2.0 with top_k=1)
- Ignore max_output_tokens for long responses

✅ **Do:**
- Match parameters to your use case
- Start with presets and adjust as needed
- Test with multiple runs to verify consistency
- Use temperature as primary control, other params as fine-tuning

## Advanced: Custom Combinations

```python
# Ultra-deterministic for critical calculations
model_input = ModelInput(
    user_prompt="Calculate precise value",
    temperature=0.0,
    top_p=0.01,
    top_k=1,
    max_output_tokens=100
)

# Controlled creativity for marketing copy
model_input = ModelInput(
    user_prompt="Write marketing copy",
    temperature=1.2,
    top_p=0.9,
    top_k=50,
    max_output_tokens=200
)
```

## Troubleshooting

**Problem:** Output is too random/incoherent
- **Solution:** Lower temperature (try 0.7), reduce top_p to 0.8

**Problem:** Output is too repetitive/boring
- **Solution:** Increase temperature (try 1.0-1.2), increase top_p to 0.95

**Problem:** Code generation has syntax errors
- **Solution:** Use temperature=0.2, top_p=0.8, top_k=40 (CODE_GENERATION preset)

**Problem:** Inconsistent results when you need consistency
- **Solution:** Set temperature=0.0, top_p=0.1, top_k=1 (DETERMINISTIC preset)

## Resources

- **Test script:** `python example_deterministic_creative.py`
- **Presets module:** `generation_presets.py`
- **Main client:** `gemini_text_client.py`

## Quick Reference Table

| Task | Preset to Use | CLI Example |
|------|---------------|-------------|
| Math calculation | DETERMINISTIC | `-t 0.0 --top-p 0.1 --top-k 1` |
| Code writing | CODE_GENERATION | `-t 0.2 --top-p 0.8 --top-k 40` |
| Data extraction | DATA_EXTRACTION | `-t 0.0 --top-p 0.1 --top-k 1` |
| Q&A | QA | `-t 0.2 --top-p 0.8 --top-k 40` |
| Summary | SUMMARY | `-t 0.4 --top-p 0.9 --top-k 40` |
| General chat | BALANCED | `-t 1.0 --top-p 0.95 --top-k 40` |
| Creative writing | STORYTELLING | `-t 1.4 --top-p 0.95 --top-k 50` |
| Brainstorming | CREATIVE | `-t 1.5 --top-p 0.95 --top-k 64` |
| Poetry | VERY_CREATIVE | `-t 2.0 --top-p 1.0` |
