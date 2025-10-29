# Gemini API: Vision & Unified Generation Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Vision Capabilities](#vision-capabilities)
3. [Unified Generation Method](#unified-generation-method)
4. [Examples](#examples)
5. [CLI Usage](#cli-usage)
6. [Advanced Features](#advanced-features)
7. [Best Practices](#best-practices)

---

## Quick Start

### Text Generation
```python
from gemini_text_client import GeminiClient, ModelInput

client = GeminiClient()

model_input = ModelInput(
    user_prompt="Explain quantum computing"
)
response = client.generate(model_input)
print(response)
```

### Vision (Single Image)
```python
model_input = ModelInput(
    user_prompt="What's in this image?",
    images=["photo.jpg"]
)
response = client.generate(model_input)
```

### Structured Output
```python
schema = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {"type": "string"}
    }
}

model_input = ModelInput(
    user_prompt="Analyze this text...",
    response_schema=schema
)
response = client.generate(model_input)  # Returns dict
```

---

## Vision Capabilities

### Supported Image Formats
- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- GIF (`.gif`)
- WebP (`.webp`)
- BMP (`.bmp`)

### Image Input Methods

#### 1. Local File Path
```python
model_input = ModelInput(
    user_prompt="Describe this image",
    images=["path/to/image.jpg"]
)
```

#### 2. Multiple Images
```python
model_input = ModelInput(
    user_prompt="Compare these images",
    images=["image1.jpg", "image2.jpg", "image3.jpg"]
)
```

#### 3. Image URL
```python
model_input = ModelInput(
    user_prompt="What's in this image?",
    images=["https://example.com/photo.jpg"]
)
```

#### 4. Raw Image Bytes
```python
with open("image.jpg", "rb") as f:
    image_bytes = f.read()

model_input = ModelInput(
    user_prompt="Analyze this",
    image_data=[image_bytes]
)
```

### Vision Use Cases

#### 1. Image Description
```python
model_input = ModelInput(
    user_prompt="Describe what you see in detail.",
    images=["scene.jpg"]
)
```

#### 2. OCR (Text Extraction)
```python
from generation_presets import DETERMINISTIC

model_input = ModelInput(
    user_prompt="Extract all text from this image.",
    images=["document.jpg"],
    **DETERMINISTIC  # Use for accuracy
)
```

#### 3. Object Detection
```python
model_input = ModelInput(
    user_prompt="List all objects visible in this image.",
    images=["photo.jpg"]
)
```

#### 4. Image Comparison
```python
model_input = ModelInput(
    user_prompt="What are the differences between these images?",
    images=["before.jpg", "after.jpg"]
)
```

#### 5. Visual Question Answering
```python
model_input = ModelInput(
    user_prompt="How many people are in this photo?",
    images=["group_photo.jpg"],
    **DETERMINISTIC
)
```

---

## Unified Generation Method

The **`generate()`** method is the recommended approach. It automatically routes to the appropriate generation method based on your input.

### Method Selection Logic
```python
def generate(self, model_input: ModelInput):
    if model_input.response_schema:
        return self.generate_structured(model_input)  # Returns dict/list
    else:
        return self.generate_text(model_input)  # Returns str (handles vision too)
```

### Why Use `generate()`?

✅ **Automatic routing** - No need to choose the method
✅ **Clean code** - One method for all use cases
✅ **Flexible** - Works with text, vision, and structured output
✅ **Future-proof** - New features work automatically

### Legacy Methods (Still Available)

```python
# Specific methods if you need them
client.generate_text(model_input)        # Returns str
client.generate_structured(model_input)  # Returns dict/list
```

---

## Examples

### Example 1: Simple Vision Task
```python
from gemini_text_client import GeminiClient, ModelInput

client = GeminiClient()

model_input = ModelInput(
    user_prompt="What objects are in this kitchen?",
    images=["kitchen.jpg"]
)

response = client.generate(model_input)
print(response)
```

**CLI:**
```bash
python gemini_text_client.py -q "What objects are in this kitchen?" -i kitchen.jpg
```

---

### Example 2: Vision + Deterministic (OCR)
```python
from generation_presets import DETERMINISTIC

model_input = ModelInput(
    user_prompt="Extract the invoice number and total amount.",
    images=["invoice.jpg"],
    **DETERMINISTIC  # temperature=0.0 for accuracy
)

response = client.generate(model_input)
```

**CLI:**
```bash
python gemini_text_client.py \
    -q "Extract invoice number and total" \
    -i invoice.jpg \
    -t 0.0 --top-p 0.1 --top-k 1
```

---

### Example 3: Vision + Structured Output
```python
schema = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {"type": "string"}
        },
        "scene_type": {
            "type": "string",
            "enum": ["indoor", "outdoor", "abstract"]
        },
        "dominant_colors": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["objects", "scene_type", "dominant_colors"]
}

model_input = ModelInput(
    user_prompt="Analyze this image and extract structured data.",
    images=["photo.jpg"],
    response_schema=schema
)

result = client.generate(model_input)
# result is a dict: {"objects": [...], "scene_type": "...", ...}
```

---

### Example 4: Multiple Images Comparison
```python
model_input = ModelInput(
    user_prompt="Compare these product photos. Which one has better lighting?",
    images=["product1.jpg", "product2.jpg", "product3.jpg"]
)

response = client.generate(model_input)
```

**CLI:**
```bash
python gemini_text_client.py \
    -q "Which photo has better lighting?" \
    -i product1.jpg product2.jpg product3.jpg
```

---

### Example 5: Creative Image Captioning
```python
from generation_presets import CREATIVE

model_input = ModelInput(
    user_prompt="Write a poetic caption for this landscape.",
    images=["landscape.jpg"],
    **CREATIVE  # temperature=1.5 for creativity
)

response = client.generate(model_input)
```

**CLI:**
```bash
python gemini_text_client.py \
    -q "Write a poetic caption" \
    -i landscape.jpg \
    -t 1.5 --top-p 0.95
```

---

### Example 6: Vision + Text + Structured (All Features)
```python
schema = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "mood": {"type": "string"},
        "suggested_story": {"type": "string"}
    }
}

model_input = ModelInput(
    user_prompt="Create a story concept based on this image.",
    sys_prompt="You are a creative writer specializing in visual storytelling.",
    images=["artwork.jpg"],
    response_schema=schema,
    temperature=1.3,  # Creative but not random
    max_output_tokens=500
)

result = client.generate(model_input)
# Returns structured dict with story concept
```

---

## CLI Usage

### Basic Vision
```bash
python gemini_text_client.py -q "What's in this image?" -i photo.jpg
```

### Multiple Images
```bash
python gemini_text_client.py \
    -q "Compare these images" \
    -i img1.jpg img2.jpg img3.jpg
```

### Vision + Generation Parameters
```bash
# Deterministic OCR
python gemini_text_client.py \
    -q "Extract all text" \
    -i document.jpg \
    -t 0.0 --top-p 0.1 --top-k 1

# Creative caption
python gemini_text_client.py \
    -q "Write a creative caption" \
    -i photo.jpg \
    -t 1.5 --top-p 0.95 --max-tokens 200
```

### Vision + System Prompt
```bash
python gemini_text_client.py \
    -q "Analyze this medical image" \
    -i xray.jpg \
    -s "You are an expert radiologist" \
    -t 0.2
```

### Vision + Structured Output (JSON file)
```bash
# Create schema file
cat > schema.json << 'EOF'
{
  "type": "object",
  "properties": {
    "objects": {"type": "array", "items": {"type": "string"}},
    "count": {"type": "integer"}
  }
}
EOF

python gemini_text_client.py \
    -q "Count objects in this image" \
    -i photo.jpg \
    -r schema.json
```

---

## Advanced Features

### 1. Vision + Custom Parameters
```python
model_input = ModelInput(
    user_prompt="Analyze this medical image",
    sys_prompt="You are an expert medical imaging analyst",
    images=["scan.jpg"],
    temperature=0.2,  # Precise but not too rigid
    top_p=0.8,
    max_output_tokens=1000
)
```

### 2. Batch Processing Images
```python
import os

client = GeminiClient()
image_dir = "photos/"

for image_file in os.listdir(image_dir):
    if image_file.endswith(('.jpg', '.png')):
        model_input = ModelInput(
            user_prompt="Describe this image briefly",
            images=[os.path.join(image_dir, image_file)]
        )
        response = client.generate(model_input)
        print(f"{image_file}: {response}\n")
```

### 3. Vision with Retry Logic
```python
# Built-in retry with exponential backoff
client = GeminiClient(
    max_retries=5,
    initial_delay=2.0
)

model_input = ModelInput(
    user_prompt="Analyze this complex scene",
    images=["complex_scene.jpg"]
)

response = client.generate(model_input)  # Auto-retries on network errors
```

### 4. Rate-Limited Vision Processing
```python
# Process many images without hitting rate limits
client = GeminiClient(
    rate_limit_calls=30,   # 30 calls
    rate_limit_period=60.0  # per 60 seconds
)

for image in image_list:
    model_input = ModelInput(
        user_prompt="Describe this",
        images=[image]
    )
    response = client.generate(model_input)  # Auto rate-limits
```

---

## Best Practices

### For Vision Tasks

#### ✅ DO:
- **Use deterministic settings for OCR/data extraction**
  ```python
  **DETERMINISTIC  # temperature=0.0
  ```
- **Use specific, clear prompts**
  ```python
  "Extract the text from this document"  # ✅ Good
  # vs
  "What's here?"  # ❌ Too vague
  ```
- **Specify what you want to extract**
  ```python
  "List all product names and prices visible in this receipt"
  ```
- **Use structured output for data extraction**
  ```python
  response_schema={...}  # Get JSON instead of text
  ```

#### ❌ DON'T:
- Use creative settings (high temperature) for accuracy-critical tasks
- Send extremely large images without consideration
- Mix too many unrelated images in one request
- Expect perfect OCR on low-quality/blurry images

### For Unified `generate()` Method

#### ✅ DO:
```python
# Always use generate() for new code
response = client.generate(model_input)
```

#### ❌ DON'T:
```python
# Avoid choosing methods manually unless necessary
response = client.generate_text(model_input)  # Unnecessary
```

### Parameter Combinations

| Use Case | Temperature | Top-P | Top-K | Example |
|----------|-------------|-------|-------|---------|
| **OCR** | 0.0 | 0.1 | 1 | Extract invoice data |
| **Object Detection** | 0.2 | 0.8 | 40 | List all items in image |
| **Image Description** | 0.7 | 0.9 | 40 | General description |
| **Creative Caption** | 1.5 | 0.95 | 64 | Poetic image caption |
| **Visual Q&A** | 0.3 | 0.8 | 40 | Answer specific questions |

### Image Quality Tips

1. **For OCR:** Use high-resolution, well-lit images
2. **For detection:** Ensure objects are clearly visible
3. **For comparison:** Use similar lighting/angles
4. **File size:** Keep under 20MB for best performance

---

## Troubleshooting

### Issue: FileNotFoundError
```python
FileNotFoundError: Image file not found: photo.jpg
```
**Solution:** Check file path is correct and file exists

### Issue: Unsupported image format
```python
ValueError: Unsupported image format: .tiff
```
**Solution:** Convert to JPEG/PNG/GIF/WebP/BMP

### Issue: Poor OCR results
**Solutions:**
- Use `**DETERMINISTIC` preset
- Ensure high-quality, well-lit image
- Increase image resolution
- Specify exact text format in prompt

### Issue: Vague image descriptions
**Solutions:**
- Make prompt more specific
- Ask for particular details
- Use structured output to force completeness

---

## Complete Reference

### ModelInput Fields for Vision
```python
@dataclass
class ModelInput:
    user_prompt: str                                    # Required
    model_name: str = 'gemini-2.5-flash'
    sys_prompt: str = ""
    assist_prompt: str = ""
    response_schema: Optional[Any] = None
    temperature: Optional[float] = None                 # 0.0-2.0
    top_p: Optional[float] = None                       # 0.0-1.0
    top_k: Optional[int] = None                         # Integer
    max_output_tokens: Optional[int] = None
    images: Optional[List[Union[str, Path]]] = None     # NEW: Image paths/URLs
    image_data: Optional[List[bytes]] = None            # NEW: Raw bytes
```

### Supported Image Formats
```python
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']
```

### CLI Arguments Reference
```bash
-q, --question          User prompt (required)
-m, --model             Model name (gemini-2.5-flash, etc.)
-s, --system-prompt     System instruction
-a, --assistant-prompt  Assistant context
-r, --response-schema   JSON schema for structured output
-t, --temperature       0.0-2.0 (creativity control)
--top-p                 0.0-1.0 (diversity)
--top-k                 Top-K sampling
--max-tokens            Max output length
-i, --images            Image paths or URLs (NEW)
--save-lmdb             Save to database
--lmdb-path             Database path
```

---

## Quick Examples Summary

```bash
# Text only
python gemini_text_client.py -q "Hello, world!"

# Single image
python gemini_text_client.py -q "What's this?" -i photo.jpg

# Multiple images
python gemini_text_client.py -q "Compare" -i img1.jpg img2.jpg

# Vision + deterministic
python gemini_text_client.py -q "Extract text" -i doc.jpg -t 0.0

# Vision + creative
python gemini_text_client.py -q "Write caption" -i pic.jpg -t 1.5

# Vision + structured
python gemini_text_client.py -q "Analyze" -i photo.jpg -r schema.json

# Vision + all params
python gemini_text_client.py \
    -q "Detailed analysis" \
    -i image.jpg \
    -s "You are an expert" \
    -t 0.3 --top-p 0.8 --max-tokens 500
```

---

## Resources

- **Example Scripts:**
  - `example_vision.py` - 11 comprehensive vision examples
  - `example_deterministic_creative.py` - Parameter demonstrations

- **Main Client:** `gemini_text_client.py`
- **Presets:** `generation_presets.py`
- **Documentation:**
  - `README_GENERATION_PARAMS.md` - Parameter guide
  - `README_VISION_UNIFIED.md` - This file

---

**Happy coding with Gemini Vision! 🖼️✨**
