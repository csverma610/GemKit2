# Gemini Object Identifier

An object identification tool that uses Google's Gemini API to analyze images through visual feature matching.

## Overview

The system performs object identification in three steps:
1. Generates 5 visual features for the target object
2. Analyzes the image for feature presence
3. Returns results with confidence scores for each feature

## Features

- Feature-based object identification (5 features per object)
- Confidence scoring (0.0-1.0 per feature)
- JSON response with Pydantic validation
- Automatic retry with exponential backoff for network failures
- Support for local file paths and URLs
- Error handling for validation, network, and API errors
- Logging for debugging and monitoring

## Requirements

### Dependencies

```bash
pip install google-genai pydantic tenacity pillow
```

### Core Packages
- `google-genai` - Google Generative AI Python SDK
- `pydantic` - Data validation
- `tenacity` - Retry logic
- `pillow` - Image processing

### Environment Variables

```bash
export GEMINI_API_KEY="your-api-key-here"
```

Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your API key:
   ```bash
   export GEMINI_API_KEY="your-key"
   ```

## Usage

### Command-Line Interface

#### Basic Usage
```bash
python gemini_object_identifier.py -i path/to/image.jpg -o "cat"
```

#### With Custom Model
```bash
python gemini_object_identifier.py \
  -i https://example.com/image.jpg \
  -o "laptop" \
  -m "gemini-2.5-flash"
```

#### Arguments
- `-i, --image_path` (required): Path or URL to the image file
- `-o, --object_name` (required): Name of the object to identify
- `-m, --model_name` (optional): Gemini model name (default: `gemini-2.5-flash`)

### Python API

```python
from gemini_object_identifier import ObjectIdentifier

# Initialize identifier
identifier = ObjectIdentifier(model_name="gemini-2.5-flash")

# Identify object in image
result = identifier.identify_object(
    image_path="path/to/image.jpg",
    object_name="dog"
)

# Process results
for feature in result.features:
    print(f"{feature.description}: {feature.matching} ({feature.confidence:.2f})")
```

### Example Output

```
--- Object Identification Report ---
Object to Identify: cat
Generated & Matched Features: Pointed ears, Whiskers, Vertical pupils, Fur coat, Tail

--- Feature Analysis ---
  - Feature: 'Pointed ears' -> MATCHED (confidence: 0.95)
  - Feature: 'Whiskers' -> MATCHED (confidence: 0.88)
  - Feature: 'Vertical pupils' -> MATCHED (confidence: 0.72)
  - Feature: 'Fur coat' -> MATCHED (confidence: 0.91)
  - Feature: 'Tail' -> MATCHED (confidence: 0.85)
------------------------

Overall Confidence Score: 86.20%
------------------------------------
```

## Architecture

### Class Structure

#### `ObjectIdentifier`
Main class for object identification.

**Methods:**
- `__init__(model_name: str)` - Initialize with Gemini model
- `identify_object(image_path: str, object_name: str) -> FeaturesResult` - Perform identification
- `_create_client() -> genai.Client` - Initialize API client
- `_generate_prompt(object_name: str) -> str` - Generate identification prompt
- `_generate_payload(prompt: str, image_path: str) -> list` - Load image and create payload
- `_call_api(payload: list) -> FeaturesResult` - Execute API call

#### Data Models

**`MatchFeature`**
```python
class MatchFeature(BaseModel):
    description: str      # Feature description
    matching: bool        # Whether feature is present
    confidence: float     # Confidence score (0.0-1.0)
```

**`FeaturesResult`**
```python
class FeaturesResult(BaseModel):
    features: List[MatchFeature]
```

## Retry Logic

The system retries operations on transient failures.

### Image Loading (`_generate_payload`)
- Max attempts: 3
- Backoff: Exponential (2s, 4s, 8s, max 10s)
- Retry conditions: `OSError`, `IOError`, `ConnectionError`

### API Calls (`_call_api`)
- Max attempts: 3
- Backoff: Exponential (2s, 4s, 8s, max 10s)
- Retry conditions: `ConnectionError`, `TimeoutError`

## Error Handling

### Input Validation
- Empty object names raise `ValueError`
- Empty image paths raise `ValueError`
- Object names are truncated to 100 characters

### Runtime Errors
- Image loading failures raise `ValueError`
- API errors are logged and return empty `FeaturesResult`
- File system errors raise `OSError`
- Network errors trigger automatic retry

### Graceful Degradation
On unrecoverable errors, returns `FeaturesResult(features=[])`.

## Configuration

### Supported Models
- `gemini-2.5-flash` (default)
- `gemini-2.5-pro`
- `gemini-1.5-pro`
- `gemini-1.5-flash`

### Logging Configuration
```python
logging.basicConfig(level=logging.DEBUG)  # Verbose output
logging.basicConfig(level=logging.WARNING)  # Minimal output
```

## Limitations

1. Fixed feature count of 5
2. Object names truncated to 100 characters
3. Image format support depends on ImageSource utility
4. Subject to Gemini API quotas and rate limits
5. Overall confidence calculation divides by total features, not matched features

## Performance Considerations

- Typical API response time: 2-5 seconds
- Large images increase processing time
- Requires stable internet connection
- Failed requests add 2-20 seconds overhead (retries)

## Troubleshooting

### Common Issues

**"GEMINI_API_KEY environment variable not set"**
- Set the environment variable before running

**"Failed to load image from {path}"**
- Check file path exists
- Verify URL is accessible
- Check image format is supported

**"Unexpected error during identification"**
- Check API key validity
- Verify network connectivity
- Review logs for details
- Check API quota

## Best Practices

1. Validate image paths before calling
2. Wrap calls in try-except blocks for production
3. Monitor logs for retry patterns
4. Use specific object names
5. Choose model based on speed vs accuracy needs

## Examples

### Batch Processing
```python
identifier = ObjectIdentifier()
images = ["cat1.jpg", "cat2.jpg", "cat3.jpg"]

for img_path in images:
    result = identifier.identify_object(img_path, "cat")
    confidence = sum(f.confidence for f in result.features if f.matching)
    print(f"{img_path}: {confidence:.2f}")
```

### Custom Confidence Threshold
```python
result = identifier.identify_object("image.jpg", "car")
high_confidence_features = [
    f for f in result.features
    if f.matching and f.confidence > 0.8
]
print(f"Found {len(high_confidence_features)} high-confidence matches")
```

### URL-Based Processing
```python
url = "https://example.com/photo.jpg"
result = identifier.identify_object(url, "building")
match_rate = sum(1 for f in result.features if f.matching) / len(result.features)
print(f"Match rate: {match_rate:.0%}")
```

## Technical Specifications

- API version: Google Generative AI SDK (latest)
- Response format: JSON with Pydantic schema validation
- Image formats: All PIL-supported formats (JPEG, PNG, GIF, BMP, etc.)
- Max object name length: 100 characters
- Feature count: Fixed at 5 per analysis

## License

This code is provided as-is for educational and development purposes.

## Support

For issues:
- Gemini API: See [Google AI Documentation](https://ai.google.dev/)
- Code issues: Review logs and error messages

## Version History

- v1.0: Initial release
- v1.1: Added retry logic
- v1.2: Refactored for modularity

---

This tool requires a valid Google Gemini API key and internet connection. Results depend on model capabilities and image quality.
