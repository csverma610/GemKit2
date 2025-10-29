# Gemini Vision - User Guide

A powerful Python tool for image analysis using Google's Gemini AI. Process images from multiple sources including files, URLs, cameras, screenshots, and even video frames!

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Features

- **Multiple Image Sources**: Process images from various sources:
  - Local image files (JPG, PNG, BMP, GIF, WebP, TIFF)
  - URLs (automatically downloads images)
  - Webcam/Camera (real-time capture)
  - Screenshots (full screen or region)
  - Video files (extracts frames)
  - Base64 encoded images
  - PIL Image objects
  - Numpy arrays

- **Powered by Gemini AI**: Uses Google's advanced Gemini 2.5 Flash model for intelligent image analysis

- **Flexible Interface**: Use as a command-line tool or import as a Python library

- **Simple Configuration**: Easy setup with environment variables

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Install Required Packages

```bash
pip install pillow numpy google-genai
```

### Optional Dependencies

For additional features, install these packages:

```bash
# For camera/webcam support and video processing
pip install opencv-python

# For screenshot capability
pip install pyautogui

# For URL image downloads
pip install requests
```

### Step 2: Set Up API Key

Get your Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey).

Set it as an environment variable:

**Linux/macOS:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
```

**Permanent Setup (recommended):**

Add to your `.bashrc`, `.zshrc`, or `.bash_profile`:
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Quick Start

### Command Line Usage

```bash
# Analyze a local image
python gemini_vision.py -i photo.jpg

# Custom prompt
python gemini_vision.py -i photo.jpg -p "What colors are in this image?"
```

### Python Library Usage

```python
from gemini_vision import GeminiVision

# Initialize
vlm = GeminiVision()

# Analyze an image
result = vlm.generate_text("photo.jpg", "Describe this image")
print(result)
```

## Usage Examples

### 1. Local Image Files

```bash
# Basic image description
python gemini_vision.py -i vacation_photo.jpg

# Specific question about the image
python gemini_vision.py -i receipt.png -p "Extract all text from this receipt"

# Detailed analysis
python gemini_vision.py -i artwork.jpg -p "Analyze the artistic style and composition"
```

### 2. Images from URLs

```bash
# Analyze an image from the web
python gemini_vision.py -i https://example.com/image.jpg -p "What is in this image?"

# Check product details
python gemini_vision.py -i https://shop.com/product.png -p "Describe this product"
```

### 3. Webcam/Camera

```bash
# Use default camera (index 0)
python gemini_vision.py -i 0 -p "What do you see?"

# Use secondary camera (index 1)
python gemini_vision.py -i 1 -p "Describe what the camera sees"
```

### 4. Screenshots

```bash
# Capture and analyze current screen
python gemini_vision.py -i screenshot -p "What is on the screen?"

# Find errors in a screenshot
python gemini_vision.py -i screenshot -p "Are there any error messages visible?"
```

### 5. Video Frames

```bash
# Analyze first frame of a video
python gemini_vision.py -i meeting_recording.mp4 -p "How many people are in this scene?"

# Describe video content
python gemini_vision.py -i tutorial.mov -p "What is being demonstrated?"
```

### 6. Python Library Examples

#### Basic Usage

```python
from gemini_vision import GeminiVision

# Create instance
vision = GeminiVision()

# Analyze local file
description = vision.generate_text("photo.jpg", "Describe this image in detail")
print(description)
```

#### Different Sources

```python
from gemini_vision import GeminiVision

vision = GeminiVision()

# From URL
url_result = vision.generate_text(
    "https://example.com/image.jpg",
    "What objects are visible?"
)

# From camera
camera_result = vision.generate_text(
    0,  # Camera index
    "Describe the scene"
)

# From screenshot
screenshot_result = vision.generate_text(
    "screenshot",
    "Summarize what's on screen"
)

# From video
video_result = vision.generate_text(
    "presentation.mp4",
    "What is the main topic?"
)
```

#### Multiple Analyses

```python
from gemini_vision import GeminiVision

vision = GeminiVision()

# Analyze multiple images
images = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
prompts = [
    "Is there a person in this image?",
    "What is the dominant color?",
    "Describe the scene"
]

for img, prompt in zip(images, prompts):
    result = vision.generate_text(img, prompt)
    print(f"{img}: {result}\n")
```

#### Custom Model

```python
from gemini_vision import GeminiVision

# Use a specific Gemini model
vision = GeminiVision(model_name="gemini-2.5-flash")

result = vision.generate_text("image.jpg", "Analyze this image")
```

## Configuration

### Command Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--image` | `-i` | Yes | - | Image source: file path, URL, "screenshot", or camera index |
| `--prompt` | `-p` | No | "Describe the image" | Text prompt for image analysis |
| `--model` | - | No | "gemini-2.5-flash" | Gemini model to use |

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Your Google Gemini API key |

### Getting Help

```bash
# Show help message with examples
python gemini_vision.py -h
```

## API Reference

### GeminiVision Class

#### Constructor

```python
GeminiVision(model_name="gemini-2.5-flash")
```

**Parameters:**
- `model_name` (str, optional): The Gemini model to use. Default: "gemini-2.5-flash"

**Example:**
```python
vision = GeminiVision()
# or
vision = GeminiVision(model_name="gemini-2.5-flash")
```

#### generate_text()

```python
generate_text(input_source, prompt) -> str
```

Analyzes an image and returns the AI-generated text response.

**Parameters:**
- `input_source` (str or int): The image source
  - File path: `"photo.jpg"`
  - URL: `"https://example.com/image.jpg"`
  - Camera: `0` (integer index)
  - Screenshot: `"screenshot"`
  - Video: `"video.mp4"`
- `prompt` (str): The question or instruction for the AI

**Returns:**
- `str`: The AI-generated response text

**Example:**
```python
result = vision.generate_text("photo.jpg", "What is in this image?")
```

## Troubleshooting

### Common Issues

#### 1. API Key Not Found

**Error:** `ValueError: API key must be set in the GEMINI_API_KEY environment variable.`

**Solution:**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

#### 2. Image File Not Found

**Error:** `Error: Input file not found`

**Solution:**
- Check the file path is correct
- Use absolute paths: `/Users/username/images/photo.jpg`
- Verify the file exists: `ls -l photo.jpg`

#### 3. Camera Not Working

**Error:** `Cannot open camera 0`

**Solutions:**
- Install opencv-python: `pip install opencv-python`
- Try different camera index: `-i 1` or `-i 2`
- Check camera permissions in system settings
- Make sure no other app is using the camera

#### 4. Screenshot Not Working

**Error:** `pyautogui required`

**Solution:**
```bash
pip install pyautogui
```

#### 5. URL Download Failed

**Error:** `Failed to download image from URL`

**Solutions:**
- Install requests: `pip install requests`
- Check internet connection
- Verify the URL is accessible in a browser
- Check if URL requires authentication

#### 6. Video Processing Issues

**Error:** `opencv-python required`

**Solution:**
```bash
pip install opencv-python
```

### Debugging Tips

1. **Test with a simple local file first:**
   ```bash
   python gemini_vision.py -i test.jpg -p "What is this?"
   ```

2. **Verify your API key:**
   ```bash
   echo $GEMINI_API_KEY
   ```

3. **Check installed packages:**
   ```bash
   pip list | grep -E 'pillow|numpy|google-genai|opencv|pyautogui|requests'
   ```

4. **Use Python to test:**
   ```python
   from gemini_vision import GeminiVision
   import os

   print("API Key set:", bool(os.getenv("GEMINI_API_KEY")))
   vision = GeminiVision()
   ```

## Advanced Usage

### Batch Processing

Process multiple images in a directory:

```python
from gemini_vision import GeminiVision
import os
from pathlib import Path

vision = GeminiVision()
image_dir = Path("./photos")

for img_file in image_dir.glob("*.jpg"):
    result = vision.generate_text(str(img_file), "Describe this image")
    print(f"\n{img_file.name}:")
    print(result)
```

### Custom Prompts for Specific Tasks

```python
from gemini_vision import GeminiVision

vision = GeminiVision()

# OCR (Text extraction)
text = vision.generate_text("document.png", "Extract all text from this image")

# Object counting
count = vision.generate_text("crowd.jpg", "How many people are in this image?")

# Color analysis
colors = vision.generate_text("painting.jpg", "List the main colors in this image")

# Safety check
safety = vision.generate_text("photo.jpg", "Is there any inappropriate content in this image?")

# Translation
translation = vision.generate_text("sign.jpg", "Translate the text in this image to English")
```

### Integration with Other Tools

```python
from gemini_vision import GeminiVision
import json

vision = GeminiVision()

# Structured output request
prompt = """
Analyze this image and return a JSON object with:
- description: brief description
- objects: list of visible objects
- colors: dominant colors
- text: any visible text
"""

result = vision.generate_text("image.jpg", prompt)
print(result)
```

### Error Handling

```python
from gemini_vision import GeminiVision

vision = GeminiVision()

try:
    result = vision.generate_text("photo.jpg", "Describe this")
    print(result)
except FileNotFoundError:
    print("Image file not found!")
except Exception as e:
    print(f"Error occurred: {e}")
```

## Tips for Better Results

1. **Be specific in your prompts:**
   - Good: "List all the animals visible in this image"
   - Better: "Identify each animal in this image and describe their location"

2. **Ask clear questions:**
   - Good: "What is this?"
   - Better: "What type of building is shown in this image and what architectural style is it?"

3. **For OCR tasks:**
   - Use: "Extract and transcribe all visible text from this image"
   - Add: "Preserve the original formatting and layout"

4. **For analysis tasks:**
   - Be explicit about what you want analyzed
   - Specify the format of the output if needed

5. **Image quality matters:**
   - Use clear, well-lit images
   - Higher resolution images generally produce better results
   - Avoid blurry or heavily compressed images

## Support and Resources

- **Gemini API Documentation:** [https://ai.google.dev/docs](https://ai.google.dev/docs)
- **Google AI Studio:** [https://makersuite.google.com](https://makersuite.google.com)
- **Python PIL Documentation:** [https://pillow.readthedocs.io](https://pillow.readthedocs.io)

## License

MIT License

## Version

Current Version: 1.0.0

---

**Happy Image Analyzing with Gemini Vision!** 🚀
