# GemKit - Gemini PDF Chat & Paper Reviewer

A comprehensive toolkit for interacting with PDF documents using Google's Gemini API, with advanced structured output support for academic paper reviews.

## Features

### 1. **PDF Paper Reviewer** (`pdf_paper_reviewer.py`)
Integrated tool combining PDF chat and structured paper reviews.

#### Review Command
Generate a comprehensive academic paper review with structured output:
```bash
python pdf_paper_reviewer.py review -i paper.pdf
python pdf_paper_reviewer.py review -i paper.pdf --model gemini-1.5-pro
python pdf_paper_reviewer.py review -i paper.pdf -o review.json
```

#### Chat Command
Start an interactive chat session with a PDF:
```bash
python pdf_paper_reviewer.py chat -i paper.pdf
python pdf_paper_reviewer.py chat -i paper.pdf -q "Summarize the methodology"
python pdf_paper_reviewer.py chat -i paper.pdf --model gemini-1.5-pro
```

### 2. **PDF Chat** (`gemini_pdf_chat.py`)
Lightweight PDF chat interface for casual questions and discussions.

```bash
python gemini_pdf_chat.py -i document.pdf
python gemini_pdf_chat.py -i document.pdf -q "What is the main contribution?"
python gemini_pdf_chat.py -i document.pdf --model gemini-1.5-pro
```

### 3. **Paper Reviewer** (`paper_reviewer.py`)
Pydantic models and logic for comprehensive academic paper reviews. Can be used programmatically or imported into other scripts.

```python
from paper_reviewer import ComprehensivePaperReview, review_paper_with_gemini

# Review a paper
review = review_paper_with_gemini(paper_content)
print(review.recommendation.decision)
print(review.overall_assessment.strengths)

# Export to JSON
with open("review.json", "w") as f:
    f.write(review.model_dump_json(indent=2))
```

## Setup

### Requirements
- Python 3.10+
- Google Generative AI SDK
- Pydantic v2

### Installation
```bash
pip install google-generativeai pydantic
```

### Configuration
Set your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Architecture

### GeminiPDFChat Class
- **Minimal, focused design** - Only handles PDF upload/chat
- **Context manager support** - Automatic cleanup with `with` statement
- **Flexible output** - Returns either plain text or structured Pydantic models

**Key Methods:**
- `load_pdf(pdf_file)` - Upload and validate PDF
- `generate_text(prompt, response_schema=None)` - Generate response with optional schema
- `_delete_pdf()` - Clean up uploaded file

### ComprehensivePaperReview Schema
Comprehensive Pydantic model covering all aspects of academic paper review:

**Sections:**
- `metadata` - Paper information
- `executive_summary` - Brief overview
- `methodology` - Methodology assessment
- `novelty` - Novelty and contribution
- `writing` - Writing quality
- `visual_elements` - Tables, figures, diagrams, images
- `literature` - Literature review assessment
- `results` - Results and analysis
- `ethical_considerations` - Ethics (optional)
- `overall_assessment` - Key strengths and weaknesses
- `specific_issues` - Major/minor issues and author questions
- `detailed_feedback` - Detailed comments for authors
- `recommendation` - Publication decision with justification

## Usage Examples

### Generate a Paper Review Programmatically
```python
from paper_reviewer import review_paper_with_gemini
import json

with open("paper.txt", "r") as f:
    paper_content = f.read()

review = review_paper_with_gemini(paper_content, model_name="gemini-1.5-pro")

# Access structured data
print(f"Recommendation: {review.recommendation.decision}")
print(f"Confidence: {review.recommendation.confidence}")
print(f"Strengths:")
for strength in review.overall_assessment.strengths:
    print(f"  - {strength}")

# Save to JSON
with open("review.json", "w") as f:
    json.dump(json.loads(review.model_dump_json()), f, indent=2)
```

### Interactive PDF Chat
```python
from pdf_paper_reviewer import GeminiPDFChat

with GeminiPDFChat(model_name="gemini-1.5-pro") as chat:
    chat.load_pdf("research_paper.pdf")

    # Ask questions
    response = chat.generate_text("What are the main results?")
    print(response)
```

### Structured Output for Custom Schemas
```python
from pdf_paper_reviewer import GeminiPDFChat
from pydantic import BaseModel, Field
from typing import List

class Summary(BaseModel):
    title: str = Field(description="Paper title")
    main_contribution: str = Field(description="Main contribution in 1-2 sentences")
    methodology: str = Field(description="Methodology overview")
    results: List[str] = Field(description="Key results")

with GeminiPDFChat() as chat:
    chat.load_pdf("paper.pdf")
    summary = chat.generate_text("Summarize this paper", response_schema=Summary)
    print(summary.title)
    print(summary.main_contribution)
```

## Design Decisions

### Simplicity First
- Removed overcomplicated dynamic import logic
- Removed dependency on `pydantic_prompt_generator`
- Focused on core functionality

### No State Management
- `response_schema` passed per-request, not stored
- No history tracking in base chat class
- Stateless design for reliability

### Flexible Output
- Returns Pydantic models when schema provided
- Returns plain text for unstructured queries
- User controls output format

### Clean CLI
- Two main entry points: `review` and `chat`
- Clear, concise examples
- Consistent error handling

## API Models Supported

- `gemini-2.5-flash` (default)
- `gemini-1.5-pro`
- Any other Gemini model available in the API

## Error Handling

All scripts include comprehensive error handling:
- Missing API key detection
- Invalid PDF validation
- Connection error recovery
- Detailed logging to file and console

## Logging

Logs are written to:
- `pdf_paper_reviewer.log` - Main integrated tool
- `pdf_chat.log` - PDF chat standalone
- Console (INFO level and above)

## Limitations

- PDF must be valid and accessible
- API key required (set via environment variable)
- Structured output depends on Gemini model quality
- Large PDFs may hit API limits

## Future Enhancements

- Batch processing multiple PDFs
- Custom review templates
- Review comparison/aggregation
- PDF annotation export
- Web interface for reviews

## License

MIT
