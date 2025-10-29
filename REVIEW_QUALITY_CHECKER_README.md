# Review Quality Checker

A tool to validate and score the quality of academic paper reviews. It checks reviews against best practices and provides specific feedback for improvement.

## Overview

The Review Quality Checker analyzes reviews to ensure they:
- **Contain specific evidence** from the paper (not generic praise/criticism)
- **Support claims** with concrete examples (Section X, Figure Y, Table Z)
- **Maintain consistency** between ratings and feedback
- **Provide actionable guidance** to authors
- **Are thorough and complete** across all review aspects

## Features

### Quality Scoring (0-100)

- **Overall Score**: Combined quality metric
- **Specificity**: How concrete and detailed is the review (avoids vague language)
- **Evidence Support**: Cites specific paper sections, figures, tables, equations
- **Consistency**: Ratings align with written feedback and recommendations
- **Completeness**: All key sections are addressed
- **Actionability**: Provides clear suggestions and guidance

### Issue Detection

Automatically detects and categorizes issues:

| Issue Type | Description | Severity |
|---|---|---|
| `missing_evidence` | Claims lack paper-specific references | High |
| `generic_feedback` | Contains vague words (good, bad, nice, etc.) | High |
| `inconsistent_rating` | Rating conflicts with feedback | Critical |
| `unsupported_claim` | Major issues lack sufficient detail | High |
| `vague_description` | Descriptions are too abstract | Medium |
| `missing_specific_section` | Required section is empty | Critical |
| `contradictory_statements` | Feedback contradicts itself | Critical |
| `insufficient_detail` | Section too short (< 10 words) | High |

## Usage

### Basic Usage

```bash
python check_review_quality.py <pdf_file> <review_json_file>
```

### Example

```bash
python check_review_quality.py paper.pdf review.json
```

### Output

The tool produces:

1. **Console Report** with:
   - Quality scores for each dimension
   - Summary assessment
   - List of issues (grouped by severity)
   - Identified strengths
   - Recommendations for improvement

2. **JSON Export** (`review_quality_assessment.json`) with detailed results

## Python API

### Quick Start

```python
from review_quality_checker import ReviewQualityChecker

checker = ReviewQualityChecker()
assessment = checker.check_review("paper.pdf", "review.json")

# Access results
print(f"Overall Score: {assessment.scores.overall}")
print(f"Issues Found: {len(assessment.issues)}")
print(f"Summary: {assessment.summary}")

# Iterate through issues
for issue in assessment.issues:
    print(f"[{issue.severity}] {issue.description}")
    print(f"  Fix: {issue.suggestion}")
```

### Classes

#### `ReviewQualityChecker`

Main class for checking review quality.

**Methods:**
- `check_review(pdf_file, review_json_file) -> ReviewQualityAssessment`
  - Analyzes a review and returns quality assessment
  - Raises `FileNotFoundError` if files don't exist
  - Raises `ValueError` if review JSON is invalid

#### `ReviewQualityAssessment`

Complete quality assessment result containing:

```python
{
    "review_file": str,                    # Path to PDF
    "review_json_file": str,               # Path to review JSON
    "scores": ReviewQualityScore,          # Numeric scores
    "issues": List[QualityIssue],         # Detected issues
    "strengths": List[str],               # What the review does well
    "recommendations": List[str],         # How to improve
    "summary": str                        # Executive summary
}
```

#### `ReviewQualityScore`

Scores (0-100) for different quality dimensions:

```python
{
    "specificity": float,            # Concrete details vs vague language
    "evidence_support": float,       # Paper-specific references
    "consistency": float,            # Ratings align with feedback
    "completeness": float,           # All sections addressed
    "actionability": float,          # Clear guidance provided
    "overall": float                 # Weighted average
}
```

#### `QualityIssue`

Individual quality issue with:

```python
{
    "issue_type": QualityIssueType,   # Type of issue
    "section": str,                   # Where in review (e.g., "methodology")
    "severity": str,                  # critical/high/medium/low
    "description": str,               # What's wrong
    "suggestion": str                 # How to fix
}
```

## Quality Thresholds

- **0-40**: Poor review quality - significant improvements needed
- **40-60**: Moderate quality - several improvements recommended
- **60-80**: Good quality - room for improvement
- **80-100**: Excellent quality - meets best practices

## Common Issues & Fixes

### Issue: Low Evidence Support Score

**Problem**: Review makes claims without citing specific parts of the paper

**Fixes**:
- Add references: "In Section 3, the authors..."
- Cite figures: "According to Figure 2..."
- Quote tables: "Table 1 shows..."
- Reference equations: "Equation (5) demonstrates..."

### Issue: Low Specificity Score

**Problem**: Review uses vague language instead of technical terms

**Vague** ❌
> "The methodology is well done and the results are interesting."

**Specific** ✅
> "The experimental design uses randomized controlled trials with adequate sample size (n=500). Results show 87% accuracy on the ImageNet validation set, exceeding the SOTA baseline by 4.2%."

### Issue: Inconsistent Ratings

**Problem**: Recommendation conflicts with written assessment

**Example** ❌
> Methodology rating: POOR
> Recommendation: ACCEPT

**Fix** ✓
- Either improve the methodology assessment with specific improvements, or
- Change recommendation to REJECT with clear justification

### Issue: Incomplete Sections

**Problem**: Required fields are empty or too short

**Fix**:
- Ensure all sections have meaningful content
- Minimum 10 words per section
- Better: 50-200 words for thorough assessment
- Include specific examples from the paper

## Best Practices for Writing High-Quality Reviews

1. **Always cite the paper**: Use section numbers, figure labels, table references
2. **Be specific**: "Paragraph 3 has unclear notation" vs "Unclear"
3. **Provide examples**: Quote specific passages or metrics
4. **Link claims to evidence**: "The methodology is weak because [specific reason from Section 2.1]"
5. **Align ratings with feedback**: Don't give "poor" rating with positive feedback
6. **Make it actionable**: "Missing ablation study for component X" vs "More ablations needed"
7. **Balance**: Include both strengths and weaknesses
8. **Justify recommendations**: Explain why accept/reject based on the analysis

## Integration with Paper Reviewer

The Quality Checker works with reviews produced by `PaperReviewer`:

```python
from paper_reviewer import PaperReviewer
from review_quality_checker import ReviewQualityChecker

# Generate initial review
reviewer = PaperReviewer()
review = reviewer.review("paper.pdf")

# Save review to JSON
import json
with open("review.json", "w") as f:
    json.dump(review.model_dump(), f, indent=2)

# Check quality of the review
checker = ReviewQualityChecker()
assessment = checker.check_review("paper.pdf", "review.json")

# View results
print(f"Review Quality Score: {assessment.scores.overall}/100")
for issue in assessment.issues:
    print(f"  - {issue.description}")
```

## Examples

### Example 1: Low-Quality Review

```
Overall Score: 35/100
Issues: 12 found (3 critical, 7 high)

Summary: Poor review quality - significant improvements needed. Found 3 critical issues that should be addressed. Found 7 high-priority issues for improvement.

Critical Issues:
1. [inconsistent_rating] in recommendation
   Issue: Recommendation 'strong_accept' conflicts with poor methodology rating
   Fix: Either improve the methodology assessment or revise the recommendation

2. [missing_specific_section] in methodology
   Issue: Field 'soundness' is empty
   Fix: Provide detailed assessment for soundness

High Issues:
1. [missing_evidence] in methodology
   Issue: Field 'soundness' lacks specific references (0 found, need 2)
   Fix: Add references to specific sections, figures, tables, or equations from the paper
```

### Example 2: High-Quality Review

```
Overall Score: 87/100
Issues: 2 found (0 critical, 1 high, 1 medium)

Summary: Excellent review quality. Found 1 high-priority issue for improvement.

Strengths:
✓ Provides 3+ key strengths - comprehensive positive assessment
✓ Identifies 3+ key weaknesses - thorough problem analysis
✓ Documents major issues - helps authors understand critical problems
✓ Asks clarifying questions - promotes dialogue with authors

Recommendations:
• Add more specific references to sections, figures, and tables from the paper
```

## Requirements

- Python 3.8+
- Pydantic v2
- paper_reviewer module

## License

Same as parent project
