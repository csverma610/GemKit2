"""
Review Quality Checker

Validates and scores the quality of academic paper reviews.
Input: PDF file and initial review (JSON)
Output: Quality assessment with specific issues and suggestions
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field

from gemkit.paper_reviewer import ComprehensivePaperReview, ReviewCategory


class QualityIssueType(str, Enum):
    """Types of quality issues found in reviews"""
    MISSING_EVIDENCE = "missing_evidence"
    GENERIC_FEEDBACK = "generic_feedback"
    INCONSISTENT_RATING = "inconsistent_rating"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    VAGUE_DESCRIPTION = "vague_description"
    MISSING_SPECIFIC_SECTION = "missing_specific_section"
    CONTRADICTORY_STATEMENTS = "contradictory_statements"
    INSUFFICIENT_DETAIL = "insufficient_detail"


class QualityIssue(BaseModel):
    """A specific quality issue found in the review"""
    issue_type: QualityIssueType = Field(description="Type of quality issue")
    section: str = Field(description="Which part of review has the issue (e.g., 'methodology', 'writing')")
    severity: str = Field(
        description="Severity level: 'critical' (affects credibility), 'high' (important), 'medium' (should improve), 'low' (minor)"
    )
    description: str = Field(description="Description of the issue")
    suggestion: str = Field(description="Specific suggestion to fix the issue")


class ReviewQualityScore(BaseModel):
    """Quality scores for different aspects of the review"""
    specificity: float = Field(ge=0, le=100, description="How specific and concrete is the review (0-100)")
    evidence_support: float = Field(ge=0, le=100, description="How well claims are supported with evidence (0-100)")
    consistency: float = Field(ge=0, le=100, description="How consistent is the review with itself (0-100)")
    completeness: float = Field(ge=0, le=100, description="How complete/thorough is the review (0-100)")
    actionability: float = Field(ge=0, le=100, description="How actionable are the suggestions (0-100)")
    overall: float = Field(ge=0, le=100, description="Overall quality score (0-100)")


class ReviewQualityAssessment(BaseModel):
    """Complete quality assessment of a review"""
    review_file: str = Field(description="Path to reviewed PDF file")
    review_json_file: str = Field(description="Path to review JSON file")

    scores: ReviewQualityScore = Field(description="Quality scores for different aspects")

    issues: List[QualityIssue] = Field(
        default_factory=list,
        description="Specific quality issues found"
    )

    strengths: List[str] = Field(
        default_factory=list,
        description="What the review does well"
    )

    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations for improving the review"
    )

    summary: str = Field(
        description="Executive summary of the quality assessment"
    )


class ReviewQualityChecker:
    """
    Checks the quality of academic paper reviews.

    This class provides methods to validate and score a review based on criteria
    such as specificity, evidence support, consistency, and completeness. It can
    identify common issues in reviews and provide recommendations for improvement.
    """

    def __init__(self):
        """
        Initializes the ReviewQualityChecker.
        """
        self.min_specificity_words = 10  # Minimum words per assessment
        self.min_evidence_references = 2  # Minimum references to paper sections

    def check_review(
        self,
        pdf_file: str,
        review_json_file: str
    ) -> ReviewQualityAssessment:
        """
        Checks the quality of a review.

        Args:
            pdf_file (str): The path to the PDF file of the paper that was reviewed.
            review_json_file (str): The path to the JSON file containing the review.

        Returns:
            ReviewQualityAssessment: A Pydantic model containing the detailed
                                     quality analysis of the review.
        """
        # Validate files exist
        if not os.path.exists(pdf_file):
            raise FileNotFoundError(f"PDF file not found: {pdf_file}")
        if not os.path.exists(review_json_file):
            raise FileNotFoundError(f"Review JSON file not found: {review_json_file}")

        # Load and parse review
        try:
            with open(review_json_file, 'r') as f:
                review_data = json.load(f)
            review = ComprehensivePaperReview(**review_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in review file: {str(e)}") from e
        except Exception as e:
            raise ValueError(f"Invalid review structure: {str(e)}") from e

        # Run quality checks
        issues = self._check_for_issues(review)
        scores = self._calculate_scores(review, issues)
        strengths = self._identify_strengths(review)
        recommendations = self._generate_recommendations(issues, review)
        summary = self._generate_summary(scores, issues)

        return ReviewQualityAssessment(
            review_file=pdf_file,
            review_json_file=review_json_file,
            scores=scores,
            issues=issues,
            strengths=strengths,
            recommendations=recommendations,
            summary=summary
        )

    def _check_for_issues(self, review: ComprehensivePaperReview) -> List[QualityIssue]:
        """Identify quality issues in the review"""
        issues = []

        # Check methodology review
        issues.extend(self._check_string_field(
            review.methodology.soundness,
            "methodology",
            "soundness"
        ))
        issues.extend(self._check_string_field(
            review.methodology.experimental_design,
            "methodology",
            "experimental_design"
        ))

        # Check novelty review
        issues.extend(self._check_string_field(
            review.novelty.originality,
            "novelty",
            "originality"
        ))
        issues.extend(self._check_string_field(
            review.novelty.significance,
            "novelty",
            "significance"
        ))

        # Check writing review
        issues.extend(self._check_string_field(
            review.writing.clarity,
            "writing",
            "clarity"
        ))
        issues.extend(self._check_string_field(
            review.writing.organization,
            "writing",
            "organization"
        ))

        # Check results review
        issues.extend(self._check_string_field(
            review.results.result_quality,
            "results",
            "result_quality"
        ))
        issues.extend(self._check_string_field(
            review.results.interpretation,
            "results",
            "interpretation"
        ))

        # Check for vague language in key sections
        issues.extend(self._check_for_generic_language(review))

        # Check consistency between ratings and feedback
        issues.extend(self._check_rating_consistency(review))

        # Check for missing evidence citations
        issues.extend(self._check_evidence_citations(review))

        return issues

    def _check_string_field(
        self,
        field_value: str,
        section: str,
        field_name: str
    ) -> List[QualityIssue]:
        """Check a string field for quality issues"""
        issues = []

        if not field_value or len(field_value.strip()) == 0:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_SPECIFIC_SECTION,
                section=section,
                severity="critical",
                description=f"Field '{field_name}' is empty",
                suggestion=f"Provide detailed assessment for {field_name}"
            ))
            return issues

        word_count = len(field_value.split())
        if word_count < self.min_specificity_words:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.INSUFFICIENT_DETAIL,
                section=section,
                severity="high",
                description=f"Field '{field_name}' is too short ({word_count} words). Needs more detail.",
                suggestion=f"Expand {field_name} to at least {self.min_specificity_words} words with specific examples"
            ))

        # Check for vague words
        vague_phrases = [
            "good", "bad", "nice", "terrible", "okay", "fine",
            "interesting", "important", "relevant", "significant"
        ]
        vague_count = sum(1 for phrase in vague_phrases if phrase in field_value.lower())
        if vague_count > 0:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.GENERIC_FEEDBACK,
                section=section,
                severity="high",
                description=f"Field '{field_name}' contains {vague_count} vague word(s): {vague_phrases[:3]}",
                suggestion=f"Replace vague words with specific technical terms and concrete examples"
            ))

        # Check for section references (e.g., "Section 3", "Figure 2", "Table 1")
        section_refs = self._count_section_references(field_value)
        if section_refs < self.min_evidence_references:
            issues.append(QualityIssue(
                issue_type=QualityIssueType.MISSING_EVIDENCE,
                section=section,
                severity="high",
                description=f"Field '{field_name}' lacks specific references ({section_refs} found, need {self.min_evidence_references})",
                suggestion=f"Add references to specific sections, figures, tables, or equations from the paper"
            ))

        return issues

    def _check_for_generic_language(self, review: ComprehensivePaperReview) -> List[QualityIssue]:
        """Check for overly generic language throughout the review"""
        generic_phrases = {
            "well-written": "Specify what aspects are well-written",
            "good work": "Explain what specifically is good",
            "needs improvement": "Specify exactly what needs to improve",
            "clear and concise": "Provide examples of clarity",
            "comprehensive": "Explain what makes it comprehensive",
            "thorough": "Give specific examples of thoroughness"
        }

        issues = []
        review_text = str(review.detailed_feedback.comments_for_authors)

        for phrase, suggestion in generic_phrases.items():
            if phrase in review_text.lower():
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.GENERIC_FEEDBACK,
                    section="detailed_feedback",
                    severity="medium",
                    description=f"Generic phrase detected: '{phrase}'",
                    suggestion=suggestion
                ))

        return issues

    def _check_rating_consistency(self, review: ComprehensivePaperReview) -> List[QualityIssue]:
        """Check consistency between ratings and written feedback"""
        issues = []

        # Map ratings to quality thresholds
        rating_to_quality = {
            ReviewCategory.EXCELLENT: 0.8,
            ReviewCategory.GOOD: 0.6,
            ReviewCategory.SATISFACTORY: 0.4,
            ReviewCategory.NEEDS_IMPROVEMENT: 0.2,
            ReviewCategory.POOR: 0.0
        }

        # If recommendation is "strong_accept" but methodology is "poor", that's inconsistent
        if review.recommendation.decision == "strong_accept":
            if review.methodology.rating in [ReviewCategory.POOR, ReviewCategory.NEEDS_IMPROVEMENT]:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INCONSISTENT_RATING,
                    section="recommendation",
                    severity="critical",
                    description="Recommendation 'strong_accept' conflicts with poor methodology rating",
                    suggestion="Either improve the methodology assessment or revise the recommendation"
                ))

        if review.recommendation.decision in ["reject", "strong_reject"]:
            if review.methodology.rating in [ReviewCategory.EXCELLENT, ReviewCategory.GOOD]:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.INCONSISTENT_RATING,
                    section="recommendation",
                    severity="critical",
                    description="Reject recommendation conflicts with good methodology rating",
                    suggestion="Clarify the major issues that warrant rejection"
                ))

        return issues

    def _check_evidence_citations(self, review: ComprehensivePaperReview) -> List[QualityIssue]:
        """Check that claims are supported with evidence"""
        issues = []

        # Check if major issues lack specificity
        for issue in review.specific_issues.major_issues:
            if len(issue.split()) < 15:
                issues.append(QualityIssue(
                    issue_type=QualityIssueType.UNSUPPORTED_CLAIM,
                    section="specific_issues",
                    severity="high",
                    description="Major issue lacks sufficient detail",
                    suggestion="Expand major issues with specific examples and explain their impact"
                ))
                break  # Only report once

        return issues

    def _calculate_scores(
        self,
        review: ComprehensivePaperReview,
        issues: List[QualityIssue]
    ) -> ReviewQualityScore:
        """Calculate quality scores"""

        # Base scores (100 = perfect, 0 = worst)
        specificity_score = 100 - (len([i for i in issues if i.issue_type == QualityIssueType.GENERIC_FEEDBACK]) * 10)
        evidence_score = 100 - (len([i for i in issues if i.issue_type == QualityIssueType.MISSING_EVIDENCE]) * 15)
        consistency_score = 100 - (len([i for i in issues if i.issue_type == QualityIssueType.INCONSISTENT_RATING]) * 20)

        # Check completeness (all sections filled)
        completeness_count = 0
        if review.methodology.soundness: completeness_count += 1
        if review.novelty.originality: completeness_count += 1
        if review.writing.clarity: completeness_count += 1
        if review.results.result_quality: completeness_count += 1
        if review.literature.comprehensiveness: completeness_count += 1
        if review.specific_issues.major_issues: completeness_count += 1
        if review.detailed_feedback.comments_for_authors: completeness_count += 1
        completeness_score = (completeness_count / 7) * 100

        # Check actionability (suggestions provided)
        actionability_count = 0
        if review.writing.suggestions: actionability_count += 1
        if review.specific_issues.questions_for_authors: actionability_count += 1
        if review.detailed_feedback.comments_for_authors: actionability_count += 1
        actionability_score = (actionability_count / 3) * 100

        # Clamp scores to 0-100
        specificity_score = max(0, min(100, specificity_score))
        evidence_score = max(0, min(100, evidence_score))
        consistency_score = max(0, min(100, consistency_score))
        completeness_score = max(0, min(100, completeness_score))
        actionability_score = max(0, min(100, actionability_score))

        overall_score = (
            specificity_score * 0.25 +
            evidence_score * 0.25 +
            consistency_score * 0.2 +
            completeness_score * 0.15 +
            actionability_score * 0.15
        )

        return ReviewQualityScore(
            specificity=round(specificity_score, 1),
            evidence_support=round(evidence_score, 1),
            consistency=round(consistency_score, 1),
            completeness=round(completeness_score, 1),
            actionability=round(actionability_score, 1),
            overall=round(overall_score, 1)
        )

    def _identify_strengths(self, review: ComprehensivePaperReview) -> List[str]:
        """Identify what the review does well"""
        strengths = []

        # Check for comprehensive coverage
        if len(review.overall_assessment.strengths) >= 3:
            strengths.append("Provides 3+ key strengths - comprehensive positive assessment")

        if len(review.overall_assessment.weaknesses) >= 3:
            strengths.append("Identifies 3+ key weaknesses - thorough problem analysis")

        if len(review.specific_issues.major_issues) > 0:
            strengths.append("Documents major issues - helps authors understand critical problems")

        if len(review.specific_issues.questions_for_authors) > 0:
            strengths.append("Asks clarifying questions - promotes dialogue with authors")

        if review.recommendation.conditions_for_acceptance:
            strengths.append("Provides conditions for acceptance - clear path forward for authors")

        if len(review.detailed_feedback.comments_for_authors) > 200:
            strengths.append("Detailed feedback - thorough guidance for improvement")

        return strengths if strengths else ["Review is complete but may benefit from more specific evidence"]

    def _generate_recommendations(
        self,
        issues: List[QualityIssue],
        review: ComprehensivePaperReview
    ) -> List[str]:
        """Generate recommendations for improving the review"""
        recommendations = []

        # Group issues by type
        issue_types = {}
        for issue in issues:
            if issue.issue_type not in issue_types:
                issue_types[issue.issue_type] = []
            issue_types[issue.issue_type].append(issue)

        # Generate recommendations based on issues
        if QualityIssueType.MISSING_EVIDENCE in issue_types:
            recommendations.append("Add specific references to sections, figures, and tables from the paper")

        if QualityIssueType.GENERIC_FEEDBACK in issue_types:
            recommendations.append("Replace vague adjectives with specific technical observations")

        if QualityIssueType.INCONSISTENT_RATING in issue_types:
            recommendations.append("Ensure ratings align with the written feedback and recommendations")

        if QualityIssueType.INSUFFICIENT_DETAIL in issue_types:
            recommendations.append("Expand key sections with more detailed analysis and examples")

        if len(review.specific_issues.major_issues) == 0:
            recommendations.append("Consider identifying any major issues that warrant the current recommendation")

        if not review.recommendation.conditions_for_acceptance and review.recommendation.decision in ["weak_accept", "borderline"]:
            recommendations.append("Specify conditions that authors must meet for acceptance")

        return recommendations if recommendations else ["Review is well-structured. Consider adding more paper-specific evidence."]

    def _count_section_references(self, text: str) -> int:
        """Count references to paper sections (Section X, Figure Y, Table Z, Equation W)"""
        import re

        patterns = [
            r"(?i)section\s+\d+",  # Section 1, Section 2.1, etc
            r"(?i)figure\s+\d+",   # Figure 1, Figure 3b, etc
            r"(?i)table\s+\d+",    # Table 1, Table 2a, etc
            r"(?i)equation\s+\d+", # Equation 1, Eq. 1, etc
            r"(?i)algorithm\s+\d+", # Algorithm 1, etc
            r"(?i)appendix\s+\w",  # Appendix A, Appendix B, etc
        ]

        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, text))

        return count

    def _generate_summary(
        self,
        scores: ReviewQualityScore,
        issues: List[QualityIssue]
    ) -> str:
        """Generate executive summary of the quality assessment"""

        critical_issues = [i for i in issues if i.severity == "critical"]
        high_issues = [i for i in issues if i.severity == "high"]

        if scores.overall >= 80:
            overall_assessment = "Excellent review quality"
        elif scores.overall >= 60:
            overall_assessment = "Good review quality with some areas for improvement"
        elif scores.overall >= 40:
            overall_assessment = "Moderate review quality - several improvements needed"
        else:
            overall_assessment = "Poor review quality - significant improvements needed"

        summary = f"{overall_assessment} (overall score: {scores.overall}/100). "

        if critical_issues:
            summary += f"Found {len(critical_issues)} critical issue(s) that should be addressed. "

        if high_issues:
            summary += f"Found {len(high_issues)} high-priority issue(s) for improvement. "

        summary += f"Key strengths: specificity ({scores.specificity:.0f}), "
        summary += f"evidence support ({scores.evidence_support:.0f}), "
        summary += f"consistency ({scores.consistency:.0f}). "

        summary += "See detailed issues and recommendations below."

        return summary
