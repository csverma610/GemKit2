#!/usr/bin/env python3
"""
Script to check the quality of an academic paper review.

Usage:
    python check_review_quality.py <pdf_file> <review_json_file>

Example:
    python check_review_quality.py paper.pdf review.json
"""

import sys
import json
from pathlib import Path
from review_quality_checker import ReviewQualityChecker


def main():
    """Main entry point"""
    if len(sys.argv) < 3:
        print("Usage: python check_review_quality.py <pdf_file> <review_json_file>")
        print("\nExample:")
        print("  python check_review_quality.py paper.pdf review.json")
        sys.exit(1)

    pdf_file = sys.argv[1]
    review_json_file = sys.argv[2]

    try:
        checker = ReviewQualityChecker()
        assessment = checker.check_review(pdf_file, review_json_file)

        # Print results
        print("\n" + "=" * 80)
        print("REVIEW QUALITY ASSESSMENT")
        print("=" * 80)

        print(f"\nReview File: {assessment.review_json_file}")
        print(f"Paper File:  {assessment.review_file}")

        # Print scores
        print("\n" + "-" * 80)
        print("QUALITY SCORES (0-100)")
        print("-" * 80)
        print(f"  Overall Quality:     {assessment.scores.overall:6.1f}")
        print(f"  Specificity:         {assessment.scores.specificity:6.1f}  (concrete details, not vague)")
        print(f"  Evidence Support:    {assessment.scores.evidence_support:6.1f}  (cites paper sections)")
        print(f"  Consistency:         {assessment.scores.consistency:6.1f}  (ratings align with feedback)")
        print(f"  Completeness:        {assessment.scores.completeness:6.1f}  (all sections addressed)")
        print(f"  Actionability:       {assessment.scores.actionability:6.1f}  (provides guidance to authors)")

        # Print summary
        print("\n" + "-" * 80)
        print("SUMMARY")
        print("-" * 80)
        print(f"\n{assessment.summary}\n")

        # Print issues
        if assessment.issues:
            print("-" * 80)
            print(f"QUALITY ISSUES ({len(assessment.issues)} found)")
            print("-" * 80)

            # Group by severity
            critical = [i for i in assessment.issues if i.severity == "critical"]
            high = [i for i in assessment.issues if i.severity == "high"]
            medium = [i for i in assessment.issues if i.severity == "medium"]
            low = [i for i in assessment.issues if i.severity == "low"]

            for severity, issues in [("CRITICAL", critical), ("HIGH", high), ("MEDIUM", medium), ("LOW", low)]:
                if issues:
                    print(f"\n{severity} SEVERITY ({len(issues)}):")
                    for i, issue in enumerate(issues, 1):
                        print(f"\n  {i}. [{issue.issue_type.value}] in {issue.section}")
                        print(f"     Issue: {issue.description}")
                        print(f"     Fix:   {issue.suggestion}")

        else:
            print("\n✅ No quality issues found!\n")

        # Print strengths
        if assessment.strengths:
            print("\n" + "-" * 80)
            print("STRENGTHS")
            print("-" * 80)
            for strength in assessment.strengths:
                print(f"  ✓ {strength}")

        # Print recommendations
        if assessment.recommendations:
            print("\n" + "-" * 80)
            print("RECOMMENDATIONS FOR IMPROVEMENT")
            print("-" * 80)
            for rec in assessment.recommendations:
                print(f"  • {rec}")

        print("\n" + "=" * 80)

        # Export to JSON
        output_file = Path(review_json_file).stem + "_quality_assessment.json"
        with open(output_file, 'w') as f:
            json.dump(assessment.model_dump(), f, indent=2)
        print(f"\n✅ Assessment saved to: {output_file}\n")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
