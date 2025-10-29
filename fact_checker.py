import argparse
import json
import os
import sys
from typing import List

from google import genai
from pydantic import BaseModel, Field

from gemini_client import GeminiClient

class Claim(BaseModel):
    """Model for representing a single claim and its fact check."""
    claim: str = Field(description="The specific claim extracted from the text")
    rating: str = Field(description="Rating of the claim: TRUE, FALSE, MISLEADING, or UNVERIFIABLE")
    explanation: str = Field(description="Detailed explanation with supporting evidence")
    sources: List[str] = Field(description="List of sources used to verify the claim")


class FactCheckResult(BaseModel):
    """Model for the complete fact check result."""
    overall_rating: str = Field(description="Overall rating: MOSTLY_TRUE, MIXED, or MOSTLY_FALSE")
    summary: str = Field(description="Brief summary of the overall findings")
    claims: List[Claim] = Field(description="List of specific claims and their fact checks")

class FactChecker:
    def __init__(self):
        self.client = GeminiClient()
        
        # The system prompt is hardcoded to ensure consistent behavior
        self.system_prompt = (
            "You are a professional fact-checker with extensive research capabilities. "
            "Your task is to evaluate claims or articles for factual accuracy. "
            "Focus on identifying false, misleading, or unsubstantiated claims."
        )

    def generate_claims(self, text: str):
        prompt = (
            "Given the following text, provide an enumerated list of verifiable claims. "
            "Do not include any other text besides the numbered list itself.\n\n"
            f"{text}"
        )
        text = self.client.generate_text(prompt)

        return response.text

    def check_claim(self, text: str):
        """
        Check the factual accuracy of a claim or article using the Gemini SDK.

        Args:
            text: The claim or article text to fact check

        Returns:
            The parsed response containing fact check results.
        """
        if not text or not text.strip():
            return {"error": "Input text is empty. Cannot perform fact check."}
            
        prompt = f"Check the veracity of the claim {text}"
        text   = self.client.generate_text(prompt)
        return text

def main():
    parser = argparse.ArgumentParser(
        description="Fact Checker CLI - Identify false or misleading claims in text using Gemini"
    )
    
    parser.add_argument("-t", "--text", type=str, help="Text to fact check")
    
    args = parser.parse_args()
    
    fchecker = FactChecker()
    results = fchecker.generate_claims(args.text)
    print(results)

if __name__ == "__main__":
    main()
