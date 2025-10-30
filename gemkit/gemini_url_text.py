
"""
A script to extract text from a URL using the Gemini API.

This script takes a URL as a command-line argument, sends it to the Gemini
model, and prints the extracted text.

Usage:
    python gemini_url_text.py <URL>
"""

import sys
from google import genai
from google.genai.types import Tool, GenerateContentConfig

def main():
    """
    The main function for the URL text extraction script.
    """
    if len(sys.argv) < 2:
        print("Usage: python gemini_url_text.py <URL>")
        sys.exit(1)

    client = genai.Client()
    model_id = "gemini-2.5-flash"

    tools = [
      {"url_context": {}},
    ]

    url = sys.argv[1]

    response = client.models.generate_content(
        model=model_id,
        contents=f"Extract the medical information from {url}",
        config=GenerateContentConfig(
            tools=tools,
        )
    )

    for each in response.candidates[0].content.parts:
        print(each.text)

    # For verification, you can inspect the metadata to see which URLs the model retrieved
    print(response.candidates[0].url_context_metadata)

if __name__ == "__main__":
    main()
