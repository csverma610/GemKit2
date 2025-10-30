"""
A script for debugging PDF chat functionality with the Gemini API.

This script uploads a PDF file, sends a prompt to the Gemini model,
and prints the response object for inspection. It is useful for
understanding the structure of the response object and for debugging
issues with PDF chat.

Usage:
    python debug_pdf_chat.py
"""

import os
import pathlib
from google import genai

def main():
    """
    The main function for the PDF chat debugging script.
    """
    # Initialize client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=api_key)

    # Load PDF
    pdf_path = pathlib.Path("book.pdf")
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found.")
        return

    print(f"Uploading {pdf_path.name}...")
    uploaded_file = client.files.upload(file=pdf_path, config=dict(mime_type='application/pdf'))
    print(f"✓ PDF uploaded: {uploaded_file.name}")

    # Test query
    prompt = "What is this book about?"
    print(f"\nSending prompt: {prompt}")

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[uploaded_file, prompt]
    )

    print(f"\nResponse object type: {type(response)}")
    print(f"Response object: {response}")
    print(f"Has text attr: {hasattr(response, 'text')}")
    if hasattr(response, 'text'):
        print(f"Response.text: {response.text}")
        print(f"Response.text type: {type(response.text)}")
    print(f"\nResponse.__dict__: {response.__dict__ if hasattr(response, '__dict__') else 'N/A'}")

    # Clean up
    client.files.delete(name=uploaded_file.name)

if __name__ == "__main__":
    main()
