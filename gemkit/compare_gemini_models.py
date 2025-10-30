import sys
import os
import openai
import time

from lmdb_storage import LMDBStorage
from gemini_client import GeminiClient

"""
A script to compare the output of different Gemini models for a given question.

This script takes a question as a command-line argument, sends it to multiple
Gemini models, and prints the output of each model, along with the time taken
to generate the response.

Usage:
    python compare_gemini_models.py "Your question here"
"""

import sys
import time

from gemini_client import GeminiClient

def main():
    """
    The main function for the model comparison script.
    """
    if len(sys.argv) < 2:
        print("Usage: python compare_gemini_models.py \"Your question here\"")
        sys.exit(1)

    question = sys.argv[1]

    results = []
    for model_name in GeminiClient.MODELS_NAME:
        print("MODEL:", model_name)
        client = GeminiClient(model_name=model_name)

        start_time = time.time()
        text = client.generate_text(question)
        end_time = time.time()
        
        print(text)
        print(len(text))
        print(f"\n(Time taken: {end_time - start_time:.2f} seconds)")
        results.append(text)

if __name__ == "__main__":
    main()

