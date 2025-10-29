import sys
import os
import openai
import time

from lmdb_storage import LMDBStorage
from gemini_client import GeminiClient

if __name__ == "__main__":
    question = sys.argv[1]

    results = []
    for i in range(4):
        model_name = GeminiClient.MODELS_NAME[i]
        print("MODEL:", model_name)
        client = GeminiClient(model_name=model_name)

        start_time = time.time()
        text = client.generate_text(question)
        end_time = time.time()
        
        print(text)
        print(len(text))
        print(f"\n(Time taken: {end_time - start_time:.2f} seconds)")
        results.append(text)

