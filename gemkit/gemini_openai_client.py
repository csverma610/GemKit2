import os
import openai
import time

from lmdb_storage import LMDBStorage

class GeminiClient:
    """
    A client to interact with the Gemini API via an OpenAI-compatible endpoint.
    """
    MODELS_NAME = ['gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-flash-latest', 'gemini-flash-lite-latest']
    DEFAULT_MODEL = 'gemini-2.5-flash'

    def __init__(self, model_name = DEFAULT_MODEL):
        self.model_name = model_name
        self._create_client()

    def _create_client(self):
        # Get the API key from an environment variable.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found.")

        self.base_url = "https://generativelanguage.googleapis.com/v1beta/openai"
        self.client = openai.OpenAI( api_key=api_key, base_url=self.base_url)

    def generate_text(self, user_prompt: str, assist_prompt: str = "", sys_prompt: str = "") -> str:
        """
        Sends a prompt to the Gemini API and returns the text response.

        Args:
            user_prompt (str): The user's question or prompt.
            assist_prompt (str): Optional assistant message for few-shot examples or context.
            sys_prompt (str): Optional system prompt to set behavior or context.

        Returns:
            str: The text response from the model.
        """
        messages = []
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})

        if assist_prompt:
            messages.append({"role": "assistant", "content": assist_prompt})

        messages.append({"role": "user", "content": user_prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except openai.APIError as e:
            raise Exception(f"Gemini API error: {e}") from e
        except openai.APIConnectionError as e:
            raise Exception(f"Failed to connect to Gemini API: {e}") from e
        except Exception as e:
            raise Exception(f"An error occurred during API call: {e}") from e

import argparse
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="CLI tool to query the Gemini API via the OpenAI-compatible endpoint."
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="The question or prompt to send to the Gemini model (required positional argument)."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=GeminiClient.DEFAULT_MODEL,
        choices=GeminiClient.MODELS_NAME,
        help=f"The Gemini model to use. Available options: {', '.join(GeminiClient.MODELS_NAME)}"
    )

    args = parser.parse_args()
    
    # Use parsed arguments
    question = args.question
    model_name = args.model

    # Create instances of the API client and storage classes.
    try:
        # Pass the parsed model name to the client constructor
        client = GeminiClient(model_name)

        print(f"\nQuery: {question}\n")
        
        # Get the response from the Gemini API.
        start_time = time.time()
        text = client.generate_text(question)
        end_time = time.time()
        
        print("\nResponse:")
        print(text)

        # Store the conversation.
        storage = LMDBStorage("geminiqa.lmdb")
        storage.put(question, text)

    except ValueError as ve:
        # Configuration issues like missing API key
        print(f"\n[CRITICAL CONFIG ERROR]: {ve}")
    except Exception as e:
        # Any other runtime issues
        print(f"\n[RUNTIME ERROR]: An unexpected error occurred: {e}")
        
    finally:
        print("\nExecution finished.")

