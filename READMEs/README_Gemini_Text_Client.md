# Gemini Text Client

A robust and feature-rich Python client for interacting with Google's Gemini large language models. This client is designed for both library and command-line usage, providing a resilient and easy-to-use interface for text generation, structured data extraction, and streaming.

It is built with production use cases in mind, incorporating features like exponential backoff, rate limiting, and a unique two-pass self-correction mechanism for highly reliable JSON output.

## Key Features

- **Unified Interface**: A single `generate()` method intelligently handles text, structured JSON, and streaming requests.
- **Robust JSON Generation**: Implements a two-pass self-correction mechanism. If the model's initial JSON output is invalid, the client automatically sends a follow-up request asking the model to fix its own mistake, dramatically increasing reliability.
- **Streaming Support**: Process responses as they are generated with support for streaming text output, ideal for interactive applications.
- **Resilient by Design**: Features automatic retries with exponential backoff for transient network errors and configurable rate limiting to prevent API abuse.
- **Flexible Configuration**: Utilizes dataclasses (`ModelConfig`, `ModelInput`) for clean, type-safe, and easy-to-manage configuration of model parameters and prompts.
- **Command-Line Interface**: A powerful CLI allows for direct interaction with the Gemini API from your terminal, supporting all major features of the client.
- **Flexible Authentication**: Use the `GEMINI_API_KEY` environment variable or pass your key directly to the client or via a CLI argument.

## Requirements

- Python 3.9+
- `google-generativeai`
- `lmdb` (optional, for saving conversations via the CLI)

## Setup

1.  **Set API Key**:
    Make your Gemini API key available as an environment variable. This is the recommended approach for security.

    ```bash
    export GEMINI_API_KEY="YOUR_API_KEY"
    ```
    Alternatively, you can pass the key directly when instantiating the client or via the `--api-key` CLI argument.

2.  **Install Dependencies**:
    Install the required Python libraries.

    ```bash
    pip install google-generativeai lmdb
    ```

## Usage

The script can be used as a Python library in your own projects or directly as a command-line tool.

### As a Library

Import the necessary classes and instantiate the `GeminiClient`.

**Example 1: Basic Text Generation**

```python
from gemini_text_client import GeminiClient, ModelConfig, ModelInput

# Configure the model
config = ModelConfig(model_name='gemini-1.5-flash', temperature=0.7)
client = GeminiClient(config)

# Prepare the input
model_input = ModelInput(user_prompt="Explain the theory of relativity in simple terms.")

# Generate the response
response = client.generate(model_input)
print(response)
```

**Example 2: Streaming Text Generation**

```python
from gemini_text_client import GeminiClient, ModelInput

client = GeminiClient()
model_input = ModelInput(user_prompt="Write a short story about a robot who discovers music.")

print("--- Streaming Story ---")
for chunk in client.generate(model_input, stream=True):
    print(chunk, end="", flush=True)
print("\n--- End of Story ---")
```

**Example 3: Robust Structured JSON Generation**

The client automatically uses a two-pass self-correction mechanism for JSON generation, ensuring a high degree of reliability.

```python
from gemini_text_client import GeminiClient, ModelInput
import json

# Define the desired JSON schema
user_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "age": {"type": "integer"}
    },
    "required": ["name", "email"]
}

client = GeminiClient()
model_input = ModelInput(
    user_prompt="Extract the user details from this sentence: 'My name is Jane Doe, I am 32 years old, and my email is jane.doe@example.com.'",
    response_schema=user_schema
)

# The generate method automatically routes to the robust generate_json method
structured_data = client.generate(model_input)

print(json.dumps(structured_data, indent=2))
```

### As a Command-Line Tool

The script provides a powerful CLI for direct interaction.

**Example 1: Simple Question**

```bash
python gemini_text_client.py -q "What are the main benefits of using Python?"
```

**Example 2: Using a Different Model and a System Prompt**

```bash
python gemini_text_client.py \
    -q "Translate 'hello world' to French." \
    -m "gemini-1.5-pro" \
    -s "You are a helpful translation assistant."
```

**Example 3: Requesting Structured JSON Output**

You can provide the schema as a JSON string or a file path.

```bash
# Schema as a string
python gemini_text_client.py \
    -q "Create a JSON object for a user named John Doe." \
    -r '{"type": "object", "properties": {"name": {"type": "string"}}}'

# Schema from a file
# echo '{"type": "object", "properties": {"city": {"type": "string"}, "country": {"type": "string"}}}' > schema.json
python gemini_text_client.py \
    -q "Where is the Eiffel Tower located?" \
    -r schema.json
```

**Example 4: Streaming a Response**

```bash
python gemini_text_client.py -q "Tell me a long story about space exploration." --stream
```

## CLI Arguments

```
usage: gemini_text_client.py [-h] -q QUESTION [--api-key API_KEY] [-m MODEL] [-s SYS_PROMPT] [-a ASSIST_PROMPT] [-r RESPONSE_SCHEMA] [--stream] [--save-lmdb] [--lmdb-path LMDB_PATH] [-t TEMPERATURE] [--top-p TOP_P] [--top-k TOP_K] [--max-tokens MAX_OUTPUT_TOKENS] [--thinking-budget THINKING_BUDGET]

CLI tool to query the Gemini API. Set GEMINI_API_KEY environment variable as a fallback.

options:
  -h, --help            show this help message and exit
  -q QUESTION, --question QUESTION
                        The question or prompt to send to the Gemini model.
  --api-key API_KEY     Your Gemini API key. If not provided, defaults to GEMINI_API_KEY environment variable.
  -m MODEL, --model MODEL
                        The Gemini model to use. Available options: gemini-1.5-flash, gemini-1.5-pro, gemini-1.5-flash-lite
  -s SYS_PROMPT, --system-prompt SYS_PROMPT
                        Optional system prompt to set the model's persona or behavior.
  -a ASSIST_PROMPT, --assistant-prompt ASSIST_PROMPT
                        Optional assistant message (role='model') for few-shot examples or context.
  -r RESPONSE_SCHEMA, --response-schema RESPONSE_SCHEMA
                        Optional JSON string or file path containing the response schema for structured output.
  --stream              Stream the response from the model.
  --save-lmdb           Save the conversation (question and response) to LMDB storage. Disabled for streaming.
  --lmdb-path LMDB_PATH
                        Path to the LMDB database file. Default: geminiqa.lmdb
  -t TEMPERATURE, --temperature TEMPERATURE
                        Temperature for sampling (0.0-1.0). Lower = deterministic, higher = creative. Default: model default
  --top-p TOP_P         Top-p (nucleus) sampling (0.0-1.0). Controls diversity. Default: model default
  --top-k TOP_K         Top-k sampling. Limits token selection to top K. Default: model default
  --max-tokens MAX_OUTPUT_TOKENS
                        Maximum number of tokens in the response. Default: model default
  --thinking-budget THINKING_BUDGET
                        Set the thinking budget for the model.


## Architectural Overview

The client is designed around three core classes:

-   **`ModelConfig`**: A dataclass that holds configuration related to the model and generation parameters (e.g., `model_name`, `temperature`, `timeout`).
-   **`ModelInput`**: A dataclass that structures the inputs for a generation request (e.g., `user_prompt`, `sys_prompt`, `response_schema`).
-   **`GeminiClient`**: The main class that orchestrates the API calls. It handles authentication, rate limiting, retries, and routing requests to the appropriate generation methods.

The robust JSON generation works via a two-pass mechanism within the `generate_json` method. The first pass attempts to get a structured response directly. If this fails (e.g., due to malformed JSON), a second call is made. This second call provides the model with the original prompt and schema again, along with a corrective instruction to fix the output, ensuring a much higher success rate.
