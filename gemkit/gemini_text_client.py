# Standard library imports
import argparse
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Union, Callable, List, Iterable

# Third-party imports
from google import genai
from google.genai import types

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Local imports
from lmdb_storage import LMDBStorage

# Configure logging
logger = logging.getLogger(__name__)

class JsonFormatter(logging.Formatter):
    """
    A logging formatter that outputs log records as JSON strings.
    """
    def format(self, record):
        log_object = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_object['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_object)


@dataclass
class ModelConfig:
    """
    Configuration for the GeminiClient, specifying model and generation parameters.

    Attributes:
        model_name (str): The name of the Gemini model to use.
        max_retries (int): The maximum number of times to retry a failed API call.
        initial_delay (float): The initial delay in seconds for exponential backoff.
        timeout (float): The timeout for API requests in seconds.
        rate_limit_calls (int): The maximum number of API calls allowed per `rate_limit_period`.
        rate_limit_period (float): The time period in seconds for rate limiting.
        temperature (Optional[float]): Controls the randomness of the output.
        top_p (Optional[float]): The nucleus sampling probability.
        top_k (Optional[int]): The top-k sampling parameter.
        max_output_tokens (Optional[int]): The maximum number of tokens to generate.
        thinking_budget (Optional[int]): The budget for the model's thinking time.
    """
    model_name: str = 'gemini-2.5-flash'
    max_retries: int = 3
    initial_delay: float = 1.0
    timeout: float = 120.0
    rate_limit_calls: int = 60
    rate_limit_period: float = 60.0
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None

@dataclass
class ModelInput:
    """
    Structures the input for a Gemini model request.

    Attributes:
        user_prompt (str): The primary text prompt from the user.
        sys_prompt (str): Optional system-level instructions to guide the model's behavior.
        assist_prompt (str): Optional assistant message for providing few-shot examples or context.
        response_schema (Optional[Any]): A Pydantic model or a JSON schema dictionary for structured output.
    """
    user_prompt: str
    sys_prompt: str = ""
    assist_prompt: str = ""
    response_schema: Optional[Any] = None

class GeminiClient:
    """
    A client for interacting with the Google Gemini API for text generation.

    This class provides a high-level interface for sending prompts to the Gemini
    API, with support for structured output, streaming, and rate limiting.
    """
    MODELS_NAME = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite']

    def __init__(self, config: ModelConfig = None, api_key: Optional[str] = None):
        """
        Initializes the GeminiClient.

        Args:
            config (ModelConfig, optional): A configuration object for the client.
            api_key (Optional[str], optional): The API key for authentication. If not provided,
                                             it's read from the GEMINI_API_KEY environment variable.
        """
        # Use default config if none provided
        if config is None:
            config = ModelConfig()

        # Validate model name
        if config.model_name not in self.MODELS_NAME:
            logger.error(f"Invalid model name '{config.model_name}'. Must be one of: {', '.join(self.MODELS_NAME)}")
            raise ValueError(
                f"Invalid model name '{config.model_name}'. "
                f"Must be one of: {', '.join(self.MODELS_NAME)}"
            )

        # Store configuration
        self.config = config
        self.model_name = config.model_name
        self.max_retries = config.max_retries
        self.initial_delay = config.initial_delay
        self.timeout = config.timeout
        self.rate_limit_calls = config.rate_limit_calls
        self.rate_limit_period = config.rate_limit_period
        self._call_times = deque()  # Track API call timestamps for rate limiting

        logger.info(
            f"Initializing GeminiClient with model: {config.model_name}, "
            f"timeout: {config.timeout}s, "
            f"rate limit: {config.rate_limit_calls} calls/{config.rate_limit_period}s"
        )

        # Prioritize the provided API key, then fallback to environment variable
        effective_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not effective_api_key:
            logger.error("API key not provided and GEMINI_API_KEY environment variable not set.")
            raise ValueError("API key must be provided either directly or through the GEMINI_API_KEY environment variable.")
        
        try:
            self.client = genai.Client(api_key=effective_api_key)
            logger.info("GeminiClient initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize genai.Client: {e}")
            raise RuntimeError(f"Failed to initialize the Gemini client, please check your API key and environment setup. Original error: {e}") from e

    def _enforce_rate_limit(self) -> None:
        """
        Enforces rate limiting by tracking API call timestamps.
        Blocks if the rate limit would be exceeded.
        """
        current_time = time.time()

        # Remove timestamps older than the rate limit period
        while self._call_times and current_time - self._call_times[0] >= self.rate_limit_period:
            self._call_times.popleft()

        # If we've hit the rate limit, wait until we can make another call
        if len(self._call_times) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (current_time - self._call_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Waiting {sleep_time:.2f}s...")
                time.sleep(sleep_time)
                # Clean up old timestamps after waiting
                current_time = time.time()
                while self._call_times and current_time - self._call_times[0] >= self.rate_limit_period:
                    self._call_times.popleft()

        # Record this API call
        self._call_times.append(time.time())

    def _prepare_generation_config(self, model_input: ModelInput, for_structured: bool = False) -> Optional[genai.types.GenerateContentConfig]:
        """Prepares the generation configuration object."""
        config_params = {}

        # Add system instruction if provided
        if model_input.sys_prompt:
            config_params['system_instruction'] = model_input.sys_prompt

        # Add generation parameters from config if provided
        if self.config.temperature is not None:
            config_params['temperature'] = self.config.temperature
        if self.config.top_p is not None:
            config_params['top_p'] = self.config.top_p
        if self.config.top_k is not None:
            config_params['top_k'] = self.config.top_k
        if self.config.max_output_tokens is not None:
            config_params['max_output_tokens'] = self.config.max_output_tokens

        # Handle thinking_budget logic
        if self.config.thinking_budget is not None:
            config_params['thinking'] = self.config.thinking_budget != 0
            if self.config.thinking_budget != 0:
                config_params['thinking_budget'] = self.config.thinking_budget
        
        # For structured output
        if for_structured:
            config_params['response_schema'] = model_input.response_schema
            config_params['response_mime_type'] = 'application/json'

        # Create config if any parameters were set
        if config_params:
            return genai.types.GenerateContentConfig(**config_params)
        return None

    def _build_api_payload(self, model_input: ModelInput, for_structured: bool = False) -> tuple[list[genai.types.Content], Optional[genai.types.GenerateContentConfig]]:
        """
        Helper function to construct the contents list and the base configuration (system instruction).

        Args:
            model_input (ModelInput): Dataclass containing all prompt and configuration settings.
            for_structured (bool): If True, prepares the config for structured JSON output.

        Returns:
            tuple[list[types.Content], Optional[types.GenerateContentConfig]]:
                The list of contents for the API call and the configuration object.

        Raises:
            ValueError: If user_prompt is empty or contains only whitespace.
        """
        # Validate user prompt
        if not model_input.user_prompt or not model_input.user_prompt.strip():
            logger.error("Empty or whitespace-only user_prompt provided")
            raise ValueError("user_prompt cannot be empty or contain only whitespace")

        contents = []
        config = self._prepare_generation_config(model_input, for_structured=for_structured)

        # 2. Handle Assistant Prompt (Contents role="model")
        if model_input.assist_prompt:
            contents.append(
                genai.types.Content(
                    role="model",
                    parts=[genai.types.Part(text=model_input.assist_prompt)]
                )
            )

        # 3. Handle User Prompt (Contents role="user")
        user_parts = [genai.types.Part(text=model_input.user_prompt)]

        contents.append(
            genai.types.Content(
                role="user",
                parts=user_parts
            )
        )

        return contents, config

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def generate_text(self, model_input: ModelInput, stream: bool = False) -> Union[str, Iterable[str]]:
        """
        Generates a text response from the Gemini model.

        Args:
            model_input (ModelInput): The input for the model, including the prompt.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            Union[str, Iterable[str]]: The generated text, or an iterable of text chunks
                                       if streaming is enabled.
        """
        logger.debug(f"Generating text with model: {self.model_name}")
        # Enforce rate limiting before API call
        self._enforce_rate_limit()

        # Use the helper function to build shared payload elements
        contents, config = self._build_api_payload(model_input)

        try:
            # API Call with retry logic
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
        except genai.errors.APIError as e:
            logger.error(f"Gemini API Error: {e}")
            raise

        logger.info("Text generation completed successfully")
        return response.text

    def generate_json(self, model_input: ModelInput, max_retries: int = 1) -> Union[dict, list]:
        """
        Generates a structured JSON response from the Gemini model.

        This method uses a two-pass self-correction mechanism. If the first attempt
        to generate a valid JSON object fails, it will make additional attempts
        to ask the model to correct its own output.

        Args:
            model_input (ModelInput): The input for the model, including the response schema.
            max_retries (int, optional): The number of self-correction attempts. Defaults to 1.

        Returns:
            Union[dict, list]: The parsed JSON data.

        Raises:
            ValueError: If a `response_schema` is not provided, or if the model fails
                        to produce a valid JSON object after all retries.
        """
        if not model_input.response_schema:
            raise ValueError("response_schema must be provided for generate_json")

        # First Pass: Attempt to get structured output directly.
        try:
            logger.info("Attempting structured JSON generation (1st pass)")
            self._enforce_rate_limit()

            contents, config = self._build_api_payload(model_input, for_structured=True)

            @retry(
                retry=retry_if_exception_type((ConnectionError, TimeoutError)),
                stop=stop_after_attempt(self.max_retries),
                wait=wait_exponential(multiplier=1, min=self.initial_delay, max=10)
            )
            def _api_call():
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config,
                )

            response = _api_call()
            # Attempt to parse the JSON from the response
            return json.loads(response.text)

        except (genai.errors.APIError, json.JSONDecodeError) as e:
            logger.warning(f"Initial structured generation failed: {e}. The model likely returned malformed JSON or text. Initiating self-correction.")
            pass
        except Exception as e:
            logger.warning(f"An unexpected error occurred during initial generation: {e}. Initiating self-correction.")
            pass

        # Second Pass (Self-Correction): If the first pass fails, ask the model to fix it.
        current_input = model_input
        for i in range(max_retries):
            logger.info(f"Attempting self-correction pass {i + 1}/{max_retries}")
            
            # Construct a new prompt for correction
            correction_prompt = (
                f"The previous attempt to generate JSON failed. Please strictly adhere to the following schema and provide only a single, valid JSON object as a raw string. Do not include any markdown or explanatory text.\n\n"
                f"Original User Prompt: {current_input.user_prompt}\n\n"
                f"JSON Schema: {json.dumps(current_input.response_schema, indent=2)}"
            )
            
            correction_input = ModelInput(
                user_prompt=correction_prompt,
                sys_prompt=current_input.sys_prompt, # Carry over original system prompt
                assist_prompt=current_input.assist_prompt
            )

            # We expect a raw text string this time, not a parsed object
            corrected_text = self.generate_text(correction_input)

            # Attempt to parse the corrected text
            try:
                # Basic cleaning of common markdown wrappers
                if corrected_text.strip().startswith("```json"):
                    cleaned_text = corrected_text.strip()[7:-3].strip()
                else:
                    cleaned_text = corrected_text
                
                return json.loads(cleaned_text)
            except json.JSONDecodeError as e:
                logger.warning(f"Self-correction pass {i + 1} failed to produce valid JSON: {e}")
                # For the next retry, we can optionally feed the bad output back in, but let's keep it simple for now.
                continue
        
        raise ValueError("Failed to generate valid JSON after all self-correction retries.")

    def generate(self, model_input: ModelInput, stream: bool = False) -> Union[str, dict, list, Any, Iterable[str]]:
        """
        A unified method for generating responses from the Gemini model.

        This method intelligently routes the request to the appropriate generation
        method based on the `ModelInput` parameters. If a `response_schema` is
        provided, it will use `generate_json`; otherwise, it will use `generate_text`.

        Args:
            model_input (ModelInput): The input for the model.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            Union[str, dict, list, Any, Iterable[str]]: The generated response, which can be
                                                       a string, a JSON object, or an iterable
                                                       of strings if streaming is enabled.
        """
        logger.debug("Using unified generate() method")

        # Determine which generation method to use
        if model_input.response_schema:
            if stream:
                logger.warning("Streaming is not supported for structured output. Ignoring stream=True.")
            logger.info("Routing to robust JSON generation")
            return self.generate_json(model_input)
        else:
            logger.info("Routing to text generation (includes vision if images provided)")
            return self.generate_text(model_input, stream=stream)

def _parse_response_schema(schema_input: Optional[str]) -> Optional[dict]:
    """Parses the response schema from a JSON string or a file path."""
    if not schema_input:
        return None
    try:
        return json.loads(schema_input)
    except json.JSONDecodeError:
        try:
            with open(schema_input, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Could not parse response schema as JSON or find file: {schema_input}"
            ) from e
        except json.JSONDecodeError as e:
            raise ValueError(
                f"File {schema_input} does not contain valid JSON"
            ) from e

def main():
    # Configure JSON logging for CLI usage
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)

    # Clear existing handlers and add the new one
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    parser = argparse.ArgumentParser(
        description="CLI tool to query the Gemini API. Set GEMINI_API_KEY environment variable as a fallback."
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        required=True,
        help="The question or prompt to send to the Gemini model."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Your Gemini API key. If not provided, defaults to GEMINI_API_KEY environment variable."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=ModelConfig().model_name,
        choices=GeminiClient.MODELS_NAME,
        help=f"The Gemini model to use. Available options: {', '.join(GeminiClient.MODELS_NAME)}"
    )
    parser.add_argument(
        "-s", "--system-prompt",
        type=str,
        default="",
        dest="sys_prompt",
        help="Optional system prompt to set the model's persona or behavior."
    )
    parser.add_argument(
        "-a", "--assistant-prompt",
        type=str,
        default="",
        dest="assist_prompt",
        help="Optional assistant message (role='model') for few-shot examples or context."
    )
    parser.add_argument(
        "-r", "--response-schema",
        type=str,
        default=None,
        dest="response_schema",
        help="Optional JSON string or file path containing the response schema for structured output."
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the response from the model."
    )
    parser.add_argument(
        "--save-lmdb",
        action="store_true",
        help="Save the conversation (question and response) to LMDB storage. Disabled for streaming."
    )
    parser.add_argument(
        "--lmdb-path",
        type=str,
        default="geminiqa.lmdb",
        help="Path to the LMDB database file. Default: geminiqa.lmdb"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling (0.0-2.0). Lower = deterministic, higher = creative. Default: model default"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling (0.0-1.0). Controls diversity. Default: model default"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling. Limits token selection to top K. Default: model default"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        dest="max_output_tokens",
        help="Maximum number of tokens in the response. Default: model default"
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Set the thinking budget for the model."
    )

    args = parser.parse_args()

    # Use parsed arguments
    question = args.question
    model_name = args.model
    sys_prompt = args.sys_prompt
    assist_prompt = args.assist_prompt
    
    try:
        response_schema = _parse_response_schema(args.response_schema)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n[SCHEMA ERROR]: {e}")
        return

    # 1. Create ModelConfig object with generation parameters
    config = ModelConfig(
        model_name=model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget
    )

    # 2. Create ModelInput object (Refactored)
    model_input = ModelInput(
        user_prompt=question,
        sys_prompt=sys_prompt,
        assist_prompt=assist_prompt,
        response_schema=response_schema
    )

    # Create instances of the API client.
    try:
        # Pass the config object and api_key to the client constructor
        client = GeminiClient(config, api_key=args.api_key)

        print(f"\nModel: {config.model_name}")
        if sys_prompt:
            print(f"System Prompt: '{sys_prompt}'")
        if assist_prompt:
            print(f"Assistant Prompt: '{assist_prompt}'")
        if response_schema:
            print(f"Response Schema: {json.dumps(response_schema, indent=2)}")
        print(f"User Query: {question}\n")

        # Get the response from the Gemini API.
        start_time = time.time()

        # Use the unified generate method
        result = client.generate(model_input, stream=args.stream)

        # Format the output
        text_to_save = None
        if args.stream:
            print("\n--- Gemini Text Response (Streaming) ---")
            # We don't accumulate the full response to save memory.
            # LMDB saving is disabled for streaming in the CLI.
            try:
                for chunk in result:
                    print(chunk, end="", flush=True)
                print()  # for newline at the end
            except TypeError:
                print("[ERROR] Could not iterate over the response. Streaming might have failed.")
        elif isinstance(result, (dict, list)):
            text_to_save = json.dumps(result, indent=2)
            print("\n--- Gemini Structured Response (JSON/Python Object) ---")
            print(text_to_save)
        else:
            text_to_save = result
            print("\n--- Gemini Text Response ---")
            print(text_to_save)

        end_time = time.time()

        print(f"\n--- Generation Time: {end_time - start_time:.2f} seconds ---\n")

        # Store the conversation if requested and not streaming
        if args.save_lmdb:
            if args.stream:
                print("[INFO]: LMDB saving is disabled for streaming responses.")
            elif text_to_save is not None:
                try:
                    storage = LMDBStorage(args.lmdb_path)
                    storage.put(question, text_to_save)
                    print(f"[INFO]: Conversation saved to LMDB at '{args.lmdb_path}'")
                except Exception as storage_error:
                    print(f"[WARNING]: Failed to save to LMDB: {storage_error}")

    except (ValueError, RuntimeError) as ve:
        # Configuration issues like missing API key or invalid model name
        print(f"\n[CRITICAL CONFIG ERROR]: {ve}")
    except (ConnectionError, TimeoutError) as e:
        # Network-related issues
        print(f"\n[NETWORK ERROR]: Failed to connect to Gemini API: {e}")
    except Exception as e:
        # Any other runtime issues
        print(f"\n[RUNTIME ERROR]: An unexpected error occurred: {e}")

    finally:
        print("\nExecution finished.")

if __name__ == "__main__":
    main()


