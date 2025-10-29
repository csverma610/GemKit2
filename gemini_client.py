# Standard library imports
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Any, Union, Iterable, List

# Third-party imports
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel

# New Imports for Multimodal Support
import mimetypes
from pathlib import Path 

# Configure logging to a file
LOG_FILE = 'gemini_client.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='a' # Append to the log file
)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """
    Configuration dataclass for GeminiClient initialization.
    Defines generation parameters.
    """
    model_name: str = 'gemini-2.5-flash'
    max_retries: int = 3
    initial_delay: float = 1.0
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None
    timeout: float = 120.0 

@dataclass
class ModelInput:
    """
    Dataclass to structure user input parameters.

    Attributes:
        user_prompt (str): The user's question or prompt (required).
        sys_prompt (str): Optional system prompt to set behavior or context.
        assist_prompt (str): Optional assistant message for few-shot examples or context.
        response_schema (Optional[Any]): Pydantic model class or raw JSON schema dict for structured output.
        images (Optional[List[Union[str, Path]]]): List of file paths pointing to images (JPG, PNG, etc.).
    """
    user_prompt: str
    sys_prompt: str = ""
    assist_prompt: str = ""
    response_schema: Optional[Any] = None
    images: Optional[List[Union[str, Path]]] = None # New image list attribute

class GeminiClient:
    """
    A core client to interact with the Gemini API, featuring retries on network and throttling errors.
    """
    MODELS_NAME = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.5-flash-lite']

    def __init__(self, config: ModelConfig = None, api_key: Optional[str] = None):
        """Initialize GeminiClient with configuration and API key."""
        if config is None:
            config = ModelConfig()

        if config.model_name not in self.MODELS_NAME:
            raise ValueError(f"Invalid model name '{config.model_name}'.")

        self.config = config
        self.model_name = config.model_name
        self.max_retries = config.max_retries
        self.initial_delay = config.initial_delay
        # Proactive rate limit tracking variables removed for simplicity.


        effective_api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not effective_api_key:
            raise ValueError("API key must be provided or set in GEMINI_API_KEY environment variable.")
            
        try:
            # Pass timeout explicitly to the client initialization
            self.client = genai.Client(
                api_key=effective_api_key
            )
            logger.info("GeminiClient initialized successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the Gemini client: {e}") from e


    def _prepare_config(self, model_input: ModelInput, for_structured: bool = False) -> Optional[genai.types.GenerateContentConfig]:
        """Prepares the generation configuration object."""
        config_params = {}

        if model_input.sys_prompt:
            config_params['system_instruction'] = model_input.sys_prompt
        
        # Add generation parameters from config
        if self.config.temperature is not None: config_params['temperature'] = self.config.temperature
        if self.config.top_p is not None: config_params['top_p'] = self.config.top_p
        if self.config.top_k is not None: config_params['top_k'] = self.config.top_k
        if self.config.max_output_tokens is not None: config_params['max_output_tokens'] = self.config.max_output_tokens
        
        # Handle thinking budget
        if self.config.thinking_budget is not None:
            config_params['thinking'] = self.config.thinking_budget != 0
            if self.config.thinking_budget != 0:
                config_params['thinking_budget'] = self.config.thinking_budget
            
        # For structured output
        if for_structured:
            # response_schema can be a Pydantic model or a raw dict. 
            # The genai client handles the conversion to JSON Schema.
            config_params['response_schema'] = model_input.response_schema
            config_params['response_mime_type'] = 'application/json'

        if config_params:
            return genai.types.GenerateContentConfig(**config_params)
        return None

    def _build_api_payload(self, model_input: ModelInput, for_structured: bool = False) -> tuple[list[genai.types.Content], Optional[genai.types.GenerateContentConfig]]:
        """Constructs the contents list and configuration object."""
        # --- 1. Basic Validation (Text is still mandatory for simplicity) ---
        if not model_input.user_prompt or not model_input.user_prompt.strip():
            # If no images are provided either, this is an empty request
            if not model_input.images:
                raise ValueError("user_prompt cannot be empty unless images are provided.")
            
        contents = []
        config = self._prepare_config(model_input, for_structured=for_structured)

        # Handle Assistant Prompt (for few-shot examples)
        if model_input.assist_prompt:
            contents.append(
                genai.types.Content(
                    role="model",
                    parts=[genai.types.Part(text=model_input.assist_prompt)]
                )
            )

        # --- 2. Handle Image Parts (New Complexity) ---
        user_parts = []
        if model_input.images:
            for image_path in model_input.images:
                try:
                    p = Path(image_path)
                    mime_type, _ = mimetypes.guess_type(p)
                    
                    if not mime_type or not mime_type.startswith('image/'):
                        logger.warning(f"Skipping file {p}: could not determine valid image MIME type.")
                        continue

                    # Read binary data and append as inline data part
                    image_data = p.read_bytes()
                    user_parts.append(
                        genai.types.Part.from_bytes(data=image_data, mime_type=mime_type)
                    )
                    logger.debug(f"Attached image file: {p} ({mime_type})")
                    
                except FileNotFoundError:
                    logger.error(f"Image file not found: {image_path}")
                    raise
                except Exception as e:
                    logger.error(f"Error processing image file {image_path}: {e}")
                    raise

        # --- 3. Handle User Text Prompt ---
        if model_input.user_prompt and model_input.user_prompt.strip():
            user_parts.append(genai.types.Part(text=model_input.user_prompt))

        # --- 4. Final Payload Construction ---
        if not user_parts:
            # Catch case where text was empty and images failed/not provided
             raise ValueError("API payload cannot be empty. Must include text or image data.")

        contents.append(
            genai.types.Content(
                role="user",
                parts=user_parts
            )
        )

        return contents, config

    def _clean_and_parse_json(self, raw_text: str) -> Union[dict, list]:
        """
        Cleans the raw text output by removing markdown wrappers and attempts JSON parsing.
        
        Raises:
            json.JSONDecodeError: If the cleaned text is not valid JSON.
        """
        cleaned_text = raw_text.strip()
        
        # Remove markdown wrappers (e.g., ```json or ```)
        if cleaned_text.startswith("```json"):
            # Assuming minimal text outside the block
            cleaned_text = cleaned_text[7:].strip().rstrip("`")
        elif cleaned_text.startswith("```"):
            # For generic code fences
            cleaned_text = cleaned_text[3:].strip().rstrip("`")
            
        # This will raise JSONDecodeError if parsing fails
        return json.loads(cleaned_text)


    def _generate_json(self, model_input: ModelInput, max_correction_retries: int = 3) -> BaseModel:
        """
        Generates structured JSON data, returning a validated Pydantic model instance.
        Features a multi-pass self-correction mechanism if the native JSON mode fails.
        """
        if not model_input.response_schema:
            raise ValueError("response_schema must be provided for JSON generation")

        # --- First Pass: Native Structured Output Attempt ---
        try:
            logger.info("Attempting structured JSON generation (1st pass - Native)")
            
            contents, config = self._build_api_payload(model_input, for_structured=True)
            
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            
            if response.parsed is not None:
                logger.info("Native structured generation succeeded.")
                # The API can return the Pydantic object directly.
                return response.parsed

            logger.warning("Native structured generation returned non-parsable output. Initiating self-correction.")

        except (genai.errors.APIError, json.JSONDecodeError) as e:
            logger.warning(f"Initial structured generation failed: {type(e).__name__}. Initiating self-correction.")
            pass
        
        # --- Subsequent Passes: Self-Correction Loop ---
        last_error = None
        for i in range(max_correction_retries):
            logger.info(f"Attempting self-correction pass {i + 1}/{max_correction_retries}")
            
            correction_prompt = (
                f"The previous attempt to generate JSON failed. Please strictly adhere to the provided schema "
                f"and provide ONLY a single, valid JSON object as a raw string. Do not include any markdown "
                f"wrappers (like ```json) or explanatory text.\n\n"
                f"Original User Prompt: {model_input.user_prompt}\n\n"
                f"JSON Schema (Reference): {json.dumps(model_input.response_schema.model_json_schema(), indent=2)}"
            )
            
            correction_input = ModelInput(
                user_prompt=correction_prompt,
                sys_prompt=model_input.sys_prompt,
                response_schema=None, # Force raw text output
                images=None
            )

            try:
                corrected_text = self.generate_content(correction_input, stream=False)
                parsed_dict = self._clean_and_parse_json(corrected_text)
                return model_input.response_schema(**parsed_dict)
            
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Self-correction pass {i + 1} failed to produce valid JSON: {e}")
                last_error = e
                time.sleep(self.config.initial_delay * (i + 1))
                continue
            
        raise ValueError(f"Failed to generate valid JSON after {max_correction_retries} self-correction retries. Last error: {last_error}")


    @retry(
        # Retry on network errors and resource exhaustion
        retry=retry_if_exception_type((ConnectionError, TimeoutError, genai.errors.APIError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    def generate_content(self, model_input: ModelInput, stream: bool = False) -> Union[str, dict, list, Any, Iterable[str]]:
        """
        Unified method for text, structured JSON, and streaming generation.

        Dispatches to _generate_json if a response_schema is present.
        Otherwise, performs text/streaming generation.
        """
        for_structured = model_input.response_schema is not None
        
        if for_structured:
            if stream:
                logger.warning("Streaming is not supported for structured output. Running non-streamed JSON generation with correction logic.")
            # Dispatch to dedicated function for JSON generation with self-correction
            return self._generate_json(model_input)
        
        # --- Non-Structured (Text/Streaming) Flow ---

        logger.debug(f"Generating content (structured=False, stream={stream})")

        contents, config = self._build_api_payload(model_input, for_structured=False)

        try:
            if stream:
                # Streaming Text Generation
                response_stream = self.client.models.generate_content_stream(
                    model=self.model_name, contents=contents, config=config
                )
                
                def stream_iterator(stream_response):
                    for chunk in stream_response:
                        if chunk.text:
                            yield chunk.text
                return stream_iterator(response_stream)
            else:
                # Single Text Generation
                response = self.client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
                return response.text

        except (genai.errors.APIError, ConnectionError, TimeoutError) as e:
            # Note: JSONDecodeError is handled in _generate_json
            logger.error(f"Text generation failed after retries: {e}")
            raise

