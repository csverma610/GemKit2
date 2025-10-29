import json
import os
from google import genai
from google.genai import types
from typing import Dict, Any, Optional, List

# --- Configuration ---
# NOTE: In a Canvas environment, the API key is automatically handled by the system
# when using the empty string. For a standalone environment, you would insert your key here.
# The SDK automatically looks for the GEMINI_API_KEY environment variable if not provided here.
API_KEY = os.getenv("GEMINI_API_KEY")
API_MODEL = 'gemini-2.5-flash-preview-05-20'

# Initialize the Gemini Client
# If API_KEY is an empty string, the client will attempt to use the environment variable.
try:
    client = genai.Client(api_key=API_KEY)
except Exception as e:
    print(f"Warning: Could not initialize Gemini Client. Ensure your API key is configured. Error: {e}")
    client = None

def fetch_gemini_grounded_response(prompt: str) -> Optional[types.GenerateContentResponse]:
    """
    Calls the Gemini API with Google Search grounding enabled using the Python SDK.

    Args:
        prompt (str): The user's query.

    Returns:
        Optional[types.GenerateContentResponse]: The API response object, or None on failure.
    """
    if not client:
        print("[ERROR] Gemini Client is not initialized.")
        return None

    # Define the system instruction for the model's behavior
    # This text is now passed directly into the config object below.
    system_prompt_text = (
        "You are a helpful, factual research assistant. Use the available tools "
        "to find up-to-date information and provide a concise, single-paragraph answer. "
        "If sources are available, you must cite them."
    )
    
    # Define the Google Search tool using the SDK's type structure
    # This is equivalent to setting "tools": [{"google_search": {}}] in the JSON payload.
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )
    
    # --- FIX: Moved system_instruction into GenerateContentConfig ---
    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        system_instruction=system_prompt_text  # Added system instruction here to satisfy older SDK
    )
    
    try:
        print(f"Sending request to Gemini API using SDK...")
        # Call the SDK method
        # Removed the 'system_instruction' keyword argument to resolve the error.
        response = client.models.generate_content(
            model=API_MODEL,
            contents=prompt,
            config=config,
        )
        return response

    except Exception as e:
        print(f"Error during API call with SDK: {e}")
        return None

def process_response(response: Optional[types.GenerateContentResponse]):
    """Processes the Gemini SDK response to extract text and citations."""
    if not response or not response.candidates:
        print("\n[ERROR] No valid response or candidates found.")
        return

    candidate = response.candidates[0]
    
    # Extract the generated text
    generated_text = candidate.content.parts[0].text if candidate.content and candidate.content.parts else 'No text generated.'
    print("\n--- Gemini Grounded Response ---")
    print(generated_text)
    print("------------------------------")

    # Extract grounding metadata (citations)
    grounding_metadata = candidate.grounding_metadata
    
    if grounding_metadata and grounding_metadata.grounding_attributions:
        sources: List[Dict[str, str]] = []
        for attribution in grounding_metadata.grounding_attributions:
            web_info = attribution.web
            if web_info and web_info.uri and web_info.title:
                sources.append({
                    'title': web_info.title,
                    'uri': web_info.uri
                })
        
        if sources:
            print("\n--- Citation Sources (Tools Used) ---")
            for i, source in enumerate(sources):
                print(f"[{i + 1}] {source['title']}")
                print(f"    Link: {source['uri']}")
            print("-------------------------------------")
        else:
            print("\n[INFO] Response was generated but no specific citations were provided.")
    else:
        print("\n[INFO] No grounding metadata (citations) found in the response.")


# --- Main Execution ---
if __name__ == "__main__":
    
    # Example query that requires up-to-date, real-world information
    user_query = "What is the biggest science news story of the week?"
    print(f"User Query: {user_query}")

    # Call the API function
    api_response = fetch_gemini_grounded_response(user_query)

    # Process and display the results
    process_response(api_response)

