"""
A script to demonstrate function calling with the Gemini API.

This script defines a function `get_current_temperature` and a corresponding
function declaration for the Gemini model. It then sends a prompt to the model
and checks if the model responds with a function call. If it does, it prints
the function name and arguments.

Usage:
    python functiona_calling.py "City Name"
"""

from google import genai
from google.genai import types
import sys

def get_current_temperature(location):
    """
    A dummy function to get the current temperature for a given location.
    """
    if location == "Ajmer":
       return "26"
    return "10"

def main():
    """
    The main function for the function calling example.
    """
    if len(sys.argv) < 2:
        print("Usage: python functiona_calling.py \"City Name\"")
        sys.exit(1)

    # Define the function declaration for the model
    weather_function = {
        "name": "get_current_temperature",
        "description": "Gets the current temperature for a given location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name, e.g. San Francisco",
                },
            },
            "required": ["location"],
        },
    }

    # Configure the client and tools
    client = genai.Client()
    tools = types.Tool(function_declarations=[weather_function])
    config = types.GenerateContentConfig(tools=[tools])

    # Send request with function declarations
    city = sys.argv[1]
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"What's the temperature in {city}?",
        config=config,
    )

    # Check for a function call
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"Function to call: {function_call.name}")
        print(f"Arguments: {function_call.args}")
        #  In a real app, you would call your function here:
        result = get_current_temperature(**function_call.args)
        print(result)
    else:
        print("No function call found in the response.")
        print(response.text)

if __name__ == "__main__":
    main()
