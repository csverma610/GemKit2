"""
A script to demonstrate how to generate structured JSON output from the
Gemini API using Pydantic models.

This script defines a `Recipe` Pydantic model, sends a prompt to the Gemini
model to generate a list of recipes, and then saves the structured output
to a JSON file.

Usage:
    python json_output.py
"""

import json
from google import genai
from pydantic import BaseModel

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

def save(recipes, filename: str = "output.json"):
    """
    Saves a list of Recipe objects to a JSON file.
    """
    if not recipes:
        print("No recipes to save.")
        return

    json_output = json.dumps(
        [recipe.model_dump() for recipe in recipes],
        indent=2
    )

    print(f"\n--- Saving All Recipes to {filename} (JSON Output) ---")
    print(json_output)

    try:
        with open(filename, 'w') as f:
            f.write(json_output)
        print(f"\nSuccessfully wrote recipes to {filename}.")
    except Exception as e:
        print(f"Failed to write file: {e}")

def main():
    """
    The main function for the JSON output example.
    """
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="List a few popular cookie recipes, and include the amounts of ingredients.",
        config={
            "response_mime_type": "application/json",
            "response_schema": list[Recipe],
        },
    )
    my_recipes: list[Recipe] = response.parsed

    save(my_recipes)

if __name__ == "__main__":
    main()

