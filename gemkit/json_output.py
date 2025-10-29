import json
from google import genai
from pydantic import BaseModel

class Recipe(BaseModel):
    recipe_name: str
    ingredients: list[str]

def save(recipes, filename: str = "output.json"):
    """
    Correctly converts the list of Pydantic Recipe objects to a single
    formatted JSON string and prints/saves it using a concise method.
    """
    if not recipes:
        print("No recipes to save.")
        return

    # Shorter Method: Use list comprehension directly within json.dumps()
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

client = genai.Client()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="List a few popular cookie recipes, and include the amounts of ingredients.",
    config={
        "response_mime_type": "application/json",
        "response_schema": list[Recipe],
    },
)
# Use instantiated objects.
my_recipes: list[Recipe] = response.parsed

save(my_recipes)

