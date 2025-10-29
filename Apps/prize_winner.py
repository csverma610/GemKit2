
import argparse
from pydantic import BaseModel, Field
from typing import List, Optional

from gemini_client import GeminiClient, ModelInput

class Affiliation(BaseModel):
    name: str
    city: str
    country: str

class Prize(BaseModel):
    year: str
    category: str
    share: str
    motivation: str
    affiliations: List[Affiliation]

class Laureate(BaseModel):
    id: str
    firstname: str
    surname: Optional[str] = None
    born: str
    died: str
    born_country: str = Field(..., alias='bornCountry')
    born_country_code: str = Field(..., alias='bornCountryCode')
    born_city: str = Field(..., alias='bornCity')
    gender: str
    prizes: List[Prize]

class LaureatesResponse(BaseModel):
    laureates: List[Laureate]

def get_nobel_prize_winners(year: Optional[int] = None, category: Optional[str] = None) -> LaureatesResponse:
    """
    Fetches Nobel Prize winners using the GeminiClient.
    """
    prompt = f"Get the Nobel Prize winners for the year {year} in the category {category}."
    prompt += " Please ensure you provide all details for each laureate, including their ID, names, birth date, death date, gender, and full birth location (city, country, and country code)."
    prompt += " For each prize they won, include the year, category, share, motivation, and any affiliations."

    gemini_client = GeminiClient()
    model_input = ModelInput(
        user_prompt=prompt,
        response_schema=LaureatesResponse
    )
    
    # The client will return a validated LaureatesResponse object directly
    return gemini_client.generate_content(model_input)


def get_impact_summary(laureate_name: str, motivation: str) -> str:
    """
    Generates a summary of the laureate's work and its impact using the GeminiClient.
    """
    prompt = f"Provide a brief, easy-to-understand summary of the major work done by {laureate_name}, who won the Nobel Prize for the following reason: '{motivation}'. Explain how their work has changed our daily lives."
    
    gemini_client = GeminiClient()
    model_input = ModelInput(user_prompt=prompt)
    
    return gemini_client.generate_content(model_input)

def display_winners(data: LaureatesResponse):
    """
    Displays the Nobel Prize winners in a readable format, including a summary of their work's impact.
    """
    for laureate in data.laureates:
        print(f"Name: {laureate.firstname} {laureate.surname or ''}")
        for prize in laureate.prizes:
            print(f"  Year: {prize.year}")
            print(f"  Category: {prize.category}")
            print(f"  Motivation: {prize.motivation}")
            
            # Generate and print the impact summary
            summary = get_impact_summary(f"{laureate.firstname} {laureate.surname or ''}", prize.motivation)
            print(f"\n  Impact Summary:\n  {summary}\n")
            
        print("-" * 20)

def main():
    """
    Main function to parse arguments and fetch/display Nobel Prize winners.
    """
    parser = argparse.ArgumentParser(description="Get information on Nobel Prize winners.")
    parser.add_argument("--year", type=int, help="Year of the prize.")
    parser.add_argument("--category", type=str, help="Category of the prize (e.g., physics, chemistry).")
    args = parser.parse_args()

    try:
        winners_data = get_nobel_prize_winners(args.year, args.category)
        display_winners(winners_data)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
