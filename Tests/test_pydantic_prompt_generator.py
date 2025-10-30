import unittest
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

from gemkit.pydantic_prompt_generator import PydanticPromptGenerator, PromptStyle

# --- Test Models ---

class Address(BaseModel):
    """Represents a physical address."""
    street: str = Field(..., min_length=1, description="Street address")
    city: str = Field(..., min_length=1, description="City name")

class ContactMethod(str, Enum):
    """Preferred contact methods."""
    EMAIL = "email"
    PHONE = "phone"

class UserProfile(BaseModel):
    """Represents a user profile."""
    user_id: int = Field(..., gt=0, description="Unique user identifier")
    name: str = Field(..., min_length=2, max_length=50, description="Full name")
    addresses: List[Address] = Field(..., min_length=1, description="List of user addresses")
    preferred_contact: Optional[ContactMethod] = Field(default=ContactMethod.EMAIL)

class TestPydanticPromptGenerator(unittest.TestCase):
    """
    Unit tests for the PydanticPromptGenerator class.
    """

    def test_detailed_prompt_generation(self):
        """
        Test the generation of a detailed prompt.
        """
        generator = PydanticPromptGenerator(UserProfile, style=PromptStyle.DETAILED)
        prompt = generator.generate_prompt()
        self.assertIn("Root Object:", prompt)
        self.assertIn("Properties:", prompt)
        self.assertIn("Nested Object Definitions:", prompt)
        self.assertIn("Example JSON Structure:", prompt)

    def test_concise_prompt_generation(self):
        """
        Test the generation of a concise prompt.
        """
        generator = PydanticPromptGenerator(UserProfile, style=PromptStyle.CONCISE)
        prompt = generator.generate_prompt()
        self.assertIn("Return a JSON object for", prompt)
        self.assertNotIn("Properties:", prompt)
        self.assertNotIn("Nested Object Definitions:", prompt)
        self.assertIn("Example JSON Structure:", prompt)

    def test_technical_prompt_generation(self):
        """
        Test the generation of a technical prompt.
        """
        generator = PydanticPromptGenerator(UserProfile, style=PromptStyle.TECHNICAL)
        prompt = generator.generate_prompt()
        self.assertIn("Generate a JSON object conforming to this schema:", prompt)
        self.assertIn("```json", prompt)
        self.assertNotIn("Example JSON Structure:", prompt)

    def test_response_validation_success(self):
        """
        Test successful validation of a JSON response.
        """
        generator = PydanticPromptGenerator(UserProfile)
        valid_data = {
            "user_id": 1,
            "name": "John Doe",
            "addresses": [{"street": "123 Main St", "city": "Anytown"}]
        }
        validated_model = generator.validate_response(valid_data)
        self.assertIsInstance(validated_model, UserProfile)
        self.assertEqual(validated_model.name, "John Doe")

    def test_response_validation_failure(self):
        """
        Test failed validation of a JSON response.
        """
        generator = PydanticPromptGenerator(UserProfile)
        invalid_data = {
            "user_id": 0,  # Fails gt=0 constraint
            "name": "J",    # Fails min_length=2 constraint
            "addresses": [] # Fails min_length=1 constraint
        }
        with self.assertRaises(Exception):
            generator.validate_response(invalid_data)

if __name__ == '__main__':
    unittest.main()