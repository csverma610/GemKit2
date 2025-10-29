import unittest
from pydantic import BaseModel, Field
from typing import List
from pydantic_prompt import PydanticPromptGenerator

class TestPydanticPromptGenerator(unittest.TestCase):

    def test_simple_model(self):
        class SimpleModel(BaseModel):
            name: str
            age: int

        prompt_generator = PydanticPromptGenerator(SimpleModel)
        prompt = prompt_generator.generate_prompt()
        self.assertIn("`name` (string): **(Required)**", prompt)
        self.assertIn("`age` (integer): **(Required)**", prompt)

    def test_model_with_description(self):
        class ModelWithDescription(BaseModel):
            """This is a test model."""
            name: str

        prompt_generator = PydanticPromptGenerator(ModelWithDescription)
        prompt = prompt_generator.generate_prompt()
        self.assertIn("*This is a test model.*", prompt)

    def test_model_with_validation(self):
        class ModelWithValidation(BaseModel):
            name: str = Field(..., min_length=2, max_length=10)
            age: int = Field(..., gt=0, lt=100)
            user_type: str = Field("reader", enum=["reader", "writer"])

        prompt_generator = PydanticPromptGenerator(ModelWithValidation)
        prompt = prompt_generator.generate_prompt()
        self.assertIn("Minimum length: 2. Maximum length: 10.", prompt)
        self.assertIn("Must be greater than: 0. ", prompt)
        self.assertIn("Must be less than: 100. ", prompt)
        self.assertIn("Allowed values: ['reader', 'writer'].", prompt)

    def test_nested_model(self):
        class NestedModel(BaseModel):
            street: str
            city: str

        class MainModel(BaseModel):
            name: str
            address: NestedModel

        prompt_generator = PydanticPromptGenerator(MainModel)
        prompt = prompt_generator.generate_prompt()
        self.assertIn("`address` (object):", prompt)
        self.assertIn('"street": "example_street"', prompt)
        self.assertIn('"city": "example_city"', prompt)

    def test_model_with_list(self):
        class Item(BaseModel):
            name: str
            price: float

        class Order(BaseModel):
            order_id: int
            items: List[Item]

        prompt_generator = PydanticPromptGenerator(Order)
        prompt = prompt_generator.generate_prompt()
        self.assertIn("`items` (array): A list of `Item` objects. **(Required)**", prompt)
        self.assertIn('"name": "example_name"', prompt)
        self.assertIn('"price": 123.45', prompt)

if __name__ == '__main__':
    unittest.main()
