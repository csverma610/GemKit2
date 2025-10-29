"""
Comprehensive unit tests for PydanticPromptGenerator

Run with: pytest test_pydantic_prompt_generator.py -v
"""

import pytest
import json
from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field, ValidationError

from pydantic_prompt_generator import (
    PydanticPromptGenerator,
    PromptStyle,
    SchemaValidationError
)


# Test Models
class SimpleModel(BaseModel):
    """Simple model for basic testing."""
    name: str
    age: int


class Address(BaseModel):
    """Address model for nested testing."""
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    zip_code: str = Field(..., pattern=r"^\d{5}$")


class Priority(str, Enum):
    """Priority enum for testing."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ComplexModel(BaseModel):
    """Complex model with various field types."""
    id: int = Field(..., gt=0, description="Unique identifier")
    name: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: Optional[int] = Field(None, ge=0, le=150)
    score: float = Field(..., ge=0.0, le=100.0, multiple_of=0.5)
    is_active: bool = True
    priority: Priority = Priority.MEDIUM
    tags: List[str] = Field(default_factory=list, max_length=5)
    addresses: List[Address] = Field(..., min_length=1)
    metadata: Optional[dict] = None


class EmptyModel(BaseModel):
    """Model with no fields."""
    pass


# Test Fixtures
@pytest.fixture
def simple_generator():
    """Fixture for simple model generator."""
    return PydanticPromptGenerator(SimpleModel)


@pytest.fixture
def complex_generator():
    """Fixture for complex model generator."""
    return PydanticPromptGenerator(ComplexModel)


# Initialization Tests
class TestInitialization:
    """Test generator initialization."""
    
    def test_valid_initialization(self):
        """Test successful initialization with valid model."""
        gen = PydanticPromptGenerator(SimpleModel)
        assert gen.model == SimpleModel
        assert gen.style == PromptStyle.DETAILED
        assert gen.include_examples is True
        assert gen.schema is not None
    
    def test_initialization_with_custom_style(self):
        """Test initialization with custom prompt style."""
        gen = PydanticPromptGenerator(SimpleModel, style=PromptStyle.CONCISE)
        assert gen.style == PromptStyle.CONCISE
    
    def test_initialization_without_examples(self):
        """Test initialization with examples disabled."""
        gen = PydanticPromptGenerator(SimpleModel, include_examples=False)
        assert gen.include_examples is False
    
    def test_invalid_model_type_instance(self):
        """Test that passing an instance raises TypeError."""
        instance = SimpleModel(name="test", age=25)
        with pytest.raises(TypeError, match="Expected a class"):
            PydanticPromptGenerator(instance)
    
    def test_invalid_model_type_not_basemodel(self):
        """Test that passing a non-BaseModel class raises TypeError."""
        class NotAModel:
            pass
        
        with pytest.raises(TypeError, match="must be a Pydantic BaseModel"):
            PydanticPromptGenerator(NotAModel)
    
    def test_empty_model_initialization(self):
        """Test initialization with empty model (should work but warn)."""
        gen = PydanticPromptGenerator(EmptyModel, validate_schema=False)
        assert gen.model == EmptyModel


# Prompt Generation Tests
class TestPromptGeneration:
    """Test prompt generation functionality."""
    
    def test_detailed_prompt_generation(self, simple_generator):
        """Test detailed prompt style generation."""
        prompt = simple_generator.generate_prompt()
        
        assert "Please provide a JSON object" in prompt
        assert "SimpleModel" in prompt
        assert "name" in prompt
        assert "age" in prompt
        assert "**Properties:**" in prompt
        assert "Example JSON Structure" in prompt
    
    def test_concise_prompt_generation(self):
        """Test concise prompt style generation."""
        gen = PydanticPromptGenerator(SimpleModel, style=PromptStyle.CONCISE)
        prompt = gen.generate_prompt()
        
        assert "Return a JSON object" in prompt
        assert "name:" in prompt
        assert "age:" in prompt
        # Should not have detailed explanations
        assert "**Properties:**" not in prompt
    
    def test_technical_prompt_generation(self):
        """Test technical prompt style generation."""
        gen = PydanticPromptGenerator(SimpleModel, style=PromptStyle.TECHNICAL)
        prompt = gen.generate_prompt()
        
        assert "conforming to this schema" in prompt
        assert "```json" in prompt
        # Should contain actual JSON schema
        assert '"properties"' in prompt or '"title"' in prompt
    
    def test_prompt_without_examples(self):
        """Test prompt generation without examples."""
        gen = PydanticPromptGenerator(SimpleModel, include_examples=False)
        prompt = gen.generate_prompt()
        
        assert "Example JSON Structure" not in prompt
        assert "```json" not in prompt
    
    def test_complex_model_prompt(self, complex_generator):
        """Test prompt generation for complex model."""
        prompt = complex_generator.generate_prompt()
        
        # Check for various field types
        assert "id" in prompt
        assert "email" in prompt
        assert "priority" in prompt
        assert "addresses" in prompt
        
        # Check for constraints
        assert "Minimum" in prompt or "minimum" in prompt
        assert "Maximum" in prompt or "maximum" in prompt
        assert "Pattern" in prompt or "pattern" in prompt
        
        # Check for nested definitions
        assert "Address" in prompt
    
    def test_prompt_includes_descriptions(self, complex_generator):
        """Test that field descriptions are included."""
        prompt = complex_generator.generate_prompt()
        
        assert "Unique identifier" in prompt
    
    def test_prompt_includes_required_markers(self, complex_generator):
        """Test that required fields are marked."""
        prompt = complex_generator.generate_prompt()
        
        assert "Required" in prompt or "required" in prompt
        assert "Optional" in prompt or "optional" in prompt
    
    def test_prompt_includes_enum_values(self, complex_generator):
        """Test that enum values are listed."""
        prompt = complex_generator.generate_prompt()
        
        assert "low" in prompt or "LOW" in prompt
        assert "medium" in prompt or "MEDIUM" in prompt
        assert "high" in prompt or "HIGH" in prompt


# Example Generation Tests
class TestExampleGeneration:
    """Test example JSON generation."""
    
    def test_simple_example_generation(self, simple_generator):
        """Test example generation for simple model."""
        example = simple_generator._generate_example_from_schema(
            simple_generator.schema
        )
        
        assert isinstance(example, dict)
        assert "name" in example
        assert "age" in example
        assert isinstance(example["name"], str)
        assert isinstance(example["age"], int)
    
    def test_complex_example_generation(self, complex_generator):
        """Test example generation for complex model."""
        example = complex_generator._generate_example_from_schema(
            complex_generator.schema
        )
        
        assert isinstance(example, dict)
        assert "id" in example
        assert "email" in example
        assert "addresses" in example
        
        # Check nested list
        assert isinstance(example["addresses"], list)
        assert len(example["addresses"]) > 0
        
        # Check nested object structure
        address = example["addresses"][0]
        assert "street" in address
        assert "city" in address
        assert "zip_code" in address
    
    def test_example_respects_constraints(self, complex_generator):
        """Test that generated examples respect field constraints."""
        example = complex_generator._generate_example_from_schema(
            complex_generator.schema
        )
        
        # Test integer constraints
        assert example["id"] > 0
        
        # Test string length constraints
        assert 2 <= len(example["name"]) <= 50
        
        # Test enum
        assert example["priority"] in ["low", "medium", "high"]
        
        # Test list constraints
        assert len(example["addresses"]) >= 1
    
    def test_example_with_pattern(self):
        """Test example generation respects regex patterns."""
        class PatternModel(BaseModel):
            zip_code: str = Field(..., pattern=r"^\d{5}$")
            email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
        
        gen = PydanticPromptGenerator(PatternModel)
        example = gen._generate_example_from_schema(gen.schema)
        
        # Check zip code pattern
        assert len(example["zip_code"]) == 5
        assert example["zip_code"].isdigit()
        
        # Check email pattern
        assert "@" in example["email"]
        assert "." in example["email"]
    
    def test_example_with_defaults(self):
        """Test that default values are used in examples."""
        class DefaultModel(BaseModel):
            status: str = "active"
            count: int = 0
        
        gen = PydanticPromptGenerator(DefaultModel)
        example = gen._generate_example_from_schema(gen.schema)
        
        assert example["status"] == "active"
        assert example["count"] == 0
    
    def test_example_with_enums(self):
        """Test enum handling in examples."""
        gen = PydanticPromptGenerator(ComplexModel)
        example = gen._generate_example_from_schema(gen.schema)
        
        assert example["priority"] in ["low", "medium", "high"]


# Schema Helper Tests
class TestSchemaHelpers:
    """Test schema helper methods."""
    
    def test_resolve_ref_valid(self, complex_generator):
        """Test resolving valid schema references."""
        ref = "#/$defs/Address"
        resolved = complex_generator._resolve_ref(ref)
        
        assert resolved is not None
        assert "properties" in resolved
        assert "street" in resolved["properties"]
    
    def test_resolve_ref_invalid(self, complex_generator):
        """Test resolving invalid references."""
        ref = "#/$defs/NonExistent"
        resolved = complex_generator._resolve_ref(ref)
        
        assert resolved is None
    
    def test_get_type_description_simple(self, simple_generator):
        """Test type description for simple types."""
        desc = simple_generator._get_type_description({"type": "string"})
        assert desc == "string"
        
        desc = simple_generator._get_type_description({"type": "integer"})
        assert desc == "integer"
    
    def test_get_type_description_array(self, complex_generator):
        """Test type description for arrays."""
        schema = {
            "type": "array",
            "items": {"type": "string"}
        }
        desc = complex_generator._get_type_description(schema)
        assert "array" in desc
        assert "string" in desc
    
    def test_get_type_description_ref(self, complex_generator):
        """Test type description for references."""
        schema = {"$ref": "#/$defs/Address"}
        desc = complex_generator._get_type_description(schema)
        assert "Address" in desc


# Validation Tests
class TestValidation:
    """Test response validation."""
    
    def test_validate_valid_response(self, simple_generator):
        """Test validation with valid data."""
        response = {"name": "John Doe", "age": 30}
        
        validated = simple_generator.validate_response(response)
        assert isinstance(validated, SimpleModel)
        assert validated.name == "John Doe"
        assert validated.age == 30
    
    def test_validate_valid_json_string(self, simple_generator):
        """Test validation with JSON string."""
        response_json = '{"name": "Jane Doe", "age": 25}'
        
        validated = simple_generator.validate_response(response_json)
        assert isinstance(validated, SimpleModel)
        assert validated.name == "Jane Doe"
    
    def test_validate_invalid_response(self, simple_generator):
        """Test validation with invalid data."""
        response = {"name": "John", "age": "not_an_int"}
        
        with pytest.raises(ValidationError):
            simple_generator.validate_response(response)
    
    def test_validate_missing_required_field(self, simple_generator):
        """Test validation with missing required field."""
        response = {"name": "John"}
        
        with pytest.raises(ValidationError):
            simple_generator.validate_response(response)
    
    def test_validate_constraint_violation(self, complex_generator):
        """Test validation with constraint violations."""
        response = {
            "id": -1,  # Should be > 0
            "name": "A",  # Too short
            "email": "invalid",
            "score": 50.0,
            "addresses": []  # Should have at least 1
        }
        
        with pytest.raises(ValidationError):
            complex_generator.validate_response(response)
    
    def test_validate_invalid_json_string(self, simple_generator):
        """Test validation with malformed JSON."""
        with pytest.raises(json.JSONDecodeError):
            simple_generator.validate_response('{"name": "John", invalid}')


# Schema Export Tests
class TestSchemaExport:
    """Test schema export functionality."""
    
    def test_get_schema_dict(self, simple_generator):
        """Test getting schema as dictionary."""
        schema = simple_generator.get_schema_dict()
        
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "title" in schema
    
    def test_get_schema_dict_is_copy(self, simple_generator):
        """Test that returned schema is a copy."""
        schema1 = simple_generator.get_schema_dict()
        schema2 = simple_generator.get_schema_dict()
        
        # Modifying one shouldn't affect the other
        schema1["modified"] = True
        assert "modified" not in schema2
    
    def test_get_schema_json(self, simple_generator):
        """Test getting schema as JSON string."""
        schema_json = simple_generator.get_schema_json()
        
        assert isinstance(schema_json, str)
        
        # Should be valid JSON
        parsed = json.loads(schema_json)
        assert isinstance(parsed, dict)
    
    def test_get_schema_json_formatting(self, simple_generator):
        """Test JSON formatting options."""
        # Default indent
        schema_default = simple_generator.get_schema_json()
        
        # Custom indent
        schema_compact = simple_generator.get_schema_json(indent=0)
        
        # Default should be more readable (longer)
        assert len(schema_default) >= len(schema_compact)


# Edge Cases and Error Handling
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_deeply_nested_model(self):
        """Test handling of deeply nested models."""
        class Level3(BaseModel):
            value: str
        
        class Level2(BaseModel):
            level3: Level3
        
        class Level1(BaseModel):
            level2: Level2
        
        gen = PydanticPromptGenerator(Level1)
        prompt = gen.generate_prompt()
        
        assert "level2" in prompt
        assert "level3" in prompt
    
    def test_optional_fields(self):
        """Test handling of optional fields."""
        class OptionalModel(BaseModel):
            required: str
            optional: Optional[str] = None
        
        gen = PydanticPromptGenerator(OptionalModel)
        prompt = gen.generate_prompt()
        
        assert "required" in prompt.lower() or "Required" in prompt
        assert "optional" in prompt.lower() or "Optional" in prompt
    
    def test_list_of_primitives(self):
        """Test handling lists of primitive types."""
        class ListModel(BaseModel):
            numbers: List[int]
            words: List[str]
        
        gen = PydanticPromptGenerator(ListModel)
        example = gen._generate_example_from_schema(gen.schema)
        
        assert isinstance(example["numbers"], list)
        assert isinstance(example["words"], list)
        if example["numbers"]:
            assert isinstance(example["numbers"][0], int)
    
    def test_multiple_of_constraint(self):
        """Test multipleOf constraint handling."""
        class MultipleModel(BaseModel):
            value: int = Field(..., multiple_of=5, ge=10)
        
        gen = PydanticPromptGenerator(MultipleModel)
        example = gen._generate_example_from_schema(gen.schema)
        
        # Value should be multiple of 5 and >= 10
        assert example["value"] % 5 == 0
        assert example["value"] >= 10
    
    def test_repr(self, simple_generator):
        """Test string representation."""
        repr_str = repr(simple_generator)
        
        assert "PydanticPromptGenerator" in repr_str
        assert "SimpleModel" in repr_str
        assert "detailed" in repr_str


# Integration Tests
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_generate_and_validate_workflow(self):
        """Test complete generate-validate workflow."""
        # Create model
        class Product(BaseModel):
            id: int = Field(..., gt=0)
            name: str = Field(..., min_length=1)
            price: float = Field(..., ge=0)
            in_stock: bool = True
        
        # Generate prompt
        gen = PydanticPromptGenerator(Product)
        prompt = gen.generate_prompt()
        
        # Verify prompt was generated
        assert len(prompt) > 0
        assert "Product" in prompt
        
        # Create valid response based on example
        example = gen._generate_example_from_schema(gen.schema)
        
        # Validate the generated example
        validated = gen.validate_response(example)
        assert isinstance(validated, Product)
    
    def test_all_prompt_styles(self):
        """Test all prompt styles produce valid output."""
        styles = [PromptStyle.DETAILED, PromptStyle.CONCISE, PromptStyle.TECHNICAL]
        
        for style in styles:
            gen = PydanticPromptGenerator(SimpleModel, style=style)
            prompt = gen.generate_prompt()
            
            assert len(prompt) > 0
            assert isinstance(prompt, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
