"""
Pydantic Prompt Generator 

This module provides utilities for generating detailed prompts from Pydantic models
that can be used with LLMs to ensure structured JSON output conforming to schemas.

"""

from pydantic import BaseModel, Field, ValidationError
from typing import Type, Dict, Any, List, Optional, Union, Set
import json
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptStyle(str, Enum):
    """Supported prompt generation styles."""
    DETAILED = "detailed"  # Full documentation with all constraints
    CONCISE = "concise"    # Minimal prompt with essential info
    TECHNICAL = "technical"  # Schema-focused with JSON schema details


class SchemaValidationError(Exception):
    """Raised when schema validation fails."""
    pass


class PydanticPromptGenerator:
    """
    Generates detailed, human-readable prompts for language models from Pydantic
    BaseModel schemas.

    This class inspects a Pydantic model and creates a prompt that instructs a
    language model to return a JSON object conforming to the model's schema. It
    handles nested models, validation constraints, enums, and complex types,
    making it easier to achieve structured output from LLMs.
    """
    
    # Supported constraint mappings
    CONSTRAINT_LABELS = {
        'minLength': 'Minimum length',
        'maxLength': 'Maximum length',
        'minimum': 'Minimum value',
        'maximum': 'Maximum value',
        'exclusiveMinimum': 'Must be greater than',
        'exclusiveMaximum': 'Must be less than',
        'multipleOf': 'Must be multiple of',
        'minItems': 'Minimum length',
        'maxItems': 'Maximum length',
        'uniqueItems': 'Items must be unique',
    }
    
    def __init__(
        self, 
        model: Type[BaseModel],
        style: PromptStyle = PromptStyle.DETAILED,
        include_examples: bool = True,
        validate_schema: bool = True
    ):
        """
        Initializes the PydanticPromptGenerator.

        Args:
            model (Type[BaseModel]): The Pydantic BaseModel class (not an instance) to
                                     generate the prompt from.
            style (PromptStyle, optional): The style of the generated prompt.
            include_examples (bool, optional): Whether to include an example JSON object
                                               in the prompt.
            validate_schema (bool, optional): Whether to validate the generated schema
                                              on initialization.
        """
        self._validate_model_type(model)
        
        self.model = model
        self.style = style
        self.include_examples = include_examples
        
        try:
            self.schema = model.model_json_schema()
        except Exception as e:
            raise SchemaValidationError(f"Failed to generate schema: {e}") from e
        
        if validate_schema:
            self._validate_schema()
        
        logger.info(f"Initialized PydanticPromptGenerator for model: {model.__name__}")
    
    def _validate_model_type(self, model: Any) -> None:
        """
        Validates that the provided model is a Pydantic BaseModel class.
        
        Args:
            model: The model to validate
            
        Raises:
            TypeError: If model is not a valid Pydantic BaseModel class
        """
        if not isinstance(model, type):
            raise TypeError(
                f"Expected a class, got instance of {type(model).__name__}. "
                "Pass the class itself, not an instance."
            )
        
        if not issubclass(model, BaseModel):
            raise TypeError(
                f"Model must be a Pydantic BaseModel subclass, got {model.__name__}"
            )
    
    def _validate_schema(self) -> None:
        """
        Validates the generated schema structure.
        
        Raises:
            SchemaValidationError: If schema is malformed
        """
        if not isinstance(self.schema, dict):
            raise SchemaValidationError("Schema must be a dictionary")
        
        if 'properties' not in self.schema and 'allOf' not in self.schema:
            logger.warning(
                f"Schema for {self.model.__name__} has no 'properties' or 'allOf'. "
                "This may indicate an empty model."
            )
    
    def _resolve_ref(self, ref: str) -> Optional[Dict[str, Any]]:
        """
        Resolves a JSON Schema $ref to its definition.
        
        Args:
            ref: The reference string (e.g., "#/$defs/Address")
            
        Returns:
            The resolved schema definition or None if not found
        """
        if not ref.startswith('#/'):
            logger.warning(f"Non-local reference not supported: {ref}")
            return None
        
        parts = ref.split('/')[1:]  # Remove leading '#'
        current = self.schema
        
        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to resolve reference {ref}: {e}")
            return None
    
    def _format_constraint(self, constraint: str, value: Any) -> str:
        """
        Formats a constraint for display in the prompt.
        
        Args:
            constraint: The constraint key
            value: The constraint value
            
        Returns:
            Formatted constraint string
        """
        label = self.CONSTRAINT_LABELS.get(constraint, constraint)
        
        if constraint == 'uniqueItems' and value:
            return "Items must be unique"
        
        return f"{label}: {value}"
    
    def _get_type_description(self, prop_details: Dict[str, Any]) -> str:
        """
        Gets a human-readable type description from property details.
        
        Args:
            prop_details: The property details from the schema
            
        Returns:
            Type description string
        """
        if '$ref' in prop_details:
            ref_name = prop_details['$ref'].split('/')[-1]
            return f"object ({ref_name})"
        
        prop_type = prop_details.get('type', 'any')
        
        if prop_type == 'array':
            items = prop_details.get('items', {})
            if '$ref' in items:
                item_type = items['$ref'].split('/')[-1]
                return f"array of {item_type} objects"
            item_type = items.get('type', 'any')
            return f"array of {item_type}"
        
        # Handle anyOf/oneOf/allOf
        if 'anyOf' in prop_details:
            types = [self._get_type_description(t) for t in prop_details['anyOf']]
            return f"one of: {', '.join(types)}"
        
        if 'oneOf' in prop_details:
            types = [self._get_type_description(t) for t in prop_details['oneOf']]
            return f"exactly one of: {', '.join(types)}"
        
        return prop_type
    
    def _generate_property_description(
        self, 
        prop_name: str, 
        prop_details: Dict[str, Any],
        required_fields: Set[str]
    ) -> str:
        """
        Generates a detailed description for a single property.
        
        Args:
            prop_name: The property name
            prop_details: The property schema details
            required_fields: Set of required field names
            
        Returns:
            Formatted property description
        """
        type_desc = self._get_type_description(prop_details)
        is_required = prop_name in required_fields
        
        parts = [f"- `{prop_name}` ({type_desc})"]
        
        # Add description
        if 'description' in prop_details:
            parts.append(f": {prop_details['description']}")
        else:
            parts.append(":")
        
        details = []
        
        # Add constraints
        for constraint, value in prop_details.items():
            if constraint in self.CONSTRAINT_LABELS or constraint == 'uniqueItems':
                details.append(self._format_constraint(constraint, value))
        
        # Add pattern
        if 'pattern' in prop_details:
            details.append(f"Pattern: `{prop_details['pattern']}`")
        
        # Add enum
        if 'enum' in prop_details:
            enum_values = ', '.join(f"`{v}`" for v in prop_details['enum'])
            details.append(f"Allowed values: {enum_values}")
        
        # Add default
        if 'default' in prop_details:
            default_val = prop_details['default']
            if isinstance(default_val, str):
                details.append(f"Default: `\"{default_val}\"`")
            else:
                details.append(f"Default: `{default_val}`")
        
        # Add required status
        if is_required:
            details.append("**(Required)**")
        else:
            details.append("(Optional)")
        
        if details:
            parts.append(" " + " | ".join(details))
        
        return "".join(parts)
    
    def _generate_detailed_prompt(self) -> str:
        """Generates a detailed prompt with full documentation."""
        lines = [
            "Please provide a JSON object that conforms to the following schema:\n",
            f"**Root Object:** `{self.schema.get('title', 'Root')}`"
        ]
        
        if 'description' in self.schema:
            lines.append(f"\n*{self.schema['description']}*")
        
        properties = self.schema.get('properties', {})
        required_fields = set(self.schema.get('required', []))
        
        if properties:
            lines.append("\n**Properties:**\n")
            
            for prop_name, prop_details in properties.items():
                prop_desc = self._generate_property_description(
                    prop_name, prop_details, required_fields
                )
                lines.append(prop_desc + "\n")
        
        # Add definitions if present
        if '$defs' in self.schema and self.style == PromptStyle.DETAILED:
            lines.append("\n**Nested Object Definitions:**\n")
            for def_name, def_schema in self.schema['$defs'].items():
                lines.append(f"\n`{def_name}`:")
                if 'description' in def_schema:
                    lines.append(f" {def_schema['description']}")
                
                # Check if it's an enum
                if 'enum' in def_schema:
                    enum_values = ', '.join(f"`{v}`" for v in def_schema['enum'])
                    lines.append(f" Allowed values: {enum_values}")
                
                lines.append("\n")
                
                def_props = def_schema.get('properties', {})
                def_required = set(def_schema.get('required', []))
                
                for prop_name, prop_details in def_props.items():
                    prop_desc = self._generate_property_description(
                        prop_name, prop_details, def_required
                    )
                    lines.append("  " + prop_desc + "\n")
        
        return "".join(lines)
    
    def _generate_concise_prompt(self) -> str:
        """Generates a concise prompt with essential information only."""
        lines = [
            f"Return a JSON object for `{self.schema.get('title', 'Root')}` with these fields:\n"
        ]
        
        properties = self.schema.get('properties', {})
        required_fields = set(self.schema.get('required', []))
        
        for prop_name, prop_details in properties.items():
            type_desc = self._get_type_description(prop_details)
            required_mark = " (required)" if prop_name in required_fields else ""
            lines.append(f"- {prop_name}: {type_desc}{required_mark}\n")
        
        return "".join(lines)
    
    def _generate_technical_prompt(self) -> str:
        """Generates a technical prompt with JSON schema reference."""
        lines = [
            f"Generate a JSON object conforming to this schema:\n\n",
            "```json\n",
            json.dumps(self.schema, indent=2),
            "\n```\n"
        ]
        return "".join(lines)
    
    def generate_prompt(self) -> str:
        """
        Generates a prompt string based on the configured style.

        Returns:
            str: A prompt that instructs a language model to return a JSON object
                 conforming to the Pydantic model's schema.
        """
        try:
            # Generate main prompt based on style
            if self.style == PromptStyle.DETAILED:
                prompt = self._generate_detailed_prompt()
            elif self.style == PromptStyle.CONCISE:
                prompt = self._generate_concise_prompt()
            elif self.style == PromptStyle.TECHNICAL:
                prompt = self._generate_technical_prompt()
            else:
                raise ValueError(f"Unsupported prompt style: {self.style}")
            
            # Add example if requested
            if self.include_examples and self.style != PromptStyle.TECHNICAL:
                prompt += "\n**Example JSON Structure:**\n```json\n"
                example = self._generate_example_from_schema(self.schema)
                prompt += json.dumps(example, indent=2)
                prompt += "\n```\n"
            
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to generate prompt: {e}")
            raise
    
    def _generate_example_value(
        self, 
        prop_details: Dict[str, Any], 
        prop_name: str
    ) -> Any:
        """
        Generates an example value for a property based on its type and constraints.
        
        Args:
            prop_details: The property schema details
            prop_name: The property name (used for generating example strings)
            
        Returns:
            An example value appropriate for the property type
        """
        # Handle allOf (enum wrapped in allOf)
        if 'allOf' in prop_details:
            for sub_schema in prop_details['allOf']:
                if 'enum' in sub_schema:
                    return sub_schema['enum'][0]
        
        # Handle enum first
        if 'enum' in prop_details:
            return prop_details['enum'][0]
        
        # Handle default
        if 'default' in prop_details:
            return prop_details['default']
        
        prop_type = prop_details.get('type')
        
        if prop_type == 'string':
            # Generate pattern-compliant example if pattern exists
            if 'pattern' in prop_details:
                pattern = prop_details['pattern']
                if pattern == r"^\d{5}$":
                    return "12345"
                elif r"@" in pattern or "email" in prop_name.lower():
                    return "user@example.com"
                # Add more common patterns as needed
            
            min_length = prop_details.get('minLength', 0)
            max_length = prop_details.get('maxLength', 20)
            example = f"example_{prop_name}"
            
            # Adjust length if needed
            if len(example) < min_length:
                example = example + "_" * (min_length - len(example))
            if len(example) > max_length:
                example = example[:max_length]
            
            return example
        
        elif prop_type == 'integer':
            minimum = prop_details.get('minimum', 1)
            maximum = prop_details.get('maximum', 100)
            exclusive_min = prop_details.get('exclusiveMinimum')
            exclusive_max = prop_details.get('exclusiveMaximum')
            
            if exclusive_min is not None:
                value = exclusive_min + 1
            else:
                value = minimum
            
            # Check if value respects maximum
            if exclusive_max is not None and value >= exclusive_max:
                value = exclusive_max - 1
            elif maximum is not None and value > maximum:
                value = maximum
            
            # Handle multipleOf
            multiple_of = prop_details.get('multipleOf')
            if multiple_of:
                value = (value // multiple_of) * multiple_of
                if value < minimum:
                    value += multiple_of
            
            return int(value)
        
        elif prop_type == 'number':
            minimum = prop_details.get('minimum', 0.0)
            maximum = prop_details.get('maximum', 100.0)
            return float((minimum + maximum) / 2)
        
        elif prop_type == 'boolean':
            return True
        
        elif prop_type == 'null':
            return None
        
        return None
    
    def _generate_example_from_schema(
        self, 
        schema: Dict[str, Any],
        depth: int = 0,
        max_depth: int = 5
    ) -> Union[Dict[str, Any], List[Any], Any]:
        """
        Generates a sample data structure from a JSON schema.
        
        Args:
            schema: The JSON schema to generate examples from
            depth: Current recursion depth (prevents infinite recursion)
            max_depth: Maximum recursion depth allowed
            
        Returns:
            An example data structure matching the schema
        """
        if depth > max_depth:
            logger.warning(f"Max recursion depth {max_depth} reached in example generation")
            return None
        
        # Handle $ref
        if '$ref' in schema:
            ref_schema = self._resolve_ref(schema['$ref'])
            if ref_schema:
                return self._generate_example_from_schema(ref_schema, depth + 1, max_depth)
            return {}
        
        # Handle allOf (merge all schemas)
        if 'allOf' in schema:
            merged_example = {}
            for sub_schema in schema['allOf']:
                sub_example = self._generate_example_from_schema(sub_schema, depth + 1, max_depth)
                if isinstance(sub_example, dict):
                    merged_example.update(sub_example)
            return merged_example
        
        # Handle anyOf/oneOf (use first option)
        if 'anyOf' in schema or 'oneOf' in schema:
            options = schema.get('anyOf') or schema.get('oneOf')
            if options:
                return self._generate_example_from_schema(options[0], depth + 1, max_depth)
        
        # Handle array type
        if schema.get('type') == 'array':
            items_schema = schema.get('items', {})
            min_items = schema.get('minItems', 1)
            
            # Generate examples based on minItems or default to 1-2 items
            num_items = max(min_items, 1) if min_items <= 2 else 2
            
            if items_schema:
                return [
                    self._generate_example_from_schema(items_schema, depth + 1, max_depth)
                    for _ in range(num_items)
                ]
            return []
        
        # Handle object with properties
        properties = schema.get('properties', {})
        if not properties:
            return {}
        
        example = {}
        required_fields = set(schema.get('required', []))
        
        for prop_name, prop_details in properties.items():
            # Generate all required fields and optional fields with defaults
            is_required = prop_name in required_fields
            has_default = 'default' in prop_details
            
            if is_required or has_default or len(example) < 5:
                # Check if it's a nested object or array
                if prop_details.get('type') == 'array':
                    items_schema = prop_details.get('items', {})
                    if items_schema:
                        if items_schema.get('type') in ['string', 'integer', 'number', 'boolean']:
                            # Handle primitive arrays
                            prim_example = self._generate_example_value(items_schema, prop_name + '_item')
                            example[prop_name] = [prim_example] if prim_example is not None else []
                        else:
                            # Handle object arrays
                            example[prop_name] = [
                                self._generate_example_from_schema(items_schema, depth + 1, max_depth)
                            ]
                    else:
                        example[prop_name] = []
                elif '$ref' in prop_details:
                    ref_schema = self._resolve_ref(prop_details['$ref'])
                    if ref_schema:
                        # Check if ref is an enum
                        if 'enum' in ref_schema:
                            example[prop_name] = ref_schema['enum'][0]
                        else:
                            example[prop_name] = self._generate_example_from_schema(
                                ref_schema, depth + 1, max_depth
                            )
                    else:
                        example[prop_name] = {}
                else:
                    example[prop_name] = self._generate_example_value(prop_details, prop_name)
        
        return example
    
    def validate_response(self, response_json: Union[str, Dict[str, Any]]) -> BaseModel:
        """
        Validates a JSON response against the Pydantic model's schema.

        Args:
            response_json (Union[str, Dict[str, Any]]): The JSON response to validate,
                                                        as a string or a dictionary.

        Returns:
            BaseModel: An instance of the Pydantic model, populated with the
                       validated data.

        Raises:
            ValidationError: If the response does not conform to the schema.
            json.JSONDecodeError: If `response_json` is an invalid JSON string.
        """
        if isinstance(response_json, str):
            data = json.loads(response_json)
        else:
            data = response_json
        
        try:
            return self.model(**data)
        except ValidationError as e:
            logger.error(f"Validation failed for {self.model.__name__}: {e}")
            raise
    
    def get_schema_dict(self) -> Dict[str, Any]:
        """
        Returns the JSON schema of the Pydantic model as a dictionary.

        Returns:
            Dict[str, Any]: The JSON schema.
        """
        return self.schema.copy()
    
    def get_schema_json(self, indent: int = 2) -> str:
        """
        Returns the JSON schema of the Pydantic model as a formatted string.

        Args:
            indent (int, optional): The indentation level for the JSON string.

        Returns:
            str: The formatted JSON schema.
        """
        return json.dumps(self.schema, indent=indent)
    
    def __repr__(self) -> str:
        """Returns a string representation of the generator."""
        return (
            f"PydanticPromptGenerator(model={self.model.__name__}, "
            f"style={self.style.value}, include_examples={self.include_examples})"
        )


# Example usage and testing
if __name__ == '__main__':
    # Example 1: Complex nested model with validation constraints
    class Address(BaseModel):
        """Represents a physical address."""
        street: str = Field(..., min_length=1, description="Street address")
        city: str = Field(..., min_length=1, description="City name")
        state: str = Field(..., min_length=2, max_length=2, description="State code (2 letters)")
        zip_code: str = Field(..., pattern=r"^\d{5}$", description="5-digit ZIP code")
        country: str = Field(default="USA", description="Country name")

    class ContactMethod(str, Enum):
        """Preferred contact methods."""
        EMAIL = "email"
        PHONE = "phone"
        SMS = "sms"
        MAIL = "mail"

    class UserProfile(BaseModel):
        """Represents a comprehensive user profile with validation."""
        user_id: int = Field(..., gt=0, description="Unique user identifier")
        name: str = Field(..., min_length=2, max_length=50, description="Full name")
        email: str = Field(
            ..., 
            pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$",
            description="Valid email address"
        )
        age: Optional[int] = Field(None, ge=18, le=120, description="User age")
        is_active: bool = Field(default=True, description="Account active status")
        user_type: str = Field(default="reader", description="User role type")
        preferred_contact: ContactMethod = Field(
            default=ContactMethod.EMAIL,
            description="Preferred method of contact"
        )
        addresses: List[Address] = Field(
            ..., 
            min_length=1,
            description="List of user addresses (at least one required)"
        )
        tags: Optional[List[str]] = Field(
            default=None,
            max_length=10,
            description="User tags for categorization"
        )

    print("=" * 80)
    print("EXAMPLE 1: Detailed Prompt Style")
    print("=" * 80)
    
    generator_detailed = PydanticPromptGenerator(
        UserProfile, 
        style=PromptStyle.DETAILED,
        include_examples=True
    )
    
    prompt = generator_detailed.generate_prompt()
    print(prompt)
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Concise Prompt Style")
    print("=" * 80)
    
    generator_concise = PydanticPromptGenerator(
        UserProfile,
        style=PromptStyle.CONCISE,
        include_examples=True
    )
    
    print(generator_concise.generate_prompt())
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Validation Test")
    print("=" * 80)
    
    # Test with valid data
    valid_response = {
        "user_id": 12345,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "is_active": True,
        "user_type": "premium",
        "preferred_contact": "email",
        "addresses": [
            {
                "street": "123 Main St",
                "city": "Springfield",
                "state": "IL",
                "zip_code": "62701",
                "country": "USA"
            }
        ],
        "tags": ["premium", "verified"]
    }
    
    try:
        validated_user = generator_detailed.validate_response(valid_response)
        print("✅ Validation successful!")
        print(f"Validated user: {validated_user.name} ({validated_user.email})")
    except ValidationError as e:
        print(f"❌ Validation failed: {e}")
    
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Schema Export")
    print("=" * 80)
    
    print("Schema JSON (first 500 chars):")
    schema_json = generator_detailed.get_schema_json()
    print(schema_json[:500] + "...\n")
    
    print(f"Generator representation: {generator_detailed}")
