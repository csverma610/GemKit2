# Pydantic Prompt Generator

A production-quality Python library for generating structured prompts from Pydantic models to guide LLMs in producing valid JSON responses.

## Features

- 🎯 **Type-Safe**: Leverages Pydantic's validation system
- 📝 **Multiple Prompt Styles**: Detailed, concise, and technical formats
- 🔍 **Constraint-Aware**: Includes field constraints in prompts
- 🏗️ **Nested Models**: Full support for complex nested structures
- ✅ **Response Validation**: Built-in validation of LLM responses
- 🔄 **Schema Export**: Export schemas as dict or JSON
- 📊 **Example Generation**: Automatic example data generation
- 🛡️ **Production-Ready**: Comprehensive error handling and logging

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from pydantic import BaseModel, Field
from pydantic_prompt_generator import PydanticPromptGenerator, PromptStyle

# Define your model
class User(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    age: int = Field(..., ge=18, le=120)

# Generate prompt
generator = PydanticPromptGenerator(User)
prompt = generator.generate_prompt()
print(prompt)

# Validate LLM response
response = '{"name": "John Doe", "email": "john@example.com", "age": 30}'
validated_user = generator.validate_response(response)
print(f"Valid user: {validated_user.name}")
```

## Prompt Styles

### Detailed (Default)
Comprehensive prompts with full documentation, constraints, and examples.

```python
generator = PydanticPromptGenerator(User, style=PromptStyle.DETAILED)
```

**Output includes:**
- Field descriptions
- All validation constraints
- Nested object definitions
- Example JSON structure
- Required/optional markers

### Concise
Minimal prompts focusing on essential information.

```python
generator = PydanticPromptGenerator(User, style=PromptStyle.CONCISE)
```

**Output includes:**
- Field names and types
- Required markers
- Basic example (optional)

### Technical
Schema-focused prompts with raw JSON schema.

```python
generator = PydanticPromptGenerator(User, style=PromptStyle.TECHNICAL)
```

**Output includes:**
- Complete JSON schema
- Technical specifications
- Schema references

## Advanced Usage

### Complex Nested Models

```python
from typing import List
from enum import Enum

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class Address(BaseModel):
    street: str = Field(..., min_length=1)
    city: str = Field(..., min_length=1)
    zip_code: str = Field(..., pattern=r"^\d{5}$")

class UserProfile(BaseModel):
    id: int = Field(..., gt=0, description="Unique identifier")
    name: str = Field(..., min_length=2, max_length=50)
    priority: Priority = Priority.MEDIUM
    addresses: List[Address] = Field(..., min_items=1)
    tags: List[str] = Field(default_factory=list, max_items=10)

generator = PydanticPromptGenerator(UserProfile)
prompt = generator.generate_prompt()
```

### Custom Configuration

```python
generator = PydanticPromptGenerator(
    model=UserProfile,
    style=PromptStyle.DETAILED,
    include_examples=True,      # Include example JSON
    validate_schema=True        # Validate schema on init
)
```

### Response Validation

```python
# Validate dictionary
response_dict = {
    "id": 123,
    "name": "Jane Doe",
    "priority": "high",
    "addresses": [
        {"street": "123 Main St", "city": "Springfield", "zip_code": "12345"}
    ]
}

try:
    validated = generator.validate_response(response_dict)
    print(f"✅ Valid: {validated}")
except ValidationError as e:
    print(f"❌ Invalid: {e}")

# Validate JSON string
response_json = '{"id": 123, "name": "Jane Doe", ...}'
validated = generator.validate_response(response_json)
```

### Schema Export

```python
# Get schema as dictionary
schema_dict = generator.get_schema_dict()

# Get schema as JSON string
schema_json = generator.get_schema_json(indent=2)

# Access raw schema
raw_schema = generator.schema
```

## Supported Pydantic Features

### Field Types
- ✅ String, Integer, Float, Boolean
- ✅ Lists and nested lists
- ✅ Optional fields
- ✅ Enums
- ✅ Nested models
- ✅ Dictionaries

### Validation Constraints
- ✅ `min_length`, `max_length` (strings)
- ✅ `minimum`, `maximum` (numbers)
- ✅ `exclusive_minimum`, `exclusive_maximum`
- ✅ `gt`, `ge`, `lt`, `le` (comparison)
- ✅ `multiple_of` (numbers)
- ✅ `pattern` (regex patterns)
- ✅ `min_items`, `max_items` (arrays)
- ✅ `unique_items` (arrays)
- ✅ Default values
- ✅ Field descriptions

### Advanced Features
- ✅ Nested model definitions
- ✅ Recursive schemas
- ✅ Schema references (`$ref`)
- ✅ `allOf`, `anyOf`, `oneOf` (partial)

## API Reference

### `PydanticPromptGenerator`

#### Constructor

```python
PydanticPromptGenerator(
    model: Type[BaseModel],
    style: PromptStyle = PromptStyle.DETAILED,
    include_examples: bool = True,
    validate_schema: bool = True
)
```

**Parameters:**
- `model`: Pydantic BaseModel class (not instance)
- `style`: Prompt generation style
- `include_examples`: Include example JSON in prompts
- `validate_schema`: Validate schema on initialization

**Raises:**
- `TypeError`: If model is not a Pydantic BaseModel class
- `SchemaValidationError`: If schema validation fails

#### Methods

##### `generate_prompt() -> str`
Generates the prompt based on configured style.

**Returns:** Formatted prompt string

**Raises:** `ValueError` for unsupported styles

##### `validate_response(response_json: Union[str, Dict]) -> BaseModel`
Validates LLM response against schema.

**Parameters:**
- `response_json`: JSON string or dictionary

**Returns:** Validated Pydantic model instance

**Raises:**
- `ValidationError`: If response doesn't match schema
- `json.JSONDecodeError`: If JSON string is invalid

##### `get_schema_dict() -> Dict[str, Any]`
Returns JSON schema as dictionary (copy).

##### `get_schema_json(indent: int = 2) -> str`
Returns formatted JSON schema string.

### `PromptStyle` (Enum)
- `DETAILED`: Full documentation with constraints
- `CONCISE`: Minimal essential information
- `TECHNICAL`: Schema-focused with JSON schema

## Error Handling

### Common Exceptions

```python
from pydantic_prompt_generator import SchemaValidationError
from pydantic import ValidationError

try:
    generator = PydanticPromptGenerator(InvalidModel)
except TypeError as e:
    print(f"Invalid model type: {e}")

try:
    validated = generator.validate_response(bad_response)
except ValidationError as e:
    print(f"Validation failed: {e}")
except SchemaValidationError as e:
    print(f"Schema error: {e}")
```

## Logging

The library uses Python's standard logging module:

```python
import logging

# Configure logging level
logging.getLogger("pydantic_prompt_generator").setLevel(logging.DEBUG)

# View detailed logs
generator = PydanticPromptGenerator(YourModel)
# Logs: "Initialized PydanticPromptGenerator for model: YourModel"
```

## Best Practices

### 1. Use Descriptive Field Descriptions
```python
class User(BaseModel):
    age: int = Field(..., ge=18, description="User's age in years (must be 18+)")
```

### 2. Leverage Enums for Fixed Options
```python
class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"

class User(BaseModel):
    status: Status = Status.ACTIVE
```

### 3. Provide Meaningful Constraints
```python
# Good: Clear constraints
email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

# Better: With description
email: str = Field(
    ..., 
    pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$",
    description="Valid email address"
)
```

### 4. Handle Validation Gracefully
```python
try:
    user = generator.validate_response(llm_response)
    # Process valid user
except ValidationError as e:
    # Log error and retry or handle gracefully
    logger.error(f"Invalid response: {e}")
    # Maybe regenerate with more explicit prompt
```

### 5. Choose Appropriate Prompt Style
- **DETAILED**: For complex models or when accuracy is critical
- **CONCISE**: For simple models or when token count matters
- **TECHNICAL**: For debugging or technical integrations

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest test_pydantic_prompt_generator.py -v

# Run specific test class
pytest test_pydantic_prompt_generator.py::TestPromptGeneration -v

# Run with coverage
pytest test_pydantic_prompt_generator.py --cov=pydantic_prompt_generator

# Run with detailed output
pytest test_pydantic_prompt_generator.py -v --tb=short
```

## Examples

See [pydantic_prompt_generator.py](computer:///home/user/pydantic_prompt_generator.py) for complete examples including:
- Basic usage
- Complex nested models
- Validation workflows
- Schema export
- All prompt styles

Run examples:
```bash
python pydantic_prompt_generator.py
```

## Performance Considerations

- Schema generation is cached after initialization
- Example generation uses recursion with depth limits (default: 5)
- Large nested models may produce lengthy prompts
- Consider `PromptStyle.CONCISE` for token optimization

## Limitations

- Pydantic V2+ required
- Limited support for custom validators
- Complex `anyOf`/`oneOf` schemas may need manual handling
- Discriminated unions not fully supported

## Contributing

Contributions welcome! Areas for improvement:
- Enhanced pattern-based example generation
- Support for custom example providers
- Additional prompt styles
- Better discriminated union support

## License

MIT License - see LICENSE file for details

## Changelog

### Version 2.0.0
- Production-quality refactor
- Comprehensive error handling
- Multiple prompt styles
- Enhanced constraint support
- Full test coverage
- Improved documentation

### Version 1.0.0
- Initial implementation
- Basic prompt generation
- Simple validation support

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review test cases for examples
