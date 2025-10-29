# Claims Generation Tool

A Python utility for extracting verifiable claims from text documents using Google's Gemini API.

## Overview

This tool analyzes text files and generates a list of factual, verifiable claims. It handles large documents by splitting them into manageable chunks, processes each chunk through the Gemini API, and deduplicates similar claims to produce a consolidated list.

## Features

- **Automatic text chunking** for large documents (configurable chunk size)
- **Claim deduplication** to remove duplicate and highly similar claims
- **Retry logic** with exponential backoff for API reliability
- **File size validation** to prevent processing of excessively large files
- **Comprehensive logging** with rotation support
- **Flexible output** as structured data (list) or formatted text

## Requirements

- Python 3.10+
- `google-genai` library
- Valid Google API credentials

## Installation

```bash
pip install google-genai
```

Set up your Google API credentials according to the [Google AI documentation](https://ai.google.dev/tutorials/setup).

## Usage

### Basic Usage

```bash
python claims_generation.py document.txt
```

### With Options

```bash
# Limit to 10 claims
python claims_generation.py document.txt --max-claims 10

# Use a different model
python claims_generation.py document.txt --model gemini-2.5-pro

# Adjust chunk size for processing
python claims_generation.py document.txt --chunk-size 5000

# Enable debug logging
python claims_generation.py document.txt --log-level DEBUG
```

### Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `filename` | Path to text file to process | Required |
| `-m, --max-claims` | Maximum number of claims to generate | No limit |
| `--model` | Gemini model to use | gemini-2.5-flash |
| `--log-file` | Path to log file | claims_generation.log |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | INFO |
| `--max-file-size` | Maximum file size in MB | 10 |
| `--max-retries` | Maximum API retry attempts | 3 |
| `--chunk-size` | Text chunk size in characters | 8000 |

## Programmatic Usage

```python
from claims_generation import ClaimsGenerator, ClaimsConfig

# Option 1: Use default configuration
generator = ClaimsGenerator()

# Option 2: Create custom configuration
config = ClaimsConfig(
    model="gemini-2.5-pro",
    max_claims=20,
    max_file_size_mb=15,
    max_retries=5,
    chunk_size_chars=10000
)
generator = ClaimsGenerator(config=config)

# Generate claims from file
claims = generator.generate_claims_from_file("document.txt")
# Returns: ['Claim 1', 'Claim 2', ...]

# Generate claims from text string
text = "Your text content here..."
claims = generator.generate_text(text)

# Format as numbered list
formatted = ClaimsGenerator.format_claims_as_numbered_list(claims)
print(formatted)
```

## Configuration

The `ClaimsConfig` dataclass contains all configurable settings:

```python
@dataclass
class ClaimsConfig:
    model: str = "gemini-2.5-flash"           # Gemini model to use
    max_claims: Optional[int] = None          # Max claims to generate (None = no limit)
    max_file_size_mb: int = 10                # Maximum file size
    max_retries: int = 3                      # API retry attempts
    chunk_size_chars: int = 8000              # Text chunk size
    retry_delay_seconds: int = 2              # Initial retry delay
    log_file: str = "claims_generation.log"   # Log file path
    log_level: str = "INFO"                   # Logging level
    log_file_max_bytes: int = 10 * 1024 * 1024  # Log file size (10MB)
    log_file_backup_count: int = 5            # Number of backup logs
    chunk_overlap_chars: int = 200            # Overlap between chunks
    similarity_length_threshold: int = 10     # Character diff for similarity
    similarity_word_overlap_ratio: float = 0.85  # Word overlap threshold (85%)
```

You can create configurations in multiple ways:

```python
# Using default values
config = ClaimsConfig()

# Overriding specific parameters
config = ClaimsConfig(model="gemini-2.5-pro", max_claims=50)

# Creating from variables
config = ClaimsConfig(
    model=args.model,
    max_claims=args.max_claims,
    chunk_size_chars=args.chunk_size
)
```

## How It Works

1. **File Validation**: Checks file size against configured limit
2. **Text Chunking**: Splits large texts into overlapping chunks at sentence boundaries
3. **Claim Extraction**: Sends each chunk to Gemini API with structured prompts
4. **Response Parsing**: Extracts individual claims from API responses
5. **Deduplication**: Removes exact duplicates and highly similar claims (>85% word overlap)
6. **Output**: Returns list of unique claims

## Error Handling

The tool includes custom exceptions for specific error cases:

- `FileSizeLimitError`: File exceeds maximum size limit
- `APIError`: API request fails after all retry attempts
- `InvalidResponseError`: API returns empty or invalid response
- `ClaimsGenerationError`: General errors (e.g., file encoding issues)

## Logging

Logs are written to both file and console:
- **File**: All log levels (DEBUG through CRITICAL)
- **Console**: WARNING and above only
- **Rotation**: Automatic log rotation at 10MB (keeps 5 backups)

Log location: `claims_generation.log` (configurable)

## Limitations

- **Processing time**: Large documents require multiple API calls
- **API costs**: Each chunk requires a separate Gemini API call
- **Deduplication**: Uses simple word overlap; may miss semantically similar claims
- **Performance**: O(n²) deduplication algorithm can be slow with 100+ claims
- **Language**: Optimized for English text
- **File format**: Supports plain text files only (UTF-8 encoding)

## Performance Considerations

- **Chunk size**: Larger chunks mean fewer API calls but may miss granular details
- **Overlap**: 200-character overlap helps maintain context across chunks
- **Retry logic**: Exponential backoff prevents overwhelming the API during transient failures
- **Deduplication**: May take several seconds with large claim sets

## Common Issues

**"File encoding error"**
- Ensure file is UTF-8 encoded
- Try converting with: `iconv -f ISO-8859-1 -t UTF-8 input.txt > output.txt`

**"API call failed"**
- Check internet connectivity
- Verify API credentials are configured
- Check API quota limits

**"No claims were generated"**
- Verify input text contains factual content
- Try adjusting chunk size
- Check log file for API response details

## Examples

### Example Input
```
The Eiffel Tower was completed in 1889. It stands 330 meters tall and
was the tallest structure in the world until 1930. Located in Paris,
France, it was designed by engineer Gustave Eiffel.
```

### Example Output
```
1. The Eiffel Tower was completed in 1889
2. The Eiffel Tower stands 330 meters tall
3. The Eiffel Tower was the tallest structure in the world until 1930
4. The Eiffel Tower is located in Paris, France
5. The Eiffel Tower was designed by engineer Gustave Eiffel
```

## License

This code is provided as-is for educational and research purposes.

## Contributing

This is a standalone utility. For issues or improvements, consider:
- Validating claims against external knowledge bases
- Implementing semantic deduplication using embeddings
- Adding support for other file formats (PDF, DOCX)
- Optimizing deduplication algorithm
- Adding batch processing capabilities

## Version History

- **1.0.0** - Initial release with basic claim generation
- **1.1.0** - Added configurable parameters and improved error handling
- **1.2.0** - Refactored to return structured data; added formatting helper
- **1.3.0** - Migrated to dataclass-based configuration (`ClaimsConfig`); simplified `__init__` to accept single config parameter

## Support

For issues related to:
- **Google Gemini API**: See [Google AI documentation](https://ai.google.dev/)
- **This tool**: Check logs for detailed error messages
