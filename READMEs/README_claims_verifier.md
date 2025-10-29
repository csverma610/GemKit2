# Claims Verifier

A production-grade tool for verifying factual claims using Google's Gemini AI with structured outputs, comprehensive error handling, and enterprise features.

## Features

- **Structured Output**: Uses Pydantic models for validated, structured verification results
- **Google Search Grounding**: Optional fact-checking with real-time web search
- **Robust Error Handling**: Automatic retry logic with exponential backoff
- **Rate Limiting**: Configurable delays between batch requests to avoid quota issues
- **Comprehensive Logging**: Multiple log levels with optional file output
- **Input Validation**: Validates and sanitizes claims before processing
- **Batch Processing**: Efficiently process multiple claims from files
- **Production Ready**: API key management, timeouts, file size limits, and more

## Installation

### Requirements

```bash
pip install google-genai pydantic tenacity
```

For testing:
```bash
pip install pytest pytest-mock
```

### API Key Setup

Set your Google API key as an environment variable:

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

Or pass it directly via the `--api-key` argument.

## Usage

### Basic Examples

**Verify a single claim:**
```bash
python claims_verifier.py "The Earth is round"
```

**With Google Search grounding:**
```bash
python claims_verifier.py "Python was created in 1991" --grounding
```

**JSON output:**
```bash
python claims_verifier.py "The sky is blue" --json
```

### Batch Processing

**Process multiple claims from a file:**
```bash
python claims_verifier.py --file claims.txt
```

**With grounding and JSON output:**
```bash
python claims_verifier.py --file claims.txt --grounding --json
```

**Custom rate limiting:**
```bash
python claims_verifier.py --file claims.txt --rate-limit 1.0
```

File format: One claim per line (empty lines are ignored)

### Logging Options

**Set log level:**
```bash
python claims_verifier.py "claim" --log-level DEBUG
```

Available levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

**Log to file:**
```bash
python claims_verifier.py --file claims.txt --log-file verification.log
```

### Advanced Options

**Custom model:**
```bash
python claims_verifier.py "claim" --model gemini-2.0-flash-exp
```

**All options combined:**
```bash
python claims_verifier.py \
  --file claims.txt \
  --grounding \
  --json \
  --model gemini-2.0-flash-exp \
  --rate-limit 1.0 \
  --log-level INFO \
  --log-file verification.log \
  --api-key "your-key"
```

## Command-Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--claim` | `-c` | Single claim to verify | - |
| `--file` | `-f` | File with claims (one per line) | - |
| `--grounding` | `-g` | Enable Google Search grounding | False |
| `--json` | `-j` | Output in JSON format | False |
| `--model` | `-m` | Gemini model to use | gemini-2.5-flash |
| `--log-level` | - | Logging level | INFO |
| `--log-file` | - | Write logs to file | None |
| `--rate-limit` | - | Delay between batch requests (seconds) | 0.5 |
| `--api-key` | - | Google API key | $GOOGLE_API_KEY |

## Output Format

### Human-Readable Output

```
Summary: The claim is factually correct based on scientific evidence.
Verification Status: True
Evidence: Multiple studies and observations confirm Earth's spherical shape.
Sources: NASA, scientific textbooks, satellite imagery
Truthfulness Score: 0.98
```

### JSON Output

```json
{
  "summary": "The claim is factually correct based on scientific evidence.",
  "verification_status": "True",
  "evidence": "Multiple studies and observations confirm Earth's spherical shape.",
  "sources": "NASA, scientific textbooks, satellite imagery",
  "truthfulness_score": 0.98
}
```

## Verification Status Values

- **True**: The claim is factually correct
- **False**: The claim is factually incorrect
- **Partially True**: The claim contains some truth but is misleading or incomplete
- **Unverifiable**: Insufficient information to verify the claim

## Programmatic Usage

```python
from claims_verifier import ClaimVerifier

# Initialize with API key
verifier = ClaimVerifier(
    model="gemini-2.5-flash",
    api_key="your-api-key"
)

# Verify a single claim
result = verifier.verify_claim(
    "The Earth revolves around the Sun",
    enable_grounding=True
)

print(f"Status: {result.verification_status.value}")
print(f"Score: {result.truthfulness_score}")
print(f"Evidence: {result.evidence}")

# Batch verification
claims = [
    "Water boils at 100°C at sea level",
    "The Moon is made of cheese",
    "Python was released in 1991"
]

results = verifier.verify_claims_batch(
    claims,
    enable_grounding=True,
    rate_limit_delay=0.5
)

for claim, result in results.items():
    print(f"{claim}: {result.verification_status.value}")
```

## Error Handling

The tool includes comprehensive error handling:

- **Validation Errors**: Invalid or empty claims
- **API Errors**: Authentication, quota, rate limits
- **Network Errors**: Automatic retry with exponential backoff (up to 3 attempts)
- **File Errors**: Missing files, permission issues, size limits

All errors are logged with appropriate severity levels.

## Configuration Limits

- **Maximum claim length**: 10,000 characters
- **Maximum file size**: 10 MB
- **Request timeout**: 60 seconds (configurable)
- **Retry attempts**: 3 (for transient errors)
- **Exponential backoff**: 2-10 seconds between retries

## Testing

Run the test suite:

```bash
pytest test_claims_verifier.py -v
```

Run with coverage:
```bash
pytest test_claims_verifier.py -v --cov=claims_verifier --cov-report=html
```

## Best Practices

1. **Use grounding for factual claims**: Enable `--grounding` for verifiable facts
2. **Rate limiting**: Keep default rate limiting (0.5s) to avoid quota issues
3. **Batch processing**: Use file input for multiple claims to leverage rate limiting
4. **Logging**: Enable DEBUG logging when troubleshooting issues
5. **Error handling**: Check return codes (0 = success, 1 = error) in scripts
6. **API key security**: Never commit API keys; use environment variables

## Production Deployment

### Environment Variables

```bash
export GOOGLE_API_KEY="your-api-key"
export LOG_LEVEL="INFO"
export LOG_FILE="/var/log/claims_verifier.log"
```

### Systemd Service Example

```ini
[Unit]
Description=Claims Verifier Service
After=network.target

[Service]
Type=simple
User=claims-verifier
WorkingDirectory=/opt/claims-verifier
Environment="GOOGLE_API_KEY=your-key"
ExecStart=/usr/bin/python3 claims_verifier.py --file /var/spool/claims/pending.txt --log-file /var/log/claims.log
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### Docker Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY claims_verifier.py .

ENV GOOGLE_API_KEY=""
ENV LOG_LEVEL="INFO"

ENTRYPOINT ["python", "claims_verifier.py"]
```

## Troubleshooting

**API Key Errors**:
```
Error: API key not found
```
Solution: Set `GOOGLE_API_KEY` environment variable or use `--api-key`

**Quota Exceeded**:
```
Error: API quota exceeded. Please try again later
```
Solution: Increase rate limiting delay or wait for quota reset

**Permission Denied**:
```
Error: Permission denied - check API key
```
Solution: Verify your API key has access to the Gemini API

**File Too Large**:
```
Error: File size exceeds maximum
```
Solution: Split large files into smaller batches (< 10 MB)

## License

This tool is provided as-is for claim verification purposes.

## Contributing

When contributing, ensure:
- All tests pass: `pytest test_claims_verifier.py -v`
- Code follows PEP 8 style guidelines
- New features include tests and documentation
- Logging is used appropriately (not print statements)

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs with `--log-level DEBUG`
3. Verify API key and quota status
4. Check Google Gemini API documentation
