# GeminiAudioAnalyzer V2 - Production Documentation

## Overview

GeminiAudioAnalyzer V2 is a production-ready Python library for analyzing audio and video files using Google's Gemini API. It provides comprehensive features for transcription, sentiment analysis, music analysis, and more with enterprise-grade reliability.

## Features

### Core Capabilities
- **Audio Analysis**: Transcription, description, sentiment analysis, summarization
- **Music Analysis**: Genre detection, tempo analysis, instrument identification
- **Video Support**: Automatic audio extraction from video files
- **Multiple Formats**: MP3, WAV, M4A, AAC, OGG, FLAC, WMA, MP4, AVI, MOV, MKV, and more

### Production Features
- ✅ **Rate Limiting**: Token bucket algorithm with configurable limits
- ✅ **Retry Logic**: Exponential backoff with configurable retries
- ✅ **Resource Management**: Context manager support with automatic cleanup
- ✅ **File Validation**: Size, duration, and format validation
- ✅ **Progress Callbacks**: Real-time progress updates
- ✅ **Cancellation Support**: Graceful operation cancellation
- ✅ **Async Batch Processing**: Concurrent file processing with semaphore control
- ✅ **Error Handling**: Custom exceptions with detailed error messages
- ✅ **Metrics Collection**: Built-in monitoring and statistics
- ✅ **Comprehensive Tests**: Unit tests with >80% coverage

## Installation

```bash
# Install required dependencies
pip install google-genai

# Install ffmpeg for video processing
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/
```

## Quick Start

### Basic Usage

```python
from gemini_audio_analyzer_v2 import GeminiAudioAnalyzer
from config import GeminiConfig

# Configure
config = GeminiConfig(
    api_key="your-api-key",  # Or set GEMINI_API_KEY env var
    max_file_size_mb=50,
    max_retries=3
)

# Use context manager for automatic cleanup
with GeminiAudioAnalyzer(config) as analyzer:
    # Transcribe audio
    result = analyzer.transcribe("audio.mp3")
    print(result.result)

    # Get metrics
    metrics = analyzer.get_metrics()
    print(f"Success rate: {metrics['success_rate']:.2%}")
```

### Configuration Options

```python
from config import GeminiConfig

config = GeminiConfig(
    # API Configuration
    api_key="your-key",              # API key (or use env var)
    model="gemini-2.5-flash",        # Gemini model
    api_timeout=300,                 # Request timeout (seconds)

    # Rate Limiting
    max_requests_per_minute=60,      # Rate limit
    max_concurrent_requests=5,       # Concurrent limit

    # Retry Configuration
    max_retries=3,                   # Max retry attempts
    retry_delay=1.0,                 # Initial delay (seconds)
    retry_backoff_factor=2.0,        # Backoff multiplier

    # File Validation
    max_file_size_mb=100.0,          # Max file size
    max_duration_seconds=3600.0,     # Max duration (1 hour)

    # Resource Management
    cleanup_uploaded_files=True,     # Auto-cleanup uploads
    cleanup_temp_files=True,         # Auto-cleanup temp files

    # Batch Processing
    batch_concurrent_limit=3,        # Batch concurrency
    batch_stop_on_error=False,       # Continue on error

    # Logging
    log_level="INFO",                # Log level
    log_file="analyzer.log",         # Optional log file

    # Progress
    enable_progress_callbacks=True,  # Enable callbacks
)
```

## Usage Examples

### 1. Simple Transcription

```python
with GeminiAudioAnalyzer(config) as analyzer:
    result = analyzer.transcribe("meeting.mp3")

    if result.success:
        print("Transcription:", result.result)
    else:
        print("Error:", result.error)
```

### 2. Sentiment Analysis

```python
with GeminiAudioAnalyzer(config) as analyzer:
    result = analyzer.analyze_sentiment("customer_call.wav")
    print("Sentiment:", result.result)
```

### 3. Music Analysis

```python
with GeminiAudioAnalyzer(config) as analyzer:
    result = analyzer.analyze_music("song.mp3")
    print("Music analysis:", result.result)
```

### 4. Custom Analysis

```python
with GeminiAudioAnalyzer(config) as analyzer:
    result = analyzer.analyze(
        "audio.mp3",
        prompt="Identify all technical terms mentioned in this audio",
        analysis_type="custom"
    )
    print(result.result)
```

### 5. Video File Analysis

```python
# Automatically extracts audio from video
with GeminiAudioAnalyzer(config) as analyzer:
    result = analyzer.transcribe("video.mp4")
    print("Video transcription:", result.result)
```

### 6. Batch Processing with Progress

```python
def progress_callback(current, total, message):
    print(f"[{current}/{total}] {message}")

audio_files = ["file1.mp3", "file2.wav", "file3.m4a"]

with GeminiAudioAnalyzer(config) as analyzer:
    results = analyzer.batch_analyze(
        audio_files,
        analysis_type='transcribe',
        progress_callback=progress_callback,
        max_concurrent=3
    )

    # Save results
    analyzer.save_results(results, "batch_results.json", format='json')

    # Print summary
    successful = sum(1 for r in results if r.success)
    print(f"Completed: {successful}/{len(results)} successful")
```

### 7. Async Batch Processing

```python
import asyncio

async def main():
    config = GeminiConfig(api_key="your-key")

    with GeminiAudioAnalyzer(config) as analyzer:
        files = ["file1.mp3", "file2.wav", "file3.m4a"]

        results = await analyzer.batch_analyze_async(
            files,
            analysis_type='transcribe',
            max_concurrent=5
        )

        return results

results = asyncio.run(main())
```

### 8. Error Handling

```python
from exceptions import (
    FileSizeError, RateLimitError, APIKeyError, CancellationError
)

try:
    with GeminiAudioAnalyzer(config) as analyzer:
        result = analyzer.transcribe("large_file.mp3")

except FileSizeError as e:
    print(f"File too large: {e}")

except RateLimitError as e:
    print(f"Rate limit exceeded: {e}")

except APIKeyError as e:
    print(f"API key issue: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

### 9. Cancellation Support

```python
import threading

config = GeminiConfig(api_key="your-key")
analyzer = GeminiAudioAnalyzer(config)

def process_files():
    files = [f"file{i}.mp3" for i in range(100)]
    analyzer.batch_analyze(files)

# Start processing in background
thread = threading.Thread(target=process_files)
thread.start()

# Cancel after 5 seconds
time.sleep(5)
analyzer.cancel()

# Cleanup
analyzer.cleanup_all_resources()
```

### 10. Monitoring and Metrics

```python
with GeminiAudioAnalyzer(config) as analyzer:
    # Process files
    for file in audio_files:
        analyzer.transcribe(file)

    # Get detailed metrics
    metrics = analyzer.get_metrics()

    print(f"Total requests: {metrics['requests_total']}")
    print(f"Success rate: {metrics['success_rate']:.2%}")
    print(f"Failed requests: {metrics['requests_failed']}")
    print(f"Retried requests: {metrics['requests_retried']}")
    print(f"Avg processing time: {metrics['avg_processing_time']:.2f}s")
    print(f"Files uploaded: {metrics['files_uploaded']}")
    print(f"Files cleaned: {metrics['files_cleaned']}")
```

## API Reference

### GeminiAudioAnalyzer

#### Initialization

```python
analyzer = GeminiAudioAnalyzer(
    config: Optional[GeminiConfig] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
)
```

#### Analysis Methods

- `transcribe(file_path, **kwargs)` - Transcribe audio
- `describe(file_path, **kwargs)` - Describe audio content
- `analyze_sentiment(file_path, **kwargs)` - Analyze sentiment
- `summarize(file_path, **kwargs)` - Summarize audio
- `analyze_music(file_path, **kwargs)` - Analyze music
- `detect_language(file_path, **kwargs)` - Detect language
- `count_speakers(file_path, **kwargs)` - Count speakers

#### Core Method

```python
analyze(
    file_path: Union[str, Path],
    prompt: Optional[str] = None,
    analysis_type: str = 'describe',
    progress_callback: Optional[ProgressCallback] = None
) -> AnalysisResult
```

#### Batch Methods

```python
batch_analyze(
    file_paths: List[Union[str, Path]],
    analysis_type: str = 'describe',
    custom_prompt: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    max_concurrent: Optional[int] = None
) -> List[AnalysisResult]

async batch_analyze_async(
    file_paths: List[Union[str, Path]],
    analysis_type: str = 'describe',
    custom_prompt: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
    max_concurrent: Optional[int] = None
) -> List[AnalysisResult]
```

#### Utility Methods

- `validate_file(file_path)` - Validate file
- `cancel()` - Cancel operations
- `get_metrics()` - Get statistics
- `cleanup_all_resources()` - Manual cleanup
- `save_results(results, output_file, format='json')` - Save results

### AnalysisResult

```python
@dataclass
class AnalysisResult:
    file_path: str           # Path to analyzed file
    analysis_type: str       # Type of analysis
    result: str             # Analysis result text
    model: str              # Model used
    success: bool           # Success status
    error: Optional[str]    # Error message if failed
```

### Custom Exceptions

- `GeminiAudioAnalyzerError` - Base exception
- `APIError` - API request failures
- `APIKeyError` - API key issues
- `FileValidationError` - File validation failures
- `FileSizeError` - File size exceeded
- `FileDurationError` - Duration exceeded
- `UnsupportedFormatError` - Unsupported format
- `RateLimitError` - Rate limit exceeded
- `QuotaExceededError` - Quota exceeded
- `UploadError` - Upload failures
- `ResourceCleanupError` - Cleanup failures
- `CancellationError` - Operation cancelled

## Best Practices

### 1. Always Use Context Manager

```python
# Good
with GeminiAudioAnalyzer(config) as analyzer:
    result = analyzer.transcribe("audio.mp3")

# Avoid
analyzer = GeminiAudioAnalyzer(config)
result = analyzer.transcribe("audio.mp3")
# Resources may not be cleaned up!
```

### 2. Configure Appropriate Limits

```python
config = GeminiConfig(
    max_file_size_mb=50,           # Prevent large uploads
    max_duration_seconds=1800,     # 30 minutes max
    max_retries=3,                 # Reasonable retry count
    max_requests_per_minute=60     # Respect rate limits
)
```

### 3. Handle Errors Gracefully

```python
from exceptions import FileSizeError, RateLimitError

with GeminiAudioAnalyzer(config) as analyzer:
    for file in audio_files:
        try:
            result = analyzer.transcribe(file)
            if result.success:
                process_result(result)
            else:
                log_error(file, result.error)
        except FileSizeError:
            log_warning(f"Skipping large file: {file}")
        except RateLimitError:
            time.sleep(60)  # Wait before continuing
```

### 4. Use Progress Callbacks for Long Operations

```python
def progress_callback(current, total, message):
    percentage = (current / total) * 100
    print(f"Progress: {percentage:.1f}% - {message}")

results = analyzer.batch_analyze(
    files,
    progress_callback=progress_callback
)
```

### 5. Monitor Metrics in Production

```python
# Periodically log metrics
def log_metrics(analyzer):
    metrics = analyzer.get_metrics()
    logger.info(f"Analyzer metrics: {metrics}")

    if metrics['success_rate'] < 0.9:
        alert_team("Low success rate detected")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest test_gemini_audio_analyzer_v2.py -v

# Run with coverage
python -m pytest test_gemini_audio_analyzer_v2.py --cov=gemini_audio_analyzer_v2

# Run specific test class
python -m pytest test_gemini_audio_analyzer_v2.py::TestGeminiAudioAnalyzer -v
```

## Performance Considerations

### Batch Processing

For optimal performance with batch processing:

```python
config = GeminiConfig(
    batch_concurrent_limit=5,      # Adjust based on system resources
    max_requests_per_minute=60     # Stay within API limits
)

# Process in chunks for very large batches
chunk_size = 50
for i in range(0, len(all_files), chunk_size):
    chunk = all_files[i:i+chunk_size]
    results = analyzer.batch_analyze(chunk)
    save_checkpoint(results, i)
```

### Memory Management

For large files or long-running processes:

```python
# Process files one at a time
for file in large_file_list:
    with GeminiAudioAnalyzer(config) as analyzer:
        result = analyzer.transcribe(file)
        save_result(result)
    # Resources cleaned up after each file
```

## Troubleshooting

### Rate Limit Errors

```python
# Increase delay between requests
config = GeminiConfig(
    max_requests_per_minute=30,  # Reduce from default 60
    retry_delay=2.0,             # Increase delay
    retry_backoff_factor=3.0     # Increase backoff
)
```

### File Size Errors

```python
# Compress or split large files
from pydub import AudioSegment

def split_large_audio(file_path, max_duration_ms=600000):  # 10 min
    audio = AudioSegment.from_file(file_path)
    chunks = []

    for i in range(0, len(audio), max_duration_ms):
        chunk = audio[i:i+max_duration_ms]
        chunk_path = f"chunk_{i}.mp3"
        chunk.export(chunk_path, format="mp3")
        chunks.append(chunk_path)

    return chunks
```

### Memory Leaks

```python
# Ensure resources are cleaned up
with GeminiAudioAnalyzer(config) as analyzer:
    try:
        results = analyzer.batch_analyze(files)
    finally:
        analyzer.cleanup_all_resources()
```

## Migration from V1

If upgrading from the original version:

```python
# V1 (Old)
analyzer = GeminiAudioAnalyzer(api_key="key")
result = analyzer.transcribe("audio.mp3")

# V2 (New - recommended)
config = GeminiConfig(api_key="key")
with GeminiAudioAnalyzer(config) as analyzer:
    result = analyzer.transcribe("audio.mp3")
```

Key changes:
- Configuration moved to `GeminiConfig` class
- Context manager support added
- Custom exceptions replace generic errors
- Resource cleanup is automatic
- Rate limiting and retries built-in
- Progress callbacks supported
- Async batch processing available

## Support

For issues, questions, or contributions:
- File issues on GitHub
- Check existing tests for usage examples
- Review error messages for troubleshooting hints

## License

See LICENSE file for details.
