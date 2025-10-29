"""
GeminiAudioAnalyzer - Production-ready audio analysis using Google's Gemini API

This module provides a comprehensive, production-quality interface for analyzing
audio files using Google's Gemini API with features including:
- Rate limiting and retry logic with exponential backoff
- Resource management and automatic cleanup
- File validation (size, duration, format)
- Async batch processing
- Progress callbacks and cancellation support
- Comprehensive error handling
- Metrics and monitoring hooks
"""

import asyncio
import logging
import os
import subprocess
import tempfile
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, Event
from typing import Optional, List, Dict, Any, Union, Callable, Protocol

from google import genai
from google.api_core import exceptions as google_exceptions

from Audio.audio_extractor import AudioExtractor
from analysis_result import AnalysisResult
from config import GeminiConfig, FileMetadata
from exceptions import (
    APIError, APIKeyError, FileValidationError, FileSizeError,
    FileDurationError, UnsupportedFormatError, RateLimitError,
    QuotaExceededError, UploadError, ResourceCleanupError, CancellationError
)


class ProgressCallback(Protocol):
    """Protocol for progress callbacks"""
    def __call__(self, current: int, total: int, message: str) -> None:
        ...


class MetricsCollector:
    """Collects metrics for monitoring"""

    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.requests_retried = 0
        self.total_processing_time = 0.0
        self.files_uploaded = 0
        self.files_cleaned = 0
        self.lock = Lock()

    def record_request(self, success: bool, processing_time: float, retried: bool = False):
        """Record a request metric"""
        with self.lock:
            self.requests_total += 1
            if success:
                self.requests_success += 1
            else:
                self.requests_failed += 1
            if retried:
                self.requests_retried += 1
            self.total_processing_time += processing_time

    def record_upload(self):
        """Record a file upload"""
        with self.lock:
            self.files_uploaded += 1

    def record_cleanup(self):
        """Record a file cleanup"""
        with self.lock:
            self.files_cleaned += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            return {
                'requests_total': self.requests_total,
                'requests_success': self.requests_success,
                'requests_failed': self.requests_failed,
                'requests_retried': self.requests_retried,
                'success_rate': self.requests_success / self.requests_total if self.requests_total > 0 else 0,
                'avg_processing_time': self.total_processing_time / self.requests_total if self.requests_total > 0 else 0,
                'files_uploaded': self.files_uploaded,
                'files_cleaned': self.files_cleaned,
            }


class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.window = 60.0  # 1 minute
        self.requests = deque()
        self.lock = Lock()

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()

        while True:
            with self.lock:
                now = time.time()

                # Remove old requests outside the window
                while self.requests and self.requests[0] < now - self.window:
                    self.requests.popleft()

                # Check if we can make a request
                if len(self.requests) < self.max_requests:
                    self.requests.append(now)
                    return True

            # Check timeout
            if timeout is not None and time.time() - start_time >= timeout:
                return False

            # Wait a bit before trying again
            time.sleep(0.1)


class GeminiAudioAnalyzer:
    """
    Production-ready audio analyzer using Google's Gemini API.

    Features:
    - Comprehensive error handling with custom exceptions
    - Rate limiting and retry logic with exponential backoff
    - Resource management with context manager support
    - File validation (size, duration, format)
    - Async batch processing with concurrency control
    - Progress callbacks and cancellation support
    - Metrics collection for monitoring

    Example:
        >>> config = GeminiConfig(api_key="your-key", max_file_size_mb=50)
        >>> with GeminiAudioAnalyzer(config) as analyzer:
        >>>     result = analyzer.transcribe("audio.mp3")
        >>>     print(result.result)
    """

    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

    DEFAULT_PROMPTS = {
        'transcribe': 'Please transcribe this audio file word-for-word. Include speaker labels if multiple speakers are present.',
        'describe': 'Describe this audio clip in detail, including content, tone, background sounds, and any notable features.',
        'sentiment': 'Analyze the sentiment and emotional tone of this audio. Identify the primary emotions expressed.',
        'summary': 'Provide a concise summary of the key points discussed in this audio.',
        'music_analysis': 'Analyze this music: identify genre, tempo, instruments, mood, and musical characteristics.',
        'speaker_count': 'How many different speakers are in this audio? Describe their characteristics.',
        'language': 'Identify the language(s) spoken in this audio and provide a confidence level.',
    }

    def __init__(
        self,
        config: Optional[GeminiConfig] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        Initialize the GeminiAudioAnalyzer.

        Args:
            config: Configuration object. If None, uses defaults.
            api_key: Google API key (overrides config). If None, uses GEMINI_API_KEY env var.
            model: Gemini model to use (overrides config).

        Raises:
            APIKeyError: If API key is not provided or invalid
        """
        # Initialize configuration
        self.config = config or GeminiConfig()

        # Override with direct parameters
        if api_key:
            self.config.api_key = api_key
        if model:
            self.config.model = model

        # Get API key from environment if not provided
        if not self.config.api_key:
            self.config.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

        if not self.config.api_key:
            raise APIKeyError(
                "API key not provided. Set GEMINI_API_KEY or GOOGLE_API_KEY "
                "environment variable, pass it as an argument, or include it in config."
            )

        # Setup logging
        self._setup_logging()

        # Configure Gemini API
        try:
            genai.configure(api_key=self.config.api_key)
            self.model = genai.GenerativeModel(self.config.model)
        except Exception as e:
            raise APIKeyError(f"Failed to configure Gemini API: {str(e)}")

        # Initialize components
        self.audio_extractor = AudioExtractor(log_level=logging.getLevelName(self.config.log_level))
        self.rate_limiter = RateLimiter(self.config.max_requests_per_minute)
        self.metrics = MetricsCollector()

        # Resource tracking
        self.uploaded_files: Dict[str, FileMetadata] = {}
        self.temp_files: List[Path] = []
        self.lock = Lock()

        # Cancellation support
        self._cancelled = Event()

        self.logger.info(
            f"GeminiAudioAnalyzer initialized with model: {self.config.model}, "
            f"max_file_size: {self.config.max_file_size_mb}MB"
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources"""
        self.cleanup_all_resources()
        return False

    # Public API methods
    def transcribe(self, file_path: Union[str, Path], **kwargs) -> AnalysisResult:
        """Transcribe an audio file."""
        return self.analyze(file_path, analysis_type='transcribe', **kwargs)

    def describe(self, file_path: Union[str, Path], **kwargs) -> AnalysisResult:
        """Get a detailed description of an audio file."""
        return self.analyze(file_path, analysis_type='describe', **kwargs)

    def analyze_sentiment(self, file_path: Union[str, Path], **kwargs) -> AnalysisResult:
        """Analyze the sentiment of an audio file."""
        return self.analyze(file_path, analysis_type='sentiment', **kwargs)

    def summarize(self, file_path: Union[str, Path], **kwargs) -> AnalysisResult:
        """Generate a summary of an audio file."""
        return self.analyze(file_path, analysis_type='summary', **kwargs)

    def analyze_music(self, file_path: Union[str, Path], **kwargs) -> AnalysisResult:
        """Analyze musical characteristics of an audio file."""
        return self.analyze(file_path, analysis_type='music_analysis', **kwargs)

    def detect_language(self, file_path: Union[str, Path], **kwargs) -> AnalysisResult:
        """Detect the language spoken in an audio file."""
        return self.analyze(file_path, analysis_type='language', **kwargs)

    def count_speakers(self, file_path: Union[str, Path], **kwargs) -> AnalysisResult:
        """Count and describe speakers in an audio file."""
        return self.analyze(file_path, analysis_type='speaker_count', **kwargs)

    def analyze(
        self,
        file_path: Union[str, Path],
        prompt: Optional[str] = None,
        analysis_type: str = 'describe',
        progress_callback: Optional[ProgressCallback] = None,
    ) -> AnalysisResult:
        """
        Analyze an audio file with a specific prompt.

        Args:
            file_path: Path to the audio or video file
            prompt: Custom prompt (if None, uses predefined prompt for analysis_type)
            analysis_type: Type of analysis
            progress_callback: Optional callback for progress updates

        Returns:
            AnalysisResult object containing the analysis

        Raises:
            CancellationError: If operation was cancelled
            FileValidationError: If file validation fails
            APIError: If API request fails
        """
        if self._cancelled.is_set():
            raise CancellationError("Operation was cancelled")

        start_time = time.time()
        uploaded_file = None
        retried = False

        try:
            # Get the prompt
            if prompt is None:
                prompt = self.DEFAULT_PROMPTS.get(
                    analysis_type,
                    self.DEFAULT_PROMPTS['describe']
                )

            # Progress callback
            if progress_callback and self.config.enable_progress_callbacks:
                progress_callback(1, 3, f"Validating file: {file_path}")

            # Validate the file
            validated_path = self.validate_file(file_path)

            # Progress callback
            if progress_callback and self.config.enable_progress_callbacks:
                progress_callback(2, 3, f"Uploading file: {file_path}")

            # Upload the file with retry logic
            uploaded_file = self._upload_with_retry(validated_path)

            # Progress callback
            if progress_callback and self.config.enable_progress_callbacks:
                progress_callback(3, 3, f"Analyzing file: {file_path}")

            # Generate content with retry logic
            self.logger.info(f"Analyzing audio with type: {analysis_type}")
            response = self._generate_content_with_retry(prompt, uploaded_file)

            # Record success metrics
            processing_time = time.time() - start_time
            self.metrics.record_request(True, processing_time, retried)

            result = AnalysisResult(
                file_path=str(file_path),
                analysis_type=analysis_type,
                result=response.text,
                model=self.config.model,
                success=True
            )

            self.logger.info(
                f"Analysis completed successfully for: {file_path} "
                f"in {processing_time:.2f}s"
            )
            return result

        except CancellationError:
            raise

        except FileValidationError as e:
            self.logger.error(f"File validation failed for {file_path}: {str(e)}")
            processing_time = time.time() - start_time
            self.metrics.record_request(False, processing_time, retried)
            return AnalysisResult(
                file_path=str(file_path),
                analysis_type=analysis_type,
                result="",
                model=self.config.model,
                success=False,
                error=f"File validation error: {str(e)}"
            )

        except (APIError, Exception) as e:
            self.logger.error(f"Analysis failed for {file_path}: {str(e)}")
            processing_time = time.time() - start_time
            self.metrics.record_request(False, processing_time, retried)
            return AnalysisResult(
                file_path=str(file_path),
                analysis_type=analysis_type,
                result="",
                model=self.config.model,
                success=False,
                error=str(e)
            )

        finally:
            # Cleanup uploaded file if configured
            if uploaded_file and self.config.cleanup_uploaded_files:
                self._schedule_file_cleanup(uploaded_file)

    async def batch_analyze_async(
        self,
        file_paths: List[Union[str, Path]],
        analysis_type: str = 'describe',
        custom_prompt: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        max_concurrent: Optional[int] = None,
    ) -> List[AnalysisResult]:
        """
        Analyze multiple audio files asynchronously with concurrency control.

        Args:
            file_paths: List of audio file paths
            analysis_type: Type of analysis to perform
            custom_prompt: Optional custom prompt to use for all files
            progress_callback: Optional callback for progress updates
            max_concurrent: Maximum concurrent operations (overrides config)

        Returns:
            List of AnalysisResult objects
        """
        if self._cancelled.is_set():
            raise CancellationError("Operation was cancelled")

        results = []
        total = len(file_paths)
        max_concurrent = max_concurrent or self.config.batch_concurrent_limit

        self.logger.info(
            f"Starting async batch analysis of {total} files "
            f"(max_concurrent={max_concurrent})"
        )

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_one(idx: int, file_path: Union[str, Path]) -> AnalysisResult:
            """Analyze a single file with semaphore"""
            async with semaphore:
                if self._cancelled.is_set():
                    raise CancellationError("Operation was cancelled")

                self.logger.info(f"Processing file {idx + 1}/{total}: {file_path}")

                # Run in thread pool since genai is synchronous
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self.analyze(
                        file_path,
                        prompt=custom_prompt,
                        analysis_type=analysis_type,
                        progress_callback=None  # Disable individual callbacks
                    )
                )

                if progress_callback and self.config.enable_progress_callbacks:
                    progress_callback(idx + 1, total, f"Completed: {file_path}")

                return result

        # Process all files
        try:
            tasks = [analyze_one(idx, fp) for idx, fp in enumerate(file_paths)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions in results
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"File {file_paths[idx]} failed: {result}")
                    results[idx] = AnalysisResult(
                        file_path=str(file_paths[idx]),
                        analysis_type=analysis_type,
                        result="",
                        model=self.config.model,
                        success=False,
                        error=str(result)
                    )

        except CancellationError:
            self.logger.warning("Batch analysis cancelled")
            raise

        successful = sum(1 for r in results if r.success)
        self.logger.info(f"Batch analysis complete: {successful}/{total} successful")

        return results

    def batch_analyze(
        self,
        file_paths: List[Union[str, Path]],
        analysis_type: str = 'describe',
        custom_prompt: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        max_concurrent: Optional[int] = None,
    ) -> List[AnalysisResult]:
        """
        Analyze multiple audio files with concurrency control (synchronous wrapper).

        Args:
            file_paths: List of audio file paths
            analysis_type: Type of analysis to perform
            custom_prompt: Optional custom prompt to use for all files
            progress_callback: Optional callback for progress updates
            max_concurrent: Maximum concurrent operations

        Returns:
            List of AnalysisResult objects
        """
        # Run async version in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run,
                        self.batch_analyze_async(
                            file_paths, analysis_type, custom_prompt,
                            progress_callback, max_concurrent
                        )
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self.batch_analyze_async(
                        file_paths, analysis_type, custom_prompt,
                        progress_callback, max_concurrent
                    )
                )
        except RuntimeError:
            # No event loop, create new one
            return asyncio.run(
                self.batch_analyze_async(
                    file_paths, analysis_type, custom_prompt,
                    progress_callback, max_concurrent
                )
            )

    # Validation methods
    def validate_file(self, file_path: Union[str, Path]) -> Path:
        """
        Comprehensive file validation.

        Args:
            file_path: Path to the file

        Returns:
            Validated Path object

        Raises:
            FileNotFoundError: If file doesn't exist
            UnsupportedFormatError: If format is not supported
            FileSizeError: If file exceeds size limit
            FileDurationError: If duration exceeds limit
        """
        path = Path(file_path)

        # Check existence
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise FileValidationError(f"Path is not a file: {file_path}")

        # Check format
        all_supported = self.SUPPORTED_AUDIO_FORMATS | self.SUPPORTED_VIDEO_FORMATS
        if path.suffix.lower() not in all_supported:
            raise UnsupportedFormatError(
                f"Unsupported format: {path.suffix}. "
                f"Supported audio: {', '.join(self.SUPPORTED_AUDIO_FORMATS)}. "
                f"Supported video: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}"
            )

        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise FileSizeError(
                f"File size ({file_size_mb:.2f}MB) exceeds limit "
                f"({self.config.max_file_size_mb}MB)"
            )

        # Check duration for audio/video files
        if self.config.max_duration_seconds is not None:
            duration = self._get_media_duration(path)
            if duration and duration > self.config.max_duration_seconds:
                raise FileDurationError(
                    f"Duration ({duration:.2f}s) exceeds limit "
                    f"({self.config.max_duration_seconds}s)"
                )

        self.logger.debug(f"Validated file: {file_path} ({file_size_mb:.2f}MB)")
        return path

    def _get_media_duration(self, file_path: Path) -> Optional[float]:
        """
        Get duration of audio/video file using ffprobe.

        Args:
            file_path: Path to media file

        Returns:
            Duration in seconds, or None if unable to determine
        """
        try:
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-show_entries', 'format=duration',
                    '-of', 'default=noprint_wrappers=1:nokey=1',
                    str(file_path)
                ],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
            self.logger.debug(f"Could not determine duration for: {file_path}")
        return None

    # Upload and retry methods
    def _upload_with_retry(self, file_path: Path) -> Any:
        """
        Upload file with retry logic and rate limiting.

        Args:
            file_path: Path to file

        Returns:
            Uploaded file object

        Raises:
            UploadError: If upload fails after retries
        """
        temp_audio_file = None
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            if self._cancelled.is_set():
                raise CancellationError("Operation was cancelled")

            try:
                # Rate limiting
                if not self.rate_limiter.acquire(timeout=self.config.api_timeout):
                    raise RateLimitError("Rate limit timeout")

                # Extract audio from video if needed
                if self.is_video_file(file_path):
                    self.logger.info(f"Video file detected. Extracting audio from: {file_path}")
                    temp_audio_file = self.extract_audio_from_video(file_path)
                    upload_path = temp_audio_file
                else:
                    upload_path = file_path

                # Upload file
                self.logger.info(f"Uploading file (attempt {attempt + 1}): {upload_path}")
                uploaded_file = genai.upload_file(path=str(upload_path))

                # Track uploaded file
                with self.lock:
                    self.uploaded_files[uploaded_file.name] = FileMetadata(
                        file_path=str(file_path),
                        gemini_file_name=uploaded_file.name,
                        upload_time=time.time(),
                        file_size=upload_path.stat().st_size,
                    )

                self.metrics.record_upload()
                self.logger.info(f"File uploaded successfully: {uploaded_file.name}")

                # Cleanup temporary file
                if temp_audio_file and temp_audio_file.exists():
                    self._cleanup_temp_file(temp_audio_file)

                return uploaded_file

            except Exception as e:
                last_exception = e
                self.logger.warning(
                    f"Upload attempt {attempt + 1} failed: {str(e)}"
                )

                # Cleanup temporary file on error
                if temp_audio_file and temp_audio_file.exists():
                    self._cleanup_temp_file(temp_audio_file)

                # Check if should retry
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (self.config.retry_backoff_factor ** attempt)
                    self.logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    self.metrics.record_request(False, 0, retried=True)
                else:
                    break

        raise UploadError(f"Failed to upload file after {self.config.max_retries + 1} attempts: {last_exception}")

    def _generate_content_with_retry(self, prompt: str, uploaded_file: Any) -> Any:
        """
        Generate content with retry logic and rate limiting.

        Args:
            prompt: Analysis prompt
            uploaded_file: Uploaded file object

        Returns:
            Response object

        Raises:
            APIError: If generation fails after retries
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            if self._cancelled.is_set():
                raise CancellationError("Operation was cancelled")

            try:
                # Rate limiting
                if not self.rate_limiter.acquire(timeout=self.config.api_timeout):
                    raise RateLimitError("Rate limit timeout")

                # Generate content
                self.logger.debug(f"Generating content (attempt {attempt + 1})")
                response = self.model.generate_content(
                    contents=[prompt, uploaded_file]
                )

                return response

            except google_exceptions.ResourceExhausted as e:
                last_exception = QuotaExceededError(f"API quota exceeded: {str(e)}")
                self.logger.error(str(last_exception))
                raise last_exception

            except google_exceptions.TooManyRequests as e:
                last_exception = RateLimitError(f"Rate limit exceeded: {str(e)}")
                self.logger.warning(
                    f"Rate limit hit on attempt {attempt + 1}: {str(e)}"
                )

                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (self.config.retry_backoff_factor ** attempt)
                    self.logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    self.metrics.record_request(False, 0, retried=True)
                else:
                    raise last_exception

            except Exception as e:
                last_exception = APIError(f"API request failed: {str(e)}")
                self.logger.warning(
                    f"Generation attempt {attempt + 1} failed: {str(e)}"
                )

                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (self.config.retry_backoff_factor ** attempt)
                    self.logger.info(f"Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    self.metrics.record_request(False, 0, retried=True)
                else:
                    break

        raise last_exception or APIError("Content generation failed")

    # Resource management
    def is_video_file(self, file_path: Union[str, Path]) -> bool:
        """Check if file is a video file."""
        return Path(file_path).suffix.lower() in self.SUPPORTED_VIDEO_FORMATS

    def extract_audio_from_video(
        self,
        video_path: Union[str, Path],
        output_format: str = 'mp3'
    ) -> Path:
        """
        Extract audio from video file.

        Args:
            video_path: Path to video file
            output_format: Audio format

        Returns:
            Path to extracted audio file
        """
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / f"{Path(video_path).stem}.{output_format}"

        # Track temp file
        with self.lock:
            self.temp_files.append(output_path)

        return self.audio_extractor.extract_audio(
            video_path=video_path,
            output_path=output_path,
            audio_format=output_format,
            audio_quality='high'
        )

    def _cleanup_temp_file(self, file_path: Path):
        """Cleanup a temporary file."""
        try:
            if file_path.exists():
                file_path.unlink()
                # Try to remove parent directory if empty
                try:
                    file_path.parent.rmdir()
                except OSError:
                    pass  # Directory not empty
                self.logger.debug(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            self.logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

    def _schedule_file_cleanup(self, uploaded_file: Any):
        """Schedule cleanup of uploaded file from Gemini."""
        try:
            genai.delete_file(uploaded_file.name)
            with self.lock:
                if uploaded_file.name in self.uploaded_files:
                    del self.uploaded_files[uploaded_file.name]
            self.metrics.record_cleanup()
            self.logger.debug(f"Cleaned up uploaded file: {uploaded_file.name}")
        except Exception as e:
            self.logger.warning(
                f"Failed to cleanup uploaded file {uploaded_file.name}: {e}"
            )

    def cleanup_all_resources(self):
        """Cleanup all tracked resources."""
        self.logger.info("Cleaning up all resources...")

        # Cleanup temporary files
        with self.lock:
            for temp_file in self.temp_files:
                self._cleanup_temp_file(temp_file)
            self.temp_files.clear()

        # Cleanup uploaded files
        if self.config.cleanup_uploaded_files:
            with self.lock:
                for file_name in list(self.uploaded_files.keys()):
                    try:
                        genai.delete_file(file_name)
                        del self.uploaded_files[file_name]
                        self.metrics.record_cleanup()
                    except Exception as e:
                        self.logger.warning(f"Failed to cleanup {file_name}: {e}")

        self.logger.info("Resource cleanup complete")

    def cancel(self):
        """Cancel ongoing operations."""
        self.logger.warning("Cancelling all operations...")
        self._cancelled.set()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return self.metrics.get_stats()

    # Utility methods (from original)
    def save_results(
        self,
        results: Union[AnalysisResult, List[AnalysisResult]],
        output_file: Union[str, Path],
        format: str = 'json'
    ):
        """Save analysis results to a file. (Same implementation as original)"""
        import json

        if isinstance(results, AnalysisResult):
            results = [results]

        output_path = Path(output_file)

        # Validate output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format == 'json':
                data = [
                    {
                        'file_path': r.file_path,
                        'analysis_type': r.analysis_type,
                        'result': r.result,
                        'model': r.model,
                        'success': r.success,
                        'error': r.error
                    }
                    for r in results
                ]

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            elif format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for r in results:
                        f.write(f"{'=' * 80}\n")
                        f.write(f"File: {r.file_path}\n")
                        f.write(f"Analysis Type: {r.analysis_type}\n")
                        f.write(f"Model: {r.model}\n")
                        f.write(f"Success: {r.success}\n")
                        if r.error:
                            f.write(f"Error: {r.error}\n")
                        f.write(f"\nResult:\n{r.result}\n")
                        f.write(f"{'=' * 80}\n\n")

            self.logger.info(f"Results saved to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise

    # Private methods
    def _setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        log_level = getattr(logging, self.config.log_level.upper())
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

            # Add file handler if configured
            if self.config.log_file:
                file_handler = logging.FileHandler(self.config.log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)


# Example usage
if __name__ == "__main__":
    import sys

    # Example 1: Basic usage with context manager
    config = GeminiConfig(
        max_file_size_mb=50,
        max_retries=3,
        log_level="INFO"
    )

    with GeminiAudioAnalyzer(config) as analyzer:
        if len(sys.argv) > 1:
            file = sys.argv[1]

            # Simple transcription
            result = analyzer.transcribe(file)
            print(result.result)

            # Get metrics
            metrics = analyzer.get_metrics()
            print(f"\nMetrics: {metrics}")
        else:
            print("Usage: python gemini_audio_analyzer_v2.py <audio_file>")

    # Example 2: Batch processing with progress callback
    # def progress_callback(current, total, message):
    #     print(f"[{current}/{total}] {message}")
    #
    # audio_files = ["file1.mp3", "file2.wav", "file3.m4a"]
    # with GeminiAudioAnalyzer(config) as analyzer:
    #     results = analyzer.batch_analyze(
    #         audio_files,
    #         analysis_type='transcribe',
    #         progress_callback=progress_callback,
    #         max_concurrent=3
    #     )
    #     analyzer.save_results(results, "batch_results.json")
