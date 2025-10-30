"""
Custom exceptions for the GemKit library.
"""


class GeminiAudioAnalyzerError(Exception):
    """Base exception for all errors raised by the GeminiAudioAnalyzer."""
    pass


class APIError(GeminiAudioAnalyzerError):
    """Raised when a request to the Gemini API fails after all retry attempts."""
    pass


class GeminiAPIError(APIError):
    """Raised for specific errors returned by the Gemini API."""
    pass


class APIKeyError(GeminiAudioAnalyzerError):
    """Raised when the Gemini API key is invalid, missing, or unauthorized."""
    pass


class FileValidationError(GeminiAudioAnalyzerError):
    """Base exception for errors related to file validation."""
    pass


class FileSizeError(FileValidationError):
    """Raised when the size of an input file exceeds the configured maximum limit."""
    pass


class FileDurationError(FileValidationError):
    """Raised when the duration of an audio or video file exceeds the configured limit."""
    pass


class UnsupportedFormatError(FileValidationError):
    """Raised when an input file has an unsupported format."""
    pass


class RateLimitError(APIError):
    """Raised when the number of requests to the Gemini API exceeds the rate limit."""
    pass


class QuotaExceededError(APIError):
    """Raised when the API usage quota has been exceeded."""
    pass


class UploadError(APIError):
    """Raised when a file upload to the Gemini API fails."""
    pass


class ResourceCleanupError(GeminiAudioAnalyzerError):
    """Raised when there is an error during the cleanup of resources, such as temporary files."""
    pass


class CancellationError(GeminiAudioAnalyzerError):
    """Raised when an operation is cancelled by the user."""
    pass
