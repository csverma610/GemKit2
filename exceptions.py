"""
Custom exceptions for GeminiAudioAnalyzer
"""


class GeminiAudioAnalyzerError(Exception):
    """Base exception for GeminiAudioAnalyzer"""
    pass


class APIError(GeminiAudioAnalyzerError):
    """Raised when Gemini API request fails"""
    pass


class GeminiAPIError(APIError):
    """Raised when a Gemini API call fails"""
    pass



class APIKeyError(GeminiAudioAnalyzerError):
    """Raised when API key is invalid or missing"""
    pass


class FileValidationError(GeminiAudioAnalyzerError):
    """Raised when file validation fails"""
    pass


class FileSizeError(FileValidationError):
    """Raised when file size exceeds limits"""
    pass


class FileDurationError(FileValidationError):
    """Raised when audio/video duration exceeds limits"""
    pass


class UnsupportedFormatError(FileValidationError):
    """Raised when file format is not supported"""
    pass


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded"""
    pass


class QuotaExceededError(APIError):
    """Raised when API quota is exceeded"""
    pass


class UploadError(APIError):
    """Raised when file upload to Gemini fails"""
    pass


class ResourceCleanupError(GeminiAudioAnalyzerError):
    """Raised when resource cleanup fails"""
    pass


class CancellationError(GeminiAudioAnalyzerError):
    """Raised when operation is cancelled"""
    pass
