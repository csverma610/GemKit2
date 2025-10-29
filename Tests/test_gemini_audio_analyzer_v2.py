"""
Comprehensive unit tests for GeminiAudioAnalyzer V2 (Production version)
Run with: pytest test_gemini_audio_analyzer_v2.py -v
"""

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, PropertyMock

from config import GeminiConfig
from exceptions import (
    APIKeyError, FileSizeError, FileDurationError, UnsupportedFormatError,
    RateLimitError, CancellationError, UploadError
)
from gemini_audio_analyzer_v2 import GeminiAudioAnalyzer, RateLimiter, MetricsCollector
from analysis_result import AnalysisResult


class TestGeminiConfig(unittest.TestCase):
    """Test configuration management"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GeminiConfig()
        self.assertEqual(config.model, "gemini-2.5-flash")
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.max_file_size_mb, 100.0)

    def test_custom_config(self):
        """Test custom configuration"""
        config = GeminiConfig(
            model="gemini-pro",
            max_retries=5,
            max_file_size_mb=50.0
        )
        self.assertEqual(config.model, "gemini-pro")
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.max_file_size_mb, 50.0)

    def test_invalid_config(self):
        """Test invalid configuration raises errors"""
        with self.assertRaises(ValueError):
            GeminiConfig(max_retries=-1)

        with self.assertRaises(ValueError):
            GeminiConfig(max_file_size_mb=-10)

        with self.assertRaises(ValueError):
            GeminiConfig(retry_backoff_factor=0.5)


class TestRateLimiter(unittest.TestCase):
    """Test rate limiter"""

    def test_rate_limiter_allows_requests(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(max_requests_per_minute=10)

        # Should allow 10 requests immediately
        for _ in range(10):
            self.assertTrue(limiter.acquire(timeout=0.1))

    def test_rate_limiter_blocks_excess(self):
        """Test rate limiter blocks requests over limit"""
        limiter = RateLimiter(max_requests_per_minute=5)

        # Acquire all tokens
        for _ in range(5):
            self.assertTrue(limiter.acquire(timeout=0.1))

        # Next request should timeout
        self.assertFalse(limiter.acquire(timeout=0.1))


class TestMetricsCollector(unittest.TestCase):
    """Test metrics collection"""

    def test_metrics_initialization(self):
        """Test metrics collector initialization"""
        metrics = MetricsCollector()
        stats = metrics.get_stats()

        self.assertEqual(stats['requests_total'], 0)
        self.assertEqual(stats['requests_success'], 0)
        self.assertEqual(stats['requests_failed'], 0)

    def test_record_success(self):
        """Test recording successful requests"""
        metrics = MetricsCollector()
        metrics.record_request(success=True, processing_time=1.5)

        stats = metrics.get_stats()
        self.assertEqual(stats['requests_total'], 1)
        self.assertEqual(stats['requests_success'], 1)
        self.assertEqual(stats['requests_failed'], 0)

    def test_success_rate(self):
        """Test success rate calculation"""
        metrics = MetricsCollector()
        metrics.record_request(success=True, processing_time=1.0)
        metrics.record_request(success=True, processing_time=1.0)
        metrics.record_request(success=False, processing_time=1.0)

        stats = metrics.get_stats()
        self.assertAlmostEqual(stats['success_rate'], 2/3, places=2)


class TestGeminiAudioAnalyzer(unittest.TestCase):
    """Test GeminiAudioAnalyzer"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_audio = Path(self.test_dir) / "test.mp3"
        self.test_audio.touch()
        self.test_audio.write_bytes(b"fake audio data" * 100)

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)

    @patch('gemini_audio_analyzer_v2.genai')
    def test_initialization_with_api_key(self, mock_genai):
        """Test analyzer initialization with API key"""
        config = GeminiConfig(api_key="test-key")

        with patch('gemini_audio_analyzer_v2.AudioExtractor'):
            analyzer = GeminiAudioAnalyzer(config)

        self.assertEqual(analyzer.config.api_key, "test-key")
        mock_genai.configure.assert_called_once_with(api_key="test-key")

    @patch.dict(os.environ, {}, clear=True)
    def test_initialization_without_api_key_fails(self):
        """Test initialization fails without API key"""
        with self.assertRaises(APIKeyError):
            GeminiAudioAnalyzer()

    @patch.dict(os.environ, {'GEMINI_API_KEY': 'env-key'})
    @patch('gemini_audio_analyzer_v2.genai')
    @patch('gemini_audio_analyzer_v2.AudioExtractor')
    def test_initialization_from_env(self, mock_extractor, mock_genai):
        """Test initialization from environment variable"""
        analyzer = GeminiAudioAnalyzer()
        self.assertEqual(analyzer.config.api_key, "env-key")

    @patch('gemini_audio_analyzer_v2.genai')
    @patch('gemini_audio_analyzer_v2.AudioExtractor')
    def test_context_manager(self, mock_extractor, mock_genai):
        """Test context manager usage"""
        config = GeminiConfig(api_key="test-key")

        with GeminiAudioAnalyzer(config) as analyzer:
            self.assertIsNotNone(analyzer)

    @patch('gemini_audio_analyzer_v2.genai')
    @patch('gemini_audio_analyzer_v2.AudioExtractor')
    def test_file_validation_success(self, mock_extractor, mock_genai):
        """Test successful file validation"""
        config = GeminiConfig(api_key="test-key", max_file_size_mb=1.0)
        analyzer = GeminiAudioAnalyzer(config)

        validated = analyzer.validate_file(self.test_audio)
        self.assertEqual(validated, self.test_audio)

    @patch('gemini_audio_analyzer_v2.genai')
    @patch('gemini_audio_analyzer_v2.AudioExtractor')
    def test_file_validation_nonexistent(self, mock_extractor, mock_genai):
        """Test validation fails for nonexistent file"""
        config = GeminiConfig(api_key="test-key")
        analyzer = GeminiAudioAnalyzer(config)

        with self.assertRaises(FileNotFoundError):
            analyzer.validate_file("/nonexistent/file.mp3")

    @patch('gemini_audio_analyzer_v2.genai')
    @patch('gemini_audio_analyzer_v2.AudioExtractor')
    def test_file_validation_unsupported_format(self, mock_extractor, mock_genai):
        """Test validation fails for unsupported format"""
        config = GeminiConfig(api_key="test-key")
        analyzer = GeminiAudioAnalyzer(config)

        test_file = Path(self.test_dir) / "test.xyz"
        test_file.touch()

        with self.assertRaises(UnsupportedFormatError):
            analyzer.validate_file(test_file)

    @patch('gemini_audio_analyzer_v2.genai')
    @patch('gemini_audio_analyzer_v2.AudioExtractor')
    def test_file_validation_size_limit(self, mock_extractor, mock_genai):
        """Test validation fails when file exceeds size limit"""
        config = GeminiConfig(api_key="test-key", max_file_size_mb=0.001)
        analyzer = GeminiAudioAnalyzer(config)

        with self.assertRaises(FileSizeError):
            analyzer.validate_file(self.test_audio)

    @patch('gemini_audio_analyzer_v2.genai')
    @patch('gemini_audio_analyzer_v2.AudioExtractor')
    def test_get_metrics(self, mock_extractor, mock_genai):
        """Test metrics retrieval"""
        config = GeminiConfig(api_key="test-key")
        analyzer = GeminiAudioAnalyzer(config)

        metrics = analyzer.get_metrics()
        self.assertIn('requests_total', metrics)
        self.assertIn('success_rate', metrics)


if __name__ == '__main__':
    unittest.main()
