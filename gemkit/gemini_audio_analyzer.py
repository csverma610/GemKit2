"""
GeminiAudioAnalyzer - A comprehensive audio analysis class using Google's Gemini API
"""

import logging
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Union

from google import genai
from Audio.audio_extractor import AudioExtractor

@dataclass
class AudioPrompts:
    """
    Predefined prompts for common audio analysis tasks.

    Usage:
        prompts = AudioPrompts()
        analyzer.generate_text(prompts.transcribe)
    """
    transcribe: str = 'Please transcribe this audio file word-for-word. Include speaker labels if multiple speakers are present.'
    describe: str = 'Describe this audio clip in detail, including content, tone, background sounds, and any notable features.'
    sentiment: str = 'Analyze the sentiment and emotional tone of this audio. Identify the primary emotions expressed.'
    summary: str = 'Provide a concise summary of the key points discussed in this audio.'
    music_analysis: str = 'Analyze this music: identify genre, tempo, instruments, mood, and musical characteristics.'
    speaker_count: str = 'How many different speakers are in this audio? Describe their characteristics.'
    language: str = 'Identify the language(s) spoken in this audio and provide a confidence level.'


class GeminiAudioAnalyzer:
    """
    A class for analyzing audio files using Google's Gemini API.

    Supports multiple analysis types including transcription, description,
    sentiment analysis, and custom prompts.
    """

    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.m4a', '.aac', '.ogg', '.flac', '.wma'}
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        log_level: int = logging.INFO
    ):
        """
        Initialize the GeminiAudioAnalyzer.

        Args:
            api_key: Google API key. If None, uses GEMINI_API_KEY environment variable.
            model: Gemini model to use for analysis.
            log_level: Logging level (default: INFO).
        """
        self._setup_logging(log_level)

        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "API key not provided. Please set GEMINI_API_KEY "
                "environment variable or pass it as an argument."
            )

        self.client = genai.Client(api_key=api_key)
        self.model_name = model

        # Initialize AudioExtractor for video-to-audio conversion
        self.audio_extractor = AudioExtractor(log_level=log_level)

        # Store uploaded file and its path
        self.uploaded_file: Optional[Any] = None
        self.uploaded_file_path: Optional[Path] = None

        self.logger.info(f"GeminiAudioAnalyzer initialized with model: {self.model_name}")

    # Core methods
    def generate_text(self, prompt: str) -> str:
        """
        Generate text analysis from the currently uploaded audio file.

        Args:
            prompt: The prompt to use for analysis

        Returns:
            Generated text string

        Raises:
            ValueError: If no file has been uploaded
            Exception: If generation fails
        """
        if self.uploaded_file is None:
            raise ValueError("No file has been uploaded. Call upload_file() first.")

        # Generate content
        self.logger.info(f"Generating text with prompt: {prompt[:50]}...")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[prompt, self.uploaded_file]
        )

        self.logger.info(f"Text generation completed successfully")
        return response.text

    def cleanup(self):
        """
        Delete the currently uploaded file from Gemini's storage.

        This method should be called when you're done with the file to free up storage.
        """
        if self.uploaded_file is None:
            self.logger.warning("No file to clean up.")
            return

        try:
            self.logger.info(f"Deleting file: {self.uploaded_file.name}")
            self.client.files.delete(name=self.uploaded_file.name)
            self.uploaded_file = None
            self.uploaded_file_path = None
            self.logger.info("File deleted successfully")
        except Exception as e:
            self.logger.error(f"Failed to delete file: {str(e)}")
            raise

    # Helper methods
    def validate_audio_file(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that the audio file exists and has a supported format.

        Args:
            file_path: Path to the audio file

        Returns:
            Path object of the validated file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check if it's a supported audio or video format
        all_supported = self.SUPPORTED_AUDIO_FORMATS | self.SUPPORTED_VIDEO_FORMATS
        if path.suffix.lower() not in all_supported:
            raise ValueError(
                f"Unsupported format: {path.suffix}. "
                f"Supported audio formats: {', '.join(self.SUPPORTED_AUDIO_FORMATS)}\n"
                f"Supported video formats: {', '.join(self.SUPPORTED_VIDEO_FORMATS)}"
            )

        self.logger.debug(f"Validated file: {file_path}")
        return path

    def upload_file(self, file_path: Union[str, Path]) -> Any:
        """
        Upload an audio file to Gemini. If a video file is provided,
        automatically extracts audio first.

        Only uploads if the file is different from the currently uploaded file.
        If the same file is already uploaded, returns the existing uploaded file.

        Args:
            file_path: Path to the audio or video file

        Returns:
            Uploaded file object

        Raises:
            Exception: If upload fails
        """
        path = self.validate_audio_file(file_path)

        # Check if this file is already uploaded
        if self.uploaded_file is not None and self.uploaded_file_path == path:
            self.logger.info(f"File already uploaded: {path}. Reusing existing upload.")
            return self.uploaded_file

        # If a different file was previously uploaded, clean it up first
        if self.uploaded_file is not None:
            self.logger.info(f"Different file requested. Cleaning up previous upload.")
            self.cleanup()

        temp_audio_file = None

        try:
            # If it's a video file, extract audio first
            if self.is_video_file(path):
                self.logger.info(f"Video file detected. Extracting audio from: {path}")
                temp_audio_file = self.extract_audio_from_video(path)
                upload_path = temp_audio_file
            else:
                upload_path = path

            self.logger.info(f"Uploading file: {upload_path}")
            uploaded_file = self.client.files.upload(path=str(upload_path))
            self.logger.info(f"File uploaded successfully: {uploaded_file.name}")

            # Store the uploaded file and its original path
            self.uploaded_file = uploaded_file
            self.uploaded_file_path = path

            # Clean up temporary file and directory if it was created
            if temp_audio_file and temp_audio_file.exists():
                temp_dir = temp_audio_file.parent
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.debug(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up temporary directory: {cleanup_error}")

            return uploaded_file

        except Exception as e:
            # Clean up temporary file and directory on error
            if temp_audio_file:
                temp_dir = temp_audio_file.parent
                try:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir)
                except Exception as cleanup_error:
                    self.logger.warning(f"Failed to clean up temporary directory: {cleanup_error}")

            self.logger.error(f"Failed to upload file {path}: {str(e)}")
            raise

    def is_video_file(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the file is a video file.

        Args:
            file_path: Path to the file

        Returns:
            True if video file, False otherwise
        """
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_VIDEO_FORMATS

    def extract_audio_from_video(self, video_path: Union[str, Path], output_format: str = 'mp3') -> Path:
        """
        Extract audio from a video file using AudioExtractor.

        Args:
            video_path: Path to the video file
            output_format: Audio format to extract (default: mp3)

        Returns:
            Path to the extracted audio file

        Raises:
            RuntimeError: If ffmpeg is not installed or extraction fails
        """
        # Create a temporary file for the extracted audio
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / f"{Path(video_path).stem}.{output_format}"

        # Use AudioExtractor to extract audio
        return self.audio_extractor.extract_audio(
            video_path=video_path,
            output_path=output_path,
            audio_format=output_format,
            audio_quality='high'
        )

    # Private methods
    def _setup_logging(self, log_level: int):
        """Setup logging configuration"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer and prompts
    analyzer = GeminiAudioAnalyzer()
    prompts = AudioPrompts()

    file = sys.argv[1]

    # Example 1: Upload and transcribe
    try:
        analyzer.upload_file(file)
        result = analyzer.generate_text(prompts.transcribe)
        print(result)
    finally:
        # Always cleanup to free up storage
        analyzer.cleanup()

    # Example 2: Multiple analyses on same file (efficient - no re-upload)
    # try:
    #     analyzer.upload_file("audio.mp3")
    #     transcription = analyzer.generate_text(prompts.transcribe)
    #     sentiment = analyzer.generate_text(prompts.sentiment)
    #     summary = analyzer.generate_text(prompts.summary)
    # finally:
    #     analyzer.cleanup()

    # Example 3: Using custom prompt directly
    # try:
    #     analyzer.upload_file("audio.mp3")
    #     result = analyzer.generate_text("What emotions are expressed in this audio?")
    # finally:
    #     analyzer.cleanup()

    print("\nGeminiAudioAnalyzer class loaded. Import and use as needed.")
