"""
Unit tests for gemini_audio_analyzer.py
Run with: pytest test_gemini_audio_analyzer.py -v
"""

import os

import pytest

from pathlib import Path

from unittest.mock import patch, MagicMock, call



from gemini_audio_analyzer import GeminiAudioAnalyzer

from analysis_result import AnalysisResult





class TestGeminiAudioAnalyzerInitialization:

    """Tests for the initialization of the GeminiAudioAnalyzer."""



    def test_init_with_api_key_arg(self):

        """Test initialization with an explicit API key argument."""

                with patch('gemini_audio_analyzer.genai') as mock_genai, patch('gemini_audio_analyzer.AudioExtractor'):

            GeminiAudioAnalyzer(api_key="explicit_key")

            mock_genai.configure.assert_called_once_with(api_key="explicit_key")



    def test_init_with_env_var(self):

        """Test initialization using an environment variable for the API key."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key"}):

                        with patch('gemini_audio_analyzer.genai') as mock_genai, patch('gemini_audio_analyzer.AudioExtractor'):

                GeminiAudioAnalyzer()

                mock_genai.configure.assert_called_once_with(api_key="env_key")



    def test_init_no_api_key_raises_error(self):

        """Test that initialization raises ValueError if no API key is found."""

        with patch.dict(os.environ, {}, clear=True):

            with pytest.raises(ValueError, match="API key not provided"):

                GeminiAudioAnalyzer()



    def test_init_custom_model(self):

        """Test initialization with a custom model name."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}):

                        with patch('gemini_audio_analyzer.genai') as mock_genai, patch('gemini_audio_analyzer.AudioExtractor'):

                analyzer = GeminiAudioAnalyzer(model="custom-model")

                assert analyzer.model_name == "custom-model"

                mock_genai.GenerativeModel.assert_called_once_with("custom-model")





class TestAnalyze:

    """Tests for the core analyze method."""



    @patch('gemini_audio_analyzer.GeminiAudioAnalyzer.upload_file')

    def test_analyze_success(self, mock_upload):

        """Test a successful analysis call."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            mock_upload.return_value = "mock_uploaded_file"

            mock_response = MagicMock()

            mock_response.text = "This is a test transcription."

            analyzer.model.generate_content.return_value = mock_response



            result = analyzer.analyze("fake/path.mp3", analysis_type="transcribe")



            assert result.success is True

            assert result.result == "This is a test transcription."

            assert result.analysis_type == "transcribe"

            analyzer.model.generate_content.assert_called_once()



    @patch('gemini_audio_analyzer.GeminiAudioAnalyzer.upload_file', side_effect=Exception("Upload failed"))

    def test_analyze_failure(self, mock_upload):

        """Test an analysis call that fails during upload."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            result = analyzer.analyze("fake/path.mp3", analysis_type="transcribe")



            assert result.success is False

            assert result.result == ""

            assert result.error == "Upload failed"

            analyzer.model.generate_content.assert_not_called()





class TestFileUpload:

    """Tests for file validation and uploading."""



    @patch('gemini_audio_analyzer.Path.exists', return_value=True)

    @patch('gemini_audio_analyzer.Path.is_file', return_value=True)

    @patch('gemini_audio_analyzer.genai.upload_file')

    def test_upload_audio_file_success(self, mock_upload, mock_is_file, mock_exists):

        """Test successfully uploading a standard audio file."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            mock_uploaded = MagicMock()

            mock_uploaded.name = "uploaded_audio.mp3"

            mock_upload.return_value = mock_uploaded



            analyzer.upload_file("fake/audio.mp3")

            mock_upload.assert_called_once_with(path='fake/audio.mp3')



    @patch('gemini_audio_analyzer.Path.exists', return_value=True)

    @patch('gemini_audio_analyzer.Path.is_file', return_value=True)

    @patch('gemini_audio_analyzer.GeminiAudioAnalyzer.extract_audio_from_video')

    @patch('gemini_audio_analyzer.genai.upload_file')

    def test_upload_video_file_extracts_audio(self, mock_upload, mock_extract, mock_is_file, mock_exists):

        """Test that a video file triggers audio extraction before upload."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            # Mock the temporary path created for the extracted audio

            temp_audio_path = Path("/tmp/fake_video.mp3")

            mock_extract.return_value = temp_audio_path



            # Mock the unlink and rmdir calls for cleanup

            with patch.object(Path, 'unlink'), patch.object(Path, 'rmdir'):

                analyzer.upload_file("fake/video.mp4")



                mock_extract.assert_called_once_with(Path("fake/video.mp4"))

                mock_upload.assert_called_once_with(path=str(temp_audio_path))



    def test_validate_file_not_found(self):

        """Test that a non-existent file raises FileNotFoundError."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            with pytest.raises(FileNotFoundError):

                analyzer.validate_audio_file("non_existent_file.mp3")



    def test_validate_unsupported_format(self):

        """Test that an unsupported file format raises ValueError."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            with patch('pathlib.Path.exists', return_value=True), \

                 patch('pathlib.Path.is_file', return_value=True):

                with pytest.raises(ValueError, match="Unsupported format"):

                    analyzer.validate_audio_file("document.txt")





class TestBatchProcessing:

    """Tests for batch analysis and result saving."""



    def test_batch_analyze(self):

        """Test batch analysis of multiple files."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            files = ["file1.mp3", "file2.wav"]

            mock_result = AnalysisResult(file_path="", analysis_type="describe", result="mock result", model="test", success=True)



            with patch.object(analyzer, 'analyze', return_value=mock_result) as mock_analyze:

                results = analyzer.batch_analyze(files, analysis_type="describe")



                assert len(results) == 2

                assert mock_analyze.call_count == 2

                mock_analyze.assert_has_calls([

                    call("file1.mp3", prompt=None, analysis_type='describe'),

                    call("file2.wav", prompt=None, analysis_type='describe')

                ])



    def test_save_results_json(self, tmp_path):

        """Test saving results to a JSON file."""

        with patch.dict(os.environ, {"GEMINI_API_KEY": "test_key"}), \

             patch('gemini_audio_analyzer.genai'), \

             patch('gemini_audio_analyzer.AudioExtractor'):

            analyzer = GeminiAudioAnalyzer()

            results = [

                AnalysisResult("file1.mp3", "transcribe", "hello", "model", True, None),

                AnalysisResult("file2.wav", "describe", "world", "model", True, None)

            ]

            output_file = tmp_path / "results.json"



            analyzer.save_results(results, output_file, format='json')



            assert output_file.exists()

            # Further checks could involve reading the file and validating its content

            # For simplicity, we just check for existence here.



