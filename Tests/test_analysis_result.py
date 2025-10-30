import unittest
from gemkit.analysis_result import AnalysisResult

class TestAnalysisResult(unittest.TestCase):
    """
    Unit tests for the AnalysisResult dataclass.
    """

    def test_creation_and_attributes(self):
        """
        Test that an AnalysisResult object can be created and its attributes are correctly set.
        """
        result = AnalysisResult(
            file_path="/path/to/file",
            analysis_type="transcription",
            result="This is a test.",
            model="gemini-2.5-flash",
            success=True,
            error=None
        )

        self.assertEqual(result.file_path, "/path/to/file")
        self.assertEqual(result.analysis_type, "transcription")
        self.assertEqual(result.result, "This is a test.")
        self.assertEqual(result.model, "gemini-2.5-flash")
        self.assertTrue(result.success)
        self.assertIsNone(result.error)

    def test_error_case(self):
        """
        Test that an AnalysisResult object can represent a failed analysis.
        """
        result = AnalysisResult(
            file_path="/path/to/file",
            analysis_type="transcription",
            result="",
            model="gemini-2.5-flash",
            success=False,
            error="File not found"
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "File not found")

if __name__ == '__main__':
    unittest.main()
