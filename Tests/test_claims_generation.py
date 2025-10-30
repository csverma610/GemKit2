import unittest
from unittest.mock import patch, MagicMock
import os

from gemkit.claims_generation import ClaimsGenerator, ClaimsConfig, FileSizeLimitError

class TestClaimsGenerator(unittest.TestCase):
    """
    Unit tests for the ClaimsGenerator class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        This involves creating a dummy file.
        """
        self.test_file = "test_document.txt"
        with open(self.test_file, "w") as f:
            f.write("This is a test document. It contains several sentences.")

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        os.remove(self.test_file)

    @patch('gemkit.claims_generation.genai.Client')
    def test_generate_claims_from_text(self, MockGeminiClient):
        """
        Test generating claims from a string of text.
        """
        # Mock the Gemini client
        mock_client_instance = MockGeminiClient.return_value
        mock_response = MagicMock()
        mock_response.text = "1. This is a claim."
        mock_client_instance.models.generate_content.return_value = mock_response

        config = ClaimsConfig()
        generator = ClaimsGenerator(config)
        claims = generator.generate_text("This is a test.")

        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0], "This is a claim.")

    def test_read_file_size_limit(self):
        """
        Test that reading a file larger than the size limit raises an error.
        """
        config = ClaimsConfig(max_file_size_mb=0.00001) # 10 bytes
        generator = ClaimsGenerator(config)

        with self.assertRaises(FileSizeLimitError):
            generator.read_file(self.test_file)

    def test_chunking(self):
        """
        Test that the text chunking mechanism works as expected.
        """
        config = ClaimsConfig(chunk_size_chars=20, chunk_overlap_chars=5)
        generator = ClaimsGenerator(config)
        text = "This is a long sentence for testing the chunking mechanism."
        chunks = generator._split_text_into_chunks(text)

        self.assertEqual(len(chunks), 4)
        self.assertTrue(chunks[0].startswith("This is a long"))
        self.assertTrue(chunks[1].startswith("sentence for"))

    def test_deduplication(self):
        """
        Test that duplicate and highly similar claims are removed.
        """
        generator = ClaimsGenerator()
        claims = [
            "This is a claim.",
            "This is a claim.",
            "This is a very similar claim.",
            "This is a different claim."
        ]
        unique_claims = generator._deduplicate_claims(claims)

        self.assertEqual(len(unique_claims), 3)
        self.assertIn("This is a claim.", unique_claims)
        self.assertIn("This is a very similar claim.", unique_claims)
        self.assertIn("This is a different claim.", unique_claims)


if __name__ == '__main__':
    unittest.main()
