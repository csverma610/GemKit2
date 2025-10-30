import unittest
from unittest.mock import patch, MagicMock
import os
import json

from gemkit.claims_verifier import ClaimVerifier, ClaimVerificationResult, VerificationStatus

class TestClaimVerifier(unittest.TestCase):
    """
    Unit tests for the ClaimVerifier class.
    """

    def setUp(self):
        """
        Set up the test environment before each test.
        This involves setting the API key and patching the `genai.Client`.
        """
        os.environ['GOOGLE_API_KEY'] = 'test_api_key'
        self.patcher = patch('gemkit.claims_verifier.genai.Client')
        self.mock_genai_client = self.patcher.start()

    def tearDown(self):
        """
        Clean up the test environment after each test.
        """
        self.patcher.stop()
        del os.environ['GOOGLE_API_KEY']

    def test_verify_claim_success(self):
        """
        Test successful verification of a single claim.
        """
        # Mock the Gemini client's response
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "summary": "Test summary",
            "verification_status": "True",
            "evidence": "Test evidence",
            "sources": "Test sources",
            "truthfulness_score": 1.0
        })
        self.mock_genai_client.return_value.models.generate_content.return_value = mock_response

        verifier = ClaimVerifier()
        result = verifier.verify_claim("The sky is blue.")

        self.assertIsInstance(result, ClaimVerificationResult)
        self.assertEqual(result.verification_status, VerificationStatus.TRUE)
        self.assertEqual(result.truthfulness_score, 1.0)

    def test_validate_claim(self):
        """
        Test the claim validation logic.
        """
        with self.assertRaises(ValueError):
            ClaimVerifier.validate_claim("")  # Empty claim

        with self.assertRaises(ValueError):
            ClaimVerifier.validate_claim("   ")  # Whitespace only

        with self.assertRaises(ValueError):
            ClaimVerifier.validate_claim("a" * (ClaimVerifier.MAX_CLAIM_LENGTH + 1))  # Too long

        sanitized_claim = ClaimVerifier.validate_claim("  This is a valid claim.  ")
        self.assertEqual(sanitized_claim, "This is a valid claim.")

    def test_verify_claims_batch(self):
        """
        Test batch verification of claims.
        """
        # Mock the Gemini client's response to return different results for each call
        mock_response_1 = MagicMock()
        mock_response_1.text = json.dumps({
            "summary": "Summary 1", "verification_status": "True", "evidence": "Evidence 1",
            "sources": "Sources 1", "truthfulness_score": 1.0
        })
        mock_response_2 = MagicMock()
        mock_response_2.text = json.dumps({
            "summary": "Summary 2", "verification_status": "False", "evidence": "Evidence 2",
            "sources": "Sources 2", "truthfulness_score": 0.0
        })
        self.mock_genai_client.return_value.models.generate_content.side_effect = [
            mock_response_1, mock_response_2
        ]

        verifier = ClaimVerifier()
        claims = ["Claim 1", "Claim 2"]
        results = verifier.verify_claims_batch(claims, rate_limit_delay=0)

        self.assertEqual(len(results), 2)
        self.assertEqual(results["Claim 1"].verification_status, VerificationStatus.TRUE)
        self.assertEqual(results["Claim 2"].verification_status, VerificationStatus.FALSE)

if __name__ == '__main__':
    unittest.main()