"""
Unit tests for claims_verifier.py

Run with: pytest test_claims_verifier.py -v
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from claims_verifier import (
    ClaimVerifier,
    ClaimVerificationResult,
    VerificationStatus
)


class TestClaimVerifier:
    """Test cases for ClaimVerifier class."""

    def test_initialization_with_api_key(self):
        """Test ClaimVerifier initialization with explicit API key."""
        with patch('claims_verifier.genai.Client') as mock_client:
            verifier = ClaimVerifier(api_key="test_key")
            assert verifier.api_key == "test_key"
            assert verifier.model == "gemini-2.5-flash"
            mock_client.assert_called_once_with(api_key="test_key")

    def test_initialization_with_env_var(self):
        """Test ClaimVerifier initialization with environment variable."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'env_key'}):
            with patch('claims_verifier.genai.Client') as mock_client:
                verifier = ClaimVerifier()
                assert verifier.api_key == "env_key"
                mock_client.assert_called_once_with(api_key="env_key")

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                ClaimVerifier()

    def test_initialization_with_custom_model(self):
        """Test ClaimVerifier with custom model."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            with patch('claims_verifier.genai.Client'):
                verifier = ClaimVerifier(model="gemini-pro")
                assert verifier.model == "gemini-pro"

    def test_validate_claim_empty_raises_error(self):
        """Test that empty claims raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ClaimVerifier.validate_claim("")

    def test_validate_claim_whitespace_only_raises_error(self):
        """Test that whitespace-only claims raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ClaimVerifier.validate_claim("   ")

    def test_validate_claim_too_long_raises_error(self):
        """Test that claims exceeding max length raise ValueError."""
        long_claim = "a" * (ClaimVerifier.MAX_CLAIM_LENGTH + 1)
        with pytest.raises(ValueError, match="exceeds maximum length"):
            ClaimVerifier.validate_claim(long_claim)

    def test_validate_claim_strips_whitespace(self):
        """Test that validation strips whitespace."""
        claim = "  valid claim  "
        result = ClaimVerifier.validate_claim(claim)
        assert result == "valid claim"

    def test_validate_claim_valid(self):
        """Test validation of a valid claim."""
        claim = "The Earth revolves around the Sun"
        result = ClaimVerifier.validate_claim(claim)
        assert result == claim

    @patch('claims_verifier.genai.Client')
    def test_verify_claim_success(self, mock_client_class):
        """Test successful claim verification."""
        # Setup mock
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = '''{
            "summary": "Test summary",
            "verification_status": "True",
            "evidence": "Test evidence",
            "sources": "Test sources",
            "truthfulness_score": 0.95
        }'''
        mock_client.models.generate_content.return_value = mock_response

        # Test
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            verifier = ClaimVerifier()
            result = verifier.verify_claim("The Earth is round")

            assert isinstance(result, ClaimVerificationResult)
            assert result.summary == "Test summary"
            assert result.verification_status == VerificationStatus.TRUE
            assert result.truthfulness_score == 0.95

    @patch('claims_verifier.genai.Client')
    def test_verify_claim_with_grounding(self, mock_client_class):
        """Test claim verification with grounding enabled."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = '''{
            "summary": "Test",
            "verification_status": "True",
            "evidence": "Evidence",
            "sources": "Sources",
            "truthfulness_score": 0.9
        }'''
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            verifier = ClaimVerifier()
            result = verifier.verify_claim("Test claim", enable_grounding=True)

            # Verify grounding was included in the call
            call_args = mock_client.models.generate_content.call_args
            config = call_args.kwargs['config']
            assert 'tools' in config
            assert config['tools'] == [{"google_search": {}}]

    @patch('claims_verifier.genai.Client')
    def test_verify_claim_invalid_claim_raises_error(self, mock_client_class):
        """Test that invalid claims raise ValueError."""
        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            verifier = ClaimVerifier()
            with pytest.raises(ValueError):
                verifier.verify_claim("")

    @patch('claims_verifier.genai.Client')
    def test_verify_claims_batch(self, mock_client_class):
        """Test batch claim verification."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = Mock()
        mock_response.text = '''{
            "summary": "Test",
            "verification_status": "True",
            "evidence": "Evidence",
            "sources": "Sources",
            "truthfulness_score": 0.9
        }'''
        mock_client.models.generate_content.return_value = mock_response

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            verifier = ClaimVerifier()
            claims = ["Claim 1", "Claim 2"]

            # Mock sleep to speed up tests
            with patch('time.sleep'):
                results = verifier.verify_claims_batch(claims, rate_limit_delay=0.1)

            assert len(results) == 2
            assert "Claim 1" in results
            assert "Claim 2" in results
            assert all(isinstance(r, ClaimVerificationResult) for r in results.values())

    @patch('claims_verifier.genai.Client')
    def test_verify_claims_batch_with_failures(self, mock_client_class):
        """Test batch verification continues on individual failures."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # First call succeeds, second fails, third succeeds
        mock_response_success = Mock()
        mock_response_success.text = '''{
            "summary": "Test",
            "verification_status": "True",
            "evidence": "Evidence",
            "sources": "Sources",
            "truthfulness_score": 0.9
        }'''

        mock_client.models.generate_content.side_effect = [
            mock_response_success,
            Exception("API Error"),
            mock_response_success
        ]

        with patch.dict(os.environ, {'GOOGLE_API_KEY': 'test_key'}):
            verifier = ClaimVerifier()
            claims = ["Claim 1", "Claim 2", "Claim 3"]

            with patch('time.sleep'):
                results = verifier.verify_claims_batch(claims, rate_limit_delay=0)

            # Should have 2 successful results despite one failure
            assert len(results) == 2


class TestClaimVerificationResult:
    """Test cases for ClaimVerificationResult model."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ClaimVerificationResult(
            summary="Test summary",
            verification_status=VerificationStatus.TRUE,
            evidence="Test evidence",
            sources="Test sources",
            truthfulness_score=0.85
        )
        assert result.summary == "Test summary"
        assert result.truthfulness_score == 0.85

    def test_truthfulness_score_validation(self):
        """Test truthfulness score is validated between 0 and 1."""
        with pytest.raises(Exception):  # Pydantic validation error
            ClaimVerificationResult(
                summary="Test",
                verification_status=VerificationStatus.TRUE,
                evidence="Evidence",
                sources="Sources",
                truthfulness_score=1.5  # Invalid
            )

    def test_empty_fields_validation(self):
        """Test that empty required fields are caught."""
        with pytest.raises(Exception):  # Pydantic validation error
            ClaimVerificationResult(
                summary="",  # Invalid
                verification_status=VerificationStatus.TRUE,
                evidence="Evidence",
                sources="Sources",
                truthfulness_score=0.5
            )


class TestVerificationStatus:
    """Test cases for VerificationStatus enum."""

    def test_enum_values(self):
        """Test that enum has correct values."""
        assert VerificationStatus.TRUE.value == "True"
        assert VerificationStatus.FALSE.value == "False"
        assert VerificationStatus.PARTIALLY_TRUE.value == "Partially True"
        assert VerificationStatus.UNVERIFIABLE.value == "Unverifiable"

    def test_enum_count(self):
        """Test that enum has expected number of values."""
        assert len(VerificationStatus) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
