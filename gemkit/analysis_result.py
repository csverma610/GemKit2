"""
This module defines the AnalysisResult dataclass for storing analysis results.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisResult:
    """Data class to store analysis results"""
    file_path: str
    analysis_type: str
    result: str
    model: str
    success: bool
    error: Optional[str] = None
