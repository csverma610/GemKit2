"""
This module defines the AnalysisResult dataclass for storing analysis results.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class AnalysisResult:
    """
    A dataclass to encapsulate the results of an analysis operation.

    This class provides a standardized structure for holding information about
    an analysis performed on a file, including the outcome, any generated data,
    and error details if the operation was unsuccessful.

    Attributes:
        file_path (str): The absolute or relative path to the file that was analyzed.
        analysis_type (str): A string indicating the type of analysis performed 
                             (e.g., 'transcription', 'sentiment_analysis').
        result (str): The output of the analysis. For text-based results, this will
                      be a string. For structured data, it may be a JSON string.
        model (str): The name of the model used for the analysis.
        success (bool): A boolean flag indicating whether the analysis was successful.
                        `True` if successful, `False` otherwise.
        error (Optional[str]): A string containing an error message if the analysis
                               failed. `None` if the analysis was successful.
    """
    file_path: str
    analysis_type: str
    result: str
    model: str
    success: bool
    error: Optional[str] = None
