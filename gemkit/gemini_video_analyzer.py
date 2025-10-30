import logging
import os
import time
from typing import Optional

from google import genai

class GeminiVideoAnalyzer:
    """
    Analyzes video files using the Google Gemini API.

    This class provides a high-level interface for uploading a video file,
    sending it to the Gemini API with a text prompt, and retrieving the
    analysis. It also handles the cleanup of uploaded files.
    """
    DEFAULT_MODEL = "gemini-2.5-flash"
    
    # File state constants
    STATE_ACTIVE = "ACTIVE"
    STATE_FAILED = "FAILED"
    
    # Supported video file extensions
    SUPPORTED_VIDEO_FORMATS = {
        '.mp4', '.mpeg', '.mpg', '.mov', '.avi', '.wmv', '.flv', 
        '.webm', '.mkv', '.3gp', '.m4v', '.ogv'
    }
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initializes the GeminiVideoAnalyzer.

        Args:
            model_name (str, optional): The name of the Gemini model to use.
                                        Defaults to "gemini-2.5-flash".
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        
        self.model_name = model_name
        self.client = genai.Client(api_key=api_key)
        self.uploaded_file: Optional[genai.File] = None
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger for this instance."""
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _remove_existing_file(self) -> None:
        """
        Removes any previously uploaded file before uploading a new one.
        """
        if self.uploaded_file:
            file_name = self.uploaded_file.name
            self.logger.info("Existing file detected. Removing it before uploading new file...")
            try:
                self.client.files.delete(name=file_name)
                self.logger.info(f"Removed existing file: {file_name}")
            except Exception as e:
                self.logger.warning(f"Failed to remove existing file: {e}")
            finally:
                self.uploaded_file = None
    
    def _wait_for_file_processing(self, timeout: int = 300, check_interval: int = 5) -> bool:
        """
        Polls the file status until it becomes ACTIVE or FAILED.
        :param timeout: Maximum time to wait in seconds
        :param check_interval: Time between status checks in seconds
        :return: True if file is ACTIVE, False otherwise
        """
        if not self.uploaded_file:
            self.logger.error("No uploaded file to wait for")
            return False
        
        self.logger.info("Waiting for file to be processed...")
        file_name = self.uploaded_file.name
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                file_info = self.client.files.get(name=file_name)
                
                self.logger.debug(f"File state: {file_info.state}")
                
                if file_info.state == self.STATE_ACTIVE:
                    self.logger.info("File is now ACTIVE and ready for use.")
                    self.uploaded_file = file_info
                    return True
                elif file_info.state == self.STATE_FAILED:
                    self.logger.error("File processing FAILED.")
                    self.uploaded_file = None
                    return False
                
                time.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error checking file status: {e}")
                return False
        
        self.logger.error(f"Timeout reached ({timeout}s). File did not become ACTIVE.")
        self.uploaded_file = None
        return False
    
    def upload_file(self, file_path: str, timeout: int = 300, check_interval: int = 5) -> Optional[genai.File]:
        """
        Uploads a video file to the Gemini service and waits for it to be processed.

        Args:
            file_path (str): The local path to the video file.
            timeout (int, optional): The maximum time to wait for processing in seconds.
            check_interval (int, optional): The interval between status checks in seconds.

        Returns:
            Optional[genai.File]: The uploaded file object, or None if an error occurs.
        """
        if not self.client:
            self.logger.error("Client is not initialized. Aborting upload.")
            return None
        
        if not os.path.exists(file_path):
            self.logger.error(f"Local file not found at path: {file_path}")
            return None
        
        # Validate file format
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension not in self.SUPPORTED_VIDEO_FORMATS:
            self.logger.error(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(sorted(self.SUPPORTED_VIDEO_FORMATS))}"
            )
            return None
        
        # Remove any existing file first
        self._remove_existing_file()
        
        # Upload the new file
        self.logger.info(f"[STEP 1/2] Attempting to upload file: {file_path}")
        try:
            self.uploaded_file = self.client.files.upload(file=file_path)
            self.logger.info(f"File uploaded successfully. Name: {self.uploaded_file.name}")
            
            # Wait for processing to complete
            if self._wait_for_file_processing(timeout, check_interval):
                return self.uploaded_file
            return None
            
        except Exception as e:
            self.logger.exception(f"Error during file upload: {e}")
            self.uploaded_file = None
            return None
    
    def generate_text(self, prompt: str = "Describe the video") -> Optional[str]:
        """
        Generates a textual analysis of the uploaded video.

        Args:
            prompt (str, optional): The prompt to send to the Gemini model.

        Returns:
            Optional[str]: The text generated by the model, or None if an error occurs.
        """
        if not self.client:
            self.logger.error("Client is not initialized. Aborting analysis.")
            return None
        
        if not self.uploaded_file:
            self.logger.error("No file uploaded. Please upload a file first.")
            return None
        
        try:
            self.logger.info(f"Generating content with prompt: '{prompt}'")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[self.uploaded_file, prompt]
            )
            self.logger.info("Content generation completed successfully.")
            return response.text
        except Exception as e:
            self.logger.exception(f"Error during content generation: {e}")
            return None
    
    def delete_file(self) -> None:
        """
        Deletes the uploaded file from the Gemini service.

        It is important to call this method when you are finished with a video
        file to avoid unnecessary storage costs.
        """
        self.logger.info("[STEP 2/2] Starting cleanup process.")
        if self.uploaded_file and self.client:
            self._remove_existing_file()
            self.logger.info("Cleanup complete.")
        else:
            self.logger.warning("Cleanup skipped: No client or uploaded file object found.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze videos using Gemini AI. Requires GEMINI_API_KEY environment variable.'
    )
    
    parser.add_argument(
        '-i', '--video_path',
        type=str,
        help='Path to the video file to analyze'
    )
    
    parser.add_argument(
        '-p', '--prompt',
        type=str,
        default='Write important class notes from this video. In the end, explain weakness and strength of the arguments',
        help='Prompt for video analysis (default: class notes with analysis)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default=GeminiVideoAnalyzer.DEFAULT_MODEL,
        help=f'Model to use (default: {GeminiVideoAnalyzer.DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '-t', '--timeout',
        type=int,
        default=300,
        help='Maximum time to wait for video processing in seconds (default: 300)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main function to handle command line execution."""
    args = parse_arguments()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    analyzer = None
    try:
        analyzer = GeminiVideoAnalyzer(model_name=args.model)
        
        if analyzer.upload_file(args.video_path, timeout=args.timeout):
            print("\n" + "=" * 60)
            print("VIDEO ANALYSIS RESULTS")
            print("=" * 60)
            
            response = analyzer.generate_text(args.prompt)
            if response:
                print(f"\n{response}")
            else:
                print("\nFailed to generate analysis. Please check the logs for details.")
            
            print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
    finally:
        if analyzer:
            analyzer.delete_file()


if __name__ == '__main__':
    import argparse
    main()
