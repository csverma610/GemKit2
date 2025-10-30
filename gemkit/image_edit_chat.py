import os
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Union

from google import genai
from PIL import Image

# Set up logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GeminiImageEditor:
    """
    An interactive image editor that uses the Google Gemini API to modify
    images based on natural language instructions.

    This class maintains a chat-like session with the Gemini model, allowing
    for a series of edits to be applied to an image.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash-image-preview"):
        """
        Initializes the GeminiImageEditor.

        Args:
            api_key (Optional[str], optional): The Google AI API key. If not provided, it will be
                                               read from the GOOGLE_AI_API_KEY environment variable.
            model (str, optional): The name of the Gemini model to use.
        """
        # Try to get API key
        self.api_key = api_key or os.getenv('GOOGLE_AI_API_KEY')
        if not self.api_key:
            print("❌ Error: No API key found!")
            print("   Set it as: export GOOGLE_AI_API_KEY='your_key_here'")
            print("   Or pass it directly: GeminiImageEditor(api_key='your_key')")
            raise ValueError("API key required")
        
        self.model = model
        self.client = genai.Client(api_key=self.api_key)
        self.chat = None
        
        # State tracking
        self.edit_counter = 0
        self.saved_file = []
        self.base_image_name = None
        
        print(f"✅ Gemini Image Editor initialized with model: {model}")
    
    def set_image(self, image_path: Union[str, Path]) -> bool:
        """
        Sets a new base image for the editing session.

        This method starts a new chat session with the Gemini model and sends
        the specified image to establish the context for subsequent edits.

        Args:
            image_path (Union[str, Path]): The path to the image file.

        Returns:
            bool: `True` if the image was set successfully, `False` otherwise.
        """
        try:
            image_path = Path(image_path)
            
            # Validate file exists
            if not image_path.exists():
                print(f"❌ Error: Image file not found: {image_path}")
                print(f"   Current directory: {Path.cwd()}")
                return False
            
            # Validate file is an image
            try:
                image = Image.open(image_path)
                print(f"📷 Loading image: {image_path.name} ({image.size[0]}x{image.size[1]})")
            except Exception as e:
                print(f"❌ Error: Cannot open as image: {image_path}")
                print(f"   {str(e)}")
                return False
            
            # Store image info
            self.base_image_name = image_path.stem
            
            # Reset state and start fresh chat
            self._reset_state()
            
            # Start new chat session
            print("🔄 Starting new chat session...")
            
            # Send image to establish context
            print("📤 Sending image to Gemini...")
            message_content = ["I want to edit this image. Please confirm you can see it and describe it briefly."]
            message_content.append(image)
            
            response = self.chat.send_message(message_content)
            self.initial_image_sent = True
            
            # Show Gemini's response about the image
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.text:
                        print(f"🤖 Gemini sees: {part.text}")
                        break
            
            print(f"✅ Ready to edit! Output files will be named: {self.base_image_name}_edit_*.png")
            return True
            
        except Exception as e:
            print(f"❌ Failed to set image: {str(e)}")
            return False
    
    def edit(self, instruction: str, save_to: str = "output") -> List[str]:
        """
        Applies an edit to the current image based on a natural language instruction.

        Args:
            instruction (str): A description of the desired edit (e.g., "make the sky blue").
            save_to (str, optional): The directory to save the resulting image(s).
                                     Defaults to "output".

        Returns:
            List[str]: A list of file paths to the saved images.
        """
        if not self._check_ready():
            return []
        
        try:
            print(f"\n🎨 Edit #{self.edit_counter + 1}: {instruction}")
            print("⏳ Processing...")
            message = [instruction]
            if edit_counter == 0: 
               message_content.append(image)
    
            # Send instruction to Gemini
            response = self.chat.send_message(message)

            self.save_response(instuctio, response)

        except Exception as e:
            print(f"❌ Edit failed: {str(e)}")
            return []
    
    def _restart(self) -> None:
        """Reset editor state."""
        self.edit_counter = 0
        self.chat = self.client.chats.create(model=self.model)
    
    def _save_images(self, instruction, response, output_dir: str) -> List[str]:
        """Save response images with user-friendly naming."""
        saved_files = []
        self.edit_counter += 1
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Update edit log
        self._update_edit_log(output_path, instruction)
        
        try:
            if not response.candidates:
                print("⚠️  No images in response")
                return saved_files
            
            image_count = 0
            for part in response.candidates[0].content.parts:
                if part.text is not None:
                    print(f"💬 Gemini says: {part.text}")
                    
                elif part.inline_data is not None:
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        image_count += 1
                        
                        # Generate filename
                        if image_count == 1:
                            filename = f"{self.base_image_name}_edit_{self.edit_counter}.png"
                        else:
                            filename = f"{self.base_image_name}_edit_{self.edit_counter}_{image_count:02d}.png"
                        
                        filepath = self._get_unique_filepath(output_path / filename)
                        
                        image.save(filepath)
                        saved_files.append(str(filepath))
                        self.all_saved_files.append(str(filepath))
                        
                    except Exception as e:
                        print(f"⚠️  Failed to save image: {str(e)}")
                        
        except Exception as e:
            print(f"❌ Error processing response: {str(e)}")
        
        return saved_files
    
if __name__ == "__main__":
    demo()
