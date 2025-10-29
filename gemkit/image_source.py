"""
ImageSource - Unified Image Source Handler for Multimodal LLMs

A comprehensive class for handling multiple image sources and converting them 
to base64 or PIL format for multimodal LLM APIs.

Supported Sources:
    - Image files (.jpg, .png, .bmp, .gif, .webp, .tiff)
    - Directories (with batch processing)
    - Video files (.mp4, .avi, .mov, .mkv, and 8 more formats)
    - Camera/Webcam (persistent connection)
    - Screenshots (full screen or region)
    - URLs (automatic download)
    - PIL Images
    - Numpy arrays
    - Base64 strings
    - Raw bytes

Usage:
    from image_source import ImageSource, SourceConfig, OutputType, ImageFormat
    
    # Simple usage
    source = ImageSource('photo.jpg')
    result = source.get_image()
    
    # With configuration
    config = SourceConfig(output_type=OutputType.BASE64, batch_size=10)
    source = ImageSource('./images', config)
    
    while source.has_more_images():
        images = source.get_images()
        for img in images:
            llm.process(img.data)

Requirements:
    pip install pillow numpy
    pip install opencv-python  # For camera and video
    pip install pyautogui      # For screenshots
    pip install requests        # For URLs

Author: AI Assistant
License: MIT
Version: 1.0.0
"""

import base64
import io
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union, List, Optional, Generator

import numpy as np
from PIL import Image


# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class OutputType(Enum):
    """Output format for images."""
    BASE64 = "base64"
    PIL = "pil"
    NUMPY = "numpy"


class ColorSpace(Enum):
    """Numpy array color space."""
    RGB = "rgb"
    RGBA = "rgba"
    BGR = "bgr"


class ImageFormat(Enum):
    """Supported image formats."""
    PNG = "PNG"
    JPEG = "JPEG"
    JPG = "JPG"
    BMP = "BMP"
    GIF = "GIF"
    WEBP = "WEBP"
    TIFF = "TIFF"


class VideoFormat(Enum):
    """Supported video formats."""
    MP4 = ".mp4"
    AVI = ".avi"
    MOV = ".mov"
    MKV = ".mkv"
    FLV = ".flv"
    WMV = ".wmv"
    WEBM = ".webm"
    M4V = ".m4v"
    MPEG = ".mpeg"
    MPG = ".mpg"
    THREE_GP = ".3gp"
    OGV = ".ogv"
    
    @classmethod
    def is_valid(cls, extension: str) -> bool:
        """Check if extension is a valid video format."""
        ext = extension.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'
        return ext in [fmt.value for fmt in cls]
    
    @classmethod
    def get_all_extensions(cls) -> List[str]:
        """Get list of all supported video extensions."""
        return [fmt.value for fmt in cls]


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ImageSourceError(Exception):
    """Base exception for ImageSource errors."""
    pass


class InvalidConfigError(ImageSourceError):
    """Raised when configuration is invalid."""
    pass


class SourceNotSetError(ImageSourceError):
    """Raised when no source is set."""
    pass


class UnsupportedOperationError(ImageSourceError):
    """Raised when operation is not supported for current source type."""
    pass


class UnsupportedFormatError(ImageSourceError):
    """Raised when file format is not supported."""
    pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Resolution:
    """Video/Image resolution."""
    width: int
    height: int
    
    def __str__(self) -> str:
        return f"{self.width}x{self.height}"
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def total_pixels(self) -> int:
        """Calculate total number of pixels."""
        return self.width * self.height
    
    @property
    def is_hd(self) -> bool:
        """Check if resolution is HD (720p or higher)."""
        return self.height >= 720
    
    @property
    def is_full_hd(self) -> bool:
        """Check if resolution is Full HD (1080p)."""
        return self.height >= 1080
    
    @property
    def is_4k(self) -> bool:
        """Check if resolution is 4K (2160p)."""
        return self.height >= 2160
    
    @classmethod
    def from_string(cls, resolution_str: str) -> 'Resolution':
        """
        Create Resolution from string like '1920x1080'.
        
        Args:
            resolution_str: String in format 'WIDTHxHEIGHT'
        
        Returns:
            Resolution object
        """
        parts = resolution_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError(f"Invalid resolution string: {resolution_str}")
        return cls(width=int(parts[0]), height=int(parts[1]))
    
    # Common resolutions as class attributes
    HD = None  # 1280x720
    FULL_HD = None  # 1920x1080
    QHD = None  # 2560x1440
    UHD_4K = None  # 3840x2160


# Initialize common resolutions
Resolution.HD = Resolution(1280, 720)
Resolution.FULL_HD = Resolution(1920, 1080)
Resolution.QHD = Resolution(2560, 1440)
Resolution.UHD_4K = Resolution(3840, 2160)

@dataclass
class VideoInfo:
    """Video metadata information."""
    fps: float
    frame_count: int
    duration: float
    resolution: Resolution
    
    @property
    def duration_str(self) -> str:
        """Get duration as formatted string (HH:MM:SS)."""
        hours = int(self.duration // 3600)
        minutes = int((self.duration % 3600) // 60)
        seconds = int(self.duration % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"
    
    @property
    def avg_frame_time(self) -> float:
        """Average time per frame in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0
    
    def __str__(self) -> str:
        return (f"VideoInfo(resolution={self.resolution}, "
                f"fps={self.fps:.2f}, duration={self.duration_str}, "
                f"frames={self.frame_count})")


@dataclass
class ImageResult:
    """Result container for image operations."""
    data: Union[str, Image.Image, np.ndarray]
    filename: Optional[str] = None
    path: Optional[str] = None
    frame_index: Optional[int] = None
    timestamp: Optional[float] = None
    resolution: Optional[Resolution] = None
    
    @property
    def is_base64(self) -> bool:
        """Check if data is base64 string."""
        return isinstance(self.data, str)
    
    @property
    def is_pil(self) -> bool:
        """Check if data is PIL Image."""
        return isinstance(self.data, Image.Image)
    
    @property
    def is_numpy(self) -> bool:
        """Check if data is a numpy array."""
        return isinstance(self.data, np.ndarray)
    
    def get_resolution(self) -> Optional[Resolution]:
        """
        Get resolution of the image.
        
        Returns:
            Resolution object if image is PIL, else None
        """
        if self.resolution:
            return self.resolution
        
        if self.is_pil:
            width, height = self.data.size
            self.resolution = Resolution(width, height)
            return self.resolution
        
        return None


@dataclass
class SourceConfig:
    """Configuration for image sources."""
    
    # Output settings
    output_type: OutputType = OutputType.BASE64
    colorspace: ColorSpace = ColorSpace.RGB
    format: ImageFormat = ImageFormat.PNG
    
    # Directory settings
    recursive: bool = False
    extensions: Optional[List[str]] = None
    max_images: Optional[int] = None
    batch_size: int = 10
    
    # Video settings
    time_or_frame: Union[float, int] = 0
    use_frame_index: bool = False
    video_interval: Optional[float] = None
    
    # Screenshot settings
    region: Optional[tuple] = None  # (x, y, width, height)
    
    # Camera settings
    camera_index: int = 0
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert string to enum if needed
        if isinstance(self.output_type, str):
            self.output_type = OutputType(self.output_type.lower())

        if isinstance(self.colorspace, str):
            try:
                self.colorspace = ColorSpace(self.colorspace.lower())
            except ValueError:
                raise InvalidConfigError(f"Unsupported colorspace: {self.colorspace}")
        
        if isinstance(self.format, str):
            try:
                self.format = ImageFormat[self.format.upper()]
            except KeyError:
                raise InvalidConfigError(f"Unsupported format: {self.format}")
        
        # Set default extensions
        if self.extensions is None:
            self.extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff']
        
        # Validate batch_size
        if self.batch_size <= 0:
            raise InvalidConfigError(f"batch_size must be positive, got {self.batch_size}")
        
        # Validate max_images
        if self.max_images is not None and self.max_images <= 0:
            raise InvalidConfigError(f"max_images must be positive, got {self.max_images}")
        
        # Validate camera_index
        if self.camera_index < 0:
            raise InvalidConfigError(f"camera_index must be non-negative, got {self.camera_index}")
        
        # Validate region
        if self.region is not None:
            if not isinstance(self.region, tuple) or len(self.region) != 4:
                raise InvalidConfigError("region must be a tuple of (x, y, width, height)")


# ============================================================================
# MAIN IMAGE SOURCE CLASS
# ============================================================================

class ImageSource:
    """
    Unified class to handle multiple image sources for multimodal LLM APIs.
    
    Examples:
        # Image file
        source = ImageSource('photo.jpg')
        result = source.get_image()
        
        # Directory batch processing
        config = SourceConfig(recursive=True, batch_size=10)
        source = ImageSource('./images', config)
        while source.has_more_images():
            images = source.get_images()
        
        # Video frames
        config = SourceConfig(video_num_frames=10)
        source = ImageSource('video.mp4', config)
        frames = source.get_images()
        
        # Camera
        source = ImageSource(0)  # Camera 0
        result = source.get_image()
        
        # URL
        source = ImageSource('https://example.com/image.jpg')
        result = source.get_image()
    """
    
    def __init__(self, source: Union[str, Path, Image.Image, np.ndarray, bytes, int] = None, 
                 config: Optional[SourceConfig] = None):
        """
        Initialize ImageSource.
        
        Args:
            source: Image source (file, directory, URL, camera index, PIL Image, etc.)
            config: SourceConfig object with settings
        """
        self.source = source
        self.source_type = None
        self.config = config if config is not None else SourceConfig()
        self._camera = None
        self._video = None
        self._directory_files = None
        self._directory_index = 0
        self._video_generator = None
        
        if source is not None:
            self.source_type = self._identify_source_type(source)
            if self.source_type == "camera":
                self._initialize_camera(source)
            elif self.source_type == "unknown":
                raise UnsupportedFormatError(f"Unsupported source: {source}")
    
    def __repr__(self) -> str:
        return (f"ImageSource(source_type={self.source_type}, "
                f"output_type={self.config.output_type.value}, "
                f"format={self.config.format.value})")
    
    def __str__(self) -> str:
        if self.source_type == "directory":
            return f"ImageSource: Directory with {self._get_total_files()} files"
        return f"ImageSource: {self.source_type or 'Not set'}"
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        try:
            self._cleanup()
        except:
            pass
    
    # ========================================================================
    # PUBLIC API
    # ========================================================================
    
    def get_image(self) -> ImageResult:
        """
        Get a single image from the current source.
        
        Returns:
            ImageResult object
        
        Raises:
            SourceNotSetError: If no source is set
            UnsupportedOperationError: If used with directory source
        """
        if self.source is None:
            raise SourceNotSetError("No source set. Call set_source() first.")
        
        if self.source_type == "directory":
            raise UnsupportedOperationError(
                "get_image() is not supported for directories. Use get_images() instead."
            )
        
        handler_name = f"_handle_{self.source_type}_source"
        handler = getattr(self, handler_name, None)
        
        if handler and callable(handler):
            return handler()
        
        raise ImageSourceError(f"Unknown or unsupported source type: {self.source_type}")
    
    def get_images(self) -> List[ImageResult]:
        """
        Get multiple images from the current source (batch operation).
        
        Supported for: directory, video, camera
        
        Returns:
            List of ImageResult objects
        
        Raises:
            UnsupportedOperationError: If source doesn't support batch operations
        """
        if self.source is None:
            raise SourceNotSetError("No source set. Call set_source() first.")
        
        if self.source_type == "directory":
            return self._get_directory_batch()
        elif self.source_type == "video_file":
            return self._get_video_frames()
        elif self.source_type == "camera":
            return self._get_camera_batch()
        else:
            raise UnsupportedOperationError(
                f"get_images() is not supported for {self.source_type}. Use get_image() instead."
            )
    
    def has_more_images(self) -> bool:
        """
        Check if there are more images available.
        
        Supported for: directory, video
        """
        if self.source_type == "directory":
            if self._directory_files is None:
                self._initialize_directory_files()
            return self._directory_index < len(self._directory_files)
        elif self.source_type == "video_file":
            if self._video is None:
                self._video = _VideoHandler(self.source)
            return self._video.has_more_frames()
        return False
    
    def reset(self):
        """Reset to start from the beginning (directory, video)."""
        if self.source_type == "directory":
            self._directory_index = 0
        elif self.source_type == "video_file":
            if self._video is None:
                self._video = _VideoHandler(self.source)
            self._video.reset()
            self._video_generator = None
        else:
            raise UnsupportedOperationError(f"reset() is not supported for {self.source_type}")
        return self
    
    def get_info(self) -> dict:
        """Get information about the current source."""
        info = {
            'source_type': self.source_type,
            'output_type': self.config.output_type.value,
            'format': self.config.format.value
        }
        
        if self.source_type == "directory":
            if self._directory_files is None:
                self._initialize_directory_files()
            total = len(self._directory_files)
            processed = self._directory_index
            info.update({
                'total_files': total,
                'processed_files': processed,
                'remaining_files': total - processed,
                'recursive': self.config.recursive,
                'batch_size': self.config.batch_size
            })
        elif self.source_type == "video_file":
            if self._video is None:
                self._video = _VideoHandler(self.source)
            info.update(self._video.get_info())
        elif self.source_type == "camera":
            info['camera_index'] = self.config.camera_index
        
        return info
    
    def set_source(self, source: Union[str, Path, Image.Image, np.ndarray, bytes, int], 
                   config: Optional[SourceConfig] = None):
        """Set or change the source."""
        self._cleanup()
        self.source = source
        self.source_type = self._identify_source_type(source)
        if config is not None:
            self.config = config
        self._directory_files = None
        self._directory_index = 0
        if self.source_type == "camera":
            self._initialize_camera(source)
        return self
    
    def set_config(self, config: SourceConfig):
        """Update configuration."""
        self.config = config
        if self.source_type == "directory":
            self._directory_files = None
            self._directory_index = 0
        return self
    
    def set_camera(self, camera_index: int = 0):
        """Set source to camera."""
        self._cleanup()
        self.source = camera_index
        self.source_type = "camera"
        self._initialize_camera(camera_index)
        return self
    
    def set_video(self, video_path: Union[str, Path]):
        """Set source to video file."""
        self._cleanup()
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not VideoFormat.is_valid(video_path.suffix):
            raise UnsupportedFormatError(
                f"Unsupported video format: {video_path.suffix}. "
                f"Supported: {', '.join(VideoFormat.get_all_extensions())}"
            )
        self.source = video_path
        self.source_type = "video_file"
        self._video = _VideoHandler(video_path)
        return self
    
    def set_directory(self, directory_path: Union[str, Path], recursive: bool = False,
                     extensions: Optional[List[str]] = None, max_images: Optional[int] = None,
                     batch_size: int = 10):
        """Set source to directory."""
        self._cleanup()
        self.source = directory_path
        self.source_type = "directory"
        self.config.recursive = recursive
        if extensions is not None:
            self.config.extensions = extensions
        if max_images is not None:
            self.config.max_images = max_images
        self.config.batch_size = batch_size
        self._directory_files = None
        self._directory_index = 0
        return self
    
    def set_screenshot(self):
        """Set source to screenshot capture."""
        self._cleanup()
        self.source = "screenshot"
        self.source_type = "screenshot"
        return self
    
    def get_video_info(self) -> VideoInfo:
        """
        Get video information (only for video sources).
        
        Returns:
            VideoInfo object with metadata
        """
        if self.source_type != "video_file":
            raise UnsupportedOperationError("get_video_info() only works with video sources")
        if self._video is None:
            self._video = _VideoHandler(self.source)
        return self._video.get_video_info()
    
    def get_source_type(self) -> str:
        """Get the type of the current source."""
        return self.source_type
    
    def close(self):
        """Close and release all resources."""
        self._cleanup()
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _identify_source_type(self, source) -> str:
        """Identify the type of the source."""
        if isinstance(source, Image.Image): return "pil_image"
        if isinstance(source, np.ndarray): return "numpy_array"
        if isinstance(source, bytes): return "bytes"
        if isinstance(source, int): return "camera"

        source_str = str(source)
        if source_str == "screenshot": return "screenshot"
        if source_str.startswith('data:image'): return "base64_data_uri"
        
        # Basic base64 check
        try:
            if base64.b64encode(base64.b64decode(source_str, validate=True)) == source_str.encode():
                return "base64_string"
        except (TypeError, ValueError):
            pass

        if source_str.startswith(('http://', 'https://')): return "url"

        path = Path(source_str)
        if path.exists():
            if path.is_dir(): return "directory"
            if VideoFormat.is_valid(path.suffix): return "video_file"
            
            image_extensions = {fmt.name.lower() for fmt in ImageFormat}
            if path.suffix.lower().replace('.', '') in image_extensions:
                return "image_file"

        return "unknown"
    
    def _initialize_camera(self, camera_index: int):
        """Initialize camera handler."""
        try:
            import cv2
        except ImportError:
            raise ImageSourceError("opencv-python required. Install: pip install opencv-python")
        self.config.camera_index = camera_index
        self._camera = _CameraHandler(camera_index)
    
    def _cleanup(self):
        """Clean up resources."""
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._video:
            self._video.release()
            self._video = None
    
    def _change_colorspace(self, pil_image: Image.Image) -> Union[str, Image.Image, np.ndarray]:
        """Convert PIL image to configured output type."""
        output_type = self.config.output_type
        
        if output_type == OutputType.PIL:
            return pil_image
        if output_type == OutputType.BASE64:
            return self._pil_to_base64(pil_image)
        
        if output_type == OutputType.NUMPY:
            colorspace = self.config.colorspace
            conversion_map = {
                ColorSpace.RGB: "RGB",
                ColorSpace.RGBA: "RGBA",
                ColorSpace.BGR: "RGB"  # BGR needs special handling after conversion
            }
            
            mode = conversion_map.get(colorspace)
            if not mode:
                raise InvalidConfigError(f"Unsupported colorspace: {colorspace}")

            converted_image = pil_image.convert(mode)
            numpy_array = np.array(converted_image)

            if colorspace == ColorSpace.BGR:
                return numpy_array[:, :, ::-1]  # Reverse color channels
            return numpy_array
        
        raise InvalidConfigError(f"Unsupported output type: {output_type}")
    
    def _pil_to_base64(self, pil_image: Image.Image) -> str:
        """Convert PIL Image to base64."""
        buffered = io.BytesIO()
        pil_image.save(buffered, format=self.config.format.value)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def _base64_to_pil(self, base64_str: str) -> Image.Image:
        """Convert base64 to PIL Image."""
        image_bytes = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_bytes))
    
    def _load_pil_from_file(self, file_path: Path) -> Image.Image:
        """Load image from file as PIL image."""
        return Image.open(file_path)
    
    def _initialize_directory_files(self):
        """Initialize directory file list."""
        directory_path = Path(self.source)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        extensions = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                     for ext in self.config.extensions]
        
        if self.config.recursive:
            image_files = [f for f in directory_path.rglob('*') if f.suffix.lower() in extensions]
        else:
            image_files = [f for f in directory_path.glob('*') if f.suffix.lower() in extensions]
        
        image_files.sort()
        if self.config.max_images:
            image_files = image_files[:self.config.max_images]
        
        self._directory_files = image_files
    
    def _get_total_files(self) -> int:
        """Get total number of files in directory."""
        if self._directory_files is None:
            self._initialize_directory_files()
        return len(self._directory_files)
    
    # Source handlers
    def _handle_pil_source(self) -> ImageResult:
        data = self._change_colorspace(self.source)
        return ImageResult(data=data)
    
    def _handle_numpy_source(self) -> ImageResult:
        pil_img = Image.fromarray(self.source.astype('uint8'))
        data = self._change_colorspace(pil_img)
        return ImageResult(data=data)
    
    def _handle_bytes_source(self) -> ImageResult:
        pil_image = Image.open(io.BytesIO(self.source))
        data = self._change_colorspace(pil_image)
        return ImageResult(data=data)
    
    def _handle_camera_source(self) -> ImageResult:
        if self._camera is None:
            self._camera = _CameraHandler(self.config.camera_index)
        pil_image = self._camera.capture()
        data = self._change_colorspace(pil_image)
        return ImageResult(data=data)
    
    def _handle_video_source(self) -> ImageResult:
        if self._video is None:
            self._video = _VideoHandler(self.source)
        
        frame_data = None
        if self.config.use_frame_index:
            frame_data = self._video.get_frame_at_index(int(self.config.time_or_frame))
        else:
            frame_data = self._video.get_frame_at_time(self.config.time_or_frame)
        
        if frame_data is None:
            raise ImageSourceError(f"Could not read frame at {self.config.time_or_frame}")

        pil_image, frame_idx = frame_data
        data = self._change_colorspace(pil_image)
        timestamp = frame_idx / self._video.fps if self._video.fps > 0 else 0
        return ImageResult(
            data=data, 
            frame_index=frame_idx, 
            timestamp=timestamp,
            resolution=self._video.resolution
        )
    
    def _handle_screenshot_source(self) -> ImageResult:
        try:
            import pyautogui
        except ImportError:
            raise ImageSourceError("pyautogui required. Install: pip install pyautogui")
        screenshot = pyautogui.screenshot(region=self.config.region) if self.config.region else pyautogui.screenshot()
        data = self._change_colorspace(screenshot)
        return ImageResult(data=data)
    
    def _handle_image_file_source(self) -> ImageResult:
        file_path = Path(self.source)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        pil_image = self._load_pil_from_file(file_path)
        data = self._change_colorspace(pil_image)
        return ImageResult(data=data, filename=file_path.name, path=str(file_path))
    
    def _handle_url_source(self) -> ImageResult:
        try:
            import requests
        except ImportError:
            raise ImageSourceError("requests required. Install: pip install requests")
        try:
            response = requests.get(self.source, timeout=10)
            response.raise_for_status()
        except Exception as e:
            raise ImageSourceError(f"Failed to download image from URL: {e}")
        pil_image = Image.open(io.BytesIO(response.content))
        data = self._change_colorspace(pil_image)
        return ImageResult(data=data)
    
    def _handle_base64_string_source(self) -> ImageResult:
        pil_image = self._base64_to_pil(self.source)
        data = self._change_colorspace(pil_image)
        return ImageResult(data=data)

    def _handle_base64_data_uri_source(self) -> ImageResult:
        self.source = self.source.split(',', 1)[1]
        return self._handle_base64_string_source()
    
    # Batch operations
    def _get_directory_batch(self) -> List[ImageResult]:
        """Get next batch of images from directory."""
        if self._directory_files is None:
            self._initialize_directory_files()
        
        start_idx = self._directory_index
        end_idx = min(start_idx + self.config.batch_size, len(self._directory_files))
        batch_files = self._directory_files[start_idx:end_idx]
        self._directory_index = end_idx
        
        results = []
        for img_file in batch_files:
            try:
                pil_image = self._load_pil_from_file(img_file)
                data = self._change_colorspace(pil_image)
                results.append(ImageResult(data=data, filename=img_file.name, path=str(img_file)))
            except Exception as e:
                logger.warning(f"Failed to load {img_file}: {e}")
        return results
    
    def _get_video_frames(self) -> List[ImageResult]:
        """Get batch of frames from video."""
        if self._video is None:
            self._video = _VideoHandler(self.source)

        if self._video_generator is None:
            self._video_generator = self._video.frame_generator(self.config.video_interval)

        results = []
        for _ in range(self.config.batch_size):
            frame_data = next(self._video_generator, None)
            if frame_data is None:
                break  # Video finished

            pil_image, frame_idx = frame_data
            self._video.frames_extracted = frame_idx # Update progress
            
            data = self._change_colorspace(pil_image)
            timestamp = frame_idx / self._video.fps if self._video.fps > 0 else 0
            results.append(ImageResult(
                data=data,
                frame_index=frame_idx,
                timestamp=timestamp,
                resolution=self._video.resolution
            ))
        return results
    
    def _get_camera_batch(self) -> List[ImageResult]:
        """Capture batch of images from camera."""
        if self._camera is None:
            self._camera = _CameraHandler(self.config.camera_index)
        results = []
        for i in range(self.config.batch_size):
            try:
                pil_image = self._camera.capture()
                data = self._change_colorspace(pil_image)
                results.append(ImageResult(data=data))
            except Exception as e:
                logger.warning(f"Failed to capture frame {i}: {e}")
                break
        return results


# ============================================================================
# INTERNAL HANDLERS
# ============================================================================

class _CameraHandler:
    """Internal camera handler."""
    
    def __init__(self, camera_index: int = 0):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImageSourceError("opencv-python required")
        self.camera_index = camera_index
        self.cap = self.cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise ImageSourceError(f"Cannot open camera {camera_index}")
    
    def capture(self) -> Image.Image:
        if not self.cap or not self.cap.isOpened():
            raise ImageSourceError("Camera not opened")
        ret, frame = self.cap.read()
        if not ret:
            raise ImageSourceError("Failed to capture")
        frame_rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


class _VideoHandler:
    """Internal video handler."""
    
    def __init__(self, video_path: Union[str, Path]):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImageSourceError("opencv-python required")
        
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        if not VideoFormat.is_valid(self.video_path.suffix):
            raise UnsupportedFormatError(f"Unsupported video: {self.video_path.suffix}")
        
        self.cap = self.cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise ImageSourceError(f"Cannot open video: {self.video_path}")
        
        self.fps = self.cap.get(self.cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        self.width = int(self.cap.get(self.cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(self.cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = Resolution(self.width, self.height)
        self.frames_extracted = 0
        
        if self.frame_count == 0:
            raise ImageSourceError("Video has no frames")
    
    def has_more_frames(self) -> bool:
        return self.frames_extracted < self.frame_count
    
    def reset(self):
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
        self.frames_extracted = 0
    
    def _frame_to_pil(self, frame) -> Image.Image:
        frame_rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return pil_image
    
    def get_frame_at_time(self, seconds: float) -> Optional[tuple[Image.Image, int]]:
        if not self.cap or not self.cap.isOpened():
            raise ImageSourceError("Video not opened")
        self.cap.set(self.cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame_idx = int(self.cap.get(self.cv2.CAP_PROP_POS_FRAMES))
        return self._frame_to_pil(frame), frame_idx
    
    def get_frame_at_index(self, frame_index: int) -> Optional[tuple[Image.Image, int]]:
        if not self.cap or not self.cap.isOpened():
            raise ImageSourceError("Video not opened")
        self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if not ret:
            return None
        return self._frame_to_pil(frame), frame_index
    
    def frame_generator(self, interval_sec: Optional[float] = None) -> Generator[tuple[Image.Image, int], None, None]:
        if not self.cap or not self.cap.isOpened():
            raise ImageSourceError("Video not opened")

        while self.cap.isOpened():
            if interval_sec:
                current_msec = self.cap.get(self.cv2.CAP_PROP_POS_MSEC)
                self.cap.set(self.cv2.CAP_PROP_POS_MSEC, current_msec + interval_sec * 1000)

            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_idx = int(self.cap.get(self.cv2.CAP_PROP_POS_FRAMES))
            yield self._frame_to_pil(frame), frame_idx
    
    def get_info(self) -> dict:
        """Get video information as dictionary (legacy support)."""
        info = self.get_video_info()
        return {
            'fps': info.fps,
            'frame_count': info.frame_count,
            'duration': info.duration,
            'width': info.resolution.width,
            'height': info.resolution.height,
            'resolution': str(info.resolution)
        }
    
    def get_video_info(self) -> VideoInfo:
        """Get video information as VideoInfo object."""
        return VideoInfo(
            fps=self.fps,
            frame_count=self.frame_count,
            duration=self.duration,
            resolution=self.resolution
        )
    
    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_image(source: Union[str, Path, Image.Image], output_type: OutputType = OutputType.BASE64) -> ImageResult:
    """
    Quick function to load a single image.
    
    Args:
        source: Image source (file path, URL, PIL Image, etc.)
        output_type: Output format (BASE64 or PIL)
    
    Returns:
        ImageResult object
    
    Example:
        result = load_image('photo.jpg')
        base64_data = result.data
    """
    config = SourceConfig(output_type=output_type)
    img_source = ImageSource(source, config)
    return img_source.get_image()


def load_images_from_directory(directory: Union[str, Path], recursive: bool = False, 
                               batch_size: int = 10, max_images: Optional[int] = None,
                               output_type: OutputType = OutputType.BASE64) -> Generator[List[ImageResult], None, None]:
    """
    Generator function to load images from directory in batches.
    
    Args:
        directory: Directory path
        recursive: Search subdirectories
        batch_size: Number of images per batch
        max_images: Maximum total images to load
        output_type: Output format
    
    Yields:
        List of ImageResult objects (one batch at a time)
    
    Example:
        for batch in load_images_from_directory('./images', batch_size=10):
            for img in batch:
                process(img.data)
    """
    config = SourceConfig(
        output_type=output_type,
        recursive=recursive,
        batch_size=batch_size,
        max_images=max_images
    )
    
    with ImageSource(directory, config) as source:
        while source.has_more_images():
            yield source.get_images()


def load_video_frames(video_path: Union[str, Path], num_frames: int = 10,
                     output_type: OutputType = OutputType.BASE64) -> List[ImageResult]:
    """
    Quick function to extract frames from video.
    
    Args:
        video_path: Path to video file
        num_frames: Number of evenly spaced frames to extract
        output_type: Output format
    
    Returns:
        List of ImageResult objects
    
    Example:
        frames = load_video_frames('video.mp4', num_frames=20)
        for frame in frames:
            print(f"Frame {frame.frame_index} at {frame.timestamp}s")
    """
    config = SourceConfig(output_type=output_type, batch_size=num_frames)
    with ImageSource(video_path, config) as source:
        return source.get_images()


def capture_from_camera(camera_index: int = 0, num_captures: int = 1,
                       output_type: OutputType = OutputType.BASE64) -> Union[ImageResult, List[ImageResult]]:
    """
    Quick function to capture from camera.
    
    Args:
        camera_index: Camera device index (0 for default)
        num_captures: Number of frames to capture
        output_type: Output format
    
    Returns:
        Single ImageResult if num_captures=1, else List[ImageResult]
    
    Example:
        # Single capture
        result = capture_from_camera()
        
        # Multiple captures
        frames = capture_from_camera(num_captures=5)
    """
    config = SourceConfig(output_type=output_type, batch_size=num_captures)
    
    with ImageSource(camera_index, config) as source:
        if num_captures == 1:
            return source.get_image()
        else:
            return source.get_images()
