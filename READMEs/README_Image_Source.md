# ImageSource: A Unified Image Handling Library for Python

`ImageSource` is a Python class that provides a unified interface for handling multiple types of image sources. It is designed to simplify the process of loading and converting images for use in other applications, such as multimodal AI models.

---

## Table of Contents

1.  [Features](#features)
2.  [Installation](#installation)
3.  [Quick Start](#quick-start)
4.  [Usage Examples](#usage-examples)
    *   [Processing a Directory of Images](#processing-a-directory-of-images)
    *   [Iterating Through a Video](#iterating-through-a-video)
    *   [Taking a Screenshot](#taking-a-screenshot)
5.  [API Reference](#api-reference)
    *   [`ImageSource` Class](#imagesource-class)
    *   [`SourceConfig` Dataclass](#sourceconfig-dataclass)
    *   [`ImageResult` Dataclass](#imageresult-dataclass)
6.  [Convenience Functions](#convenience-functions)
7.  [Contributing](#contributing)
8.  [License](#license)

---

## Features

*   **Multiple Source Types**: Handles a range of inputs including files, directories, videos, webcams, screenshots, URLs, and in-memory data.
*   **Flexible Output**: Converts images into `Base64`, `PIL`, or `NumPy` formats.
*   **Batch Processing**: Processes images from directories or videos in batches.
*   **Configuration**: A dataclass is used to set parameters for image processing.
*   **Type Hinting**: The codebase is fully type-hinted.

### Supported Sources
*   Image Files (`.jpg`, `.png`, etc.)
*   Directories (with optional recursion)
*   Video Files (`.mp4`, `.avi`, etc.)
*   Webcams
*   Screenshots (full screen or regional)
*   URLs
*   In-Memory Data (PIL Images, NumPy arrays, raw bytes, Base64 strings)

---

## Installation

This class has several third-party dependencies. They can be installed using `pip`:

```bash
pip install pillow numpy opencv-python pyautogui requests
```

---

## Quick Start

This example demonstrates loading an image from a file and displaying it. The recommended way to use `ImageSource` is with a `with` statement to ensure resources are automatically managed.

```python
from image_source import ImageSource, OutputType, SourceConfig

# Use a 'with' statement to ensure the source is properly closed
with ImageSource('path/to/your/image.jpg') as img_source:
    # Configure the output to be a PIL Image
    img_source.set_config(SourceConfig(output_type=OutputType.PIL))

    # Get the image and display it
    image_result = img_source.get_image()
    image_result.data.show()
```

---

## Resource Management (Using the `with` Statement)

The `ImageSource` class is a context manager, which means it can be used with a `with` statement. This is the recommended approach, especially when working with sources that need to be explicitly released, such as video files or webcams.

The `with` statement ensures that the `.close()` method is automatically called, even if errors occur, preventing resource leaks.

---

## Usage Examples

### Processing a Directory of Images

This example recursively finds all `.png` and `.jpg` images in a directory and processes them in batches of 5, returning each image as a NumPy array.

```python
from image_source import ImageSource, SourceConfig, OutputType, ColorSpace

config = SourceConfig(
    recursive=True,
    batch_size=5,
    extensions=['.png', '.jpg'],
    output_type=OutputType.NUMPY,
    colorspace=ColorSpace.RGB
)
dir_source = ImageSource('path/to/image_directory', config)

while dir_source.has_more_images():
    image_batch = dir_source.get_images()
    print(f"Processing a batch of {len(image_batch)} images...")
    for image_result in image_batch:
        print(f"  - Image shape: {image_result.data.shape}")
```

### Iterating Through a Video

This example processes a video by extracting one frame every 5 seconds, handling them in batches of 4.

```python
from image_source import ImageSource, SourceConfig

config = SourceConfig(
    video_interval=5.0, # Time in seconds between frames
    batch_size=4
)
video_source = ImageSource('path/to/your/video.mp4', config)

while video_source.has_more_images():
    frames = video_source.get_images()
    if not frames:
        break
    
    print(f"Processing a batch of {len(frames)} frames...")
    for frame in frames:
        print(f"  - Frame Index: {frame.frame_index}, Timestamp: {frame.timestamp:.2f}s")
```

### Taking a Screenshot

This example captures a specific region of the screen.

```python
from image_source import ImageSource, SourceConfig, OutputType

config = SourceConfig(
    region=(100, 100, 500, 500), # (x, y, width, height)
    output_type=OutputType.PIL
)
screenshot_source = ImageSource("screenshot", config)
screenshot_result = screenshot_source.get_image()
screenshot_result.data.show()
```

---

## API Reference

### `ImageSource` Class

The main class for handling image sources.

**`ImageSource(source, config=None)`**
*   Initializes the class.
*   **Parameters:**
    *   `source`: The image source (e.g., file path, URL, camera index).
    *   `config`: An optional `SourceConfig` object.

**`get_image()`**
*   Returns a single `ImageResult` from the source.

**`get_images()`**
*   Returns a list of `ImageResult` objects (for batch operations).

**`has_more_images()`**
*   Returns `True` if there are more images to process.

**`set_source(source, config=None)`**
*   Changes the source and optionally updates the configuration.

**`set_config(config)`**
*   Updates the configuration.

**`set_screenshot()`**
*   Sets the source to capture a screenshot.

**`get_info()`**
*   Returns a dictionary with information about the current source.

**`close()`**
*   Releases any open resources (like a camera or video file).

### `SourceConfig` Dataclass

Used to configure the behavior of an `ImageSource` instance.

*   `output_type` (OutputType): The desired output format. Default: `OutputType.BASE64`.
*   `colorspace` (ColorSpace): The color space for NumPy output. Default: `ColorSpace.RGB`.
*   `format` (ImageFormat): The image format for Base64 encoding. Default: `ImageFormat.PNG`.
*   `recursive` (bool): Whether to search directories recursively. Default: `False`.
*   `batch_size` (int): The number of images/frames to process per batch. Default: `10`.
*   `camera_index` (int): The index of the camera to use. Default: `0`.

### `ImageResult` Dataclass

A container for the processed image and its metadata.

*   `data`: The processed image data.
*   `filename`: The original filename, if applicable.
*   `path`: The absolute path to the file, if applicable.
*   `frame_index`: The frame number, for video sources.
*   `timestamp`: The timestamp in seconds, for video sources.
*   `resolution`: The image resolution.

---

## Error Handling

The `ImageSource` class will raise specific exceptions when it encounters problems. This allows you to handle errors gracefully in your application.

*   `ImageSourceError`: The base exception for all errors from this library.
*   `InvalidConfigError`: Raised when the `SourceConfig` contains invalid or conflicting settings.
*   `SourceNotSetError`: Raised when you attempt to get an image before a source has been set.
*   `UnsupportedOperationError`: Raised when an operation is not supported for the current source type (e.g., calling `get_image()` on a directory source).
*   `UnsupportedFormatError`: Raised when a file format is not supported.

Example:
```python
from image_source import ImageSource, ImageSourceError

try:
    with ImageSource('path/to/invalid_file.xyz') as img_source:
        img_source.get_image()
except ImageSourceError as e:
    print(f"An error occurred: {e}")
```

## Thread Safety

The `ImageSource` class is **not thread-safe**. An instance of `ImageSource` maintains state, such as the current position in a directory or video file. Using the same instance across multiple threads will lead to unpredictable behavior.

For multi-threaded applications, create a separate `ImageSource` instance for each thread.

## Convenience Functions

These functions are available for common tasks:

*   `load_image(source, ...)`: Loads a single image.
*   `load_images_from_directory(directory, ...)`: A generator that yields batches of images from a directory.
*   `load_video_frames(video_path, ...)`: Extracts a specified number of frames from a video.
*   `capture_from_camera(camera_index=0, ...)`: Captures one or more images from a webcam.

---

## Contributing

Contributions are welcome. Please feel free to submit a pull request or open an issue.

---

## License

This project is licensed under the MIT License.
