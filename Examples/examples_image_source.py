"""
ImageSource Examples - Complete Usage Guide

This file demonstrates all features of the ImageSource class with practical examples.
Each function shows a different source type and use case.
"""

from image_source import (
    ImageSource, SourceConfig, OutputType, ImageFormat, VideoFormat,
    ImageSourceError, InvalidConfigError, SourceNotSetError, UnsupportedFormatError
)
from PIL import Image
import numpy as np


def example_format_validation():
    """Example: File format validation."""
    print("\n=== Example 0: Format Validation ===")
    
    # Valid video formats
    print("Supported video formats:")
    print(", ".join(VideoFormat.get_all_extensions()))
    
    # Try unsupported video format
    try:
        source = ImageSource('video.xyz')
        result = source.get_image()
    except UnsupportedFormatError as e:
        print(f"\n✓ Caught unsupported format: {e}")
    
    # Try unsupported image format
    try:
        source = ImageSource('image.xyz')
        result = source.get_image()
    except UnsupportedFormatError as e:
        print(f"✓ Caught unsupported format: {e}")
    
    # Valid formats
    print("\n✓ Valid video: .mp4, .avi, .mov, .mkv, etc.")
    print("✓ Valid image: .jpg, .png, .bmp, .gif, etc.")


def example_image_file():
    """Example: Load and process a single image file."""
    print("\n=== Example 1: Image File ===")
    
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.JPEG
    )
    
    try:
        source = ImageSource('photo.jpg', config)
        result = source.get_image()
        
        print(f"Loaded: {result.filename}")
        print(f"Path: {result.path}")
        print(f"Is Base64: {result.is_base64}")
        print(f"Data length: {len(result.data)}")
        
        # Send to LLM
        # llm_response = llm.generate(image=result.data, prompt="Describe this image")
        # print(f"LLM Response: {llm_response}")
        
    except FileNotFoundError:
        print("Error: Image file not found")


def example_pil_image():
    """Example: Work with PIL Image objects."""
    print("\n=== Example 2: PIL Image ===")
    
    # Create a PIL image
    pil_img = Image.new('RGB', (200, 200), color='blue')
    
    # Get as Base64
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.PNG
    )
    
    source = ImageSource(pil_img, config)
    result = source.get_image()
    
    print(f"Output type: {'Base64' if result.is_base64 else 'PIL'}")
    print(f"Data length: {len(result.data)}")
    
    # Change to PIL output
    config.output_type = OutputType.PIL
    source.set_config(config)
    result = source.get_image()
    
    print(f"Output type: {'Base64' if result.is_base64 else 'PIL'}")
    print(f"Image size: {result.data.size}")
    
    # Save the PIL image
    # result.data.save('output.png')


def example_numpy_array():
    """Example: Convert numpy arrays to images."""
    print("\n=== Example 3: Numpy Array ===")
    
    # Create a random numpy array (simulating image data)
    np_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    config = SourceConfig(
        output_type=OutputType.PIL,
        format=ImageFormat.PNG
    )
    
    source = ImageSource(np_array, config)
    result = source.get_image()
    
    print(f"Array shape: {np_array.shape}")
    print(f"Output type: {'Base64' if result.is_base64 else 'PIL'}")
    print(f"Image size: {result.data.size}")


def example_directory_batch():
    """Example: Process images from directory in batches."""
    print("\n=== Example 4: Directory Batch Processing ===")
    
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.JPEG,
        recursive=True,
        extensions=['.jpg', '.png', '.jpeg'],
        max_images=100,
        batch_size=10
    )
    
    try:
        source = ImageSource('./images', config)
        
        print(f"Source: {source}")
        
        # Get initial info
        info = source.get_info()
        print(f"Total files: {info['total_files']}")
        print(f"Batch size: {info['batch_size']}")
        print(f"Recursive: {info['recursive']}")
        
        # Process in batches
        batch_count = 0
        while source.has_more_images():
            images = source.get_images()
            batch_count += 1
            
            print(f"\nBatch {batch_count}: {len(images)} images")
            
            for result in images:
                print(f"  - {result.filename}")
                # Process with LLM
                # response = llm.analyze(result.data)
                # print(f"    Analysis: {response}")
            
            # Show progress
            info = source.get_info()
            progress = (info['processed_files'] / info['total_files']) * 100
            print(f"Progress: {info['processed_files']}/{info['total_files']} ({progress:.1f}%)")
        
        print("\nAll images processed!")
        
    except FileNotFoundError:
        print("Error: Directory not found")


def example_video_frames():
    """Example: Extract frames from video."""
    print("\n=== Example 5: Video Frame Extraction ===")
    
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.JPEG,
        video_num_frames=10  # Extract 10 evenly spaced frames
    )
    
    try:
        source = ImageSource('video.mp4', config)
        
        # Get video info
        info = source.get_video_info()
        print(f"Video duration: {info['duration']:.2f}s")
        print(f"FPS: {info['fps']:.2f}")
        print(f"Resolution: {info['width']}x{info['height']}")
        print(f"Total frames: {info['frame_count']}")
        
        # Extract frames
        print("\nExtracting frames...")
        frames = source.get_images()
        
        print(f"Extracted {len(frames)} frames:")
        for result in frames:
            print(f"  Frame {result.frame_index} at {result.timestamp:.2f}s")
            # Process frame
            # analysis = llm.analyze(result.data)
            # print(f"    {analysis}")
        
        # Alternative: Extract by time interval
        print("\n--- Using time interval ---")
        config.video_interval = 2.0  # Every 2 seconds
        config.video_num_frames = None
        source.set_config(config)
        source.reset()
        
        frames = source.get_images()
        print(f"Extracted {len(frames)} frames (every 2s)")
        
    except FileNotFoundError:
        print("Error: Video file not found")


def example_video_single_frame():
    """Example: Get a single frame from video."""
    print("\n=== Example 6: Video Single Frame ===")
    
    config = SourceConfig(
        output_type=OutputType.PIL,
        time_or_frame=5.0,  # Get frame at 5 seconds
        use_frame_index=False
    )
    
    try:
        source = ImageSource('video.mp4', config)
        result = source.get_image()
        
        print(f"Frame at {result.timestamp:.2f}s")
        print(f"Frame index: {result.frame_index}")
        
        # Save the frame
        # result.data.save('frame_at_5s.jpg')
        
        # Get frame by index instead
        config.use_frame_index = True
        config.time_or_frame = 100  # Frame 100
        source.set_config(config)
        
        result = source.get_image()
        print(f"\nFrame {result.frame_index} at {result.timestamp:.2f}s")
        
    except FileNotFoundError:
        print("Error: Video file not found")


def example_camera_single():
    """Example: Capture single image from camera."""
    print("\n=== Example 7: Camera Single Capture ===")
    
    config = SourceConfig(
        output_type=OutputType.PIL,
        format=ImageFormat.JPEG
    )
    
    try:
        # Just pass camera index directly!
        source = ImageSource(0, config)  # Camera 0
        
        print("Capturing from camera...")
        result = source.get_image()
        
        print(f"Captured! Image size: {result.data.size}")
        
        # Save the capture
        # result.data.save('camera_capture.jpg')
        
        source.close()
        print("Camera released")
        
    except ImageSourceError as e:
        print(f"Camera error: {e}")


def example_camera_batch():
    """Example: Capture multiple frames from camera."""
    print("\n=== Example 8: Camera Batch Capture ===")
    
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.JPEG,
        batch_size=5  # Capture 5 frames at once
    )
    
    try:
        source = ImageSource(0, config)
        
        print("Capturing 5 frames from camera...")
        frames = source.get_images()
        
        print(f"Captured {len(frames)} frames:")
        for i, result in frames:
            print(f"  Frame {i}: {len(result.data)} bytes")
            # Process with LLM
            # response = llm.analyze(result.data)
        
        # Capture more batches
        print("\nCapturing 3 more batches...")
        for batch_num in range(3):
            frames = source.get_images()
            print(f"  Batch {batch_num + 1}: {len(frames)} frames")
        
        source.close()
        print("\nCamera released")
        
    except ImageSourceError as e:
        print(f"Camera error: {e}")


def example_screenshot():
    """Example: Capture screenshot."""
    print("\n=== Example 9: Screenshot ===")
    
    config = SourceConfig(
        output_type=OutputType.PIL,
        format=ImageFormat.PNG
    )
    
    try:
        # Full screen
        source = ImageSource('screenshot', config)
        result = source.get_image()
        
        print(f"Screenshot captured: {result.data.size}")
        # result.data.save('screenshot.png')
        
        # Specific region
        config.region = (100, 100, 800, 600)  # x, y, width, height
        source.set_config(config)
        
        result = source.get_image()
        print(f"Region screenshot: {result.data.size}")
        
    except ImageSourceError as e:
        print(f"Screenshot error: {e}")


def example_url():
    """Example: Download image from URL."""
    print("\n=== Example 10: URL Image ===")
    
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.JPEG
    )
    
    # Example URLs (replace with real URLs to test)
    urls = [
        "https://picsum.photos/200/300",  # Random image
        "https://via.placeholder.com/150",  # Placeholder
    ]
    
    for url in urls:
        try:
            print(f"\nDownloading from: {url}")
            source = ImageSource(url, config)
            
            result = source.get_image()
            
            print(f"✓ Downloaded successfully!")
            print(f"  Data length: {len(result.data)} bytes")
            print(f"  Is Base64: {result.is_base64}")
            
            # Can also get as PIL for inspection
            config.output_type = OutputType.PIL
            source.set_config(config)
            result = source.get_image()
            print(f"  Image size: {result.data.size}")
            print(f"  Image mode: {result.data.mode}")
            
            # Reset config
            config.output_type = OutputType.BASE64
            
        except ImageSourceError as e:
            print(f"✗ Download error: {e}")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    # Multiple URLs example
    print("\n--- Processing Multiple URLs ---")
    image_urls = [
        "https://example.com/image1.jpg",
        "https://example.com/image2.jpg",
        "https://example.com/image3.jpg",
    ]
    
    print(f"Processing {len(image_urls)} URLs...")
    for i, url in enumerate(image_urls, 1):
        try:
            source = ImageSource(url, config)
            result = source.get_image()
            print(f"  {i}. Downloaded: {len(result.data)} bytes")
            # Process with LLM
            # response = llm.analyze(result.data)
        except Exception as e:
            print(f"  {i}. Failed: {e}")


def example_error_handling():
    """Example: Proper error handling."""
    print("\n=== Example 11: Error Handling ===")
    
    # Invalid config
    try:
        config = SourceConfig(
            video_interval=2.0,
            video_num_frames=10  # Can't set both!
        )
    except InvalidConfigError as e:
        print(f"Config error: {e}")
    
    # No source set
    try:
        source = ImageSource()
        result = source.get_image()
    except SourceNotSetError as e:
        print(f"Source error: {e}")
    
    # Wrong method for source
    try:
        source = ImageSource('./images')
        result = source.get_image()  # Should use get_images()
    except Exception as e:
        print(f"Operation error: {e}")
    
    # File not found
    try:
        source = ImageSource('nonexistent.jpg')
        result = source.get_image()
    except FileNotFoundError as e:
        print(f"File error: {e}")


def example_context_manager():
    """Example: Using context manager for auto cleanup."""
    print("\n=== Example 12: Context Manager ===")
    
    config = SourceConfig(
        output_type=OutputType.BASE64,
        recursive=True,
        batch_size=5
    )
    
    # Automatically closes/releases resources
    try:
        with ImageSource('./images', config) as source:
            print(f"Source: {source}")
            
            processed = 0
            while source.has_more_images():
                images = source.get_images()
                processed += len(images)
                print(f"Processed: {processed} images")
        
        print("Resources automatically cleaned up!")
        
    except FileNotFoundError:
        print("Directory not found")


def example_debugging():
    """Example: Debugging and introspection."""
    print("\n=== Example 13: Debugging ===")
    
    pil_img = Image.new('RGB', (100, 100), color='red')
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.PNG,
        batch_size=10
    )
    
    source = ImageSource(pil_img, config)
    
    # Debug info
    print(f"repr: {repr(source)}")
    print(f"str:  {str(source)}")
    print(f"Source type: {source.get_source_type()}")
    
    # Get detailed info
    info = source.get_info()
    print(f"Info: {info}")
    
    # Check result
    result = source.get_image()
    print(f"Result is Base64: {result.is_base64}")
    print(f"Result is PIL: {result.is_pil}")


def example_llm_integration():
    """Example: Real-world LLM integration workflow."""
    print("\n=== Example 14: LLM Integration Workflow ===")
    
    # Process directory of images with LLM
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.JPEG,
        recursive=True,
        batch_size=10
    )
    
    try:
        source = ImageSource('./product_images', config)
        
        results = []
        
        print("Processing images with LLM...")
        while source.has_more_images():
            images = source.get_images()
            
            for result in images:
                print(f"Analyzing: {result.filename}")
                
                # Send to LLM
                # response = llm.generate(
                #     image=result.data,
                #     prompt="Describe this product image in detail"
                # )
                
                # Store results
                # results.append({
                #     'filename': result.filename,
                #     'path': result.path,
                #     'description': response
                # })
            
            # Show progress
            info = source.get_info()
            print(f"Progress: {info['processed_files']}/{info['total_files']}")
        
        print(f"\nProcessed {len(results)} images!")
        
        # Save results
        # import json
        # with open('results.json', 'w') as f:
        #     json.dump(results, f, indent=2)
        
    except FileNotFoundError:
        print("Directory not found")


def example_video_analysis():
    """Example: Analyze video with LLM frame by frame."""
    print("\n=== Example 15: Video Analysis with LLM ===")
    
    config = SourceConfig(
        output_type=OutputType.BASE64,
        format=ImageFormat.JPEG,
        video_interval=1.0  # Analyze every second
    )
    
    try:
        source = ImageSource('presentation.mp4', config)
        
        info = source.get_video_info()
        print(f"Analyzing {info['duration']:.1f}s video...")
        
        frames = source.get_images()
        
        timeline = []
        for result in frames:
            print(f"Frame at {result.timestamp:.1f}s...")
            
            # Analyze frame
            # description = llm.generate(
            #     image=result.data,
            #     prompt="What's happening in this frame?"
            # )
            
            # timeline.append({
            #     'time': result.timestamp,
            #     'frame': result.frame_index,
            #     'description': description
            # })
        
        print(f"\nAnalyzed {len(frames)} frames")
        
        # Generate summary
        # summary = llm.generate(
        #     prompt=f"Summarize this video: {timeline}"
        # )
        # print(f"Summary: {summary}")
        
    except FileNotFoundError:
        print("Video not found")


def main():
    """Run all examples."""
    print("=" * 60)
    print("ImageSource - Complete Examples")
    print("=" * 60)
    
    # Format validation
    example_format_validation()
    
    # Basic sources
    example_image_file()
    example_pil_image()
    example_numpy_array()
    
    # Batch sources
    example_directory_batch()
    example_video_frames()
    example_video_single_frame()
    example_camera_single()
    example_camera_batch()
    
    # Other sources
    example_screenshot()
    example_url()
    
    # Advanced usage
    example_error_handling()
    example_context_manager()
    example_debugging()
    
    # Real-world scenarios
    example_llm_integration()
    example_video_analysis()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
