# WebScreenShot Software Contract

## Overview
WebScreenShot is a Python tool for capturing screenshots of websites using Playwright. It provides multiple capture modes to handle different use cases.

---

## Core Capabilities

### 1. Single Viewport Screenshot (Default Mode)
**What it does:**
- Captures one screenshot based on the browser viewport size
- Scrolls through the page to load lazy-loaded images
- Waits for network idle before capturing
- Returns True on success, False on error

**Configuration:**
- `viewport_width`: Custom width in pixels (default: browser default)
- `viewport_height`: Custom height in pixels (default: browser default)
- `scroll_delay`: Delay between scroll actions (default: 1.0 seconds)
- `final_wait`: Wait time after network idle (default: 5000ms)

**CLI:**
```bash
python web_screenshot.py --url https://example.com
python web_screenshot.py --url https://example.com --output output.png
```

**Python:**
```python
with WebScreenShot() as shot:
    shot.capture(url, output_path)
```

**Output:** Single PNG file at specified path or auto-generated timestamp filename

---

### 2. Full-Page Single Screenshot (Auto-Adjust Mode)
**What it does:**
- Captures the ENTIRE webpage in a single PNG file
- Auto-adjusts viewport to match full page dimensions
- Handles multi-page websites by expanding viewport
- No scrolling needed - entire page captured in one shot
- Respects maximum viewport limits (default: 7680x43200)

**Configuration:**
- `viewport_auto_adjust`: Set to True to enable
- `max_viewport_width`: Maximum width limit (default: 7680px)
- `max_viewport_height`: Maximum height limit (default: 43200px)
- Inherits other timeout and scroll settings

**CLI:**
```bash
python web_screenshot.py --url https://example.com --full-page
python web_screenshot.py --url https://example.com --full-page --output full_page.png
```

**Python:**
```python
config = ScreenShotConfig(viewport_auto_adjust=True)
with WebScreenShot() as shot:
    shot.config = config
    shot.capture(url, output_path)
```

**Output:** Single PNG file containing entire webpage

---

### 3. Multi-Page Screenshot (Separate Files Mode)
**What it does:**
- Captures webpage in viewport-sized chunks
- Saves each chunk as a separate PNG file (page_1.png, page_2.png, etc.)
- Automatically calculates number of pages needed
- Loads lazy content before capturing each page
- Creates output directory automatically if not specified

**Configuration:**
- Uses viewport dimensions for page sizing
- `scroll_delay`: Delay between page captures (default: 1.0 seconds)
- `final_wait`: Wait before each page capture (default: 5000ms)

**CLI:**
```bash
python web_screenshot.py --url https://example.com --multi-page
python web_screenshot.py --url https://example.com --multi-page --output my_screenshots
```

**Python:**
```python
with WebScreenShot() as shot:
    shot.capture_multiple_pages(url, output_dir)
```

**Output:** Directory with multiple PNG files:
- `page_1.png`
- `page_2.png`
- `page_3.png`
- ... (as many as needed)

---

## Configuration Methods

### Method 1: Default Configuration
All parameters use built-in defaults.

```python
shot = WebScreenShot()
```

### Method 2: YAML Configuration File
Load settings from a YAML file.

```python
shot = WebScreenShot(config_file='web_screenshot_config.yaml')
```

**YAML Structure:**
```yaml
browser:
  headless: true

timeouts:
  navigation_timeout: 60000
  action_timeout: 60000
  final_wait: 5000

scroll:
  delay: 1.0

viewport:
  width: null
  height: null
  auto_adjust: false
  max_width: 7680
  max_height: 43200

logging:
  verbose: false
```

---

## Default Configuration Values

| Parameter | Default | Type | Notes |
|-----------|---------|------|-------|
| `navigation_timeout` | 60000 | int | Milliseconds |
| `action_timeout` | 60000 | int | Milliseconds |
| `scroll_delay` | 1.0 | float | Seconds |
| `final_wait` | 5000 | int | Milliseconds |
| `viewport_width` | None | Optional[int] | Uses browser default if None |
| `viewport_height` | None | Optional[int] | Uses browser default if None |
| `viewport_auto_adjust` | False | bool | Only for full-page mode |
| `max_viewport_width` | 7680 | int | Limit for auto-adjust |
| `max_viewport_height` | 43200 | int | Limit for auto-adjust |
| `headless` | True | bool | Browser visibility |
| `verbose` | False | bool | Logging level |

---

## Error Handling

### URL Validation
- Validates URL scheme (must be http or https)
- Validates URL format (must have scheme and netloc)
- **On error:** Creates blank white image (800x600 default)
- **Return value:** False

### Network Errors
- Waits for `networkidle` state before capturing
- Respects timeout settings
- **On error:** Returns False and creates blank image

### Directory Creation (Multi-Page)
- Automatically creates output directory with parents
- Creates with timestamp if not specified
- **On error:** Returns False

### File System Errors
- Handles permission errors gracefully
- Logs errors with details
- **On error:** Returns False and appropriate error message

---

## URL Handling

### Valid URLs
- Must have scheme: `http://` or `https://`
- Must have netloc (domain)
- Examples: `https://example.com`, `https://example.com/path?query=1`

### Invalid URLs
- Missing scheme: `example.com`
- Invalid scheme: `ftp://example.com`
- Malformed URLs

### On Invalid URL
- Logs warning message
- Creates blank white image as fallback
- Returns False
- Continues execution without raising exception

---

## Return Values

All capture methods return `bool`:
- **True:** Screenshot captured successfully and saved
- **False:** Error occurred (blank image created or operation failed)

---

## Resource Management

### Browser Lifecycle
- Browser launches on first use
- Browser reused for multiple captures
- Browser closed when context manager exits (`with` statement)

### Memory
- Browser instance cached during context manager
- Proper cleanup on exit
- Page instances created and closed per capture

### File Handles
- Output files closed after write
- Logs written to stderr/stdout
- No dangling file handles

---

## CLI Interface

### Required Arguments
- `--url / -u`: Website URL to capture (required)

### Optional Arguments
- `--output / -o`: Output file/directory path (optional)
- `--full-page / -f`: Capture entire page in one screenshot (flag)
- `--multi-page / -m`: Capture as separate page files (flag)

### Constraints
- Cannot use `--full-page` and `--multi-page` together
- Program exits with code 1 on error
- Program exits with code 0 on success

### Examples
```bash
# Default mode
python web_screenshot.py --url https://example.com

# Full page
python web_screenshot.py -u https://example.com -f

# Multi-page
python web_screenshot.py -u https://example.com -m --output screenshots

# With output
python web_screenshot.py --url https://example.com --output my_screenshot.png
```

---

## Lazy-Loaded Images

### Handling
- Converts lazy-loaded images to eager on page load
- Scrolls through entire page to trigger image loading
- Waits for network idle between scrolls

### Scroll Mechanism
- Scrolls by viewport height increments
- Respects `scroll_delay` between scroll actions
- Continues until entire page scrolled

---

## Logging

### Logging Levels
- **Verbose=False:** WARNING level only
- **Verbose=True:** INFO level with timestamps

### Logged Information
- Browser launch/close events
- Page navigation and load states
- Screenshot capture start/end
- Errors and warnings
- Multi-page capture progress

---

## Context Manager Support

### Usage
```python
with WebScreenShot() as shot:
    shot.capture(url, output)
    shot.capture(url2, output2)
```

### Guarantees
- Browser automatically opens on entry (`__enter__`)
- Browser automatically closes on exit (`__exit__`)
- Works even if exceptions occur in with-block

---

## Limitations

1. **Single Process:** Screenshots captured sequentially, not in parallel
2. **Single Browser:** One browser instance per WebScreenShot object
3. **Network Dependent:** Requires active internet connection
4. **JavaScript:** Waits for dynamic content but respects timeout limits
5. **Memory:** Full-page mode limited by `max_viewport_height` (default 43200)
6. **Viewport Max:** Auto-adjust respects maximum dimensions
7. **File System:** Output directory must be writable

---

## Not In Scope

The following are NOT guaranteed by this software:

- Authentication/Login to websites
- Cookie handling or session management
- JavaScript execution beyond Playwright defaults
- PDF conversion or other formats
- Video/animation capture
- Network speed optimization
- CSS/JavaScript debugging
- Content extraction or parsing
- DOM manipulation beyond lazy-load handling

---

## Version & Dependencies

- **Python Version:** 3.7+
- **Dependencies:**
  - `playwright` (browser automation)
  - `pillow` (image handling)
  - `pyyaml` (configuration files)

---

## Summary

| Capability | Supported | Notes |
|-----------|-----------|-------|
| Single viewport screenshot | ✅ | Default mode |
| Full-page single screenshot | ✅ | Auto-adjust viewport |
| Multi-page separate files | ✅ | Separate PNG per viewport chunk |
| Custom viewport size | ✅ | Via config |
| Lazy-loaded images | ✅ | Auto-handled |
| YAML configuration | ✅ | Optional |
| Error recovery | ✅ | Blank image fallback |
| Context manager | ✅ | Auto cleanup |
| CLI interface | ✅ | Full-featured |
| Headless browser | ✅ | Default |
| Verbose logging | ✅ | Optional |

