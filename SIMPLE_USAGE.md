# Simple Usage Guide

## The Simplest Way to Use WebScreenShot

### 1. Basic Screenshot (One Liner)

```python
from web_screenshot import WebScreenShot

WebScreenShot().capture("https://example.com", "output.png")
```

That's it! The screenshot will be saved to `output.png`.

### 2. Command Line (Even Simpler)

```bash
python web_screenshot.py https://example.com
```

Screenshot saved with auto-generated timestamp filename.

### 3. Capture Entire Page in One Screenshot

```python
from web_screenshot import WebScreenShot

WebScreenShot(viewport_auto_adjust=True).capture("https://example.com", "full_page.png")
```

Or from command line:
```bash
python web_screenshot.py https://example.com --auto-adjust
```

### 4. Multiple Screenshots (Reuse Browser)

```python
from web_screenshot import WebScreenShot

with WebScreenShot() as shot:
    shot.capture("https://example.com", "site1.png")
    shot.capture("https://github.com", "site2.png")
    shot.capture("https://stackoverflow.com", "site3.png")
```

### 5. Use Config File for Defaults

Create `web_screenshot_config.yaml`:
```yaml
viewport:
  auto_adjust: true  # Always capture full page in one shot
logging:
  verbose: true      # See what's happening
```

Then use it:
```python
from web_screenshot import WebScreenShot

WebScreenShot(config_file="web_screenshot_config.yaml").capture("https://example.com")
```

Or from command line:
```bash
python web_screenshot.py https://example.com --config web_screenshot_config.yaml
```

## Common Scenarios

### Scenario 1: I just want a screenshot quickly
```bash
python web_screenshot.py https://example.com
```

### Scenario 2: I want to see progress
```bash
python web_screenshot.py https://example.com --verbose
```

### Scenario 3: The page is very long, capture it all at once
```bash
python web_screenshot.py https://example.com --auto-adjust
```

### Scenario 4: Specific viewport size (e.g., mobile)
```bash
python web_screenshot.py https://example.com --width 375 --height 667
```

### Scenario 5: Automate multiple screenshots in Python
```python
from web_screenshot import WebScreenShot

urls = ["https://site1.com", "https://site2.com", "https://site3.com"]

with WebScreenShot() as shot:
    for i, url in enumerate(urls):
        shot.capture(url, f"screenshot_{i}.png")
```

## Default Behavior (No Config Needed)

By default, WebScreenShot:
- ✅ Uses headless browser (runs in background)
- ✅ Waits 60 seconds max for pages to load
- ✅ Scrolls through page to load lazy images
- ✅ Creates blank image if URL is invalid
- ✅ Auto-generates filename with timestamp if not specified
- ✅ Handles errors gracefully

## When to Use Config File

Only use a config file if you want to:
- Change default timeouts
- Always use auto-adjust mode
- Set custom max dimensions
- Change logging levels

Otherwise, the defaults work great for most use cases!
