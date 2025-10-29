# WebScreenShot - Quick Start

## Installation

```bash
pip install playwright pyyaml pillow
playwright install chromium
```

## Three Ways to Use

### 1. Simple Function (Easiest)

```python
from web_screenshot import screenshot

# Basic
screenshot("https://example.com", "output.png")

# Full page in one shot
screenshot("https://example.com", "output.png", viewport_auto_adjust=True)

# See progress
screenshot("https://example.com", "output.png", verbose=True)
```

### 2. Class (For Multiple Screenshots)

```python
from web_screenshot import WebScreenShot

with WebScreenShot() as ws:
    ws.capture("https://example.com", "site1.png")
    ws.capture("https://github.com", "site2.png")
```

### 3. Command Line (No Code)

```bash
# Basic
python web_screenshot.py https://example.com

# Full page capture
python web_screenshot.py https://example.com --auto-adjust

# See progress
python web_screenshot.py https://example.com --verbose

# Custom output
python web_screenshot.py https://example.com my_screenshot.png

# Use config file
python web_screenshot.py https://example.com --config my_config.yaml
```

## Key Features

| Feature | How to Use |
|---------|------------|
| **Full page in one screenshot** | `viewport_auto_adjust=True` or `--auto-adjust` |
| **Custom viewport** | `viewport_width=1920, viewport_height=1080` or `--width 1920 --height 1080` |
| **See progress** | `verbose=True` or `--verbose` |
| **Use config file** | `config_file="config.yaml"` or `--config config.yaml` |
| **Reuse browser** | Use `with WebScreenShot() as ws:` |

## Config File Template

Save as `web_screenshot_config.yaml`:

```yaml
viewport:
  auto_adjust: true  # Capture full page in one shot
  max_width: 7680
  max_height: 43200

timeouts:
  navigation_timeout: 60000
  action_timeout: 60000
  final_wait: 5000

logging:
  verbose: true
```

Then use it:
```python
from web_screenshot import screenshot
screenshot("https://example.com", config_file="web_screenshot_config.yaml")
```

Or:
```bash
python web_screenshot.py https://example.com --config web_screenshot_config.yaml
```

## That's It!

For more details, see:
- `SIMPLE_USAGE.md` - Common scenarios
- `WEB_SCREENSHOT_README.md` - Full documentation
- `web_screenshot_example.py` - Working examples
