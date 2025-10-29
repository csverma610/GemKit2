# Rate Limiting & Popup Handling Improvements

## Problem
When capturing multiple pages, frequent requests triggered rate-limiting popups showing "too many requests sent from the IP address".

## Solution
Enhanced `web_screenshot.py` with several defensive mechanisms:

### 1. **Request Throttling** (web_screenshot.py:507-508)
- Adds configurable delay between each HTTP request
- Default: 0.5 seconds per request
- Configurable via `request_delay` in config

```yaml
throttling:
  request_delay: 0.5  # seconds between requests
```

### 2. **Page Capture Delays** (web_screenshot.py:336-338)
- Adds delay between capturing successive pages
- Default: 2 seconds between pages
- Prevents hammering server with consecutive captures

```yaml
throttling:
  page_delay: 2.0  # seconds between page captures
```

### 3. **Automatic Popup Dismissal** (web_screenshot.py:515-558)
- Detects and closes common popup types:
  - Alert dialogs
  - Modal windows
  - Close buttons
  - Cookie notices
  - Rate-limit warning dialogs
- Fires before each page screenshot

### 4. **Optional Throttling Control**
- Toggle throttling on/off via `enable_request_throttling` config
- Disabled = faster but risker (for trusted sites)
- Enabled = slower but safer (default)

```yaml
throttling:
  enable_request_throttling: true  # Set to false to disable
```

## Configuration Example

Create `web_screenshot_config.yaml`:

```yaml
throttling:
  page_delay: 2.0              # Wait 2s between pages
  request_delay: 0.5           # Wait 0.5s between requests
  enable_request_throttling: true

# Adjust if needed:
timeouts:
  navigation_timeout: 60000
  final_wait: 5000
```

## Usage

### Multi-page capture (with rate limiting protection)
```python
shot = WebScreenShot('web_screenshot_config.yaml')
shot.capture_multiple_pages('https://example.com', 'output_dir')
```

### For fast sites (disable throttling)
```yaml
throttling:
  enable_request_throttling: false
  page_delay: 1.0
```

## Implementation Details

- **Request routing** (web_screenshot.py:507-508): Routes all requests through throttle handler
- **Popup detection** (web_screenshot.py:527-544): Uses multiple CSS selectors for broad popup coverage
- **Graceful degradation**: If popup detection fails, capture still proceeds
- **Logging**: Debug logs show which popups were dismissed

## Performance Impact

- With throttling: ~2.5s per page (2s delay + 0.5s per request * N requests)
- Without throttling: Original speed (risky for sites with rate limits)

Adjust `page_delay` and `request_delay` based on your target sites' tolerance.
