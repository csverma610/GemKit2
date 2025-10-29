# LMDB Storage

A Python wrapper for LMDB (Lightning Memory-Mapped Database) that provides a simple key-value storage interface with optional compression and logging.

## Overview

This module provides a straightforward way to store and retrieve string key-value pairs using LMDB. It handles compression automatically for large values and includes built-in logging for operations monitoring.

## Features

- Key-value storage with string keys and values
- Automatic gzip compression for values exceeding a configurable threshold
- Optional logging with per-instance log files
- Context manager support for automatic resource cleanup
- JSON import/export functionality
- Configurable database capacity and compression settings
- Key size validation

## Requirements

```bash
pip install lmdb
```

## Basic Usage

### Simple Example

```python
from lmdb_storage import LMDBStorage

# Create storage instance
storage = LMDBStorage("mydata.lmdb")

# Store data
storage.put("user:1", "John Doe")
storage.put("user:2", "Jane Smith")

# Retrieve data
name = storage.get("user:1")
print(name)  # Output: John Doe

# Check if key exists
if storage.exists("user:1"):
    print("Key exists")

# Delete a key
storage.delete("user:1")

# Close when done
storage.close()
```

### Using Context Manager (Recommended)

```python
with LMDBStorage("mydata.lmdb") as storage:
    storage.put("config:theme", "dark")
    theme = storage.get("config:theme")
    # Automatically closed when exiting the block
```

## Configuration

### Using Individual Parameters

```python
storage = LMDBStorage(
    db_path="mydata.lmdb",
    capacity_mb=200,
    enable_logging=True,
    compression_threshold=1024
)
```

### Using Configuration Dataclass

```python
from lmdb_storage import LMDBStorage, LMDBConfig

config = LMDBConfig(
    db_path="mydata.lmdb",
    capacity_mb=200,
    enable_logging=True,
    compression_threshold=1024,
    max_key_size=511
)

storage = LMDBStorage(config=config)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `db_path` | str | `"storage.lmdb"` | Path to the LMDB database file |
| `capacity_mb` | int | `100` | Maximum database size in megabytes |
| `enable_logging` | bool | `True` | Enable/disable logging |
| `compression_threshold` | int | `100` | Size in bytes above which values are compressed |
| `max_key_size` | int | `511` | Maximum key size in bytes (LMDB limit) |

## API Reference

### Core Methods

#### `put(key: str, value: str) -> bool`
Stores a key-value pair. Returns `True` on success, `False` on failure.

```python
success = storage.put("user:1", "John Doe")
```

#### `get(key: str) -> Optional[str]`
Retrieves a value by key. Returns `None` if key doesn't exist.

```python
value = storage.get("user:1")
```

#### `exists(key: str) -> bool`
Checks if a key exists without retrieving the value.

```python
if storage.exists("user:1"):
    print("Key found")
```

#### `delete(key: str) -> bool`
Deletes a key. Returns `True` if deleted, `False` if key not found.

```python
storage.delete("user:1")
```

#### `clear() -> int`
Deletes all entries. Returns the number of deleted entries.

```python
count = storage.clear()
print(f"Deleted {count} entries")
```

### Utility Methods

#### `num_keys() -> int`
Returns the total number of keys in the database.

```python
total = storage.num_keys()
```

#### `get_keys(as_generator: bool = False) -> list | generator`
Retrieves all keys. Use `as_generator=True` for memory efficiency with large databases.

```python
# As list
keys = storage.get_keys()

# As generator (memory efficient)
for key in storage.get_keys(as_generator=True):
    print(key)
```

#### `get_stats() -> dict`
Returns database statistics from LMDB.

```python
stats = storage.get_stats()
print(stats)
```

### JSON Import/Export

#### `export_to_json(json_file_path: str) -> bool`
Exports all key-value pairs to a JSON file.

```python
storage.export_to_json("backup.json")
```

Output format:
```json
[
    {"key": "user:1", "value": "John Doe"},
    {"key": "user:2", "value": "Jane Smith"}
]
```

#### `import_from_json(json_file_path: str) -> bool`
Imports key-value pairs from a JSON file.

```python
storage.import_from_json("backup.json")
```

### Resource Management

#### `close()`
Closes the database and releases resources. Called automatically when using context manager.

```python
storage.close()
```

## Compression

Values larger than `compression_threshold` bytes are automatically compressed using gzip. Compression is transparent to the user:

```python
# Small value - stored uncompressed
storage.put("small", "Hello")

# Large value - automatically compressed
large_text = "x" * 10000
storage.put("large", large_text)

# Retrieval is the same regardless
value = storage.get("large")  # Automatically decompressed
```

## Logging

When logging is enabled, operations are logged to a file named after the database file:

- Database file: `mydata.lmdb`
- Log file: `mydata_lmdb.log`

Disable logging if not needed:

```python
storage = LMDBStorage("mydata.lmdb", enable_logging=False)
```

## Limitations

- Keys must not exceed 511 bytes when UTF-8 encoded (LMDB limitation, configurable via `max_key_size`)
- Both keys and values must be strings
- Empty keys and `None` values are rejected
- Database size is fixed at initialization (specified by `capacity_mb`)

## Error Handling

Methods return `False` or `None` on failure rather than raising exceptions. Check return values for error detection:

```python
if not storage.put("", "value"):
    print("Failed to store - empty key")

value = storage.get("nonexistent")
if value is None:
    print("Key not found")
```

Detailed error information is available in the log file when logging is enabled.

## Complete Example

```python
from lmdb_storage import LMDBStorage, LMDBConfig

# Configure storage
config = LMDBConfig(
    db_path="products.lmdb",
    capacity_mb=50,
    compression_threshold=500
)

# Use with context manager
with LMDBStorage(config=config) as storage:
    # Store products
    storage.put("prod:001", "Laptop - 16GB RAM, 512GB SSD, $1299")
    storage.put("prod:002", "Monitor - 27-inch 4K, $599")
    storage.put("prod:003", "Keyboard - Mechanical RGB, $149")

    # Retrieve and display
    product = storage.get("prod:001")
    print(f"Product: {product}")

    # List all products
    print(f"\nTotal products: {storage.num_keys()}")
    for key in storage.get_keys():
        print(f"{key}: {storage.get(key)}")

    # Export to JSON
    storage.export_to_json("products_backup.json")

    # Statistics
    stats = storage.get_stats()
    print(f"\nDatabase stats: {stats}")
```

## Performance Considerations

- LMDB uses memory-mapped files for fast access
- Database size is preallocated (set `capacity_mb` appropriately)
- Use `get_keys(as_generator=True)` for large databases to reduce memory usage
- Compression trades CPU time for storage space (adjust `compression_threshold` based on your needs)
- Logging adds overhead; disable for performance-critical applications

## License

This code is provided as-is for use in your projects.
