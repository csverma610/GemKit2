#!/usr/bin/env python3
"""
LMDB CLI Tool - Command Line Interface for viewing LMDB database contents
"""

import argparse
import sys
import os
from lmdb_storage import LMDBStorage

def main():
    if len(sys.argv) < 3:
        print("Usage: python lmdb2json.py <database_path> <json_path>")
        print("Example: python lmdb2json.py geminiqa.lmdb geminiqa.json")
        sys.exit(1)
    
    db_path = sys.argv[1]
    db = LMDBStorage(db_path)
    keys = db.get_keys()
    for key in keys:
        print(key)
    json_path = sys.argv[2]
    db.export_to_json(json_path)

if __name__ == "__main__":
    main()
