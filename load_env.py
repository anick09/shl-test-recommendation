"""
Load environment variables from .env file
"""
import os
from pathlib import Path

def load_env():
    env_file = Path('.env')
    if env_file.exists():
        # Read with UTF-8 encoding and handle BOM
        with open(env_file, 'r', encoding='utf-8-sig') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and value:
                        os.environ[key] = value
                        print(f"Loaded: {key}")

if __name__ == "__main__":
    load_env()

