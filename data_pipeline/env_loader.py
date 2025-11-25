"""
Load environment variables from .env file
"""

import os
from pathlib import Path

def load_env_file(env_file='.env'):
    """Load environment variables from .env file"""
    env_path = Path(__file__).parent / env_file
    
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                # Strip whitespace and ignore comments/blank lines
                raw = line.strip()
                if not raw or raw.startswith('#') or '=' not in raw:
                    continue

                # Split into key and value at first '=' and strip spaces
                key, value = raw.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Remove surrounding quotes from value if present
                if (value.startswith('"') and value.endswith('"')) or (
                    value.startswith("'") and value.endswith("'")
                ):
                    value = value[1:-1]

                # Remove inline comments after the value (e.g. value # comment)
                if '#' in value:
                    # only strip if '#' appears after a space or at start of comment
                    # keep hashes that are part of the value (rare)
                    val_parts = value.split('#')
                    # take the first part as the actual value
                    value = val_parts[0].strip()

                # Finally set the environment variable
                if key:
                    os.environ[key] = value

# Load environment variables when module is imported
load_env_file()
