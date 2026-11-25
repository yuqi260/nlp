"""Load the project's two API settings without executing configuration text."""
import os
from pathlib import Path


def load_local_config(path: Path):
    if not path.is_file():
        return
    allowed = {'WEREWOLF_API_KEY', 'WEREWOLF_API_BASE_URL'}
    for line_number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, separator, value = line.partition('=')
        name, value = name.strip(), value.strip()
        if not separator or name not in allowed:
            raise ValueError(f'invalid configuration line {line_number}')
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
            value = value[1:-1]
        if value and not os.environ.get(name):
            os.environ[name] = value
