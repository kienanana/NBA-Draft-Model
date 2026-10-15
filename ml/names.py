"""Stable player-name matching between prospect pools and draft results."""

import re
import unicodedata


ALIASES = {
    "jabari smith jr": "jabari smith",
    "alex sarr": "alexandre sarr",
}


def player_key(name: str) -> str:
    """Normalize spelling and the known historical draft/pool aliases."""
    ascii_name = unicodedata.normalize("NFKD", str(name))
    ascii_name = "".join(char for char in ascii_name if not unicodedata.combining(char))
    key = " ".join(re.findall(r"[a-z0-9]+", ascii_name.casefold()))
    return ALIASES.get(key, key)
