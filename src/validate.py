import re


PLATE_PATTERNS = [
    r"^[A-Z]{3}[0-9]{3}[A-Z]?$",   # ex: RAB123A or RAB123
    r"^[A-Z]{2}[0-9]{3}[A-Z]{2}$", # optional second format
]


def is_valid_plate(text: str) -> bool:
    if not text:
        return False

    if len(text) < 5 or len(text) > 8:
        return False

    for pattern in PLATE_PATTERNS:
        if re.match(pattern, text):
            return True

    return False