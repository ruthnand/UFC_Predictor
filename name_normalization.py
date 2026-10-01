"""Shared normalization for fighter-name lookup keys and URL slugs."""

import unicodedata


def normalize_fighter_name(value):
    """Return an accent-insensitive, lowercase, whitespace-normalized name."""
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    without_accents = "".join(
        character
        for character in decomposed
        if not unicodedata.combining(character)
    )
    return " ".join(without_accents.lower().split())


def fighter_name_slug(*parts):
    """Build an ASCII UFC-style slug from one or more name components."""
    normalized = normalize_fighter_name(
        " ".join(str(part) for part in parts if part and str(part).strip())
    )
    return normalized.replace(" ", "-")
