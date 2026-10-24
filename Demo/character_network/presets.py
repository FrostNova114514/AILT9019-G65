"""Built-in alias tables for the two evaluation novels.

These tables are intentionally small and explicit. They are *test fixtures*, not
general knowledge: each one encodes the conventions of a single book (for example,
in *Pride and Prejudice* ``"Miss Bennet"`` conventionally means Jane, the eldest
daughter, while ``"Miss Elizabeth"`` means Elizabeth).
"""

from __future__ import annotations


from .names import normalize_aliases

PRIDE_AND_PREJUDICE: dict[str, str] = {
    # Bennet family
    "Elizabeth": "Elizabeth Bennet",
    "Eliza": "Elizabeth Bennet",
    "Lizzy": "Elizabeth Bennet",
    "Miss Eliza": "Elizabeth Bennet",
    "Miss Elizabeth": "Elizabeth Bennet",
    "Jane": "Jane Bennet",
    "Miss Bennet": "Jane Bennet",
    "Miss Jane": "Jane Bennet",
    "Mary": "Mary Bennet",
    "Kitty": "Kitty Bennet",
    "Catherine Bennet": "Kitty Bennet",
    "Lydia": "Lydia Bennet",
    "Mr. Bennet": "Mr Bennet",
    "Mr Bennet": "Mr Bennet",
    "Mrs. Bennet": "Mrs Bennet",
    "Mrs Bennet": "Mrs Bennet",
    # Bingley household
    "Bingley": "Charles Bingley",
    "Charles": "Charles Bingley",
    "Mr. Bingley": "Charles Bingley",
    "Caroline": "Caroline Bingley",
    "Miss Bingley": "Caroline Bingley",
    "Mrs. Hurst": "Mrs Hurst",
    "Mr. Hurst": "Mr Hurst",
    "Louisa": "Mrs Hurst",
    # Darcy household
    "Darcy": "Fitzwilliam Darcy",
    "Mr. Darcy": "Fitzwilliam Darcy",
    "Georgiana": "Georgiana Darcy",
    "Miss Darcy": "Georgiana Darcy",
    "Colonel Fitzwilliam": "Colonel Fitzwilliam",
    "Lady Catherine": "Lady Catherine de Bourgh",
    "Lady Catherine de Bourgh": "Lady Catherine de Bourgh",
    "Miss de Bourgh": "Anne de Bourgh",
    "Anne": "Anne de Bourgh",
    # Longbourn neighbours and Meryton
    "Charlotte": "Charlotte Lucas",
    "Charlotte Lucas": "Charlotte Lucas",
    "Maria Lucas": "Maria Lucas",
    "Lady Lucas": "Lady Lucas",
    "Sir William": "Sir William Lucas",
    "Sir William Lucas": "Sir William Lucas",
    "Mr. Collins": "William Collins",
    "Collins": "William Collins",
    "Mr. Wickham": "George Wickham",
    "Wickham": "George Wickham",
    "Mrs. Philips": "Mrs Philips",
    "Mr. Denny": "Mr Denny",
    "Mrs. Long": "Mrs Long",
    "Mrs. Gardiner": "Mrs Gardiner",
    "Mr. Gardiner": "Mr Gardiner",
    "Colonel Forster": "Colonel Forster",
    "Mrs. Forster": "Mrs Forster",
}

JANE_EYRE: dict[str, str] = {
    # Gateshead
    "Jane": "Jane Eyre",
    "Janet": "Jane Eyre",
    "Jane Eyre": "Jane Eyre",
    "Mrs. Reed": "Mrs Reed",
    "Mrs Reed": "Mrs Reed",
    "Mr. Reed": "Mr Reed",
    "John Reed": "John Reed",
    "Eliza Reed": "Eliza Reed",
    "Georgiana Reed": "Georgiana Reed",
    "Bessie": "Bessie",
    "Bessie Leaven": "Bessie",
    "Miss Abbot": "Miss Abbot",
    "Mr. Lloyd": "Mr Lloyd",
    # Lowood
    "Helen": "Helen Burns",
    "Helen Burns": "Helen Burns",
    "Brocklehurst": "Mr Brocklehurst",
    "Mr. Brocklehurst": "Mr Brocklehurst",
    "Miss Temple": "Miss Temple",
    "Temple": "Miss Temple",
    "Miss Scatcherd": "Miss Scatcherd",
    # Thornfield
    "Rochester": "Edward Rochester",
    "Mr. Rochester": "Edward Rochester",
    "Mr Rochester": "Edward Rochester",
    "Edward": "Edward Rochester",
    "Fairfax": "Mrs Fairfax",
    "Mrs. Fairfax": "Mrs Fairfax",
    "Adèle": "Adèle Varens",
    "Adele": "Adèle Varens",
    "Adèle Varens": "Adèle Varens",
    "Céline": "Céline Varens",
    "Blanche": "Blanche Ingram",
    "Ingram": "Blanche Ingram",
    "Miss Ingram": "Blanche Ingram",
    "Bertha": "Bertha Mason",
    "Bertha Antoinetta": "Bertha Mason",
    "Grace Poole": "Grace Poole",
    "Poole": "Grace Poole",
    "Mr. Mason": "Richard Mason",
    "Richard": "Richard Mason",
    "Mr. Briggs": "Mr Briggs",
    # Marsh End / Ferndean
    "St. John": "St John Rivers",
    "St John": "St John Rivers",
    "St. John Rivers": "St John Rivers",
    "Diana": "Diana Rivers",
    "Mary": "Mary Rivers",
    "Hannah": "Hannah",
    "Rosamond": "Rosamond Oliver",
}

PRESETS: dict[str, dict[str, str]] = {
    "Pride and Prejudice": PRIDE_AND_PREJUDICE,
    "Jane Eyre": JANE_EYRE,
}


def get_preset(label: str | None) -> dict[str, str]:
    """Return a normalized copy of a named preset, or an empty mapping for ``"None"``.

    Keys are case-folded and values are cleaned, which is what the analysis pipeline
    expects.
    """

    if not label:
        return {}
    return normalize_aliases(PRESETS.get(label, {}))


def get_preset_raw(label: str | None) -> dict[str, str]:
    """Return the preset with its original spelling, for display in an editor."""

    if not label:
        return {}
    return dict(PRESETS.get(label, {}))


def preset_labels() -> list[str]:
    """Return the display labels of the built-in presets."""

    return list(PRESETS)
