QUALITY_TOKENS = [
    "maj", "min", "dom7", "maj7", "min7",
    "dim", "dim7", "aug", "sus",
    "dom9", "maj9", "min9",
    "maj6", "min6",
    "other",
]

# Maps raw quality suffix from the chord string -> family token
QUALITY_MAP: dict[str, str] = {
    # Major
    "": "maj", "maj": "maj", "M": "maj",
    # Minor
    "min": "min", "m": "min",
    # Dominant 7
    "7": "dom7", "9": "dom7", "11": "dom7", "13": "dom7",
    "7b9": "dom7", "7s9": "dom7", "7b5": "dom7", "7s11": "dom7",
    # Major 7
    "maj7": "maj7", "maj9": "maj7", "M7": "maj7",
    # Minor 7
    "min7": "min7", "m7": "min7", "min9": "min7", "m9": "min7",
    # Diminished
    "dim": "dim", "o": "dim",
    "dim7": "dim7", "o7": "dim7", "hdim7": "dim7",
    # Augmented
    "aug": "aug", "+": "aug",
    # Suspended
    "sus2": "sus", "sus4": "sus", "sus": "sus",
    # 6th chords
    "6": "maj6", "maj6": "maj6",
    "min6": "min6", "m6": "min6",
    # Extended minor
    "min11": "min7", "min13": "min7",
}

# Maps quality family -> suffix used when decoding back to a chord string
QUALITY_TO_SUFFIX: dict[str, str] = {
    "maj": "", "min": "min", "dom7": "7", "maj7": "maj7", "min7": "min7",
    "dim": "dim", "dim7": "dim7", "aug": "aug", "sus": "sus4",
    "dom9": "9", "maj9": "maj9", "min9": "min9",
    "maj6": "6", "min6": "min6",
    "other": "?",
}
