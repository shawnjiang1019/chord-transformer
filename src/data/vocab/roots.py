ROOT_TOKENS = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

# Maps any spelling of a pitch to the canonical sharp form
ENHARMONIC = {
    "Cb": "B",  "Db": "Cs", "Eb": "Ds", "Fb": "E",
    "Gb": "Fs", "Ab": "Gs", "Bb": "As",
    "Bs": "C",  "Es": "F",  # rare but possible
}
