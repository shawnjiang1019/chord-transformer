SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]

STRUCTURE_TOKENS = [
    "<intro>", "<verse>", "<chorus>", "<bridge>",
    "<interlude>", "<solo>", "<instrumental>", "<outro>",
]

DECADE_TOKENS = [f"<{d}s>" for d in range(1890, 2030, 10)]

GENRE_TOKENS = [
    "<pop>", "<rock>", "<country>", "<alternative>", "<pop_rock>", "<punk>",
    "<metal>", "<rap>", "<soul>", "<jazz>", "<reggae>", "<electronic>",
    "<unk_genre>",
]
