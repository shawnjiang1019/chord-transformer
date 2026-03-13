"""Tests for chromatic transposition augmentation."""

from src.data.augmentation import transpose_chord, augment_sequence


class TestTransposeChord:
    """Tests for transpose_chord."""

    def test_no_transposition(self):
        assert transpose_chord("C", 0) == "C"
        assert transpose_chord("Amin", 0) == "Amin"

    def test_basic_transposition(self):
        assert transpose_chord("C", 2) == "D"
        assert transpose_chord("C", 7) == "G"

    def test_wrap_around(self):
        assert transpose_chord("A", 3) == "C"
        assert transpose_chord("B", 1) == "C"

    def test_suffix_preserved(self):
        assert transpose_chord("Amin", 3) == "Cmin"
        assert transpose_chord("Fs7", 1) == "G7"
        assert transpose_chord("Cadd9", 2) == "Dadd9"
        assert transpose_chord("Emin7", 5) == "Amin7"

    def test_enharmonic_normalization(self):
        # Bb -> As internally, then transpose
        assert transpose_chord("Bb", 0) == "As"
        assert transpose_chord("Bb", 2) == "C"
        assert transpose_chord("Dbmin", 0) == "Csmin"

    def test_sharp_root(self):
        assert transpose_chord("Fs7", 0) == "Fs7"
        assert transpose_chord("Cs", 2) == "Ds"

    def test_full_chromatic_cycle(self):
        # Transposing by 12 should return to the same chord
        assert transpose_chord("Amin7", 12) == "Amin7"

    def test_section_markers_unchanged(self):
        assert transpose_chord("<intro_1>", 5) == "<intro_1>"
        assert transpose_chord("<verse_2>", 7) == "<verse_2>"
        assert transpose_chord("<chorus_1>", 11) == "<chorus_1>"

    def test_slash_chord_suffix(self):
        # Bass note is part of the suffix, preserved as-is
        assert transpose_chord("C/G", 2) == "D/G"

    def test_unrecognized_input(self):
        # Non-chord strings returned unchanged
        assert transpose_chord("???", 5) == "???"


class TestAugmentSequence:
    """Tests for augment_sequence."""

    def test_returns_12_transpositions(self):
        seq = ["C", "F", "G"]
        result = augment_sequence(seq)
        assert len(result) == 12

    def test_first_is_original(self):
        seq = ["C", "Amin", "F", "G"]
        result = augment_sequence(seq)
        assert result[0] == seq

    def test_transposition_correctness(self):
        seq = ["C", "F", "G"]
        result = augment_sequence(seq)
        # +2 semitones: C->D, F->G, G->A
        assert result[2] == ["D", "G", "A"]
        # +7 semitones: C->G, F->C, G->D
        assert result[7] == ["G", "C", "D"]

    def test_section_markers_preserved(self):
        seq = ["<verse_1>", "C", "F", "G"]
        result = augment_sequence(seq)
        for transposition in result:
            assert transposition[0] == "<verse_1>"

    def test_all_transpositions_same_length(self):
        seq = ["C", "Amin", "F", "G7", "E7"]
        result = augment_sequence(seq)
        for transposition in result:
            assert len(transposition) == len(seq)
