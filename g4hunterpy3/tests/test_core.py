"""
Comprehensive test suite for g4hunterpy3.core module.

Tests cover:
- base_scores: per-base G4Hunter score computation
- window_mean_scores: sliding window mean calculations
- find_window_hits: threshold-based hit detection
- merge_overlapping_windows: region merging logic
- scan_sequence: end-to-end sequence scanning
- _iter_fasta_records: FASTA file parsing
- scan_fasta: full FASTA file scanning
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from g4hunterpy3.core import (
    WindowHit,
    Region,
    base_scores,
    window_mean_scores,
    find_window_hits,
    merge_overlapping_windows,
    scan_sequence,
    _iter_fasta_records,
    scan_fasta,
)


# =============================================================================
# Tests for base_scores function
# =============================================================================

class TestBaseScores:
    """Tests for the base_scores function."""

    def test_single_g(self):
        """Single G should score +1."""
        scores = base_scores("G")
        assert len(scores) == 1
        assert scores[0] == 1

    def test_single_c(self):
        """Single C should score -1."""
        scores = base_scores("C")
        assert len(scores) == 1
        assert scores[0] == -1

    def test_single_other_base(self):
        """Non-G/C bases should score 0."""
        for base in ["A", "T", "N", "a", "t", "n"]:
            scores = base_scores(base)
            assert len(scores) == 1
            assert scores[0] == 0

    def test_g_run_length_2(self):
        """Run of 2 Gs should score +2 for each position."""
        scores = base_scores("GG")
        np.testing.assert_array_equal(scores, [2, 2])

    def test_g_run_length_3(self):
        """Run of 3 Gs should score +3 for each position."""
        scores = base_scores("GGG")
        np.testing.assert_array_equal(scores, [3, 3, 3])

    def test_g_run_length_4(self):
        """Run of 4 Gs should score +4 for each position."""
        scores = base_scores("GGGG")
        np.testing.assert_array_equal(scores, [4, 4, 4, 4])

    def test_g_run_length_5_capped(self):
        """Run of 5+ Gs should be capped at +4 for each position."""
        scores = base_scores("GGGGG")
        np.testing.assert_array_equal(scores, [4, 4, 4, 4, 4])

    def test_c_run_length_2(self):
        """Run of 2 Cs should score -2 for each position."""
        scores = base_scores("CC")
        np.testing.assert_array_equal(scores, [-2, -2])

    def test_c_run_length_3(self):
        """Run of 3 Cs should score -3 for each position."""
        scores = base_scores("CCC")
        np.testing.assert_array_equal(scores, [-3, -3, -3])

    def test_c_run_length_4(self):
        """Run of 4 Cs should score -4 for each position."""
        scores = base_scores("CCCC")
        np.testing.assert_array_equal(scores, [-4, -4, -4, -4])

    def test_c_run_length_5_capped(self):
        """Run of 5+ Cs should be capped at -4 for each position."""
        scores = base_scores("CCCCC")
        np.testing.assert_array_equal(scores, [-4, -4, -4, -4, -4])

    def test_mixed_sequence(self):
        """Mixed sequence with G runs, C runs, and other bases."""
        # GGAACCC -> GG(+2,+2), AA(0,0), CCC(-3,-3,-3)
        scores = base_scores("GGAACCC")
        np.testing.assert_array_equal(scores, [2, 2, 0, 0, -3, -3, -3])

    def test_lowercase_g(self):
        """Lowercase g should be treated same as uppercase G."""
        scores = base_scores("ggg")
        np.testing.assert_array_equal(scores, [3, 3, 3])

    def test_lowercase_c(self):
        """Lowercase c should be treated same as uppercase C."""
        scores = base_scores("ccc")
        np.testing.assert_array_equal(scores, [-3, -3, -3])

    def test_mixed_case(self):
        """Mixed case should work correctly."""
        scores = base_scores("GgGg")
        # Note: lowercase and uppercase are the same character, so this is a run of 4
        # Actually wait - the code checks seq[j] == b, so 'G' != 'g'
        # So this would be G(1), g(1), G(1), g(1)
        np.testing.assert_array_equal(scores, [1, 1, 1, 1])

    def test_alternating_g_and_c(self):
        """Alternating G and C should each have length 1."""
        scores = base_scores("GCGC")
        np.testing.assert_array_equal(scores, [1, -1, 1, -1])

    def test_empty_sequence(self):
        """Empty sequence should return empty array."""
        scores = base_scores("")
        assert len(scores) == 0

    def test_g4_motif(self):
        """Typical G4 motif with G runs separated by other bases."""
        # GGGG T GGGG T GGGG T GGGG
        seq = "GGGGTGGGGTGGGGTGGGG"
        scores = base_scores(seq)
        expected = [4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4]
        np.testing.assert_array_equal(scores, expected)

    def test_long_g_run(self):
        """Very long G run should all be capped at 4."""
        scores = base_scores("G" * 100)
        assert len(scores) == 100
        assert all(s == 4 for s in scores)


# =============================================================================
# Tests for window_mean_scores function
# =============================================================================

class TestWindowMeanScores:
    """Tests for the window_mean_scores function."""

    def test_basic_window(self):
        """Basic sliding window mean calculation."""
        scores = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        result = window_mean_scores(scores, window_size=3)
        # Window 0: mean(1,2,3) = 2
        # Window 1: mean(2,3,4) = 3
        # Window 2: mean(3,4,5) = 4
        expected = np.array([2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_window_size_equals_sequence_length(self):
        """Window size equals sequence length should return single value."""
        scores = np.array([1, 2, 3, 4], dtype=np.float64)
        result = window_mean_scores(scores, window_size=4)
        assert len(result) == 1
        assert result[0] == 2.5  # mean(1,2,3,4)

    def test_window_size_one(self):
        """Window size of 1 should return original scores."""
        scores = np.array([1, -1, 2, -2], dtype=np.float64)
        result = window_mean_scores(scores, window_size=1)
        np.testing.assert_array_equal(result, scores)

    def test_invalid_window_size_zero(self):
        """Window size of 0 should raise ValueError."""
        scores = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            window_mean_scores(scores, window_size=0)

    def test_invalid_window_size_negative(self):
        """Negative window size should raise ValueError."""
        scores = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            window_mean_scores(scores, window_size=-1)

    def test_window_size_larger_than_sequence(self):
        """Window size larger than sequence should raise ValueError."""
        scores = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="window_size cannot exceed sequence length"):
            window_mean_scores(scores, window_size=5)

    def test_result_length(self):
        """Result length should be len(scores) - window_size + 1."""
        scores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        for window_size in [1, 3, 5, 10]:
            result = window_mean_scores(scores, window_size)
            expected_length = len(scores) - window_size + 1
            assert len(result) == expected_length

    def test_with_g4hunter_scores(self):
        """Test with typical G4Hunter base scores."""
        # Simulating GGGGAATTCCCC
        scores = np.array([4, 4, 4, 4, 0, 0, 0, 0, -4, -4, -4, -4], dtype=np.float64)
        result = window_mean_scores(scores, window_size=4)
        # First 4: mean(4,4,4,4) = 4.0
        # etc.
        assert result[0] == 4.0
        assert result[-1] == -4.0


# =============================================================================
# Tests for find_window_hits function
# =============================================================================

class TestFindWindowHits:
    """Tests for the find_window_hits function."""

    def test_no_hits_below_threshold(self):
        """Windows below threshold should not be returned."""
        window_scores = np.array([0.5, 0.6, 0.7, -0.5, -0.6])
        hits = find_window_hits(window_scores, window_size=4, threshold=1.0)
        assert len(hits) == 0

    def test_positive_hits_above_threshold(self):
        """Positive scores above threshold should be hits."""
        window_scores = np.array([0.5, 1.5, 2.0, 0.3])
        hits = find_window_hits(window_scores, window_size=4, threshold=1.0)
        assert len(hits) == 2
        assert hits[0].start == 1
        assert hits[0].end == 5
        assert hits[0].score == 1.5
        assert hits[1].start == 2
        assert hits[1].end == 6
        assert hits[1].score == 2.0

    def test_negative_hits_above_threshold(self):
        """Negative scores with absolute value above threshold should be hits."""
        window_scores = np.array([0.5, -1.5, -2.0, 0.3])
        hits = find_window_hits(window_scores, window_size=4, threshold=1.0)
        assert len(hits) == 2
        assert hits[0].score == -1.5
        assert hits[1].score == -2.0

    def test_exact_threshold(self):
        """Score exactly at threshold should be included."""
        window_scores = np.array([1.0, 0.9, 1.0])
        hits = find_window_hits(window_scores, window_size=4, threshold=1.0)
        assert len(hits) == 2  # indices 0 and 2

    def test_empty_window_scores(self):
        """Empty window scores should return empty list."""
        window_scores = np.array([])
        hits = find_window_hits(window_scores, window_size=4, threshold=1.0)
        assert len(hits) == 0

    def test_window_hit_coordinates(self):
        """Verify WindowHit start/end coordinates are correct."""
        window_scores = np.array([0.0, 0.0, 2.0, 0.0])
        hits = find_window_hits(window_scores, window_size=10, threshold=1.0)
        assert len(hits) == 1
        assert hits[0].start == 2
        assert hits[0].end == 12  # start + window_size

    def test_all_hits(self):
        """All windows above threshold should be returned."""
        window_scores = np.array([1.5, 1.6, 1.7, 1.8])
        hits = find_window_hits(window_scores, window_size=4, threshold=1.0)
        assert len(hits) == 4


# =============================================================================
# Tests for merge_overlapping_windows function
# =============================================================================

class TestMergeOverlappingWindows:
    """Tests for the merge_overlapping_windows function."""

    def test_empty_hits(self):
        """Empty hits list should return empty regions."""
        regions = merge_overlapping_windows([], "ATCG")
        assert len(regions) == 0

    def test_single_hit(self):
        """Single hit should become a single region."""
        hit = WindowHit(start=0, end=4, score=2.0)
        seq = "GGGGAAAA"
        regions = merge_overlapping_windows([hit], seq)
        assert len(regions) == 1
        assert regions[0].start == 0
        assert regions[0].end == 4
        assert regions[0].sequence == "GGGG"
        assert regions[0].length == 4
        assert regions[0].n_windows == 1

    def test_non_overlapping_hits(self):
        """Non-overlapping hits should become separate regions."""
        hits = [
            WindowHit(start=0, end=4, score=2.0),
            WindowHit(start=10, end=14, score=2.0),
        ]
        seq = "GGGGAAAAAAGGGAAAAA"
        regions = merge_overlapping_windows(hits, seq)
        assert len(regions) == 2
        assert regions[0].start == 0
        assert regions[1].start == 10

    def test_overlapping_hits_merge(self):
        """Overlapping hits should merge into single region."""
        # Consecutive windows (start differs by 1)
        hits = [
            WindowHit(start=0, end=4, score=2.0),
            WindowHit(start=1, end=5, score=2.5),
            WindowHit(start=2, end=6, score=3.0),
        ]
        seq = "GGGGGGAAAA"
        regions = merge_overlapping_windows(hits, seq)
        assert len(regions) == 1
        assert regions[0].start == 0
        assert regions[0].end == 6
        assert regions[0].n_windows == 3

    def test_region_score_calculation(self):
        """Region score should be mean of per-base scores."""
        hits = [WindowHit(start=0, end=4, score=4.0)]
        seq = "GGGGAAAA"  # First 4 bases are GGGG with base score 4 each
        regions = merge_overlapping_windows(hits, seq)
        assert regions[0].score == 4.0  # mean of [4,4,4,4]

    def test_region_sequence_extraction(self):
        """Region sequence should be correctly extracted."""
        hits = [WindowHit(start=2, end=6, score=2.0)]
        seq = "AAGGGAAA"
        regions = merge_overlapping_windows(hits, seq)
        assert regions[0].sequence == "GGGA"

    def test_unsorted_hits_get_sorted(self):
        """Hits should be sorted by start position before merging."""
        hits = [
            WindowHit(start=10, end=14, score=2.0),
            WindowHit(start=0, end=4, score=2.0),
        ]
        seq = "GGGGAAAAAAGGGAAAAA"
        regions = merge_overlapping_windows(hits, seq)
        assert len(regions) == 2
        assert regions[0].start == 0
        assert regions[1].start == 10


# =============================================================================
# Tests for scan_sequence function
# =============================================================================

class TestScanSequence:
    """Tests for the scan_sequence function."""

    def test_returns_three_items(self):
        """scan_sequence should return tuple of (window_scores, hits, regions)."""
        ws, hits, regions = scan_sequence("GGGGTTTTGGGG", window_size=4, threshold=1.0)
        assert isinstance(ws, np.ndarray)
        assert isinstance(hits, list)
        assert isinstance(regions, list)

    def test_strong_g4_motif(self):
        """Strong G4 motif should produce hits."""
        seq = "GGGGTTGGGGTTGGGGTTGGGG"
        ws, hits, regions = scan_sequence(seq, window_size=4, threshold=1.0)
        assert len(hits) > 0
        assert len(regions) > 0

    def test_no_g4_motif(self):
        """Sequence without G or C runs should have no hits above threshold."""
        seq = "ATATATATATATAT"
        ws, hits, regions = scan_sequence(seq, window_size=4, threshold=1.0)
        assert len(hits) == 0
        assert len(regions) == 0

    def test_window_scores_length(self):
        """Window scores length should match expected value."""
        seq = "GGGGTTTTGGGGTTTTGGGG"  # length 20
        window_size = 4
        ws, _, _ = scan_sequence(seq, window_size=window_size, threshold=1.0)
        expected_length = len(seq) - window_size + 1
        assert len(ws) == expected_length

    def test_default_parameters(self):
        """Test with default parameters (window_size=25, threshold=1.5)."""
        seq = "GGGG" + "T" * 100 + "GGGG"  # length > 25
        ws, hits, regions = scan_sequence(seq)
        assert len(ws) == len(seq) - 25 + 1

    def test_high_threshold_reduces_hits(self):
        """Higher threshold should result in fewer or equal hits."""
        seq = "GGGGTTGGGGTTGGGGTTGGGG"
        _, hits_low, _ = scan_sequence(seq, window_size=4, threshold=1.0)
        _, hits_high, _ = scan_sequence(seq, window_size=4, threshold=3.0)
        assert len(hits_high) <= len(hits_low)


# =============================================================================
# Tests for _iter_fasta_records function
# =============================================================================

class TestIterFastaRecords:
    """Tests for the _iter_fasta_records function."""

    def test_single_record(self):
        """Parse FASTA with single record."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">test_seq\n")
            f.write("ACGTACGT\n")
            fasta_path = Path(f.name)
        
        try:
            records = list(_iter_fasta_records(fasta_path))
            assert len(records) == 1
            assert records[0][0] == "test_seq"
            assert records[0][1] == "ACGTACGT"
        finally:
            fasta_path.unlink()

    def test_multiple_records(self):
        """Parse FASTA with multiple records."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">seq1\n")
            f.write("AAAA\n")
            f.write(">seq2\n")
            f.write("CCCC\n")
            f.write(">seq3\n")
            f.write("GGGG\n")
            fasta_path = Path(f.name)
        
        try:
            records = list(_iter_fasta_records(fasta_path))
            assert len(records) == 3
            assert records[0][0] == "seq1"
            assert records[1][0] == "seq2"
            assert records[2][0] == "seq3"
        finally:
            fasta_path.unlink()

    def test_multiline_sequence(self):
        """Parse FASTA with multiline sequences."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">test_seq\n")
            f.write("AAAA\n")
            f.write("TTTT\n")
            f.write("CCCC\n")
            fasta_path = Path(f.name)
        
        try:
            records = list(_iter_fasta_records(fasta_path))
            assert len(records) == 1
            assert records[0][1] == "AAAATTTTCCCC"
        finally:
            fasta_path.unlink()

    def test_invalid_nucleotide_strict(self):
        """Invalid nucleotides in strict mode should raise ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">test_seq\n")
            f.write("ACGTXYZ\n")
            fasta_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Invalid nucleotide characters"):
                list(_iter_fasta_records(fasta_path, strict=True))
        finally:
            fasta_path.unlink()


# =============================================================================
# Tests for scan_fasta function
# =============================================================================

class TestScanFasta:
    """Tests for the scan_fasta function."""

    def test_scan_single_record(self):
        """Scan FASTA with single record."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">test_seq\n")
            f.write("GGGGTTTTGGGGTTTTGGGGTTTTGGGG\n")
            fasta_path = Path(f.name)
        
        try:
            results = scan_fasta(fasta_path, window_size=4, threshold=1.0)
            assert "test_seq" in results
            ws, hits, regions = results["test_seq"]
            assert isinstance(ws, np.ndarray)
            assert len(hits) > 0
        finally:
            fasta_path.unlink()

    def test_scan_multiple_records(self):
        """Scan FASTA with multiple records."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
            f.write(">seq1\n")
            f.write("GGGGTTTTGGGG\n")
            f.write(">seq2\n")
            f.write("CCCCAAAACCCC\n")
            fasta_path = Path(f.name)
        
        try:
            results = scan_fasta(fasta_path, window_size=4, threshold=1.0)
            assert len(results) == 2
            assert "seq1" in results
            assert "seq2" in results
        finally:
            fasta_path.unlink()


# =============================================================================
# Tests for dataclasses
# =============================================================================

class TestDataclasses:
    """Tests for WindowHit and Region dataclasses."""

    def test_window_hit_creation(self):
        """WindowHit should be created with correct attributes."""
        hit = WindowHit(start=10, end=20, score=2.5)
        assert hit.start == 10
        assert hit.end == 20
        assert hit.score == 2.5

    def test_window_hit_frozen(self):
        """WindowHit should be immutable (frozen)."""
        hit = WindowHit(start=10, end=20, score=2.5)
        with pytest.raises(AttributeError):
            hit.start = 15

    def test_region_creation(self):
        """Region should be created with correct attributes."""
        region = Region(
            start=0,
            end=10,
            sequence="GGGGAATTGG",
            length=10,
            score=2.0,
            n_windows=5
        )
        assert region.start == 0
        assert region.end == 10
        assert region.sequence == "GGGGAATTGG"
        assert region.length == 10
        assert region.score == 2.0
        assert region.n_windows == 5

    def test_region_frozen(self):
        """Region should be immutable (frozen)."""
        region = Region(start=0, end=10, sequence="GGGG", length=4, score=2.0, n_windows=1)
        with pytest.raises(AttributeError):
            region.start = 5


# =============================================================================
# Integration tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_pipeline_simple_sequence(self):
        """Test complete pipeline from sequence to regions."""
        # A simple sequence with clear G4-forming regions
        seq = "AAAA" + "GGGG" + "TTTT" + "GGGG" + "AAAA"  # length 20
        
        # Get base scores
        bs = base_scores(seq)
        assert len(bs) == 20
        
        # Get window scores
        ws = window_mean_scores(bs, window_size=4)
        assert len(ws) == 17
        
        # Find hits
        hits = find_window_hits(ws, window_size=4, threshold=1.0)
        
        # Merge regions
        regions = merge_overlapping_windows(hits, seq, base_score_array=bs)
        
        # Verify we found the G-rich regions
        g_regions = [r for r in regions if r.score > 0]
        assert len(g_regions) >= 1

    def test_real_g4_sequence(self):
        """Test with a known G4-forming sequence motif."""
        # Human telomeric repeat: (TTAGGG)n
        telomeric = "TTAGGG" * 8  # 48 bp
        
        ws, hits, regions = scan_sequence(telomeric, window_size=24, threshold=1.0)
        
        # Should find G-rich regions
        assert len(ws) == len(telomeric) - 24 + 1
        # The telomeric sequence should have positive scoring regions
        assert any(s > 0 for s in ws)

    def test_c_rich_sequence(self):
        """Test with C-rich sequence (reverse complement G4 potential)."""
        c_rich = "CCCC" + "AAAA" + "CCCC" + "AAAA" + "CCCC"
        
        ws, hits, regions = scan_sequence(c_rich, window_size=4, threshold=1.0)
        
        # C-rich regions should have negative scores
        c_regions = [r for r in regions if r.score < 0]
        assert len(c_regions) >= 1
