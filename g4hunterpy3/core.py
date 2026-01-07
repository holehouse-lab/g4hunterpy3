"""
Core algorithms for G4Hunter scanning.

The G4Hunter approach assigns a per-base score based on runs of G or C:
- G runs contribute positive scores (1..4), capped at 4.
- C runs contribute negative scores (-1..-4), capped at -4.
- Other bases contribute 0.

A sliding window average over these base scores yields a "G4Hunter score"
per window. Windows whose absolute score exceeds a threshold are reported
as candidate G4-forming regions.

This module provides a small, testable API around those steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union, Dict

import numpy as np
import protfasta
from protfasta.protfasta_exceptions import ProtfastaException

@dataclass(frozen=True)
class WindowHit:
    """A single scoring window that passes the threshold.

    Parameters
    ----------
    start : int
        0-based start index of the window.
    end : int
        0-based end index (exclusive) of the window.
    score : float
        Mean G4Hunter score for the window (mean of base scores in window).
    """
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class Region:
    """A merged region formed by overlapping/adjacent WindowHits.

    Parameters
    ----------
    start : int
        0-based start index of the region.
    end : int
        0-based end index (exclusive) of the region.
    sequence : str
        Sequence slice for the region (original sequence[start:end]).
    length : int
        Region length, equal to end - start.
    score : float
        Region score. By default, this implementation uses the mean of
        the per-base scores across the region (rounded to 2 decimals),
        matching the original script's behavior.
    n_windows : int
        Number of window hits that were merged into this region.
    """
    start: int
    end: int
    sequence: str
    length: int
    score: float
    n_windows: int


def base_scores(seq: Union[str, "np.ndarray"]) -> np.ndarray:
    """Compute per-base G4Hunter scores for a sequence.

    This is a refactor of the original `BaseScore` routine. Runs of G (or g)
    contribute positive scores and runs of C (or c) contribute negative
    scores. The magnitude is capped at 4 and applied to every base in the run.

    Parameters
    ----------
    seq : str
        Input DNA sequence (may contain lower/upper case).

    Returns
    -------
    numpy.ndarray
        Array of shape (len(seq),) with integer scores in [-4, 4].

    Notes
    -----
    - Any character other than G/g/C/c receives score 0.
    - Runs longer than 4 are scored as 4 (or -4) for all positions in the run.
    """
    if not isinstance(seq, str):
        seq = "".join(seq.tolist())  # type: ignore[union-attr]
    n = len(seq)
    scores = np.zeros(n, dtype=np.int16)

    i = 0
    while i < n:
        b = seq[i]
        if b in ("G", "g", "C", "c"):
            # identify run
            j = i + 1
            while j < n and seq[j] == b:
                j += 1
            run_len = j - i
            capped = 4 if run_len >= 4 else run_len
            val = capped if b in ("G", "g") else -capped
            scores[i:j] = val
            i = j
        else:
            i += 1

    # check nothing funky happened...
    assert scores.size == n
    
    return scores


def window_mean_scores(scores: np.ndarray, window_size: int) -> np.ndarray:
    """Compute sliding-window mean scores.

    Parameters
    ----------
    scores : numpy.ndarray
        Per-base score array (output of `base_scores`).
    window_size : int
        Window length in bases (k in the original script).

    Returns
    -------
    numpy.ndarray
        Array of shape (len(scores) - window_size + 1,) containing the mean
        score for each window starting at i.

    Raises
    ------
    ValueError
        If window_size is < 1 or larger than the sequence length.
    """
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if window_size > scores.size:
        raise ValueError("window_size cannot exceed sequence length")

    # Fast sliding mean using convolution (O(n)).
    kernel = np.ones(window_size, dtype=np.float64) / float(window_size)
    smoothed =  np.convolve(scores.astype(np.float64), kernel, mode="valid")
    return smoothed

def find_window_hits(
    window_scores: np.ndarray,
    window_size: int,
    threshold: float,
) -> List[WindowHit]:
    """Identify scoring windows whose absolute score passes a threshold.

    Parameters
    ----------
    window_scores : numpy.ndarray
        Sliding-window mean scores (output of `window_mean_scores`).
    window_size : int
        Window length in bases.
    threshold : float
        Threshold applied to absolute window score.

    Returns
    -------
    list of WindowHit
        Each hit corresponds to one window start position i. WindowHit uses
        0-based indexing with end exclusive.
    """
    hits: List[WindowHit] = []
    if window_scores.size == 0:
        return hits

    mask = np.abs(window_scores) >= float(threshold)
    hit_starts = np.nonzero(mask)[0]
    for s in hit_starts.tolist():
        hits.append(WindowHit(start=int(s), end=int(s + window_size), score=float(window_scores[s])))
    return hits


def merge_overlapping_windows(
    hits: Sequence[WindowHit],
    seq: str,
    base_score_array: Optional[np.ndarray] = None,
) -> List[Region]:
    """Merge overlapping/adjacent window hits into regions.

    The original script merged windows when their start positions were
    consecutive (difference of 1). This produces regions that are the union
    of a run of overlapping windows.

    Parameters
    ----------
    hits : sequence of WindowHit
        Window hits, ideally sorted by start.
    seq : str
        Original sequence the hits are defined on.
    base_score_array : numpy.ndarray, optional
        Per-base score array for `seq`. If not supplied, it will be computed.

    Returns
    -------
    list of Region
        Merged regions with region score computed as mean per-base score
        across the region (rounded to 2 decimals), consistent with the
        original script output.

    Notes
    -----
    - If `hits` is empty, returns an empty list.
    - Windows are treated as overlapping if the next start is <= current_end - 1.
      For consecutive starts and fixed window size, this matches the original.
    """
    if not hits:
        return []
    if base_score_array is None:
        base_score_array = base_scores(seq)

    # Ensure sorted
    hits_sorted = sorted(hits, key=lambda h: h.start)

    regions: List[Region] = []
    cur_start = hits_sorted[0].start
    cur_end = hits_sorted[0].end
    n_windows = 1

    for h in hits_sorted[1:]:
        if h.start <= cur_end - 1:
            # Overlapping (or adjacent by one base for consecutive starts)
            cur_end = max(cur_end, h.end)
            n_windows += 1
        else:
            region_seq = seq[cur_start:cur_end]
            region_score = float(np.round(base_score_array[cur_start:cur_end].mean(), 2))
            regions.append(
                Region(
                    start=cur_start,
                    end=cur_end,
                    sequence=region_seq,
                    length=cur_end - cur_start,
                    score=region_score,
                    n_windows=n_windows,
                )
            )
            cur_start, cur_end, n_windows = h.start, h.end, 1

    # Final region
    region_seq = seq[cur_start:cur_end]
    region_score = float(np.round(base_score_array[cur_start:cur_end].mean(), 2))
    regions.append(
        Region(
            start=cur_start,
            end=cur_end,
            sequence=region_seq,
            length=cur_end - cur_start,
            score=region_score,
            n_windows=n_windows,
        )
    )
    return regions


def scan_sequence(
    seq: str,
    window_size: int = 25,
    threshold: float = 1.5,
) -> Tuple[np.ndarray, List[WindowHit], List[Region]]:
    """Run G4Hunter scoring on a single DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence to scan.
    window_size : int, default=25
        Sliding window length in bases.
    threshold : float, default=1.5
        Absolute score threshold for calling windows as hits.

    Returns
    -------
    window_scores : numpy.ndarray
        Sliding-window mean score array.
    hits : list of WindowHit
        Per-window hits whose absolute score >= threshold.
    regions : list of Region
        Merged hit regions.

    Examples
    --------
    >>> ws, hits, regions = scan_sequence("GGGGTTTTGGGG", window_size=4, threshold=1.0)
    >>> len(hits) > 0
    True
    """
    bs = base_scores(seq)    
    ws = window_mean_scores(bs, window_size)
    hits = find_window_hits(ws, window_size, threshold)
    regions = merge_overlapping_windows(hits, seq, base_score_array=bs)
    return ws, hits, regions


def _iter_fasta_records(path: Union[str, Path],
                        strict: bool = True) -> Iterator[Tuple[str, str]]:
    """
    Yield (header, sequence) from a FASTA file.

    This function uses protfasta.read_fasta() to parse FASTA files. Note

    Parameters
    ----------
    path : str or pathlib.Path
        Path to FASTA file.
    strict : bool, default=True
        If True, raise an error for invalid nucleotide characters.

    Yields
    ------
    tuple of (str, str)
        Record id and sequence (no whitespace).
    """
    # Use protfasta to read the file, returning a list of [header, sequence] pairs
    
    records = protfasta.read_fasta(str(path), return_list=True, invalid_sequence_action='ignore')
    
    if strict:
        valid_nucleotides = set("ACGTU")
    else:
        valid_nucleotides = set("ACGTUN-")

    # validate for nucleotides 
    for header, sequence in records:
        invalid_chars = set(sequence.upper()) - valid_nucleotides
        if invalid_chars:
            raise ValueError(f"Invalid nucleotide characters found in record {header}: {invalid_chars}")
    
    for header, sequence in records:
        yield str(header), str(sequence)


def scan_fasta(
    fasta_path: Union[str, Path],
    window_size: int = 25,
    threshold: float = 1.5,
) -> Dict[str, Tuple[np.ndarray, List[WindowHit], List[Region]]]:
    """Scan every record in a FASTA file.

    Parameters
    ----------
    fasta_path : str or pathlib.Path
        Path to FASTA file.
    window_size : int, default=25
        Sliding window length in bases.
    threshold : float, default=1.5
        Absolute score threshold for calling windows as hits.

    Returns
    -------
    dict
        Mapping from record id to (window_scores, hits, regions).
    """
    results: Dict[str, Tuple[np.ndarray, List[WindowHit], List[Region]]] = {}
    for header, seq in _iter_fasta_records(fasta_path):
        results[header] = scan_sequence(seq, window_size=window_size, threshold=threshold)
    return results
