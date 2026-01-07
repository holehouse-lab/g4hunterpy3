"""
Command-line interface for the G4Hunter refactor.

This is a Python 3.8-friendly replacement for the original Python 2 script.
It reads a FASTA file, scores each record with G4Hunter, and writes:

1) A per-window hit file: <record>-W<k>-S<threshold>.txt
2) A merged region file: <record>-Merged.txt
3) Optional PDF plot(s) of the sliding-window scores

Run:
    python -m g4hunter.cli -i input.fasta -o outdir -w 25 -s 1.5 --plot
"""

from __future__ import annotations

import argparse
from pathlib import Path

from g4hunterpy3.core import _iter_fasta_records, scan_sequence
from g4hunterpy3.plotting import simple_plot, complex_plot
import sys


# ................................................................................................
#
def _write_window_hits(path: Path, header: str, seq: str, hits: list) -> None:
    """    
    Write per-window hits to a text file (similar to original output).
    
    Parameters
    ----------
    path : Path
        Output file path.
    header : str
        FASTA record header.
    seq : str
        Full sequence.
    hits : list
        List of hit objects with start, end, and score attributes.

    Returns
    -------
    None
        No return value but writes to file.
    """

    with path.open("w", encoding="utf-8") as f:
        f.write(f">{header}\n")
        f.write("Start\tEnd\tSequence\tLength\tScore\n")
        for h in hits:
            window_seq = seq[h.start:h.end]

            # note +1 to convert 0-based to 1-based coordinates
            f.write(f"{h.start+1}\t{h.end+1}\t{window_seq}\t{len(window_seq)}\t{h.score}\n")


# ................................................................................................
#
def _write_merged_regions(path: Path, header: str, regions: list) -> None:
    """
    Write merged regions to a text file (similar to original output).

    Parameters
    ----------
    path : Path
        Output file path.

    header : str
        FASTA record header.

    regions : list
        List of merged region objects with start, end, sequence, length, 
        and score attributes.

    Returns
    -------
    None
        No return value but writes to file.
    """

    with path.open("w", encoding="utf-8") as f:
        f.write(f">{header}\n")
        f.write("Start\tEnd\tSequence\tLength\tScore\tNBR\n")
        for idx, r in enumerate(regions, start=1):
            f.write(f"{r.start+1}\t{r.end+1}\t{r.sequence}\t{r.length}\t{r.score}\t{idx}\n")



def main() -> int:
    """Entry point for the g4hunter CLI."""


    # build argument parser
    # ...............................................................
    args = argparse.ArgumentParser(
        prog="g4hunter",
        description="Predict G4 propensity using G4Hunter algorithm (see Bedrat et al. 2016 NAR). G4Hunterpy3 is a Python 3 refactor.",)

    args.add_argument("-i", "--input", required=True, help="Input FASTA file")
    args.add_argument("-o", "--output", required=True, help="Output directory")
    args.add_argument("-w", "--window", type=int, default=25, help="Window size (k)")
    args.add_argument("--info", action="store_true", help="Print information about the sequnces being analyzed and exit")
    
    args.add_argument(
        "-s",
        "--score",
        type=float,
        default=1.2,
        help="Absolute score threshold for calling hits. If windows score below this they are not considered")

    args.add_argument(
        "--simple-plot",
        action="store_true",
        help="Also write a PDF plot of the sliding-window scores per record")

    args.add_argument(
        "--complex-plot",
        action="store_true",
        help="Also write a PDF plot of the sliding-window scores per record")

    args.add_argument(
        "--complex-plot-nbins",
        type=int,
        default=1000,
        help="Number of bins for the complex plot. Note that 1000 bins is approriate for a large genome, but ")
    
    args.add_argument(
        "--complex-plot-percentile",
        type=int,
        default=95,
        help="Percentile to use for y-axis limit in complex plot")

    # ...............................................................

    # parse CLI arguments
    args = args.parse_args()
    
    # fix paths
    input_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for header, seq in _iter_fasta_records(input_path):

        if args.info:
            print(f">{header}\nLength: {len(seq)}")
            

        # calculate window scores, hits, and merged regions
        window_scores, hits, regions = scan_sequence(
            seq, window_size=args.window, threshold=args.score
        )

        safe_header = header.replace("/", "_")
        hits_path = out_dir / f"{safe_header}-W{args.window}-S{args.score}.txt"
        merged_path = out_dir / f"{safe_header}-Merged.txt"

        if args.info:
            print(f"Number of window hits: {len(hits)}")
            print(f"Number of merged regions: {len(regions)}")
            print(f"writing results to:\n  {hits_path}\n  {merged_path}\n")

        _write_window_hits(hits_path, header, seq, hits)
        _write_merged_regions(merged_path, header, regions)

        if args.simple_plot:
            plot_path = out_dir / f"{safe_header}-ScorePlot.pdf"
            simple_plot(window_scores, plot_path)

        if args.complex_plot:
            plot_path = out_dir / f"{safe_header}-ComplexScorePlot.pdf"
            complex_plot(hits, 
                         genome_length=len(seq), 
                         out_pdf=plot_path, 
                         nbins=args.complex_plot_nbins, 
                         percentile_to_use=args.complex_plot_percentile)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
