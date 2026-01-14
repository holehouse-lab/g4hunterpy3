g4hunterpy3
==============================
[//]: # (Badges)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)
![Maintained](https://img.shields.io/badge/Maintained%3F-yes-green.svg)
![Last Commit](https://img.shields.io/github/last-commit/holehouse-lab/g4hunterpy3)
![License](https://img.shields.io/github/license/holehouse-lab/g4hunterpy3)

Python 3 implementation of G4Hunter as a full Python package. G4Hunter predicts G-quadruplex (G4) propensity in DNA sequences using a sliding window approach based on the algorithm described by Bedrat et al. (2016).

We created this package because we wanted to use G4Hunter



---

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Command-Line Interface (CLI)](#command-line-interface-cli)
  - [Basic Usage](#basic-usage)
  - [Options](#options)
  - [Examples](#examples)
  - [Output Files](#output-files)
- [How to Cite](#how-to-cite)
- [Copyright](#copyright)

---

## About

G4Hunter is a bioinformatic tool designed to predict the formation propensity of **G-quadruplexes (G4)**, which are four-stranded nucleic acid structures stabilized by guanine quartets. Unlike previous pattern-matching algorithms that rely on rigid consensus sequences, G4Hunter uses a scoring system based on **G-richness** and **G-skewness**.

---

### Core Scoring Principles

The algorithm assigns a score to every nucleotide in a sequence to reflect its contribution to (or competition with) G4 formation:

* 
**Neutral Bases:** Adenine (A) and Thymine (T) are assigned a score of **0**.


* **Guanine (G):** Assigned positive scores based on the length of the G-run:
* A single G = **1** 


* Each G in a GG run = **2** 


* Each G in a GGG run = **3** 


* Each G in a run of 4 or more Gs = **4** 




* 
**Cytosine (C):** Assigned the same values as G, but **negative** (e.g., a CCC run scores -3 per C). This penalizes regions where high C-content would favor stable duplex formation over G4 structures.



### Calculation Method

1. 
**Sliding Window:** The algorithm typically uses a sliding window (default: **25 nt**) to compute the arithmetic mean of the scores within that window.


2. 
**G4Hscore:** The resulting mean value is the **G4Hscore**.


3. **Thresholding:** A threshold () is applied to extract G-quadruplex Forming Sequences (G4FS). Typical values for  range between **1.0 and 2.0**. 

	* Thresold of 1.2: This is a recommended compromise for identifying many true G4 motifs while maintaining reasonable precision. 
	* Threshold of 1.5: This is recommended for high-confidence predictions (precision >90%) where stable G4 formation is likely.

	
For more information, [see the manuscript](https://academic.oup.com/nar/article/44/4/1746/1854457)


## Installation

Install from source:

```bash
git clone https://github.com/holehouse-lab/g4hunterpy3.git
cd g4hunterpy3
pip install .
```

Or install in development mode:

```bash
pip install -e .
```

### Dependencies

- Python >= 3.8
- numpy
- matplotlib
- protfasta

---

## Command-Line Interface (CLI)

After installation, the `g4hunterpy3` command becomes available in your terminal. This CLI tool reads FASTA files containing DNA sequences, calculates G4Hunter scores using a sliding window approach, and outputs hit regions and merged regions to text files.

### Basic Usage

```bash
g4hunterpy3 -i <input.fasta> -o <output_directory> [options]
```

### Options

| Option | Short | Required | Default | Description |
|--------|-------|----------|---------|-------------|
| `--input` | `-i` | **Yes** | — | Path to the input FASTA file containing DNA sequence(s) |
| `--output` | `-o` | **Yes** | — | Path to the output directory where results will be saved |
| `--window` | `-w` | No | `25` | Window size (k) for the sliding window analysis |
| `--score` | `-s` | No | `1.2` | Absolute score threshold for calling hits. Windows scoring below this threshold (in absolute value) are not reported |
| `--info` | — | No | `false` | Print information about the sequences being analyzed (length, number of hits, number of merged regions) |
| `--simple-plot` | — | No | `false` | Generate a simple PDF plot of the sliding-window scores for each sequence |
| `--complex-plot` | — | No | `false` | Generate a complex PDF plot with binned visualization of sliding-window scores |
| `--complex-plot-nbins` | — | No | `1000` | Number of bins for the complex plot (appropriate for large genomes) |
| `--complex-plot-percentile` | — | No | `95` | Percentile to use for y-axis limit in complex plot |

### Examples

#### Basic analysis with default parameters

```bash
g4hunterpy3 -i sequences.fasta -o results/
```

This runs G4Hunter with:
- Window size: 25
- Score threshold: 1.2

#### Custom window size and threshold

```bash
g4hunterpy3 -i genome.fasta -o output/ -w 30 -s 1.5
```

This uses:
- Window size: 30 bp
- Score threshold: 1.5 (more stringent, fewer hits)

#### Get sequence information

```bash
g4hunterpy3 -i sequences.fasta -o results/ --info
```

This prints details about each sequence including:
- Sequence length
- Number of window hits
- Number of merged regions
- Output file paths

#### Generate visualization plots

```bash
# Simple score plot
g4hunterpy3 -i sequences.fasta -o results/ --simple-plot

# Complex binned plot (useful for large sequences/genomes)
g4hunterpy3 -i genome.fasta -o results/ --complex-plot --complex-plot-nbins 500

# Both plots with custom percentile for y-axis
g4hunterpy3 -i genome.fasta -o results/ --simple-plot --complex-plot --complex-plot-percentile 99
```

#### Full example with all options

```bash
g4hunterpy3 \
    -i my_sequences.fasta \
    -o ./g4_results/ \
    -w 25 \
    -s 1.2 \
    --info \
    --simple-plot \
    --complex-plot \
    --complex-plot-nbins 1000 \
    --complex-plot-percentile 95
```

### Output Files

For each sequence record in the input FASTA file, the CLI generates the following output files:

#### 1. Per-Window Hit File

**Filename:** `<sequence_header>-W<window_size>-S<threshold>.txt`

Contains all windows that pass the score threshold:

| Column | Description |
|--------|-------------|
| Start | 1-based start position of the window |
| End | 1-based end position of the window |
| Sequence | The nucleotide sequence of the window |
| Length | Length of the window (equals window size) |
| Score | G4Hunter score for the window |

#### 2. Merged Region File

**Filename:** `<sequence_header>-Merged.txt`

Contains merged regions formed by overlapping or adjacent window hits:

| Column | Description |
|--------|-------------|
| Start | 1-based start position of the merged region |
| End | 1-based end position of the merged region |
| Sequence | The nucleotide sequence of the merged region |
| Length | Total length of the merged region |
| Score | Mean G4Hunter score across the region |
| NBR | Region number (sequential identifier) |

#### 3. Plot Files (optional)

- **Simple Plot:** `<sequence_header>-ScorePlot.pdf` — A straightforward visualization of G4Hunter scores across the sequence
- **Complex Plot:** `<sequence_header>-ComplexScorePlot.pdf` — A binned visualization suitable for large sequences/genomes

### Understanding G4Hunter Scores

- **Positive scores** indicate G-rich regions (potential G4-forming on the forward strand)
- **Negative scores** indicate C-rich regions (potential G4-forming on the reverse strand)
- **Score magnitude** reflects the G4-forming propensity:
  - |score| ≥ 1.2: Moderate propensity
  - |score| ≥ 1.5: High propensity
  - |score| ≥ 2.0: Very high propensity

---

## How to Cite

Please cite the original G4Hunter paper [1], and link to this repository so folks can reproduce analysis with this implementation.

[1] Bedrat, A., Lacroix, L. & Mergny, J.-L. Re-evaluation of G-quadruplex propensity with G4Hunter. Nucleic Acids Res. 44, 1746–1759 (2016). [Link](https://doi.org/10.1093/nar/gkw006)

---

## Copyright

Copyright (c) 2025-2026, Alex Holehouse

---

#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.11.
