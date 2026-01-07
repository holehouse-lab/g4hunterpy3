
import numpy as np
import matplotlib.pyplot as plt  # local import so CLI works without mpl
from pathlib import Path

# ................................................................................................
#
def simple_plot(scores: np.ndarray, 
                out_pdf: Path, 
                dpi: int = 300,
                line_color = "red",
                line_width = 0.8) -> None:
    """
    Save a PDF plot of the sliding-window scores.

    Parameters
    ----------
    scores : np.ndarray
        Array of sliding-window scores.

    out_pdf : Path
        Output PDF file path.

    dpi : int, optional
        Dots per inch for the output PDF, by default 300.
    Returns
    -------
    None
        No return value but writes to file.
    """
    
    # define x values
    x = np.arange(scores.size)

    # build figure
    fig = plt.figure()
    plt.plot(x, scores, line_color, linewidth=line_width)
    plt.xlim(0, max(1, scores.size))
    plt.xlabel("Window start (0-based)")
    plt.ylabel("G4Hunter score (mean)")
    plt.grid(True)
    fig.savefig(str(out_pdf), dpi=dpi)
    plt.close(fig)

# ................................................................................................
#


# ................................................................................................
#
def complex_plot(hits: list,                  
                 genome_length: int,                                                    
                 out_pdf: Path, 
                 nbins: int = 1000,
                 percentile_to_use: int = 95,                 
                 dpi: int = 300):
    
    """
    Save a PDF plot of the sliding-window scores.

    Parameters
    ----------

    hits : list
        List of hit objects with start, end, and score attributes.
        Each hit represents a window with a calculated G4Hunter score.

    genome_length : int
        Length of the genome/sequence being analyzed; this is needed
        to map the hits to the full length sequence.
    
    out_pdf : Path
        Output PDF file path.

    nbins : int, optional
        Number of bins for the complex plot, by default 1000.

    percentile_to_use : int, optional
        Percentile of scores to use within each bin (e.g., 95 for 
        95th percentile), by default 95.

    dpi : int, optional
        Dots per inch for the output PDF, by default 300.
        
    Returns
    -------
    None
        No return value but writes to file.
    """

    ## sanity check stuff 

    if nbins < 1:
        raise ValueError("nbins must be >= 1")
    
    if percentile_to_use < 0 or percentile_to_use > 100:
        raise ValueError("percentile_to_use must be between 0 and 100")
    
    if genome_length < 1:
        raise ValueError("genome_length must be >= 1")  
    
    if dpi < 100:
        raise ValueError("dpi must be >= 100")
    
    # build up scores and positions arrays from hits
    scores = []
    positions = []
    for h in hits:            
        scores.append(h.score)
        positions.append(h.start)  # 1-based positions

    scores = np.array(scores)
    positions = np.array(positions)
    
    ## Build full-length array with NaNs for missing - initialize with NaNs
    ## and then overwrite with the real values so we leave "missing" positions as NaN
    full = np.full(genome_length, np.nan)
    
    full[positions - 1] = scores   # positions are 1-based, so correct

    # get strand-agnostic G4 propensity
    signal = np.abs(full)

    # build "edges"
    edges = np.linspace(0, genome_length, nbins + 1, dtype=int)

    # this should be true but we check to be sure
    assert len(signal) == genome_length

    # initialize binned arrays
    binned = np.full(nbins, np.nan)
    coverage = np.zeros(nbins)

    for i in range(nbins):

        # get a local segment of the signal (absolute G4 score)
        seg = signal[edges[i]:edges[i+1]]

        # returns a boolean array where True indicates valid (non-NaN) data
        valid = np.isfinite(seg)
        

        # fraction of positions with data (e.g. if 1 = all positions have data, 0 = none)
        coverage[i] = valid.mean()  

        # if we had ANY valid data in this segment...
        if valid.any():
            
            # get xxx-percentile of valid data in this segment. We take 95th percentile
            # to capture the strongest G4 signal in this bin, while ignoring outliers. This
            # serves to highlight regions with strong G4 propensity while avoiding overemphasis 
            # on local overlapping regions (e.g. if you had a run of G4s overlapping each other,
            # you don't want that to dominate the entire bin).
            binned[i] = np.nanpercentile(seg, percentile_to_use)

    # Make it a 1-row "heatmap"
    heat = np.ma.masked_invalid(binned.reshape(1, -1))

    fig = plt.figure(figsize=(8, 0.8), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color='lightgray')  # missing bins

    im = ax.imshow(
        heat, aspect='auto', interpolation='nearest',
        extent=[1, genome_length, 0, 1], cmap=cmap)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(1, genome_length)
    #ax2.set_xlabel("HSV-1 genome position (bp)")
    #cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    #cb.set_label("G4Hunter |score| (95th percentile per bin)")

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    #ax2.plot(np.linspace(1, genome_length, nbins), coverage, color='k',lw=0.4)

    coverage_positions = np.linspace(1, genome_length, nbins)
    bar_width = coverage_positions[1]-coverage_positions[0]
    ax2.bar(coverage_positions, coverage, color='k',width=bar_width)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("cov.")
    ax2.set_xlabel("")
    ax2.set_xticks(np.arange(0, genome_length+1, 20000))
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    fig.suptitle(f"G4 propensity (bin_size={edges[1]-edges[0]} bps, window percentile={percentile_to_use}%)", y=1.25, fontsize=8)

    fig.savefig(str(out_pdf), dpi=dpi)
    # ................................................................................................
    #
