
import numpy as np
import matplotlib.pyplot as plt  # local import so CLI works without mpl
from pathlib import Path
import matplotlib

# so we can edit text if PDFs generated
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {'family' : 'arial',
    	'weight' : 'normal'}

matplotlib.rc('font', **font)

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
                 score: float = 1.2,
                 percentile_to_use: int = 95,                 
                 dpi: int = 300,
                 figsize: tuple = (8, 1.5),
                 colorbar_vmax: float = 3.0,
                 colorbar_vmin: float = None,
                 highlight_regions: list = None,
                 strand_agnostic: bool = True   
                 ):
    
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

    score : float, optional
        Score threshold used for calling hits, used to set floor
        on colorbar. By default 1.2.

    percentile_to_use : int, optional
        Percentile of scores to use within each bin (e.g., 95 for 
        95th percentile), by default 95.

    dpi : int, optional
        Dots per inch for the output PDF, by default 300.
        
    figsize : tuple, optional
        Figure size as (width, height) in inches, by default (8, 2.0).

    colorbar_vmax = float, optional
        Maximum value for colorbar, by default 3.0.
        
    colorbar_vmin = float, optional
        Mininum value for colorbar, by default max(score, 0.0).

    highlight_regions : list, optional
        List of [start, end] pairs defining regions to highlight on the
        x-axis. Each element should be a list or tuple with two integers
        representing the start and end positions (1-based) of the region
        to highlight. Highlighted regions are shown as yellow vertical
        spans with alpha=0.5. By default None (no highlighting).

    strand_agnostic : bool, optional
        If True, use absolute G4 scores (ignoring strand) for plotting.
        If False, use raw scores (which can be negative for C-rich
        regions). Set to true for dsDNA sequences where both strands 
        can form G4s. By default True.
        
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
    
    # Set colorbar limits based on strand_agnostic mode
    if colorbar_vmin is None:
        if strand_agnostic:
            colorbar_vmin = max(score, 0.0)
        else:
            # For strand-specific, use symmetric limits around zero
            colorbar_vmin = -colorbar_vmax

    
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
    if strand_agnostic:
        signal = np.abs(full)
    else:
        signal = full

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

        # returns a boolean array where True indicates valid (non-NaN) 
        # data
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

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.25, 1, 1], hspace=0.3)

    # Top subplot for colorbar
    cbar_ax = fig.add_subplot(gs[0, 0])
    cbar_ax.axis('off')  # Hide the axes
    
    # Middle subplot for heatmap
    ax = fig.add_subplot(gs[1, 0])
    
    # Use appropriate colormap based on strand_agnostic setting
    if strand_agnostic:
        cmap = plt.cm.Reds.copy()
    else:
        # Use diverging colormap for strand-specific: blue (C-rich/negative) to red (G-rich/positive)
        cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color='lightgray')  # missing bins

    im = ax.imshow(
        heat, aspect='auto', interpolation='nearest',
        extent=[1, genome_length, 0, 1], cmap=cmap, vmin=colorbar_vmin, vmax=colorbar_vmax)
    ax.set_yticks([])
    ax.set_xlim(1, genome_length)
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_ylabel("G4 propensity", rotation=0, labelpad=35, va='center')
    
    # Add highlight regions if provided
    if highlight_regions is not None:
        for region in highlight_regions:
            start, end = region[0], region[1]
            ax.axvspan(start, end, lw=0, color='y', alpha=0.5)
    
    # Create colorbar in the top subplot area
    cb = plt.colorbar(im, ax=cbar_ax, orientation='horizontal', fraction=0.8, pad=0.1)    
    # Set only min/max ticks from actual data
    valid_data = binned[np.isfinite(binned)]
    if len(valid_data) > 0:
        vmin, vmax = colorbar_vmin, colorbar_vmax
        cb.set_ticks([])  # Remove automatic ticks
        # Manually position min/max labels at the edges
        cb.ax.text(-0.12, 0.9, f'{vmin:.2f}', ha='left', va='top', fontsize=6, transform=cb.ax.transAxes)
        cb.ax.text(1.12, 0.9, f'{vmax:.2f}', ha='right', va='top', fontsize=6, transform=cb.ax.transAxes)
    
    # Bottom subplot for coverage
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax)
    

    coverage_positions = np.linspace(1, genome_length, nbins)
    bar_width = coverage_positions[1]-coverage_positions[0]
    ax2.bar(coverage_positions, coverage, color='k',width=bar_width)


    if highlight_regions is not None:
        for region in highlight_regions:
            start, end = region[0], region[1]
            ax2.axvspan(start, end, lw=0, color='y', alpha=0.5)
    

    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Coverage", rotation=0, labelpad=20, va='center')
    ax2.set_xticks(np.arange(0, genome_length+1, 20000))
    ax2.set_xlabel("Position (bp)")
    
    # Add common y-axis label
    fig.suptitle(f"G4 propensity (bin_size={edges[1]-edges[0]} bps, window percentile={percentile_to_use}%)", y=0.98, fontsize=8)
    plt.tight_layout(pad=1.0)

    fig.savefig(str(out_pdf), dpi=dpi, bbox_inches='tight')
    # ................................................................................................
    #
