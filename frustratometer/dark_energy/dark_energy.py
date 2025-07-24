import numpy as np
import pandas as pd

# stats
from scipy.stats import pearsonr

## amino acid abundance
from collections import Counter

#plot
from matplotlib import pyplot as plt, cm, colors
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import matplotlib
import seaborn as sns

# pdb view
import py3Dmol
import zipfile
import tarfile





def compute_dark_energy_matrix(EVO_matrix: np.array,
                               FOLD_matrix: np.array,
                               Tsel_from: str = 'variance',
                               kB = 0.001985875,
                               Tsel = None,
                               return_evo_rescaled = False):
    
    """
        Compute the Dark Energy Variations.

        Args:
            np.ndarray: Evolutionary Energy Variations (matrix).
            np.ndarray: Folding Energy Variations (matrix).
            Tsel_from (str): Method to compute Tsel ('variance', 'slope_folding_to_evo', 'slope_evo_to_folding', 'user_defined').
            return_evo_rescaled (bool): Whether to return the rescaled evolutionary matrix.

        Returns:
            np.ndarray: Dark Energy Variations.
            np.ndarray: Rescaled evolutionary matrix (optional).
            
    """
    valid = {'variance', 'slope_folding_to_evo', 'slope_evo_to_folding', 'user_defined'}
    
    if Tsel_from not in valid:
        raise ValueError("Error: Tsel_from must be one of %r." % valid)
        return 0                 
                               
    elif Tsel_from == 'variance':
        slope = (np.nanvar(FOLD_matrix)**.5) / (np.nanvar(EVO_matrix)**.5) 

        EVO_rescaled_matrix =  EVO_matrix * slope
        Tsel = slope/kB
        
    elif Tsel_from == 'user_defined':
        EVO_rescaled_matrix =  EVO_matrix * kB * Tsel            
    
    elif Tsel_from == 'slope_folding_to_evo':

        slope = compute_slope(FOLD_matrix.flatten(), EVO_matrix.flatten())                           
        EVO_rescaled_matrix =  EVO_matrix * slope
        Tsel = slope/kB

    elif Tsel_from == 'slope_evo_to_folding':

        slope = compute_slope(EVO_matrix.flatten(),FOLD_matrix.flatten())                           
        EVO_rescaled_matrix =  EVO_matrix / slope
        Tsel = 1/(slope*kB)

    DARK_ENERGY_matrix =  EVO_rescaled_matrix - FOLD_matrix
    
    if return_evo_rescaled:
        return DARK_ENERGY_matrix,EVO_rescaled_matrix,Tsel
    else:
        return DARK_ENERGY_matrix,Tsel

    
def compute_N_eff(evo_matrix): 
    
    """
         Compute effective number of amino acids (N_eff) for each site

        Args:
            np.ndarray: Energy Variations (matrix).
    
        Returns:
            np.array: Effective number of amino acids (N_eff)
            
    """
    # Step 1: Compute Boltzmann weights (w_i) for all sites
    w_i = np.exp(-evo_matrix)  # Shape: (L, 20)
    # Step 2: Compute partition function (Z_i) for each site
    Z_i = np.sum(w_i, axis=1, keepdims=True)  # Shape: (L, 1)
    # Step 3: Compute probabilities (P_i) for all sites
    P_i = w_i / Z_i  # Shape: (L, 20)
    # Step 4: Compute entropy (S) for each site
    S = -np.sum(P_i * np.log(P_i + 1e-10), axis=1)  # Shape: (L,)
    # Adding 1e-10 to avoid log(0), which is undefined
    # Step 5: Compute effective number of amino acids (N_eff) for each site
    N_eff = np.exp(S)  # Shape: (L,)
    return N_eff

def compute_slope(x, y):
        """Compute slope (fixing intercept at 0) while excluding NaN pairs."""
        valid = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(valid) == 0:
            return np.nan  # No valid pairs
        return np.sum(x[valid] * y[valid]) / np.sum(x[valid]**2)



def compute_aa_freq(seq, _AA, include_gaps=True):
    """
    Calculates amino acid frequencies in a given sequence using minimal memory.

    Parameters
    ----------
    seq : str
        Amino acid sequence (1-letter code, with gaps as '-')
    _AA : str
        String of amino acids defining the order (e.g., "-ACDEFGHIKLMNPQRSTVWY")
    include_gaps : bool
        Whether to include gaps ('-') in the frequency output.

    Returns
    -------
    aa_freq : np.array
        Array of amino acid counts in the order of _AA
    """
    counter = Counter(seq)
    aa_freq = np.array([counter.get(aa, 0) for aa in _AA])

    if not include_gaps:
        gap_index = _AA.find('-')
        if gap_index != -1:
            aa_freq[gap_index] = 0

    return aa_freq

def weighted_nanmean(matrix, weights, axis=0):
    """
    Compute the weighted mean along an axis, ignoring NaN values.

    Args:
        matrix (np.ndarray): Input matrix (may contain NaN values).
        weights (np.ndarray): Weights array (must have the same shape as the matrix).
        axis (int): Axis along which to compute the mean (default: 0).

    Returns:
        np.ndarray: Weighted mean along the specified axis.
    """
    # Create a mask for NaN values
    mask = np.isnan(matrix)
    
    # Set weights to 0 where the matrix has NaN values
    weights = np.where(mask, 0, weights)
    
    # Replace NaN values with 0 to avoid affecting the sum
    matrix = np.where(mask, 0, matrix)
    
    # Compute the weighted sum
    weighted_sum = np.sum(matrix * weights, axis=axis)
    
    # Compute the sum of weights (ignoring NaNs)
    sum_weights = np.sum(weights, axis=axis)
    
    # Avoid division by zero by setting sum_weights to NaN where it is zero
    sum_weights = np.where(sum_weights == 0, np.nan, sum_weights)
    
    # Compute the weighted mean
    weighted_mean = weighted_sum / sum_weights
    
    return weighted_mean

def flat_matrix(matrix,
                to = 'pos',
                aa_freq = None):
    if to=='pos':
        if aa_freq is not None:
            values = weighted_nanmean(matrix, aa_freq, axis=1)             
        else:
            values = np.nanmean(matrix,axis=1)
    elif to=='aa':
        values = np.nanmean(matrix,axis=0)
    else:
        raise ValueError("Error: to must be 'pos' or 'aa'")
        return 0
    return values

def compute_pearson(x, y):
        """Compute Pearson correlation coefficient and p-value while excluding NaN pairs."""
        valid = ~np.isnan(x) & ~np.isnan(y)
        if np.sum(valid) < 2:
            return np.nan, np.nan  # Need at least 2 valid pairs for Pearson
        return pearsonr(x[valid], y[valid])

def compute_slope(x, y):
    """Compute slope (fixing intercept at 0) while excluding NaN pairs."""
    valid = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(valid) == 0:
        return np.nan  # No valid pairs
    return np.sum(x[valid] * y[valid]) / np.sum(x[valid]**2)


def plot_mutational_matrix(matrix: np.array,
                           alphabet = '-ACDEFGHIKLMNPQRSTVWY',
                           highlighted_sites = None,
                           vmin= None,
                           vmax= None,
                           ax = None,
                           cbar_ax = None,
                           pdb_beg = 1):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(10,3.3))
    
    # Create heatmap and get colorbar
    heatmap = sns.heatmap(matrix.T, cmap='RdBu_r', center=0, ax=ax, 
                         vmax=vmax, vmin=vmin, cbar_ax=cbar_ax)
    
    # Explicitly set colorbar label if using custom cbar_ax
   
    
    ax.set_yticks(np.arange(len(alphabet))+0.5, list(alphabet))

    if highlighted_sites is None:
        # Default: show all positions (Matplotlib will auto-thin)
        xticks = ax.get_xticks()

        xlabels = [str(int(i + pdb_beg)) for i in xticks-.5]
        ax.set_xticks(xticks, xlabels)
    else:    
        ax.set_xticks(highlighted_sites+.5, ['*']*len(highlighted_sites))
        
    if cbar_ax is not None:
        return heatmap

    
    

def create_color_bar(cmap, vmin, vmax, center_value,label='Tsel'):
    
    fig, ax = plt.subplots(figsize=(8, 0.5))
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=ax, orientation='horizontal')
    cb.set_label(label)
    plt.show()

def map_hist_to_colors(h_, vmin, vmax, center_value, cmap="coolwarm",hex=True):
    norm = Normalize(vmin, vmax)
    center_norm = (center_value - vmin) / (vmax - vmin)  # Normalize center value
    #cmap = cm.get_cmap(cmap)
    cmap = matplotlib.colormaps[cmap]
    cmap_colors = cmap(np.linspace(0, 1, 256))
    cmap_colors[128] = [1, 1, 1, 1]  # Set the center of the colormap to white
    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center_value, vmax=vmax)
    rgba_values = cmap(norm(h_))
    if hex:
        colors = [mcolors.rgb2hex(rgba) for rgba in rgba_values]
    else:
        colors=rgba_values
    return colors


def view_3dmol(ali_seq, colors, pdb_filename, zip_file_path=None, zip_=False, chain='A',
                      clean_not_A_chain=True, highlight_residues=None,
                      second_chain=None, second_chain_surface=True,
                      second_chain_opacity=0.3, second_chain_color='lightgray'):
    
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')

    if zip_:
        if zip_file_path.endswith('.zip'):
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                with zip_ref.open(pdb_filename, 'r') as pdb_file:
                    pdb_content = pdb_file.read().decode('utf-8')
        elif zip_file_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(zip_file_path, 'r:gz') as tar_ref:
                member = tar_ref.getmember(pdb_filename)
                with tar_ref.extractfile(member) as pdb_file:
                    pdb_content = pdb_file.read().decode('utf-8')
    else:
        pdb_content = open(pdb_filename, 'r').read()

    view.addModel(pdb_content, 'pdb')
    view.setBackgroundColor('white')
    view.setStyle({'cartoon': {'color': 'white'}})

    if clean_not_A_chain:
        for c in 'BCDEFGHIJKLMNOPQRSTUVWXYZ':
            if c != chain:# and (second_chain is None or c != second_chain):
            
            #if c != chain and (second_chain is None or c != second_chain):
                view.setStyle({'chain': c}, {'opacity': 0})

    for i, res in enumerate(ali_seq):
        if res > 0:
            view.addStyle({'chain': chain, 'resi': [str(res)]}, {'cartoon': {'color': colors[i]}})

    if highlight_residues is not None:
        resi_strs = [str(r) for r in highlight_residues]
        view.addStyle({'chain': chain, 'resi': resi_strs, 'atom': 'CB'},
                      {'sphere': {'color': 'yellow', 'radius': 1.3}})
        view.addStyle({'chain': chain, 'resi': resi_strs, 'resn': 'GLY', 'atom': 'CA'},
                      {'sphere': {'color': 'yellow', 'radius': 1.3}})

    if second_chain and second_chain_surface:
        view.setStyle({'chain': second_chain}, {'opacity': 0})
        view.addSurface(py3Dmol.VDW,
                        {'opacity': second_chain_opacity, 'color': second_chain_color},
                        {'chain': second_chain})

    view.zoomTo(viewer=(0, 0))
    return view



def plot_correlation_color(A, B, C,correlation_type, T_sel, index=None, alphabet='ACDEFGHIKLMNPQRSTVWY',kB=0.001985875,
                     show_r=True, show_p=True, show_slope=True, show_T_sel=True, A_label = '', B_label='',
                          cmap='RdBu_r',vmax=None, alpha=0.5,edgecolors='black', ax=None, lw=1, s=5):
    """
    Plot correlations for a specific type and index, handling NaN values.

    Parameters:
        A, B: Input matrices (sequence_length x num_amino_acids).
        C: color scale
        correlation_type: "all_values", "sequence_position_level", or "amino_acid_level".
        index: For "sequence_position_level" or "amino_acid_level", specify the position or amino acid index.
        alphabet: Amino acid single-letter codes.
        show_r: Whether to include Pearson r in the label.
        show_p: Whether to include p-value in the label.
        show_slope: Whether to include slope in the label.
        show_T_sel: Whether to include T_sel (slope / kB) in the label.
    """

    # Extract data based on correlation type
    
    if correlation_type == "all_values":
        x = A.flatten()
        y = B.flatten()
        if C is not None:
            c = C.flatten()
        title = "Correlation (All Values)"
        xlabel = A_label
        ylabel = B_label
    elif correlation_type == "sequence_position_level":
        if index is None or index >= A.shape[0]:
            raise ValueError("Invalid sequence position index.")
        x = A[index, :]
        y = B[index, :]
        if C is not None:
            c =  C[index, :]
        title = f"Correlation (Sequence Position {index + 1})"
        xlabel = A_label+f" at Position {index + 1}"
        ylabel = B_label+f" at Position {index + 1}"
    elif correlation_type == "amino_acid_level":
        if index is None or index >= A.shape[1]:
            raise ValueError("Invalid amino acid index.")
        x = A[:, index]
        y = B[:, index]
        if C is not None:
            c =  C[:, index]
        aa_code = alphabet[index]  # Get amino acid code from alphabet
        title = f"Correlation (Amino Acid {aa_code})"
        xlabel = A_label+ f" for {aa_code}"
        ylabel = A_label+ f" for {aa_code}"
    else:
        raise ValueError("Invalid correlation type.")

    # Compute slope, Pearson r, p-value, and T_sel, excluding NaN pairs
    slope = compute_slope(x, y)
    r, p = compute_pearson(x, y)
    #if fold=='X':
    #    T_sel = 1/ slope / kB if not np.isnan(slope) else np.nan
    #else:
    #    T_sel = slope / kB if not np.isnan(slope) else np.nan

    
    # Build the label dynamically based on user preferences
    label_parts = []
    if show_r:
        label_parts.append(f"r = {r:.3f}")
    if show_p:
        label_parts.append(f"p = {p:.3e}")
    if show_slope:
        label_parts.append(f"slope = {slope:.3f}")
  #  if show_T_sel:
  #      label_parts.append(f"T_sel = {T_sel:.3f}")
    label = ", ".join(label_parts)

    
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    valid = ~np.isnan(x) & ~np.isnan(y)  # Mask for valid pairs

    ax.scatter(x[valid], y[valid], c=c[valid], cmap=cmap, norm=mcolors.CenteredNorm(vcenter=0, halfrange=vmax), alpha=alpha, label=label, edgecolors=edgecolors,lw=lw, s=s, zorder=2)
    if not np.isnan(slope):
       # plt.plot(x[valid], slope * x[valid], color="grey", label="Regression Line (slope)",alpha=0.2)
        #ax.plot([np.nanmin(A),np.nanmax(A)], np.array([np.nanmin(A),np.nanmax(A)])/(T_sel*kB),
        #color="k", label=rf"$T_{{\mathrm{{sel}}}}$ = {T_sel:.0f}K", alpha=1, zorder=3, lw=1)
        ax.plot(
            [np.nanmin(A), np.nanmax(A)],
            np.array([np.nanmin(A), np.nanmax(A)]) / (T_sel * kB),
            color="k",
            label=rf"$T_{{\mathrm{{sel}}}}^{{\mathrm{{fold}}}}$ = {T_sel:.0f} K",
            alpha=1,
            zorder=3,
            lw=1
        )
    ax.set_xlim([np.nanmin(A),np.nanmax(A)])
    ax.set_ylim([np.nanmin(B),np.nanmax(B)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
  #  plt.title(title)
    ax.legend(loc='lower right')
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.grid(visible=True,zorder=1)


def plot_flat_with_highlighted_sites(dem_flat,
                                     active_sites,
                                     pdb_seq, 
                                     chunk_size = 100,
                                     ax = None):
    # Calculate the number of chunks needed
    num_chunks = (len(pdb_seq) + chunk_size - 1) // chunk_size
    
    tlayout = False

    if ax is None:
    # Create subplots
        fig, ax = plt.subplots(num_chunks, 1, figsize=(10, 1.5 * num_chunks))
        tlayout = True
    # If there's only one chunk, ax will not be an array, so we need to handle that case
    if num_chunks == 1:
        ax = [ax]

    # Iterate over each chunk
    for i in range(num_chunks):
        # Define the start and end of the chunk
        start = i * chunk_size
        
        end = start + chunk_size
        
        # For the last chunk, extend the x-axis to the full chunk size
        if i == num_chunks - 1:
            # Plot the bar plot for the current chunk
            ax[i].bar(pdb_seq[start:], height=dem_flat[start:], color='k', 
                      edgecolor='k', linewidth=0.5)

            # Find active sites within the current chunk
            active_in_chunk = [site for site in active_sites if start <= site < len(pdb_seq)]

            # Plot the active sites in red
            ax[i].bar(pdb_seq[active_in_chunk], height=dem_flat[active_in_chunk], color='y',
                      edgecolor='y', linewidth=0.5)
            # Set the x-axis limits for the last subplot to the full chunk size
            ax[i].set_xlim([pdb_seq[start]-1, pdb_seq[start] + chunk_size +1])
        else:
            # Plot the bar plot for the current chunk
            ax[i].bar(pdb_seq[start:end], height=dem_flat[start:end], color='k', 
                      edgecolor='k', linewidth=0.5)

            # Find active sites within the current chunk
            active_in_chunk = [site for site in active_sites if start <= site < end]

            # Plot the active sites in red
            ax[i].bar(pdb_seq[active_in_chunk], height=dem_flat[active_in_chunk], color='y',
                      edgecolor='y', linewidth=0.5)
            ax[i].set_xlim([pdb_seq[start]-1, pdb_seq[start] + chunk_size +1])

        # Optional: Set y-axis limits and add a threshold line
        ax[i].set_ylim([min(dem_flat)*1.1,max(dem_flat)*1.1 ])
        # ax[i].axhline(dem_threshold, color='grey', ls='--', alpha=0.3)

    # Adjust layout for better spacing
    if tlayout:
        plt.tight_layout()
        plt.show()
    return 0
    

