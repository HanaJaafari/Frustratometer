"""dark_energy: Dark Energy functions

This module includes functions related to dark energy calculation, including plots and PDB visualization using py3dmol

"""
from .dark_energy import *

__all__ = ['compute_dark_energy_matrix',
           'compute_N_eff', 
           'compute_slope', 
           'compute_aa_freq',
           'weighted_nanmean', 
           'flat_matrix',
           'compute_pearson',
           'compute_slope',
           'plot_mutational_matrix', 
           'create_color_bar',
           'map_hist_to_colors', 
           'view_3d_exon_hist', # change name and update
           'plot_correlation_color', 
           'plot_flat_with_highlighted_sites'
          ]
