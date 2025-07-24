import numpy as np
from .. import dark_energy



class DarkEnergy:
    def __init__(self, EVO_matrix, FOLD_matrix, seq, 
                 alphabet='-ACDEFGHIKLMNPQRSTVWY', kB=0.001985875, Tsel=None):
        """
        Initialize the class with EVO_matrix, FOLD_matrix, sequence, and alphabet.

        Args:
            EVO_matrix (np.ndarray): Evolutionary matrix (sequence_length x num_amino_acids).
            FOLD_matrix (np.ndarray): Folding energy matrix (sequence_length x num_amino_acids).
            seq (str): Protein sequence.
            alphabet (str): Amino acid alphabet (default: '-ACDEFGHIKLMNPQRSTVWY').
            kB (float): Boltzmann constant (default: 0.001985875 kcal/mol/K).
            Tsel (float): Selection temperature (optional).
        """
        self.EVO_matrix = EVO_matrix
        self.FOLD_matrix = FOLD_matrix
        self.seq = seq
        self.alphabet = alphabet
        self.kB = kB
        self.Tsel = Tsel
        self.DEM = None
        self.EVO_r = None
        self.N_eff = None
        self.N_eff_fold = None
        self.N_eff_func = None

    def compute_dark_energy_matrix(self, Tsel_from='variance', return_evo_rescaled=True):
        """
        Compute the Dark Energy Variations.

        Args:
            
            Tsel_from (str): Method to compute Tsel ('variance', 'slope_folding_to_evo', 'slope_evo_to_folding', 'user_defined').
            return_evo_rescaled (bool): Whether to return the rescaled evolutionary matrix.

        Returns:
            np.ndarray: Dark Energy Variations.
            np.ndarray: Rescaled evolutionary matrix (optional).
            
        """
        self.DEM, self.EVO_r,self.Tsel = dark_energy.compute_dark_energy_matrix(
            self.EVO_matrix, self.FOLD_matrix, Tsel_from, self.kB, self.Tsel, return_evo_rescaled
        )
        
        return self.DEM, self.EVO_r

    def compute_all_N_effs(self):
    
        self.N_eff = dark_energy.compute_N_eff(-self.EVO_matrix)
        self.N_eff_fold = dark_energy.compute_N_eff(-self.FOLD_matrix/self.kB/self.Tsel)
        self.N_eff_func = dark_energy.compute_N_eff(self.DEM/self.kB/self.Tsel)
    
        return self.N_eff,self.N_eff_fold,self.N_eff_func
    
    def compute_aa_freq(self, include_gaps=True):
        """
        Compute amino acid frequencies from the sequence.

        Args:
            include_gaps (bool): Whether to include gaps in the frequency calculation.

        Returns:
            np.ndarray: Amino acid frequencies.
        """
        return dark_energy.compute_aa_freq(self.seq, self.alphabet, include_gaps)

    def flat_matrix(self, matrix, to='pos', aa_freq=None):
        """
        Flatten the matrix by averaging over amino acids or sequence positions.

        Args:
            matrix (np.ndarray): Input matrix (sequence_length x num_amino_acids).
            to (str): Type of flattening ('pos' for positions, 'aa' for amino acids).
            aa_freq (np.ndarray): Amino acid frequencies (optional).

        Returns:
            np.ndarray: Flattened matrix.
        """
        return dark_energy.flat_matrix(matrix, to, aa_freq)

    
    
    def plot_correlation_color(self, A_label=r'$\Delta E^{fold}$', B_label=r'$\Delta \Psi^{evo}$',
                        correlation_type ="all_values", show_r=False, show_p=False, show_slope=False, show_T_sel=True, index=None,cmap='RdBu_r',alpha=0.5, lw=1, s=5,vmax=None):
        """
        Plot correlations for a specific type and index, handling NaN values.

        Parameters:
            A, B: Input matrices (sequence_length x num_amino_acids).
            correlation_type: "all_values", "sequence_position_level", or "amino_acid_level".
            index: For "sequence_position_level" or "amino_acid_level", specify the position or amino acid index.
        """


        return dark_energy.plot_correlation_color(self.FOLD_matrix, self.EVO_matrix, self.DEM, correlation_type, self.Tsel, index,   alphabet=self.alphabet,kB=self.kB, show_r=show_r, show_p=show_p, show_slope=show_slope, show_T_sel=show_T_sel, A_label = A_label, B_label=B_label,cmap=cmap,alpha=alpha, lw=lw, s=s,vmax=vmax)
        
       

    
    def plot_mutational_matrix(self, highlighted_sites=None,
                               vmin= None,
                               vmax= None,
                               pdb_beg = 1):
        """
        Plot the Dark Energy Matrix (DEM) with highlighted sites.

        Args:
            highlighted_sites (list or np.ndarray): Indices of sites to highlight.
        """
        dark_energy.plot_mutational_matrix(self.DEM, alphabet=self.alphabet, highlighted_sites=highlighted_sites,
                              vmin= vmin, vmax= vmax, pdb_beg = pdb_beg)
        
        
        
    def create_color_bar(self, cmap, vmin, vmax, center_value, label='Tsel'):
        """
        Create a color bar for visualization.

        Args:
            cmap (str): Colormap name.
            vmin (float): Minimum value for the colormap.
            vmax (float): Maximum value for the colormap.
            center_value (float): Center value for the colormap.
            label (str): Label for the color bar.
        """
        dark_energy.create_color_bar(cmap, vmin, vmax, center_value, label)

    def map_hist_to_colors(self, h_, vmin, vmax, center_value, cmap="coolwarm", hex=True):
        """
        Map histogram values to colors using a colormap.

        Args:
            h_ (np.ndarray): Histogram values.
            vmin (float): Minimum value for the colormap.
            vmax (float): Maximum value for the colormap.
            center_value (float): Center value for the colormap.
            cmap (str): Colormap name.
            hex (bool): Whether to return colors in hex format.

        Returns:
            list: List of colors.
        """
        return dark_energy.map_hist_to_colors(h_, vmin, vmax, center_value, cmap, hex)

    def view_3dmol(self, ali_seq, colors, pdb_filename, zip_file_path=None, zip_=False, chain='A',
                         clean_not_A_chain = True,highlight_residues = None,
                                second_chain=None, second_chain_surface=True,
                                second_chain_opacity=0.3, second_chain_color='lightgray'):
        """
        Visualize a PDB structure with colored residues using py3Dmol.

        Args:
            ali_seq (np.ndarray): Array of residue numbers.
            colors (list): List of colors for each residue.
            pdb_filename (str): Path to the PDB file.
            zip_file_path (str): Path to the zip file containing the PDB (optional).
            zip_ (bool): Whether the PDB file is inside a zip archive.

        Returns:
            py3Dmol.view: 3D visualization object.
        """
        return dark_energy.view_3dmol(ali_seq, colors, pdb_filename, zip_file_path, zip_,chain, clean_not_A_chain,
                                 highlight_residues = highlight_residues,
                                second_chain=second_chain, second_chain_surface=second_chain_surface,
                                  second_chain_opacity=second_chain_opacity, second_chain_color=second_chain_color)


    
    def visualize_pdb(self, pdb_filename,
                      zip_file_path=None, zip_=False, highlight_sites=None,pdb_seq_num =None, cmap='RdBu_r', label='site-average $\Delta E_{Dark}$',chain='A', clean_not_A_chain=True,vmax=None, highlight_residues = None,
                                     second_chain=None, second_chain_surface=True,
                                       second_chain_opacity=0.3, second_chain_color='lightgray'):
        """
        Visualize the PDB structure with a color scale or highlighted sites.

        Args:
            pdb_filename (str): Path to the PDB file.
            zip_file_path (str): Path to the zip file containing the PDB (optional).
            zip_ (bool): Whether the PDB file is inside a zip archive.
            highlight_sites (list): List of sites to highlight (optional).
            cmap (str): Colormap name.
            label (str): Label for the color bar.
        """
        # Compute flattened DEM values
        DEM_flat = self.flat_matrix(self.DEM, 'pos', self.compute_aa_freq())

        # Create color bar
        if vmax is None:
            vmax = max(DEM_flat)
        vmin = -vmax
        # Highlight specific sites if provided
        if highlight_sites is not None:
            colors_ = np.array(['white'] * len(self.seq))
            colors_[highlight_sites] = 'red'
        # Map DEM values to colors
        else:
            self.create_color_bar(cmap, vmin, vmax, 0, label) 
            colors_ = self.map_hist_to_colors(DEM_flat, vmin, vmax, 0, cmap=cmap)

        if pdb_seq_num is None:
            pdb_seq_num_ = np.arange(len(self.seq)) + 1
        else:
            pdb_seq_num_=pdb_seq_num
        # Visualize the PDB structure
        
        view = self.view_3dmol(pdb_seq_num_, colors_, pdb_filename, zip_file_path, zip_, chain,
                                     clean_not_A_chain,highlight_residues=highlight_residues,
                                     second_chain=second_chain, second_chain_surface=second_chain_surface,
                                       second_chain_opacity=second_chain_opacity, second_chain_color=second_chain_color)
        view.show()
        

    def plot_flat_with_highlighted_sites(self,
                                 active_sites,
                                 pdb_seq, 
                                 aa_freq = None,
                                 chunk_size = 100):



        return dark_energy.plot_flat_with_highlighted_sites(self.flat_matrix(self.DEM, to='pos', aa_freq=aa_freq),
                                 active_sites,
                                 pdb_seq, 
                                 chunk_size = chunk_size)