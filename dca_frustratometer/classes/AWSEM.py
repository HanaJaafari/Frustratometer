"""Provide the primary functions."""
import prody
import scipy.spatial.distance as sdist
import numpy as np
from ..utils import _path

#Import other modules
from .. import pdb
from .. import frustration
from .DCA import PottsModel

__all__ = ['AWSEMFrustratometer']

##################
# PFAM functions #
##################


class AWSEMFrustratometer(PottsModel):
    # AWSEM parameters
    r_min = .45
    r_max = .65
    r_minII = .65
    r_maxII = .95
    eta = 50  # eta actually has unit of nm^-1.
    eta_sigma = 7.0
    rho_0 = 2.6

    min_sequence_separation_rho = 2
    min_sequence_separation_contact = 10  # means j-i > 9

    eta_switching = 10
    k_contact = 4.184
    burial_kappa = 4.0
    burial_ro_min = [0.0, 3.0, 6.0]
    burial_ro_max = [3.0, 6.0, 9.0]

    gamma_se_map_3_letters = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                              'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                              'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                              'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19,
                              'NGP': 0, 'IPR': 14, 'IGL': 7}
    burial_gamma = np.fromfile(f'{_path}/data/burial_gamma').reshape(20, 3)
    gamma_ijm = np.fromfile(f'{_path}/data/gamma_ijm').reshape(2, 20, 20)
    water_gamma_ijm = np.fromfile(f'{_path}/data/water_gamma_ijm').reshape(2, 20, 20)
    protein_gamma_ijm = np.fromfile(f'{_path}/data/protein_gamma_ijm').reshape(2, 20, 20)
    q = 20
    aa_map_awsem = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18]
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem, aa_map_awsem, indexing='ij')

    def __init__(self,
                 pdb_file,
                 chain=None,
                 sequence_cutoff=None):
        self._pdb_file = pdb_file
        self._chain = chain
        self._sequence = pdb.get_sequence(self.pdb_file, self.chain)
        self.structure = prody.parsePDB(self.pdb_file)
        selection_CB = self.structure.select('name CB or (resname GLY IGL and name CA)')
        resid = selection_CB.getResindices()
        self.N = len(resid)
        resname = [self.gamma_se_map_3_letters[aa] for aa in selection_CB.getResnames()]

        coords = selection_CB.getCoords()
        r = sdist.squareform(sdist.pdist(coords)) / 10
        distance_mask = ((r < 1) - np.eye(len(r)))
        sequence_mask_rho = abs(np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= self.min_sequence_separation_rho
        sequence_mask_contact = abs(
            np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= self.min_sequence_separation_contact
        mask = ((r < 1) - np.eye(len(r)))
        rho = 0.25 * (1 + np.tanh(self.eta * (r - self.r_min))) * \
              (1 + np.tanh(self.eta * (self.r_max - r))) * sequence_mask_rho
        rho_r = (rho).sum(axis=1)
        rho_b = np.expand_dims(rho_r, 1)
        rho1 = np.expand_dims(rho_r, 0)
        rho2 = np.expand_dims(rho_r, 1)
        sigma_water = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (
                1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0)))
        sigma_protein = 1 - sigma_water
        theta = 0.25 * (1 + np.tanh(self.eta * (r - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - r)))
        thetaII = 0.25 * (1 + np.tanh(self.eta * (r - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - r)))
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + \
                           np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b))
        J_index = np.meshgrid(range(self.N), range(self.N), range(self.q), range(self.q), indexing='ij', sparse=False)
        h_index = np.meshgrid(range(self.N), range(self.q), indexing='ij', sparse=False)

        burial_energy = -0.5 * self.k_contact * self.burial_gamma[h_index[1]] * burial_indicator[:, np.newaxis, :]
        direct = self.gamma_ijm[0, J_index[2], J_index[3]] * theta[:, :, np.newaxis, np.newaxis]

        water_mediated = thetaII[:, :, np.newaxis, np.newaxis] * sigma_water[:, :, np.newaxis, np.newaxis] * \
                         self.water_gamma_ijm[0, J_index[2], J_index[3]]
        protein_mediated = thetaII[:, :, np.newaxis, np.newaxis] * sigma_protein[:, :, np.newaxis, np.newaxis] * \
                           self.protein_gamma_ijm[0, J_index[2], J_index[3]]
        contact_energy = -self.k_contact * np.array([direct, water_mediated, protein_mediated]) * \
                         sequence_mask_contact[np.newaxis, :, :, np.newaxis, np.newaxis]

        # Set parameters
        self._distance_cutoff = 10
        self._sequence_cutoff = 2

        # Compute fast properties
        self.distance_matrix = r * 10
        self.potts_model = {}
        self.potts_model['h'] = -burial_energy.sum(axis=-1)[:, self.aa_map_awsem]
        self.potts_model['J'] = -contact_energy.sum(axis=0)[:, :, self.aa_map_awsem_x, self.aa_map_awsem_y]
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        #
        # def __init__(self,
        #              pdb_file: str,
        #              chain: str,
        #              potts_model_file: str,
        #              sequence_cutoff: typing.Union[float, None],
        #              distance_cutoff: typing.Union[float, None],
        #              distance_matrix_method='minimum'
        #              ):
        #     self.pdb_file = pdb_file
        #     self.chain = chain
        #     self.sequence = get_protein_sequence_from_pdb(self.pdb_file, self.chain)
        #
        #     # Set parameters
        #     self._potts_model_file = potts_model_file
        #     self._sequence_cutoff = sequence_cutoff
        #     self._distance_cutoff = distance_cutoff
        #     self._distance_matrix_method = distance_matrix_method
        #
        #     # Compute fast properties
        #     self.distance_matrix = get_distance_matrix_from_pdb(self.pdb_file, self.chain, self.distance_matrix_method)
        #     self.potts_model = load_potts_model(self.potts_model_file)
        #     self.aa_freq = compute_aa_freq(self.sequence)
        #     self.contact_freq = compute_contact_freq(self.sequence)
        #     self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        #
        #     # Initialize slow properties
        #     self._native_energy = None
        #     self._decoy_fluctuation = {}
