"""Provide the primary functions."""
import prody
import typing
import scipy.spatial.distance as sdist
import numpy as np
from ..utils import _path

#Import other modules
from .. import pdb
from .. import frustration
# from .DCA import PottsModel
from .Frustratometer import Frustratometer

__all__ = ['AWSEMFrustratometer']

##################
# PFAM functions #
##################


class AWSEMFrustratometer(Frustratometer):
    # AWSEM parameters
    r_min = 4.5 # A
    r_minII = r_max = 6.5 # A
    r_maxII = 9.5 # A
    
    eta = 5  # A^-1.
    eta_sigma = 7.0
    rho_0 = 2.6

    min_sequence_separation_rho = 2
    min_sequence_separation_contact = 10  # means j-i > 9

    eta_switching = 10 
    k_contact = 4.184 #kJ
    burial_kappa = 4.0
    burial_ro_min = [0.0, 3.0, 6.0]
    burial_ro_max = [3.0, 6.0, 9.0]

    burial_gamma = np.fromfile(f'{_path}/data/burial_gamma').reshape(20, 3)
    gamma_ijm = np.fromfile(f'{_path}/data/gamma_ijm').reshape(2, 20, 20)
    water_gamma_ijm = np.fromfile(f'{_path}/data/water_gamma_ijm').reshape(2, 20, 20)
    protein_gamma_ijm = np.fromfile(f'{_path}/data/protein_gamma_ijm').reshape(2, 20, 20)
    q = 20

    aa_map_awsem_list = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18] #A gap is equivalent to Alanine
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem_list, aa_map_awsem_list, indexing='ij') 
    
    @classmethod
    def from_full_pdb(cls,pdb_file:str, chain:str=None, sequence:str= None,
                 sequence_cutoff:typing.Union[float, None] = 2,
                 distance_cutoff:typing.Union[float, None] = 10,
                 distance_matrix_method:str="CB"):
        object=cls(pdb_file,chain,sequence,sequence_cutoff,distance_cutoff,
                distance_matrix_method)

    @classmethod
    def from_pdb_segment(cls,pdb_file:str, chain:str=None, sequence:str= None,
                        sequence_cutoff:typing.Union[float, None] = 2,
                        distance_cutoff:typing.Union[float, None] = 10,
                        distance_matrix_method:str="CB",subsequence_initial_boundary: int=None,
                        subsequence_final_boundary: int=None):
        object=cls(pdb_file,chain,sequence,sequence_cutoff,distance_cutoff,
                distance_matrix_method,subsequence_initial_boundary,subsequence_final_boundary)

    def __init__(self, 
                 pdb_file:str, chain:str=None, sequence:str= None,
                 sequence_cutoff:typing.Union[float, None] = None,
                 distance_cutoff:typing.Union[float, None] = None,
                 distance_matrix_method:str="CB",subsequence_initial_boundary: int=None,
                 subsequence_final_boundary: int=None):
        super().__init__(pdb_file, chain,sequence,sequence_cutoff,distance_cutoff,distance_matrix_method)

        self.structure = prody.parsePDB(self.pdb_file, chain=chain)
        selection_CB = self.structure.select('name CB or (resname GLY IGL and name CA)')
        resid = selection_CB.getResindices()
        self.resid=resid

        assert len(resid) == len(self._sequence), 'The pdb is incomplete'
        #resname = [self.gamma_se_map_3_letters[aa] for aa in selection_CB.getResnames()]

        #Calculate sequence mask
        self.subsequence_initial_boundary=subsequence_initial_boundary
        self.subsequence_final_boundary=subsequence_final_boundary

        sequence_mask_rho = abs(np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= self.min_sequence_separation_rho
        if self.subsequence_initial_boundary!=None and self.subsequence_final_boundary!=None:
            sequence_mask_rho[self.subsequence_final_boundary+1:,:]=False
            sequence_mask_rho[:,self.subsequence_final_boundary+1:]=False
            if self.subsequence_initial_boundary!=0:
                sequence_mask_rho[:self.subsequence_initial_boundary,:]=False
                sequence_mask_rho[:,:self.subsequence_initial_boundary]=False

        sequence_mask_contact = abs(np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= self.min_sequence_separation_contact
        if self.subsequence_initial_boundary!=None and self.subsequence_final_boundary!=None:
            sequence_mask_contact[self.subsequence_final_boundary+1:,:]=False
            sequence_mask_contact[:,self.subsequence_final_boundary+1:]=False
            if self.subsequence_initial_boundary!=0:
                sequence_mask_contact[:self.subsequence_initial_boundary,:]=False
                sequence_mask_contact[:,:self.subsequence_initial_boundary]=False
        self.sequence_mask_rho=sequence_mask_rho
        self.sequence_mask_contact=sequence_mask_contact
        # Calculate rho
        rho = 0.25 
        rho *= (1 + np.tanh(self.eta * (self.distance_matrix- self.r_min))) 
        rho *= (1 + np.tanh(self.eta * (self.r_max - self.distance_matrix)))
        rho *= sequence_mask_rho
        self.rho=rho
        
        #Calculate sigma water
        rho_r = (rho).sum(axis=1)
        self.rho_r=rho_r
        rho_b = np.expand_dims(rho_r, 1)
        rho1 = np.expand_dims(rho_r, 0)
        rho2 = np.expand_dims(rho_r, 1)
        sigma_water = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (
                1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0)))
        sigma_protein = 1 - sigma_water
        theta = 0.25 * (1 + np.tanh(self.eta * (self.distance_matrix - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - self.distance_matrix)))
        thetaII = 0.25 * (1 + np.tanh(self.eta * (self.distance_matrix - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - self.distance_matrix)))
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + \
                           np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b))
        J_index = np.meshgrid(range(self.N), range(self.N), range(self.q), range(self.q), indexing='ij', sparse=False)
        h_index = np.meshgrid(range(self.N), range(self.q), indexing='ij', sparse=False)

        if self.subsequence_initial_boundary!=None and self.subsequence_final_boundary!=None:
            burial_indicator[self.subsequence_final_boundary+1:,:]=0
            if self.subsequence_initial_boundary!=0:
               burial_indicator[:self.subsequence_initial_boundary,:]=0 

        self.burial_indicator=burial_indicator
        burial_energy = -0.5 * self.k_contact * self.burial_gamma[h_index[1]] * burial_indicator[:, np.newaxis, :]
        self.burial_energy=burial_energy
        direct = self.gamma_ijm[0, J_index[2], J_index[3]] * theta[:, :, np.newaxis, np.newaxis]

        water_mediated = thetaII[:, :, np.newaxis, np.newaxis] * sigma_water[:, :, np.newaxis, np.newaxis] * \
                         self.water_gamma_ijm[0, J_index[2], J_index[3]]
        protein_mediated = thetaII[:, :, np.newaxis, np.newaxis] * sigma_protein[:, :, np.newaxis, np.newaxis] * \
                           self.protein_gamma_ijm[0, J_index[2], J_index[3]]
        contact_energy = -self.k_contact * np.array([direct, water_mediated, protein_mediated]) * \
                         sequence_mask_contact[np.newaxis, :, :, np.newaxis, np.newaxis]

        # Compute fast properties
        self.potts_model = {}
        self.potts_model['h'] = -burial_energy.sum(axis=-1)[:, self.aa_map_awsem_list]
        self.potts_model['J'] = -contact_energy.sum(axis=0)[:, :, self.aa_map_awsem_x, self.aa_map_awsem_y]