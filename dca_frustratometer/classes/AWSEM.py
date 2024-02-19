"""Provide the primary functions."""
import scipy.spatial.distance as sdist
import numpy as np
from ..utils import _path
from .. import frustration
import typing
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

    # The following files have a different aminoacid order than the expected by the Frustratometer class.
    # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    #  'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    q = 20

    #Map of AWSEM order to Frustratometer order
    aa_map_awsem_list = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18] #A gap is equivalent to Alanine
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem_list, aa_map_awsem_list, indexing='ij')

    electrostatics_screening_length = 10
    
    def __init__(self, 
                 pdb_structure,
                 distance_cutoff_contact = None, #9.5 for frustratometer    
                 min_sequence_separation_rho = 2,
                 min_sequence_separation_contact = 10,
                 min_sequence_separation_electrostatics = 3,
                 expose_indicator_functions=False,
                 electrostatics = False,
                 burial_gamma_file = f'{_path}/data/burial_gamma',
                 gamma_ijm_file = f'{_path}/data/gamma_ijm',
                 water_gamma_ijm_file = f'{_path}/data/water_gamma_ijm',
                 protein_gamma_ijm_file =f'{_path}/data/protein_gamma_ijm',
                 ):
        
        self.burial_gamma = np.fromfile(burial_gamma_file).reshape(20, 3)
        self.gamma_ijm = np.fromfile(gamma_ijm_file).reshape(2, 20, 20)
        self.water_gamma_ijm = np.fromfile(water_gamma_ijm_file).reshape(2, 20, 20)
        self.protein_gamma_ijm = np.fromfile(protein_gamma_ijm_file).reshape(2, 20, 20)


        self.sequence=pdb_structure.sequence
        self.structure=pdb_structure.structure
        self.chain=pdb_structure.chain
        self.pdb_file=pdb_structure.pdb_file
        self.init_index_shift=pdb_structure.init_index_shift

        self.distance_matrix=pdb_structure.distance_matrix
        self.distance_cutoff_contact=distance_cutoff_contact
        self.sequence_cutoff=None
        self.distance_cutoff=None

        self.minimally_frustrated_threshold=.78
        
        self._decoy_fluctuation = {}
        #self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self.mask = frustration.compute_mask(self.distance_matrix, distance_cutoff=self.distance_cutoff_contact, sequence_distance_cutoff = min_sequence_separation_contact)
        selection_CB = self.structure.select('name CB or (resname GLY IGL and name CA)')
        resid = selection_CB.getResindices()
        self.resid=resid
        self.N=len(self.resid)

        assert self.N == len(self.sequence), "The pdb is incomplete. Try setting 'repair_pdb=True' when constructing the Structure object."
        #resname = [self.gamma_se_map_3_letters[aa] for aa in selection_CB.getResnames()]

        #Calculate sequence mask
        sequence_mask_rho = frustration.compute_mask(self.distance_matrix, distance_cutoff=None, sequence_distance_cutoff = min_sequence_separation_rho)#abs(np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= min_sequence_separation_rho
        sequence_mask_contact = frustration.compute_mask(self.distance_matrix, distance_cutoff=self.distance_cutoff_contact, sequence_distance_cutoff = min_sequence_separation_contact)#abs(np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= min_sequence_separation_contact

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

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.eta * (self.distance_matrix - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - self.distance_matrix)))
        thetaII = 0.25 * (1 + np.tanh(self.eta * (self.distance_matrix - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - self.distance_matrix)))
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b))
        direct_indicator = theta[:, :, np.newaxis, np.newaxis]
        water_indicator = thetaII[:, :, np.newaxis, np.newaxis] * sigma_water[:, :, np.newaxis, np.newaxis]
        protein_indicator = thetaII[:, :, np.newaxis, np.newaxis] * sigma_protein[:, :, np.newaxis, np.newaxis]
        
        if expose_indicator_functions:
            self.burial_indicator = burial_indicator
            self.direct_indicator = direct_indicator
            self.water_indicator = water_indicator
            self.protein_indicator = protein_indicator


        J_index = np.meshgrid(range(self.N), range(self.N), range(self.q), range(self.q), indexing='ij', sparse=False)
        h_index = np.meshgrid(range(self.N), range(self.q), indexing='ij', sparse=False)

        #Burial energy
        burial_energy = -0.5 * self.k_contact * self.burial_gamma[h_index[1]] * burial_indicator[:, np.newaxis, :]
        self.burial_energy=burial_energy

        #Contact energy
        direct = direct_indicator * self.gamma_ijm[0, J_index[2], J_index[3]]
        water_mediated = water_indicator * self.water_gamma_ijm[0, J_index[2], J_index[3]]
        protein_mediated = protein_indicator  * self.protein_gamma_ijm[0, J_index[2], J_index[3]]
        contact_energy = -self.k_contact * np.array([direct, water_mediated, protein_mediated]) * sequence_mask_contact[np.newaxis, :, :, np.newaxis, np.newaxis]

        # Compute electrostatics
        if electrostatics:
            electrostatics_mask = frustration.compute_mask(self.distance_matrix, distance_cutoff=None, sequence_distance_cutoff=min_sequence_separation_electrostatics)
            # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
            charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            charges2 = charges[:,np.newaxis]*charges[np.newaxis,:]

            electrostatics_indicator = 4.15 * 4.184 / (self.distance_matrix + 1E-6) *\
                                       np.exp(-self.distance_matrix / self.electrostatics_screening_length) * electrostatics_mask
            electrostatics_energy = (charges2[np.newaxis,np.newaxis,:,:]*electrostatics_indicator[:,:,np.newaxis,np.newaxis])

            contact_energy = np.append(contact_energy, electrostatics_energy[np.newaxis,:,:,:,:], axis=0)
        self.contact_energy = contact_energy

        # Compute fast properties
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.potts_model = {}
        self.potts_model['h'] = -burial_energy.sum(axis=-1)[:, self.aa_map_awsem_list]
        self.potts_model['J'] = -contact_energy.sum(axis=0)[:, :, self.aa_map_awsem_x, self.aa_map_awsem_y]
        self._native_energy=None