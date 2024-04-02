"""Provide the primary functions."""
import numpy as np
from ..utils import _path
from .. import frustration
from .Frustratometer import Frustratometer
from pydantic import BaseModel, Field
from pydantic.types import Path
from typing import List,Optional
import numpy as np

__all__ = ['AWSEM']


class AWSEMParameters(BaseModel):
    
    k_contact: float = Field(4.184, description="Coefficient for contact potential. (kJ/mol)")
    
    #Density
    eta: float = Field(5.0, description="Sharpness of the distance-based switching function (Angstrom^-1).")
    rho_0: float = Field(2.6, description="Density cutoff defining buried residues.")
    min_sequence_separation_rho: Optional[int] = Field(2, description="Minimum sequence separation for density calculation.")

    #Burial potential
    burial_kappa: float = Field(4.0, description="Sharpness of the density-based switching function for the burial potential wells")
    burial_ro_min: List[float] = Field([0.0, 3.0, 6.0], description="Minimum radii for burial potential wells. (Angstrom)")
    burial_ro_max: List[float] = Field([3.0, 6.0, 9.0], description="Maximum radii for burial potential wells. (Angstrom)")
    
    #Direct contacts
    min_sequence_separation_contact: Optional[int] = Field(0, description="Minimum sequence separation for contact calculation.")
    distance_cutoff_contact: Optional[float] = Field(9.5, description="Distance cutoff for contact calculation. (Angstrom)")
    r_min: float = Field(4.5, description="Minimum distance for direct contact potential. (Angstrom)")
    r_max: float = Field(6.5, description="Maximum distance for direct contact potential. (Angstrom)")
    
    #Mediated contacts
    r_minII: float = Field(6.5, description="Minimum distance for mediated contact potential. (Angstrom)")
    r_maxII: float = Field(9.5, description="Maximum distance for mediated contact potential. (Angstrom)")
    eta_sigma: float = Field(7.0, description="Sharpness of the density-based switching function between protein-mediated and water-mediated contacts.")

    #Gamma files
    burial_gamma_file: Path = Field(f'{_path}/data/burial_gamma', description="File containing the burial gamma values.")
    direct_gamma_file: Path = Field(f'{_path}/data/gamma_ijm', description="File containing the gamma_ijm values.")
    water_gamma_file: Path = Field(f'{_path}/data/water_gamma_ijm', description="File containing the water gamma_ijm values.")
    protein_gamma_file: Path = Field(f'{_path}/data/protein_gamma_ijm', description="File containing the protein gamma_ijm values.")
    
    #Membrane
    eta_switching: int = Field(10, description="Switching distance for the membrane switching function")

    #Membrane gamma files
    membrane_burial_file: Path = Field(f'{_path}/data/membrane_gamma_ijm', description="File containing the membrane gamma_ijm values.")
    membrane_direct_gamma_file: Path = Field(f'{_path}/data/membrane_gamma_ijm', description="File containing the membrane gamma_ijm values.")
    membrane_water_gamma_file: Path = Field(f'{_path}/data/membrane_gamma_ijm_water', description="File containing the membrane gamma_ijm water values.")
    membrane_protein_gamma_file: Path = Field(f'{_path}/data/membrane_gamma_ijm_protein', description="File containing the membrane gamma_ijm protein values.")

    #Electrostatics
    min_sequence_separation_electrostatics: Optional[int] = Field(1, description="Minimum sequence separation for electrostatics calculation.")
    k_electrostatics: float = Field(17.3636, description="Coefficient for electrostatic interactions. (kJ/mol)")
    electrostatics_screening_length: float = Field(10, description="Screening length for electrostatic interactions. (Angstrom)")

    class Config:
        extra = 'forbid'

class AWSEM(Frustratometer):

    #Mapping to DCA
    q = 20
    aa_map_awsem_list = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18] #A gap is equivalent to Alanine
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem_list, aa_map_awsem_list, indexing='ij')

    def __init__(self, 
                 pdb_structure,
                 expose_indicator_functions=False,
                 **parameters):
        
        #Set attributes
        p = AWSEMParameters(**parameters)
        for field, value in p:
            setattr(self, field, value)
        
        #Gamma files
        self.burial_gamma = np.fromfile(p.burial_gamma_file).reshape(20, 3)
        self.direct_gamma = np.fromfile(p.direct_gamma_file).reshape(2, 20, 20)
        self.water_gamma = np.fromfile(p.water_gamma_file).reshape(2, 20, 20)
        self.protein_gamma = np.fromfile(p.protein_gamma_file).reshape(2, 20, 20)

        #Structure details
        self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
        self.sequence=pdb_structure.sequence
        self.structure=pdb_structure.structure
        self.chain=pdb_structure.chain
        self.pdb_file=pdb_structure.pdb_file
        self.init_index_shift=pdb_structure.init_index_shift
        self.distance_matrix=pdb_structure.distance_matrix
        selection_CB = self.structure.select('name CB or (resname GLY IGL and name CA)')

        resid = selection_CB.getResindices()
        self.resid=resid
        self.N=len(self.resid)
        assert self.N == len(self.sequence), "The pdb is incomplete. Try setting 'repair_pdb=True' when constructing the Structure object."


        sequence_mask_rho = frustration.compute_mask(self.distance_matrix, 
                                                     distance_cutoff=None, 
                                                     sequence_distance_cutoff = p.min_sequence_separation_rho)
        sequence_mask_contact = frustration.compute_mask(self.distance_matrix, 
                                                     distance_cutoff=p.distance_cutoff_contact, 
                                                     sequence_distance_cutoff = p.min_sequence_separation_contact)
        
        self._decoy_fluctuation = {}
        self.minimally_frustrated_threshold=.78

        # Calculate rho
        rho = 0.25 
        rho *= (1 + np.tanh(p.eta * (self.distance_matrix- p.r_min)))
        rho *= (1 + np.tanh(p.eta * (p.r_max - self.distance_matrix)))
        rho *= sequence_mask_rho
        self.rho=rho
        
        #Calculate sigma water
        rho_r = (rho).sum(axis=1)
        self.rho_r=rho_r
        rho_b = np.expand_dims(rho_r, 1)
        rho1 = np.expand_dims(rho_r, 0)
        rho2 = np.expand_dims(rho_r, 1)
        sigma_water = 0.25 * (1 - np.tanh(p.eta_sigma * (rho1 - p.rho_0))) * (1 - np.tanh(p.eta_sigma * (rho2 - p.rho_0)))
        sigma_protein = 1 - sigma_water

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(p.eta * (self.distance_matrix - p.r_min))) * (1 + np.tanh(p.eta * (p.r_max - self.distance_matrix)))
        thetaII = 0.25 * (1 + np.tanh(p.eta * (self.distance_matrix - p.r_minII))) * (1 + np.tanh(p.eta * (p.r_maxII - self.distance_matrix)))
        burial_indicator = np.tanh(p.burial_kappa * (rho_b - p.burial_ro_min)) + np.tanh(p.burial_kappa * (p.burial_ro_max - rho_b))
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
        burial_energy = -0.5 * p.k_contact * self.burial_gamma[h_index[1]] * burial_indicator[:, np.newaxis, :]
        self.burial_energy=burial_energy

        #Contact energy
        direct = direct_indicator * self.direct_gamma[0, J_index[2], J_index[3]]
        water_mediated = water_indicator * self.water_gamma[0, J_index[2], J_index[3]]
        protein_mediated = protein_indicator  * self.protein_gamma[0, J_index[2], J_index[3]]
        contact_energy = -p.k_contact * np.array([direct, water_mediated, protein_mediated]) * sequence_mask_contact[np.newaxis, :, :, np.newaxis, np.newaxis]

        # Compute electrostatics
        if p.k_electrostatics!=0:
            self.sequence_cutoff=min(p.min_sequence_separation_electrostatics, p.min_sequence_separation_contact)
            self.distance_cutoff=None
            
            
            electrostatics_mask = frustration.compute_mask(self.distance_matrix, distance_cutoff=None, sequence_distance_cutoff=p.min_sequence_separation_electrostatics)
            # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
            charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            charges2 = charges[:,np.newaxis]*charges[np.newaxis,:]

            electrostatics_indicator = p.k_electrostatics / (self.distance_matrix + 1E-6) * np.exp(-self.distance_matrix / p.electrostatics_screening_length) * electrostatics_mask
            electrostatics_energy = (charges2[np.newaxis,np.newaxis,:,:]*electrostatics_indicator[:,:,np.newaxis,np.newaxis])

            contact_energy = np.append(contact_energy, electrostatics_energy[np.newaxis,:,:,:,:], axis=0)
        else:
            self.sequence_cutoff=p.min_sequence_separation_contact
            self.distance_cutoff=p.distance_cutoff_contact
        self.mask = frustration.compute_mask(self.distance_matrix, distance_cutoff=self.distance_cutoff, sequence_distance_cutoff = self.sequence_cutoff)

        self.contact_energy = contact_energy

        # Compute fast properties
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        self.potts_model = {}
        self.potts_model['h'] = -burial_energy.sum(axis=-1)[:, self.aa_map_awsem_list]
        self.potts_model['J'] = -contact_energy.sum(axis=0)[:, :, self.aa_map_awsem_x, self.aa_map_awsem_y]
        self._native_energy=None