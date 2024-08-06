import numpy as np
from ..utils import _path
from .. import frustration
from .Frustratometer import Frustratometer
from .Gamma import Gamma
from pydantic import BaseModel, Field, ConfigDict
from pydantic.types import Path
from typing import List,Optional,Union

__all__ = ['AWSEM']

class AWSEMParameters(BaseModel):
    model_config = ConfigDict(extra='ignore', arbitrary_types_allowed=True)
    """Default parameters for AWSEM energy calculations."""
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
    gamma: Union[Path,Gamma] = Field(_path/'data'/'AWSEM_2015.json', description="File or Gamma object containing the Gamma values")
    r_minII: float = Field(6.5, description="Minimum distance for mediated contact potential. (Angstrom)")
    r_maxII: float = Field(9.5, description="Maximum distance for mediated contact potential. (Angstrom)")
    eta_sigma: float = Field(7.0, description="Sharpness of the density-based switching function between protein-mediated and water-mediated contacts.")
    

    #Membrane
    membrane_gamma: Union[Path,Gamma] = Field(_path/'data'/'AWSEM_membrane_2015.json', description="File or Gamma object containing the membrane Gamma values (for membrane proteins)")
    eta_switching: int = Field(10, description="Switching distance for the membrane switching function")

    #Electrostatics
    min_sequence_separation_electrostatics: Optional[int] = Field(1, description="Minimum sequence separation for electrostatics calculation.")
    k_electrostatics: float = Field(17.3636, description="Coefficient for electrostatic interactions. (kJ/mol)")
    electrostatics_screening_length: float = Field(10, description="Screening length for electrostatic interactions. (Angstrom)")


class AWSEM(Frustratometer):
    #Mapping to DCA
    q = 20
    aa_map_awsem_list = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18] #A gap has no energy
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem_list, aa_map_awsem_list, indexing='ij')

    def __init__(self, 
                 pdb_structure,
                 sequence=None,
                 expose_indicator_functions=False,
                 **parameters):
        
        #Set attributes
        p = AWSEMParameters(**parameters)
        if p.min_sequence_separation_contact is None:
            p.min_sequence_separation_contact = 1
        if p.min_sequence_separation_rho is None:
            p.min_sequence_separation_rho = 1
        if p.min_sequence_separation_electrostatics is None:
            p.min_sequence_separation_electrostatics = 1

        for field, value in p:
            setattr(self, field, value)
        
        #Gamma parameters
        if isinstance(p.gamma, Gamma):
            gamma = p.gamma
        elif isinstance(p.gamma, Path):
            gamma = Gamma(p.gamma)
        else:
            raise ValueError("Gamma parameter must be a path or a Gamma object.")
                
        self.gamma=gamma
        self.burial_gamma = gamma['Burial'].T
        self.direct_gamma = gamma['Direct'][0]
        self.protein_gamma = gamma['Protein'][0]
        self.water_gamma = gamma['Water'][0]

        #Structure details
        self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
        if sequence is None:
            self.sequence=pdb_structure.sequence
        else:
            self.sequence=sequence
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
                                                     maximum_contact_distance=None, 
                                                     minimum_sequence_separation = p.min_sequence_separation_rho)
        sequence_mask_contact = frustration.compute_mask(self.distance_matrix, 
                                                     maximum_contact_distance=p.distance_cutoff_contact, 
                                                     minimum_sequence_separation = p.min_sequence_separation_contact)
        
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
            self.indicators=[self.burial_indicator[:,0],self.burial_indicator[:,1],self.burial_indicator[:,2], 
                             self.direct_indicator[:,:,0,0], self.water_indicator[:,:,0,0], self.protein_indicator[:,:,0,0]]


        J_index = np.meshgrid(range(self.N), range(self.N), range(self.q), range(self.q), indexing='ij', sparse=False)
        h_index = np.meshgrid(range(self.N), range(self.q), indexing='ij', sparse=False)

        #Burial energy
        burial_energy = -0.5 * p.k_contact * self.burial_gamma[h_index[1]] * burial_indicator[:, np.newaxis, :]
        self.burial_energy=burial_energy

        #Contact energy
        direct = direct_indicator * self.direct_gamma[J_index[2], J_index[3]]
        water_mediated = water_indicator * self.water_gamma[J_index[2], J_index[3]]
        protein_mediated = protein_indicator  * self.protein_gamma[J_index[2], J_index[3]]
        contact_energy = -p.k_contact * np.array([direct, water_mediated, protein_mediated]) * sequence_mask_contact[np.newaxis, :, :, np.newaxis, np.newaxis]

        # Compute electrostatics
        if p.k_electrostatics!=0:
            self.sequence_cutoff=min(p.min_sequence_separation_electrostatics, p.min_sequence_separation_contact)
            self.distance_cutoff=None
            
            
            electrostatics_mask = frustration.compute_mask(self.distance_matrix, maximum_contact_distance=None, minimum_sequence_separation=p.min_sequence_separation_electrostatics)
            # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
            charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            charges2 = charges[:,np.newaxis]*charges[np.newaxis,:]

            electrostatics_indicator = p.k_electrostatics / (self.distance_matrix + 1E-6) * np.exp(-self.distance_matrix / p.electrostatics_screening_length) * electrostatics_mask
            electrostatics_energy = (charges2[np.newaxis,np.newaxis,:,:]*electrostatics_indicator[:,:,np.newaxis,np.newaxis])

            contact_energy = np.append(contact_energy, electrostatics_energy[np.newaxis,:,:,:,:], axis=0)
        else:
            self.sequence_cutoff=p.min_sequence_separation_contact
            self.distance_cutoff=p.distance_cutoff_contact
        self.mask = frustration.compute_mask(self.distance_matrix, maximum_contact_distance=self.distance_cutoff, minimum_sequence_separation = self.sequence_cutoff)

        self.contact_energy = contact_energy

        # Compute fast properties
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        self.potts_model = {}
        self.potts_model['h'] = -burial_energy.sum(axis=-1)[:, self.aa_map_awsem_list]
        self.potts_model['J'] = -contact_energy.sum(axis=0)[:, :, self.aa_map_awsem_x, self.aa_map_awsem_y]
        
        # Set the gap energy to zero
        self.potts_model['h'][:, 0] = 0
        self.potts_model['J'][:, :, 0, :] = 0
        self.potts_model['J'][:, :, :, 0] = 0
        self._native_energy=None

    def compute_configurational_decoy_statistics(self, n_decoys=4000,aa_freq=None):
        # ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        _AA='ARNDCQEGHILKMFPSTWYV'
        if aa_freq is None:
            seq_index = np.array([_AA.find(aa) for aa in self.sequence])
            N=self.N
        else:
            N=self.N*10
            total = sum(aa_freq)
            probabilities = [freq / total for freq in aa_freq.ravel()]
            seq_index = np.random.choice(a=len(aa_freq), size=N, p=probabilities)
        
        distances = np.triu(self.distance_matrix)
        distances = distances[(distances<self.distance_cutoff_contact) & (distances>0)]

        rho_b = np.expand_dims(self.rho_r, 1) #(n,1)
        rho1 = np.expand_dims(self.rho_r, 0) #(1,n)
        rho2 = np.expand_dims(self.rho_r, 1) #(n,1)

        sigma_water = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0))) #(n,n)
        sigma_protein = 1 - sigma_water #(n,n)

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - distances))) #(c,)
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b)) #(n,3)
           
        charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        electrostatics_indicator = np.exp(-distances / self.electrostatics_screening_length) / distances

        decoy_energies=np.zeros(n_decoys)
        #decoy_data=[None]*n_decoys
        #decoy_data_columns=['decoy_i','rand_i_resno','rand_j_resno','ires_type','jres_type','i_resno','j_resno','rij','rho_i','rho_j','water_energy','burial_energy_i','burial_energy_j','electrostatic_energy','tert_frust_decoy_energies']
        for i in range(n_decoys):
            c=np.random.randint(0,len(distances))
            n1=np.random.randint(0,self.N)
            n2=np.random.randint(0,self.N)
            qi1=np.random.randint(0,N)
            qi2=np.random.randint(0,N)
            q1=seq_index[qi1]
            q2=seq_index[qi2]

            
            burial_energy1 = (-0.5 * self.k_contact * self.burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
            burial_energy2 = (-0.5 * self.k_contact * self.burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
            
            direct = theta[c] * self.direct_gamma[q1, q2]
            water_mediated = sigma_water[n1,n2] * thetaII[c] * self.water_gamma[q1,q2]
            protein_mediated = sigma_protein[n1,n2] * thetaII[c] * self.protein_gamma[q1,q2]
            contact_energy = -self.k_contact * (direct+water_mediated+protein_mediated)
            electrostatics_energy = self.k_electrostatics * electrostatics_indicator[c]*charges[q1]*charges[q2]

            decoy_energies[i]=(burial_energy1+burial_energy2+contact_energy+electrostatics_energy)
            #decoy_data[i]=[i, qi1, qi2, q1, q2, n1, n2, distances[c], self.rho_r[n1], self.rho_r[n2], contact_energy/4.184, burial_energy1/4.184, burial_energy2/4.184, electrostatics_energy/4.184, decoy_energies[i]]
            
        mean_decoy_energy = np.mean(decoy_energies)
        std_decoy_energy = np.std(decoy_energies)
        return mean_decoy_energy, std_decoy_energy
    
    def compute_configurational_energies(self):
        _AA='ARNDCQEGHILKMFPSTWYV'
        seq_index = np.array([_AA.find(aa) for aa in self.sequence])
        distances = np.triu(self.distance_matrix)
        distances = distances[(distances<self.distance_cutoff_contact) & (distances>0)]
        n_contacts=len(distances)

        n = self.distance_matrix.shape[0]  # Assuming self.distance_matrix is defined and square
        tri_upper_indices = np.triu_indices(n, k=1)  # k=1 excludes the diagonal
        valid_pairs = (self.distance_matrix[tri_upper_indices] < self.distance_cutoff_contact) & \
                      (self.distance_matrix[tri_upper_indices] > 0)
        indices1,indices2 = (tri_upper_indices[0][valid_pairs], tri_upper_indices[1][valid_pairs])

        # for n1,n2,c in zip(indices1,indices2,range(n_contacts)):
        #     assert self.distance_matrix[n1,n2] == distances[c]
        
        rho_b = np.expand_dims(self.rho_r, 1) #(n,1)
        rho1 = np.expand_dims(self.rho_r, 0) #(1,n)
        rho2 = np.expand_dims(self.rho_r, 1) #(n,1)

        sigma_water = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0))) #(n,n)
        sigma_protein = 1 - sigma_water #(n,n)

        #Calculate theta and indicators
        theta = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - distances))) # (c,)
        thetaII = 0.25 * (1 + np.tanh(self.eta * (distances - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - distances))) #(c,)
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b)) #(n,3)
           
        charges = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        electrostatics_indicator = np.exp(-distances / self.electrostatics_screening_length) / distances

        # decoy_data_columns=['decoy_i','i_resno','j_resno','ires_type','jres_type','aa1','aa2','rij','rho_i','rho_j','water_energy','burial_energy_i','burial_energy_j','electrostatic_energy','total_energies']
        # decoy_data=[]
        configurational_energies=np.zeros((n,n))
        for c in range(n_contacts):
            n1=indices1[c]
            n2=indices2[c]
            q1=seq_index[n1]
            q2=seq_index[n2]

            burial_energy1 = (-0.5 * self.k_contact * self.burial_gamma[q1] * burial_indicator[n1]).sum(axis=0)
            burial_energy2 = (-0.5 * self.k_contact * self.burial_gamma[q2] * burial_indicator[n2]).sum(axis=0)
            
            direct = theta[c] * self.direct_gamma[q1, q2]
            water_mediated = sigma_water[n1,n2] * thetaII[c] * self.water_gamma[q1,q2]
            protein_mediated = sigma_protein[n1,n2] * thetaII[c] * self.protein_gamma[q1,q2]
            contact_energy = -self.k_contact * (direct+water_mediated+protein_mediated)
            electrostatics_energy = self.k_electrostatics * electrostatics_indicator[c]*charges[q1]*charges[q2]

            energy=(burial_energy1+burial_energy2+contact_energy+electrostatics_energy)
            configurational_energies[n1,n2]=energy
            configurational_energies[n2,n1]=energy
            # decoy_data+=[[c, n1, n2, q1, q2, _AA[q1],_AA[q2], distances[c], self.rho_r[n1], self.rho_r[n2], contact_energy/4.184, burial_energy1/4.184, burial_energy2/4.184, electrostatics_energy/4.184, energy/4.184]]
        # import pandas as pd
        return configurational_energies #, pd.DataFrame(decoy_data, columns=decoy_data_columns)
    
    def configurational_frustration(self,aa_freq=None, correction=0, n_decoys=4000):
        mean_decoy_energy, std_decoy_energy = self.compute_configurational_decoy_statistics(n_decoys=n_decoys,aa_freq=aa_freq)
        return -(self.compute_configurational_energies()-mean_decoy_energy)/(std_decoy_energy+correction)