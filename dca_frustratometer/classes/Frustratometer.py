"""Provide the primary functions."""
from pathlib import Path
from .. import pdb
from .. import dca

#Import other modules
from ..utils import _path
from .. import frustration

__all__=['PottsModel']
##################
# PFAM functions #
##################


# Class wrapper
class Frustratometer:

    # @property
    # def sequence_cutoff(self):
    #     return self._sequence_cutoff

    # @sequence_cutoff.setter
    # def sequence_cutoff(self, value):
    #     self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
    #     self._sequence_cutoff = value
    #     self._native_energy = None
    #     self._decoy_fluctuation = {}

    # @property
    # def distance_cutoff(self):
    #     return self._distance_cutoff

    # @distance_cutoff.setter
    # def distance_cutoff(self, value):
    #     self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
    #     self._distance_cutoff = value
    #     self._native_energy = None
    #     self._decoy_fluctuation = {}

    def native_energy(self,sequence=None,ignore_contacts_with_gaps=False):
        if sequence is None:
            sequence=self.sequence
        else:
            self._native_energy=frustration.compute_native_energy(sequence, self.potts_model, self.mask,ignore_contacts_with_gaps)
        if not self._native_energy:
            self._native_energy=frustration.compute_native_energy(sequence, self.potts_model, self.mask,ignore_contacts_with_gaps)
        return self._native_energy

    def sequences_energies(self, sequences, split_couplings_and_fields=False):
        return frustration.compute_sequences_energy(sequences, self.potts_model, self.mask, split_couplings_and_fields)

    def fields_energy(self, sequence=None):
        if sequence is None:
            sequence=self.sequence
        return frustration.compute_fields_energy(sequence, self.potts_model)

    def couplings_energy(self, sequence=None,ignore_contacts_with_gaps=False):
        if sequence is None:
            sequence=self.sequence
        return frustration.compute_couplings_energy(sequence, self.potts_model, self.mask,ignore_contacts_with_gaps)
        
    def decoy_fluctuation(self, sequence=None,kind='singleresidue'):
        if sequence is None:
            sequence=self.sequence
            if kind in self._decoy_fluctuation:
                return self._decoy_fluctuation[kind]
        if kind == 'singleresidue':
            fluctuation = frustration.compute_singleresidue_decoy_energy_fluctuation(sequence, self.potts_model, self.mask)
        elif kind == 'mutational':
            fluctuation = frustration.compute_mutational_decoy_energy_fluctuation(sequence, self.potts_model, self.mask)
        elif kind == 'configurational':
            fluctuation = frustration.compute_configurational_decoy_energy_fluctuation(sequence, self.potts_model, self.mask)
        elif kind == 'contact':
            fluctuation = frustration.compute_contact_decoy_energy_fluctuation(sequence, self.potts_model, self.mask)
        else:
            raise Exception("Wrong kind of decoy generation selected")
        self._decoy_fluctuation[kind] = fluctuation
        return self._decoy_fluctuation[kind]

    def decoy_energy(self, kind='singleresidue'):
        return self.native_energy() + self.decoy_fluctuation(kind=kind)

    def scores(self):
        return frustration.compute_scores(self.potts_model)

    def frustration(self, sequence=None, kind='singleresidue', aa_freq=None, correction=0):
        if sequence is None:
            sequence=self.sequence
        decoy_fluctuation = self.decoy_fluctuation(sequence=sequence,kind=kind)
        if kind == 'singleresidue':
            if aa_freq is not None:
                aa_freq = self.aa_freq
            return frustration.compute_single_frustration(decoy_fluctuation, aa_freq, correction)
        elif kind in ['mutational', 'configurational', 'contact']:
            if aa_freq is not None:
                aa_freq = self.contact_freq
            return frustration.compute_pair_frustration(decoy_fluctuation, aa_freq, correction)

    def plot_decoy_energy(self, kind='singleresidue', method='clustermap'):
        native_energy = self.native_energy()
        decoy_energy = self.decoy_energy(kind=kind)
        if kind == 'singleresidue':
            g = frustration.plot_singleresidue_decoy_energy(decoy_energy, native_energy, method)
            return g

    def roc(self):
        return frustration.compute_roc(self.scores(), self.distance_matrix, self.distance_cutoff)

    def plot_roc(self):
        frustration.plot_roc(self.roc())

    def auc(self):
        """Computes area under the curve of the receiver-operating characteristic.
           Function intended"""
        return frustration.compute_auc(self.roc())

    def vmd(self, single='singleresidue', pair='mutational', aa_freq=None, correction=0, max_connections=100):
        tcl_script = frustration.write_tcl_script(self.pdb_file, self.chain,
                                      self.frustration(single, aa_freq=aa_freq, correction=correction),
                                      self.frustration(pair, aa_freq=aa_freq, correction=correction),
                                      max_connections=max_connections)
        frustration.call_vmd(self.pdb_file, tcl_script)

    def view_frustration(self, single='singleresidue', pair='mutational', aa_freq=None, correction=0, max_connections=100):
        import numpy as np
        import py3Dmol
        pdb_filename = self.pdb_file
        shift=self.init_index_shift+1
        pair_frustration=self.frustration(kind=pair)*np.triu(self.mask)
        residues=np.arange(len(self.sequence))
        r1, r2 = np.meshgrid(residues, residues, indexing='ij')
        sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel()]).T
        minimally_frustrated = sel_frustration[sel_frustration[:, -1] < -0.78]
        frustrated = sel_frustration[sel_frustration[:, -1] > 1]
        
        view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js')
        view.addModel(open(pdb_filename,'r').read(),'pdb')

        view.setBackgroundColor('white')
        view.setStyle({'cartoon':{'color':'white'}})
        
        for i,j,f in frustrated:
            view.addLine({'start':{'chain':'A','resi':[str(i+shift)]},'end':{'chain':'A','resi':[str(j+shift)]},
                        'color':'red', 'dashed':False,'linewidth':3})
        
        for i,j,f in minimally_frustrated:
            view.addLine({'start':{'chain':'A','resi':[str(i+shift)]},'end':{'chain':'A','resi':[str(j+shift)]},
                        'color':'green', 'dashed':False,'linewidth':3})

        view.zoomTo(viewer=(0,0))

        return view

    def view_frustration_distribution(self,kind="mutational"):
        import pandas as pd
        import numpy as np

        if kind in ['mutational', 'configurational']:
            frustration_values=self.frustration(kind=kind)
            #Calculate each residue's center of mass (COM)
            import MDAnalysis as mda
            from scipy.spatial.distance import pdist,squareform

            u = mda.Universe(self.pdb_file)
            protein = u.select_atoms('protein')
            residue_COM_values=protein.center_of_mass(compound="residues")
            COM_distance_matrix = squareform(pdist(residue_COM_values))

            flattened_COM_distance_matrix=COM_distance_matrix[np.triu_indices(COM_distance_matrix.shape[0], k = 1)]
            flattened_frustration_values=frustration_values[np.triu_indices(frustration_values.shape[0], k = 1)]
            frustration_class=[]
            for i in flattened_frustration_values:
                if i<-1:
                    frustration_class.append("Frustrated")
                elif i>0.78:
                    frustration_class.append("Minimally Frustrated")
                else:
                    frustration_class.append("Neutral")
            
            frustration_dataframe=pd.DataFrame(data=np.array([flattened_COM_distance_matrix,flattened_frustration_values,frustration_class]).T,
                                                columns=["Distance_ij","F_ij","Type"])
            frustration_dataframe[["Distance_ij","F_ij"]]=frustration_dataframe[["Distance_ij","F_ij"]].astype(float)

            return frustration_dataframe

            
            



