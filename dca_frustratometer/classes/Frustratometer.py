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
        minimally_frustrated = sel_frustration[sel_frustration[:, -1] > 1]
        frustrated = sel_frustration[sel_frustration[:, -1] < -.78]
        
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

    def view_frustration_pair_distribution(self,kind="mutational"):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns

        def frustration_type(x):
            if x>.78:
                frustration_class="Minimally Frustrated"
            elif x<-1:
                frustration_class="Frustrated"
            else:
                frustration_class="Neutral"
            return frustration_class
        ###
        frustration_values=self.frustration(kind=kind)

        ca_distance_matrix=pdb.get_distance_matrix(self.pdb_file,self.chain,method="CA")
        i_index,j_index=np.triu_indices(ca_distance_matrix.shape[0], k = 1)

        midpoint_ca_distance_matrix=ca_distance_matrix/2
        flattened_ca_distance_matrix=ca_distance_matrix[i_index,j_index]
        flattened_midpoint_ca_distance_matrix=midpoint_ca_distance_matrix[i_index,j_index]
        contact_pairs=list((zip(i_index,j_index)))
        ###
        if kind=="singleresidue":
            residue_i_frustration=[];residue_j_frustration=[]
            for pair in contact_pairs:
                residue_i_frustration.append(frustration_values[pair[0]])
                residue_j_frustration.append(frustration_values[pair[1]])

            frustration_dataframe=pd.DataFrame(data=np.array([contact_pairs,flattened_ca_distance_matrix,residue_i_frustration,residue_j_frustration]).T,
                                    columns=["i,j Pair","Distance_ij","F_i","F_j"])
            #Classify frustration type
            frustration_dataframe["F_i_Type"]=frustration_dataframe["F_i"].apply(lambda x: frustration_type(x))
            frustration_dataframe["F_j_Type"]=frustration_dataframe["F_j"].apply(lambda x: frustration_type(x))
            #Keep only residues with same frustration classes
            frustration_dataframe=frustration_dataframe.loc[frustration_dataframe["F_i_Type"]==frustration_dataframe["F_j_Type"]]
            frustration_dataframe["F_ij_Type"]=frustration_dataframe["F_j_Type"]

        elif kind in ['mutational', 'configurational']:
            flattened_frustration_values=frustration_values[i_index,j_index]
            
            frustration_dataframe=pd.DataFrame(data=np.array([contact_pairs,flattened_midpoint_ca_distance_matrix,flattened_frustration_values]).T,
                                                columns=["i,j Pair","Distance_ij","F_ij"])
            frustration_dataframe=frustration_dataframe.dropna(subset=["F_ij"])
            #Classify frustration type
            frustration_dataframe["F_ij_Type"]=frustration_dataframe["F_ij"].apply(lambda x: frustration_type(x))
        ###
        maximum_distance=frustration_dataframe['Distance_ij'].max()
        #Bin by residue pair distances
        frustration_dataframe['bin'] = pd.cut(frustration_dataframe['Distance_ij'], bins=np.arange(0,maximum_distance,.1), labels=[f'{l}-{l+.1}' for l in np.arange(0,maximum_distance-.1,.1)])
        frustration_dataframe=frustration_dataframe.dropna(subset=["bin"]).sort_values('bin')

        frustration_distribution_dictionary={"Distances":[],"Minimally Frustrated":[],"Frustrated":[],"Neutral":[]}
        for bin_value in frustration_dataframe['bin'].unique():
            lower_bin_value=float(bin_value.split("-")[0])
            upper_bin_value=float(bin_value.split("-")[1])
            frustration_distribution_dictionary["Distances"].append(lower_bin_value)
            
            for frustration_class in ["Minimally Frustrated","Frustrated","Neutral"]:
                bin_select_frustration_values_dataframe=frustration_dataframe.loc[((frustration_dataframe["bin"]==bin_value) & (frustration_dataframe["F_ij_Type"]==frustration_class))]
                hollow_sphere_volume=((4*np.pi)/3)*((upper_bin_value**3)-(lower_bin_value**3))
                distribution_function=len(bin_select_frustration_values_dataframe)/hollow_sphere_volume
                frustration_distribution_dictionary[frustration_class].append(distribution_function)
        ###
        frustration_distribution_dataframe=pd.DataFrame.from_dict(frustration_distribution_dictionary)

        plt.figure(figsize=(10,5))

        sns.lineplot(data=frustration_distribution_dataframe,x="Distances",y="Minimally Frustrated",color="green",label="Minimally Frustrated")
        sns.lineplot(data=frustration_distribution_dataframe,x="Distances",y="Frustrated",color="red",label="Frustrated")
        sns.lineplot(data=frustration_distribution_dataframe,x="Distances",y="Neutral",color="gray",label="Neutral")
        plt.xlabel("Distance (A)");plt.ylabel("g(r)")
        plt.xlim([0,20])
        plt.legend(loc="best")
        plt.show()

            
            



