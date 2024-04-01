"""Provide the primary functions."""
from pathlib import Path
from .. import pdb
from .. import dca

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    def native_energy(self,sequence:str = None,ignore_contacts_with_gaps:bool = False):
        if sequence is None:
            sequence=self.sequence
        else:
            return frustration.compute_native_energy(sequence, self.potts_model, self.mask,ignore_contacts_with_gaps)
        if not self._native_energy:
            self._native_energy=frustration.compute_native_energy(sequence, self.potts_model, self.mask,ignore_contacts_with_gaps)
        return self._native_energy

    def sequences_energies(self, sequences:np.array, split_couplings_and_fields:bool = False):
        return frustration.compute_sequences_energy(sequences, self.potts_model, self.mask, split_couplings_and_fields)

    def fields_energy(self, sequence=None):
        if sequence is None:
            sequence=self.sequence
        return frustration.compute_fields_energy(sequence, self.potts_model)

    def couplings_energy(self, sequence:str = None,ignore_contacts_with_gaps:bool = False):
        if sequence is None:
            sequence=self.sequence
        return frustration.compute_couplings_energy(sequence, self.potts_model, self.mask,ignore_contacts_with_gaps)
        
    def decoy_fluctuation(self, sequence:str = None,kind:str = 'singleresidue',mask:np.array = None):
        if sequence is None:
            sequence=self.sequence
            if kind in self._decoy_fluctuation:
                return self._decoy_fluctuation[kind]
        if not isinstance(mask, np.ndarray):
            mask=self.mask
        if kind == 'singleresidue':
            fluctuation = frustration.compute_singleresidue_decoy_energy_fluctuation(sequence, self.potts_model, mask)
        elif kind == 'mutational':
            fluctuation = frustration.compute_mutational_decoy_energy_fluctuation(sequence, self.potts_model, mask)
        elif kind == 'configurational':
            fluctuation = frustration.compute_configurational_decoy_energy_fluctuation(sequence, self.potts_model, mask)
        elif kind == 'contact':
            fluctuation = frustration.compute_contact_decoy_energy_fluctuation(sequence, self.potts_model, mask)
        else:
            raise Exception("Wrong kind of decoy generation selected")
        self._decoy_fluctuation[kind] = fluctuation
        return self._decoy_fluctuation[kind]

    def decoy_energy(self, kind:str = 'singleresidue'):
        return self.native_energy() + self.decoy_fluctuation(kind=kind)

    def scores(self):
        return frustration.compute_scores(self.potts_model)

    def frustration(self, sequence:str = None, kind:str = 'singleresidue', mask:np.array = None, aa_freq:np.array = None, correction:int = 0):
        if sequence is None:
            sequence=self.sequence
        if not isinstance(mask, np.ndarray):
            mask=self.mask
        decoy_fluctuation = self.decoy_fluctuation(sequence=sequence,kind=kind, mask=mask)
        if kind == 'singleresidue':
            if aa_freq is None:
                aa_freq = self.aa_freq
            return frustration.compute_single_frustration(decoy_fluctuation, aa_freq, correction)
        elif kind in ['mutational', 'configurational', 'contact']:
            if aa_freq is None:
                aa_freq = self.contact_freq
            return frustration.compute_pair_frustration(decoy_fluctuation, aa_freq, correction)

    def plot_decoy_energy(self, kind:str = 'singleresidue', method:str = 'clustermap'):
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

    def vmd(self, single:str = 'singleresidue', pair:str = 'mutational', aa_freq:np.array = None, correction:int = 0, max_connections:int = 100):
        tcl_script = frustration.write_tcl_script(self.pdb_file, self.chain,
                                      self.frustration(single, aa_freq=aa_freq, correction=correction),
                                      self.frustration(pair, aa_freq=aa_freq, correction=correction),
                                      max_connections=max_connections)
        frustration.call_vmd(self.pdb_file, tcl_script)

    def view_frustration(self, single:str = 'singleresidue', pair:str = 'mutational', aa_freq:np.array = None, correction:int = 0, max_connections:int = 100):
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

    def view_frustration_pair_distribution(self,kind:str ="singleresidue",include_long_range_contacts:bool =True):
        def frustration_type(x):
            if x>.78:
                frustration_class="Minimally Frustrated"
            elif x<-1:
                frustration_class="Frustrated"
            else:
                frustration_class="Neutral"
            return frustration_class
        ###
        #Ferrerio et al. (2007) pair distribution analysis included long-range contacts.
        if include_long_range_contacts==True:
            mask=frustration.compute_mask(self.distance_matrix, distance_cutoff=None, sequence_distance_cutoff = 2)
        else:
            mask=self.mask

        frustration_values=self.frustration(kind=kind,mask=mask)
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
            frustration_values=frustration_values[i_index,j_index]
            
            frustration_dataframe=pd.DataFrame(data=np.array([contact_pairs,flattened_midpoint_ca_distance_matrix,frustration_values]).T,
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
        with sns.plotting_context("poster"):
            plt.figure(figsize=(10,5))

            sns.lineplot(data=frustration_distribution_dataframe,x="Distances",y="Minimally Frustrated",color="green",label="Minimally Frustrated")
            sns.lineplot(data=frustration_distribution_dataframe,x="Distances",y="Frustrated",color="red",label="Frustrated")
            sns.lineplot(data=frustration_distribution_dataframe,x="Distances",y="Neutral",color="gray",label="Neutral")
            plt.xlabel("Distance (A)");plt.ylabel("g(r)")
            plt.xlim([0,20])
            plt.legend(loc="best")
            plt.show()

    def view_frustration_histogram(self,kind:str = "singleresidue"):
        def frustration_type(x):
            if x>.78:
                frustration_class="Minimally Frustrated"
            elif x<-1:
                frustration_class="Frustrated"
            else:
                frustration_class="Neutral"
            return frustration_class

        frustration_values=self.frustration(kind=kind)
        cb_distance_matrix=self.distance_matrix
        i_index,j_index=np.triu_indices(cb_distance_matrix.shape[0], k = 1)

        flattened_cb_distance_matrix=cb_distance_matrix[i_index,j_index]
        contact_pairs=list((zip(i_index,j_index)))
        ###
        if kind=="singleresidue":
            frustration_dataframe=pd.DataFrame(data=np.array([range(0,len(frustration_values)),frustration_values]).T,
                                    columns=["Residue Index","F_i"])
            #Classify frustration type
            frustration_dataframe["F_i_Type"]=frustration_dataframe["F_i"].apply(lambda x: frustration_type(x))

            #Plot histogram of all frustration values.
            with sns.plotting_context("poster"):
                plt.figure(figsize=(10,5))

                g=sns.histplot(data=frustration_dataframe,x="F_i",bins=20)
                ymin, ymax = g.get_ylim()
                g.vlines(x=[-1, .78], ymin=ymin, ymax=ymax, colors=['gray', 'gray'], ls='--', lw=2)
                plt.title(f"N={len(frustration_dataframe)}")
                plt.show()
            ###
            print(f"{((len(frustration_dataframe.loc[frustration_dataframe['F_i_Type']=='Minimally Frustrated'])/len(frustration_dataframe))*100):.2f}% Residues are Minimally Frustrated")
            print(f"{((len(frustration_dataframe.loc[frustration_dataframe['F_i_Type']=='Frustrated'])/len(frustration_dataframe))*100):.2f}% Residues are Frustrated")
            print(f"{((len(frustration_dataframe.loc[frustration_dataframe['F_i_Type']=='Neutral'])/len(frustration_dataframe))*100):.2f}% Residues are Neutral")

        elif kind in ['mutational', 'configurational']:
            frustration_values=frustration_values[i_index,j_index]
            
            frustration_dataframe=pd.DataFrame(data=np.array([contact_pairs,flattened_cb_distance_matrix,frustration_values]).T,
                                                columns=["i,j Pair","Original_Distance_ij","F_ij"])
            frustration_dataframe=frustration_dataframe.dropna(subset=["F_ij"])
            #Classify frustration type
            frustration_dataframe["F_ij_Type"]=frustration_dataframe["F_ij"].apply(lambda x: frustration_type(x))
            frustration_dataframe["Contact_Type"]=np.where(frustration_dataframe["Original_Distance_ij"]<6.5,"Direct","Water-Mediated")
            #Plot histogram of all frustration values.
            with sns.plotting_context("poster"):
                fig,axes=plt.subplots(1,2,figsize=(15,5),sharex=True)

                g=sns.histplot(data=frustration_dataframe.loc[frustration_dataframe["Contact_Type"]=="Direct"],x="F_ij",bins=20,ax=axes[0])
                ymin, ymax = g.get_ylim()
                g.vlines(x=[-1, .78], ymin=ymin, ymax=ymax, colors=['gray', 'gray'], ls='--', lw=2)
                axes[0].title.set_text(f"Direct Contacts (N={len(frustration_dataframe.loc[frustration_dataframe['Contact_Type']=='Direct'])})")
                ###
                g=sns.histplot(data=frustration_dataframe.loc[frustration_dataframe["Contact_Type"]=="Water-Mediated"],x="F_ij",bins=20,ax=axes[1])
                ymin, ymax = g.get_ylim()
                g.vlines(x=[-1, .78], ymin=ymin, ymax=ymax, colors=['gray', 'gray'], ls='--', lw=2)
                axes[1].title.set_text(f"Protein- & Water-\nMediated Contacts (N={len(frustration_dataframe.loc[frustration_dataframe['Contact_Type']=='Water-Mediated'])})")

                plt.tight_layout()
                plt.show()
            ###
            direct_contact_frustration_dataframe=frustration_dataframe.loc[frustration_dataframe["Contact_Type"]=="Direct"]
            print(f"{((len(direct_contact_frustration_dataframe.loc[direct_contact_frustration_dataframe['F_ij_Type']=='Minimally Frustrated'])/len(direct_contact_frustration_dataframe))*100):.2f}% Direct Contacts are Minimally Frustrated")
            print(f"{((len(direct_contact_frustration_dataframe.loc[direct_contact_frustration_dataframe['F_ij_Type']=='Frustrated'])/len(direct_contact_frustration_dataframe))*100):.2f}% Direct Contacts are Frustrated")
            print(f"{((len(direct_contact_frustration_dataframe.loc[direct_contact_frustration_dataframe['F_ij_Type']=='Neutral'])/len(direct_contact_frustration_dataframe))*100):.2f}% Direct Contacts are Neutral")





