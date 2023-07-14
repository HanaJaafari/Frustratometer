"""Provide the primary functions."""
from pathlib import Path
from .. import pdb
from .. import dca

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.spatial.distance as sdist

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
            self._native_energy=frustration.compute_native_energy(sequence, self.potts_model, self.mask,ignore_contacts_with_gaps)
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
            if aa_freq is not None:
                aa_freq = self.aa_freq
            return frustration.compute_single_frustration(decoy_fluctuation, aa_freq, correction)
        elif kind in ['mutational', 'configurational', 'contact']:
            if aa_freq is not None:
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
        #Ferrerio et al. (2007) pair distribution analysis included long-range contacts (distance>9.5).
        if include_long_range_contacts==True:
            mask=frustration.compute_mask(self.distance_matrix, distance_cutoff=None, sequence_distance_cutoff = 2)
        else:
            mask=self.mask

        frustration_values=self.frustration(kind=kind,mask=mask)
        residue_ca_coordinates=(self.structure.select('calpha').getCoords())

        if kind=="singleresidue":
            sel_frustration = np.column_stack((residue_ca_coordinates,np.expand_dims(frustration_values, axis=1)))

        elif kind in ["configurational","mutational"]:
            i,j=np.meshgrid(range(0,len(self.sequence)),range(0,len(self.sequence)))
            midpoint_coordinates=(residue_ca_coordinates[i.flatten(),:]+ residue_ca_coordinates[j.flatten(),:])/2
            sel_frustration = np.column_stack((midpoint_coordinates, frustration_values.ravel()))

        bins=100
        maximum_shell_radius=20
        r=np.linspace(0,maximum_shell_radius,num=bins)
        r_m=(r[1:]+r[:-1])/2
        shell_vol = 4/3 * np.pi * (r[1:]**3-r[:-1]**3)
        maximum_shell_vol=4/3 * np.pi *(maximum_shell_radius**3)

        minimally_frustrated_contacts=(sdist.pdist(sel_frustration[sel_frustration[:, -1] >.78][:,:-1]))
        frustrated_contacts=(sdist.pdist(sel_frustration[sel_frustration[:, -1] <-1][:,:-1]))
        neutral_contacts=(sdist.pdist(sel_frustration[(sel_frustration[:, -1] > -1) & (sel_frustration[:, -1] < .78)][:,:-1]))
        total_contacts_count=(len(sel_frustration[sel_frustration[:,-1]!=np.nan][:,-1]))

        minimally_frustrated_hist,_ = np.histogram(minimally_frustrated_contacts,bins=r)
        frustrated_hist,_ = np.histogram(frustrated_contacts,bins=r)
        neutral_hist,_=np.histogram(neutral_contacts,bins=r)
        
        with sns.plotting_context("poster"):
            plt.figure(figsize=(15,12))

            g=sns.lineplot(x=r_m,y=np.divide(maximum_shell_vol*minimally_frustrated_hist,(total_contacts_count**2)*shell_vol),color="green",label="Minimally Frustrated")
            g=sns.lineplot(x=r_m,y=np.divide(maximum_shell_vol*frustrated_hist,(total_contacts_count**2)*shell_vol),color="red",label="Frustrated")
            g=sns.lineplot(x=r_m,y=np.divide(maximum_shell_vol*neutral_hist,(total_contacts_count**2)*shell_vol),color="gray",label="Neutral")
            plt.xlabel("Pair Distance (A)"); plt.ylabel("g(r)")
            plt.legend()
            plt.show()


    def view_frustration_histogram(self,kind:str = "singleresidue"):
        frustration_values=self.frustration(kind=kind)

        r=np.linspace(-4,4,num=100)

        if kind=="singleresidue":
            minimally_frustrated=[i for i in frustration_values if i>.78]
            frustrated=[i for i in frustration_values if i<-1]
            neutral=[i for i in frustration_values if -1<i<.78]

            #Plot histogram of all frustration values.
            with sns.plotting_context("poster"):
                plt.figure(figsize=(10,5))

                g=sns.histplot(x=minimally_frustrated,bins=r,color="green")
                g=sns.histplot(x=frustrated,bins=r,color="red")
                g=sns.histplot(x=neutral,bins=r,color="gray")

                ymin, ymax = g.get_ylim()
                g.vlines(x=[-1, .78], ymin=ymin, ymax=ymax, colors=['black', 'black'], ls='--', lw=2)
                plt.title(f"{len(frustration_values)} Residues")
                plt.xlabel("$F_{i}$")
                plt.show()
            print(f"{(len(minimally_frustrated)/len(frustration_values))*100:.2f}% of Residues are Minimally Frustrated")
            print(f"{(len(frustrated)/len(frustration_values))*100:.2f}% of Residues are Frustrated")
            print(f"{(len(neutral)/len(frustration_values))*100:.2f}% of Residues are Neutral")

        elif kind in ["configurational","mutational"]:
            cb_distance_matrix=self.distance_matrix

            sel_frustration = np.array([cb_distance_matrix.ravel(), frustration_values.ravel()]).T
            sel_frustration=sel_frustration[~np.isnan(sel_frustration[:, 1])]
            minimally_frustrated = sel_frustration[sel_frustration[:, 1] > .78]
            frustrated = sel_frustration[sel_frustration[:, 1] < -1]
            neutral=sel_frustration[(sel_frustration[:, 1] > -1) & (sel_frustration[:, 1] < .78)]

            #Plot histogram of all frustration values.
            with sns.plotting_context("poster"):
                fig,axes=plt.subplots(1,2,figsize=(15,5),sharex=True)

                g=sns.histplot(x=minimally_frustrated[minimally_frustrated[:,0]<6.5][:,1],bins=r,ax=axes[0],color="green")
                g=sns.histplot(x=frustrated[frustrated[:,0]<6.5][:,1],bins=r,ax=axes[0],color="red")
                g=sns.histplot(x=neutral[neutral[:,0]<6.5][:,1],bins=r,ax=axes[0],color="gray")

                ymin, ymax = g.get_ylim()
                g.vlines(x=[-1, .78], ymin=ymin, ymax=ymax, colors=['black', 'black'], ls='--', lw=2)
                axes[0].title.set_text(f"Direct Contacts\n(N={len(sel_frustration[sel_frustration[:,0]<6.5])})")
                axes[0].set_xlabel("$F_{ij}$")
                ###
                g=sns.histplot(x=minimally_frustrated[minimally_frustrated[:,0]>6.5][:,1],bins=r,ax=axes[1],color="green")
                g=sns.histplot(x=frustrated[frustrated[:,0]>6.5][:,1],bins=r,ax=axes[1],color="red")
                g=sns.histplot(x=neutral[neutral[:,0]>6.5][:,1],bins=r,ax=axes[1],color="gray")

                ymin, ymax = g.get_ylim()
                g.vlines(x=[-1, .78], ymin=ymin, ymax=ymax, colors=['black', 'black'], ls='--', lw=2)
                axes[1].title.set_text(f"Water-Mediated and\nProtein-Mediated Contacts\n(N={len(sel_frustration[sel_frustration[:,0]>6.5])})")
                axes[1].set_xlabel("$F_{ij}$")

                plt.tight_layout()
                plt.show()
            ###
            print(f"{(len(minimally_frustrated[minimally_frustrated[:,0]<6.5])/len(sel_frustration[sel_frustration[:,0]<6.5]))*100:.2f}% of Direct Contacts are Minimally Frustrated")
            print(f"{(len(frustrated[frustrated[:,0]<6.5])/len(sel_frustration[sel_frustration[:,0]<6.5]))*100:.2f}% of Direct Contacts are Frustrated")
            print(f"{(len(neutral[neutral[:,0]<6.5])/len(sel_frustration[sel_frustration[:,0]<6.5]))*100:.2f}% of Direct Contacts are Neutral")
            print("###")
            print(f"{(len(minimally_frustrated[minimally_frustrated[:,0]>6.5])/len(sel_frustration[sel_frustration[:,0]>6.5]))*100:.2f}% of Water-Mediated Contacts are Minimally Frustrated")
            print(f"{(len(frustrated[frustrated[:,0]>6.5])/len(sel_frustration[sel_frustration[:,0]>6.5]))*100:.2f}% of Water-Mediated Contacts are Frustrated")
            print(f"{(len(neutral[neutral[:,0]>6.5])/len(sel_frustration[sel_frustration[:,0]>6.5]))*100:.2f}% of Water-Mediated Contacts are Neutral")





