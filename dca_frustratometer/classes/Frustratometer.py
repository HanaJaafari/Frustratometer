"""Provide the primary functions."""
import typing
import os
from pathlib import Path

#Import other modules
from ..utils import _path
from .. import pdb
from .. import filter
from .. import dca
from .. import map
from .. import align
from .. import frustration
from .. import pfam

__all__=['PottsModel']
##################
# PFAM functions #
##################


# Class wrapper
class Frustratometer:
    def __init__(self,pdb_file:str,chain:str,sequence:str=None,sequence_cutoff: typing.Union[float, None] = None,distance_cutoff: typing.Union[float, None] = None,distance_matrix_method="minimum"):
        self._pdb_file = Path(pdb_file)
        self._chain = chain
        if sequence is None:
            self._sequence = pdb.get_sequence(self.pdb_file, self.chain)
        else:
            self._sequence = sequence
        self._distance_matrix_method=distance_matrix_method
        self.distance_matrix = pdb.get_distance_matrix(self.pdb_file, self.chain, self.distance_matrix_method)

        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff

        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def sequence(self):
        return self._sequence
    
    # Set a new sequence in case someone needs to calculate the energy of a diferent sequence with the same structure
    @sequence.setter
    def sequence(self, value):
        assert len(value) == len(self._sequence)
        self._sequence = value

    @property
    def pdb_file(self):
        return str(self._pdb_file)

    @property
    def pdb_name(self, value):
        """
        Returns PDBid from pdb name
        """
        assert self._pdb_file.exists()
        return self._pdb_file.stem

    @property
    def chain(self):
        return self._chain

    @property
    def sequence_cutoff(self):
        return self._sequence_cutoff

    @sequence_cutoff.setter
    def sequence_cutoff(self, value):
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._sequence_cutoff = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def distance_cutoff(self):
        return self._distance_cutoff

    @distance_cutoff.setter
    def distance_cutoff(self, value):
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._distance_cutoff = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def distance_matrix_method(self):
        return self._distance_matrix_method

    @distance_matrix_method.setter
    def distance_matrix_method(self, value):
        self.distance_matrix = pdb.get_distance_matrix(self._pdb_file, self._chain, value)
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._distance_matrix_method = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    def native_energy(self, sequence=None):
        if sequence is None:
            if self._native_energy:
                return self._native_energy
            else:
                return frustration.compute_native_energy(self.sequence, self.potts_model, self.mask)
        else:
            return frustration.compute_native_energy(sequence, self.potts_model, self.mask)

    def sequences_energies(self, sequences, split_couplings_and_fields=False):
        return frustration.compute_sequences_energy(sequences, self.potts_model, self.mask, split_couplings_and_fields)

    def fields_energy(self, sequence=None):
        if sequence is None:
            sequence=self._sequence
        return frustration.compute_fields_energy(sequence, self.potts_model, self.mask)

    def couplings_energy(self, sequence=None):
        if sequence is None:
            sequence=self._sequence
        return frustration.compute_couplings_energy(sequence, self.potts_model, self.mask)
        
    def decoy_fluctuation(self, kind='singleresidue'):
        if kind in self._decoy_fluctuation:
            return self._decoy_fluctuation[kind]
        if kind == 'singleresidue':
            fluctuation = frustration.compute_singleresidue_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)
        elif kind == 'mutational':
            fluctuation = frustration.compute_mutational_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)
        elif kind == 'configurational':
            fluctuation = frustration.compute_configurational_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)
        elif kind == 'contact':
            fluctuation = frustration.compute_contact_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)

        else:
            raise Exception("Wrong kind of decoy generation selected")
        self._decoy_fluctuation[kind] = fluctuation
        return self._decoy_fluctuation[kind]

    def decoy_energy(self, kind='singleresidue'):
        return self.native_energy() + self.decoy_fluctuation(kind)

    def scores(self):
        return frustration.compute_scores(self.potts_model)

    def frustration(self, kind='singleresidue', aa_freq=None, correction=0):
        decoy_fluctuation = self.decoy_fluctuation(kind)
        if kind == 'singleresidue':
            if aa_freq is not None:
                aa_freq = self.aa_freq
            return frustration.compute_single_frustration(decoy_fluctuation, aa_freq, correction)
        elif kind in ['mutational', 'configurational', 'contact']:
            if aa_freq is not None:
                aa_freq = self.contact_freq
            return frustration.compute_pair_frustration(decoy_fluctuation, aa_freq, correction)

    def plot_decoy_energy(self, kind='singleresidue'):
        native_energy = self.native_energy()
        decoy_energy = self.decoy_energy(kind)
        if kind == 'singleresidue':
            frustration.plot_singleresidue_decoy_energy(decoy_energy, native_energy)

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
        pair_frustration=self.frustration(pair)*np.triu(self.mask)
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
            view.addLine({'start':{'chain':'A','resi':[str(i+1)]},'end':{'chain':'A','resi':[str(j+1)]},
                        'color':'red', 'dashed':False,'linewidth':3})
        
        for i,j,f in minimally_frustrated:
            view.addLine({'start':{'chain':'A','resi':[str(i+1)]},'end':{'chain':'A','resi':[str(j+1)]},
                        'color':'green', 'dashed':False,'linewidth':3})

        view.zoomTo(viewer=(0,0))

        return view

