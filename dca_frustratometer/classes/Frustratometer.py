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

    def sequences_energies(self, sequences, split_couplings_and_fields=False):
        return frustration.compute_sequences_energy(sequences, self.potts_model, self.mask, split_couplings_and_fields)

    def fields_energy(self, sequence=None):
        if sequence is None:
            sequence=self._sequence
        return frustration.compute_fields_energy(sequence, self.potts_model)

    def couplings_energy(self, sequence=None):
        if sequence is None:
            sequence=self._sequence
        return frustration.compute_couplings_energy(sequence, self.potts_model, self.mask)
        
    def decoy_fluctuation(self, kind='singleresidue',store=True):
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
        #Set to False to save memory.
        if store==True:
            self._decoy_fluctuation[kind] = fluctuation
            return self._decoy_fluctuation[kind]
        else:
            return fluctuation

    def decoy_energy(self, kind='singleresidue'):
        return self.native_energy + self.decoy_fluctuation(kind)

    def scores(self):
        return frustration.compute_scores(self.potts_model)

    def frustration(self, kind='singleresidue', aa_freq=None, correction=0):
        decoy_fluctuation = self._decoy_fluctuation(kind)
        if kind == 'singleresidue':
            if aa_freq is None:
                aa_freq = self._aa_freq
            return frustration.compute_single_frustration(decoy_fluctuation, aa_freq, correction)
        elif kind in ['mutational', 'configurational', 'contact']:
            if aa_freq is None:
                aa_freq = self._contact_freq
            return frustration.compute_pair_frustration(decoy_fluctuation, aa_freq, correction)

    def plot_decoy_energy(self, kind='singleresidue', method='clustermap'):
        native_energy = self.native_energy
        decoy_energy = self.decoy_energy(kind)
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
            view.addLine({'start':{'chain':'A','resi':[str(i+shift)]},'end':{'chain':'A','resi':[str(j+shift)]},
                        'color':'red', 'dashed':False,'linewidth':3})
        
        for i,j,f in minimally_frustrated:
            view.addLine({'start':{'chain':'A','resi':[str(i+shift)]},'end':{'chain':'A','resi':[str(j+shift)]},
                        'color':'green', 'dashed':False,'linewidth':3})

        view.zoomTo(viewer=(0,0))

        return view


    @property
    def sequence(self):
        return self._sequence
    
    @sequence.setter
    def sequence(self, value):
        assert len(value) == len(self._sequence)
        self._sequence = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def pdb_file(self):
        return str(self._pdb_file)

    @pdb_file.setter
    def pdb_file(self, value):
        self._pdb_file = Path(value)
        self._distance_matrix = pdb.get_distance_matrix(self.pdb_file, self.chain, method=value)
        self._mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._potts_model = {}
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def pdbID(self):
        """
        Returns PDBid from pdb name
        """
        assert self._pdb_file.exists()
        return self._pdbID

    @property
    def chain(self):
        return self._chain

    @chain.setter
    def chain(self, value):
        self._chain = value

    @property
    def pfamID(self):
        """
        Returns pfamID from pdb name
        """
        return self._pfamID
    
    @pfamID.setter
    def pfamID(self, value):
        self._pfamID=value

    @property
    def alignment_type(self):
        return self._alignment_type

    @alignment_type.setter
    def alignment_type(self, value):
        self._alignment_type = value

    @property
    def alignment_sequence_database(self):
        return self._alignment_sequence_database

    @alignment_sequence_database.setter
    def alignment_sequence_database(self, value):
        self._alignment_sequence_database = value

    @property
    def download_all_alignment_files(self):
        return self._download_all_alignment_files

    @download_all_alignment_files.setter
    def download_all_alignment_files(self, value):
        self._download_all_alignment_files = value

    @property
    def alignment_files_directory(self):
        return self._alignment_files_directory

    @alignment_files_directory.setter
    def alignment_files_directory(self, value):
        self._alignment_files_directory = value

    @property
    def alignment_output_file(self):
        return self._alignment_output_file

    @alignment_output_file.setter
    def alignment_output_file(self, value):
        self._alignment_output_file = value

    @property
    def sequence_cutoff(self):
        return self._sequence_cutoff

    @sequence_cutoff.setter
    def sequence_cutoff(self, value):
        self._mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, value)
        self._sequence_cutoff = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def distance_cutoff(self):
        return self._distance_cutoff

    @distance_cutoff.setter
    def distance_cutoff(self, value):
        self._mask = frustration.compute_mask(self.distance_matrix, value, self.sequence_cutoff)
        self._distance_cutoff = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def distance_matrix_method(self):
        return self._distance_matrix_method

    @distance_matrix_method.setter
    def distance_matrix_method(self, value):
        self._distance_matrix = pdb.get_distance_matrix(self.pdb_file, self.chain, method=value)
        self._mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._distance_matrix_method = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, value):
        assert value.shape==(len(self._sequence),len(self._sequence)), "Distance matrix dimensions are incorrect"
        self._distance_matrix = value
        self._mask = frustration.compute_mask(value, self._distance_cutoff, self._sequence_cutoff)
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._distance_matrix=None
        self._sequence_cutoff=None
        self._distance_cutoff=None
        self._mask = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def aa_freq(self):
        if not self._aa_freq:
            self._aa_freq = frustration.compute_aa_freq(self.sequence)
        return self._aa_freq

    @aa_freq.setter
    def aa_freq(self, value):
        assert len(value)==21, "AA frequencies must be calculated for all 20 amino acids."
        self._aa_freq=value
        self._decoy_fluctuation={}

    @property
    def contact_freq(self):
        if not self._contact_freq:
            self._contact_freq = frustration.compute_contact_freq(self.sequence)
        return self._contact_freq

    @contact_freq.setter
    def contact_freq(self, value):
        assert value.shape==(21,21), "Contact frequencies must be calculated for all 20 amino acids."
        self._contact_freq=value
        self._decoy_fluctuation={}             

    @property
    def potts_model_file(self):
        return self._potts_model_file

    @potts_model_file.setter
    def potts_model_file(self, value):
        self._potts_model = dca.matlab.load_potts_model(value)
        self._potts_model_file = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def potts_model(self):
        return self._potts_model

    @potts_model.setter
    def potts_model(self, value):
        self._potts_model = value
        self._potts_model_file = None
        self._native_energy = None
        self._decoy_fluctuation = {}
    
    @property
    def native_energy(self,sequence=None):
        if sequence is None:
            sequence=self._sequence
        if not self._native_energy:
            self._native_energy=frustration.compute_native_energy(sequence, self.potts_model, self.mask)
        return self._native_energy
