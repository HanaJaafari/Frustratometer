"""Provide the primary functions."""
import typing
import numpy as np
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
from .Frustratometer import Frustratometer

__all__=['PottsModel']
##################
# PFAM functions #
##################


# Class wrapper
class PottsModel(Frustratometer):

    @classmethod
    def from_distance_matrix(cls,
                 potts_model: dict,
                 distance_matrix : np.array,
                 sequence: str,
                 sequence_cutoff: typing.Union[float, None] = None,
                 distance_cutoff: typing.Union[float, None] = None):
        
        self = cls()
        # Set initialization variables
        self._potts_model = potts_model
        self._potts_model_file = None

        self._pdb_file = None
        self._chain = None
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = None

        # Compute fast properties
        self._sequence = sequence
        self.distance_matrix = distance_matrix
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_potts_model_file(cls,
                              potts_model_file: str,
                              pdb_file: str,
                              chain: str,
                              sequence_cutoff: typing.Union[float, None] = None,
                              distance_cutoff: typing.Union[float, None] = None,
                              distance_matrix_method='minimum'):
        self = cls()

        # Set initialization variables
        self._potts_model_file = potts_model_file
        self._pdb_file = Path(pdb_file)
        self._chain = chain
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = distance_matrix_method

        # Compute fast properties
        self._potts_model = dca.matlab.load_potts_model(self.potts_model_file)
        self._sequence = pdb.get_sequence(self.pdb_file, self.chain)
        self.distance_matrix = pdb.get_distance_matrix(self.pdb_file, self.chain, self.distance_matrix_method)

        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_pottsmodel(cls,
                        potts_model: dict,
                        pdb_file: str,
                        chain: str,
                        sequence_cutoff: typing.Union[float, None] = None,
                        distance_cutoff: typing.Union[float, None] = None,
                        distance_matrix_method='minimum'):
        self = cls()

        # Set initialization variables
        self._potts_model = potts_model
        self._potts_model_file = None
        self._pdb_file = Path(pdb_file)
        self._chain = chain
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = distance_matrix_method

        # Compute fast properties
        self._sequence = pdb.get_sequence(self.pdb_file, self.chain)
        self.distance_matrix = pdb.get_distance_matrix(self.pdb_file, self.chain, self.distance_matrix_method)
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)
        self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_pfam_alignment(cls,
                            sequence:   str,
                            PFAM_ID: str,
                            pdb_file:   str,
                            pdb_chain:  str,
                            download_all_alignment_files: bool,
                            alignment_files_directory: str,
                            sequence_cutoff: typing.Union[float, None] = None,
                            distance_cutoff: typing.Union[float, None] = None,
                            distance_matrix_method='minimum'):
        self = cls()

        # Set initialization variables
        #Can provide pdb file path or a protein sequence
        self._potts_model_file = None
        self._pdb_file=Path(pdb_file)
        self._pdb_chain=pdb_chain
        self._download_all_alignment_files = download_all_alignment_files
        self._alignment_files_directory=Path(alignment_files_directory)
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = distance_matrix_method

        # Compute fast properties
        if sequence==None:
            self._sequence = pdb.get_sequence(self.pdb_file, self.pdb_chain)
            self.distance_matrix = pdb.get_distance_matrix(self.pdb_file, self.pdb_chain, self.distance_matrix_method)
            self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        else:
            self._sequence=sequence
            self.distance_matrix=None
            self.mask = None
        
        if PFAM_ID==None:
            # TODO Identify protein family given protein sequence
            if sequence==None:
                self._PFAM_ID=map.get_pfamID(self.pdb_file,self.chain)
        else:
            self._PFAM_ID=PFAM_ID

        self._alignment_file=pfam.download_full_alignment(self.PFAM_ID,self.alignment_files_directory)
        self._filtered_alignment_file=filter.convert_and_filter_alignment(self.alignment_file,self.download_all_alignment_files,self.alignment_files_directory)
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_hmmer_alignment(cls,
                             sequence:   str,
                             PFAM_ID: str,
                             pdb_file:   str,
                             pdb_chain:  str,
                             download_all_alignment_files: bool,
                             alignment_files_directory: str,
                             alignment_output_file:  bool,
                             alignment_sequence_database:    str,
                             sequence_cutoff: typing.Union[float, None] = None,
                             distance_cutoff: typing.Union[float, None] = None,
                             distance_matrix_method='minimum'):
        self = cls()

        # Set initialization variables
        #Can provide pdb file or a protein sequence
        self._potts_model_file = None
        self._pdb_file=Path(pdb_file)
        self._pdb_chain=pdb_chain
        self._download_all_alignment_files = download_all_alignment_files
        self._alignment_files_directory=Path(alignment_files_directory)
        self._alignment_output_file=alignment_output_file
        self._alignment_sequence_database=alignment_sequence_database
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = distance_matrix_method

        # Compute fast properties
        if sequence==None:
            self._sequence = pdb.get_sequence(self.pdb_file, self.pdb_chain)
            self.distance_matrix = pdb.get_distance_matrix(self.pdb_file, self.pdb_chain, self.distance_matrix_method)
            self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        else:
            self._sequence=sequence
            self.distance_matrix=None
            self.mask = None

        self._alignment_file=align.generate_hmmer_alignment(self.pdb_file,self.sequence,self.alignment_files_directory,self.alignment_output_file,self.alignment_sequence_database)
        self._filtered_alignment_file=filter.convert_and_filter_alignment(self.alignment_file,self.download_all_alignment_files,self.alignment_files_directory)
        self.aa_freq = frustration.compute_aa_freq(self.sequence)
        self.contact_freq = frustration.compute_contact_freq(self.sequence)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_alignment(cls):
        # Compute dca
        import pydca.plmdca
        plmdca_inst = pydca.plmdca.PlmDCA(
            'test',
            'protein',
            seqid=0.8,
            lambda_h=1.0,
            lambda_J=20.0,
            num_threads=10,
            max_iterations=500,
        )

        # compute DCA scores summarized by Frobenius norm and average product corrected
        potts_model = plmdca_inst.get_potts_model()

