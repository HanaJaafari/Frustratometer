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
    def from_potts_model_file(cls,pdb_structure,
                              potts_model_file: str,
                              reformat_potts_model: bool = False,
                              sequence_cutoff: typing.Union[float, None] = None,
                              distance_cutoff: typing.Union[float, None] = None):
        self = cls()

        # Set initialization variables
        self.structure=pdb_structure.structure
        self.chain=pdb_structure.chain
        self.sequence=pdb_structure.sequence
        self.pdb_file=pdb_structure.pdb_file
        self.potts_model_file=potts_model_file
        self.reformat_potts_model=reformat_potts_model
        self.init_index_shift=pdb_structure.init_index_shift

        self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
        self.filtered_aligned_sequence=pdb_structure.filtered_aligned_sequence
        self.aligned_sequence=pdb_structure.aligned_sequence

        self.mapped_distance_matrix=pdb_structure.mapped_distance_matrix
        self.distance_matrix=self.mapped_distance_matrix
        self.sequence_cutoff=sequence_cutoff
        self.distance_cutoff=distance_cutoff
        
        if self.distance_cutoff==None:
            example_matrix=np.ones((len(self.filtered_aligned_sequence),len(self.filtered_aligned_sequence)))
            self.mask = frustration.compute_mask(example_matrix, self.distance_cutoff, self.sequence_cutoff)
        else:
            self.mask = frustration.compute_mask(self.mapped_distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        self.minimally_frustrated_threshold=1

        # Compute fast properties
        self.potts_model = dca.matlab.load_potts_model(self.potts_model_file)
        if self.reformat_potts_model:
            self.potts_model["h"]=self.potts_model["h"].T
            self.potts_model["J"]= self.potts_model["familycouplings"].reshape(int(len(self.filtered_aligned_sequence)),21,int(len(self.filtered_aligned_sequence)),21).transpose(0,2,1,3)

        if self.filtered_aligned_sequence is not None:
            self.aa_freq = frustration.compute_aa_freq(self.sequence)
            self.contact_freq = frustration.compute_contact_freq(self.sequence)
        else:
            self.aa_freq = None
            self.contact_freq = None   

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
    def from_alignment(cls,
                       alignment: str,
                       pdb_file: str,
                       chain: str,
                       sequence_cutoff: typing.Union[float, None] = None,
                       distance_cutoff: typing.Union[float, None] = None,
                       distance_matrix_method='minimum'):
        
        # Compute dca
        potts_model = dca.pydca.plmdca(alignment)
        return cls.from_pottsmodel(potts_model, 
                                   pdb_file=pdb_file, 
                                   chain=chain, 
                                   sequence_cutoff=sequence_cutoff, 
                                   distance_cutoff=distance_cutoff, 
                                   distance_matrix_method=distance_matrix_method)



    # @property
    # def sequence(self):
    #     return self._sequence
    
    # # Set a new sequence in case someone needs to calculate the energy of a diferent sequence with the same structure
    # @sequence.setter
    # def sequence(self, value):
    #     assert len(value) == len(self._sequence)
    #     self._sequence = value

    # @property
    # def pdb_file(self):
    #     return str(self._pdb_file)

    # # @pdb_file.setter
    # # def pdb_file(self, value):
    # #     self._pdb_file = Path(value)

    # @property
    # def pdb_name(self, value):
    #     """
    #     Returns PDBid from pdb name
    #     """
    #     assert self._pdb_file.exists()
    #     return self._pdb_file.stem

    # @property
    # def chain(self):
    #     return self._chain

    # # @chain.setter
    # # def chain(self, value):
    # #     self._chain = value

    # @property
    # def pfamID(self, value):
    #     """
    #     Returns pfamID from pdb name
    #     """
    #     return self._pfamID

    # @property
    # def alignment_type(self, value):
    #     return self._alignment_type

    # # @alignment_type.setter
    # # def alignment_type(self, value):
    # #     self._alignment_type = value

    # @property
    # def alignment_sequence_database(self, value):
    #     return self._alignment_sequence_database

    # # @alignment_sequence_database.setter
    # # def alignment_sequence_database(self, value):
    # #     self._alignment_sequence_database = value

    # @property
    # def download_all_alignment_files(self, value):
    #     return self._download_all_alignment_files

    # # @download_all_alignment_files.setter
    # # def download_all_alignment_files(self, value):
    # #     self._download_all_alignment_files = value

    # @property
    # def alignment_files_directory(self, value):
    #     return self._alignment_files_directory

    # # @alignment_files_directory.setter
    # # def alignment_files_directory(self, value):
    # #     self._alignment_files_directory = value

    # @property
    # def alignment_output_file(self, value):
    #     return self._alignment_output_file

    # # @alignment_output_file.setter
    # # def alignment_output_file(self, value):
    # #     self._alignment_output_file = value

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

    # @property
    # def distance_matrix_method(self):
    #     return self._distance_matrix_method

    # @distance_matrix_method.setter
    # def distance_matrix_method(self, value):
    #     self.distance_matrix = pdb.get_distance_matrix(self._pdb_file, self._chain, value)
    #     self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
    #     self._distance_matrix_method = value
    #     self._native_energy = None
    #     self._decoy_fluctuation = {}

    # @property
    # def potts_model_file(self):
    #     return self._potts_model_file

    # @potts_model_file.setter
    # def potts_model_file(self, value):
    #     if value == None:
    #         print("Generating PDB alignment using Jackhmmer")
    #         align.create_alignment_jackhmmer(self.sequence, self.pdb_name,
    #                                    output_file="dcaf_{}_alignment.sto".format(self.pdb_name))
    #         filter.convert_and_filter_alignment(self.pdb_name)
    #         dca.matlab.compute_plm(self.pdb_name)
    #         raise ValueError("Need to generate potts model")
    #     else:
    #         self.potts_model = dca.matlab.load_potts_model(value)
    #         self._potts_model_file = value
    #         self._native_energy = None
    #         self._decoy_fluctuation = {}

    # @property
    # def potts_model(self):
    #     return self._potts_model

    # @potts_model.setter
    # def potts_model(self, value):
    #     self._potts_model = value
    #     self._potts_model_file = None
    #     self._native_energy = None
    #     self._decoy_fluctuation = {}

