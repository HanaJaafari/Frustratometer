"""Provide the primary functions."""
from typing import Union
import numpy as np
from pathlib import Path
from typing import Union


#Import other modules
from .. import pdb
from .. import filter
from .. import dca
from .. import map
from .. import align
from .. import frustration
from .. import pfam
from .Frustratometer import Frustratometer

__all__=['DCA']
##################
# PFAM functions #
##################


# Class wrapper
class DCA(Frustratometer):
    """
    The DCA class contains many class methods that can be used, depending on whether they have already calculated the DCA couplings and fields parameters.

    If the user already has calculated these parameters, the "from_potts_model_file" or "from_pottsmodel" methods can be used. Otherwise, the user can 
    locally generate these parameters using the pyDCA package. In this case, the user can try using the "from_distance_matrix," "from_pfam_alignment,"
    or "from_hmmer_alignment" methods.
    """
    # @classmethod
    # def from_distance_matrix(cls,
    #              potts_model: dict,
    #              distance_matrix : np.array,
    #              sequence: str =None,
    #              sequence_cutoff: Union[float, None] = None,
    #              distance_cutoff: Union[float, None] = None)->object:

    #     """
    #     Generate DCA object from previously generated distance matrix.

    #     Parameters
    #     ----------
    #     pdb_structure : object
    #         Structure object generated by Structure class
    #     sequence_cutoff : float
    #         Sequence seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.
    #     distance_cutoff : float
    #         Distance seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.

        
    #     Returns
    #     -------
    #     DCA object
    #     """
        
    #     self = cls()
    #     # Set initialization variables
    #     self._potts_model = potts_model
    #     self._potts_model_file = None
    #     self._pdb_file = None
    #     self._chain = None
    #     self._sequence_cutoff = sequence_cutoff
    #     self._distance_cutoff = distance_cutoff
    #     self._distance_matrix_method = None
    #     self._sequence = sequence

    #     # Compute fast properties
    #     self.distance_matrix = distance_matrix
    #     self.aa_freq = frustration.compute_aa_freq(self.sequence)
    #     self.contact_freq = frustration.compute_contact_freq(self.sequence)
    #     self.mask = frustration.compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

    #     # Initialize slow properties
    #     self._native_energy = None
    #     self._decoy_fluctuation = {}
    #     return self

    @classmethod
    def from_potts_model_file(cls,pdb_structure: object,
                              potts_model_file: Union[Path,str] = None,
                              reformat_potts_model: bool = False,
                              sequence_cutoff: Union[float, None] = None,
                              distance_cutoff: Union[float, None] = None)->object:

        """
        Generate DCA object from previously generated potts model file.

        Parameters
        ----------
        pdb_structure : object
            Structure object generated by Structure class
        potts_model_file :  Path or str
            File path of potts model file
        reformat_potts_model : bool
            If True, the fields matrix will be transposed, while the couplings matrix will be reformatted into a (NxNx21x21) matrix. 
            This reformatting is necessary for some potts model files generated by the mfDCA Matlab algorithm.
            Default is False.
        sequence_cutoff : float
            Sequence seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.
        distance_cutoff : float
            Distance seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.

        
        Returns
        -------
        DCA object
        """
        self = cls()

        # Set initialization variables
        self.structure=pdb_structure.structure
        self._chain=pdb_structure.chain
        self._sequence=pdb_structure.sequence
        self._pdb_file=pdb_structure.pdb_file
        self._potts_model_file=potts_model_file
        self._sequence_cutoff=sequence_cutoff
        self._distance_cutoff=distance_cutoff
        self._distance_matrix_method=None

        self.reformat_potts_model=reformat_potts_model
        self.init_index_shift=pdb_structure.init_index_shift

        self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
        self.filtered_aligned_sequence=pdb_structure.filtered_aligned_sequence
        self.aligned_sequence=pdb_structure.aligned_sequence

        self.mapped_distance_matrix=pdb_structure.mapped_distance_matrix
        self.distance_matrix=self.mapped_distance_matrix

        
        if self.distance_cutoff==None:
            example_matrix=np.ones((len(self.filtered_aligned_sequence),len(self.filtered_aligned_sequence)))
            self.mask = frustration.compute_mask(example_matrix, self.distance_cutoff, self.sequence_cutoff)
        else:
            self.mask = frustration.compute_mask(self.mapped_distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        self.minimally_frustrated_threshold=1

        # Compute fast properties
        self._potts_model = dca.matlab.load_potts_model(self.potts_model_file)
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
    def from_pottsmodel(cls,pdb_structure : object,
                        potts_model: dict,
                        reformat_potts_model: bool = False,
                        sequence_cutoff: Union[float, None] = None,
                        distance_cutoff: Union[float, None] = None)->object:
        """
        Generate DCA object from previously generated potts model.

        Parameters
        ----------
        pdb_structure : object
            Structure object generated by Structure class
        potts_model :  dict
            Dictionary of potts model file, containing fields and couplings keys.
        reformat_potts_model : bool
            If True, the fields matrix will be transposed, while the couplings matrix will be reformatted into a (NxNx21x21) matrix. 
            This reformatting is necessary for some potts model files generated by the mfDCA Matlab algorithm.
            Default is False.
        sequence_cutoff : float
            Sequence seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.
        distance_cutoff : float
            Distance seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.

        
        Returns
        -------
        DCA object
        """
        self = cls()

        # Set initialization variables
        self.structure=pdb_structure.structure
        self._chain=pdb_structure.chain
        self._sequence=pdb_structure.sequence
        self._pdb_file=pdb_structure.pdb_file
        self._potts_model=potts_model
        self._sequence_cutoff=sequence_cutoff
        self._distance_cutoff=distance_cutoff
        self._distance_matrix_method=None

        self.reformat_potts_model=reformat_potts_model
        self.init_index_shift=pdb_structure.init_index_shift

        self.full_to_aligned_index_dict=pdb_structure.full_to_aligned_index_dict
        self.filtered_aligned_sequence=pdb_structure.filtered_aligned_sequence
        self.aligned_sequence=pdb_structure.aligned_sequence

        self.mapped_distance_matrix=pdb_structure.mapped_distance_matrix
        self.distance_matrix=self.mapped_distance_matrix

        if self.distance_cutoff==None:
            example_matrix=np.ones((len(self.filtered_aligned_sequence),len(self.filtered_aligned_sequence)))
            self.mask = frustration.compute_mask(example_matrix, self.distance_cutoff, self.sequence_cutoff)
        else:
            self.mask = frustration.compute_mask(self.mapped_distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        self.minimally_frustrated_threshold=1

        # Compute fast properties
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
    def from_pfam_alignment(cls,pdb_structure : object,
                            alignment_output_file_name: Union[Path,str],
                            filtered_alignment_output_file_name: Union[Path,str],
                            PFAM_ID: str = None,
                            DCA_format : str = "plmDCA",
                            sequence_cutoff: Union[float, None] = None,
                            distance_cutoff: Union[float, None] = None)->object:
        """
        Generate DCA object from a locally downloaded PFAM alignment file that will be used to generate a Potts Model file.

        Parameters
        ----------
        pdb_structure : object
            Structure object generated by Structure class
        alignment_output_file_name : Path or str
            File name of generated alignment file. The file will be in stockholm format.
        filtered_alignment_output_file_name : Path or str
            File name of generated filtered alignment file. The file will be in Fasta format.
        PFAM_ID :  str
            PFAM ID associated with structure object
        DCA_format : str
            Current option is "plmDCA"
        sequence_cutoff : float
            Sequence seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.
        distance_cutoff : float
            Distance seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.
        
        Returns
        -------
        DCA object
        """
        self = cls()

        self.structure=pdb_structure.structure
        self._chain=pdb_structure.chain
        self._sequence=pdb_structure.sequence
        self._pdb_file=pdb_structure.pdb_file
        self._sequence_cutoff=sequence_cutoff
        self._distance_cutoff=distance_cutoff
        self._distance_matrix_method=None

        self.alignment_output_file_name = alignment_output_file_name
        self.filtered_alignment_output_file_name = filtered_alignment_output_file_name

        self.DCA_format=DCA_format
        
        self.mapped_distance_matrix=pdb_structure.mapped_distance_matrix
        self.distance_matrix=self.mapped_distance_matrix
        self.mask = frustration.compute_mask(self.mapped_distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        self.minimally_frustrated_threshold=1

        if PFAM_ID==None:
            self.PFAM_ID=map.get_pfamID(self.pdb_file,self.chain)
        else:
            self.PFAM_ID=PFAM_ID

        self.alignment_file=pfam.download_aligment(self.PFAM_ID,self.alignment_output_file_name)
        self.filtered_alignment_file=filter.filter_alignment(self.alignment_output_file_name,self.filtered_alignment_output_file_name)

        self.potts_model=dca.pydca.plmdca(str(self.filtered_alignment_file))

        self.aa_freq = None
        self.contact_freq = None 

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_hmmer_alignment(cls,pdb_structure : object,
                            alignment_output_file_name: Union[Path,str],
                            filtered_alignment_output_file_name: Union[Path,str],
                            query_sequence_database_file : Union[Path,str],
                            DCA_format : str = "plmDCA",
                            sequence_cutoff: Union[float, None] = None,
                            distance_cutoff: Union[float, None] = None)->object:
        """
        Generate DCA object from a locally generated jackhmmer alignment file that will be used to generate a Potts Model file.
        The protein sequence of the structure object will be used as the query sequence by the Jackhmmer algorithm.

        Parameters
        ----------
        pdb_structure : object
            Structure object generated by Structure class
        alignment_output_file_name : Path or str
            File name of generated alignment file. The file will be in stockholm format.
        filtered_alignment_output_file_name : Path or str
            File name of generated filtered alignment file. The file will be in Fasta format.
        query_sequence_database_file : Path or str
            File name of sequence database.
        DCA_format : str
            Current option is "plmDCA"
        sequence_cutoff : float
            Sequence seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.
        distance_cutoff : float
            Distance seperation cutoff; the couplings terms of contacts that are separated by more than this cutoff will be ignored.
        
        Returns
        -------
        DCA object
        """
        self = cls()

        self.structure=pdb_structure.structure
        self._chain=pdb_structure.chain
        self._sequence=pdb_structure.sequence
        self._pdb_file=pdb_structure.pdb_file
        self._potts_model=potts_model
        self._sequence_cutoff=sequence_cutoff
        self._distance_cutoff=distance_cutoff
        self._distance_matrix_method=None

        self._alignment_output_file_name = alignment_output_file_name
        self._filtered_alignment_output_file_name = filtered_alignment_output_file_name
        self._query_sequence_database_file=query_sequence_database_file

        self._DCA_format=DCA_format
        
        self.mapped_distance_matrix=pdb_structure.mapped_distance_matrix
        self.distance_matrix=self.mapped_distance_matrix
        self.mask = frustration.compute_mask(self.mapped_distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        self.minimally_frustrated_threshold=1

        self._alignment_file=align.jackhmmer(self.sequence,self.alignment_file)
        self._filtered_alignment_file=filter.filter_alignment(self.alignment_output_file_name,self.filtered_alignment_output_file_name,self.query_sequence_database_file)

        self._potts_model=dca.pydca.plmdca(self.filtered_alignment_file)

        self.aa_freq = None
        self.contact_freq = None 

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self


    # @classmethod
    # def from_alignment(cls,
    #                    alignment: str,
    #                    pdb_file: str,
    #                    chain: str,
    #                    sequence_cutoff: Union[float, None] = None,
    #                    distance_cutoff: Union[float, None] = None,
    #                    distance_matrix_method='minimum')->object:
        
    #     # Compute dca
    #     potts_model = dca.pydca.plmdca(alignment)
    #     return cls.from_pottsmodel(potts_model, 
    #                                pdb_file=pdb_file, 
    #                                chain=chain, 
    #                                sequence_cutoff=sequence_cutoff, 
    #                                distance_cutoff=distance_cutoff, 
    #                                distance_matrix_method=distance_matrix_method)



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

    # @pdb_file.setter
    # def pdb_file(self, value):
    #     self._pdb_file = Path(value)

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

    # @chain.setter
    # def chain(self, value):
    #     self._chain = value

    @property
    def pfamID(self, value):
        """
        Returns pfamID from pdb name
        """
        return self._pfamID

    @property
    def alignment_type(self, value):
        return self._alignment_type

    # @alignment_type.setter
    # def alignment_type(self, value):
    #     self._alignment_type = value

    @property
    def alignment_sequence_database(self, value):
        return self._alignment_sequence_database

    # @alignment_sequence_database.setter
    # def alignment_sequence_database(self, value):
    #     self._alignment_sequence_database = value

    @property
    def download_all_alignment_files(self, value):
        return self._download_all_alignment_files

    # @download_all_alignment_files.setter
    # def download_all_alignment_files(self, value):
    #     self._download_all_alignment_files = value

    @property
    def alignment_files_directory(self, value):
        return self._alignment_files_directory

    # @alignment_files_directory.setter
    # def alignment_files_directory(self, value):
    #     self._alignment_files_directory = value

    @property
    def alignment_output_file(self, value):
        return self._alignment_output_file

    # @alignment_output_file.setter
    # def alignment_output_file(self, value):
    #     self._alignment_output_file = value

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

    @property
    def potts_model_file(self):
        return self._potts_model_file

    @potts_model_file.setter
    def potts_model_file(self, value):
        if value == None:
            print("Generating PDB alignment using Jackhmmer")
            align.create_alignment_jackhmmer(self.sequence, self.pdb_name,
                                       output_file="dcaf_{}_alignment.sto".format(self.pdb_name))
            filter.convert_and_filter_alignment(self.pdb_name)
            dca.matlab.compute_plm(self.pdb_name)
            raise ValueError("Need to generate potts model")
        else:
            self.potts_model = dca.matlab.load_potts_model(value)
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

