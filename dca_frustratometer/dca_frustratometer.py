"""Provide the primary functions."""
import typing
import prody
import scipy.spatial.distance as sdist
import numpy as np
import os
from pathlib import Path
from .utils import _path


##################
# PFAM functions #
##################


# Class wrapper
class PottsModel:

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
        self._potts_model = load_potts_model(self.potts_model_file)
        self._sequence = get_protein_sequence_from_pdb(self.pdb_file, self.chain)
        self.distance_matrix = get_distance_matrix_from_pdb(self.pdb_file, self.chain, self.distance_matrix_method)

        self.aa_freq = compute_aa_freq(self.sequence)
        self.contact_freq = compute_contact_freq(self.sequence)
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

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
        self._sequence = get_protein_sequence_from_pdb(self.pdb_file, self.chain)
        self.distance_matrix = get_distance_matrix_from_pdb(self.pdb_file, self.chain, self.distance_matrix_method)
        self.aa_freq = compute_aa_freq(self.sequence)
        self.contact_freq = compute_contact_freq(self.sequence)
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_pdb_file(cls,
                      alignment_type: str,
                      alignment_sequence_database: str,
                      pdb_file: str,
                      chain: str,
                      download_all_alignment_files: bool,
                      alignment_files_directory: str,
                      alignment_output_file: bool,
                      sequence_cutoff: typing.Union[float, None] = None,
                      distance_cutoff: typing.Union[float, None] = None,
                      distance_matrix_method='minimum'):
        self = cls()

        # Set initialization variables
        self._potts_model_file = None
        self._alignment_type=alignment_type
        self._alignment_sequence_database=alignment_sequence_database
        self._pdb_file = Path(pdb_file)
        self._pdb_name=os.path.basedir(pdb_file)[:4]
        self._chain = chain
        self._download_all_alignment_files = download_all_alignment_files
        self._alignment_files_directory=Path(alignment_files_directory)
        self._alignment_output_file=alignment_output_file
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = distance_matrix_method

        # Compute fast properties
        self._sequence = get_protein_sequence_from_pdb(self.pdb_file, self.chain)
        self._pfamID=get_pfamID(self.pdb_name,self.chain)
        self._alignment_file=generate_alignment(self.pdb_name,self.pfamID,self.sequence,self.alignment_type,self.alignment_files_directory,self.alignment_output_file,self.alignment_sequence_database)
        self._filtered_alignment_file=convert_and_filter_alignment(self.alignment_file,self.download_all_alignment_files,self.alignment_files_directory)
        self.distance_matrix = get_distance_matrix_from_pdb(self.pdb_file, self.chain, self.distance_matrix_method)
        self.aa_freq = compute_aa_freq(self.sequence)
        self.contact_freq = compute_contact_freq(self.sequence)
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        return self

    @classmethod
    def from_alignment(cls):
        # Compute dca
        import pydca.plmdca
        plmdca_inst = pydca.plmdca.PlmDCA(
            new_alignment_file,
            'protein',
            seqid=0.8,
            lambda_h=1.0,
            lambda_J=20.0,
            num_threads=10,
            max_iterations=500,
        )

        # compute DCA scores summarized by Frobenius norm and average product corrected
        potts_model = plmdca_inst.get_potts_model()

    @property
    def sequence(self):
        return self._sequence
    #The sequence is dependent on the pdb so no need to provide option to set this.
    # @sequence.setter
    # def sequence(self, value):
    #     assert len(value) == len(self._sequence)
    #     self._sequence = value

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
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._sequence_cutoff = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def distance_cutoff(self):
        return self._distance_cutoff

    @distance_cutoff.setter
    def distance_cutoff(self, value):
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._distance_cutoff = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def distance_matrix_method(self):
        return self._distance_matrix_method

    @distance_matrix_method.setter
    def distance_matrix_method(self, value):
        self.distance_matrix = get_distance_matrix_from_pdb(self._pdb_file, self._chain, value)
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
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
            create_alignment_jackhmmer(self.sequence, self.pdb_name,
                                       output_file="dcaf_{}_alignment.sto".format(self.pdb_name))
            convert_and_filter_alignment(self.pdb_name)
            compute_plm(self.pdb_name)
            raise ValueError("Need to generate potts model")
        else:
            self.potts_model = load_potts_model(value)
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

    def native_energy(self, sequence=None):
        if sequence is None:
            if self._native_energy:
                return self._native_energy
            else:
                return compute_native_energy(self.sequence, self.potts_model, self.mask)
        else:
            return compute_native_energy(sequence, self.potts_model, self.mask)

    def decoy_fluctuation(self, kind='singleresidue'):
        if kind in self._decoy_fluctuation:
            return self._decoy_fluctuation[kind]
        if kind == 'singleresidue':
            fluctuation = compute_singleresidue_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)
        elif kind == 'mutational':
            fluctuation = compute_mutational_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)
        elif kind == 'configurational':
            fluctuation = compute_configurational_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)
        elif kind == 'contact':
            fluctuation = compute_contact_decoy_energy_fluctuation(self.sequence, self.potts_model, self.mask)

        else:
            raise Exception("Wrong kind of decoy generation selected")
        self._decoy_fluctuation[kind] = fluctuation
        return self._decoy_fluctuation[kind]

    def decoy_energy(self, kind='singleresidue'):
        return self.native_energy() + self.decoy_fluctuation(kind)

    def scores(self):
        return compute_scores(self.potts_model)

    def frustration(self, kind='singleresidue', aa_freq=None, correction=0):
        decoy_fluctuation = self.decoy_fluctuation(kind)
        if kind == 'singleresidue':
            if aa_freq is not None:
                aa_freq = self.aa_freq
            return compute_single_frustration(decoy_fluctuation, aa_freq, correction)
        elif kind in ['mutational', 'configurational', 'contact']:
            if aa_freq is not None:
                aa_freq = self.contact_freq
            return compute_pair_frustration(decoy_fluctuation, aa_freq, correction)

    def plot_decoy_energy(self, kind='singleresidue'):
        native_energy = self.native_energy()
        decoy_energy = self.decoy_energy(kind)
        if kind == 'singleresidue':
            plot_singleresidue_decoy_energy(decoy_energy, native_energy)

    def roc(self):
        return compute_roc(self.scores(), self.distance_matrix, self.distance_cutoff)

    def plot_roc(self):
        plot_roc(self.roc())

    def auc(self):
        """Computes area under the curve of the receiver-operating characteristic.
           Function intended"""
        return compute_auc(self.roc())

    def vmd(self, single='singleresidue', pair='mutational', aa_freq=None, correction=0, max_connections=100):
        tcl_script = write_tcl_script(self.pdb_file, self.chain,
                                      self.frustration(single, aa_freq=aa_freq, correction=correction),
                                      self.frustration(pair, aa_freq=aa_freq, correction=correction),
                                      max_connections=max_connections)
        call_vmd(self.pdb_file, tcl_script)


class AWSEMFrustratometer(PottsModel):
    # AWSEM parameters
    r_min = .45
    r_max = .65
    r_minII = .65
    r_maxII = .95
    eta = 50  # eta actually has unit of nm^-1.
    eta_sigma = 7.0
    rho_0 = 2.6

    min_sequence_separation_rho = 2
    min_sequence_separation_contact = 10  # means j-i > 9

    eta_switching = 10
    k_contact = 4.184
    burial_kappa = 4.0
    burial_ro_min = [0.0, 3.0, 6.0]
    burial_ro_max = [3.0, 6.0, 9.0]

    gamma_se_map_3_letters = {'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
                              'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
                              'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
                              'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19}
    burial_gamma = np.fromfile(f'{_path}/data/burial_gamma').reshape(20, 3)
    gamma_ijm = np.fromfile(f'{_path}/data/gamma_ijm').reshape(2, 20, 20)
    water_gamma_ijm = np.fromfile(f'{_path}/data/water_gamma_ijm').reshape(2, 20, 20)
    protein_gamma_ijm = np.fromfile(f'{_path}/data/protein_gamma_ijm').reshape(2, 20, 20)
    q = 20
    aa_map_awsem = [0, 0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18]
    aa_map_awsem_x, aa_map_awsem_y = np.meshgrid(aa_map_awsem, aa_map_awsem, indexing='ij')

    def __init__(self,
                 pdb_file,
                 chain=None,
                 sequence_cutoff=None):
        self.pdb_file = pdb_file
        self.chain = chain
        self._sequence = get_protein_sequence_from_pdb(self.pdb_file, self.chain)
        self.structure = prody.parsePDB(self.pdb_file)
        selection_CB = self.structure.select('name CB or (resname GLY and name CA)')
        resid = selection_CB.getResindices()
        self.N = len(resid)
        resname = [self.gamma_se_map_3_letters[aa] for aa in selection_CB.getResnames()]

        coords = selection_CB.getCoords()
        r = sdist.squareform(sdist.pdist(coords)) / 10
        distance_mask = ((r < 1) - np.eye(len(r)))
        sequence_mask_rho = abs(np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= self.min_sequence_separation_rho
        sequence_mask_contact = abs(
            np.expand_dims(resid, 0) - np.expand_dims(resid, 1)) >= self.min_sequence_separation_contact
        mask = ((r < 1) - np.eye(len(r)))
        rho = 0.25 * (1 + np.tanh(self.eta * (r - self.r_min))) * \
              (1 + np.tanh(self.eta * (self.r_max - r))) * sequence_mask_rho
        rho_r = (rho).sum(axis=1)
        rho_b = np.expand_dims(rho_r, 1)
        rho1 = np.expand_dims(rho_r, 0)
        rho2 = np.expand_dims(rho_r, 1)
        sigma_water = 0.25 * (1 - np.tanh(self.eta_sigma * (rho1 - self.rho_0))) * (
                1 - np.tanh(self.eta_sigma * (rho2 - self.rho_0)))
        sigma_protein = 1 - sigma_water
        theta = 0.25 * (1 + np.tanh(self.eta * (r - self.r_min))) * (1 + np.tanh(self.eta * (self.r_max - r)))
        thetaII = 0.25 * (1 + np.tanh(self.eta * (r - self.r_minII))) * (1 + np.tanh(self.eta * (self.r_maxII - r)))
        burial_indicator = np.tanh(self.burial_kappa * (rho_b - self.burial_ro_min)) + \
                           np.tanh(self.burial_kappa * (self.burial_ro_max - rho_b))
        J_index = np.meshgrid(range(self.N), range(self.N), range(self.q), range(self.q), indexing='ij', sparse=False)
        h_index = np.meshgrid(range(self.N), range(self.q), indexing='ij', sparse=False)

        burial_energy = -0.5 * self.k_contact * self.burial_gamma[h_index[1]] * burial_indicator[:, np.newaxis, :]
        direct = self.gamma_ijm[0, J_index[2], J_index[3]] * theta[:, :, np.newaxis, np.newaxis]

        water_mediated = thetaII[:, :, np.newaxis, np.newaxis] * sigma_water[:, :, np.newaxis, np.newaxis] * \
                         self.water_gamma_ijm[0, J_index[2], J_index[3]]
        protein_mediated = thetaII[:, :, np.newaxis, np.newaxis] * sigma_protein[:, :, np.newaxis, np.newaxis] * \
                           self.protein_gamma_ijm[0, J_index[2], J_index[3]]
        contact_energy = -self.k_contact * np.array([direct, water_mediated, protein_mediated]) * \
                         sequence_mask_contact[np.newaxis, :, :, np.newaxis, np.newaxis]

        # Set parameters
        self._distance_cutoff = 10
        self._sequence_cutoff = 2

        # Compute fast properties
        self.distance_matrix = r * 10
        self.potts_model = {}
        self.potts_model['h'] = -burial_energy.sum(axis=-1)[:, self.aa_map_awsem]
        self.potts_model['J'] = -contact_energy.sum(axis=0)[:, :, self.aa_map_awsem_x, self.aa_map_awsem_y]
        self.aa_freq = compute_aa_freq(self.sequence)
        self.contact_freq = compute_contact_freq(self.sequence)
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}
        #
        # def __init__(self,
        #              pdb_file: str,
        #              chain: str,
        #              potts_model_file: str,
        #              sequence_cutoff: typing.Union[float, None],
        #              distance_cutoff: typing.Union[float, None],
        #              distance_matrix_method='minimum'
        #              ):
        #     self.pdb_file = pdb_file
        #     self.chain = chain
        #     self.sequence = get_protein_sequence_from_pdb(self.pdb_file, self.chain)
        #
        #     # Set parameters
        #     self._potts_model_file = potts_model_file
        #     self._sequence_cutoff = sequence_cutoff
        #     self._distance_cutoff = distance_cutoff
        #     self._distance_matrix_method = distance_matrix_method
        #
        #     # Compute fast properties
        #     self.distance_matrix = get_distance_matrix_from_pdb(self.pdb_file, self.chain, self.distance_matrix_method)
        #     self.potts_model = load_potts_model(self.potts_model_file)
        #     self.aa_freq = compute_aa_freq(self.sequence)
        #     self.contact_freq = compute_contact_freq(self.sequence)
        #     self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        #
        #     # Initialize slow properties
        #     self._native_energy = None
        #     self._decoy_fluctuation = {}
