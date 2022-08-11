"""Provide the primary functions."""
import typing

from Bio.PDB import PDBParser
from Bio import AlignIO
import prody
import scipy.spatial.distance as sdist
import pandas as pd
import numpy as np
import itertools
from datetime import datetime
import os
import urllib.request
import scipy.io


def get_protein_sequence_from_pdb(pdb: str,
                                  chain: str
                                  ) -> str:
    """
    Get a protein sequence from a pdb file

    Parameters
    ----------
    pdb : str,
        PDB file location.
    chain: str,
        Chain ID of the selected protein.

    Returns
    -------
    sequence : str
        Protein sequence.
    """
    """
    Get a protein sequence from a PDB file
    
    :param pdb: PDB file location
    :param chain: chain name of PDB file to get sequence
    :return: protein sequence
    """

    parser = PDBParser()
    structure = parser.get_structure('name', pdb)
    protein_residues = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL'}
    Letter_code = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                   'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                   'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                   'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    residues = [residue for residue in structure.get_residues() if (
            residue.has_id('CA') and residue.get_parent().get_id() == str(chain) and residue.resname not in [' CA',
                                                                                                             'PBC'])]
    sequence = ''.join([Letter_code[r.resname] for r in residues])
    return sequence


def get_distance_matrix_from_pdb(pdb: str,
                                 chain: str,
                                 method: str = 'minimum'
                                 ) -> np.array:
    """
    Get a residue distance matrix from a pdb protein
    :param pdb: PDB file location
    :param chain: chain name of PDB file to get sequence
    :param method: method to calculate the distance between residue [minimum, CA, CB]
    :return: distance matrix
    """
    '''Returns the distance matrix of the aminoacids on a sequence. The distance used is
    the minimum distance between two residues (for example the distance of the atoms on a H-bond)'''
    structure = prody.parsePDB(pdb)
    if method == 'CA':
        selection = structure.select('protein and name CA', chain=chain)
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'CB':
        selection = structure.select('protein and name CB', chain=chain)
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'minimum':
        selection = structure.select('protein', chain=chain)
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        distance_matrix = pd.DataFrame(data=distance_matrix, columns=range(len(distance_matrix)),
                                       index=range(len(distance_matrix)), dtype=float)
        residues = pd.Series(selection.getResindices()).unique()
        D = np.zeros((len(residues), len(residues))) + 1000
        for ij, ab in zip(itertools.combinations(range(len(residues)), 2), itertools.combinations(residues, 2)):
            i, j = ij
            a, b = ab
            d = distance_matrix.iloc[selection.getResindices() == a, selection.getResindices() == b].values.min()
            D[i, j] = d
            D[j, i] = d
        return D


def pseudofam_pfam_download_alignments(protein_family, alignment_type):
    if alignment_type == "both":
        all_alignment_types = ["seed", "full"]
    else:
        all_alignment_types = [alignment_type]

    for category in all_alignment_types:
        alignment_file_name = "%s_%s.aln" % (protein_family, category)
        os.system(f"wget -O {alignment_file_name} http://pfam.xfam.org/family/{protein_family}/alignment/{category}")


def pseudofam_retrieve_and_filter_alignment(file_name):
    # Convert full MSA in stockholm format to fasta format
    input_handle = open(f"{file_name}_full.aln", "rU")
    output_handle = open(f"{file_name}_msa.fasta", "w")
    alignments = AlignIO.parse(input_handle, "stockholm")
    AlignIO.write(alignments, output_handle, "fasta")
    output_handle.close()
    input_handle.close()

    # Remove inserts and columns that are completely composed of gaps from MSA
    alignment = AlignIO.read(open(f"{file_name}_msa.fasta"), "fasta")
    output_handle = open(f"{file_name}_gap_filtered_msa.fasta", "w")

    index_mask = []
    for i, record in enumerate(alignment):
        index_mask += [i for i, x in enumerate(list(record.seq)) if x != x.upper()]
    for i in range(len(alignment[0].seq)):
        if alignment[:, i] == ''.join(["-"] * len(alignment)):
            index_mask.append(i)
    index_mask = sorted(list(set(index_mask)))

    for i, record in enumerate(alignment):
        aligned_sequence = [list(record.seq)[i] for i in range(len(list(record.seq))) if i not in index_mask]

        output_handle.write(">%s\n" % record.id + "".join(aligned_sequence) + '\n')
    output_handle.close()


def generate_protein_sequence_alignments(protein_family, pdb_name, build_msa_files,
                                         database_name, dca_frustratometer_directory):
    if not os.path.exists(f"{protein_family}_full.aln"):
        if build_msa_files:
            if database_name == "Uniparc":
                database_file = f"{dca_frustratometer_directory}/uniparc_active.fasta"
            else:
                database_file = f"{dca_frustratometer_directory}/Uniprot_Sequence_Database_Files/uniprot_sprot.fasta"
            os.system(
                f"jackhmmer -A {protein_family}_{pdb_name}_{chain_name}_full.aln -N 1 --popen 0 --pextend 0 --chkhmm {protein_family}_{pdb_name}_{chain_name} --chkali {protein_family}_{pdb_name}_{chain_name} {protein_family}_{pdb_name}_{chain_name}_sequences.fasta {database_file}")
            file_name = f"{protein_family}_{pdb_name}_{chain_name}"
        else:
            pseudofam_pfam_download_alignments([protein_family], alignment_type="both",
                                               pfam_alignment_path=f"{os.getcwd()}/")
            file_name = protein_family
    # Reformat and filter MSA file
    pseudofam_retrieve_and_filter_alignment(file_name)


def generate_potts_model(file_name, gap_threshold, DCA_Algorithm, alignment_dca_files_directory,
                         dca_frustratometer_directory):
    if not os.path.exists(f"{file_name}_msa_dca_gap_threshold_{gap_threshold}.mat"):
        import matlab.engine
        eng = matlab.engine.start_matlab()

        if DCA_Algorithm == "mfDCA":
            os.system(f"cp {dca_frustratometer_directory}/DCA_Algorithms/mfDCA/DCAparameters.m .")
            eng.DCAparameters(f"{file_name}_gap_filtered_msa.fasta", 1, 1.0)
        else:
            os.chdir(f"{dca_frustratometer_directory}/DCA_Algorithms/plmDCA-master/plmDCA_asymmetric_v2")
            eng.plmDCA_asymmetric(f"{alignment_dca_files_directory}/{file_name}_gap_filtered_msa.fasta",
                                  f"{alignment_dca_files_directory}/dca.mat",
                                  0.2, 1)
            os.chdir(alignment_dca_files_directory)

        os.system(f"mv dca.mat {file_name}_msa_dca_gap_threshold_{gap_threshold}.mat")


def load_potts_model(potts_model_file):
    return scipy.io.loadmat(potts_model_file)


def compute_mask(distance_matrix: np.array,
                 distance_cutoff: typing.Union[float, None] = None,
                 sequence_distance_cutoff: typing.Union[int, None] = None) -> np.array:
    seq_len = len(distance_matrix)
    mask = np.ones([seq_len, seq_len])
    if sequence_distance_cutoff is not None:
        sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
        mask *= sequence_distance > sequence_distance_cutoff
    if distance_cutoff is not None:
        mask *= distance_matrix <= distance_cutoff
    return mask.astype(np.bool8)


def compute_native_energy(seq: str,
                          potts_model: dict,
                          mask: np.array) -> float:
    AA = '-ACDEFGHIKLMNPQRSTVWY'

    seq_index = np.array([AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    #    h_sum=0
    #    for i in range(seq_len):
    #        h = -(potts_model['h'].T)[i, seq_index[i]]
    #        h_sum+=h
    h = -potts_model['h'][range(seq_len), seq_index]
    j = -potts_model['J'][range(seq_len), :, seq_index, :][:, range(seq_len), seq_index]

    j_prime = j * mask
    energy = h.sum() + j_prime.sum() / 2
    return energy


def compute_singleresidue_decoy_energy_fluctuation(seq: str,
                                                   potts_model: dict,
                                                   mask: np.array) -> np.array:
    AA = '-ACDEFGHIKLMNPQRSTVWY'

    seq_index = np.array([AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, aa1 = np.meshgrid(np.arange(seq_len), np.arange(21), indexing='ij', sparse=True)

    decoy_energy = np.zeros([seq_len, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1

    j_correction = np.zeros([seq_len, seq_len, 21])
    # J correction interactions with other aminoacids
    reduced_j = potts_model['J'][range(seq_len), :, seq_index, :].astype(np.float32)
    j_correction += reduced_j[:, pos1, seq_index[pos1]] * mask[:, pos1]
    j_correction -= reduced_j[:, pos1, aa1] * mask[:, pos1]

    # J correction, interaction with self aminoacids
    decoy_energy += j_correction.sum(axis=0)

    return decoy_energy


def compute_mutational_decoy_energy_fluctuation(seq: str,
                                                potts_model: dict,
                                                mask: np.array, ) -> np.array:
    AA = '-ACDEFGHIKLMNPQRSTVWY'

    seq_index = np.array([AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, pos2, aa1, aa2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), np.arange(21), np.arange(21),
                                       indexing='ij', sparse=True)

    decoy_energy = np.zeros([seq_len, seq_len, 21, 21])
    decoy_energy -= (potts_model['h'][pos1, aa1] - potts_model['h'][pos1, seq_index[pos1]])  # h correction aa1
    decoy_energy -= (potts_model['h'][pos2, aa2] - potts_model['h'][pos2, seq_index[pos2]])  # h correction aa2

    j_correction = np.zeros([seq_len, seq_len, 21, 21])
    for pos, aa in enumerate(seq_index):
        # J correction interactions with other aminoacids
        reduced_j = potts_model['J'][pos, :, aa, :].astype(np.float32)
        j_correction += reduced_j[pos1, seq_index[pos1]] * mask[pos, pos1]
        j_correction -= reduced_j[pos1, aa1] * mask[pos, pos1]
        j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= reduced_j[pos2, aa2] * mask[pos, pos2]
    # J correction, interaction with self aminoacids
    j_correction -= potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += potts_model['J'][pos1, pos2, aa1, seq_index[pos2]] * mask[pos1, pos2]  # Added mistakenly
    j_correction += potts_model['J'][pos1, pos2, seq_index[pos1], aa2] * mask[pos1, pos2]  # Added mistakenly
    j_correction -= potts_model['J'][pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # Correct combination
    decoy_energy += j_correction

    return decoy_energy


def compute_singleresidue_decoy_energy(seq: str,
                                       potts_model: dict,
                                       mask: np.array, ):
    return compute_native_energy(seq, potts_model, mask) + compute_singleresidue_decoy_energy_fluctuation(seq,
                                                                                                          potts_model,
                                                                                                          mask)


def compute_mutational_decoy_energy(seq: str,
                                    potts_model: dict,
                                    mask: np.array, ):
    return compute_native_energy(seq, potts_model, mask) + compute_mutational_decoy_energy_fluctuation(seq,
                                                                                                       potts_model,
                                                                                                       mask)


def compute_aa_freq(sequence):
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq_index = np.array([AA.find(aa) for aa in sequence])
    return np.array([(seq_index == i).sum() for i in range(21)])


def compute_contact_freq(sequence):
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq_index = np.array([AA.find(aa) for aa in sequence])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)], dtype=np.float64)
    aa_freq /= aa_freq.sum()
    contact_freq = (aa_freq[:, np.newaxis] * aa_freq[np.newaxis, :])
    return contact_freq


def compute_singleresidue_frustration(decoy_energy, native_energy, aa_freq=None):
    if aa_freq is None:
        aa_freq = np.ones(21)
    mean_energy = (aa_freq * decoy_energy).sum(axis=1) / aa_freq.sum()
    std_energy = np.sqrt(((aa_freq * (decoy_energy - mean_energy[:, np.newaxis]) ** 2) / aa_freq.sum()).sum(axis=1))
    frustration = (native_energy - mean_energy) / std_energy
    return frustration


def compute_mutational_frustration(decoy_energy: np.array,
                                   native_energy: float,
                                   contact_freq: typing.Union[None, np.array]) -> np.array:
    """
    Computes mutational frustration
    Parameters
    ----------
    :param decoy_energy:
    :param native_energy:
    :param contact_freq:
    Returns
    -------
    :return:
    """
    if contact_freq is None:
        contact_freq = np.ones([21, 21])
    seq_len = decoy_energy.shape[0]
    average = np.average(decoy_energy.reshape(seq_len * seq_len, 21 * 21), weights=contact_freq.flatten(), axis=-1)
    variance = np.average((decoy_energy.reshape(seq_len * seq_len, 21 * 21) - average[:, np.newaxis]) ** 2,
                          weights=contact_freq.flatten(), axis=-1)
    mean_energy = average.reshape(seq_len, seq_len)
    std_energy = np.sqrt(variance).reshape(seq_len, seq_len)
    frustration = (native_energy - mean_energy) / std_energy
    return frustration


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


# Class wrapper
class PottsModel:
    def __init__(self,
                 pdb_file: str,
                 chain: str,
                 potts_model_file: str,
                 sequence_cutoff: typing.Union[float, None],
                 distance_cutoff: typing.Union[float, None],
                 distance_matrix_method='minimum'
                 ):
        self.pdb = pdb_file
        self.chain = chain
        self.potts_model_file = potts_model_file
        self.sequence = get_protein_sequence_from_pdb(self.pdb, self.chain)
        self.distance_matrix = get_distance_matrix_from_pdb(self.pdb, self.chain, distance_matrix_method)
        self.potts_model = load_potts_model(potts_model_file)
        self.sequence_cutoff = sequence_cutoff
        self.distance_cutoff = distance_cutoff
        self.aa_freq = compute_aa_freq(self.sequence)
        self.contact_freq = compute_contact_freq(self.sequence)

    def native_energy(self):
        mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        return compute_native_energy(self.sequence, self.potts_model, mask)

    def decoy_energy(self, type='singleresidue'):
        mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        if type == 'singleresidue':
            return compute_singleresidue_decoy_energy(self.sequence, self.potts_model, mask)
        elif type == 'mutational':
            return compute_mutational_decoy_energy(self.sequence, self.potts_model, mask)

    def frustration(self, type='singleresidue', aa_freq=None):
        native_energy = self.native_energy()
        decoy_energy = self.decoy_energy(type)
        if type == 'singleresidue':
            if aa_freq is not None:
                aa_freq = self.aa_freq
            return compute_singleresidue_frustration(decoy_energy, native_energy, aa_freq)
        elif type == 'mutational':
            if aa_freq is not None:
                aa_freq = self.contact_freq
            return compute_mutational_frustration(decoy_energy, native_energy, aa_freq)


# Function if script invoked on its own
def main(pdb_name, chain_name, atom_type, DCA_Algorithm, build_msa_files, database_name,
         gap_threshold, dca_frustratometer_directory):
    # PDB DCA frustration analysis directory
    protein_dca_frustration_directory = f"{os.getcwd()}/{datetime.today().strftime('%m_%d_%Y')}_{pdb_name}_{chain_name}_DCA_Frustration_Analysis"
    if not os.path.exists(protein_dca_frustration_directory):
        os.mkdir(protein_dca_frustration_directory)
    os.chdir(protein_dca_frustration_directory)
    ###
    # Importing PDB structure
    if not os.path.exists(f"./{pdb_name[:4]}.pdb"):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_name[:4]}.pdb", f"./{pdb_name[:4]}.pdb")

    pdb_sequence = get_protein_sequence_from_pdb(f"./{pdb_name[:4]}.pdb", chain_name)

    # Identify PDB's protein family
    pdb_pfam_mapping_dataframe = pd.read_csv(f"{dca_frustratometer_directory}/pdb_chain_pfam.csv", header=1, sep=",")
    protein_family = pdb_pfam_mapping_dataframe.loc[((pdb_pfam_mapping_dataframe["PDB"] == pdb_name.lower())
                                                     & (pdb_pfam_mapping_dataframe[
                                                            "CHAIN"] == chain_name)), "PFAM_ID"].values[0]

    # Save PDB sequence
    with open(f"./{protein_family}_{pdb_name}_{chain_name}_sequences.fasta", "w") as f:
        f.write(">{}_{}\n{}\n".format(pdb_name, chain_name, pdb_sequence))

    # Generate PDB contact distance matrix
    distance_matrix = get_distance_matrix_from_pdb(f"./{pdb_name[:4]}.pdb", chain_name)

    # Generate PDB alignment files
    alignment_dca_files_directory = f"{protein_dca_frustration_directory}/{datetime.today().strftime('%m_%d_%Y')}_{protein_family}_PFAM_Alignment_DCA_Files"
    file_name = protein_family
    if build_msa_files:
        alignment_dca_files_directory = f"{protein_dca_frustration_directory}/{datetime.today().strftime('%m_%d_%Y')}_{protein_family}_{pdb_name}_{chain_name}_Jackhmmer_Alignment_DCA_Files"
        file_name = f"{protein_family}_{pdb_name}_{chain_name}"

    if not os.path.exists(alignment_dca_files_directory):
        os.mkdir(alignment_dca_files_directory)

    os.system(f"cp ./{protein_family}_{pdb_name}_{chain_name}_sequences.fasta {alignment_dca_files_directory}")
    os.chdir(alignment_dca_files_directory)

    generate_protein_sequence_alignments(protein_family, pdb_name, build_msa_files,
                                         database_name,
                                         dca_frustratometer_directory)

    generate_potts_model(file_name, gap_threshold, DCA_Algorithm,
                         alignment_dca_files_directory, dca_frustratometer_directory)

    # Compute PDB sequence native DCA energy
    os.chdir(protein_dca_frustration_directory)

    potts_model = load_potts_model(
        f"{alignment_dca_files_directory}/{file_name}_msa_dca_gap_threshold_{gap_threshold}.mat")
    mask = compute_mask(distance_matrix, distance_cutoff=4, sequence_distance_cutoff=0)
    e = compute_native_energy(pdb_sequence, potts_model, mask)
    print(e)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_name", type=str, required=True, help="PDB Name")
    parser.add_argument("--chain_name", type=str, required=True, help="Chain Name")
    parser.add_argument("--atom_type", type=str, default="CB", help="Atom Type Used for Residue Contact Map")
    parser.add_argument("--DCA_Algorithm", type=str, default="mfDCA",
                        help="DCA Algorithm Used (options=mfDCA or plmDCA)")
    parser.add_argument("--build_msa_files", action='store_false', help="Build MSA with Full Coverage of PDB")
    parser.add_argument("--database_name", default="Uniprot",
                        help="Database used in seed msa (options are Uniparc or Uniprot)")
    parser.add_argument("--gap_threshold", type=float, default=0.2, help="Continguous gap threshold applied to MSA")
    parser.add_argument("--dca_frustratometer_directory", type=str, help="Directory of DCA Frustratometer Scripts")

    args = parser.parse_args()

    pdb_name = args.pdb_name
    chain_name = args.chain_name
    atom_type = args.atom_type
    DCA_Algorithm = args.DCA_Algorithm
    build_msa_files = args.build_msa_files
    database_name = args.database_name
    gap_threshold = args.gap_threshold
    dca_frustratometer_directory = args.dca_frustratometer_directory

    main(pdb_name, chain_name, atom_type, DCA_Algorithm, build_msa_files, database_name,
         gap_threshold, dca_frustratometer_directory)
