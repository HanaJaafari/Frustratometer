"""Provide the primary functions."""
import typing

from Bio.PDB import PDBParser
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


def load_potts_model(potts_model_file):
    return scipy.io.loadmat(potts_model_file)


def compute_native_energy(seq: str,
                          potts_model: str,
                          distance_matrix: np.array,
                          distance_cutoff: typing.Union[float, None] = None,
                          sequence_distance_cutoff: typing.Union[int, None] = None) -> float:
    AA = '-ACDEFGHIKLMNPQRSTVWY'

    seq_index = np.array([AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    h = -potts_model['h'][range(seq_len), seq_index]
    j = -potts_model['J'][range(seq_len), :, seq_index, :][:, range(seq_len), seq_index]

    mask = np.ones([seq_len, seq_len])
    if sequence_distance_cutoff is not None:
        sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
        mask *= sequence_distance > sequence_distance_cutoff
    if distance_cutoff is not None:
        mask *= distance_matrix <= distance_cutoff
    j_prime = j * mask
    energy = h.sum() + j_prime.sum() / 2
    return energy


def compute_singleresidue_decoy_energy(seq: str,
                                       potts_model: str,
                                       distance_matrix: np.array,
                                       distance_cutoff: typing.Union[float, None] = None,
                                       sequence_distance_cutoff: typing.Union[int, None] = None) -> np.array:
    AA = '-ACDEFGHIKLMNPQRSTVWY'

    seq_index = np.array([AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    decoys = np.repeat(np.repeat(np.array(seq_index)[np.newaxis, np.newaxis, :], 21, 1), seq_len, 0)
    for i in range(21):
        decoys[range(seq_len), i, range(seq_len)] = i
    # Position of mutation, Mutation, position in sequence
    mut_pos, mut_aa, seq_pos = np.meshgrid(range(seq_len), range(21), range(seq_len), indexing='ij')

    # Compute energy
    h_decoy = -potts_model['h'][seq_pos, decoys]
    j_decoy = -potts_model['J'][seq_pos, :, decoys, :][mut_pos, mut_aa, :, seq_pos, decoys]
    mask = np.ones([seq_len, seq_len])
    if sequence_distance_cutoff is not None:
        sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
        mask *= sequence_distance > sequence_distance_cutoff
    if distance_cutoff is not None:
        mask *= distance_matrix <= distance_cutoff
    j_decoy_prime = j_decoy * mask
    decoy_energy = h_decoy.sum(axis=-1) + j_decoy_prime.sum(axis=-1).sum(axis=-1) / 2
    return decoy_energy


def compute_aa_freq(sequence):
    AA = '-ACDEFGHIKLMNPQRSTVWY'
    seq_index = np.array([AA.find(aa) for aa in sequence])
    return np.array([(seq_index == i).sum() for i in range(21)])


def compute_singleresidue_frustration(decoy_energy, native_energy, aa_freq=None):
    if aa_freq is None:
        aa_freq = np.ones(21)
    mean_energy = (aa_freq * decoy_energy).sum(axis=1) / aa_freq.sum()
    std_energy = np.sqrt(((aa_freq * (decoy_energy - mean_energy[:, np.newaxis]) ** 2) / aa_freq.sum()).sum(axis=1))
    frustration = (native_energy - mean_energy) / std_energy
    return frustration


def compute_mutational_frustration():
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

    def native_energy(self):
        return compute_native_energy(self.sequence, self.potts_model, self.distance_matrix,
                                     self.distance_cutoff, self.sequence_cutoff)

    def decoy_energy(self, type='singleresidue'):
        if type == 'singleresidue':
            return compute_singleresidue_decoy_energy(self.seq, self.potts_model, self.distance_matrix,
                                                      self.distance_cutoff, self.sequence_cutoff)

    def frustration(self, type='singleresidue'):
        native_energy = self.native_energy()
        decoy_energy = self.decoy_energy(type)
        if type == 'singleresidue':
            return compute_singleresidue_frustration(decoy_energy, native_energy, self.aa_freq)

# Function if script invoked on its own
def main(pdb_name,chain_name,atom_type,PFAM_version,build_msa_files,database_name,
         gap_threshold,dca_frustratometer_directory):
    #PDB DCA frustration analysis directory
    protein_dca_frustration_directory=f"{os.getcwd()}/{datetime.today().strftime('%m_%d_%Y')}_{pdb_name}_{chain_name}_DCA_Frustration_Analysis"
    if not os.path.exists(protein_dca_frustration_directory):
        os.mkdir(protein_dca_frustration_directory)
    os.chdir(protein_dca_frustration_directory)
    ###
    #Importing PDB structure
    if not os.path.exists(f"./{pdb_name[:4]}.pdb"):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_name[:4]}.pdb", f"./{pdb_name[:4]}.pdb")
    
    pdb_sequence = get_protein_sequence_from_pdb(f"./{pdb_name[:4]}.pdb", chain_name)
    
    #Identify PDB's protein family
    pdb_pfam_mapping_dataframe=pd.read_csv(f"{dca_frustratometer_directory}/pdb_chain_pfam.csv",header=1,sep=",")
    protein_family=pdb_pfam_mapping_dataframe.loc[((pdb_pfam_mapping_dataframe["PDB"]==pdb_name.lower()) 
                                                  & (pdb_pfam_mapping_dataframe["CHAIN"]==chain_name)),"PFAM_ID"].values[0]
    
    #Save PDB sequence
    with open(f"./{protein_family}_{pdb_name}_{chain_name}_sequences.fasta","w") as f:
        f.write(">{}_{}\n{}\n".format(pdb_name,chain_name,pdb_sequence))
    
    #Generate PDB contact distance matrix
    distance_matrix = get_distance_matrix_from_pdb(f"./{pdb_name[:4]}.pdb", chain_name)

    #Generate PDB alignment files
    potts_model=load_potts_model('examples/data/PottsModel1l63A.mat')
    e = compute_native_energy(seq, 'examples/data/PottsModel1l63A.mat', distance_matrix,
                              distance_cutoff=4, sequence_distance_cutoff=0)
    print(e)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_name", type=str, required=True,help="PDB Name")
    parser.add_argument("--chain_name", type=str, required=True,help="Chain Name")
    parser.add_argument("--atom_type", type=str, default="CB",help="Atom Type Used for Residue Contact Map")
    parser.add_argument("--PFAM_version", action="store_true",default=27, help="PFAM version for DCA Files (Default is PFAM 27)")
    parser.add_argument("--build_msa_files",action='store_false',help="Build MSA with Full Coverage of PDB")
    parser.add_argument("--database_name", default="Uniprot",help="Database used in seed msa (options are Uniparc or Uniprot)")
    parser.add_argument("--gap_threshold",type=float,default=0.2,help="Continguous gap threshold applied to MSA")
    parser.add_argument("--dca_frustratometer_directory",type=str,help="Directory of DCA Frustratometer Scripts")
    
    args = parser.parse_args()
    
    pdb_name=args.pdb_name
    chain_name=args.chain_name
    atom_type=args.atom_type
    PFAM_version=args.PFAM_version
    build_msa_files=args.build_msa_files
    database_name=args.database_name
    gap_threshold=args.gap_threshold
    dca_frustratometer_directory=args.dca_frustratometer_directory
    
    main(pdb_name,chain_name,atom_type,PFAM_version,build_msa_files,database_name,
         gap_threshold,dca_frustratometer_directory)
