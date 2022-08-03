"""Provide the primary functions."""
import typing

from Bio.PDB import PDBParser
import prody
import scipy.spatial.distance as sdist
import pandas as pd
import numpy as np
import itertools
import scipy.io

def get_protein_sequence_from_pdb(pdb: str,
                                  chain: str
                                  ) -> str :
    """
    Get a protein sequence from a PDB file
    :param pdb: PDB file location
    :param chain: chain name of PDB file to get sequence
    :return: protein sequence
    """

    parser = PDBParser()
    structure=parser.get_structure('name',pdb)
    protein_residues = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                        'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                        'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                        'SER', 'THR', 'TRP', 'TYR', 'VAL'}
    Letter_code = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                   'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                   'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                   'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}
    residues=[residue for residue in structure.get_residues() if (residue.has_id('CA') and  residue.get_parent().get_id()==str(chain) and residue.resname not in [' CA','PBC'])]
    sequence=''.join([Letter_code[r.resname] for r in residues])
    return sequence


def get_distance_matrix_from_pdb(pdb: str,
                                 chain: str,
                                 method: str = 'minimum'
                                 ) -> np.array :
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
        distance_matrix = pd.DataFrame(data=distance_matrix, columns=range(len(distance_matrix)), index=range(len(distance_matrix)), dtype=float)
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
                          potts_model_file: str,
                          distance_matrix: np.array,
                          distance_cutoff: typing.Union[float, None] = None,
                          sequence_distance_cutoff: typing.Union[int, None] = None):

    AA = '-ACDEFGHIKLMNPQRSTVWY'

    potts_model = load_potts_model(potts_model_file)

    seq_index = np.array([AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    h = -potts_model['h'][range(seq_len), seq_index]
    j = -potts_model['J'][range(seq_len), :, seq_index, :][:, range(seq_len), seq_index]

    mask=np.ones([seq_len,seq_len])
    if sequence_distance_cutoff is not None:
        sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
        mask *= sequence_distance > sequence_distance_cutoff
    if distance_cutoff is not None:

        mask *= distance_matrix > distance_cutoff
    j_prime = j * (sequence_distance > sequence_distance_cutoff) * (distance_matrix <= distance_cutoff)
    energy = h.sum() + j_prime.sum() / 2
    return energy

def compute_frustration(with_attribution=True):
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

#Function if script invoked on its own
def main():
    seq = get_protein_sequence_from_pdb('examples/data/1l63.pdb', 'A')
    distance_matrix = get_distance_matrix_from_pdb('examples/data/1l63.pdb', 'A')
    e = compute_native_energy(seq, 'examples/data/PottsModel1l63A.mat', distance_matrix, distance_cutoff=4, sequence_distance_cutoff=0)
    print(e)


if __name__ == "__main__":
    main()

