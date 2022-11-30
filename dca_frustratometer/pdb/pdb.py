from Bio.PDB import PDBParser
import prody
import scipy.spatial.distance as sdist
import pandas as pd
import numpy as np
import itertools


def download(pdbID: str):
    """
    Downloads a single pdb file
    """
    import urllib.request
    urllib.request.urlretrieve('http://www.rcsb.org/pdb/files/%s.pdb' % pdbID, "%s%s.pdb" % (directory, pdbID))

def get_sequence(pdb_file: str, 
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
    structure = parser.get_structure('name', pdb_file)
    Letter_code = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                   'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                   'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                   'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
                   'NGP': 'A', 'IPR': 'P', 'IGL': 'G'}
    residues = [residue for residue in structure.get_residues() if (
            residue.has_id('CA') and residue.get_parent().get_id() == str(chain) and residue.resname not in [' CA',
                                                                                                             'PBC'])]
    sequence = ''.join([Letter_code[r.resname] for r in residues])
    return sequence

def get_distance_matrix(pdb_file: str, 
                        chain: str, 
                        method: str = 'minimum'
                        ) -> np.array:
    """
    Get a residue distance matrix from a pdb protein
    :param pdb_file: PDB file location
    :param chain: chain name of PDB file to get sequence
    :param method: method to calculate the distance between residue [minimum, CA, CB]
    :return: distance matrix
    """
    '''Returns the distance matrix of the aminoacids on a sequence. The distance used is
    the minimum distance between two residues (for example the distance of the atoms on a H-bond)'''
    structure = prody.parsePDB(pdb_file)
    if method == 'CA':
        selection = structure.select('protein and name CA', chain=chain)
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'CB':
        selection = structure.select('protein and (name CB) or (resname GLY and name CA)', chain=chain)
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        return distance_matrix
    elif method == 'minimum':
        selection = structure.select('protein', chain=chain)
        distance_matrix = sdist.squareform(sdist.pdist(selection.getCoords()))
        resids = selection.getResindices()
        residues = pd.Series(resids).unique()
        selections = np.array([resids == a for a in residues])
        dm = np.zeros((len(residues), len(residues)))
        for i, j in itertools.combinations(range(len(residues)), 2):
            d = distance_matrix[selections[i]][:, selections[j]].min()
            dm[i, j] = d
            dm[j, i] = d
        return dm

