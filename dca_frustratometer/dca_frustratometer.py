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
import subprocess
from pathlib import Path

_AA = '-ACDEFGHIKLMNPQRSTVWY'
_path = Path(__file__).parent.absolute()


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


def get_distance_matrix_from_pdb(pdb_file: str,
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


def create_alignment_jackhmmer(fasta_file,
                               database=f'{_path}/Databases/uniparc_active.fasta',
                               output='results.sto'
                               ):
    # jackhmmer and databases required
    import subprocess
    subprocess.call(['jackhmmer', '-A', output, '--noali', '-E', '1E-8', '--incE', '1E-10',
                     fasta_file, database])


def remove_gaps():
    pass


def sto2fasta():
    pass


def compute_plm(fasta_alignment,
                outputfile,
                lambda_h=0.01,
                lambda_J=0.01,
                reweighting_threshold=0.1,
                nr_of_cores=1,
                outputDistribution="Distribution.txt",
                outputMatrix='matrix.mat'
                ):
    # MATLAB needs Bioinfomatics toolbox and Image processing toolbox to parse the sequences
    # Functions need to be compiled with 'matlab -nodisplay -r "mexAll"'
    try:
        import matlab.engine
    except ImportError:
        subprocess.call(['matlab', '-nodisplay', '-r',
                         "plmDCA_symmetric_mod7('3ks3A_nogaps20.fa','scores_3ks3A.txt',0.01,0.01,0.1,1,'PottsModel3ks3A.mat');quit"])
        return outputMatrix
    eng = matlab.engine.start_matlab()
    eng.addpath('%s/plm' % _path, nargout=0)
    eng.addpath('%s/plm/functions' % _path, nargout=0)
    eng.addpath('%s/plm/3rd_party_code/minFunc' % _path, nargout=0)
    print('plmDCA_symmetric_mod7', fasta_alignment, outputfile, lambda_h, lambda_J, reweighting_threshold, nr_of_cores,
          outputDistribution, outputMatrix)
    eng.plmDCA_symmetric_mod7(fasta_alignment, outputfile, lambda_h, lambda_J, reweighting_threshold, nr_of_cores,
                              outputDistribution, outputMatrix, nargout=0)  # , stdout=out )
    return outputMatrix


def pseudofam_pfam_download_alignments(protein_family, alignment_type,
                                       alignment_dca_files_directory):
    if alignment_type == "both":
        all_alignment_types = ["seed", "full"]
    else:
        all_alignment_types = [alignment_type]

    for category in all_alignment_types:
        alignment_file_name = f"{alignment_dca_files_directory}/{protein_family}_{category}.aln"
        subprocess.call(["wget", "-O", alignment_file_name,
                         f"http://pfam.xfam.org/family/{protein_family}/alignment/{category}"])


def pseudofam_retrieve_and_filter_alignment(file_name, alignment_dca_files_directory):
    # Convert full MSA in stockholm format to fasta format
    input_handle = open(f"{alignment_dca_files_directory}/{file_name}_full.aln", "rU")
    output_handle = open(f"{alignment_dca_files_directory}/{file_name}_msa.fasta", "w")

    alignments = AlignIO.parse(input_handle, "stockholm")
    AlignIO.write(alignments, output_handle, "fasta")
    output_handle.close()
    input_handle.close()

    # Remove inserts and columns that are completely composed of gaps from MSA
    alignment = AlignIO.read(open(f"{alignment_dca_files_directory}/{file_name}_msa.fasta"), "fasta")
    output_handle = open(f"{alignment_dca_files_directory}/{file_name}_gap_filtered_msa.fasta", "w")

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
                                         database_name, alignment_dca_files_directory,
                                         dca_frustratometer_directory):
    if not os.path.exists(f"{alignment_dca_files_directory}/{protein_family}_full.aln"):
        if build_msa_files:
            if database_name == "Uniparc":
                database_file = f"{dca_frustratometer_directory}/uniparc_active.fasta"
            else:
                database_file = f"{dca_frustratometer_directory}/Uniprot_Sequence_Database_Files/uniprot_sprot.fasta"
            subprocess.call(["jackhmmer", "-A",
                             f"{alignment_dca_files_directory}/{protein_family}_{pdb_name}_{chain_name}_full.aln",
                             "-N", "1", "--popen", "0", "--pextend", "0", "--chkhmm",
                             f"{protein_family}_{pdb_name}_{chain_name}", "--chkali",
                             f"{protein_family}_{pdb_name}_{chain_name}",
                             f"{alignment_dca_files_directory}/{protein_family}_{pdb_name}_{chain_name}_sequences.fasta",
                             database_file])
            file_name = f"{protein_family}_{pdb_name}_{chain_name}"
        else:
            alignment_type = "both"
            pseudofam_pfam_download_alignments(protein_family, alignment_type,
                                               alignment_dca_files_directory)
            file_name = protein_family
        # Reformat and filter MSA file
        pseudofam_retrieve_and_filter_alignment(file_name, alignment_dca_files_directory)


def generate_potts_model(file_name, gap_threshold, DCA_Algorithm, alignment_dca_files_directory,
                         dca_frustratometer_directory):
    if not os.path.exists(f"{alignment_dca_files_directory}/{file_name}_msa_dca_gap_threshold_{gap_threshold}.mat"):
        import matlab.engine
        eng = matlab.engine.start_matlab()

        if DCA_Algorithm == "mfDCA":
            subprocess.call(["cp",
                             f"{dca_frustratometer_directory}/DCA_Algorithms/mfDCA/DCAparameters.m",
                             alignment_dca_files_directory])
            os.chdir(alignment_dca_files_directory)
            eng.DCAparameters(f"{alignment_dca_files_directory}/{file_name}_gap_filtered_msa.fasta", 1, 1.0)
        else:
            os.chdir(f"{dca_frustratometer_directory}/DCA_Algorithms/plmDCA-master/plmDCA_asymmetric_v2")
            eng.plmDCA_asymmetric(f"{alignment_dca_files_directory}/{file_name}_gap_filtered_msa.fasta",
                                  f"{alignment_dca_files_directory}/dca.mat",
                                  0.2, 1)

        subprocess.call(["mv", f"{alignment_dca_files_directory}/dca.mat",
                         f"{alignment_dca_files_directory}/{file_name}_msa_dca_gap_threshold_{gap_threshold}.mat"])


def load_potts_model(potts_model_file):
    return scipy.io.loadmat(potts_model_file)


def compute_mask(distance_matrix: np.array,
                 distance_cutoff: typing.Union[float, None] = None,
                 sequence_distance_cutoff: typing.Union[int, None] = None) -> np.array:
    """
    Computes mask for couplings based on maximum distance cutoff and minimum sequence separation.
    The cutoffs are inclusive
    :param distance_matrix:
    :param distance_cutoff:
    :param sequence_distance_cutoff:
    :return:
    """
    seq_len = len(distance_matrix)
    mask = np.ones([seq_len, seq_len])
    if sequence_distance_cutoff is not None:
        sequence_distance = sdist.squareform(sdist.pdist(np.arange(seq_len)[:, np.newaxis]))
        mask *= sequence_distance >= sequence_distance_cutoff
    if distance_cutoff is not None:
        mask *= distance_matrix <= distance_cutoff
    return mask.astype(np.bool8)


def compute_native_energy(seq: str,
                          potts_model: dict,
                          mask: np.array) -> float:
    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)
    pos1, pos2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij', sparse=True)
    aa1, aa2 = np.meshgrid(seq_index, seq_index, indexing='ij', sparse=True)

    h = -potts_model['h'][range(seq_len), seq_index]
    j = -potts_model['J'][pos1, pos2, aa1, aa2]

    j_prime = j * mask
    energy = h.sum() + j_prime.sum() / 2
    return energy


def compute_singleresidue_decoy_energy_fluctuation(seq: str,
                                                   potts_model: dict,
                                                   mask: np.array) -> np.array:
    """
    $ \Delta H_i = \Delta h_i + \sum_k\Delta j_{ik} $

    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
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
    """
    $$ \Delta DCA_{ij} = H_i - H_{i'} + H_{j}-H_{j'}
    + J_{ij} -J_{ij'} + J_{i'j'} - J_{i'j}
    + \sum_k {J_{ik} - J_{i'k} + J_{jk} -J_{j'k}}
    $$
    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
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

def compute_configurational_decoy_energy_fluctuation(seq: str,
                                                potts_model: dict,
                                                mask: np.array, ) -> np.array:
    """
    $$ \Delta DCA_{ij} = H_i - H_{i'} + H_{j}-H_{j'}
    + J_{ij} -J_{ij'} + J_{i'j'} - J_{i'j}
    + \sum_k {J_{ik} - J_{i'k} + J_{jk} -J_{j'k}}
    $$
    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """
    seq_index = np.array([_AA.find(aa) for aa in seq])
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
        j_correction -= reduced_j[pos1, aa1] * mask.mean()
        j_correction += reduced_j[pos2, seq_index[pos2]] * mask[pos, pos2]
        j_correction -= reduced_j[pos2, aa2] * mask.mean()
    # J correction, interaction with self aminoacids
    j_correction -= potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Taken two times
    j_correction += potts_model['J'][pos1, pos2, aa1, seq_index[pos2]] * mask.mean()  # Added mistakenly
    j_correction += potts_model['J'][pos1, pos2, seq_index[pos1], aa2] * mask.mean()  # Added mistakenly
    j_correction -= potts_model['J'][pos1, pos2, aa1, aa2] * mask.mean()  # Correct combination
    decoy_energy += j_correction

    return decoy_energy


def compute_contact_decoy_energy_fluctuation(seq: str,
                                             potts_model: dict,
                                             mask: np.array) -> np.array:
    """
    $$ \Delta DCA_{ij} = \Delta j_{ij} $$
    :param seq:
    :param potts_model:
    :param mask:
    :return:
    """

    seq_index = np.array([_AA.find(aa) for aa in seq])
    seq_len = len(seq_index)

    # Create decoys
    pos1, pos2, aa1, aa2 = np.meshgrid(np.arange(seq_len), np.arange(seq_len), np.arange(21), np.arange(21),
                                       indexing='ij', sparse=True)

    decoy_energy = np.zeros([seq_len, seq_len, 21, 21])
    decoy_energy += potts_model['J'][pos1, pos2, seq_index[pos1], seq_index[pos2]] * mask[pos1, pos2]  # Old coupling
    decoy_energy -= potts_model['J'][pos1, pos2, aa1, aa2] * mask[pos1, pos2]  # New Coupling

    return decoy_energy


def compute_decoy_energy(seq: str, potts_model: dict, mask: np.array, kind='singleresidue'):
    """
    Calculates the decoy energy (Obsolete)
    :param seq:
    :param potts_model:
    :param mask:
    :param kind:
    :return:
    """

    native_energy = compute_native_energy(seq, potts_model, mask)
    if kind == 'singleresidue':
        return native_energy + compute_singleresidue_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'mutational':
        return native_energy + compute_mutational_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'configurational':
        return native_energy + compute_configurational_decoy_energy_fluctuation(seq, potts_model, mask)
    elif kind == 'contact':
        return native_energy + compute_contact_decoy_energy_fluctuation(seq, potts_model, mask)


def compute_aa_freq(sequence, include_gaps=True):
    seq_index = np.array([_AA.find(aa) for aa in sequence])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)])
    if not include_gaps:
        aa_freq[0] = 0
    return aa_freq


def compute_contact_freq(sequence):
    seq_index = np.array([_AA.find(aa) for aa in sequence])
    aa_freq = np.array([(seq_index == i).sum() for i in range(21)], dtype=np.float64)
    aa_freq /= aa_freq.sum()
    contact_freq = (aa_freq[:, np.newaxis] * aa_freq[np.newaxis, :])
    return contact_freq


def compute_single_frustration(decoy_fluctuation,
                               aa_freq=None,
                               correction=0):
    if aa_freq is None:
        aa_freq = np.ones(21)
    mean_energy = (aa_freq * decoy_fluctuation).sum(axis=1) / aa_freq.sum()
    std_energy = np.sqrt(
        ((aa_freq * (decoy_fluctuation - mean_energy[:, np.newaxis]) ** 2) / aa_freq.sum()).sum(axis=1))
    frustration = -mean_energy / (std_energy + correction)
    return frustration


def compute_pair_frustration(decoy_fluctuation,
                             contact_freq: typing.Union[None, np.array],
                             correction=0) -> np.array:
    if contact_freq is None:
        contact_freq = np.ones([21, 21])
    decoy_energy = decoy_fluctuation
    seq_len = decoy_fluctuation.shape[0]
    average = np.average(decoy_energy.reshape(seq_len * seq_len, 21 * 21), weights=contact_freq.flatten(), axis=-1)
    variance = np.average((decoy_energy.reshape(seq_len * seq_len, 21 * 21) - average[:, np.newaxis]) ** 2,
                          weights=contact_freq.flatten(), axis=-1)
    mean_energy = average.reshape(seq_len, seq_len)
    std_energy = np.sqrt(variance).reshape(seq_len, seq_len)
    contact_frustration = -mean_energy / (std_energy + correction)
    return contact_frustration


def compute_scores(potts_model: dict) -> np.array:
    """
    Computes contact scores based on the Frobenius norm

    CN[i,j] = F[i,j] - F[i,:] * F[:,j] / F[:,:]

    Parameters
    ----------
    potts_model :  dict
        Potts model containing the couplings in the "J" key

    Returns
    -------
    scores : np.array
        Score matrix (N x N)
    """
    j = potts_model['J']
    n, _, __, q = j.shape
    norm = np.linalg.norm(j.reshape(n * n, q * q), axis=1).reshape(n, n)  # Frobenius norm
    norm_mean = np.mean(norm, axis=0) / (n - 1) * n
    norm_mean_all = np.mean(norm) / (n - 1) * n
    corr_norm = norm - norm_mean[:, np.newaxis] * norm_mean[np.newaxis, :] / norm_mean_all
    corr_norm[np.diag_indices(n)] = 0
    corr_norm = np.mean([corr_norm, corr_norm.T], axis=0)  # Symmetrize matrix
    return corr_norm


def compute_roc(scores, distance_matrix, cutoff):
    scores = sdist.squareform(scores)
    distance = sdist.squareform(distance_matrix)
    results = np.array([scores, distance])
    results = results[:, results[0, :].argsort()[::-1]]  # Sort results by score
    contacts = results[1] <= cutoff
    not_contacts = ~contacts
    tpr = np.concatenate([[0], contacts.cumsum() / contacts.sum()])
    fpr = np.concatenate([[0], not_contacts.cumsum() / not_contacts.sum()])
    return np.array([fpr, tpr])


def compute_auc(roc):
    fpr, tpr = roc
    auc = np.sum(tpr[:-1] * (fpr[1:] - fpr[:-1]))
    return auc


def plot_roc(roc):
    import matplotlib.pyplot as plt
    plt.plot(roc[0], roc[1])
    plt.xlabel('False positive rate (1-specificity)')
    plt.ylabel('True positive rate (sensiticity)')
    plt.suptitle('Receiver operating characteristic')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], '--')


def plot_singleresidue_decoy_energy(decoy_energy, native_energy):
    import seaborn as sns
    g = sns.clustermap(decoy_energy, cmap='RdBu_r',
                       vmin=native_energy - decoy_energy.std() * 3,
                       vmax=native_energy + decoy_energy.std() * 3)
    AA_dict = {str(i): _AA[i] for i in range(len(_AA))}
    new_ticklabels = []
    for t in g.ax_heatmap.get_xticklabels():
        t.set_text(AA_dict[t.get_text()])
        new_ticklabels += [t]
    g.ax_heatmap.set_xticklabels(new_ticklabels)


def write_tcl_script(pdb_file, chain, single_frustration, pair_frustration, tcl_script='frustration.tcl',
                     max_connections=100):
    fo = open(tcl_script, 'w+')
    structure = prody.parsePDB(pdb_file)
    selection = structure.select('protein', chain=chain)
    residues = np.unique(selection.getResindices())

    fo.write(f'[atomselect top all] set beta 0\n')
    # Single residue frustration
    for r, f in zip(residues, single_frustration):
        # print(f)
        fo.write(f'[atomselect top "chain {chain} and residue {r}"] set beta {f}\n')

    # Mutational frustration:
    r1, r2 = np.meshgrid(residues, residues, indexing='ij')
    sel_frustration = np.array([r1.ravel(), r2.ravel(), pair_frustration.ravel()]).T
    minimally_frustrated = sel_frustration[sel_frustration[:, -1] < -0.78]
    s = np.argsort(minimally_frustrated[:, -1])
    minimally_frustrated = minimally_frustrated[s][:max_connections]
    fo.write('draw color green\n')
    for r1, r2, f in minimally_frustrated:
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        fo.write(f'draw line $pos1 $pos2 style solid width 2\n')

    frustrated = sel_frustration[sel_frustration[:, -1] > 1]
    s = np.argsort(frustrated[:, -1])[::-1]
    frustrated = frustrated[s][:max_connections]
    fo.write('draw color red\n')
    for r1, r2, f in frustrated:
        fo.write(f'lassign [[atomselect top "resid {r1} and name CA and chain {chain}"] get {{x y z}}] pos1\n')
        fo.write(f'lassign [[atomselect top "resid {r2} and name CA and chain {chain}"] get {{x y z}}] pos2\n')
        fo.write('draw line $pos1 $pos2 style solid width 2\n')
    fo.write('''mol delrep top 0
            mol color Beta
            mol representation NewCartoon 0.300000 10.000000 4.100000 0
            mol selection all
            mol material Opaque
            mol addrep top
            color scale method GWR
            ''')
    fo.close()
    return tcl_script


def call_vmd(pdb_file, tcl_script):
    import subprocess
    return subprocess.Popen(['vmd', '-e', tcl_script, pdb_file], stdin=subprocess.PIPE)


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
        self.pdb_file = pdb_file
        self.chain = chain
        self.sequence = get_protein_sequence_from_pdb(self.pdb_file, self.chain)

        # Set parameters
        self._potts_model_file = potts_model_file
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = distance_matrix_method

        # Compute fast properties
        self.distance_matrix = get_distance_matrix_from_pdb(self.pdb_file, self.chain, self.distance_matrix_method)
        self.potts_model = load_potts_model(self.potts_model_file)
        self.aa_freq = compute_aa_freq(self.sequence)
        self.contact_freq = compute_contact_freq(self.sequence)
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)

        # Initialize slow properties
        self._native_energy = None
        self._decoy_fluctuation = {}

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
        self.distance_matrix = get_distance_matrix_from_pdb(self.pdb_file, self.chain, value)
        self.mask = compute_mask(self.distance_matrix, self.distance_cutoff, self.sequence_cutoff)
        self._distance_matrix_method = value
        self._native_energy = None
        self._decoy_fluctuation = {}

    @property
    def potts_model_file(self):
        return self._potts_model_file

    @potts_model_file.setter
    def potts_model_file(self, value):
        self.potts_model = load_potts_model(value)
        self._potts_model_file = value
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
        elif kind in ['mutational', 'configurational','contact']:
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


# Function if script invoked on its own
def main(pdb_name, chain_name, atom_type, DCA_Algorithm, build_msa_files, database_name,
         gap_threshold, dca_frustratometer_directory):
    # PDB DCA frustration analysis directory
    protein_dca_frustration_calculation_directory = f"{os.getcwd()}/{datetime.today().strftime('%m_%d_%Y')}_{pdb_name}_{chain_name}_DCA_Frustration_Analysis"
    if not os.path.exists(protein_dca_frustration_calculation_directory):
        os.mkdir(protein_dca_frustration_calculation_directory)
    ###
    # Importing PDB structure
    if not os.path.exists(f"{protein_dca_frustration_calculation_directory}/{pdb_name[:4]}.pdb"):
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_name[:4]}.pdb",
                                   f"{protein_dca_frustration_calculation_directory}/{pdb_name[:4]}.pdb")

    pdb_sequence = get_protein_sequence_from_pdb(f"{protein_dca_frustration_calculation_directory}/{pdb_name[:4]}.pdb",
                                                 chain_name)

    # Identify PDB's protein family
    pdb_pfam_mapping_dataframe = pd.read_csv(f"{dca_frustratometer_directory}/pdb_chain_pfam.csv", header=1, sep=",")
    protein_family = pdb_pfam_mapping_dataframe.loc[((pdb_pfam_mapping_dataframe["PDB"] == pdb_name.lower())
                                                     & (pdb_pfam_mapping_dataframe[
                                                            "CHAIN"] == chain_name)), "PFAM_ID"].values[0]

    # Save PDB sequence
    with open(
            f"{protein_dca_frustration_calculation_directory}/{protein_family}_{pdb_name}_{chain_name}_sequences.fasta",
            "w") as f:
        f.write(">{}_{}\n{}\n".format(pdb_name, chain_name, pdb_sequence))

    # Generate PDB contact distance matrix
    distance_matrix = get_distance_matrix_from_pdb(
        f"{protein_dca_frustration_calculation_directory}/{pdb_name[:4]}.pdb",
        chain_name)
    # Generate PDB alignment files
    alignment_dca_files_directory = f"{protein_dca_frustration_calculation_directory}/{datetime.today().strftime('%m_%d_%Y')}_{protein_family}_PFAM_Alignment_DCA_Files"
    file_name = protein_family
    if build_msa_files:
        alignment_dca_files_directory = f"{protein_dca_frustration_calculation_directory}/{datetime.today().strftime('%m_%d_%Y')}_{protein_family}_{pdb_name}_{chain_name}_Jackhmmer_Alignment_DCA_Files"
        file_name = f"{protein_family}_{pdb_name}_{chain_name}"

    if not os.path.exists(alignment_dca_files_directory):
        os.mkdir(alignment_dca_files_directory)

    subprocess.call(["cp",
                     f"{protein_dca_frustration_calculation_directory}/{protein_family}_{pdb_name}_{chain_name}_sequences.fasta",
                     alignment_dca_files_directory])

    generate_protein_sequence_alignments(protein_family, pdb_name, build_msa_files,
                                         database_name, alignment_dca_files_directory,
                                         dca_frustratometer_directory)
    ###
    generate_potts_model(file_name, gap_threshold, DCA_Algorithm,
                         alignment_dca_files_directory, dca_frustratometer_directory)
    os.chdir(protein_dca_frustration_calculation_directory)

    # Compute PDB sequence native DCA energy
    potts_model = load_potts_model(
        f"{alignment_dca_files_directory}/{file_name}_msa_dca_gap_threshold_{gap_threshold}.mat")
    e = compute_native_energy(pdb_sequence, potts_model, distance_matrix,
                              distance_cutoff=4, sequence_distance_cutoff=0)
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
