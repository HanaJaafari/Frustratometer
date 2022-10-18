"""Provide the primary functions."""
import typing

from Bio.PDB import PDBParser
from Bio import AlignIO
import prody
import scipy.spatial.distance as sdist
import pandas as pd
import numpy as np
import itertools
import os
import urllib.request
import scipy.io
import subprocess
from pathlib import Path
# import pydca.plmdca
import logging
import gzip
import tempfile

_path = Path(__file__).parent.absolute()
_AA = '-ACDEFGHIKLMNPQRSTVWY'


##################
# PFAM functions #
##################

def create_directory(path):
    """
    Creates a directory after checking it does not exists

    Parameters
    ----------
    path :  str or Path
        Location of the new directory
    
    Returns
    -------
    path: Path
        Path of the new directory

    """
    path=Path(path)
    if not path.exists() and not path.is_symlink():
        logging.debug(f"Creating {path}")
        Path.mkdir(path)
    return path


def create_pfam_database(name='PFAM_current',
                         url="https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.full.uniprot.gz", ):
    """
    Downloads and creates a pfam database in the Database folder

    Parameters
    ----------
    url :  str
        Adress of pfam database
    name: str
        Name of the new database folder

    Returns
    -------
    alignment_path: Path
        Path of the alignments
    """

    # Create database directory
    databases_path = create_directory(_path / 'databases')
    data_path = create_directory(databases_path / name)
    alignments_path = create_directory(data_path / 'Alignments')

    # Get file name
    file_name = url.split('/')[-1]

    # Download pfam alignments
    logging.debug(f"Downloading {url} to {data_path}/{file_name}")
    urllib.request.urlretrieve(f"{url}", f"{data_path}/{file_name}")

    # Split PFAM alignments
    with gzip.open(f'{data_path}/{file_name}') as in_file:
        new_lines = ''
        acc = 'Unknown'
        for line in in_file:
            line = line.decode('utf-8')
            if line.strip() == '//':
                if len(new_lines) > 0:
                    with open(f'{alignments_path}/{acc}.sto', 'w+') as out_file:
                        logging.debug(f'{alignments_path}/{acc}.sto')
                        out_file.write(new_lines)
                new_lines = ''
                acc = 'Unknown'
                continue
            new_lines += line
            l = line.strip().split()
            if len(l) == 3 and l[0] == "#=GF" and l[1] == "AC":
                acc = l[2]
        if len(new_lines) > 0:
            with open(f'{alignments_path}/{acc}.sto', 'w+') as out_file:
                logging.debug(f'{alignments_path}/{acc}.sto')
                out_file.write(new_lines)
    return alignments_path


def get_pfamID(pdbID, chain):
    """
    Returns PFAM and Uniprot IDs

    Parameters
    ----------
    pdbID :  str
        pdbID (4 characters)
    chain : str
        Select chain from pdb
    Returns
    -------
    pfamID : str
        PFAM family ID
    """

    # TODO fix function
    df = pd.read_csv(f'{_path}/data/pdb_chain_pfam.csv', header=1)
    if sum((df['PDB'] == pdbID.lower()) & (df['CHAIN'] == chain.upper())) != 0:
        #Assumes one domain for the PDB
        pfamID = df.loc[(df['PDB'] == pdbID.lower()) & (df['CHAIN'] == chain.upper())]["PFAM_ID"].values[0]
    else:
        print('PFAM ID is unavailable')
        pfamID = 'null'
    return pfamID


def get_pfam_map(pdbID, chain):
    # TODO fix function
    import pandas as pd
    df = pd.read_table('%s/pdb_pfam_map.txt' % basedir, header=0)
    if sum((df['PDB_ID'] == pdbID.upper()) & (df['CHAIN_ID'] == chain.upper())) != 0:
        start = df.loc[(df['PDB_ID'] == pdbID.upper()) & (df['CHAIN_ID'] == chain.upper())]["PdbResNumStart"].values[0]
        end = df.loc[(df['PDB_ID'] == pdbID.upper()) & (df['CHAIN_ID'] == chain.upper())]["PdbResNumEnd"].values[0]
    else:
        print('data not found')
        pfamID = 'null'
    return int(start), int(end)


def download_alignment_from_interpro(pfamID,
                                     output_file=None,
                                     alignment_type='uniprot'):
    """'
    Retrieves a pfam family alignment from interpro

    Parameters
    ----------
    pfamID : str,
        ID of PFAM family. ex: PF00001
    alignment_type: str,
        alignment type to retrieve. Options: full, seed, uniprot
    output_file: str
        location of the output file. Default: Temporary file

    Returns
    -------
    output : Path
        location of alignment
    
    """
    import tempfile
    from urllib.request import urlopen
    import gzip

    url = f'https://www.ebi.ac.uk/interpro/api/entry/pfam/{pfamID}/?annotation=alignment:{alignment_type}'
    logging.debug(f'Downloading {url} to {output_file}')

    if output_file is None:
        output = tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_interpro.sto')
        output_file = Path(output.name)
    else:
        output_file = Path(output_file)

    output = urlopen(url).read()
    alignment = gzip.decompress(output)
    output_file.write_bytes(alignment)
    return output_file

def filter_alignment(alignment_file, 
                     output_file = None,
                     alignment_format = "stockholm"):
    """
    Filter PDB alignment
    :param alignment_file
    :return: 
    """
    '''Returns PDB MSA (fasta format) with column-spanning gaps and insertions removed'''
    import tempfile

    # Parse the alignment
    alignment = AlignIO.read(alignment_file, alignment_format)

    # Create a numpy array of the alignment
    alignment_array=[]
    for record in alignment:
        alignment_array+=[np.array(record.seq)]
    alignment_array=np.array(alignment_array)

    # Substitute lower case letters with gaps
    for letter in np.unique(alignment_array):
        if letter!=letter.upper():
            alignment_array[alignment_array==letter]='-'

    # Take only columns that are not all gaps
    not_all_gaps=(alignment_array=='-').sum(axis=0)<len(alignment_array)
    new_alignment=alignment_array[:,not_all_gaps]

    # Create temporary file if needed
    if output_file is None:
        # Temporary file
        output = tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_alignment.fa', delete=False)
        output_file = Path(output.name)
    else:
        # Other file
        output_file = Path(output_file)

    # Write filtered alignment to file
    for record,new_seq in zip(alignment,new_alignment):
        output_file.write_text(f">{record.id}\n{''.join(new_seq)}\n")
        
    return output_file

def create_pottsmodel_from_alignment_pydca(fasta_sequence,
                                           sequence_type='protein',
                                           seqid=0.8, 
                                           lambda_h=1.0,
                                           lambda_J=20.0,
                                           num_threads=10,
                                           max_iterations=500):
    plmdca_inst = pydca.plmdca.PlmDCA(fasta_sequence,
                                      sequence_type,
                                      seqid,
                                      lambda_h,
                                      lambda_J,
                                      num_threads,
                                      max_iterations)
    return plmdca_inst.get_potts_model()

def download_pdb(pdbID):
    """
    Downloads a single pdb file
    """
    import urllib.request
    urllib.request.urlretrieve('http://www.rcsb.org/pdb/files/%s.pdb' % pdbID, "%s%s.pdb" % (directory, pdbID))


def stockholm2fasta(pfamID):
    """
    Converts stockholm alignment to fasta    
    """
    from Bio import AlignIO
    # rewrite Stockholm alignment in FASTA format
    input_handle = open("%s%s.stockholm" % (directory, pfamID), "rU")
    output_handle = open("%s%s.fasta" % (directory, pfamID), "w")
    alignments = AlignIO.parse(input_handle, "stockholm")
    AlignIO.write(alignments, output_handle, "fasta")
    output_handle.close()
    input_handle.close()


def filter_fasta(gap_threshold, pfamID, pdbID, chain, seq, resnos):
    """
    Filters and maps sequence to fasta alignment
    """

    from Bio import AlignIO
    import numpy
    import subprocess
    # gap_threshold=0.25
    pfam_start, pfam_end = get_pfam_map(pdbID, chain)
    mapped_seq = seq[resnos.index(pfam_start):resnos.index(pfam_end) + 1]

    # print mapped fasta file
    f = open('%s%s%s_pfam_mapped.fasta' % (directory, pdbID, chain), 'w')
    f.write('>%s:%s pdb mapped to pfam\n' % (pdbID, chain))
    f.write(mapped_seq)
    f.close()

    submit = ("%s/muscle3.8.31_i86linux64 -profile -in1 %s%s.fasta -in2 %s%s%s_pfam_mapped.fasta -out %s%s%s.fasta" % (
        basedir, directory, pfamID, directory, pdbID, chain, directory, pdbID, chain))

    # print(submit)

    process = subprocess.Popen(submit.split(), stdout=subprocess.PIPE)
    process.communicate()

    # Filter sequences based on gaps in input sequence and gap threshold
    alignment = AlignIO.read(open("%s%s%s.fasta" % (directory, pdbID, chain)), "fasta")
    targetseq = alignment[-1].seq
    targetname = alignment[-1].name
    if targetseq == '':
        print("targetseq not found")

    output_handle = open("%s%s_msa_filtered.fasta" % (directory, pdbID), "w")
    target_array = numpy.array([list(targetseq)], numpy.character)
    bools = target_array != b'-'
    sequences_passed_threshold = 0
    for i, record in enumerate(alignment):
        record_array = numpy.array([list(record.seq)], numpy.character)
        aligned_sequence = record_array[bools]
        if float(numpy.sum(aligned_sequence == b'-')) / len(aligned_sequence) < gap_threshold:
            output_handle.write(">%s\n" % record.id + "".join(aligned_sequence.astype(str)).upper() + '\n')
            sequences_passed_threshold += 1
    output_handle.close()

    fastaseq = ''.join(target_array[bools].astype(str)).upper()

    stat_output = open(stat_output_file_name, "w")
    stat_output.write("FASTA_alignments " + str(len(alignment)) + "\n")
    stat_output.write("Filtered_alignments " + str(sequences_passed_threshold) + "\n")
    stat_output.close()

    return fastaseq, sequences_passed_threshold

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

def generate_filtered_alignment(alignment_source,pfamID,sequence,pdb_name,download_all_alignment_files_status,alignment_files_directory):
    """
    Generates gap-filtered alignment based on user input.
    Options are generating an alignment using Jackhmmer (select "alignment_source='jackhmmer'")
    or downloading generated PFAM (select "alignment_source='full'") 
    or Uniprot alignments (select "alignment_source='uniprot'").

    Parameters
    ----------
    alignment_source :  str
        Protein alignment source

    Returns
    -------
    alignment_file_name: str
        Alignment file name
    """
    if alignment_source=="full" or alignment_source=="uniprot":
        alignment_file=download_alignment_PFAM_or_uniprot(pfamID,alignment_source,download_all_alignment_files_status,alignment_files_directory)
    elif alignment_source=="jackhmmer":
        alignment_file=create_alignment_jackhmmer(sequence,pdb_name,
                                                  download_all_alignment_files_status,
                                                  alignment_files_directory)
    else:
        print("Incorrect alignment type input")
    
    if not alignment_file==None:
        filtered_alignment_file=convert_and_filter_alignment(alignment_file,
                                                            download_all_alignment_files_status,
                                                            alignment_files_directory)
    else:
        filtered_alignment_file=None
    return filtered_alignment_file


def download_alignment_PFAM_or_uniprot(pfamID,alignment_source,download_all_alignment_files_status,alignment_files_directory):
    """
    Downloads family PFAM or uniprot alignment from Interpro database

    Parameters
    ----------
    pfamID :  str
        Protein Family PFAM ID

    Returns
    -------
    alignment_path: Path
        Path of the alignments
    """
    url = f'https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfamID}/?annotation=alignment:{alignment_source}&download'

    if download_all_alignment_files_status is None:
        output = tempfile.NamedTemporaryFile(mode="w", prefix=f"{pfamID}_", suffix=f'_{alignment_source}.sto',dir=alignment_files_directory,delete=False)
        output_file = Path(output.name)
    else:
        output_file = Path(f"{alignment_files_directory}/{pfamID}_{alignment_source}.sto")
    
    zipped_alignment = urllib.request.urlopen(url).read()
    unzipped_alignment = gzip.decompress(zipped_alignment)
    output_file.write_bytes(unzipped_alignment)
    return output_file


def create_alignment_jackhmmer_deprecated(sequence, pdb_name,
                               database=f'{_path}/Uniprot_Sequence_Database_Files/uniprot_sprot.fasta',
                               output_file=None,
                               log_file=None):
    """

    :param sequence:
    :param PDB Name:
    :param database:
    :param output_file:
    :param log_file:
    :return:
    """
    # TODO: pass options for jackhmmer
    # jackhmmer and databases required
    import subprocess
    import tempfile

    if output_file is None:
        output = tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_alignment.sto')
        output_file = output.name

    with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_sequence.fa') as fasta_file:
        fasta_file.write(f'>{pdb_name}\n{sequence}\n')
        fasta_file.flush()
        if log_file is None:
            subprocess.call(
                ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
                 '--incE', '1E-10', fasta_file.name, database], stdout=subprocess.DEVNULL)
        else:
            with open(log_file, 'w') as log:
                subprocess.call(
                    ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
                     '--incE', '1E-10', fasta_file.name, database], stdout=log)

def create_alignment_jackhmmer(sequence, pdb_name,download_all_alignment_files_status,
                               alignment_files_directory,
                               database=f'{_path}/Uniprot_Sequence_Database_Files/uniprot_sprot.fasta'):
    """

    :param sequence:
    :param PDB Name:
    :param database:
    :param output_file:
    :param log_file:
    :return:
    """
    # TODO: pass options for jackhmmer
    # jackhmmer and databases required
    
    if alignment_files_directory==None:
        jackhmmer_alignments_directory=os.getcwd()
    else:
        jackhmmer_alignments_directory=alignment_files_directory
        
    import tempfile

    if download_all_alignment_files_status is False:
        output = tempfile.NamedTemporaryFile(mode="w", prefix=pdb_name, suffix='_alignment.sto')
        output_file = output.name
    else:
        output_file=open(f"","w")

    with tempfile.NamedTemporaryFile(mode="w", prefix=pdb_name, suffix='_sequence.fa') as fasta_file:
        fasta_file.write(f'>{pdb_name}\n{sequence}\n')
        fasta_file.flush()
        if log_file is None:
            subprocess.call(
                ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
                 '--incE', '1E-10', fasta_file.name, database], stdout=subprocess.DEVNULL)
        else:
            with open(log_file, 'w') as log:
                subprocess.call(
                    ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
                     '--incE', '1E-10', fasta_file.name, database], stdout=log)
    return output.name

def convert_and_filter_alignment(alignment_file,download_all_alignment_files_status,alignment_files_directory):
    """
    Filter PDB alignment
    :param alignment_file
    :return: 
    """
    '''Returns PDB MSA (fasta format) with column-spanning gaps and insertions removed'''
    alignment_file_name=alignment_file.name.replace(".sto","")
    # Convert full MSA in stockholm format to fasta format
    output_handle = open(f"{alignment_file_name}.fasta", "w")
    alignments = AlignIO.parse(alignment_file, "stockholm")
    AlignIO.write(alignments, output_handle, "fasta")
    output_handle.close()

    # Remove inserts and columns that are completely composed of gaps from MSA
    alignment = AlignIO.read(f"{alignment_file_name}.fasta", "fasta")
    output_handle = open(f"{alignment_file_name}_gaps_filtered.fasta", "w")

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

    output_file_name=f"{alignment_file_name}_gaps_filtered.fasta"
    return output_file_name


def compute_plm(protein_name, reweighting_threshold=0.1, nr_of_cores=1):
    """
    Calculate Potts Model Fields and Couplings Terms
    :param protein_name
    :param reweighting_threshold
    :param nr_of_cores
    """
    '''Returns matrix consisting of Potts Model Fields and Couplings terms'''
    # MATLAB needs Bioinfomatics toolbox and Image processing toolbox to parse the sequences
    # Functions need to be compiled with 'matlab -nodisplay -r "mexAll"'
    # See: https://www.mathworks.com/help/matlab/call-mex-functions.html
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
        eng.addpath('%s/plmDCA_asymmetric_v2_with_h' % _path, nargout=0)
        eng.addpath('%s/plmDCA_asymmetric_v2_with_h/functions' % _path, nargout=0)
        eng.addpath('%s/plmDCA_asymmetric_v2_with_h/3rd_party_code/minFunc' % _path, nargout=0)
        print('plmDCA_asymmetric', protein_name, reweighting_threshold, nr_of_cores)
        eng.plmDCA_asymmetric(protein_name, reweighting_threshold, nr_of_cores, nargout=0)  # , stdout=out )
    except ImportError:
        subprocess.call(['matlab', '-nodisplay', '-r',
                         f"plmDCA_asymmetric({protein_name},{reweighting_threshold},{nr_of_cores},nargout=0);quit"])



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
    r"""
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
    r"""
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
    r"""
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
    r"""
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
                      alignment_source: str,
                      pdb_file: str,
                      chain: str,
                      download_all_alignment_files_status=True,
                      alignment_files_directory=None,
                      sequence_cutoff: typing.Union[float, None] = None,
                      distance_cutoff: typing.Union[float, None] = None,
                      distance_matrix_method='minimum'):
        self = cls()

        # Set initialization variables
        self._potts_model_file = None
        self._alignment_source=alignment_source
        self._pdb_file = Path(pdb_file)
        self._pdb_name=os.path.basedir(pdb_file)[:4]
        self._chain = chain
        self._download_all_alignment_files_status = download_all_alignment_files_status
        self._alignment_files_directory=alignment_files_directory
        self._sequence_cutoff = sequence_cutoff
        self._distance_cutoff = distance_cutoff
        self._distance_matrix_method = distance_matrix_method

        # Compute fast properties
        self._pfamID=get_pfamID(self.pdb_name,self.chain)
        self._filtered_alignment_file=generate_filtered_alignment(self.alignment_source,self.pfamID,self.sequence,self.pdb_name,self.download_all_alignment_files_status,self.alignment_files_directory)
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

    @sequence.setter
    def sequence(self, value):
        assert len(value) == len(self._sequence)
        self._sequence = value

    @property
    def pdb_file(self):
        return str(self._pdb_file)

    @pdb_file.setter
    def pdb_file(self, value):
        self._pdb_file = Path(value)

    @property
    def pdb_name(self, value):
        """
        Returns PDBid from pdb name
        """
        return self._pdb_file.stem

    @property
    def chain(self):
        return self._chain

    @chain.setter
    def chain(self, value):
        self._chain = value

    @property
    def pfamID(self, value):
        """
        Returns pfamID from pdb name
        """
        return self._pfamID

    @property
    def alignment_source(self, value):
        return self._alignment_source

    @property
    def download_all_alignment_files_status(self, value):
        return self._download_all_alignment_files_status

    @property
    def alignment_files_directory(self, value):
        return self._alignment_files_directory

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
