import os
import urllib.request
import subprocess
from pathlib import Path
import gzip
import tempfile
from ..utils import _path

def generate_hmmer_alignment(pdb_file,protein_sequence,alignment_files_directory=os.getcwd(),
    alignment_output_file=False,alignment_sequence_database="swissprot"):
    """
    Generates alignment using jackhmmer

    Parameters
    ----------
    pdb_file :  str
        PDB file path
    protein_sequence :  str
        Protein sequence
    alignment_files_directory:  str
        If selected TRUE for download_all_alignment_files_status, 
        provide filepath. Default is current directory. 
    alignment_output_file:   bool
        If True, will record alignment output messages;
        Arguments: True OR False (Default is False)
    alignment_sequence_database: str
        Sequence database used to generation MSA
        Arguments: "swissprot" OR "trembl" (Default is "swissprot")

    Returns
    -------
    output_file: Path
        Path of the alignment file
    """
    if pdb_file:
        pdb_name=os.path.basename(pdb_file)[:4]
    else:
        pdb_name="protein"
    output_file = Path(f"{alignment_files_directory}/{pdb_name}_jackhmmer_MSA.sto")
        
    jackhmmer_sequence_database=Path(f"{_path}/databases/uniprot_{jackhmmer_sequence_database}.fa")
    with tempfile.NamedTemporaryFile(mode="w", prefix=f"{pdb_name}_", suffix='_sequence.fa',dir=alignment_files_directory) as fasta_file:
        fasta_file.write(f'>{pdb_name}\n{protein_sequence}\n')
        fasta_file.flush()
        if alignment_output_file is False:
            subprocess.call(
                ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
                '--incE', '1E-10', fasta_file.name, alignment_sequence_database], stdout=subprocess.DEVNULL)
        else:
            with open(Path(f"{alignment_files_directory}/jackhmmer_output.txt"), 'w') as log:
                subprocess.call(
                    ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
                    '--incE', '1E-10', fasta_file.name, alignment_sequence_database], stdout=log)

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