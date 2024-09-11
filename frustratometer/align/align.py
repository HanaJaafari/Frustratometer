import subprocess
from pathlib import Path
import tempfile

def jackhmmer(sequence,
              output_file,
              database,
              log=subprocess.DEVNULL,
              dry_run: bool = False,
              **kwargs)->Path:
    """
    Generates alignment using jackhmmer

    Parameters
    ----------
    sequence :  str
        Protein sequence
    output_file : str
        If True, will record alignment output messages;
        Arguments: True OR False (Default is False)
    database: str
        Location of the sequence database used to generation MSA
    log: File handle
        jackhmmer output Default:DEVNULL
    dry_run: bool (default: False)
        Save the temporary fasta_file and print the command instead of running
    **kwargs: 
        Other arguments that can be passed to jackhmmer.
        More information can be found by executing `jackhmmer -h`
        arguments without a value such as --noali should be passed as `noali=True`
        Common kwargs:
            N: number of iterations
            E: E-value threshold


    Returns
    -------
    output_file: Path
        Path of the alignment file
    """
    jackhmmer_path='jackhmmer'
    command_kwargs={"A": output_file,
                    "N":5,
                    "E": 1E-8,
                    "noali": True,
                    "incE": 1E-10}

    database=Path(database)
    if not database.exists():
      raise IOError(f'Database not found in path:{str(database)}')
    
    #Update arguments
    command_kwargs.update(kwargs)

    #Create commands
    commands=[jackhmmer_path]
    for key,value in command_kwargs.items():
        #Add two dashes if key in command is larger than a single letter
        if len(key)==1:
            key='-'+key
        else:
            key='--'+key

        #Only add the term if the value is True, else add the value
        if value is True:
             commands+=[key]
        elif value is False:
            pass
        else:
            commands+=[key,str(value)]
       
    #Creates a temporary file with the Query sequence to run jackhmmer
    if dry_run:
        with open(output_file+'_sequence.fasta','w+') as fasta_file:
            fasta_file.write(f'>Query\n{sequence}\n')
            fasta_file.flush()
            commands+=[fasta_file.name, database]
            print(' '.join([str(a) for a in commands]))
        return

    with tempfile.NamedTemporaryFile(mode="w", prefix=f"dcaf_", suffix='_sequence.fasta') as fasta_file:
        fasta_file.write(f'>Query\n{sequence}\n')
        fasta_file.flush()
        commands+=[fasta_file.name, database]
        print(commands)
        subprocess.call(commands, stdout=log)
    return Path(output_file)

def extract_sequences_from_alignment(alignment_file,
                                     output_file,
                                     database):
    """
    Extracts the complete sequences of an stockholm alignment from a database to refine the alignment if needed
    """
    
    import Bio.AlignIO
    import Bio.SeqIO

    alignment = Bio.AlignIO.read(alignment_file,'stockholm')
    database = Bio.SeqIO.parse('/home/cb/Development/DCA_Frustratometer/dca_frustratometer/databases/Uniprot/uniprot_sprot.fasta','fasta')
    alignment_records=[]
    for record in alignment:
        alignment_records+=[record.name]

    with open(output_file,'w+') as output_handle:
        for record in database:
            if record.name in alignment_records:
                output_handle.write(record.format('fasta'))
    
    return output_file
    
    
    


#output=jackhmmer('GSWTEHKSPDGRTYYYNTETKQSTWEKPDD','jackhmmer_test.sto','/home/cb/Development/DCA_Frustratometer/dca_frustratometer/databases/Uniprot/uniprot_sprot.fasta')
#extract_sequences_from_alignment(output,'selected_sequences.fa','/home/cb/Development/DCA_Frustratometer/dca_frustratometer/databases/Uniprot/uniprot_sprot.fasta')
#output2=jackhmmer('GSWTEHKSPDGRTYYYNTETKQSTWEKPDD','jackhmmer_test2.sto','/home/cb/Development/DCA_Frustratometer/tests/data/selected_sequences.fa')



# def generate_hmmer_alignment(protein_sequence,alignment_files_directory=os.getcwd(),
#     alignment_output_file=False,alignment_sequence_database="swissprot",iterations=5):
#     """
#     Generates alignment using jackhmmer

#     Parameters
#     ----------
#     pdb_file :  str
#         PDB file path
#     protein_sequence :  str
#         Protein sequence
#     alignment_files_directory:  str
#         If selected TRUE for download_all_alignment_files_status, 
#         provide filepath. Default is current directory. 
#     alignment_output_file:   bool
#         If True, will record alignment output messages;
#         Arguments: True OR False (Default is False)
#     alignment_sequence_database: str
#         Sequence database used to generation MSA
#         Arguments: "swissprot" OR "trembl" (Default is "swissprot")
#     iterations: int
#         Number of iterations for jackhmmer alignment

#     Returns
#     -------
#     output_file: Path
#         Path of the alignment file
#     """
#     output_file = (f"{alignment_files_directory}/{pdb_name}_jackhmmer_MSA.sto")
        
#     jackhmmer_sequence_database=(f"/media/cb/SATA8TB/Databases/Uniprot/uniparc_active.fasta")
#     with tempfile.NamedTemporaryFile(mode="w", prefix=f"{pdb_name}_", suffix='_sequence.fasta',dir=alignment_files_directory) as fasta_file:
#         fasta_file.write(f'>{pdb_name}\n{protein_sequence}\n')
#         fasta_file.flush()
#         if alignment_output_file is False:
#             subprocess.call(
#                 ['jackhmmer', '-A', output_file, '-N', str(iterations),'--noali', '-E', '1E-8',
#                 '--incE', '1E-10', fasta_file.name, jackhmmer_sequence_database], stdout=subprocess.DEVNULL)
#         else:
#             with open(Path(f"{alignment_files_directory}/jackhmmer_output.txt"), 'w') as log:
#                 subprocess.call(
#                     ['jackhmmer', '-A', output_file, '-N', str(iterations), '--noali', '-E', '1E-8',
#                     '--incE', '1E-10', fasta_file.name, jackhmmer_sequence_database], stdout=log)

#     return Path(output_file)

# def create_alignment_jackhmmer_deprecated(sequence, pdb_name,
#                                database=f'{_path}/../Uniprot_Sequence_Database/uniprot_sprot.fasta',
#                                output_file=None,
#                                log_file=None):
#     """

#     :param sequence:
#     :param PDB Name:
#     :param database:
#     :param output_file:
#     :param log_file:
#     :return:
#     """
#     # TODO: pass options for jackhmmer
#     # jackhmmer and databases required
#     import subprocess
#     import tempfile

#     if output_file is None:
#         output = tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_alignment.sto')
#         output_file = output.name

#     with tempfile.NamedTemporaryFile(mode="w", prefix="dcaf_", suffix='_sequence.fa') as fasta_file:
#         fasta_file.write(f'>{pdb_name}\n{sequence}\n')
#         fasta_file.flush()
#         if log_file is None:
#             subprocess.call(
#                 ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
#                  '--incE', '1E-10', fasta_file.name, database], stdout=subprocess.DEVNULL)
#         else:
#             with open(log_file, 'w') as log:
#                 subprocess.call(
#                     ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
#                      '--incE', '1E-10', fasta_file.name, database], stdout=log)

# def create_alignment_jackhmmer(sequence, pdb_name,download_all_alignment_files_status,
#                                alignment_files_directory,
#                                database=f'{_path}/../Uniprot_Sequence_Database/uniprot_swissprot.fasta'):
#     """

#     :param sequence:
#     :param PDB Name:
#     :param database:
#     :param output_file:
#     :param log_file:
#     :return:
#     """
#     # TODO: pass options for jackhmmer
#     # jackhmmer and databases required
    
#     if alignment_files_directory==None:
#         jackhmmer_alignments_directory=os.getcwd()
#     else:
#         jackhmmer_alignments_directory=alignment_files_directory
        
#     import tempfile

#     if download_all_alignment_files_status is False:
#         output = tempfile.NamedTemporaryFile(mode="w", prefix=pdb_name, suffix='_alignment.sto')
#         output_file = output.name
#     else:
#         output_file=open(f"","w")

#     with tempfile.NamedTemporaryFile(mode="w", prefix=pdb_name, suffix='_sequence.fa') as fasta_file:
#         fasta_file.write(f'>{pdb_name}\n{sequence}\n')
#         fasta_file.flush()
#         if log_file is None:
#             subprocess.call(
#                 ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
#                  '--incE', '1E-10', fasta_file.name, database], stdout=subprocess.DEVNULL)
#         else:
#             with open(log_file, 'w') as log:
#                 subprocess.call(
#                     ['jackhmmer', '-A', output_file, '--noali', '-E', '1E-8',
#                      '--incE', '1E-10', fasta_file.name, database], stdout=log)
#     return output.name