
from Bio import AlignIO

import numpy as np

import os

from pathlib import Path


def filter_alignment(alignment_file, 
                     output_file = None,
                     alignment_format = "stockholm"):
    """
    Filter PDB alignment
    :param alignment_file
    :return: 
    """
    '''Returns PDB MSA (fasta format) with column-spanning gaps and insertions removed'''
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

    output_file = Path(output_file)

    # Write filtered alignment to file
    text=''
    for record,new_seq in zip(alignment,new_alignment):
        text+=f">{record.id}\n{''.join(new_seq)}\n"
    output_file.write_text(text)
        
    return output_file


#Convert to stockholm
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

def filter_alignment_no_memory(alignment_file, 
                               output_file, 
                               alignment_format = "stockholm"):
    """
    Produces a column gap and insertion filtered MSA file. 

    Parameters
    ----------
    alignment_file :  Path
        MSA file name (full path)
    alignment_files_directory:  str
        If selected TRUE for download_all_alignment_files_status, 
        provide filepath. Default is current directory. 

    Returns
    -------
    filtered_fasta_alignment_file: Path
        Path of the filtered MSA
    """
    
    # Remove inserts and columns that are completely composed of gaps from MSA
    alignment = AlignIO.parse(alignment_file, "stockholm")
    output_handle = open(output_file, "w")

    with open(alignment_file) as alignment_handle:
        alignment = AlignIO.read(alignment_handle, "stockholm")
        index_mask = []
        for i, record in enumerate(alignment):
            index_mask += [i for i, x in enumerate(list(record.seq)) if x != x.upper()]
        for i in range(len(alignment[0].seq)):
            if alignment[:, i] == ''.join(["-"] * len(alignment)):
                index_mask.append(i)
        index_mask = sorted(list(set(index_mask)))
    
    with open(alignment_file) as alignment_handle, open(output_file, "w") as output_handle:
        alignment = AlignIO.read(alignment_handle, "stockholm")
        for i, record in enumerate(alignment):
            aligned_sequence = [list(record.seq)[i] for i in range(len(list(record.seq))) if i not in index_mask]
            output_handle.write(">%s\n" % record.id + "".join(aligned_sequence) + '\n')

    return Path(output_file)