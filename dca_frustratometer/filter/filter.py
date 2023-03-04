from Bio import AlignIO
import numpy as np
from pathlib import Path

def filter_alignment(alignment_file, 
                     output_file = None,
                     alignment_format = "stockholm"):
    """
    Filter stockholm alignment by removing unaligned sections.
    Filters by saving the alignment into memory.

    Parameters
    ----------
    alignment_file : Path
        location of file
    output_file : Path
        location of the output_file
    alignment_format : str
        Biopython alignment format (Default: stockholm)
    Returns
    -------
    output_file : Path
        location of the output_file
    """
    filter_query=False
    
    # Parse the alignment
    alignment = AlignIO.read(alignment_file, alignment_format)

    # Create a numpy array of the alignment
    alignment_array=[]
    for record in alignment:
        alignment_array+=[np.array(record.seq)]
        if record.name=='Query':
            query_seq=np.array(record.seq)
            filter_query=True
    alignment_array=np.array(alignment_array)

    # If there is a query sequence only take sequences in query
    if filter_query:
        alignment_array=alignment_array[:,query_seq!='-']

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


def filter_alignment_lowmem(alignment_file, 
                               output_file, 
                               alignment_format = "stockholm"):
    """
    Filter stockholm alignment by removing unaligned sections.
    Filters by reading the file without saving to memory.

    Parameters
    ----------
    alignment_file : Path
        location of file
    output_file : Path
        location of the output_file
    alignment_format : str
        Biopython alignment format (Default: stockholm)
    Returns
    -------
    output_file : Path
        location of the output_file
    """
    #TODO: Implement filter query for jackhmmer

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