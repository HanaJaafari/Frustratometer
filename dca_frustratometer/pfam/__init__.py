"""PFAM functions

This module includes functions to download and manipulate PFAM alignments.
There are two options to obtain alignments from PFAM: downloading the database
locally or downloading specific alignments.

If you want to download the database locally use the download function to
download or update the database and the get function to select a single
alignment.

If you want to retrieve a single alignment use the alignemnt function

Examples:
    
        $ dca_frustratometer.pfam.database('/media/databases',
        https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.full.uniprot.gz)

Todo:
    * Create a function to get single alignments

"""

from .pfam import *