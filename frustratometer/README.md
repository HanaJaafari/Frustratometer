# Developer notes

This in an overview of the organization of the following folders and notes for future development efforts.

## Overall Design Approach

The software is designed with two programming paradigms in mind:

- **Object-Oriented**: By using classes, we try to encapsulate the data and its expected behavior, providing a structured way to represent Structures, which can be converted to AWSEM or DCA models and then be used in subsequent analysis.
  
- **Functional**: Functions are used for specific tasks, offering flexibility and modularity. These tasks include filtering alignments, computing energies, and generating visualizations, among others.

We hope that this combined approach will allow a clean way to manage the data flow using the objects, while offering the flexibility of diverting the flow using the functions.

## Folder Structure and Functionalities

The project is structured into several directories, each dedicated to specific types of operations:

### `classes`
- **Purpose**: Defines core classes for the project.
- **Key Classes**:
  - `DCA`: Represents a protein with DCA information, including methods to calculate energies, frustrations, decoy statistics, and visualizations.
  - `AWSEM`: Manages AWSEM energy calculations with methods for computing energies, frustrations, decoy statistics, and visualizations
  - `Structure`: Manages protein structures, handling PDB files, chains, sequences, and distance matrices.
  - `Map`: Maps between different sequence representations (for example to translate models of different sizes using alignments)
  - `Frustratometer`: Base class for Potts-model like models like AWSEM and DCA that provides frustration functions.

### `frustration`
- **Purpose**: Contains functions related to the calculations of frustration in protein structures.
- **Energy Funtions**:
  - `compute_mask`: Creates a mask to define which residue pairs to include in frustration analysis based on distance and sequence separation.
  - `compute_native_energy`: Calculates the native energy of a protein sequence using its Potts model and mask, with options to ignore gap residues.
  - `compute_sequences_energy`: Efficiently computes energies for multiple protein sequences.
- **Decoy Energy Fluctuation Functions**:
  - `compute_singleresidue_decoy_energy_fluctuation`: Calculates energy changes when single residues are mutated.
  - `compute_mutational_decoy_energy_fluctuation`: Calculates energy changes for pairwise mutations within contact distance.
  - `compute_configurational_decoy_energy_fluctuation`: Similar to mutational, but approximates effects on non-contacting residues.
  - `compute_contact_decoy_energy_fluctuation`: Focuses on changes in pairwise interaction energies upon mutations.
- **Frustration Calculation Functions**:
  - `compute_single_frustration`: Quantifies single-residue frustration based on decoy energy fluctuations.
  - `compute_pair_frustration`: Quantifies pairwise frustration based on decoy energy fluctuations.
- **Other Functions**:
  - `compute_scores`: Computes contact scores using the Frobenius norm of couplings.
  - `compute_roc`: Generates a receiver operating characteristic (ROC) curve for assessing contact prediction.
  - `compute_auc`: Calculates the area under the ROC curve.
- **Visualization functions**:
  - `plot_roc`: Visualizes the ROC curve.
  - `plot_singleresidue_decoy_energy`: Creates a heatmap or clustermap of single-residue decoy energies.
  - `write_tcl_script`: Generates a Tcl script for visualizing frustration in VMD.
  - `call_vmd`: Opens a PDB file in VMD and runs a Tcl script for visualization.

### `dca`
- **Purpose**: Handles calculations related to Direct Coupling Analysis using both PyDCA and MATLAB.
- **Key Files**:
  - `pydca.py`: Functions for PLMDCA and MFDCa calculations using PyDCA.
  - `matlab.py`: Functions for computing and loading Potts models using MATLAB scripts.

### `filter`
- **Purpose**: Provides functions to filter and process multiple sequence alignments.
- **Key Files**:
  - `filter_alignment.py`: Filters a Stockholm alignment by removing unaligned sections and gaps.
  - `filter_alignment_lowmem.py`: A memory-efficient version of alignment filtering.

### `optimization`
- **Purpose**: Implements optimization algorithms for protein sequences.

### `align`
- **Purpose**: Provides functions for generating alignments.
- **Key Files**:
  - `jackhmmer.py`: Runs Jackhmmer to generate alignments from sequences and databases.
  - `extract_sequences_from_alignment.py`: Extracts full sequences from a Stockholm alignment file using a specified database.

### `pfam`
- **Purpose**: Manages operations related to PFAM alignments.
- **Key Files**:
  - `download_database.py`: Downloads and organizes the entire PFAM database locally.
  - `get_alignment.py`: Retrieves specific alignments from the local PFAM database.
  - `download_alignment.py`: Downloads a specific PFAM alignment directly from Interpro.

### `map`
- **Purpose**: Offers functions for mapping PDB information to PFAM IDs.
- **Key Files**:
  - `get_pfamID.py`: Retrieves the PFAM ID for a given PDB file and chain.

### `pdb`
- **Purpose**: Handles operations related to Protein Data Bank (PDB) files.
- **Key Files**:
  - `download.py`: Downloads a PDB file given its ID.
  - `get_sequence.py`: Extracts the protein sequence from a PDB file.
  - `repair_pdb.py`: Repairs a PDB file by inserting missing residues and removing heteroatoms.
  - `get_distance_matrix.py`: Calculates the distance matrix for specified atoms in a PDB file using various methods.

### `utils`
- **Purpose**: Utility functions that don't fit anywhere yet
- **Key Files**:
  - `create_directory.py`: Ensures the creation of a directory if it does not exist.