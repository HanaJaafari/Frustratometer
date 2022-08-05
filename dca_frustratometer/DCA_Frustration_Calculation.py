#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Author: Hana Jaafari
#Date: July 19, 2022
#Purpose: This script aims to calculate the DCA frustration of a given PDB.

import os
import subprocess
import argparse
import pandas as pd
from datetime import datetime
import urllib.request
import glob
from scipy.io import loadmat
from Bio.PDB import *
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--pdb_name", type=str, required=True,help="PDB Name")
parser.add_argument("--chain_name", type=str, required=True,help="Chain Name")
parser.add_argument("--PFAM_version", action="store_true",default=27, help="PFAM version for DCA Files (Default is PFAM 27)")
parser.add_argument("--build_msa_files",action='store_false',help="Build MSA with Full Coverage of PDB")
parser.add_argument("--database_name", default="Uniprot",help="Database used in seed msa (options are Uniparc or Uniprot)")
parser.add_argument("--gap_threshold",type=float,default=0.2,help="Continguous gap threshold applied to MSA")

args = parser.parse_args()

pdb_name=args.pdb_name
chain_name=args.chain_name
PFAM_version=args.PFAM_version
build_msa_files=args.build_msa_files
database_name=args.database_name
gap_threshold=args.gap_threshold
###
#Directory of DCA frustratometer
dca_frustratometer_directory="/scratch/hkj1/DCA_Frustratometer"

#PDB DCA frustration analysis directory
protein_dca_frustration_directory=f"{os.getcwd()}/{datetime.today().strftime('%m_%d_%Y')}_{pdb_name}_{chain_name}_DCA_Frustration_Analysis"
if not os.path.exists(protein_dca_frustration_directory):
    os.mkdir(protein_dca_frustration_directory)
os.chdir(protein_dca_frustration_directory)


# # Download PDB

# In[ ]:


#Importing PDB structure
if not os.path.exists(f"./{pdb_name[:4]}.pdb"):
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_name[:4]}.pdb", f"./{pdb_name[:4]}.pdb")

#Extract PDB sequence
structure = PDBParser().get_structure(pdb_name[:4], f"./{pdb_name[:4]}.pdb")
ppb=PPBuilder()
for pp in ppb.build_peptides(structure[0][chain_name], aa_only=False):
    pdb_sequence=pp.get_sequence() 
pdb_sequence_length=len(pdb_sequence)
###
#Identify PDB's protein family
pdb_pfam_mapping_dataframe=pd.read_csv(f"{dca_frustratometer_directory}/pdb_chain_pfam.csv",header=1,sep=",")
protein_family=pdb_pfam_mapping_dataframe.loc[((pdb_pfam_mapping_dataframe["PDB"]==pdb_name.lower()) 
                                              & (pdb_pfam_mapping_dataframe["CHAIN"]==chain_name)),"PFAM_ID"].values[0]
###
#Save PDB sequence
with open(f"./{protein_family}_{pdb_name}_{chain_name}_sequences.fasta","w") as f:
    f.write(">{}_{}\n{}\n".format(pdb_name,chain_name,pdb_sequence))


# # Align Protein Sequence and Generate DCA Parameters

# In[ ]:


os.system(f"cp {dca_frustratometer_directory}/Generate_DCA_Frustration_Files.py .")

subprocess.call(["python","./Generate_DCA_Frustration_Files.py","--protein_family",protein_family,"--pdb_name",
                 f"{pdb_name}_{chain_name}","--DCA_frustratometer_directory",dca_frustratometer_directory])
os.chdir(protein_dca_frustration_directory)


# In[ ]:


matlab_parameters_file=glob.glob(f"./*.mat")[0]

plm_dca_parameters = loadmat(matlab_parameters_file)
fields=plm_dca_parameters["h"].T
couplings=plm_dca_parameters["familycouplings"].reshape(int(pdb_sequence_length),21,int(pdb_sequence_length),21).transpose(0,2,1,3)
print("Fields are"); print(fields)
print("Couplings are"); print(couplings)

symmetric_dca=True
column_gap_threshold=None
distance_threshold=16


# In[ ]:


#Calculate Cb-Cb distance map to apply distance threshold on coupling terms.
print("####")
print(f"Generating {pdb_name} Contact Map")
print("####")

os.system(f"cp {dca_frustratometer_directory}/Generate_PDB_Contact_Map.py .")
subprocess.call(["python","./Generate_PDB_Contact_Map.py","--pdb_name",f"{pdb_name}.pdb"])

distance_map=np.loadtxt(f"./{pdb_name}_CB_CB_Distance_Map.txt")


# # Calculate PDB's Native DCA Energy

# In[ ]:


print("####")
print(f"Calculate DCA Energy of {pdb_name}")
print("####")

if not build_msa_files:
    pdb_to_pfam_map=pickle.load(open(f"{pdb_name}_cleaned_to_filtered_aligned_sequence_map.pickle","rb"))
    aligned_file=f"./{protein_family}_{pdb_name}_sequences_aligned_{PFAM_version}_gaps_and_inserts_filtered_gap_threshold_{gap_threshold}.fasta"
else:
    pdb_to_pfam_map=None
    aligned_file=f"./{protein_family}_{pdb_name}_{chain_name}_sequences.fasta"
###
import pseudogeneup as pse
sequence_file_type="native"
native_pdb_DCA_energy, full_info_native_pdb_DCA_energy= pse.pseudofam_evaluate_energies_of_sequences_in_fasta(aligned_file,sequence_file_type,
                                                                                      protein_family,
                                                                                      pdb_to_pfam_map=pdb_to_pfam_map,
                                                                                      distance_matrix=distance_map,
                                                                                      distance_threshold=float(distance_threshold),
                                                                                      fields=fields,couplings=couplings,
                                                                                      subsampling_frequency=1,
                                                                                      symmetric_dca=symmetric_dca,
                                                                                      column_gap_threshold=column_gap_threshold,
                                                                                      gap_threshold=gap_threshold,
                                                                                      PFAM_version=PFAM_version)


# # Calculate Single Residue Mutational Frustration

# In[1]:


plm_dca_letters = ['-', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 
                   'T', 'V', 'W', 'Y']


# In[ ]:


print("####")
print(f"Calculate DCA Energy of {pdb_name} Single Residue Mutants")
print("####")

for residue_index in range(len(pdb_sequence)):
    all_residue_i_mutant_sequences_file=open(f"./{protein_family}_{pdb_name}_{chain_name}_mutated_residue_{residue_index}_sequences.fasta","w")
    
    for mutation_residue in plm_dca_letters:
        if mutation_residue != pdb_sequence[residue_index]:
            mutated_residue_i_sequence=pdb_sequence
            original_residue=pdb_sequence[residue_index]
            mutated_residue_i_sequence[residue_index]=mutation_residue
            
            with open(f"./{protein_family}_{pdb_name}_{chain_name}_mutated_residue_{residue_index}_sequences.fasta","a") as f:
                f.write(">{}_{}_{}_{}_{}\n{}\n".format(pdb_name,chain_name,original_residue_residue_index,mutation_residue,mutated_residue_i_sequence))
                
    sequence_file_type="mutated"
    all_residue_i_mutant_sequences_file_name=f"./{protein_family}_{pdb_name}_{chain_name}_mutated_residue_{residue_index}_sequences.fasta"
    
    mutated_pdb_DCA_energy, full_info_mutated_pdb_DCA_energy= pse.pseudofam_evaluate_energies_of_sequences_in_fasta(all_residue_i_mutant_sequences_file_name,
                                                                                                                    sequence_file_type,protein_family,
                                                                                                                    pdb_to_pfam_map=pdb_to_pfam_map,
                                                                                                                    distance_matrix=distance_map,
                                                                                                                    distance_threshold=float(distance_threshold),
                                                                                                                    fields=fields,couplings=couplings,
                                                                                                                    subsampling_frequency=1,
                                                                                                                    symmetric_dca=symmetric_dca,
                                                                                                                    column_gap_threshold=column_gap_threshold,
                                                                                                                    gap_threshold=gap_threshold,
                                                                                                                    PFAM_version=PFAM_version)

