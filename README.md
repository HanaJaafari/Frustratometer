Frustratometer
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/HanaJaafari/Frustratometer/workflows/CI/badge.svg)](https://github.com/HanaJaafari/Frustratometer/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/HanaJaafari/Frustratometer/graph/badge.svg?token=JKDOXOYPRS)](https://codecov.io/gh/HanaJaafari/Frustratometer)


A python implementation of the protein frustratometer.

![https://xkcd.com/173/](https://imgs.xkcd.com/comics/movie_seating.png)

The frustratometer is based on the principle of minimal frustration postulated by Wolynes et al. The main idea is that proteins have evolved under a selective pressure to fold into a native structure, which is energetically minimized, but also to not fold into incorrect conformations, where the energy is maximized. In that principle it is expected that when we measure the energy of a correctly folded protein, most of the interactions between the aminoacids will be minimized, compared to a different aminoacid in the same position or the same aminoacid in a different position. When this energy is minimized we define a particular residue to be minimally frustrated.

Under most circunstances aminoacids are minimally frustrated in the protein, but pockets where aminoacids are of interest for protein function. In short, a frustrated residue or interaction indicates that the particular aminoacid would be minimized under a different configuration, or a different sequence, which reveals possible competing evolutionary pressures on its selection. We have identified that these pockets usually correspond to regions of functional importance, for example, that may form part of a catalitic domain, a hinge domain, or a binding region.

This principle of minimum frustration has been shown using the AWSEM forcefield but can be extended to any other forcefield, including atomistic forcefields, or a pseudo-forcefield like DCA, as implemented here.

In this module we implement a version of the frustratometer based on Direct Coupling analysis.

## Installation

This module is currently under development.
To install this modules please clone this repository and install it using the following commands.

    git clone HanaJaafari/Frustratometer
    cd Frustratometer
    conda install -c conda-forge --file requirements.txt
    pip install -e .

A clean python environment is recommended.

## Usage

### Loading Protein Structures

The awsem frustatometer package includes a prody based Structure class to load the structure and calculate the properties needed for the AWSEM and DCA Frustratometers.

```python
import frustratometer

# Define the path to your PDB file
pdb_path = Path('data/my_protein.pdb')
# Load the structure
structure = frustratometer.Structure.full_pdb(pdb_path)
structure.sequence #The sequence of the structure
```

### Creating an AWSEM Model Instance

After loading the structure, create an AWSEM model instance with the desired parameters. Here we provide some typical configurations that can be found elsewhere.

```python
## Single residue frustration with electrostatics
model_singleresidue = frustratometer.AWSEM(structure, min_sequence_separation_contact=2) 
## Single residue frustration without electrostatics
model_singleresidue_noelectrostatics = frustratometer.AWSEM(structure, min_sequence_separation_contact=2, k_electrostatics=0) 
## Mutational/Configurational frustration with electrostatics
model_mutational = frustratometer.AWSEM(structure) 
## Mutational/Configurational frustration without electrostatics
model_mutational_noelectrostatics = frustratometer.AWSEM(structure, k_electrostatics=0)
## Mutational frustration with sequence separation of 12
model_mutational_seqsep12 = frustratometer.AWSEM(structure, min_sequence_separation_rho=13)
## Typical openAWSEM
model_openAWSEM = frustratometer.AWSEM(structure, min_sequence_separation_contact = 10, distance_cutoff_contact = None)
```

### Calculating Residue Densities

To calculate the density of residues in the structure.

```python
calculated_densities = model_singleresidue.rho_r
print(calculated_densities)
```

### Calculating Frustration Indices

Frustration indices can be calculated for single residues or mutationally. This measurement helps identify energetically favorable or unfavorable interactions within the protein structure.

#### Single Residue Frustration

```python
# Calculate single residue frustration
single_residue_frustration = model_singleresidue.frustration(kind='singleresidue')
print(single_residue_frustration)
```

#### Single Residue Decoy Fluctuation
The frustratometer package also allows the quick calculation of the energies of all single residue and mutational decoys.

```python
# Calculate mutational frustration
mutational_frustration = model_singleresidue.decoy_fluctuation(kind='singleresidue')
print(mutational_frustration)
```

#### Mutational Frustration

```python
# Calculate mutational frustration
mutational_frustration = model_mutational.frustration(kind='mutational')
print(mutational_frustration)
```

### Energy Calculations

You can calculate different energy contributions, including fields energy (pseudo one-body terms like burial), couplings energy (pseudo two-body terms like contact and electrostatics), and their combination to determine the native energy of the protein structure.

#### Fields Energy

```python
fields_energy = model_openAWSEM.fields_energy()
print(fields_energy)
```

#### Couplings Energy

```python
couplings_energy = model_openAWSEM.couplings_energy()
print(couplings_energy)
```

#### Native Energy

Native energy can be considered as a combination of fields and couplings energy contributions.

```python
native_energy = model_openAWSEM.native_energy()
print(native_energy)
```

## Conclusion

The Frustratometer AWSEM package offers many functionalities for analyzing protein structures. By calculating residue densities, frustration indices, and various energy contributions, researchers can gain insights into the stability, energetics, and potentially functional aspects of protein conformations.

## Other flavors

The frustratometer has been implemented in other ways by our group:

Web-server

R

Lammps

Atomistic


## Copyright

Copyright (c) 2022-2024, Carlos Bueno, Hana Jaafari


## Acknowledgements
 
Project skeleton based on the [Computational Molecular Science Python Cookiecutter] (https://github.com/molssi/cookiecutter-cms) version 1.6.

## References

### frustratometeR

Atilio O Rausch, Maria I Freiberger, Cesar O Leonetti, Diego M Luna, Leandro G Radusky, Peter G Wolynes, Diego U Ferreiro, R Gonzalo Parra, FrustratometeR: an R-package to compute local frustration in protein structures, point mutants and MD simulations, Bioinformatics, 2021;, btab176, https://doi.org/10.1093/bioinformatics/btab176

