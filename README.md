Frustratometer
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/cabb99/Frustratometer/workflows/CI/badge.svg)](https://github.com/cabb99/Frustratometer/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/HanaJaafari/Frustratometer/graph/badge.svg?token=JKDOXOYPRS)](https://codecov.io/gh/HanaJaafari/Frustratometer)
[![Documentation Status](https://readthedocs.org/projects/frustratometer/badge/?version=latest)](https://frustratometer.readthedocs.io/en/latest/?badge=latest)



A python implementation of the protein frustratometer.

<p align="center">
    <a href="https://xkcd.com/173/">
        <img src="https://imgs.xkcd.com/comics/movie_seating.png" alt="XKCD depiction of frustration at the movies" title="At least amino acids don't create directed graphs">
    </a> <br>
    <em> XKCD comic showing a clasic example of frustration at the movie theather </em>
</p>


The frustratometer is based on the principle of minimal frustration postulated by [Wolynes et al.](https://doi.org/10.1073/pnas.84.21.7524) Proteins have evolved under a selective pressure to fold into a single native structure, which is energetically minimized by the evolutionary process, but also to not fold into the many possible incorrect glassy conformations, where the energy is maximized by the evolutionary process. In that principle it is expected that when we measure the energy of a correctly folded protein, most of the interactions between the amino acids will be minimized, compared to both a different amino acid in the same position (mutational) and the same amino acid in a different position (configurational). When this energy is minimized for a particular residue or interaction we define it to be minimally frustrated.

Under most circunstances amino acids are minimally frustrated in the protein, but pockets where amino acids are of interest for protein function can remain frustrated, as they become minimally frustrated when achieving the corresponding function. In short, a frustrated residue or interaction indicates that the particular amino acid would be minimized under a different environment, which reveals possible competing evolutionary pressures on its selection. We have identified that these pockets usually correspond to regions of functional importance, for example, that may form part of a catalitic domain, a hinge domain, or a binding region.

This principle of minimum frustration has been shown using the AWSEM forcefield but can be extended to any other forcefield, including atomistic forcefields, or a pseudo-forcefield like DCA, as implemented here.

## Installation

To install this modules please clone this repository and install it using the following commands.

    git clone HanaJaafari/Frustratometer
    cd Frustratometer
    conda install -c conda-forge --file requirements.txt
    pip install -e .

There are additional packages that can be installed and are detailed in the [documentation](https://frustratometer.readthedocs.io/en/latest).
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

[frustratometer-server](http://frustratometer.qb.fcen.uba.ar/)

[frustratometeR](https://github.com/proteinphysiologylab/frustratometeR)

[AWSEM-MD](https://github.com/adavtyan/awsemmd)

[Atomistic](https://www.nature.com/articles/s41467-020-19560-9)

## Copyright

Copyright (c) 2022-2024, Carlos Bueno, Hana Jaafari


## Acknowledgements
 
This work was supported by National Science Foundation (NSF) Center for Theoretical Biological Physics (NSF PHY-2019745) and NSF grants PHY-1522550.
Project skeleton based on the [Computational Molecular Science Python Cookiecutter] (https://github.com/molssi/cookiecutter-cms) version 1.6.

## References

Atilio O Rausch, Maria I Freiberger, Cesar O Leonetti, Diego M Luna, Leandro G Radusky, Peter G Wolynes, Diego U Ferreiro, R Gonzalo Parra, FrustratometeR: an R-package to compute local frustration in protein structures, point mutants and MD simulations, Bioinformatics, 2021;, btab176, https://doi.org/10.1093/bioinformatics/btab176

Ferreiro DU, Komives EA, Wolynes PG. Frustration, function and folding. Curr Opin Struct Biol. 2018;48: 68–73. doi:10.1016/j.sbi.2017.09.006. PubMed PMID: 29101782

Parra RG, Schafer NP, Radusky LG, Tsai M-Y, Guzovsky AB, Wolynes PG, et al. Protein Frustratometer 2: a tool to localize energetic frustration in protein  molecules, now with electrostatics. Nucleic Acids Res. 2016;44: W356-60. doi:10.1093/nar/gkw304. PubMed PMID: 27131359

Wolynes PG. Evolution, energy landscapes and the paradoxes of protein folding. Biochimie. 2015;119: 218–230. doi:10.1016/j.biochi.2014.12.007. PubMed PMID: 25530262

Ferreiro DU, Komives E a., Wolynes PG. Frustration in Biomolecules. Q Rev Biophys. 2013;47: 1–97. doi:10.1017/S0033583514000092. PubMed PMID: 25225856

Jenik M, Parra RG, Radusky LG, Turjanski A, Wolynes PG, Ferreiro DU. Protein frustratometer: A tool to localize energetic frustration in protein molecules. Nucleic Acids Res. 2012;40: 348–351. doi:10.1093/nar/gks447. PubMed PMID: 22645321
