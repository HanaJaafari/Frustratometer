DCA Frustratometer
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/HanaJaafari/dca_frustratometer/workflows/CI/badge.svg)](https://github.com/HanaJaafari/dca_frustratometer/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/HanaJaafari/DCA Frustratometer/branch/master/graph/badge.svg)](https://codecov.io/gh/HanaJaafari/DCA_Frustratometer/branch/master)

A python implementation of the protein frustratometer.

![https://xkcd.com/173/](https://imgs.xkcd.com/comics/movie_seating.png)

The frustratometer is based on the principle of minimal frustration postulated by Wolynes et al. The main idea is that proteins have evolved under a selective pressure to fold into a native structure, which is energetically minimized, but also to not fold into incorrect conformations, where the energy is maximized. In that principle it is expected that when we measure the energy of a correctly folded protein, most of the interactions between the aminoacids will be minimized, compared to a different aminoacid in the same position or the same aminoacid in a different position. When this energy is minimized we define a particular residue to be minimally frustrated.

Under most circunstances aminoacids are minimally frustrated in the protein, but pockets where aminoacids are of interest for protein function. In short, a frustrated residue or interaction indicates that the particular aminoacid would be minimized under a different configuration, or a different sequence, which reveals possible competing evolutionary pressures on its selection. We have identified that these pockets usually correspond to regions of functional importance, for example, that may form part of a catalitic domain, a hinge domain, or a binding region.

This principle of minimum frustration has been shown using the AWSEM forcefield but can be extended to any other forcefield, including atomistic forcefields, or a pseudo-forcefield like DCA, as implemented here.

In this module we implement a version of the frustratometer based on Direct Coupling analysis.

## Installation

This module is currently under development.
To install this modules please clone this repository and install it using the following commands.

    git clone HanaJaafari/dca_frustratometer
    cd dca_frustratometer
    conda install -c conda-forge --file requirements.txt
    pip install -e .

A clean python environment is recommended.

## Usage

...

## Other flavors

The frustratometer has been implemented in other ways by our group:

Web-server

R

Lammps

Atomistic


## Copyright

Copyright (c) 2022, Carlos Bueno, Hana Jaafari


## Acknowledgements
 
Project skeleton based on the [Computational Molecular Science Python Cookiecutter] (https://github.com/molssi/cookiecutter-cms) version 1.6.

## References

### frustratometeR

Atilio O Rausch, Maria I Freiberger, Cesar O Leonetti, Diego M Luna, Leandro G Radusky, Peter G Wolynes, Diego U Ferreiro, R Gonzalo Parra, FrustratometeR: an R-package to compute local frustration in protein structures, point mutants and MD simulations, Bioinformatics, 2021;, btab176, https://doi.org/10.1093/bioinformatics/btab176

