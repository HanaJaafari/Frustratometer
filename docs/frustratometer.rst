Frustratometer
==============================

.. image:: https://github.com/HanaJaafari/Frustratometer/workflows/CI/badge.svg
    :target: https://github.com/HanaJaafari/Frustratometer/actions?query=workflow%3ACI

.. image:: https://codecov.io/gh/HanaJaafari/Frustratometer/graph/badge.svg?token=JKDOXOYPRS
    :target: https://codecov.io/gh/HanaJaafari/Frustratometer

.. image:: https://readthedocs.org/projects/frustratometer/badge/?version=latest
    :target: https://frustratometer.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status
    
A python implementation of the protein frustratometer.

.. image:: https://imgs.xkcd.com/comics/movie_seating.png
    :target: https://xkcd.com/173/

The frustratometer is based on the principle of minimal frustration postulated by Wolynes et al. The main idea is that proteins have evolved under a selective pressure to fold into a native structure, which is energetically minimized, but also to not fold into incorrect conformations, where the energy is maximized. In that principle, it is expected that when we measure the energy of a correctly folded protein, most of the interactions between the amino acids will be minimized, compared to a different amino acid in the same position or the same amino acid in a different position. When this energy is minimized, we define a particular residue to be minimally frustrated.

Under most circumstances, amino acids are minimally frustrated in the protein, but pockets where amino acids are of interest for protein function. In short, a frustrated residue or interaction indicates that the particular amino acid would be minimized under a different configuration, or a different sequence, which reveals possible competing evolutionary pressures on its selection. We have identified that these pockets usually correspond to regions of functional importance, for example, that may form part of a catalytic domain, a hinge domain, or a binding region.

This principle of minimum frustration has been shown using the AWSEM forcefield but can be extended to any other forcefield, including atomistic forcefields, or a pseudo-forcefield like DCA, as implemented here.

In this module, we implement a version of the frustratometer based on Direct Coupling analysis.

Copyright
---------

Copyright (c) 2022-2024, Carlos Bueno, Hana Jaafari

Acknowledgements
----------------

This work was supported by National Science Foundation (NSF) Center for Theoretical Biological Physics (NSF PHY-2019745) and NSF grants PHY-1522550.

Project skeleton based on the Computational Molecular Science Python Cookiecutter version 1.6.

