# (Finite Size) Density Matrix Renomalization Group

Solves for ground state of a spin-1/2 Heisenberg chain - A toy DMRG realization by Python

The code is developed under Block DMRG formalism (not Matrix Product States). The input are as annotated in the main.py, i.e. the interaction matrix, number of sites in the chain, and size of the Hilbert space to keep. 

- - - - - - - - - - - - - - - - - - -
main.py: Finite size DMRG

Warmup.py: For warming up the finite size algorithm, containing two functions:
           infinite sie DMRG and Sweep.

Block.py: A class representing Block, with attributes needed for both infinite and finite size DMRG

helper.py: Defines tensor product, export logfile, and plot chain geometry
           
