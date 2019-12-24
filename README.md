# (Finite Size) Density Matrix Renomalization Group

Solves for ground state of a spin-1/2 Heisenberg chain - A toy DMRG realization by Python

The code is developed under Block DMRG formalism (not Matrix Product States). The input are as annotated in the main.py, i.e. the interaction matrix, number of sites in the chain, and size of the Hilbert space to keep. 

- - - - - - - - - - - - - - - - - - -
main.py: Finite size DMRG

Warmup.py: For warming up the finite size algorithm, containing two functions:
           infinite sie DMRG and Sweep.

Wavefunction.py: A class representing wavefunction, formalized as a bipartite universe.

Block.py: A class representing block or sub-block in the system, with attributes needed for both infinite and finite size DMRG

Memory.py: A class for the storage of temporary results in different steps, useful in both infinite and finite size DMRG. 
        subject to update

helper.py: Defines tensor product, export logfile, and plot chain geometry
           
logfile.log: the output file, printing geometry, Hilbert space size ... etc, for every step in the algorithm
