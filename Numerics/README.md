# Reaction-Diffusion simulation with C++, SFML, & Cuda

Based on the [Coding Train video](https://www.youtube.com/watch?v=BV9ny785UNc) and the subsequent C++/SFML [contribution by 'Deedone'](https://github.com/Deedone/Small-projects/tree/master/Reaction-diffusion)

This version is simplified and rewritten to include CUDA support for the simulation, so larger simulation grids can run faster.
Also, the reaction part of the equation was abstracted to a function, so they can be swapped out for models other than the [Gray-Scott](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/)

## How to run

- Make sure you have the SFML and CUDA libraries installed (latest).
- Build: 'make all'
- Run: './diffusion'
