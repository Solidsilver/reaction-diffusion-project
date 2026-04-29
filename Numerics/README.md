# Reaction-Diffusion Simulation

Based on the [Coding Train video](https://www.youtube.com/watch?v=BV9ny785UNc) and the subsequent C++/SFML [contribution by 'Deedone'](https://github.com/Deedone/Small-projects/tree/master/Reaction-diffusion)

This repository contains multiple implementations of a Gray-Scott reaction-diffusion system.

## Project Structure

- `cuda/` - Original CUDA implementation using C++, SFML, and CUDA.
- `metal-swift/` - Apple Metal/Swift implementation for Apple Silicon.
- `Samples/` - Sample outputs and recordings.

The reaction part of the equation is abstracted to functions, so they can be swapped out for models other than the [Gray-Scott](https://groups.csail.mit.edu/mac/projects/amorphous/GrayScott/)

## Samples

1000x1000 Realtime (f: 0.055, k:0.062)
![alt text](./Samples/rdiff_cuda_default.gif "sample1")

## How to run (CUDA)

```bash
cd cuda
make all
./diffusion
```

- Make sure you have the SFML and CUDA libraries installed (latest).

## How to run (Metal/Swift)

```bash
cd metal-swift
swift run
```

- Requires macOS 11+ and Xcode with Swift 5.5+.
- Runs a 1920x1080 simulation using Metal compute shaders on the GPU.
- Press **Space** to start/pause the simulation.
- Default configuration uses 100 substeps per frame and a mix of rectangle and circle seeds.
- You can customize the initial seed shapes (rectangles and circles) and simulation parameters in `Sources/DiffusionMetal/main.swift`.
