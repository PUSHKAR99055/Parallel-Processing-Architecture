# Parallel-Processing-Architecture

Abstract—This report outlines the Single Source Shortest Path (SSSP) algorithm for finding the shortest path between two points in a matrix using CUDA parallel computing. It covers the algorithm, challenges, and implementation details.

Index Terms—SSSP, CUDA, parallel computing.

I. Introduction
The problem has two parts: calculating the product matrix of two input matrices while identifying the first two minima and their indices, and then finding the shortest path between these points. OpenCV is used for visualizing the computed path.

II. Parallel Matrix Multiplication and Minima
A. Parallel Matrix Multiplication
Tiled matrix multiplication improves performance by optimizing shared memory usage in CUDA.

B. Block Reduction
Block reduction identifies the minimum value and its location using a stride-based approach. Threads compare values in shared memory iteratively until the minimum is found.

C. Second Minimum
The first minimum is replaced with the maximum float value, and the block reduction kernel is rerun to find the second minimum.

III. Single Source Shortest Weighted Path (SSSP)
A. Graph Representation
The matrix is represented as a graph with vertices, edges, and weights. Intermediate costs and paths are stored in arrays, with atomic functions ensuring memory consistency.

B. SSSP Algorithm
Two CUDA kernels are used:

Kernel 1 updates intermediate costs and paths for neighbors of the source.
Kernel 2 refines final costs, enabling threads to update their neighbors iteratively until convergence.
