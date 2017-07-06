# DSGDpp

This project contains implementations of various parallel algorithms for computing low-rank matrix factorizations. Both shared-memory and shared-nothing (via MPI) implementations are provided. The following algorithms are currently implemented:

- DSGD
- DSGD++
- Asynchronous SGD
- CSGD
- Parallel SGD with locking
- Lock-free parallel SGD (Hogwild)
- Alternating least squares
- Non-negative matrix factorization.

## Quick start

Follow the installation instructions given in the INSTALL file. First time installation and compilation may take while.

To generate some synthetic data, run:

    cd build/tools
    ./generateSyntheticData

The script `generateSyntheticData` will generate the following files:
- `/tmp/train.mmc`: a matrix with some training data
- `/tmp/test.mmc`: a matrix with some test data
- `/tmp/W.mma` and `/tmp/H.mma`: initial factor matrices

To run DSGD++ on a single-machine with 4 threads, type:

    ./mfdsgdpp --tasks-per-rank=4 --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --rank=10 --update="Nzsl_L2(1)" --regularize="None" --loss="Nzsl_L2(1)" --decay="BoldDriver(0.001)" --epochs=50 --trace=/tmp/trace.R

This gives you output such as:

        5 mpi2 | Initialized mpi2 on 1 rank(s)
        6 mpi2 | Starting task managers (parallel mode)...
        6 mpi2 | Started task manager at rank 0
      108 main | Input
      108 main |     Input file: /tmp/train.mmc
      108 main |     Input test file: /tmp/test.mmc
      108 main |     Input row factors: /tmp/W.mma
      108 main |     Input column factors: /tmp/H.mma
      108 main | Output
      108 main |     Output row factors: Disabled
      108 main |     Output column factors: Disabled
      108 main |     Trace: /tmp/trace.R (trace)
      108 main | Parallelization
      108 main |     MPI ranks: 1
      108 main |     Tasks per rank: 4
      108 main | DSGD++ options
      108 main |     Seed: 1499165899
      108 main |     Epochs: 50
      108 main |     Factorization rank: 10
      108 main |     Update function: Nzsl_L2(1)
      108 main |     Regularize function: None
      108 main |     Loss function: Nzsl_L2(1)
      108 main |     Decay: BoldDriver(0.001)
      108 main |     Balancing: Disabled
      108 main |     Absolute function: Disabled
      108 main |     Truncation: Disabled
      108 main |     SGD step sequence: WOR
      108 main |     DSGD stratum sequence: COWOR
      109 mf   | File '/tmp/train.mmc' is not blocked; it will be blocked automatically
      110 mf   | Constructing blocks (0,0) (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (0,7) (1,0) (1,1) (1,2) (1,3) (1,4) (1,5) (1,6) (1,7) (2,0) (2,1) (2,2) (2,3) (2,4) (2,5) (2,6) (2,7) (3,0) (3,1) (3,2) (3,3) (3,4) (3,5) (3,6) (3,7)  of '/tmp/train.mmc'
     3664 mf   | File '/tmp/test.mmc' is not blocked; it will be blocked automatically
     3664 mf   | Constructing blocks (0,0) (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (0,7) (1,0) (1,1) (1,2) (1,3) (1,4) (1,5) (1,6) (1,7) (2,0) (2,1) (2,2) (2,3) (2,4) (2,5) (2,6) (2,7) (3,0) (3,1) (3,2) (3,3) (3,4) (3,5) (3,6) (3,7)  of '/tmp/test.mmc'
     3959 mf   | Test matrix: 100000 x 100000, 100000 nonzeros, 4 x 8 blocks
     3961 mf   | Data matrix: 100000 x 100000, 1000000 nonzeros, 4 x 8 blocks
    Time for generating data and test matrices: 3.85172s
     3961 mf   | File '/tmp/W.mma' is not blocked; it will be blocked automatically
     3961 mf   | Constructing blocks (0,0) (1,0) (2,0) (3,0)  of '/tmp/W.mma'
     4540 mf   | Row factor matrix: 100000 x 10, 4 x 1 blocks
     4540 mf   | File '/tmp/H.mma' is not blocked; it will be blocked automatically
     4541 mf   | Constructing blocks (0,0) (0,1) (0,2) (0,3) (0,4) (0,5) (0,6) (0,7)  of '/tmp/H.mma'
     5209 mf   | Column factor matrix: 10 x 100000, 1 x 8 blocks
     6450 main | Using NzslLoss for test data
     6450 mf   | Starting DSGD++ (polling delay: 500 microseconds)
     6450 mf   | Using COWOR order for selecting strata
     6450 mf   | Using WOR order for selecting training points
     6869 mf   | Loss: 1.00091e+09 (0.418913s)
     6936 mf   | Test loss: 9.98073e+07 (0.066328s)
     6936 mf   | Step size: 0.001 (0s)
     6936 mf   | Starting epoch 1
     7457 mf   | Finished epoch 1 (0.521064s)
     7999 mf   | Loss: 9.94145e+08 (0.542368s)
     8077 mf   | Test loss: 9.98114e+07 (0.077686s)
    ...   

To get some idea about the performance of the algorithm, start an `R` console and type:

    source("mfplot.R")
    source("/tmp/trace.R")                               # created by running DSGD++ above
    mfplot(trace, x.is="time$elapsed", log="y")          # plots the training loss over time

To run DSGD++ on two machines (here both localhost) with 4 threads each, type:

    mpirun --hosts localhost,localhost ./mfdsgdpp --tasks-per-rank=4 --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --rank=10 --update="Nzsl_L2(1)" --regularize="None" --loss="Nzsl_L2(1)" --decay="BoldDriver(0.001)" --epochs=50 --trace=/tmp/trace.R

## Preparing data 

To use DSGDpp, convert your data to a [market matrix format](http://math.nist.gov/MatrixMarket/formats.html). We also provide tools to generate and convert matrices (e.g., `mfgenerate`, `mfconvert`, `mfsample`, `mfdblock`).

To run a factorization in a shared-nothing environment, the data must lie in a shared filesystem mounted at the same location on all machines.

Note that many algorithms require the data to be blocked in some form. The required blocking will be created on the fly, but we did not optimize the performance of this step. For faster data loading, matrices should be blocked up front via the `mfdblock` tool.

## Tools

We provide a number of tools to run factorizations and prepare data. All tools are located under `build/tools`. Run a tool executable without arguments for help on its parameters. E.g., runnung `./mfdsgdpp` produces

    mfsgdpp [options]
    
    Options:
      --help                produce help message
      --input-file arg      filename of data matrix
      --input-test-file arg filename of test matrix
      --input-row-file arg  filename of initial row factors
      --input-col-file arg  filename of initial column factors
      --output-row-file arg filename of final row factors                                               
      --output-col-file arg filename of final column factors                                            
      --trace arg           filename of trace [trace.R]                                                 
      --trace-var arg       variable name for trace [traceVar]                                          
      --epochs arg          number of epochs to run [10]                                                
      --tasks-per-rank arg  number of concurrent tasks per rank [1]                                     
      --sgd-order arg       order of SGD steps [WOR] (e.g., "SEQ", "WR", "WOR")                         
      --stratum-order arg   order of strata [COWOR] (e.g., "SEQ", "RSEQ", "WR",                         
                            "WOR", "COWOR")                                                             
      --seed arg            seed for random number generator (system time if not                        
                            set)                                                                        
      --rank arg            rank of factorization                                                       
      --update arg          SGD update function (e.g., "Sl", "Nzsl", "GklData")                         
      --regularize arg      SGD regularization function (e.g., "None", "L2(0.05)",                      
                            "Nzl2(0.05)")
      --loss arg            loss function (e.g., "Nzsl", "Nzsl_L2(0.5)"))
      --abs                 if present, absolute values are taken after every SGD 
                            step
      --truncate arg        if present, truncatation is enabled (e.g., --truncate 
                            "(-1000, 1000)"
      --decay arg           decay function (constant, bold driver, or auto)
      --balance arg         Type of balancing (None, L2, Nzl2)

The source code contains additional information about accepted parameter values. 

### Factorizating matrices

To run a method in a shared-nothing environment, simply prefix the command with `mpirun --hosts <list-of-hostnames>`.

| Tool | Method(s) |
| ---- | ------ |
| mfasgd | Asynchronous SGD |
| mfdsgdpp | DSGD++ |
| psgdL2Lock | Parallel SGD with locking, L2 loss |
| psgdL2NoLock | Parallel SGD without locking (Hogwild), L2 loss |
| psgdNZL2Lock | Parallel SGD with locking, NZL2 loss |
| psgdNZL2NoLock | Parallel SGD without locking (Hogwild), NZL2 loss |
| mfdap | Alternating least squares, non-negative matrix factorization (via multiplicative updates) |
| mfdsgd | DSGD |
| mfsgd | Sequential SGD |
| stratified-psgd | CSGD (cache-conscious parallel SGD) |

### Utilities

| Tool | Description |
| ---- | ----------- |
| averageOutMatrix | Center a matrix (substract mean from every observed entry) |
| mfconvert | Convert between different file formats |
| mfcreateInitialFactors | Create initial factor matrices |
| mfcreateRandomMatrixFile | Create a blocked random matrix |
| mfgenerate | Geneate random matrices |
| mfproject | Drops rows/columns with fewer entries than a given threshold |
| mfdblock | Blocks an input matrix |
| mfsample | Creates a sample of an input matrix (e.g., used for automatic step size selection) |
| mfprepare | Combines mfsample, mfproject, and mfcreateInitialFactors |

### Citations

  - **CSGD**
  
  
    F. Makari, C. Teflioudi, R. Gemulla, P. J. Haas, Y. Sismanis
    
    *Shared-Memory and Shared-Nothing Stochastic Gradient Descent Algorithms for Matrix Completion.*
    
    In KAIS, 2013
    

  - **DSGD++, Asynchronous SGD, Parallel SGD with locking**
  
  
    C. Teflioudi, F. Makari, R. Gemulla
    
    *Distributed Matrix Completion.*
    
    In ICDM, 2012.
    

  - **DSGD**

    
    R. Gemulla, E. Nijkamp, P. J. Haas, Y. Sismanis
    
    *Large-Scale Matrix Factorization with Distributed Stochastic Gradient Descent*
    
    In KDD, 2011
    

  - **Lock-free parallel SGD (Hogwild)**
  
  
    F. Niu, B. Recht, C. Re, and S. J. Wright, 
    
    *Hogwild!: A lock-free approach to parallelizing stochastic gradient descent*
    
    In NIPS, 2011
    

  - **Alternating least squares**

  
    Y. Zhou, D. Wilkinson, R. Schreiber, and R. Pan, 
    
    *Large-scale parallel collaborative filtering for the Netflix Prize*
    
    In AAIM, 2008
