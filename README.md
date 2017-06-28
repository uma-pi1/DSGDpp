# DSGDpp

This project contains code performing matrix factorizations in a parallel and distributed fashion. 

Quick start
===========

    cd build/tools

    ./generateSyntheticData

The script generateSyntheticData will generate in /tmp:
- a matrix with training data: train.mmc
- a matrix with test data: test.mmc
- 2 initial factor matrices: W.mma and H.mma

Then run the method of your choice in parallel:

    ./mfdsgdpp --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=2 --rank=10 --update="Nzsl_Nzl2(1)" --regularize="None" --loss="Nzsl_Nzl2(1)" --truncate="(-100,100)" --decay="BoldDriver(0.01)" --epochs=3

or distributed:

    mpirun --hosts localhost,localhost ./mfdsgdpp --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=2 --rank=10 --update="Nzsl_Nzl2(1)" --regularize="None" --loss="Nzsl_Nzl2(1)" --truncate="(-100,100)" --decay="BoldDriver(0.01)"



Preparing data 
==================

For the input matrices we use market matrix formats. See: http://math.nist.gov/MatrixMarket/formats.html. Use the script *generateSyntheticData* to generate toy examples of such matrices.


Running the factorization 
===============================

In the folder build/tools you can find tools that can run factorization from the command line. For more details for each method please consult our publication:

F. Makari, C. Teflioudi, R. Gemulla, P. J. Haas, Y. Sismanis

*Shared-Memory and Shared-Nothing Stochastic Gradient Descent Algorithms for Matrix Completion. *

In KAIS, 2013

To see how to use a tool, simply run it without arguments.
The usual parameters for the tools are:

--input-file: the matrix with the training data. E.g., /tmp/train.mmc

--input-test-file: the matrix with the test data. E.g., /tmp/test.mmc

--input-row-file: the matrix with the initial row factors. E.g., /tmp/W.mma 

--input-col-file: the matrix with the initial column factors. E.g., /tmp/H.mma

--tasks-per-rank: the number of threads to work per machine. E.g., 2

--epochs: the number of iterations (passes over all the training data points). E.g.: 20

--update: the update that needs to be performed. "Nzsl_Nzl2(1)" means Non-Zero-Squared-Loss with weighted L2 regularization and regularization parameter=1 as objective function.

--loss: the loss function to report after each epoch. E.g., "Nzsl_Nzl2(1)"

--regularize: This is experimental. Please always keep it ="None"

--rank: the rank of the factorization. E.g, 10, 50, 100

--decay: the step size selection mechanism. E.g. "BoldDriver(0.01)" will use BoldDriver (please refer to our publication for a reference on how BoldDriver works) with initial step size = 0.01

Below you can find a list of the implemented methods with example invocations. We also state if the methods can run in parallel (shared memory) and/or distributed (shared-nothing)

DSGD++ (parallel and distributed)
--------------------------------

To run DSGD++ locally with 2 threads

    ./mfdsgdpp --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=2 --rank=10 --update="Nzsl_Nzl2(1)" --regularize="None" --loss="Nzsl_Nzl2(1)" --truncate="(-100,100)" --decay="BoldDriver(0.01)" --epochs=3

To use MPI to distribute on many machines (substitute localhost with the machine names)

    mpirun --hosts localhost,localhost ./mfdsgdpp --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=2 --rank=10 --update="Nzsl_Nzl2(1)" --regularize="None" --loss="Nzsl_Nzl2(1)" --truncate="(-100,100)" --decay="BoldDriver(0.01)"

PSGD without locking aka HOGWILD (parallel)
-------------------------------------------
implemented  weighted L2 regularization only

    ./psgdNZL2NoLock --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=2

PSGD with locking (parallel)
-------------------------------------------
implemented  weighted L2 regularization only

    ./psgdNZL2Lock --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=2

ALS (parallel and distributed)
------------------------------

    mpirun --hosts localhost,localhost ./mfdap --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=1  --loss="Nzsl_Nzl2(1)" --epochs=3

ASGD (distributed)
------------------

    mpirun --hosts localhost,localhost ./mfasgd --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=1 --update="Nzsl_Nzl2(1)" --regularize="None" --loss="Nzsl_Nzl2(1)" --truncate="(-100,100)" --decay="BoldDriver(0.01)" --rank=10

CSGD  (parallel)
-----------------
    ./stratified-psgd --input-file=/tmp/train.mmc --input-test-file=/tmp/test.mmc  --input-row-file=/tmp/W.mma --input-col-file=/tmp/H.mma --tasks-per-rank=2 --eps0=0.001 --cache=256


Other tools
=================

In the folder build/tools you can find the following tools:

Create Initial Factor Matrices
------------------------------

Use the tool *mfcreateInitialFactors*. With this tool you can create the 2 initial factor matrices for the factorization of rank *rank* of an original input matrix of the form *size1* x *size2*. 
The values of these matrices can be drawn from specific distributions (e.g., "Normal(0,1)", "Uniform(-1,1)").

Average-out Original Matrix
---------------------------

In case you are working with real data, you might want to the average of all observed training points from the input and input-test matrix. To do this you can use the tool *averageOutMatrix*.

Sample and Project Input Matrix
-------------------------------

For using adaptive decay (instead of bold driver) as a step size selection mechanism, 
you will need to create a sample matrix from your input matrix and learn the best step size to use on this sample matrix. The tool *mfsample* can create such a sample matrix for you.
This tool samples the  sparse input matrix and decides the size of the projected matrix according to the following parameters:
It decides the rows and columns of the projected matrix either in terms of absolute numbers or as fraction of the rows and columns of the original matrix. 
Alternatively, one can supply the tool with the number of non-zero entries to be sampled. 
Additionally to the sample matrix, the tool can also generate files that contain the mappings for rows and columns between the original and sampled matrix.  

This tool (mfsample) does NOT eliminate empty rows and columns from the resulting projected matrix. 
To eliminate empty (or nearly empty) rows and columns use the tool *mfproject*. 
This tool keeps only rows and columns of the input matrix whose number of nonzero entries is above some threshold (with default threshold 0). 
It can also repeat this elimination process until all rows and columns pass the threshold.

*mfprepare* is a tool that combines the functionality of mfsample, mfproject and mfcreateInitialFactors, i.e.:
- (1) It samples the input matrix
- (2) It projects out the zero row and columns
- (3) It creates initial factors for a specific rank and according to some distribution

 Example invocation:
 
    ./mfprepare --input-file=/someDir/v.mmc --nnz=10000 --values="Uniform(0,1)" --rank=10
    
Convert Matrices Between Different Formats
------------------------------------------

To do so use *mfconvert*.
Supported extensions are .mma .mmc .bsb .bst .bdb .bdt

Generate a Martix Based on a Distribution
------------------------------------------

Use the tool *mfgenerate*. You can specify the dimension of the matrix, how sparse or dense it should be and from which distribution the values should be drawn.


Blocking of Matrices
--------------------

Reading the matrices normally takes place sequentially. The factorization algorithms then partition the matrices in some way before performing the factorization. 
If the matrices are partitioned before-hand, they can be read in parallel, reducing the reading overhead significantly. To partition the matrices use the *mfdblock* tool.

An example invocation could be:

    ./mfdblock --blocks1=2 --blocks2=4 --threads=2 --format="mmc" --input-file=/tmp/train.mmc --output-base-file=/tmp/blockedTrain
    
this will create in the /tmp folder the following files:

    blockedTrain-0-0.mmc
    blockedTrain-0-1.mmc
    blockedTrain-0-2.mmc
    blockedTrain-0-3.mmc
    blockedTrain-1-0.mmc
    blockedTrain-1-1.mmc
    blockedTrain-1-2.mmc
    blockedTrain-1-3.mmc
    blockedTrain.xml
    
A factorization method can then be called by using the .xml file as input-file.

Generate Large Synthetic Matrices On-The-Fly
--------------------------------------------

When using large synthetic data, lots of time might be wasted in reading/writing the matrices from/to the disk.
*mfcreateRandomMatrixFile* isa  tool that creates a file descriptor for generating synthetic matrices on the fly  in a parallel or distributed manner.
For a given experiment you will need 2 such files: 
- (i) a file containing the original factors + the data matrix + the test matrix (optionally) 
- (ii) a file containing the starting points (initial factors). 

E.g.:

    ./mfcreateRandomMatrixFile --size1=1000000 --size2=1000000 --nnz=10000000 --nnzTest=1000000 --rank=10 --values="Normal(0,10)" --noise="Normal(0,1)" --blocks1=10 --blocks2=10 --output-file=/tmp/synthetic.rm
Will create (i) and 

    ./mfcreateRandomMatrixFile --size1=1000000 --size2=1000000  --rank=10 --values="Uniform(-1,1)" --blocks1=10 --blocks2=10 --output-file=/tmp/syntheticFactors.rm
will create (ii).


We can then call DSGDpp as:

    mpirun --hosts localhost,localhost ./mfdsgdpp --input-file=/tmp/synthetic.rm --input-test-file=/tmp/synthetic.rm --input-row-file=/tmp/syntheticFactors.rm --input-col-file=/tmp/syntheticFactors.rm --tasks-per-rank=2 --rank=10 --update="Nzsl_Nzl2(1)" --regularize="None" --loss="Nzsl_Nzl2(1)" --truncate="(-100,100)" --decay="BoldDriver(0.01)"

Before creating a file take into consideration the following: 
- (1) parallelization happens in row-chunk manner. Please make sure that the blocks1 dimension is enough fine-grained, so that the threads of each node will have enough row-blocks to work on. 
- (2) make sure that the blocks1, blocks2 values are at least as much as the most fine-blocked matrix that you want your file to be able to create. 
- (3) blocks_in_file mod blocks_of_the_matrix = 0. 
- (4) rows/columns_of_the_matrix mod blocks_in_file = 0.




Tracing
=========================


