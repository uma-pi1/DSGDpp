//    Copyright 2017 Rainer Gemulla
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
/*
 * generateDistributedMatrix.h
 *
 *  Created on: May 15, 2012
 *      Author: chteflio

 */

#ifndef MF_MATRIX_IO_GENERATEDISTRIBUTEDMATRIX_H_
#define MF_MATRIX_IO_GENERATEDISTRIBUTEDMATRIX_H_

#include <mpi2/mpi2.h>
#include <log4cxx/logger.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/io/randomMatrixDescriptor.h>

namespace mf{
/*
 * 	generates the factor matrices from a RandomMatrixDescriptor
 *	in a distributed or parallel manner
 *
 *	W: blocks1 x 1
 *	H: 1 x blocks1
 * */
template<typename M>
DistributedMatrix<M> generateFactor(const RandomMatrixDescriptor& f, const std::string& name,mf_size_type blocks1,mf_size_type blocks2,
		bool rowBlocks, int tasksPerRank);
/*
 * 	generates the data / test matrix from a RandomMatrixDescriptor
 *	in a distributed or parallel manner
 * */
template<typename M>
void generateDataMatrices(const RandomMatrixDescriptor& f, DistributedMatrix<M>& dv, int tasksPerRank,DistributedMatrix<M>* dvTest=NULL);

/*
 * Gets a pair of Factors W,H. in the form of distributed Matrices with the default names "W", "H"
 * W: blocks1 x 1
 * H: 1 x block2
 * If fileW is a .rm file it generates the Factors, otherwise it loads the matrices from the files fileW,fileH
 *
 * if forAsgd=true and you load the data, the code uses 1 taskPerRank to load them
 *
 *
 */
std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> getFactors(const std::string& fileW,
		const std::string& fileH, int tasksPerRank, int worldSize,
		mf_size_type blocks1, mf_size_type blocks2, bool forAsgd);
/*
 * generates or load the data / test matrix from a file(s)
 *
 * returns a vector of data matrices in the form of distributed Matrices with the  names V: name , Vtest: name+test
 * V:blocks1 x blocks2
 *
 * If fileV is a .rm file it generates the Matrices, otherwise it loads the matrices from the files fileV,fileVtest
 * if fileVtest!=NULL it returns both data and test matrices, otherwise only the data matrix
 *
 * if forAsgd=true and you load the data, the code uses 1 taskPerRank to load them
 *
 * if forDap=true the test matrix needs to be blocked tasksPerRank*worldSize x tasksPerRank*worldSize
 * and not worldSize x tasksPerRank*worldSize as the data matrix
 * */
template<typename M>
std::vector<DistributedMatrix<M> > getDataMatrices(const std::string& fileV, const std::string& name, bool partitionByRow,
		int tasksPerRank, int worldSize, mf_size_type blocks1, mf_size_type blocks2, bool forAsgd, bool forDap,
		std::string* fileVtest=NULL);
}

#include <mf/matrix/io/generateDistributedMatrix_impl.h>
#endif /* MF_MATRIX_IO_GENERATEDISTRIBUTEDMATRIX_H_ */
