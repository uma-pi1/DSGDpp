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
 * loadDistributedMatrix.h
 *
 *  Created on: Jul 13, 2011
 *      Author: chteflio
 */

#ifndef MF_MATRIX_IO_LOADDISTRIBUTEDMATRIX_H_
#define MF_MATRIX_IO_LOADDISTRIBUTEDMATRIX_H_

#include <mpi2/mpi2.h>

#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/io/descriptor.h>

namespace mf {

/** loads a blocked matrix from disk into a DestributedMatrix in memory.
 *
 * @param f descriptor for input files
 * @param name the name of the matrix
 * @param partitionByRow read row-wise or column-wise
 * @param tasksPerRank how many tasks to use for parallel reading on each rank
 *
 * @tparam M matrix type
 */
template<typename M>
DistributedMatrix<M> loadMatrix(const BlockedMatrixFileDescriptor& f,
		const std::string& name, bool partitionByRow, int tasksPerRank = 1);

/** loads a blocked matrix from disk into a DestributedMatrix in memory.
 *
 * @param file a descriptor file for input files (.xml) or the input file itself (.mmc)
 * @param name the name of the matrix
 * @param partitionByRow read row-wise or column-wise
 * @param tasksPerRank how many tasks to use for parallel reading on each rank
 * @param worldSize the number of nodes available for the distribution
 * @param blocks1 the # of row-blocks if the input is an mmc file
 * @param blocks2 the # of column-blocks if the input is an mmc file
 *
 * @tparam M matrix type
 */
template<typename M>
DistributedMatrix<M> loadMatrix(const std::string& file,
		const std::string& name, bool partitionByRow, int tasksPerRank = 1, int worldSize=1,
		mf_size_type blocks1=1,mf_size_type blocks2=1,bool test=false);

} // namespace mf

#include <mf/matrix/io/loadDistributedMatrix_impl.h>

#endif /* MF_MATRIX_IO_LOADDISTRIBUTEDMATRIX_H_ */
