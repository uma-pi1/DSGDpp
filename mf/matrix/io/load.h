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
#ifndef MF_MATRIX_IO_LOAD_H
#define MF_MATRIX_IO_LOAD_H

#include <utility>

#include <mpi2/mpi2.h>

#include <mf/matrix/io/format.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>


namespace mf {

/** Loads a distributed matrix from an unblocked file, picking block offsets and storage
 * locations automatically.
 * Each block will get roughly the same number of rows and columns.
 *
 * @param name Name of the output matrix (must be unique across the cluster)
 * @param blocks1 number of row blocks
 * @param blocks2 number of column blocks
 * @param partitionByRow whether the matrix should be partitioned by row (if false, partitioning
 	 	 is performed by column)
 * @param fname name of file to read from
 * @param format file format
 * @return handle for the newly created distributed matrix
 *
 * @tparam M matrix type
 */
template<typename M>
DistributedMatrix<M> loadMatrix(
		const std::string& name, mf_size_type blocks1, mf_size_type blocks2,
		bool partitionByRow,
		const std::string& fname, MatrixFileFormat format = AUTOMATIC);

/** Loads a distributed matrix from an unblocked file, taking block offsets as input (to be used together with clustering).
 * Each block will get roughly the same number of rows and columns.
 *
 * @param name Name of the output matrix (must be unique across the cluster)
 * @param blockOffsets1 vector of row blocksOffsets
 * @param blockOffsets2 vector of column blockOffsets
 * @param partitionByRow whether the matrix should be partitioned by row (if false, partitioning
 	 	 is performed by column)
 * @param fname name of file to read from
 * @param format file format
 * @return handle for the newly created distributed matrix
 *
 * @tparam M matrix type
 */
//template<typename M>
//DistributedMatrix<M> loadMatrix(
//		const std::string& name, std::vector<mf_size_type> blockOffsets1, std::vector<mf_size_type> blockOffsets2,
//		bool partitionByRow,
//		const std::string& fname, MatrixFileFormat format = AUTOMATIC);

/** Loads a distributed matrix from an unblocked file.
 *
 * @param name Name of the output matrix (must be unique across the cluster)
 * @param blockLocations (blocks1 x blocks2 matrix containing the rank of where to store each block)
 * @param[in,out] blockOffsets1 Offsets of row blocks (automatically computed if empty)
 * @param[in,out] blockOffsets2 Offsets of column blocks (automatically computed if empty)
 * @param fname name of file to read from
 * @param format file format
 * @return handle for the newly created distributed matrix
 *
 * @tparam M matrix type
 */
template<typename M>
DistributedMatrix<M> loadMatrix(
		const std::string& name, const boost::numeric::ublas::matrix<int>& blockLocations,
		const std::vector<mf_size_type>& blockOffsets1, // can be empty
		const std::vector<mf_size_type>& blockOffsets2, // can be empty
		const std::string& fname, MatrixFileFormat format = AUTOMATIC);


/** Distributes a matrix across a cluster, picking block offsets and storage
 * locations automatically.
 * Each block will get roughly the same number of rows and columns.
 * This method is currently fully sequential and inefficient.
 *
 * @param name Name of the output matrix (must be unique across the cluster)
 * @param blocks1 number of row blocks
 * @param blocks2 number of column blocks
 * @param partitionByRow whether the matrix should be partitioned by row (if false, partitioning
 	 	 is performed by column)
 * @param m matrix to distribute
 * @return handle for the newly created distributed matrix
 * @tparam M matrix type
 */
template<typename M>
DistributedMatrix<M> distributeMatrix(
		const std::string& name, mf_size_type blocks1, mf_size_type blocks2, bool partitionByRow,
		const M& m);

} // namespace mf

#include <mf/matrix/io/load_impl.h>

#endif
