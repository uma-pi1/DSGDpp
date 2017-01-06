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
#ifndef MF_MATRIX_IO_STORE_H
#define MF_MATRIX_IO_STORE_H

#include <mpi2/mpi2.h>

#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>


namespace mf {

/** Writes the given matrix into a set of files. Each block is written to one file.
 *
 * @param m input matrix
 * @param f descriptor for output files
 * @param tasksPerRank how many tasks to use for parallel writing on each rank
 * @tparam M matrix type
 */
template<typename M>
void storeMatrix(const DistributedMatrix<M>& m, const BlockedMatrixFileDescriptor& f,
		int tasksPerRank = 1);

} // namespace mf

#include <mf/matrix/io/store_impl.h>

#endif
