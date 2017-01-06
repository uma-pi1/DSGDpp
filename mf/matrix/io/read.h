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
#ifndef MF_MATRIX_IO_READ_H
#define MF_MATRIX_IO_READ_H

#include <mf/types.h>
#include <mf/matrix/io/format.h>

namespace mf {

/** Reads a matrix from a file into memory.
 *
 * @param fname file name
 * @param[out] m output matrix
 * @param format file format
 * @tparam M matrix type
 */
template<typename M>
void readMatrix(const std::string& fname, M& m, MatrixFileFormat format = AUTOMATIC);


/** Reads some blocks of a matrix from a file into memory. This method is efficient for
 * the matrix-file formats, but inefficient for all other formats.
 *
 * @param fname file name
 * @param blocks1 number of row blocks
 * @param blocks2 number of column blocks
 * @param sortedBlockList list of blocks to read (must be sorted by row, then column)
 * @param[in,out] blockOffsets1 offsets of row blocks (automatically computed if emtpy)
 * @param[in,out] blockOffsets2 offsets of column blocks (automatically computed if emtpy)
 * @param[out] size1 total number of rows in the matrix
 * @param[out] size2 total number of columns in the matrix
 * @param[out] blocks a list of read blocks (same order as sortedBlockList)
 * @param format file format
 * @tparam M matrix type
 * @tparam SparseOut (internal) whether the output matrix type is sparse
 */
template<typename M>
void readMatrixBlocks(const std::string& fname,
		mf_size_type blocks1, mf_size_type blocks2,
		const std::vector<std::pair<mf_size_type, mf_size_type> >& sortedBlockList,
		std::vector<mf_size_type>& blockOffsets1, std::vector<mf_size_type>& blockOffsets2,
		mf_size_type& size1, mf_size_type& size2, std::vector<M*>& blocks,
		MatrixFileFormat format = AUTOMATIC);


/** Specialization of mf::readMatrixBlocks for sparse matrices. */
template<class L, std::size_t IB, class IA, class TA>
void readMatrixBlocks(
		const std::string& fname,
		mf_size_type blocks1, mf_size_type blocks2,
		const std::vector<std::pair<mf_size_type, mf_size_type> >& sortedBlockList,
		std::vector<mf_size_type>& blockOffsets1, std::vector<mf_size_type>& blockOffsets2,
		mf_size_type& size1, mf_size_type& size2,
		std::vector<boost::numeric::ublas::coordinate_matrix<double, L, IB, IA, TA>*>& blocks,
		MatrixFileFormat format = AUTOMATIC);

} // namespace mf

#include <mf/matrix/io/read_impl.h>

#endif
