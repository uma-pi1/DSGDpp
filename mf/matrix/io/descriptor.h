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
#ifndef MF_MATRIX_IO_DESCRIPTOR_H
#define MF_MATRIX_IO_DESCRIPTOR_H

#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/io/format.h>

namespace mf {

/** Describes a blocked matrix that is stored one file per block. Descriptors are the main
 * tool to quickly load and store distributed matrices. The descriptor contains
 * information about the matrix itself (such as dimensions, number of blocks, block offsets)
 * and storage information (such as where the matrix is stored and in which format). */
struct BlockedMatrixFileDescriptor {
	/** Creates a descriptor for a distributed sparse matrix. The descriptor can be used to store
	 * the matrix to disk (see mf::storeMatrix).
	 *
	 * @param m input matrix
	 * @param path path at which to put files
	 * @param baseFilename prefix of filename
	 * @param format file format
	 */
	template<class T, class L, std::size_t IB, class IA, class TA>
	static BlockedMatrixFileDescriptor create(
			const DistributedMatrix<boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA> >& m,
			const std::string& path,
			const std::string& baseFilename, MatrixFileFormat format = MM_COORD);

	/** Creates a descriptor for a distributed dense matrix. The descriptor can be used to store
	 * the matrix to disk (see mf::storeMatrix).
	 *
	 * @param m input matrix
	 * @param path path at which to put files
	 * @param baseFilename prefix of filename
	 * @param format file format
	 */
	template<class T, class L, class A>
	static BlockedMatrixFileDescriptor create(
			const DistributedMatrix<boost::numeric::ublas::matrix<T,L,A> >& m,
			const std::string& path,
			const std::string& baseFilename, MatrixFileFormat format = MM_ARRAY);

	/** Stores the descriptor as XML file. */
    void load(const std::string &filename);

    /** Reads a descriptor from an XML file. */
    void save(const std::string &filename);


	mf_size_type size1;
	mf_size_type size2;
	mf_size_type blocks1;
	mf_size_type blocks2;
	std::vector<mf_size_type> blockOffsets1;
	std::vector<mf_size_type> blockOffsets2;
	std::string path;
	boost::numeric::ublas::matrix<std::string> filenames;
	MatrixFileFormat format;
};

} // namespace mf

#include <mf/matrix/io/descriptor_impl.h>

#endif
