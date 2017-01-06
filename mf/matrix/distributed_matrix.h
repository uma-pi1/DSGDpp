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
#ifndef MF_MATRIX_MATRIX_H
#define MF_MATRIX_MATRIX_H

#include <string>
#include <iostream>

#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <util/exception.h>
#include <util/io.h>

#include <mpi2/mpi2.h>
#include <mf/types.h>

namespace mf {

/** Compute the default variable name for block (b1,b2) for a distributed matrix with the
 * specified name.
 *
 * @param name name of a distributed matrix
 * @param b1 row block
 * @param b2 column block
 * @return default variable name for storing block (b1,b2)
 */
std::string defaultBlockName(const std::string& name, mf_size_type b1, mf_size_type b2);

/** Computes the block offsets along a dimension (rows/columns) by trying to create blocks
 * of equal size.
 *
 * @param size size of dimension (number of rows/columns)
 * @param blocks number of blocks in the dimension
 * @param[out] blockOffsets starting offsets for each block (this vector will have size() equal
 *                          to blocks)
 */
void computeDefaultBlockOffsets(mf_size_type size, mf_size_type blocks,
		std::vector<mf_size_type>& blockOffsets);

/** Determines locations at which to store the blocks of a blocks1 x blocks2 distributed matrix.
 * Tries to evenly distribute the blocks among the available nodes subject to the condition that
 * each row (partitionByRow = true) or each column (partitionByRow = false) is stored on
 * a single node.
 *
 * @param worldSize number of nodes
 * @param blocks1 number of row blocks
 * @param blocks2 number of column blocks
 * @param partitionByRow whether to partition by rows (true) or by columns (false)
 * @param[out] blockLocations a blocks1 x blocks2 matrix containing the computed block locations
 */
void computeDefaultBlockLocations(unsigned worldSize,
		mf_size_type blocks1, mf_size_type blocks2,
		bool partitionByRow, boost::numeric::ublas::matrix<int>& blockLocations);

/** Computes the default remote variables for a distributed matrix of the specified name
 * and block locations. Uses mf::defaultBlockName.
 *
 * @param name name of a distributed matrix
 * @param blockLocations a blocks1 x blocks2 matrix of block locations
 * @param[out] a blocks1 x blocks2 matrix of remote variables
 */
void computeDefaultBlockVars(
		const std::string& name,
		const boost::numeric::ublas::matrix<int>& blockLocations,
		boost::numeric::ublas::matrix<mpi2::RemoteVar>& blockVars);

/** Computes the number of rows/columns of a given block from the block offsets.
 *
 * @param b block number (0-based)
 * @param size (total number of rows/columns)
 * @param blockOffsets (block offsets, i.e., starting row/column numbers of
 *                     each block)
 * @return number of rows/columns in block b
 */
inline mf_size_type blockSize(
		mf_size_type b,
		mf_size_type size,
		const std::vector<mf_size_type>& blockOffsets) {
	mf_size_type blocks = blockOffsets.size();
	if (b != blocks - 1) {
		return blockOffsets[b+1]-blockOffsets[b];
	} else {
		return size - blockOffsets[b];
	}
}

/** A blocked matrix with blocks distributed across an mpi2 cluster.
 *
 * This class describes properties of the matrix (e.g., size), how the matrix
 * is blocked (e.g., block sizes), and where the data is stored. A distributed
 * matrix is obtained by dividing an size1 x size2 input matrix into blocks1 x blocks2 blocks,
 * and stored each block separately (possibly at different nodes). More
 * specifically, each of the blocks is stored in some local mpi2 environment
 * (see mpi2::env()) of one of the nodes in the cluster. This class only describes
 * data locations but does not contain the actual data.
 *
 * @tparam M matrix type of each block
 */
template <typename M>
class DistributedMatrix {
public:
	/** matrix type of each block */
	typedef M Matrix;

	// -- construction ----------------------------------------------------------------------------

	/** Serialization constructor. Do not use directly. */
	DistributedMatrix(mpi2::SerializationConstructor _)	{
	}

	/** Complete constructor.
	 *
	 * @param name A unique matrix name. The variable names used for storing the blocks are
	 *             constructed from this name.
	 * @param size1 Number of rows in the matrix
	 * @param size2 Number of columns in the matrix
	 * @param blockOffsets1 Vector of the block row offsets (number of first row in each block).
	 *                      The distributed matrix has blockOffsets1.size() row blocks.
	 * @param blockOffsets2 Vector of the block column offsets (number of first column in each block).
	 *                      The distributed matrix has blockOffsets2.size() column blocks.
	 * @param blockLocations A blocks1 x blocks2 matrix containing the rank at which each block
	 *                       is stored (or will be stored)
	 */
	DistributedMatrix(std::string name,
			mf_size_type size1, mf_size_type size2,
			const std::vector<mf_size_type>& blockOffsets1, const std::vector<mf_size_type>& blockOffsets2,
			const boost::numeric::ublas::matrix<int>& blockLocations)
	: name_(name),
	  size1_(size1), size2_(size2),
	  blocks1_(blockOffsets1.size()), blocks2_(blockOffsets2.size()),
	  blockOffsets1_(blockOffsets1), blockOffsets2_(blockOffsets2),
	  blocks_(blocks1_, blocks2_, mpi2::RemoteVar(mpi2::UNINITIALIZED)) {
		computeDefaultBlockVars(name, blockLocations, blocks_);
	}

	/** Automatic choice of block offsets */
	DistributedMatrix(std::string name,
			mf_size_type size1, mf_size_type size2,
			const boost::numeric::ublas::matrix<int>& blockLocations)
	: name_(name), size1_(size1), size2_(size2),
	  blocks1_(blockLocations.size1()), blocks2_(blockLocations.size2()),
	  blocks_(blocks1_, blocks2_, mpi2::RemoteVar(mpi2::UNINITIALIZED)) {
		// determine block offsets
		computeDefaultBlockOffsets(size1, blocks1_, blockOffsets1_);
		computeDefaultBlockOffsets(size2, blocks2_, blockOffsets2_);
		computeDefaultBlockVars(name, blockLocations, blocks_);
	}

	/** Automatic choice of block locations */
	DistributedMatrix(std::string name,
			mf_size_type size1, mf_size_type size2,
			const std::vector<mf_size_type>& blockOffsets1, const std::vector<mf_size_type>& blockOffsets2,
			bool partitionByRow = true)
	: name_(name), size1_(size1), size2_(size2),
	  blocks1_(blockOffsets1.size()), blocks2_(blockOffsets2.size()),
	  blockOffsets1_(blockOffsets1), blockOffsets2_(blockOffsets2),
	  blocks_(blocks1_, blocks2_, mpi2::RemoteVar(mpi2::UNINITIALIZED)) {
		// determine locations
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		boost::mpi::communicator& world = tm.world();
		int worldSize = world.size();
		boost::numeric::ublas::matrix<int> blockLocations;
		computeDefaultBlockLocations(world.size(), blocks1_, blocks2_,
				partitionByRow, blockLocations);
		computeDefaultBlockVars(name, blockLocations, blocks_);
	}

	/** Automatic choice of block offsets and block locations */
	DistributedMatrix(std::string name,
			mf_size_type size1, mf_size_type size2,
			mf_size_type blocks1, mf_size_type blocks2,
			bool partitionByRow = true)
	: name_(name), size1_(size1), size2_(size2),
	  blocks1_(blocks1), blocks2_(blocks2),
	  blocks_(blocks1_, blocks2_, mpi2::RemoteVar(mpi2::UNINITIALIZED)) {
		// determine block offsets
		computeDefaultBlockOffsets(size1, blocks1_, blockOffsets1_);
		computeDefaultBlockOffsets(size2, blocks2_, blockOffsets2_);

		// determine locations
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		boost::mpi::communicator& world = tm.world();
		int worldSize = world.size();
		boost::numeric::ublas::matrix<int> blockLocations;
		computeDefaultBlockLocations(worldSize, blocks1_, blocks2_,
				partitionByRow, blockLocations);
		computeDefaultBlockVars(name, blockLocations, blocks_);
	}

	/** Creates empty blocks for this matrix on all nodes. Will throw an exception if any
	 * of the block exist already. */
	void create();

	/** Removes all blocks of the matrix from memory at all nodes. The blocks must have been created
	 * before.
	 */
	void erase() {
		// this could be parallelized, but it's currently not performance critical...
		for (mf_size_type b1 = 0; b1 < blocks1(); b1++) {
			for (mf_size_type b2 = 0; b2 < blocks2(); b2++) {
				mpi2::RemoteVar rv = block(b1,b2);
				rv.erase<M>();
			}
		}
	}

	// -- getters --------------------------------------------------------------------------------

	/** Returns the name of this matrix */
	const std::string& name() const {
		return name_;
	}

	/** Returns the number of rows of this matrix */
	mf_size_type size1() const {
		return size1_;
	}

	/** Returns the number of columns of this matrix */
	mf_size_type size2() const {
		return size2_;
	}

	/** Returns the number of row blocks of this matrix */
	mf_size_type blocks1() const {
		return blocks1_;
	}

	/** Returns the number of column blocks of this matrix */
	mf_size_type blocks2() const {
		return blocks2_;
	}

	/** Returns the number of the first row in blocks (b1,*) */
	mf_size_type blockOffset1(mf_size_type b1) const {
		return blockOffsets1_[b1];
	}

	/** Returns the number of the first column in blocks (*,b2) */
	mf_size_type blockOffset2(mf_size_type b2) const {
		return blockOffsets2_[b2];
	}

	/** Returns the number of rows in blocks (b1,*)  */
	mf_size_type blockSize1(mf_size_type b1) const {
		return blockSize(b1, size1_, blockOffsets1_);
	}

	/** Returns the number of columns in blocks (*,b2) */
	mf_size_type blockSize2(mf_size_type b2) const {
		return blockSize(b2, size2_, blockOffsets2_);
	}

	/** Returns a vector of the block row offsets (number of first row in each block) */
	const std::vector<mf_size_type>& blockOffsets1() const {
		return blockOffsets1_;
	}

	/** Returns a vector of the block column offsets (number of first column in each block) */
	const std::vector<mf_size_type>& blockOffsets2() const {
		return blockOffsets2_;
	}

	/** Returns a remote variable referencing block (b1,b2). */
	mpi2::RemoteVar block(mf_size_type b1, mf_size_type b2) const {
		return blocks_(b1,b2);
	}

	/** Returns a blocks1_ x blocks2_ matrix containing the remote variable pointing to each
	 * blocks data. */
	const boost::numeric::ublas::matrix<mpi2::RemoteVar>& blocks() const {
		return blocks_;
	}

private:
	/** A unique matrix name. The variable names used for storing the blocks are constructed
	 * from this name (via mf::blockVarName()). */
	 std::string name_;

	/** Number of rows in the matrix */
	mf_size_type size1_;

	 /** Number of columns in the matrix */
	mf_size_type size2_;

	/** Number of row blocks */
	mf_size_type blocks1_;

	/** Number of column blocks */
	mf_size_type blocks2_;

	/** Vector of the block row offsets (number of first row in each block). */
	std::vector<mf_size_type> blockOffsets1_;

	/** Vector of the block column offsets (number of first column in each block). */
	std::vector<mf_size_type> blockOffsets2_;

	/** A blocks1_ x blocks2_ matrix containing the remote variable pointing to each
	 * blocks data. */
	boost::numeric::ublas::matrix<mpi2::RemoteVar> blocks_;

	// serialization code
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & name_;
		ar & size1_;
		ar & size2_;
		ar & blocks1_;
		ar & blocks2_;
		ar & blockOffsets1_;
		ar & blockOffsets2_;
		ar & blocks_;
	}
};

} // namespace mf

MPI2_SERIALIZATION_CONSTRUCTOR1(mf::DistributedMatrix);

namespace mf {

/** A distributed sparse matrix (row-major storage) */
typedef DistributedMatrix<SparseMatrix> DistributedSparseMatrix;

/** A distributed sparse matrix (column-major storage) */
typedef DistributedMatrix<SparseMatrixCM> DistributedSparseMatrixCM;

/** A distributed dense matrix (row-major storage) */
typedef DistributedMatrix<DenseMatrix> DistributedDenseMatrix;

/** A distributed dense matrix (column-major storage) */
typedef DistributedMatrix<DenseMatrixCM> DistributedDenseMatrixCM;
}

namespace mf {

/** Formatted output of a distributed matrix (just the description, no data) */
template<typename CharT, typename Traits, typename M>
std::basic_ostream<CharT, Traits>& operator<<(
		std::basic_ostream<CharT, Traits>& out, const DistributedMatrix<M>& m) {
	using rg::operator<<;
	out << "Name:        " << m.name() << std::endl;
	std::string typeName;
	try {
		typeName = mpi2::TypeTraits<M>::name();
	} catch (...) {
		typeName = typeid(M).name();
	}
	out << "Type:        " << typeName << std::endl;
	out << "Size:        " << m.size1() << "x" << m.size2() << std::endl;
	out << "Blocks:      " << m.blocks1() << "x" << m.blocks2() << std::endl;
	out << "Block size1: "; for (mf_size_type b1=0; b1<m.blocks1(); b1++) out << m.blockSize1(b1) << " "; out << std::endl;
	out << "Block size2: "; for (mf_size_type b2=0; b2<m.blocks2(); b2++) out << m.blockSize2(b2) << " "; out << std::endl;
	out << "Offsets1:    " << m.blockOffsets1() << std::endl;
	out << "Offsets2:    " << m.blockOffsets2() << std::endl;
	out << "Blocks:      " << m.blocks();

	return out;
};

} // namespace mf

// precompiled templates (not used)
/*
namespace boost { namespace numeric { namespace ublas {
extern template class boost::numeric::ublas::matrix<int, boost::numeric::ublas::row_major>;
extern template class boost::numeric::ublas::matrix<unsigned, boost::numeric::ublas::row_major>;
extern template class coordinate_matrix<double, boost::numeric::ublas::row_major>;
extern template class coordinate_matrix<double, boost::numeric::ublas::column_major>;
extern template class boost::numeric::ublas::matrix<double, boost::numeric::ublas::row_major>;
extern template class boost::numeric::ublas::matrix<double, boost::numeric::ublas::column_major>;
}}}
namespace mf {
extern template class DistributedMatrix<SparseMatrix>;
extern template class DistributedMatrix<SparseMatrixCM>;
extern template class DistributedMatrix<DenseMatrix>;
extern template class DistributedMatrix<DenseMatrixCM>;
}
*/

#include <mf/matrix/distributed_matrix_impl.h>

#endif
