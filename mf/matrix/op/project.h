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
/** \file
 * Methods for extracting submatrices.
 */

#ifndef MF_MATRIX_OP_PROJECT_H
#define MF_MATRIX_OP_PROJECT_H

// TODO: make (some of) these methods distributed

#include <boost/foreach.hpp>

#include <util/exception.h>
#include <util/io.h>
#include <util/random.h>

#include <mf/matrix/distribute.h>

namespace mf {

/**A submatrix of some sparse matrix. Holds data as well as the indexes of the rows/columns
 * from the original matrix.
 */
struct ProjectedSparseMatrix {
	SparseMatrix data;
	std::vector<mf_size_type> map1;
	std::vector<mf_size_type> map2;
	mf_size_type size1; /**< size1 of original matrix */
	mf_size_type size2; /**< size2 of original matrix */

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & data;
		ar & map1;
		ar & map2;
		ar & size1;
		ar & size2;
	}
};

}

MPI2_TYPE_TRAITS(mf::ProjectedSparseMatrix);

namespace mf {

/** Extract a submatrix from a given matrix.
 *
 * @param V input matrix
 * @param[out] result matrix
 * @param indexes1 which rows to select
 * @param indexes2 which columns to select
 */
void projectSubmatrix(const SparseMatrix& V, SparseMatrix& result,
		const std::vector<mf_size_type>& indexes1,
		const std::vector<mf_size_type>& indexes2);

/** Extract a submatrix from a given matrix by selecting a random set of rows and columns.
 *
 * @param random a pseudo random number generator
 * @param V input matrix
 * @param[out] sample the output matrix
 * @param n1 how many rows to select
 * @param n2 how many columns to select
 */
void projectRandomSubmatrix(rg::Random32& random, const SparseMatrix& V,
		ProjectedSparseMatrix& sample,
		mf_size_type n1, mf_size_type n2=0);

/** Selects the submatrix of nonempty rows and columns from a given matrix.
 *
 * @param m input matrix
 * @param[out] result output matrix
 */
//void projectNonempty(const SparseMatrix& m, ProjectedSparseMatrix& result);

/** Removes empty rows and columns from the given matrix.
 *
 * @param[in,out] m input matrix, will be overwritten
 */
//void projectNonempty(SparseMatrix& m);

/** Selects the submatrix of nonempty rows and columns from a given submatrix.
 *
 * @param m input submatrix, already projected from some original matrix
 * @param[out] result output submatrix, row/column indexes refer to the original matrix
 */
//void projectNonempty(const ProjectedSparseMatrix& m, ProjectedSparseMatrix& result);

/** Selects the submatrix of nonempty rows and columns from a given submatrix.
 *
 * @param m[in,out] input/output submatrix, row/column indexes refer to the original matrix from which
 * the input matrix was created
 */
//void projectNonempty(ProjectedSparseMatrix& m);

/** Removes rows and columns whose number of nonzero entries is less than threshold k.
 *
 * @param[in,out] m input matrix, will be overwritten
 * @param[in] t threshold of nonzero entries in each row and each column
 */
void projectFrequent(SparseMatrix& m, mf_size_type t);

/** Removes rows and columns whose number of nonzero entries is less than threshold k.
 *
 * @param[in,out] m input matrix, will be overwritten
 * @param[in] t threshold of nonzero entries in each row and each column
 */
void projectFrequent(ProjectedSparseMatrix& m, mf_size_type t);

/** Selects the submatrix whose number of nonzero entries in each row and each column is above threshold k from a given submatrix.
 *
 * @param m input matrix
 * @param[out] result output matrix
 * @param[in] k threshold of nonzero entries in each row and each column
 */
void projectFrequent(const SparseMatrix& m, ProjectedSparseMatrix& result, mf_size_type t);

/** Selects the submatrix whose number of nonzero entries in each row and each column is above threshold k from a given submatrix.
 *
 * @param m input submatrix, already projected from some original matrix
 * @param[out] result output submatrix, row/column indexes refer to the original matrix
 * @param[in] k threshold of nonzero entries in each row and each column
 */
void projectFrequent(const ProjectedSparseMatrix& m, ProjectedSparseMatrix& result, mf_size_type t);

/** Selects a submatrix consisting of the given set of rows (and all columns).
 *
 * @param m input matrix
 * @param[out] result output matrix
 * @param indexes1 which rows to select
 * @tparam M a dense matrix type
 */
template<typename M>
void project1(const M& m, M& result, std::vector<mf_size_type> indexes1);

/** Selects a submatrix consisting of the given set of columns (and all rows).
 *
 * @param m input matrix
 * @param[out] result output matrix
 * @param indexes2 which columns to select
 * @tparam M a dense matrix type
 */
template<typename M>
void project2(const M& m, M& result, std::vector<mf_size_type> indexes2);

/** Splits a vector of row/column indexes according to a row/column blocking. Assumes
 * that input index list is sorted.
 *
 * @param indexes input vector of row or column indexes
 * @param blockOffsets vector of block offsets
 * @param[out] blockIndexes output vector containing the indexes in each block (relative to the
 * blockoffset)
 */
void splitIndexes(const std::vector<mf_size_type>& indexes,
		const std::vector<mf_size_type>& blockOffsets,
		std::vector<std::vector<mf_size_type> >& blockIndexes);


/** Selects a submatrix consisting of the given set of rows (and all columns).
 *
 * @param m input matrix
 * @param[out] result output matrix
 * @param indexes1 which rows to select
 * @param taskPerRank how many tasks to launch at each rank
 * @tparam M a dense matrix type
 */
template<typename M>
void project1(const DistributedMatrix<M>& m, M& result,
		std::vector<mf_size_type> indexes1, int tasksPerRank=1);

/** Selects a submatrix consisting of the given set of columns (and all rows).
 *
 * @param m input matrix
 * @param[out] result output matrix
 * @param indexes2 which columns to select
 * @param taskPerRank how many tasks to launch at each rank
 * @tparam M a dense matrix type
 */
template<typename M>
void project2(const DistributedMatrix<M>& m, M& result,
		std::vector<mf_size_type> indexes2, int tasksPerRank=1);

/** Extracts the consecutive submatrix given by the row range start1 through stop1 (exclusive)
 * and column range start2 through stop2 (exclusive).
 *
 * @param source input matrix
 * @param[out] target output matrix
 * @param start1 first row (inclusive)
 * @param stop1 last row (exclusive)
 * @param start2 first column (inclusive)
 * @param stop2 last column (exclusive)
 * @tparam Min type of input matrix
 * @tparam Mout type of output matrix
 */
template<typename Min, typename Mout>
void projectSubrange(const Min& source, Mout& target,
		mf_size_type start1, mf_size_type stop1, mf_size_type start2, mf_size_type stop2);

}

#include <mf/matrix/op/project_impl.h>

#endif
