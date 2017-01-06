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
#ifndef MF_MATRIX_OP_NNZ_H
#define MF_MATRIX_OP_NNZ_H

#include <mf/id.h>
#include <mf/matrix/coordinate.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/op/sum.h>

#include <mf/matrix/coordinate.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

inline mf_size_type nnz(const SparseMatrix& m) { return m.nnz(); }

inline mf_size_type nnz(const SparseMatrixCM& m) { return m.nnz(); }

/** Counts the number of nonzero entries in each row / column of the matrix */
inline void nnz12(const SparseMatrix& m, std::vector<mf_size_type>& nnz1,
		std::vector<mf_size_type>& nnz2,
		mf_size_type& nnz12max) {
	// start with zeroes
	nnz1.resize(m.size1());
	nnz2.resize(m.size2());
	std::fill(nnz1.begin(), nnz1.end(), 0);
	std::fill(nnz2.begin(), nnz2.end(), 0);

	const SparseMatrix::index_array_type& index1 = rowIndexData(m);
	const SparseMatrix::index_array_type& index2 = columnIndexData(m);
	for (mf_size_type p=0; p<m.nnz(); p++) {
		nnz1[index1[p]]++;
		nnz2[index2[p]]++;
	}
	// max==0 if called from Sgd
	if (nnz12max==0){ // make sure that I don't overwrite something in max
		for (mf_size_type i=0;i<nnz1.size();i++){
			if (nnz12max<nnz1[i]) nnz12max=nnz1[i];
		}
		for (mf_size_type i=0;i<nnz2.size();i++){
			if (nnz12max<nnz2[i]) nnz12max=nnz2[i];
		}
	}
}

/** Counts the number of nonzero entries in each row / column of the matrix */
inline void nnz12Incremental(const SparseMatrix& m,
		std::vector<mf_size_type>& nnz1, mf_size_type nnz1offset,
		std::vector<mf_size_type>& nnz2, mf_size_type nnz2offset,
		mf_size_type& nnz12max) {
	const SparseMatrix::index_array_type& index1 = rowIndexData(m);
	const SparseMatrix::index_array_type& index2 = columnIndexData(m);
	for (mf_size_type p=0; p<m.nnz(); p++) {
		nnz1[index1[p] + nnz1offset]++;
		nnz2[index2[p] + nnz2offset]++;
	}
	// max==0 if called from Sgd
	if (nnz12max==0){ // make sure that I don't overwrite something in max
		for (mf_size_type i=nnz1offset;i<nnz1offset+m.size1();i++){
			if (nnz12max<nnz1[i]) nnz12max=nnz1[i];
		}
		for (mf_size_type j=nnz2offset;j<nnz2offset+m.size2();j++){
			if (nnz12max<nnz2[j]) nnz12max=nnz2[j];
		}
	}
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	template<typename M>
	mf_size_type NnzTaskF(M& m) {
		return nnz(m);
	}

	template<typename M>
	struct NnzTask : public PerBlockTaskReturn<M, mf_size_type, NnzTaskF<M>, ID_NNZ> {
		typedef PerBlockTaskReturn<M, mf_size_type, NnzTaskF<M>, ID_NNZ> Task;
	};

	template<typename M>
	inline std::pair< std::vector<mf_size_type>, std::vector<mf_size_type> > Nnz12TaskF(M& m) {
		std::pair< std::vector<mf_size_type>, std::vector<mf_size_type> > result;
		mf_size_type max;
		nnz12(m, result.first, result.second, max);
		return result;
	}

	template<typename M>
	struct Nnz12Task : public PerBlockTaskReturn<M, std::pair< std::vector<mf_size_type>, std::vector<mf_size_type> >, Nnz12TaskF<M>, ID_NNZ12> {
		typedef PerBlockTaskReturn<M, std::pair< std::vector<mf_size_type>, std::vector<mf_size_type> >, Nnz12TaskF<M>, ID_NNZ12> Task;
	};
}

template<typename M>
inline void nnz(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<mf_size_type>& nnzs, int tasksPerRank=1) {
	runTaskOnBlocks< detail::NnzTask<M> >(m, nnzs, tasksPerRank);
}

template<typename M>
inline mf_size_type nnz(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	boost::numeric::ublas::matrix<mf_size_type> nnzs(m.blocks1(), m.blocks2());
	nnz(m, nnzs, tasksPerRank);
	return sum(nnzs);
}

/** Counts the number of nonzero entries in each row / column of a distributed matrix.
 * Current implementation is not efficient (neither memory nor CPU, but that's OK for now). */
inline void nnz12(const DistributedSparseMatrix& m,
		std::vector<mf_size_type>& nnz1,
		std::vector<mf_size_type>& nnz2,
		mf_size_type& nnz12max,
		unsigned tasksPerRank = 1) {
	boost::numeric::ublas::matrix<
		detail::Nnz12Task<SparseMatrix>::Return
		> localNnzPairs(m.blocks1(), m.blocks2());
	runTaskOnBlocks< detail::Nnz12Task<SparseMatrix> >(m, localNnzPairs, tasksPerRank, false);

	// clear result
	nnz1.clear();
	nnz1.resize(m.size1(), 0);
	nnz2.clear();
	nnz2.resize(m.size2(), 0);

	// fill result
	for (mf_size_type b1 = 0; b1<m.blocks1(); b1++) {
		mf_size_type b1offset = m.blockOffsets1()[b1];
		for (mf_size_type b2 = 0; b2<m.blocks2(); b2++) {
			const detail::Nnz12Task<SparseMatrix>::Return& localPair = localNnzPairs(b1,b2);
			mf_size_type b2offset = m.blockOffsets2()[b2];

			for (mf_size_type i = 0; i<localPair.first.size(); i++) {
				nnz1[i + b1offset] += localPair.first[i];
			}
			for (mf_size_type j = 0; j<localPair.second.size(); j++) {
				nnz2[j + b2offset] += localPair.second[j];
			}
		}
	}
	// find the maximum entry of both nnz1 and nnz2
	nnz12max=0;
	for (mf_size_type i=0;i<nnz1.size();i++){
		if (nnz12max<nnz1[i]) nnz12max=nnz1[i];
	}
	for (mf_size_type i=0;i<nnz2.size();i++){
			if (nnz12max<nnz2[i]) nnz12max=nnz2[i];
	}
}

} // namespace mf

#endif
