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
#ifndef MF_MATRIX_OP_CROSSPROD_H
#define MF_MATRIX_OP_CROSSPROD_H

#include <mf/id.h>
#include <mf/matrix/distribute.h>

namespace mf {

// -- sequential methods ----------------------------------------------------------------

// these methods are carefully optimized; do not touch

/** Returns hh' */
inline DenseMatrixCM tcrossprod(const DenseMatrixCM& h) {
	// initialize
	unsigned r = h.size1();
	unsigned n = h.size2();
	DenseMatrixCM result(r,r);
	result.clear();
	const DenseMatrixCM::array_type& hValues = h.data();
	DenseMatrixCM::array_type& resultValues = result.data();

	// compute upper half of result
	mf_size_type pj = 0;
	for (mf_size_type j=0; j<n; j++) {
		mf_size_type pi = 0;
		for (mf_size_type i=0; i<r; i++) {
			for (mf_size_type k=i; k<r; k++) {
				resultValues[pi+k] += hValues[i + pj] * hValues[k + pj];
			}
			pi += r;
		}
		pj += r;
	}

	// copy to lower half
	mf_size_type pi = r;
	for (mf_size_type i=1; i<r; i++) {
		mf_size_type pk = 0;
		for (mf_size_type k=0; k<i; k++) {
			resultValues[pi+k] = resultValues[pk+i];
			pk += r;
		}
		pi += r;
	}

	return result;
}

/** Returns w'w */
inline DenseMatrix crossprod(const DenseMatrix& w) {
	// initialize
	unsigned m = w.size1();
	unsigned r = w.size2();
	DenseMatrix result(r,r);
	result.clear();
	const DenseMatrix::array_type& wValues = w.data();
	DenseMatrix::array_type& resultValues = result.data();

	// compute upper half of result
	mf_size_type pi = 0;
	for (mf_size_type i=0; i<m; i++) {
		mf_size_type pj = 0;
		for (mf_size_type j=0; j<r; j++) {
			for (mf_size_type k=j; k<r; k++) {
				resultValues[k+pj] += wValues[pi + j] * wValues[pi + k];
			}
			pj += r;
		}
		pi += r;
	}

	// copy to lower half
	mf_size_type pj = r;
	for (unsigned j=1; j<r; j++) {
		mf_size_type pk = 0;
		for (unsigned k=0; k<j; k++) {
			resultValues[k+pj] = resultValues[j+pk];
			pk += r;
		}
		pj += r;
	}
	return result;
}

/** Compute the sum of matrices stored in the given matrix*/
template<typename M>
M matrixSum(boost::numeric::ublas::matrix<M>& m){
	unsigned r = m(0,0).size1();
	M result(r,r);
	for (unsigned i=0; i<r; i++) {
		for (unsigned j=0; j<r; j++) {
			result(i,j) = 0;
		}
	}
	for (unsigned i=0; i<m.size1(); i++) {
		for (unsigned j=0; j<m.size2(); j++) {
			result += m(i,j);
		}
	}
	return result;
}


// -- distributed methods ---------------------------------------------------------------

namespace detail {
	template<typename M>
	M TCrossprodTaskF(M& m) {
		return tcrossprod(m);
	}

	template<typename M>
	struct TCrossprodTask : public PerBlockTaskReturn<M, M, TCrossprodTaskF<M>, ID_TCROSSPROD> {
		typedef PerBlockTaskReturn<M, M, TCrossprodTaskF<M>, ID_TCROSSPROD> Task;
	};

	template<typename M>
	M CrossprodTaskF(M& m) {
		return crossprod(m);
	}
	template<typename M>
	struct CrossprodTask : public PerBlockTaskReturn<M, M, CrossprodTaskF<M>, ID_CROSSPROD> {
		typedef PerBlockTaskReturn<M, M, CrossprodTaskF<M>, ID_CROSSPROD> Task;
	};
}

/** Returns mm' */
template<typename M>
inline void tcrossprod(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<M>& result, int tasksPerRank=1) {
	runTaskOnBlocks< detail::TCrossprodTask<M> >(m, result, tasksPerRank);
}

/** Returns mm' */
template<typename M>
inline M tcrossprod(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	boost::numeric::ublas::matrix<M> mmts(m.blocks1(), m.blocks2());
	tcrossprod(m, mmts, tasksPerRank);
	return matrixSum(mmts);
}


/** Returns m'm */
template<typename M>
inline void crossprod(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<M>& result, int tasksPerRank=1) {
	runTaskOnBlocks< detail::CrossprodTask<M> >(m, result, tasksPerRank);
}

/** Returns m'm */
template<typename M>
inline M crossprod(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	boost::numeric::ublas::matrix<M> mmts(m.blocks1(), m.blocks2());
	crossprod(m, mmts, tasksPerRank);
	return matrixSum(mmts);
}

}

#endif
