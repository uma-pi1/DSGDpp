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
#ifndef MF_LOSS_KL_H
#define MF_LOSS_KL_H

#include <mf/matrix/coordinate.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>


namespace mf {

template<class T, class L, std::size_t IB, class IA, class TA, typename W, typename H>
inline T kl(const boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>& v,
		const W& w, const H& h) {

	typedef typename boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA> M;
	typename M::value_type result = 0;

	const typename M::index_array_type& index1 = rowIndexData(v);
	const typename M::index_array_type& index2 = columnIndexData(v);
	const typename M::value_array_type& values = v.value_data();
	mf_size_type rank = w.size2();
	for (mf_size_type i=0; i<v.nnz(); i++) {
		typename M::value_type ip = 0;
		mf_size_type i1 = index1[i];
		mf_size_type i2 = index2[i];
		typename M::value_type value = values[i];
		for (mf_size_type r=0; r<rank; r++) {
			ip += w(i1,r) * h(r,i2);
		}
		result += value * log(value / fabs(ip));
	}
	return result;
}

 namespace detail {

template<typename M1, typename M2, typename M3>
struct KlTask {
	static const std::string id() { return std::string("__mf/matrix/kl_")
			+ mpi2::TypeTraits<M1>::name() + "_" + mpi2::TypeTraits<M2>::name() + "_" + mpi2::TypeTraits<M3>::name(); }
	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		runFunctionPerAssignedBlock3<M1,M2,M3, typename M1::value_type>(ch, &f);
	}
	static inline typename M1::value_type f(const M1& v, const M2& w, const M3& h) {
		return kl(v, w, h);
	}
};

} 

inline void kl(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		boost::numeric::ublas::matrix<SparseMatrix::value_type>& result, int tasksPerRank=1) {
	runTaskOnBlocks3(v, w, h, result,
			detail::KlTask<SparseMatrix,DenseMatrix,DenseMatrixCM>::id(),
			tasksPerRank);
}

inline SparseMatrix::value_type kl(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		int tasksPerRank=1) {
	boost::numeric::ublas::matrix<SparseMatrix::value_type> result(v.blocks1(), v.blocks2());
	kl(v, w, h, result, tasksPerRank);
	return sum(result);
}

} // namespace mf



#endif
