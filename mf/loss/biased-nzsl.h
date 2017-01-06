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
#ifndef MF_LOSS_BIASED_NZSL_H
#define MF_LOSS_BIASED_NZSL_H

#include <mf/loss/loss.h>
#include <mf/matrix/coordinate.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/op/sum.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

template<typename W, typename H>
inline double biasedNzsl(const SparseMatrix& v,
		const W& w, const H& h) {
	typedef SparseMatrix M;
	typename M::value_type result = 0;

	const typename M::index_array_type& index1 = rowIndexData(v);
	const typename M::index_array_type& index2 = columnIndexData(v);
	const typename M::value_array_type& values = v.value_data();
	mf_size_type rank = w.size2();
	for (mf_size_type i=0; i<v.nnz(); i++) {
		typename M::value_type ip = 0;
		typename M::size_type i1 = index1[i];
		typename M::size_type i2 = index2[i];
		typename M::value_type value = values[i];
		for (mf_size_type r=1; r<rank; r++) {
			ip += w(i1,r) * h(r,i2);
		}
		typename M::value_type diff = value - ip - w(i1,0) - h(0,i2);
		result += diff*diff;
	}

	return result;
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	template<typename M1, typename M2, typename M3>
	struct BiasedNzslTask {
		static const std::string id() { return std::string("__mf/loss/_BiasedNzslTask")
				+ mpi2::TypeTraits<M1>::name() + "_" + mpi2::TypeTraits<M2>::name() + "_" + mpi2::TypeTraits<M3>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			runFunctionPerAssignedBlock3<M1,M2,M3, typename M1::value_type>(ch, &f);
		}
		static inline typename M1::value_type f(const M1& v, const M2& w, const M3& h) {
			return biasedNzsl(v, w, h);
		}
	};
}

inline void biasedNzsl(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		boost::numeric::ublas::matrix<SparseMatrix::value_type>& result, int tasksPerRank=1) {
	runTaskOnBlocks3(v, w, h, result,
			detail::BiasedNzslTask<SparseMatrix,DenseMatrix,DenseMatrixCM>::id(),
			tasksPerRank);
}

inline SparseMatrix::value_type biasedNzsl(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		int tasksPerRank=1) {
	boost::numeric::ublas::matrix<SparseMatrix::value_type> result(v.blocks1(), v.blocks2());
	biasedNzsl(v, w, h, result, tasksPerRank);
	return sum(result);
}

// -- Loss ----------------------------------------------------------------------------------------

struct BiasedNzslLoss : public LossConcept, DistributedLossConcept {
	BiasedNzslLoss() {};
	BiasedNzslLoss(mpi2::SerializationConstructor _) { };

	double operator()(const FactorizationData<>& data) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of BiasedNzslLoss not yet implemented, using sequential computation.");
		}
		return biasedNzsl(data.v, data.w, data.h);
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return biasedNzsl(data.dv, data.dw, data.dh, data.tasksPerRank);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::BiasedNzslLoss);


#endif
