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
#ifndef MF_LOSS_GKL_DATA_H
#define MF_LOSS_GKL_DATA_H

#include <math.h>

#include <mf/matrix/coordinate.h>
#include <mf/id.h>
#include <mf/loss/loss.h>
#include <mf/ap/aptask.h>
#include <mf/matrix/distribute.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

/** Compute the data part of the gkl loss only */
inline double gklData(const SparseMatrix& v,const DenseMatrix& w, const DenseMatrixCM& h) {
	double result = 0;

	// compute data part
	const SparseMatrix::index_array_type& index1 = rowIndexData(v);
	const SparseMatrix::index_array_type& index2 = columnIndexData(v);
	const SparseMatrix::value_array_type& values = v.value_data();
	mf_size_type rank = w.size2();
	for (mf_size_type i=0; i<v.nnz(); i++) {
		double ip = 0;
		mf_size_type i1 = index1[i];
		mf_size_type i2 = index2[i];
		double value = values[i];
		if (values[i] == 0) continue; // no loss occurs at zeros (shouldn't be in a sparse matrix anyway, but just in case...)
		for (mf_size_type r=0; r<rank; r++) {
			ip += w(i1,r) * h(r,i2);
		}
		ip = fabs(ip); // just to be safe
		if (ip == 0) {
			result = INFINITY;
		} else {
			result += value * log(value / ip) - value;
		}
	}

	return result;
	//return std::max(0., result); // avoid rounding errors
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	typedef ApTaskW<mf::gklData, ID_GKL_DATA_AP> GklApTaskW;
}

inline double gklData(const DistributedSparseMatrix& v, const DistributedDenseMatrix& w,
		const std::string& hUnblockedName, int tasksPerRank=1) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<SparseMatrix,double,detail::GklApTaskW::Arg>(
						v, result,
						boost::bind(detail::GklApTaskW::constructArg, _1, _2, _3, boost::cref(w), boost::cref(hUnblockedName)),
						detail::GklApTaskW::id(), tasksPerRank);
	return std::accumulate(result.data().begin(), result.data().end(), 0.);
}

namespace detail {
	template<typename M1, typename M2, typename M3>
	struct GklDataTask {
		static const std::string id() { return std::string("__mf/loss/GklDataTask_")
				+ mpi2::TypeTraits<M1>::name() + "_" + mpi2::TypeTraits<M2>::name() + "_" + mpi2::TypeTraits<M3>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			runFunctionPerAssignedBlock3<M1,M2,M3, typename M1::value_type>(ch, &f);
		}
		static inline typename M1::value_type f(const M1& v, const M2& w, const M3& h) {
			return gklData(v, w, h);
		}
	};
}

inline void gklData(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		boost::numeric::ublas::matrix<SparseMatrix::value_type>& result, int tasksPerRank=1) {
	runTaskOnBlocks3(v, w, h, result,
			detail::GklDataTask<SparseMatrix,DenseMatrix,DenseMatrixCM>::id(),
			tasksPerRank);
}

inline SparseMatrix::value_type gklData(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		int tasksPerRank=1) {
	boost::numeric::ublas::matrix<SparseMatrix::value_type> result(v.blocks1(), v.blocks2());
	gklData(v, w, h, result, tasksPerRank);
	return sum(result);
}

// -- Loss ----------------------------------------------------------------------------------------

struct GklDataLoss : public LossConcept {
	GklDataLoss() {};
	GklDataLoss(mpi2::SerializationConstructor _) { };

	double operator()(const FactorizationData<>& data) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of GklDataLoss not yet implemented, using sequential computation.");
		}
		return gklData(data.v, data.w, data.h);
	}
	double operator()(const DsgdFactorizationData<>& data) {
		return gklData(data.dv, data.dw, data.dh, data.tasksPerRank);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::GklDataLoss);



#endif
