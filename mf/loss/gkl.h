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
#ifndef MF_LOSS_GKL_H
#define MF_LOSS_GKL_H

#include <mf/loss/gkl-data.h>
#include <mf/loss/gkl-model.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

inline double gkl(const SparseMatrix& v,const DenseMatrix& w, const DenseMatrixCM& h) {
	double result = gklData(v, w, h) + gklModel(w, h);
	return std::max(0., result); // avoid rounding errors
}

// -- distributed ---------------------------------------------------------------------------------

inline SparseMatrix::value_type gkl(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		int tasksPerRank=1) {
	double result = gklData(v, w, h, tasksPerRank) + gklModel(w, h, tasksPerRank);
	return std::max(0., result); // avoid rounding errors
}

inline double gkl(const DistributedSparseMatrix& v, const DistributedDenseMatrix& w,
		const DistributedDenseMatrixCM& h, const std::string& hUnblockedName, int tasksPerRank=1) {
	double result = gklData(v, w, hUnblockedName, tasksPerRank) + gklModel(w, h, tasksPerRank);
	return std::max(0., result); // avoid rounding errors
}


// -- Loss ----------------------------------------------------------------------------------------

struct GklLoss : public LossConcept {
	GklLoss() { };
	GklLoss(mpi2::SerializationConstructor _) { };


	double operator()(const FactorizationData<>& data) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of GklLoss not yet implemented, using sequential computation.");
		}
		return gkl(data.v, data.w, data.h);
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return gkl(data.dv, data.dw, data.dh, data.tasksPerRank);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

} // namespace mf

MPI2_TYPE_TRAITS(mf::GklLoss);

#endif
