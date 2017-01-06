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
#ifndef MF_LOSS_SL_H
#define MF_LOSS_SL_H

#include <mf/loss/sl-data.h>
#include <mf/loss/sl-model.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

inline double sl(const SparseMatrix& v,const DenseMatrix& w, const DenseMatrixCM& h) {
	return slData(v, w, h) + slModel(w, h);
}

// -- distributed ---------------------------------------------------------------------------------

inline double sl(const DistributedSparseMatrix& v, const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h, int tasksPerRank = 1) {
	return slData(v, w, h, tasksPerRank) + slModel(w, h, tasksPerRank);
}

inline double sl(const DistributedSparseMatrix& v, const DistributedDenseMatrix& w,
		const DistributedDenseMatrixCM& h, const std::string& hUnblockedName, int tasksPerRank=1) {
	return slData(v, w, hUnblockedName, tasksPerRank) + slModel(w, h, tasksPerRank);
}

// -- Loss ----------------------------------------------------------------------------------------

struct SlLoss : public LossConcept {
	SlLoss() { };
	SlLoss(mpi2::SerializationConstructor _) { };


	double operator()(const FactorizationData<>& data) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of SlLoss not yet implemented, using sequential computation.");
		}

		return sl(data.v, data.w, data.h);
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return sl(data.dv, data.dw, data.dh, data.tasksPerRank);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::SlLoss);

#endif
