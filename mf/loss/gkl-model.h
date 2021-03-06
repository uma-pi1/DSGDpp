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
#ifndef MF_LOSS_GKL_MODEL_H
#define MF_LOSS_GKL_MODEL_H


#include <mf/loss/loss.h>
#include <mf/matrix/op/sumofprod.h>

namespace mf {

inline double gklModel(const DenseMatrix& w, const DenseMatrixCM& h) {
	return sumOfProd(w, h);
}

inline double gklModel(const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h, int tasksPerRank) {
	return sumOfProd(w, h, tasksPerRank);
}


// -- Loss ----------------------------------------------------------------------------------------

struct GklModelLoss : public LossConcept {
	GklModelLoss() { };
	GklModelLoss(mpi2::SerializationConstructor _) { };

	double operator()(const FactorizationData<>& data) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of GklModelLoss not yet implemented, using sequential computation.");
		}
		return gklModel(data.w, data.h);
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return gklModel(data.dw, data.dh, data.tasksPerRank);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::GklModelLoss);

#endif
