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
#ifndef MF_LOSS_NZRMSE_H
#define MF_LOSS_NZRMSE_H

#include <math.h>
#include <mf/loss/nzsl.h>

namespace mf {

struct NzRmseLoss : public LossConcept, DistributedLossConcept {
	NzRmseLoss() {};
	NzRmseLoss(mpi2::SerializationConstructor _) { };

	double operator()(const FactorizationData<>& data) {
		return sqrt(nzsl(data.v, data.w, data.h, data.tasks) / data.nnz);
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return sqrt(nzsl(data.dv, data.dw, data.dh, data.tasksPerRank) / data.nnz);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::NzRmseLoss);



#endif
