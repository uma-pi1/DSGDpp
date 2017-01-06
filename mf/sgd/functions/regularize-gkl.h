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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_GKL_H
#define MF_SGD_FUNCTIONS_REGULARIZE_GKL_H

#include <mf/sgd/functions/functions.h>
#include <mf/factorization.h>
#include <mf/matrix/op/sums.h>

namespace mf {

struct RegularizeGkl : public RegularizeConcept {
	RegularizeGkl() { };
	RegularizeGkl(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& job, const double eps) {
		if (job.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeGkl not yet implemented, using sequential computation.");
		}

		boost::numeric::ublas::vector<double> uW = sums2(job.w);
		boost::numeric::ublas::vector<double> uH = sums1(job.h);
		uW *= eps;
		uH *= eps;

		// update W
		for (unsigned i=0; i<job.m; i++){
			for (unsigned k=0; k<job.r; k++){
				job.w(i,k) -= uH[k];
			}
		}
		// update H
		for (unsigned k=0; k<job.r; k++){
			for (unsigned j=0; j<job.n; j++){
				job.h(k,j) -= uW[k];
			}
		}
	}

	inline bool rescaleStratumStepsize() {
		return false;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}

};

}

MPI2_TYPE_TRAITS(mf::RegularizeGkl);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::RegularizeGkl);

#endif
