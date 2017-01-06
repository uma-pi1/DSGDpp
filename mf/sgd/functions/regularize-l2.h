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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_L2_H
#define MF_SGD_FUNCTIONS_REGULARIZE_L2_H

#include <mf/sgd/functions/functions.h>
#include <mf/factorization.h>

namespace mf {

struct RegularizeL2 : public RegularizeConcept {
	RegularizeL2(double lambda) : lambda(lambda) { };
	RegularizeL2(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& job, const double eps) {
		if (job.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeL2 not yet implemented, using sequential computation.");
		}

		if (lambda==0) return;
		double factor;
		factor = 1 - eps * 2 * lambda;
		if (factor < 0.5) {
			// smoothing to avoid too big reduction; shrinks at speed 1/x
			// once eps is small enough, this branch won't be reached anymore
			factor = 0.25 / (1 - factor);
		}
		for (unsigned p=0; p<job.m*job.r; p++) {
			job.wValues[p] *= factor;
		}
		for (unsigned p=0; p<job.r*job.n; p++) {
			job.hValues[p] *= factor;
		}
	}

	inline bool rescaleStratumStepsize() {
		return true;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & lambda;
	}

	double lambda;
};

}

MPI2_TYPE_TRAITS(mf::RegularizeL2);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::RegularizeL2);

#endif
