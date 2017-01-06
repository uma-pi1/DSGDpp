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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_L1_H
#define MF_SGD_FUNCTIONS_REGULARIZE_L1_H

#include <mf/sgd/functions/functions.h>
#include <mf/factorization.h>

namespace mf {

struct RegularizeL1 : public RegularizeConcept {
	RegularizeL1(double lambda) : lambdaW(lambda), lambdaH(lambda) { };
	RegularizeL1(double lambdaW, double lambdaH) : lambdaW(lambdaW), lambdaH(lambdaH) { };
	RegularizeL1(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& job, const double eps) {
		if (job.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeL1 not yet implemented, using sequential computation.");
		}

		if (lambdaW !=0) {
			double v = eps * lambdaW;
			for (unsigned p=0; p<job.m*job.r; p++) {
				double w = job.wValues[p];
				if (w >= 0) { // we also change w if it is zero (subgradient)
					w -= v;
				} else {
					w += v;
				}
				job.wValues[p] = w;
			}
		}
		if (lambdaH != 0) {
			double v = eps * lambdaH;
			for (unsigned p=0; p<job.r*job.n; p++) {
				double h = job.hValues[p];
				if (h >= 0) { // we also change h if it is zero (subgradient)
					h -= v;
				} else {
					h += v;
				}
				job.hValues[p] = h;
			}
		}
	}

	inline bool rescaleStratumStepsize() {
		return true;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & lambdaW;
		ar & lambdaH;
	}

	double lambdaW;
	double lambdaH;
};

}

MPI2_TYPE_TRAITS(mf::RegularizeL1);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::RegularizeL1);

#endif
