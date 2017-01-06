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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_NZL2_H
#define MF_SGD_FUNCTIONS_REGULARIZE_NZL2_H

#include <mf/sgd/functions/functions.h>
#include <mf/factorization.h>

namespace mf {

struct RegularizeNzl2 : public RegularizeConcept {
	RegularizeNzl2(double lambda) : lambda(lambda) { };
	RegularizeNzl2(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& job, double eps) {
		if (job.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeNzl2 not yet implemented, using sequential computation.");
		}

		if (lambda==0) return;
		mf_size_type max=job.nnz12max;
		//std::cout<<"Max in Regularize: "<<max<<std::endl;

		double eps2Lambda = 2 * eps * lambda;
		if (1-eps2Lambda*max<0.5){ // find eps that will smooth the factor in the worst case
			eps=(1/(2*lambda*max))*(1-(0.25/(eps2Lambda*max)));
			eps2Lambda = 2 * eps * lambda;
		}

		// TODO: rewrite to avoid division by r
		for (mf_size_type p=0; p<job.m*job.r; p++) {
			job.wValues[p] *= 1 - ( eps2Lambda * (*job.nnz1)[(p/job.r) + job.nnz1offset] );
		}
		for (mf_size_type p=0; p<job.r*job.n; p++) {
			job.hValues[p] *= 1 - ( eps2Lambda * (*job.nnz2)[(p/job.r) + job.nnz2offset] );
		}
	}

	inline bool rescaleStratumStepsize() {
		return true;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, unsigned int version) {
		ar & lambda;
	}

	double lambda;
};

}

MPI2_TYPE_TRAITS(mf::RegularizeNzl2);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::RegularizeNzl2);

#endif
