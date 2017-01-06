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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_SL_H
#define MF_SGD_FUNCTIONS_REGULARIZE_SL_H

#include <mf/sgd/functions/functions.h>
#include <mf/factorization.h>
#include <mf/matrix/op/crossprod.h>

namespace mf {

struct RegularizeSl : public RegularizeConcept {
	RegularizeSl() { };
	RegularizeSl(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& job, const double eps) {
		if (job.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeSl not yet implemented, using sequential computation.");
		}

		DenseMatrixCM HHT = tcrossprod(job.h);
		DenseMatrix WTW = crossprod(job.w);
		mf_size_type r = job.r;

		// update W
		const DenseMatrixCM::array_type& hht = HHT.data();
		mf_size_type ir = 0;
		for (unsigned i=0; i<job.m; i++){
			mf_size_type jr = 0;
			for (unsigned j=0; j<job.r; j++){
				double gradW = 0;
				for (unsigned k=0; k<job.r; k++){
					gradW += job.wValues[k + ir] * hht[k + jr];
				}
				job.wValues[j + ir] -= (2. * eps * gradW);
				jr += r;
			}
			ir += r;
		}

		// update H
		const DenseMatrix::array_type& wtw = WTW.data();
		mf_size_type jr = 0;
		for (unsigned j=0; j<job.n; j++) {
			mf_size_type ir = 0;
			for (unsigned i=0; i<job.r; i++) {
				double gradH = 0;
				for (unsigned k=0; k<job.r; k++){
					gradH += wtw[k + ir] * job.hValues[k + jr];
				}
				job.hValues[i + jr] -= (2. * eps * gradH);
				ir += r;
			}
			jr += r;
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

MPI2_TYPE_TRAITS(mf::RegularizeSl);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::RegularizeSl);

#endif
