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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_ABS_H
#define MF_SGD_FUNCTIONS_REGULARIZE_ABS_H

#include <math.h>

#include <boost/serialization/serialization.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <mf/sgd/functions/functions.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/sgd/sgd.h>
#include <mf/types.h>

namespace mf {

template<typename R>
struct RegularizeAbs : public RegularizeConcept {
	typedef R Regularize;

	RegularizeAbs(Regularize regularize) : regularize(regularize)
	{
	};

	inline void operator()(FactorizationData<>& data, const double eps) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeAbs not yet implemented, using sequential computation.");
		}

		regularize(data, eps);
		for (mf_size_type i=0; i<data.m; i++) {
			for (mf_size_type z=0; z<data.r; z++) {
				data.w(i,z) = fabs(data.w(i,z));
			}
		}
		for (mf_size_type z=0; z<data.r; z++) {
			for (mf_size_type j=0; j<data.n; j++) {
				data.h(z,j) = fabs(data.h(z,j));
			}
		}
	}

	inline bool rescaleStratumStepsize() {
		return regularize.rescaleStratumStepsize();
	}

private:
	Regularize regularize;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & regularize;
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR1(mf::RegularizeAbs);

#endif
