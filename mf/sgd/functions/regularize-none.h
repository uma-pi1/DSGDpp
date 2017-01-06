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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_NONE_H
#define MF_SGD_FUNCTIONS_REGULARIZE_NONE_H

#include <mf/sgd/functions/functions.h>
#include <mf/factorization.h>

namespace mf {

struct RegularizeNone : public RegularizeConcept {
	RegularizeNone() { };
	RegularizeNone(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& job, const double eps) {
	}

	inline bool rescaleStratumStepsize() {
		return true;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::RegularizeNone);


#endif
