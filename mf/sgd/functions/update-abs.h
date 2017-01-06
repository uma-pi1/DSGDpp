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
#ifndef MF_SGD_FUNCTIONS_UPDATE_ABS_H
#define MF_SGD_FUNCTIONS_UPDATE_ABS_H

#include <math.h>

#include <boost/serialization/serialization.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <mf/sgd/functions/functions.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/sgd/sgd.h>
#include <mf/types.h>

namespace mf {

template<typename U>
struct UpdateAbs : public UpdateConcept {
	typedef U Update;

	UpdateAbs(Update update) : update(update) {
	};

	UpdateAbs(mpi2::SerializationConstructor _) : update(mpi2::UNINITIALIZED)
	{
	};

	inline void operator()(FactorizationData<>& data,
			const unsigned i, const unsigned j,	const double x,
			const double eps) {
		update(data, i, j, x, eps);
		for (unsigned z=0; z<data.r; z++) {
			data.w(i,z) = fabs(data.w(i,z));
			data.h(z,j) = fabs(data.h(z,j));
		}
	}

private:
	Update update;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & update;
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR1(mf::UpdateAbs);

#endif
