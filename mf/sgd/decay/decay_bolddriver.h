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
#ifndef MF_SGD_DECAY_DECAY_BOLDDRIVER_H
#define MF_SGD_DECAY_DECAY_BOLDDRIVER_H

#include <boost/serialization/serialization.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <util/random.h>

#include <mf/sgd/decay/decay.h>

namespace mf {

/** The bold driver in an adaptive decay function that increases the step size by a small
 * fraction (default 5%) when the loss
 * decreased since the last iteration. It significantly decreases the step size (default 50%)
 * when the loss has increased.
 */
class BoldDriver : public AdaptiveDecayConcept , public DistributedAdaptiveDecayConcept {
public:
	BoldDriver(mpi2::SerializationConstructor _) : eps_(0), decrease_(0), increase_(0) { };
	BoldDriver(double eps, double decrease = 0.5, double increase=1.05)
	: eps_(eps), decrease_(decrease), increase_(increase) { };

	inline double operator()(FactorizationData<>& data, double* previousLoss, double* currentLoss, rg::Random32& random) {
		if (previousLoss == NULL) return eps_;
		if (*previousLoss <= *currentLoss) {
			eps_ *= decrease_;
		} else {
			eps_ *= increase_;
		}
		return eps_;
	}

	inline double operator()(DsgdFactorizationData<>& data, double* previousLoss, double* currentLoss, rg::Random32& random) {
		if (previousLoss == NULL) return eps_;
		if (*previousLoss <= *currentLoss) {
			eps_ *= decrease_;
		} else {
			eps_ *= increase_;
		}
		return eps_;
	}

	inline double operator()(DsgdPpFactorizationData<>& data, double* previousLoss, double* currentLoss, rg::Random32& random) {
		if (previousLoss == NULL) return eps_;
		if (*previousLoss <= *currentLoss) {
			eps_ *= decrease_;
		} else {
			eps_ *= increase_;
		}
		return eps_;
	}

	inline double operator()(AsgdFactorizationData<>& data, double* previousLoss, double* currentLoss, rg::Random32& random) {
		if (previousLoss == NULL) return eps_;
		if (*previousLoss <= *currentLoss) {
			eps_ *= decrease_;
		} else {
			eps_ *= increase_;
		}
		return eps_;
	}

private:
	double eps_;
	const double decrease_;
	const double increase_;

	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & eps_;
		ar & const_cast<double&>(decrease_);
		ar & const_cast<double&>(increase_);
	}
};

}

MPI2_TYPE_TRAITS(mf::BoldDriver);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::BoldDriver)

#endif
