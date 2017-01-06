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
#ifndef MF_SGD_DECAY_DECAY_CONSTANT_H
#define MF_SGD_DECAY_DECAY_CONSTANT_H

#include <boost/serialization/serialization.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <mf/sgd/decay/decay.h>
#include <mf/factorization.h>
#include <mf/sgd/asgd-factorization.h>
#include <mf/sgd/dsgd-factorization.h>
#include <mf/sgd/dsgdpp-factorization.h>

namespace mf {

/** Constant decay */
class DecayConstant : public StaticDecayConcept, public AdaptiveDecayConcept {
public:
	DecayConstant(mpi2::SerializationConstructor _) : eps_(0) { };
	DecayConstant(double eps) : eps_(eps) { };

	inline double operator()(unsigned _) const { return eps_; }

	// added by me
	inline double operator()(FactorizationData<>& data, double* prevLoss, double* curLoss, rg::Random32& random){ return eps_; }

	inline double operator()(FactorizationData<>& data){ return eps_; }

	inline double operator()(DsgdFactorizationData<>& data, double* prevLoss, double* curLoss, rg::Random32& random){ return eps_; }

	inline double operator()(DsgdFactorizationData<>& data){ return eps_; }

	inline double operator()(DsgdPpFactorizationData<>& data, double* prevLoss, double* curLoss, rg::Random32& random){ return eps_; }
	inline double operator()(DsgdPpFactorizationData<>& data){ return eps_; }

	inline double operator()(AsgdFactorizationData<>& data){ return eps_; }
	inline double operator()(AsgdFactorizationData<>& data, double* prevLoss, double* curLoss, rg::Random32& random){ return eps_; }
	/**/

private:
	const double eps_;

	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & const_cast<double&>(eps_);
	}
};

}

MPI2_TYPE_TRAITS(mf::DecayConstant);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::DecayConstant)

#endif
