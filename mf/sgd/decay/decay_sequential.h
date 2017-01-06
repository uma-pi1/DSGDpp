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
/*
 * decay_sequential.h
 *
 *  Created on: Dec 27, 2011
 *      Author: chteflio
 */

#ifndef MF_SGD_DECAY_DECAY_SEQUENTIAL_H_
#define MF_SGD_DECAY_DECAY_SEQUENTIAL_H_

#include <boost/serialization/serialization.hpp>
#include <mpi2/types.h>
#include <mpi2/uninitialized.h>
#include <util/random.h>
#include <mf/sgd/decay/decay.h>

namespace mf {

/** The sequential decay calculates the step in the next epoch as a function of the epoch (n).
 *  eps(n)=a/(n+1+A)^alpha where n=0,1,2,..., #epochs
 *  Given to the constructor are the parameters:
 *  	eps(0)
 *  	A
 *  	alpha
 *  The algorithm then calculates the correct a and continues with finding the step size sequence
 */
class SequentialDecay : public StaticDecayConcept {
public:
	SequentialDecay(mpi2::SerializationConstructor _) : eps_(0), eps0_(0), alpha_(0), A_(0), a_(0),n_(0) { };
	SequentialDecay(double eps0, double alpha=1, double A = 0)
	: eps0_(eps0),alpha_(alpha), A_(A), a_(eps0*pow(1+A,alpha)),n_(0) { };

	inline double nextStep(){
		eps_=a_/pow(n_+1+A_,alpha_);
		n_++;
		return eps_;
	}

	// the following declarations are in order to fit the DefaultRunner CONSIDER REWRITING
	inline double operator()(FactorizationData<>& data, double* prevLoss, double* curLoss, rg::Random32& random){ return nextStep(); }

	inline double operator()(FactorizationData<>& data){ return nextStep(); }

	inline double operator()(DsgdFactorizationData<>& data, double* prevLoss, double* curLoss, rg::Random32& random){ return nextStep(); }

	inline double operator()(DsgdFactorizationData<>& data){ return nextStep(); }


private:
	double eps_;
	mf_size_type n_;
	const double eps0_;
	const double A_;
	const double a_;
	const double alpha_;

	friend class boost::serialization::access;

	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & eps_;
		ar & n_;
		ar & const_cast<double&>(eps0_);
		ar & const_cast<double&>(A_);
		ar & const_cast<double&>(a_);
		ar & const_cast<double&>(alpha_);
	}
};

}

MPI2_TYPE_TRAITS(mf::SequentialDecay);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::SequentialDecay)

#endif /* MF_SGD_DECAY_DECAY_SEQUENTIAL_H_ */
