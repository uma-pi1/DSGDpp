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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_TRUNCATE_H
#define MF_SGD_FUNCTIONS_REGULARIZE_TRUNCATE_H

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
struct RegularizeTruncate : public RegularizeConcept {
	typedef R Regularize;

	RegularizeTruncate(Regularize regularize, double min, double max)
	: regularize(regularize), wmin(min), wmax(max), hmin(min), hmax(max)
	{
	};

	RegularizeTruncate(Regularize regularize, double wmin, double wmax, double hmin, double hmax)
	: regularize(regularize), wmin(wmin), wmax(wmax), hmin(hmin), hmax(hmax)
	{
	};

	RegularizeTruncate(mpi2::SerializationConstructor _)
	: regularize(mpi2::UNINITIALIZED), wmin(FP_NAN), wmax(FP_NAN), hmin(FP_NAN), hmax(FP_NAN)
	{
	};

	inline void operator()(FactorizationData<>& data, const double eps) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeTruncate not yet implemented, using sequential computation.");
		}

		regularize(data, eps);

		mf_size_type s = data.m * data.r;
		for (mf_size_type p = 0; p<s; p++) {
			double& w = data.wValues[p];
			if (w<=wmin) { // eps softens up the boundary a bit
				w = wmin+eps;
			} else if (w >= wmax) {
				w = wmax-eps;
			}
		}

		s = data.r * data.n;
		for (mf_size_type p = 0; p<s; p++) {
			double& h = data.hValues[p];
			if (h<=hmin) {
				h = hmin+eps;
			} else if (h >= hmax) {
				h = hmax-eps;
			}
		}
	}

	inline bool rescaleStratumStepsize() {
		return regularize.rescaleStratumStepsize();
	}

private:
	Regularize regularize;
	const double wmin;
	const double wmax;
	const double hmin;
	const double hmax;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & regularize;
		ar & const_cast<double&>(wmin);
		ar & const_cast<double&>(wmax);
		ar & const_cast<double&>(hmin);
		ar & const_cast<double&>(hmax);
	}
};


// do nothing if no regularization!
template<>
struct RegularizeTruncate<RegularizeNone> : public RegularizeConcept {
	typedef RegularizeNone Regularize;

	RegularizeTruncate(Regularize regularize, double min, double max)
	: regularize(regularize), wmin(min), wmax(max), hmin(min), hmax(max)
	{
	};

	RegularizeTruncate(Regularize regularize, double wmin, double wmax, double hmin, double hmax)
	: regularize(regularize), wmin(wmin), wmax(wmax), hmin(hmin), hmax(hmax)
	{
	};

	RegularizeTruncate(mpi2::SerializationConstructor _)
	: regularize(mpi2::UNINITIALIZED), wmin(FP_NAN), wmax(FP_NAN), hmin(FP_NAN), hmax(FP_NAN)
	{
	};

	inline void operator()(FactorizationData<>& data, const double eps) {
	}

	inline bool rescaleStratumStepsize() {
		return regularize.rescaleStratumStepsize();
	}

private:
	Regularize regularize;
	const double wmin;
	const double wmax;
	const double hmin;
	const double hmax;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & regularize;
		ar & const_cast<double&>(wmin);
		ar & const_cast<double&>(wmax);
		ar & const_cast<double&>(hmin);
		ar & const_cast<double&>(hmax);
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR1(mf::RegularizeTruncate);

#endif
