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
#ifndef MF_SGD_FUNCTIONS_UPDATE_TRUNCATE_H
#define MF_SGD_FUNCTIONS_UPDATE_TRUNCATE_H

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
struct UpdateTruncate : public UpdateConcept {
	typedef U Update;

	UpdateTruncate(Update update, double min, double max)
	: update(update), wmin(min), wmax(max), hmin(min), hmax(max)
	{
	};

	UpdateTruncate(Update update, double wmin, double wmax, double hmin, double hmax)
	: update(update), wmin(wmin), wmax(wmax), hmin(hmin), hmax(hmax)
	{
	};

	UpdateTruncate(mpi2::SerializationConstructor _)
	: update(mpi2::UNINITIALIZED), wmin(FP_NAN), wmax(FP_NAN), hmin(FP_NAN), hmax(FP_NAN)
	{
	};

	inline void operator()(FactorizationData<>& data,
			const unsigned i, const unsigned j,	const double x,
			const double eps) {
		update(data, i, j, x, eps);

		mf_size_type pi = i*data.r;
		mf_size_type pj = j*data.r;
		for (unsigned z=0; z<data.r; z++) {
			double& w = data.wValues[pi + z];
			if (w<=wmin) { // eps softens up the boundary a bit
				w = wmin+eps;
			} else if (w >= wmax) {
				w = wmax-eps;
			}

			double& h = data.hValues[z + pj];
			if (h<=hmin) {
				h = hmin+eps;
			} else if (h >= hmax) {
				h = hmax-eps;
			}
		}
	}
	//Update update;
private:
	Update update;
	const double wmin;
	const double wmax;
	const double hmin;
	const double hmax;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & update;
		ar & const_cast<double&>(wmin);
		ar & const_cast<double&>(wmax);
		ar & const_cast<double&>(hmin);
		ar & const_cast<double&>(hmax);
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR1(mf::UpdateTruncate);

#endif
