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
#ifndef MF_SGD_FUNCTIONS_REGULARIZE_MAXNORM_H
#define MF_SGD_FUNCTIONS_REGULARIZE_MAXNORM_H

#include <math.h>

#include <boost/serialization/serialization.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <mf/sgd/functions/functions.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/sgd/sgd.h>
#include <mf/types.h>
#include <mf/matrix/op/sums.h>

namespace mf {

template<typename R>
struct RegularizeMaxNorm : public RegularizeConcept {
	typedef R Regularize;

	RegularizeMaxNorm(Regularize regularize, double b)
	: regularize(regularize), b(b)
	{
	};

	RegularizeMaxNorm(mpi2::SerializationConstructor _)
	: regularize(mpi2::UNINITIALIZED), b(FP_NAN)
	{
	};

	inline void operator()(FactorizationData<>& data, const double eps) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of RegularizeMaxNorm not yet implemented, using sequential computation.");
		}

		regularize(data, eps);

		boost::numeric::ublas::vector<double> result1 = squaredSums1(data.w);
		for (mf_size_type i=0; i<result1.size(); i++) {
			if (result1[i] > b) {
				for (mf_size_type j=0; j<data.r; j++) {
					data.wValues[i*data.r+j] /= b; // assume data.w is in row-major
				}
			}
		}

		boost::numeric::ublas::vector<double> result2 = squaredSums2(data.h);
		for (mf_size_type j=0; j<result2.size(); j++) {
			if (result2[j] > b) {
				for (mf_size_type i=0; i<data.r; i++) {
					data.hValues[j*data.r+i] /= b; // assume data.h is in column-major
				}
			}
		}
	}

private:
	Regularize regularize;
	const double b;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & regularize;
		ar & const_cast<double&>(b);
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR1(mf::RegularizeMaxNorm);

#endif
