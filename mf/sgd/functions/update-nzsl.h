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
#ifndef MF_SGD_FUNCTIONS_UPDATE_NZSL_H
#define MF_SGD_FUNCTIONS_UPDATE_NZSL_H

#include <boost/serialization/serialization.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <mf/sgd/functions/functions.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/sgd/sgd.h>
#include <mf/types.h>

namespace mf {

struct UpdateNzsl : public UpdateConcept {
	UpdateNzsl() { };
	UpdateNzsl(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& data,
			const unsigned i, const unsigned j,	const double x,
			const double eps) {
		mf_size_type ir = i*data.r;
		mf_size_type jr = j*data.r;

		double wh = 0;
		for (unsigned z=0; z<data.r; z++) {
			wh += data.wValues[ir + z] * data.hValues[z + jr];
		}

		double f = eps * -2. * (x-wh);
		for (unsigned z=0; z<data.r; z++) {
			double temp = data.wValues[ir + z];
			data.wValues[ir + z] -= f * data.hValues[z + jr];
			data.hValues[z + jr] -= f * temp;
		}
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

// much slower, not sure why
struct UpdateNzslBoost {
	UpdateNzslBoost() { };
	UpdateNzslBoost(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& job,
			const unsigned int i, const unsigned int j,	const unsigned int x,
			const double eps) {
		boost::numeric::ublas::matrix_row<DenseMatrix> row = boost::numeric::ublas::row(job.w, i);
		boost::numeric::ublas::matrix_column<DenseMatrixCM> col = boost::numeric::ublas::column(job.h, j);
		const double innerProduct = boost::numeric::ublas::inner_prod(row, col);

		temp = row;
		row -= eps * ( -2 * ( x - innerProduct ) * col );
		col -= eps * ( -2 * ( x - innerProduct ) * temp );
	}

private:
	boost::numeric::ublas::vector<double> temp;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::UpdateNzsl);
MPI2_TYPE_TRAITS(mf::UpdateNzslBoost);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::UpdateNzsl);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::UpdateNzslBoost);

#endif
