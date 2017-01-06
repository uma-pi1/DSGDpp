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
 * update-nzsl-l1.h
 *
 *  Created on: Jan 7, 2014
 *      Author: chteflio
 */

#ifndef MF_SGD_FUNCTIONS_UPDATE_NZSL_L1_H_
#define MF_SGD_FUNCTIONS_UPDATE_NZSL_L1_H_


#include <boost/serialization/serialization.hpp>

#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

#include <mf/sgd/functions/functions.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/sgd/sgd.h>
#include <mf/types.h>


namespace mf {

struct UpdateNzslL1 : public UpdateConcept {
	UpdateNzslL1(double lambda) : lambda(lambda) { };
	UpdateNzslL1(mpi2::SerializationConstructor _) { };

	inline void operator()(FactorizationData<>& data,
			const unsigned i, const unsigned j,	const double x,
			const double eps) {
		mf_size_type ir = i*data.r;
		mf_size_type jr = j*data.r;

		double wh = 0;
		for (unsigned z=0; z<data.r; z++) {
			wh += data.wValues[ir + z] * data.hValues[z + jr];
		}

		double f1 = eps * -2. * (x-wh);

		double f3 = eps * lambda / (*data.nnz1)[i + data.nnz1offset];
		double f4 = eps * lambda / (*data.nnz2)[j + data.nnz2offset];
		for (unsigned z=0; z<data.r; z++) {

			double temp = data.wValues[ir + z];


			if(data.wValues[ir + z] > 0){
				data.wValues[ir + z] -= f1 * data.hValues[z + jr] + f3;
			}else if(data.wValues[ir + z] < 0){
				data.wValues[ir + z] -= f1 * data.hValues[z + jr] - f3;
			}
			else{
				double value = fabs(f1 * data.hValues[z + jr]) - f3;
				value = (value > 0 ? value : 0);

				if((x-wh)*data.hValues[z + jr] >= 0){
					data.wValues[ir + z] -= -value;
				}else{
					data.wValues[ir + z] += -value;
				}

			}


			if (data.hValues[z + jr] > 0){
				data.hValues[z + jr] -= f1 * temp + f4;
			}else if(data.hValues[z + jr] < 0){
				data.hValues[z + jr] -= f1 * temp - f4;
			}
			else{
				double value = fabs(f1 * temp) - f4;
				value = (value > 0 ? value : 0);

				if((x-wh)*temp >= 0){
					data.hValues[z + jr] -= -value;
				}else{
					data.hValues[z + jr] += -value;
				}

			}


		}

	}

private:
	double lambda;

	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & lambda;
	}


};

}

MPI2_TYPE_TRAITS(mf::UpdateNzslL1);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::UpdateNzslL1);

#endif /* MF_SGD_FUNCTIONS_UPDATE_NZSL_L1_H_ */
