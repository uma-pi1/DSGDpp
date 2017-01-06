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
#ifndef MF_MATRIX_OP_SUMOFPROD_H
#define MF_MATRIX_OP_SUMOFPROD_H

#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/op/sum.h>
#include <mf/matrix/op/sums.h>

namespace mf {

/** Computes the sum of the product of the given matrices */
inline double sumOfProd(const DenseMatrix& w, const DenseMatrixCM& h) {
	boost::numeric::ublas::vector<double> sumW = sums2(w);
	boost::numeric::ublas::vector<double> sumH = sums1(h);
	return boost::numeric::ublas::inner_prod(sumW, sumH);
}

inline double sumOfProd(const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h, int tasksPerRank) {
	boost::numeric::ublas::vector<double> sumW = sums2(w, tasksPerRank);
	boost::numeric::ublas::vector<double> sumH = sums1(h, tasksPerRank);
	return boost::numeric::ublas::inner_prod(sumW, sumH);
}

} // namespace mf



#endif
