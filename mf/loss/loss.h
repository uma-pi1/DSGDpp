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
#ifndef MF_LOSS_LOSS_H
#define MF_LOSS_LOSS_H

#include <math.h>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <mf/matrix/distributed_matrix.h>
#include <mf/factorization.h>
#include <mf/sgd/dsgd-factorization.h>
#include <mf/sgd/dsgdpp-factorization.h>
#include <mf/sgd/asgd-factorization.h>

namespace mf {

/** A loss function computes the value of the loss for a given factorization.
 *
 * \par Methods:
 * <table>
 * <tr><th>Signature</th><th>Description</th></tr>
 * <tr><td>double operator()(const FactorizationData<>& data)</td>
 * <td>Computes the value of the current loss</td></tr>
 * </table>
 */
class LossConcept {
};

/** A distributed loss function computes the value of the loss for a given
 * distributed factorization.
 *
 * \par Methods:
 * <table>
 * <tr><th>Signature</th><th>Description</th></tr>
 * <tr><td>double operator()(const DsgdFactorizationData<>& data)</td>
 * <td>Computes the value of the current loss (optional)</td></tr>
 * <tr><td>double operator()(const DapFactorizationData<>& data)</td>
 * <td>Computes the value of the current loss (optional)</td></tr>
 * <tr><td>double operator()(const AsgdFactorizationData<>& data)</td>
 * <td>Computes the value of the current loss (optional)</td></tr>
 * </table>
 */
class DistributedLossConcept {
};

struct NoLoss : public LossConcept, DistributedLossConcept {
	NoLoss() {};
	NoLoss(mpi2::SerializationConstructor _) { };

	double operator()(const FactorizationData<>& data) {
		return 0.;
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return 0.;
	}

	double operator()(const DsgdPpFactorizationData<>& data) {
		return 0.;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::NoLoss);

namespace mf {

template<typename Loss1, typename Loss2>
class SumLoss : public LossConcept, public DistributedLossConcept {
public:
	SumLoss(mpi2::SerializationConstructor _)
	: loss1(mpi2::UNINITIALIZED), loss2(mpi2::UNINITIALIZED) {
	}

	SumLoss(Loss1 loss1, Loss2 loss2) : loss1(loss1), loss2(loss2) { };

	template<typename FD>
	double operator()(const FD& data) {
		return loss1(data) + loss2(data);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & loss1;
		ar & loss2;
	}

	Loss1 loss1;
	Loss2 loss2;
};

}

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::SumLoss);

#endif
