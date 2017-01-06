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
/** \file
 * Methods for matrix factorization via ASGD.
 */

#ifndef MF_SGD_ASGD_H
#define MF_SGD_ASGD_H

#include <mpi2/mpi2.h>

#include <mf/sgd/sgd.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/asgd-factorization.h>

namespace mf {


template<typename Update, typename Regularize>
struct AsgdJob : public AsgdFactorizationData<>, public Dsgd<Update,Regularize> {
	AsgdJob(const DistributedSparseMatrix& dv,
			DistributedDenseMatrix &dw, DistributedDenseMatrixCM& dh,
			Update update, Regularize regularize,
			SgdOrder order = STRATUM_ORDER_WOR, StratumOrder stratumOrder = STRATUM_ORDER_WOR,
			unsigned tasksPerRank=1, bool averageDeltas = false)
	: AsgdFactorizationData<>(dv, dw, dh, tasksPerRank),
	  Dsgd<Update,Regularize>(update, regularize, order, stratumOrder),
	  averageDeltas(averageDeltas) {
	}

	AsgdJob(AsgdFactorizationData<> job,
			Update update, Regularize regularize, SgdOrder order = STRATUM_ORDER_WR,
			bool averageDeltas = false)
	: AsgdFactorizationData<>(job),
	  Dsgd<Update,Regularize>(update, regularize, order),
	  averageDeltas(averageDeltas) {
	}

	AsgdJob(mpi2::SerializationConstructor _)
	: AsgdFactorizationData<>(mpi2::UNINITIALIZED), Dsgd<Update,Regularize>(mpi2::UNINITIALIZED) {
	}

	bool averageDeltas;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<AsgdFactorizationData<>  >(*this);
		ar & boost::serialization::base_object<Dsgd<Update, Regularize> >(*this);
		ar & averageDeltas;
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::AsgdJob);

namespace mf {

/** Runs a distributed ASGD-based factorization job */
class AsgdRunner {
public:
	AsgdRunner(rg::Random32& random) : random_(random) {
	}

	/** Runs a number of ASGD epochs using a distributed adaptive decay function. This is the most
	 * commonly used method to run DSGD. Here, an epoch consists of a number
	 * of ASGD update steps (as many as training points) and a single ASGD regularize step.
	 *
	 * @param job DSGD parameters and data
	 * @param loss instance of loss function
	 * @param epochs number of epochs to run
	 * @param decay instance of decay function
	 * @param trace a trace that will be filled with information about the SGD run
	 *
 	 * @tparam Update type of update function (model of UpdateConcept)
 	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 	 * @tparam Loss type of loss function (model of DistributedLossConcept)
 	 * @tparam AdaptiveDecay type of decay function (model of DistributedAdaptiveDecayConcept)
	 */
	template<typename Update, typename Regularize, typename DistributedLoss,
		typename DistributedAdaptiveDecay,
		typename TestData, typename TestLoss>
	void run(AsgdJob<Update, Regularize>& job,
			DistributedLoss& loss,
			mf_size_type epochs,
			DistributedAdaptiveDecay& decay,
			Trace& trace,
			BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE,
			TestData* testData = NULL, TestLoss *testLoss = NULL);

	template<typename Update, typename Regularize, typename DistributedLoss,
		typename DistributedAdaptiveDecay>
	void run(AsgdJob<Update, Regularize>& job, DistributedLoss& loss,
			mf_size_type epochs, DistributedAdaptiveDecay& decay,
			Trace& trace, BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE);


private:
	template<typename Update, typename Regularize>
	void epoch(AsgdJob<Update, Regularize>& job, double eps);

	rg::Random32& random_;
};

}

#include <mf/sgd/asgd_impl.h>

#endif
