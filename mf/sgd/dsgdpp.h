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
 * Methods for matrix factorization via DSGD.
 */

#ifndef MF_SGD_DSGDPP_H
#define MF_SGD_DSGDPP_H

#include <mpi2/mpi2.h>

#include <mf/sgd/sgd.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/dsgdpp-factorization.h>


namespace mf {


/** Describes a distributed SGD-based factorization job. Consists of a distributed SGD
 * algorithm (mf::Dsgd)
 * and a description of the data (mf::DsgdPpFactorizationData).
 *
 * @tparam Update type of update function (model of UpdateConcept)
 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 */
template<typename Update, typename Regularize>
struct DsgdPpJob : public DsgdPpFactorizationData<>, public Dsgd<Update,Regularize> {
	DsgdPpJob(const DistributedSparseMatrix& dv,
			DistributedDenseMatrix &dw, DistributedDenseMatrixCM& dh,
			Update update, Regularize regularize,
			SgdOrder order = STRATUM_ORDER_WOR, StratumOrder stratumOrder = STRATUM_ORDER_WOR,
			unsigned tasksPerRank=1)
	: DsgdPpFactorizationData<>(dv, dw, dh, tasksPerRank),
	  Dsgd<Update,Regularize>(update, regularize, order, stratumOrder) {
	}

	DsgdPpJob(DsgdPpFactorizationData<> job,
                  Update update, Regularize regularize, SgdOrder order = STRATUM_ORDER_WR, 
                  StratumOrder stratumOrder = STRATUM_ORDER_WOR)
	: DsgdPpFactorizationData<>(job),
        Dsgd<Update,Regularize>(update, regularize, order, stratumOrder) {
	}

	DsgdPpJob(DsgdPpFactorizationData<>& job,
                  Dsgd<Update,Regularize>& sgd)
	: DsgdPpFactorizationData<>(job), Dsgd<Update,Regularize>(sgd) {
	}

	DsgdPpJob(mpi2::SerializationConstructor _)
	: DsgdPpFactorizationData<>(mpi2::UNINITIALIZED), Dsgd<Update,Regularize>(mpi2::UNINITIALIZED) {
	}

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<DsgdPpFactorizationData<>  >(*this);
		ar & boost::serialization::base_object<Dsgd<Update, Regularize> >(*this);
	}
};


} // namespace mf

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::DsgdPpJob);

namespace mf {

/** Runs a distributed SGD-based factorization job */
class DsgdPpRunner {
public:
	DsgdPpRunner(rg::Random32& random) : random_(random) {
	}

	/** Runs a number of DSGD epochs using a distributed adaptive decay function. This is the most
	 * commonly used method to run DSGD. Here, an epoch consists of a number
	 * of DSGD update steps (as many as training points) and a single DSGD regularize step.
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
	void run(DsgdPpJob<Update, Regularize>& job,
			DistributedLoss& loss,
			mf_size_type epochs,
			DistributedAdaptiveDecay& decay,
			Trace& trace,
			BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE,
			TestData* testData = NULL, TestLoss *testLoss = NULL);

	template<typename Update, typename Regularize, typename DistributedLoss,
		typename DistributedAdaptiveDecay>
	void run(DsgdPpJob<Update, Regularize>& job, DistributedLoss& loss,
			mf_size_type epochs, DistributedAdaptiveDecay& decay,
			Trace& trace, BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE);

	/** Runs a single DSGD epoch using a fixed step size. An epoch consists of a number
	 * of SGD update steps (as many as data points) and a single SGD regularize step.
	 *
	 * @param job SGD parameters and data
	 * @param eps step size to use
	 * @param number of concurrent tasks to use on each rank
	 *
 	 * @tparam Update type of update function (model of UpdateConcept)
 	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
	 */
	template<typename Update, typename Regularize>
	void epoch(DsgdPpJob<Update, Regularize>& job, double eps);

private:
	rg::Random32& random_;
};

}

#include <mf/sgd/dsgdpp_impl.h>

#endif
