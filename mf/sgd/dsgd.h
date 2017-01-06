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

#ifndef MF_SGD_DSGD_H
#define MF_SGD_DSGD_H

#include <mpi2/mpi2.h>

#include <mf/sgd/sgd.h>
#include <mf/sgd/dsgd-factorization.h>

namespace mf {

/** Order of selection of strata */
enum StratumOrder { STRATUM_ORDER_SEQ, STRATUM_ORDER_RSEQ, STRATUM_ORDER_WR, STRATUM_ORDER_WOR, STRATUM_ORDER_COWOR };

/** Describes an distributed SGD algorithm in terms of (1) an update function that performs a step
 * on a data point, (2) a regularize function that performs a step on the model.
 *
 * @tparam Update type of update function (model of UpdateConcept)
 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 */
template<typename Update, typename Regularize>
struct Dsgd : public Sgd<Update, Regularize> {
	Dsgd(Update update, Regularize regularize,
			SgdOrder order = SGD_ORDER_WOR, StratumOrder stratumOrder = STRATUM_ORDER_WOR,
			bool mapReduce = false)
	: Sgd<Update, Regularize>(update, regularize, order), stratumOrder(stratumOrder), mapReduce(mapReduce) {
	}

	Dsgd(Dsgd<Update,Regularize>& o)
	:  Sgd<Update, Regularize>(o.update, o.regularize, o.order),
	   stratumOrder(o.stratumOrder), mapReduce(o.mapReduce) {
	}

	Dsgd(mpi2::SerializationConstructor _)
	: Sgd<Update,Regularize>(mpi2::UNINITIALIZED) {
	}

	StratumOrder stratumOrder;
	bool mapReduce;

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<Sgd<Update,Regularize> >(*this);
		ar & stratumOrder;
		ar & mapReduce;
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::Dsgd);

namespace mf {


/** Describes a distributed SGD-based factorization job. Consists of a distributed SGD
 * algorithm (mf::Dsgd)
 * and a description of the data (mf::DsgdFactorizationData).
 *
 * @tparam Update type of update function (model of UpdateConcept)
 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 */
template<typename Update, typename Regularize>
struct DsgdJob : public DsgdFactorizationData<>, public Dsgd<Update,Regularize> {
	DsgdJob(const DistributedSparseMatrix& dv,
			DistributedDenseMatrix &dw, DistributedDenseMatrixCM& dh,
			Update update, Regularize regularize,
			SgdOrder order = SGD_ORDER_WOR, StratumOrder stratumOrder = STRATUM_ORDER_WOR,
			bool mapReduce = false, unsigned tasksPerRank=1)
	: DsgdFactorizationData<>(dv, dw, dh, tasksPerRank),
	  Dsgd<Update,Regularize>(update, regularize, order, stratumOrder, mapReduce) {
	}

	DsgdJob(DsgdFactorizationData<> job,
                Update update, Regularize regularize, 
                SgdOrder order = SGD_ORDER_WOR, StratumOrder stratumOrder = STRATUM_ORDER_WOR,
                bool mapReduce=false)
	: DsgdFactorizationData<>(job),
        Dsgd<Update,Regularize>(update, regularize, order, stratumOrder, mapReduce) {
	}

	DsgdJob(DsgdFactorizationData<>& job,
			Dsgd<Update,Regularize>& sgd)
	: DsgdFactorizationData<>(job), Dsgd<Update,Regularize>(sgd) {
	}

	DsgdJob(mpi2::SerializationConstructor _)
	: DsgdFactorizationData<>(mpi2::UNINITIALIZED), Dsgd<Update,Regularize>(mpi2::UNINITIALIZED) {
	}

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<DsgdFactorizationData<>  >(*this);
		ar & boost::serialization::base_object<Dsgd<Update, Regularize> >(*this);
	}
};


} // namespace mf

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::DsgdJob);

namespace mf {

/** Runs a distributed SGD-based factorization job */
class DsgdRunner {
public:
	DsgdRunner(rg::Random32& random) : random_(random) {
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
	void run(DsgdJob<Update, Regularize>& job,
			DistributedLoss& loss,
			mf_size_type epochs,
			DistributedAdaptiveDecay& decay,
			Trace& trace,
			BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE,
			TestData* testData = NULL, TestLoss *testLoss = NULL);

	template<typename Update, typename Regularize, typename DistributedLoss,
		typename DistributedAdaptiveDecay>
	void run(DsgdJob<Update, Regularize>& job, DistributedLoss& loss,
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
	void epoch(DsgdJob<Update, Regularize>& job, double eps);

private:
	rg::Random32& random_;
};

}

#include <mf/sgd/dsgd_impl.h>

#endif
