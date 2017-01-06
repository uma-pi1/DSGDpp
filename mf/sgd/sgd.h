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
 * Methods for matrix factorization via SGD.
 */

#ifndef MF_SGD_SGD_H
#define MF_SGD_SGD_H

#include <vector>

#include <log4cxx/logger.h>

#include <util/evaluation.h>
#include <util/io.h>
#include <util/random.h>

#include <mpi2/mpi2.h>

#include <mf/matrix/op/balance.h>
#include <mf/factorization.h>
#include <mf/trace.h>
#include <mf/sgd/functions/regularize-none.h>

namespace mf {

/** Order of selection of training points */
enum SgdOrder { SGD_ORDER_SEQ, SGD_ORDER_WR, SGD_ORDER_WOR };


/** Describes an SGD algorithm in terms of (1) an update function that performs a step
 * on a data point, (2) a regularize function that performs a step on the model, and (3)
 * the order in which to process training points.
 *
 * @tparam Update type of update function (model of UpdateConcept)
 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 */
template<typename Update, typename Regularize>
struct Sgd {
	Sgd(Update update, Regularize regularize, SgdOrder order = SGD_ORDER_WOR)
	: update(update), regularize(regularize), order(order) {
	}

	/** Update functor. Required signature is
	 * operator()(FactorizationJob job, unsigned i, unsigned j, unsigned x, double eps)
	 */
	Update update;

	/** Regularization functor. Required signature is
	 * operator()(FactorizationJob job, double eps)
	 */
	Regularize regularize;

	/** Order of SGD steps */
	SgdOrder order;

	Sgd(mpi2::SerializationConstructor _)
	: update(mpi2::UNINITIALIZED), regularize(mpi2::UNINITIALIZED), order(SGD_ORDER_WR) { };

private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & update;
		ar & regularize;
		ar & order;
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::Sgd);



namespace mf {

/** Describes an SGD-based factorization job. Consists of an SGD algorithm (mf::Sgd)
 * and a description of the data (mf::FactorizationData).
 *
 * @tparam Update type of update function (model of UpdateConcept)
 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 */
template<typename Update, typename Regularize>
struct SgdJob : public FactorizationData<>, public Sgd<Update,Regularize> {
	SgdJob(const SparseMatrix& v, DenseMatrix &w, DenseMatrixCM& h,
			Update update, Regularize regularize, SgdOrder order = SGD_ORDER_WR)
	: FactorizationData<>(v, w, h), Sgd<Update,Regularize>(update, regularize, order) {
	}

	SgdJob(FactorizationData<> data,
			Update update, Regularize regularize, SgdOrder order = SGD_ORDER_WR)
	: FactorizationData<>(data), Sgd<Update,Regularize>(update, regularize, order) {
	}
};

class PsgdRunner;

/** Runs an SGD-based factorization job */
class SgdRunner {
public:
	SgdRunner(rg::Random32& random) : random_(random) {
	}

	/** Runs a number of SGD epochs using an adaptive decay function. This is the most
	 * commonly used method to run SGD. Here, an epoch consists of a number
	 * of SGD update steps (as many as training points) and a single SGD regularize step.
	 *
	 * @param job SGD parameters and data
	 * @param loss instance of loss function
	 * @param epochs number of epochs to run
	 * @param decay instance of decay function
	 * @param trace a trace that will be filled with information about the SGD run
	 *
 	 * @tparam Update type of update function (model of UpdateConcept)
 	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 	 * @tparam Loss type of loss function (model of LossConcept)
 	 * @tparam AdaptiveDecay type of decay function (model of AdaptiveDecayConcept)
	 */
	template<typename Update, typename Regularize, typename Loss, typename AdaptiveDecay,
		typename TestData, typename TestLoss>
	void run(SgdJob<Update, Regularize>& job,
			Loss& loss,
			mf_size_type epochs,
			AdaptiveDecay& decay,
			Trace& trace,
			BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE,
			TestData* testData = NULL, TestLoss *testLoss = NULL);

	template<typename Update, typename Regularize, typename Loss, typename AdaptiveDecay>
	void run(SgdJob<Update, Regularize>& job,
			Loss& loss,
			mf_size_type epochs,
			AdaptiveDecay& decay,
			Trace& trace,
			BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE);

	/** Runs a number of SGD update steps using a decay function.
	 *
	 * @param job SGD parameters and data
	 * @param epochs number of steps to run (a step corresponds to processing one training point)
	 * @param decay instance of decay function
	 *
 	 * @tparam Update type of update function (model of UpdateConcept)
 	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
 	 * @tparam StaticDecay type of decay function (model of StaticDecayConcept)
	 */
	template<typename Update, typename Regularize, typename StaticDecay>
	void update(SgdJob<Update, Regularize>& job, mf_size_type steps, StaticDecay& decay);

	/** Runs a number of SGD update steps using a fixed step size.
	 *
	 * @param job SGD parameters and data
	 * @param epochs number of steps to run (a step corresponds to processing one training point)
	 * @param eps step size to use
	 *
 	 * @tparam Update type of update function (model of UpdateConcept)
 	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
	 */
	template<typename Update, typename Regularize>
	void update(SgdJob<Update, Regularize>& job, mf_size_type steps, double eps);

	/** Runs a single SGD epoch using a fixed step size. An epoch consists of a number
	 * of SGD update steps (as many as data points) and a single SGD regularize step.
	 *
	 * @param job SGD parameters and data
	 * @param eps step size to use
	 *
 	 * @tparam Update type of update function (model of UpdateConcept)
 	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
	 */
	template<typename Update, typename Regularize>
	void epoch(SgdJob<Update, Regularize>& job, double eps);

	/** Runs a single SGD epoch using a fixed step size. An epoch consists of a number
	 * of SGD update steps (as many as data points) and a single SGD regularize step.
	 *
	 * @param job SGD parameters and data
	 * @param epsUpdate step size to use for updates
	 * @param epsRegularize step size to use for regularization
	 *
	 * @tparam Update type of update function (model of UpdateConcept)
	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
	 */
	template<typename Update, typename Regularize>
	void epoch(SgdJob<Update, Regularize>& job, double epsUpdate, double epsRegularize);

	/** Runs a single SGD regularize step.
	 *
	 * @param job SGD parameters and data
	 * @param eps step size to use
	 *
 	 * @tparam Update type of update function (model of UpdateConcept)
 	 * @tparam Regularize type of regularize function (model of RegularizeConcept)
	 */
	template<typename Update, typename Regularize>
	void regularize(SgdJob<Update, Regularize>& job, double eps);

	static void permute(rg::Random32& random, std::vector<mf_size_type>& permutation, int n, int k);

	/** Runs steps SGD steps in sequential order */
	template<typename Update, typename Regularize, typename Decay>
	void updateSequential(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay);

	/** Runs steps SGD steps in sequential order */
	template<typename Update, typename Regularize, typename Decay>
	static void updateSequential(SgdJob<Update, Regularize>& job, Decay& decay,
			mf_size_type begin, mf_size_type end, mf_size_type decayOffset);

	/** Runs steps SGD steps in WR order */
	template<typename Update, typename Regularize, typename Decay>
	void updateWr(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay);

	/** Runs steps SGD steps in WR order */
	template<typename Update, typename Regularize, typename Decay>
	static void updateWr(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay,
			rg::Random32& random, mf_size_type begin, mf_size_type end, mf_size_type decayOffset);

	/** Runs steps SGD steps in WOR order. */
	template<typename Update, typename Regularize, typename Decay>
	void updateWor(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay);

	/** Runs steps SGD steps in WOR order. */
	template<typename Update, typename Regularize, typename Decay>
	static void updateWor(SgdJob<Update, Regularize>& job, Decay& decay,
			rg::Random32& random, mf_size_type begin, mf_size_type end, mf_size_type decayOffset,
			const std::vector<mf_size_type>& permutation);

private:
	rg::Random32& random_;
	std::vector<mf_size_type> permutation_; // temp space for WOR ordering
	rg::Timer t;
};

}

#include <mf/sgd/sgd_impl.h>

#endif
