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
#ifndef MF_SGD_PSGD_H
#define MF_SGD_PSGD_H

#include <mf/sgd/sgd.h>

namespace mf {

enum PsgdShuffle {
	PSGD_SHUFFLE_SEQ,                      /**< shuffling is performed sequentially after each epoch */
	PSGD_SHUFFLE_PARALLEL,                 /**< shuffling is performed in parallel (replacing one of the SGD tasks) */
	PSGD_SHUFFLE_PARALLEL_ADDITIONAL_TASK  /**< shuffling is performed in parallel (in an additional task) */
};

template<typename Update, typename Regularize>
struct PsgdJob : public SgdJob<Update,Regularize> {
    using FactorizationData<>::tasks;

	PsgdJob(const SparseMatrix& v, DenseMatrix &w, DenseMatrixCM& h,
			Update update, Regularize regularize, SgdOrder order = SGD_ORDER_WR,
			int tasks = 1, PsgdShuffle shuffle = PSGD_SHUFFLE_PARALLEL)
	: SgdJob<Update,Regularize>(v, w, h, update, regularize, order), shuffle(shuffle) {
		this->tasks = tasks;
	}

	PsgdJob(FactorizationData<> data,
			Update update, Regularize regularize, SgdOrder order = SGD_ORDER_WR,
			int tasks = 1, PsgdShuffle shuffle= PSGD_SHUFFLE_PARALLEL)
	: SgdJob<Update,Regularize>(data, update, regularize, order), shuffle(shuffle) {
		this->tasks = tasks;
	}

public:
	PsgdShuffle shuffle;
};

/** Runs a PSGD-based factorization job */
class PsgdRunner {
public:
	PsgdRunner(rg::Random32& random) : random_(random), nextPermutation(true) {
	}

	/** Runs a number of Hogwild SGD epochs using a distributed adaptive decay function. This is the most
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
	template<typename Update, typename Regularize, typename Loss,
		typename AdaptiveDecay,
		typename TestData, typename TestLoss>
	void run(PsgdJob<Update, Regularize>& job,
			Loss& loss,
			mf_size_type epochs,
			AdaptiveDecay& decay,
			Trace& trace,
			BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE,
			TestData* testData = NULL, TestLoss *testLoss = NULL);

	template<typename Update, typename Regularize, typename Loss,
		typename AdaptiveDecay>
	void run(PsgdJob<Update, Regularize>& job, Loss& loss,
			mf_size_type epochs, AdaptiveDecay& decay,
			Trace& trace, BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE);

	/** Runs a single Hogwild SGD epoch using a fixed step size. An epoch consists of a number
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
	void epoch(PsgdJob<Update, Regularize>& job, double eps);

	template<typename Update, typename Regularize>
	void updateSequential(PsgdJob<Update, Regularize>& job, double eps);

	template<typename Update, typename Regularize>
	void updateWr(PsgdJob<Update, Regularize>& job, double eps);

	template<typename Update, typename Regularize>
	void updateWor(PsgdJob<Update, Regularize>& job, double eps);

	void setPrngState(const rg::Random32& random) {
		random_ = random;
	}

private:
	rg::Random32& random_;
	std::vector<mf_size_type> permutation_;
	std::vector<mf_size_type> permutation2_;
	bool nextPermutation;
};

}

#include <mf/sgd/psgd_impl.h>

#endif
