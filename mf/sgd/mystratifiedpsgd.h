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
 * Methods for matrix factorization via Stratified PSGD.
 */

#ifndef MF_SGD_STRATIFIEDPSGD_H
#define MF_SGD_STRATIFIEDPSGD_H

#include <mpi2/mpi2.h>
#include <boost/thread/mutex.hpp>
#include <boost/thread/locks.hpp>
#include <boost/shared_array.hpp>
#include <mf/sgd/sgd.h>
//#include <mf/sgd/dsgd-factorization.h>

namespace mf {


template<typename Update, typename Regularize>
struct StratifiedPsgd : public Sgd<Update, Regularize> {
	StratifiedPsgd(Update update, Regularize regularize,
			SgdOrder order = SGD_ORDER_WOR)
	: Sgd<Update, Regularize>(update, regularize, order) {
	}

	StratifiedPsgd(StratifiedPsgd<Update,Regularize>& o)
	:  Sgd<Update, Regularize>(o.update, o.regularize, o.order) {
	}

	StratifiedPsgd(mpi2::SerializationConstructor _)
	: Sgd<Update,Regularize>(mpi2::UNINITIALIZED) {
	}


private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<Sgd<Update,Regularize> >(*this);
	}
};

}

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::StratifiedPsgd);

namespace mf {


template<typename Update, typename Regularize>
struct StratifiedPsgdJob : public SgdJob<Update,Regularize> {
    using FactorizationData<>::tasks;

	StratifiedPsgdJob(const SparseMatrix& v, DenseMatrix &w, DenseMatrixCM& h,
			Update update, Regularize regularize, SgdOrder order = SGD_ORDER_WR,
			int tasks = 1)
	: SgdJob<Update,Regularize>(v, w, h, update, regularize, order) {
		this->tasks = tasks;
	}

	StratifiedPsgdJob(FactorizationData<> data,
			Update update, Regularize regularize, SgdOrder order = SGD_ORDER_WR,
			int tasks = 1)
	: SgdJob<Update,Regularize>(data, update, regularize, order) {
		this->tasks = tasks;
	}
	
// 	boost::shared_array<boost::mutex> locks_;

private:

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & boost::serialization::base_object<SgdJob<Update, Regularize> >(*this);
	}
};


} // namespace mf

MPI2_SERIALIZATION_CONSTRUCTOR2(mf::StratifiedPsgdJob);

namespace mf {

/** Runs a distributed SGD-based factorization job */
class StratifiedPsgdRunner {
public:
	StratifiedPsgdRunner(rg::Random32& random) : random_(random) {
	}


	StratifiedPsgdRunner(rg::Random32& random, std::vector<mf_size_type>& permutation,
			std::vector<mf_size_type>& offsets) : random_(random), permutation_(permutation), offsets_(offsets) {
	}

	template<typename Update, typename Regularize, typename Loss,
		typename AdaptiveDecay,
		typename TestData, typename TestLoss>
	void run(StratifiedPsgdJob<Update, Regularize>& job,
			Loss& loss,
			mf_size_type epochs,
			AdaptiveDecay& decay,
			Trace& trace,
			BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE,
			TestData* testData = NULL, TestLoss *testLoss = NULL);

	template<typename Update, typename Regularize, typename Loss,
		typename AdaptiveDecay>
	void run(StratifiedPsgdJob<Update, Regularize>& job, Loss& loss,
			mf_size_type epochs, AdaptiveDecay& decay,
			Trace& trace, BalanceType balanceType = BALANCE_NONE, BalanceMethod balanceMethod = BALANCE_SIMPLE);


	template<typename Update, typename Regularize>
	void epoch(StratifiedPsgdJob<Update, Regularize>& job, double eps);

	

private:

	
	std::vector<mf_size_type> permutation_;
	std::vector<mf_size_type> offsets_;
	rg::Random32& random_;
};

}


#include "mystratifiedpsgd_impl.h"

#endif
