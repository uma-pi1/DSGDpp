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
#include <mf/sgd/psgd.h> // help for compilers

namespace mf {

template<typename Update, typename Regularize, typename Loss,
	typename AdaptiveDecay,typename TestData,typename TestLoss>
void PsgdRunner::run(PsgdJob<Update, Regularize>& job, Loss& loss,
		mf_size_type epochs, AdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod,
		TestData* testData, TestLoss *testLoss) {
	std::stringstream ss;
	ss << "sgd tasks: " << job.tasks - (job.order == SGD_ORDER_WOR && job.shuffle == PSGD_SHUFFLE_PARALLEL ? 1 : 0);
	ss << ", ";
	if (job.order == SGD_ORDER_WOR) {
		ss << "shuffle tasks: " << (job.shuffle == PSGD_SHUFFLE_SEQ ? 0 : 1);
		ss << ", ";
	}
	ss << "polling delay: " << mpi2::TaskManager::getInstance().pollDelay() << " microseconds";

	LOG4CXX_INFO(detail::logger, "Starting PSGD (" << ss.str() << ")");

	detail::defaultRunner(job, loss, epochs, decay,
			boost::bind(&PsgdRunner::epoch<Update,Regularize>, this, _1, _2),
			trace, random_, balanceType, balanceMethod, testData, testLoss);

	LOG4CXX_INFO(detail::logger, "Finished PSGD");
}

template<typename Update, typename Regularize, typename Loss,
	typename AdaptiveDecay>
void PsgdRunner::run(PsgdJob<Update, Regularize>& job, Loss& loss,
		mf_size_type epochs, AdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod) {
	run(job, loss, epochs, decay, trace, balanceType, balanceMethod, (FactorizationData<>*)NULL, (NoLoss*)NULL);
}

namespace detail {
	template<typename Update, typename Regularize>
	struct PsgdUpdateSeqTask {
		static const std::string id() { return std::string("__mf/sgd/PsgdUpdateSeqTask_")
				+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			// receive data descriptor
			mpi2::PointerIntType pJob;
			double eps;
			mpi2::PointerIntType pSplits;
			ch.recv(*mpi2::unmarshal(pJob, eps, pSplits));
			PsgdJob<Update, Regularize>& job = *mpi2::intToPointer<PsgdJob<Update, Regularize> >(pJob);
			std::vector<mf_size_type>& splits = *mpi2::intToPointer<std::vector<mf_size_type> >(pSplits);

			// run SGD steps
			int id = info.groupId();
			mf_size_type begin = splits[id];
			mf_size_type end = splits[id+1];
			DecayConstant decay(eps);
			SgdRunner::updateSequential(job, decay, begin, end, begin);

			// signal that we are done
			ch.send();
		}
	};
}

template<typename Update, typename Regularize>
void PsgdRunner::updateSequential(PsgdJob<Update, Regularize>& job, double eps) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	unsigned tasks = job.tasks;
	std::vector<mf_size_type> splits = mpi2::split(job.nnz, job.tasks);

	// spawn task-1 parallel threads and also run sgd in this thread
	std::vector<mpi2::Channel> channels;
	if (tasks > 1) {
		tm.spawn<detail::PsgdUpdateSeqTask<Update, Regularize> >(tm.world().rank(), tasks-1, channels);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&job), eps, mpi2::pointerToInt(&splits)));
	}
	DecayConstant decay(eps);
	SgdRunner::updateSequential(job, decay, splits[tasks-1], splits[tasks], splits[tasks-1]);

	// wait for other threads to finish
	mpi2::economicRecvAll(channels, tm.pollDelay());
}

namespace detail {
	template<typename Update, typename Regularize>
	struct PsgdUpdateWrTask {
		static const std::string id() { return std::string("__mf/sgd/PsgdUpdateWrTask_")
				+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			rg::Random32 random = mpi2::getSeed(ch);

			// receive data descriptor
			mpi2::PointerIntType pJob;
			double eps;
			mpi2::PointerIntType pSplits;
			ch.recv(*mpi2::unmarshal(pJob, eps, pSplits));
			PsgdJob<Update, Regularize>& job = *mpi2::intToPointer<PsgdJob<Update, Regularize> >(pJob);
			std::vector<mf_size_type>& splits = *mpi2::intToPointer<std::vector<mf_size_type> >(pSplits);

			// run SGD steps
			int id = info.groupId();
			mf_size_type steps = splits[id+1]-splits[id];
			DecayConstant decay(eps);
			SgdRunner::updateWr(job, steps, decay, random, 0, job.nnz, 0);

			// signal that we are done
			ch.send();
		}
	};
}

template<typename Update, typename Regularize>
void PsgdRunner::updateWr(PsgdJob<Update, Regularize>& job, double eps) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	unsigned tasks = job.tasks;
	std::vector<mf_size_type> splits = mpi2::split(job.nnz, job.tasks);

	// spawn task-1 parallel threads and also run sgd in this thread
	std::vector<mpi2::Channel> channels;
	if (tasks > 1) {
		tm.spawn<detail::PsgdUpdateWrTask<Update, Regularize> >(tm.world().rank(), tasks-1, channels);
		mpi2::seed(channels, random_);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&job), eps, mpi2::pointerToInt(&splits)));
	}
	DecayConstant decay(eps);
	SgdRunner::updateWr(job, splits[tasks]-splits[tasks-1], decay, random_, 0, job.nnz, 0);

	// wait for other threads to finish
	mpi2::economicRecvAll(channels, tm.pollDelay());
}

namespace detail {
	struct PsgdShuffleTask {
		static const std::string id() { return std::string("__mf/sgd/PsgdShuffleTask"); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			rg::Random32 random = mpi2::getSeed(ch);

			// receive data
			mpi2::PointerIntType pPermutation;
			mf_size_type n;
			ch.recv(*mpi2::unmarshal(pPermutation, n));
			std::vector<mf_size_type>& permutation = *mpi2::intToPointer<std::vector<mf_size_type> >(pPermutation);

			// shuffle
			SgdRunner::permute(random, permutation, n, n);

			// notify
			ch.economicSend( mpi2::TaskManager::getInstance().pollDelay() );
		}
	};

	template<typename Update, typename Regularize>
	struct PsgdUpdateWorTask {
		static const std::string id() { return std::string("__mf/sgd/PsgdUpdateWorTask_")
				+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			rg::Random32 random = mpi2::getSeed(ch);

			// receive data descriptor
			mpi2::PointerIntType pJob;
			double eps;
			mpi2::PointerIntType pPermutation;
			mpi2::PointerIntType pSplits;
			ch.recv(*mpi2::unmarshal(pJob, eps, pPermutation, pSplits));

			PsgdJob<Update, Regularize>& job = *mpi2::intToPointer<PsgdJob<Update, Regularize> >(pJob);
			std::vector<mf_size_type>& permutation = *mpi2::intToPointer<std::vector<mf_size_type> >(pPermutation);
			std::vector<mf_size_type>& splits = *mpi2::intToPointer<std::vector<mf_size_type> >(pSplits);

			// run SGD steps
			int id = info.groupId();
			mf_size_type begin = splits[id];
			mf_size_type end = splits[id+1];
			DecayConstant decay(eps);
			SgdRunner::updateWor(job, decay, random, begin, end, begin, permutation);

			// signal that we are done
			ch.send();
		}
	};
}

template<typename Update, typename Regularize>
void PsgdRunner::updateWor(PsgdJob<Update, Regularize>& job, double eps) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();

	// determine how many SGD tasks to run
	int tasks = job.tasks;
	if (job.shuffle == PSGD_SHUFFLE_PARALLEL) tasks--;

	// check if we need to shuffle and if so, do so
	bool forcedSeqShuffle = false;
	std::vector<mf_size_type>& permutation =
			nextPermutation ? permutation_ : permutation2_;
	if (job.shuffle == PSGD_SHUFFLE_SEQ || permutation.size() != job.nnz || tasks == 0) {
		if (tasks == 0) {
			LOG4CXX_WARN(detail::logger, "Not enough tasks for parallel shuffling; using sequential shuffle");
			tasks = 1;
			forcedSeqShuffle = true;
		}
		SgdRunner::permute(random_, permutation, job.nnz, job.nnz);
	}
	// at this point, permutation points to a valid permutation

	// spawn task-1 parallel threads for SGD (current thread is also used --> tasks threads in total)
	std::vector<mf_size_type> splits = mpi2::split(job.nnz, tasks);
	std::vector<mpi2::Channel> channels;
	std::vector<boost::mpi::request> reqs;
	if (tasks > 1) {
		tm.spawn<detail::PsgdUpdateWorTask<Update, Regularize> >(tm.world().rank(), tasks-1, channels);
		mpi2::seed(channels, random_);
		mpi2::sendAll(channels, mpi2::marshal(
				mpi2::pointerToInt(&job),
				eps,
				mpi2::pointerToInt(&permutation),
				mpi2::pointerToInt(&splits)));
		reqs = mpi2::irecvAll(channels);
	}

	// check if we should start a shuffle task for the next epoch
	if (job.shuffle != PSGD_SHUFFLE_SEQ && !forcedSeqShuffle) {
		std::vector<mf_size_type>& next =
				nextPermutation ? permutation2_ : permutation_;

		mpi2::Channel ch = tm.spawn<detail::PsgdShuffleTask>(tm.world().rank());
		mpi2::seed(ch, random_);
		ch.send( mpi2::marshal(mpi2::pointerToInt(&next), job.nnz) );
		reqs.push_back( ch.irecv() );
	}

	// run sgd in this thread
	DecayConstant decay(eps);
	SgdRunner::updateWor(job, decay, random_, splits[tasks-1], splits[tasks], splits[tasks-1], permutation);

	// wait until all threads are done
	mpi2::economicWaitAll(reqs, tm.pollDelay());

	// switch permutations
	if (job.shuffle != PSGD_SHUFFLE_SEQ) nextPermutation = !nextPermutation;
}

template<typename Update, typename Regularize>
void PsgdRunner::epoch(PsgdJob<Update, Regularize>& job, double eps) {
	// SGD steps
	switch  (job.order) {
	case SGD_ORDER_SEQ:
		updateSequential(job, eps);
		break;
	case SGD_ORDER_WR:
		updateWr(job, eps);
		break;
	case SGD_ORDER_WOR:
		updateWor(job, eps);
		break;
	}

	// regularization step
	job.regularize(job, eps);
};

}

