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
 * Implementation for sgd/dsgd.h
 * DO NOT INCLUDE DIRECTLY
 */

#include <algorithm>

#include <mf/sgd/asgd.h> // help for compilers

#include <mf/matrix/op/shuffle.h>

namespace mf {

namespace detail {
	struct AsgdInitTask {
		static const std::string id() { return std::string("__mf/sgd/AsgdInitTask_"); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			DistributedSparseMatrix dv(mpi2::UNINITIALIZED);
			ch.recv(dv);
			unsigned groupId = info.groupId();
			mpi2::RemoteVar var = dv.block(groupId, 0);
			const SparseMatrix& localV = *var.getLocal<SparseMatrix>();
			mpi2::env().create("asgd_locks1", new boost::shared_array<boost::mutex>(new boost::mutex[localV.size1()]));
			mpi2::env().create("asgd_locks2", new boost::shared_array<boost::mutex>(new boost::mutex[localV.size2()]));
			mpi2::env().create("asgd_h_cache", new DenseMatrixCM(*mpi2::env().get<DenseMatrixCM>("asgd_h_work"))); // TODO: get name from fact. data
			rg::Random32* random = new rg::Random32();
			mpi2::env().create("asgd_runner_random", random);
			mpi2::env().create("asgd_runner", new PsgdRunner(*random)); // runner stored in env so that we can reuse permutation vector
			ch.send();
		}
	};

	struct AsgdDestroyTask {
		static const std::string id() { return std::string("__mf/sgd/AsgdDestroyTask_"); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			mpi2::env().erase<boost::shared_array<boost::mutex> >("asgd_locks1");
			mpi2::env().erase<boost::shared_array<boost::mutex> >("asgd_locks2");
			mpi2::env().erase<DenseMatrixCM>("asgd_h_cache");
			mpi2::env().erase<PsgdRunner>("asgd_runner");
			mpi2::env().erase<rg::Random32>("asgd_runner_random");
			ch.send();
		}
	};

	struct AsgdShuffleTask {
		static const std::string id() { return std::string("__mf/sgd/AsgdShuffleTask_"); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			mpi2::logBeginEvent("shuffle");

			std::vector<mpi2::Channel>& pairwiseChannels = info.pairwiseChannels();
			mf_size_type d = pairwiseChannels.size();
			int groupId = info.groupId();

			// get relevant data
			DenseMatrixCM& localH = *mpi2::env().get<DenseMatrixCM>("asgd_h_work");
			DenseMatrixCM& cachedH = *mpi2::env().get<DenseMatrixCM>("asgd_h_cache");
			boost::shared_array<boost::mutex>& locks2 =
					*mpi2::env().get<boost::shared_array<boost::mutex> >("asgd_locks2");
			mf_size_type n = localH.size2();
			mf_size_type r = localH.size1();
			DistributedMatrix<DenseMatrixCM> masterH(mpi2::UNINITIALIZED);
			bool averageDeltas;
			ch.recv(*mpi2::unmarshal(masterH, averageDeltas));
			DenseMatrixCM& masterHblock = *masterH.block(0, groupId).getLocal<DenseMatrixCM>();
			double weight = averageDeltas ? 1./d : 1;

			// compute delta between local and cache; update cache
			DenseMatrixCM deltaH(localH.size1(), localH.size2());
			for (mf_size_type j=0; j<n; j++) {
				boost::mutex::scoped_lock lock2(locks2[j]);
				for (mf_size_type i=0; i<r; i++) {
					deltaH(i,j) = localH(i,j)-cachedH(i,j);
					cachedH(i,j) = localH(i,j);
				}
			}

			// send the delta to the respective nodes (including myself; could be optimized)
			std::vector<boost::mpi::request> reqs;
			std::vector<mf_size_type> splits = mpi2::split(n, d); // TODO: splits need to be chosen conformingly to H (which by default is just this)
			for (int i=0; i<d; i++) {
				mf_size_type begin = splits[i] * r;
				mf_size_type end = splits[i+1] * r;
				reqs.push_back( pairwiseChannels[i].isend((double *)&deltaH.data()[begin], end-begin) );
			}

			// receive the delta from all nodes
			mf_size_type jbegin = splits[groupId];
			mf_size_type jend = splits[groupId+1];
			std::vector<DenseMatrixCM> deltas(d);
			for (int i=0; i<d; i++) {
				deltas[i].resize(r, jend-jbegin, false);
				reqs.push_back( pairwiseChannels[i].irecv((double *)&deltas[i].data()[0], (jend-jbegin)*r) );
			}

			// wait until communication finished
			mpi2::economicWaitAll(reqs, mpi2::TaskManager::getInstance().pollDelay());

			// add the deltas to the master
			for (int i=0; i<d; i++) {
				boost::numeric::ublas::noalias(masterHblock) += deltas[i] * weight;
			}

			// send/recv the master to/from all nodes
			reqs.clear();
			for (int i=0; i<d; i++) {
				mf_size_type begin = splits[i] * r;
				mf_size_type end = splits[i+1] * r;
				reqs.push_back( pairwiseChannels[i].isend((double *)&masterHblock.data()[0], (jend-jbegin)*r) );
				reqs.push_back( pairwiseChannels[i].irecv((double *)&deltaH.data()[begin], end-begin) );
			}
			mpi2::economicWaitAll(reqs, mpi2::TaskManager::getInstance().pollDelay());

			// update work and cached H
			for (mf_size_type j=0; j<n; j++) {
				boost::mutex::scoped_lock lock2(locks2[j]);
				for (mf_size_type i=0; i<r; i++) {
					double delta = deltaH(i,j) - cachedH(i,j);
					cachedH(i,j) = deltaH(i,j);
					localH(i,j) += delta;
				}
			}

			ch.send();

			mpi2::logEndEvent("shuffle");
		}
	};

	template<typename Update, typename Regularize>
	struct AsgdTask {
		static const std::string id() { return std::string("__mf/sgd/AsgdTask_")
			+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			rg::Random32 random = mpi2::getSeed(ch);
			// receive factorization data
			AsgdJob<Update, Regularize> job(mpi2::UNINITIALIZED);
			double eps;
			ch.recv(*mpi2::unmarshal(job, eps));

			// get the data
			unsigned groupId = info.groupId();
			mpi2::RemoteVar var = job.dv.block(groupId, 0);
			const SparseMatrix& localV = *var.getLocal<SparseMatrix>();
			var = job.dw.block(groupId, 0);
			DenseMatrix& localW = *var.getLocal<DenseMatrix>();
			DenseMatrixCM& localH = *mpi2::env().get<DenseMatrixCM>("asgd_h_work");

			// create locked updates and put column locks in environment
			boost::shared_array<boost::mutex>& locks1 =
					*mpi2::env().get<boost::shared_array<boost::mutex> >("asgd_locks1");
			boost::shared_array<boost::mutex>& locks2 =
					*mpi2::env().get<boost::shared_array<boost::mutex> >("asgd_locks2");
			UpdateLock<Update> updateLock(job.update, localV.size1(), localV.size2(), locks1, locks2);

			// run an epoch
			mpi2::logBeginEvent("computation");
			PsgdJob<UpdateLock<Update>,Regularize> localJob(
					localV,
					localW,
					localH,
					updateLock,
					job.regularize,
					job.order,
					job.tasksPerRank
			);
			PsgdRunner& runner = *mpi2::env().get<PsgdRunner>("asgd_runner");
			runner.setPrngState(random);
			runner.epoch(localJob, eps); // note: random not changed!
			mpi2::logEndEvent("computation");

			// send
			ch.send();
		}
	};

} // mf::detail

template<typename Update, typename Regularize>
void AsgdRunner::epoch(AsgdJob<Update, Regularize>& job, double eps) {
	// start the ASGD task on all ranks
	std::vector<mpi2::Channel> sgdChannels;
	mpi2::TaskManager::getInstance().spawnAll<detail::AsgdTask<Update,Regularize> >(sgdChannels);
	mpi2::seed(sgdChannels, random_);
	mpi2::sendAll(sgdChannels, mpi2::marshal(job, eps));

	// shuffle repeatedly until ASGD tasks are done
	int noShuffles = 0;
	std::vector<boost::mpi::request> sgdRequests = mpi2::irecvAll(sgdChannels);
	do {
		// check whether SGD tasks are done
		for (int i=0; i<sgdRequests.size();) {
			boost::optional<boost::mpi::status> msg = sgdRequests[i].test();
			if (msg) {
				if (sgdRequests.size() > 1) sgdRequests[i] = sgdRequests[sgdRequests.size()-1];
				sgdRequests.pop_back();
			} else {
				i++;
			}
		}

		// start the shuffle tasks on all ranks
		// (Once more even after SGD is done; need to make sure everybody has same local copy.)
		std::vector<mpi2::Channel> shuffleChannels;
		mpi2::TaskManager::getInstance().spawnAll<detail::AsgdShuffleTask>(shuffleChannels, true);
		mpi2::sendAll(shuffleChannels, mpi2::marshal(job.dh, job.averageDeltas));
		noShuffles++;

		// wait for shuffle tasks to finish
		mpi2::economicRecvAll(shuffleChannels, mpi2::TaskManager::getInstance().pollDelay());
	} while (sgdRequests.size() > 0);

	LOG4CXX_INFO(detail::logger, "Synchronized " << noShuffles << " times");
}

template<typename Update, typename Regularize, typename Loss,
	typename DistributedAdaptiveDecay,typename TestData,typename TestLoss>
void AsgdRunner::run(AsgdJob<Update, Regularize>& job, Loss& loss,
		mf_size_type epochs, DistributedAdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod,
		TestData* testData, TestLoss *testLoss) {
	LOG4CXX_INFO(detail::logger, "Starting ASGD (polling delay: " << mpi2::TaskManager::getInstance().pollDelay() << " microseconds)");

	// print information about stratum order
	switch (job.stratumOrder) {
	case STRATUM_ORDER_SEQ:
		LOG4CXX_INFO(detail::logger, "Using SEQ order for selecting strata");
		break;
	case STRATUM_ORDER_WR:
		LOG4CXX_INFO(detail::logger, "Using WR order for selecting strata");
		break;
	case STRATUM_ORDER_WOR:
		LOG4CXX_INFO(detail::logger, "Using WOR order for selecting strata");
		break;
	}

	if (job.averageDeltas) {
		LOG4CXX_INFO(detail::logger, "Deltas will be averaged");
	} else {
		LOG4CXX_INFO(detail::logger, "Deltas will not be averaged");
	}

	// create a copy of H on all ranks
	LOG4CXX_INFO(detail::logger, "Unblocking H...");
	const std::string hUnblockedName = "asgd_h_work";
	mpi2::createCopyAll(hUnblockedName, AsgdFactorizationData<>::H(0,0));
	unblockAll(job.dh, hUnblockedName);

	// initialize ASGD
	LOG4CXX_INFO(detail::logger, "Initializing...");
	std::vector<mpi2::Channel> channels;
	mpi2::TaskManager::getInstance().spawnAll<detail::AsgdInitTask>(channels);
	mpi2::sendAll(channels, job.dv);
	mpi2::recvAll(channels);

	// run ASGD
	detail::defaultRunner(job, loss, epochs, decay,
			boost::bind(&AsgdRunner::epoch<Update,Regularize>, this, _1, _2),
			trace, random_, balanceType, balanceMethod, testData, testLoss);

	// destroy ASGD
	mpi2::TaskManager::getInstance().spawnAll<detail::AsgdDestroyTask>(channels);
	mpi2::recvAll(channels);

	// delete the copy of H from all ranks
	mpi2::eraseAll<AsgdFactorizationData<>::H>(hUnblockedName);

	LOG4CXX_INFO(detail::logger, "Finished ASGD");
}

template<typename Update, typename Regularize, typename DistributedLoss,
	typename DistributedAdaptiveDecay>
void AsgdRunner::run(AsgdJob<Update, Regularize>& job, DistributedLoss& loss,
		mf_size_type epochs, DistributedAdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod) {
	run(job, loss, epochs, decay, trace, balanceType, balanceMethod, (DsgdFactorizationData<>*)NULL, (NoLoss*)NULL);
}



} // mf
