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

#include <mf/sgd/dsgd.h> // help for compilers

#include <mf/matrix/op/shuffle.h>

namespace mf {

template<typename Update, typename Regularize, typename DistributedLoss,
	typename DistributedAdaptiveDecay,typename TestData,typename TestLoss>
void DsgdRunner::run(DsgdJob<Update, Regularize>& job, DistributedLoss& loss,
		mf_size_type epochs, DistributedAdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod,
		TestData* testData, TestLoss *testLoss) {
	LOG4CXX_INFO(detail::logger, "Starting DSGD (polling delay: " << mpi2::TaskManager::getInstance().pollDelay() << " microseconds)");

	// print information about stratum order
	switch (job.stratumOrder) {
	case STRATUM_ORDER_SEQ:
		LOG4CXX_INFO(detail::logger, "Using SEQ order for selecting strata");
		break;
	case STRATUM_ORDER_RSEQ:
		LOG4CXX_INFO(detail::logger, "Using RSEQ order for selecting strata");
		break;
	case STRATUM_ORDER_WR:
		LOG4CXX_INFO(detail::logger, "Using WR order for selecting strata");
		break;
	case STRATUM_ORDER_WOR:
		LOG4CXX_INFO(detail::logger, "Using WOR order for selecting strata");
		break;
	case STRATUM_ORDER_COWOR:
		LOG4CXX_INFO(detail::logger, "Using COWOR order for selecting strata");
		break;
	default:
		RG_THROW(rg::InvalidArgumentException, rg::paste("Invalid stratum order: ", job.stratumOrder));
	}

	if (job.mapReduce) {
		LOG4CXX_INFO(detail::logger, "Using slow MapReduce-style implementation");
	} else {
		LOG4CXX_INFO(detail::logger, "Using fast DSGD+ implementation");
	}

	detail::defaultRunner(job, loss, epochs, decay,
			boost::bind(&DsgdRunner::epoch<Update,Regularize>, this, _1, _2),
			trace, random_, balanceType, balanceMethod, testData, testLoss);

	LOG4CXX_INFO(detail::logger, "Finished DSGD");
}

template<typename Update, typename Regularize, typename DistributedLoss,
	typename DistributedAdaptiveDecay>
void DsgdRunner::run(DsgdJob<Update, Regularize>& job, DistributedLoss& loss,
		mf_size_type epochs, DistributedAdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod) {
	run(job, loss, epochs, decay, trace, balanceType, balanceMethod, (DsgdFactorizationData<>*)NULL, (NoLoss*)NULL);
}


namespace detail {

template<typename Update, typename Regularize>
struct DsgdTask {
	static const std::string id() { return std::string("__mf/sgd/DsgdTask_")
			+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		rg::Random32 random = mpi2::getSeed(ch);

		// receive task tags and figure out my id
		std::vector<mpi2::Channel>& channels = info.pairwiseChannels();
		int id = info.groupId();
		const mf_size_type d = info.groupSize();

		// receive data descriptor
		DsgdJob<Update,Regularize> job(mpi2::UNINITIALIZED);
		double eps;
		boost::numeric::ublas::matrix<mf_size_type> schedule(d, d);
		ch.recv(*mpi2::unmarshal(job, eps, schedule));

		// run
		DenseMatrixCM *H = NULL;
		DenseMatrixCM *Hprev  = NULL;
		if (ch.world().size() > 1  || job.mapReduce) {
			H =  new DenseMatrixCM(0,0);
			Hprev = new DenseMatrixCM(0,0);
		}
		SgdRunner runner(random);
		for (mf_size_type subepoch = 0; subepoch < d; subepoch++) {
			LOG4CXX_DEBUG(detail::logger, id << ": "
					<< "Starting subepoch " << subepoch);

			mpi2::logBeginEvent("subepoch");
			mpi2::logBeginEvent("communication");

			mf_size_type b1 = id;
			mf_size_type b2 = schedule(subepoch, id);

			// get W and V
			mpi2::RemoteVar rv = job.dw.block(b1,0); // compiler yells if I don't use a temp...!
			DenseMatrix *bW = rv.getLocal<DenseMatrix>();
			rv = job.dv.block(b1, b2);
			SparseMatrix *bV = rv.getLocal<SparseMatrix>();

			// get H
			if (job.mapReduce) {
				mpi2::RemoteVar vH = job.dh.block(0,b2);
				vH.getCopy(*H);
			} else {
				// fetch H directly from previous task / send my previous H to next task
				std::swap(H, Hprev);
				if (ch.world().size() == 1) { // single node
					mpi2::RemoteVar vH = job.dh.block(0,b2);
					H = vH.getLocal<DenseMatrixCM>();
				} else if (subepoch == 0) { // first epoch: read from env
					mpi2::RemoteVar vH = job.dh.block(0,b2);
					vH.getCopy(*H);
				} else { // subsequent epoch: communicate directly
					boost::mpi::request reqs[4];
					int numReqs = 0;
					mpi2::PointerIntType
						pH_cur = mpi2::pointerToInt(H),								// current pointer to H
						pHprev_cur = mpi2::pointerToInt(Hprev),                     // current pointer to Hprev
						pH_new = mpi2::pointerToInt(H),                             // new pointer to H
						pHprev_new = mpi2::pointerToInt(Hprev);                     // new pointer to Hprev
					bool exchangePointersH = false, exchangePointersHprev = false;

					// send the previous block of H to the next task
					mf_size_type idNext;
					for (idNext=0; idNext<d; idNext++) {
						if (schedule(subepoch,idNext) == schedule(subepoch-1,id)) break;
					}
					if (channels[idNext].remote().rank == channels[idNext].local().rank) {
						reqs[numReqs++] = channels[idNext].isend(pHprev_cur); // send pointer
						exchangePointersHprev = true;
					} else {
						reqs[numReqs++] = channels[idNext].isend(*Hprev);     // send data
					}

					// receive the next block of H from the previous task
					mf_size_type idPrev;
					for (idPrev=0; idPrev<d; idPrev++) {
						if (schedule(subepoch-1,idPrev) == b2) break;
					}
					if (channels[idPrev].remote().rank == channels[idPrev].local().rank) {
						reqs[numReqs++] = channels[idPrev].irecv(pH_new); // recv pointer
						exchangePointersH = true;
					} else {
						reqs[numReqs++] = channels[idPrev].irecv(*H);     // recv data
					}

					// wait for communication to finish
					boost::mpi::wait_all(reqs, reqs+numReqs);

					// if a pointer was received, we send back our pointer (pointers will be exchanged)
					// similarly, if a pointer was sent, we receive a new pointer
					numReqs = 0;
					if (exchangePointersH) reqs[numReqs++] = channels[idPrev].isend(pH_cur); // send pointer
					if (exchangePointersHprev) reqs[numReqs++] = channels[idNext].irecv(pHprev_new); // receive pointer
					boost::mpi::wait_all(reqs, reqs+numReqs);

					// update my pointers in case pointers were exchanged
					if (exchangePointersH) H = mpi2::intToPointer<DenseMatrixCM>(pH_new);
					if (exchangePointersHprev) Hprev = mpi2::intToPointer<DenseMatrixCM>(pHprev_new);
				}
			}
			mpi2::logEndEvent("communication");

			// run the SGD
			mpi2::logBeginEvent("computation");
			FactorizationData<> jobData(*bV, *bW, *H, job.nnz1(), job.dv.blockOffset1(b1),
					job.nnz2(), job.dv.blockOffset2(b2),job.nnz12max);
			SgdJob<Update,Regularize> sgdJob(jobData, job.update, job.regularize, job.order);
			double epsRegularize = job.regularize.rescaleStratumStepsize() ? eps/d : eps;
			runner.epoch(sgdJob, eps, epsRegularize); // regularize called d times per row/column block!
			mpi2::logEndEvent("computation");

			// store H back
			if (job.mapReduce) {
				// store H in every epoch
				mpi2::logBeginEvent("communication");
				mpi2::RemoteVar vH = job.dh.block(0,b2);
				vH.setCopy(*H);
				mpi2::logEndEvent("communication");
			} else {
				// store H in last epoch
				if (subepoch == d-1 && ch.world().size()>1) {
					mpi2::logBeginEvent("communication");
					mpi2::RemoteVar vH = job.dh.block(0,b2);
					vH.setCopy(*H);
					mpi2::logEndEvent("communication");
				}
			}

			// signal that subepoch is done
			LOG4CXX_DEBUG(detail::logger, id << ": "
					<< "Finished subepoch " << subepoch);

			// wait for go to next subepoch (for single-node and MapReduce version only)
			if (ch.world().size() == 1 || job.mapReduce) {
				mpi2::logBeginEvent("barrier");
				barrier(channels);
				mpi2::logEndEvent("barrier");
			}

			mpi2::logEndEvent("subepoch");
		}

		// signal done
		if (ch.world().size() > 1) {
			delete H;
			delete Hprev;
		}
		ch.send();
	}
};
}

namespace detail {
	/**
	 *
	 * @param w world size
	 * @param t tasks per rank
	 * @param stratumOrder
	 * @param random
	 * @return
	 */
	inline boost::numeric::ublas::matrix<mf_size_type> computeDsgdSchedule(mf_size_type w, mf_size_type t,
			StratumOrder stratumOrder, rg::Random32& random) {
		mf_size_type d = w*t;
		boost::numeric::ublas::matrix<mf_size_type> schedule(d, d);
		for (mf_size_type subepoch=0; subepoch<d; subepoch++) {
			mf_size_type s = subepoch;
			for (mf_size_type id=0; id<d; id++) {
				schedule(subepoch,id) = (s + id) % d;
			}
		}
		switch (stratumOrder) {
		case STRATUM_ORDER_SEQ:
			// leave as is
			break;
		case STRATUM_ORDER_RSEQ:
		{
			// randomize blocks
			std::vector<mf_size_type> blocks(d);
			for (mf_size_type b=0; b<d; b++) blocks[b] = b;
			rg::shuffle(blocks.begin(), blocks.end(), random);

			// update schedule
			for (mf_size_type subepoch=0; subepoch<d; subepoch++) {
				for (mf_size_type id=0; id<d; id++) {
					schedule(subepoch, id) = blocks[ schedule(subepoch,id) ];
				}
			}
		}
			break;
		case STRATUM_ORDER_WR:
			// shuffle each row
			for (mf_size_type subepoch=0; subepoch<d; subepoch++) {
				rg::shuffle(
						schedule.data().begin() + subepoch*d,
						schedule.data().begin() + (subepoch+1)*d,
						random);
			}
			break;
		case STRATUM_ORDER_WOR:
			// shuffle rows and columns
			mf::shuffle(schedule, random);
			break;
		case STRATUM_ORDER_COWOR:
		{
			// blocks will be grouped into sets of t blocks
			// the grouping is determined by the blocks vector computed below (the first t blocks,
			// the second t blocks, ...)
			std::vector<mf_size_type> blocks(d);
			for (mf_size_type b=0; b<d; b++) blocks[b] = b;
			rg::shuffle(blocks.begin(), blocks.end(), random);

			// get DSGD WOR schedule for the groups
			boost::numeric::ublas::matrix<mf_size_type> groupSchedule(w, w);
			groupSchedule = computeDsgdSchedule(w, 1, STRATUM_ORDER_WOR, random);

			// compute the final schedule
			for (mf_size_type bi=0; bi<w; bi++) {
				mf_size_type biOffset = bi*t;
				for (mf_size_type bj=0; bj<w; bj++) {
					mf_size_type bjOffset = bj*t;
					mf_size_type groupOffset = groupSchedule(bi, bj)*t;

					// get DSGD schedule within group
					boost::numeric::ublas::matrix<mf_size_type> localSchedule = computeDsgdSchedule(t, 1, STRATUM_ORDER_WOR, random);
					for (mf_size_type ti=0; ti<t; ti++) {
						for (mf_size_type tj=0; tj<t; tj++) {
							mf_size_type block = blocks[ groupOffset + localSchedule(ti,tj) ];
							schedule(biOffset + ti, bjOffset+tj) = block;
						}
					}
				}
			}
		}
			break;
		default:
			RG_THROW(rg::InvalidArgumentException, rg::paste("Invalid stratum order: ", stratumOrder));
		}

		return schedule;
	}
}

template<typename Update, typename Regularize>
void DsgdRunner::epoch(DsgdJob<Update, Regularize>& job, double eps) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	unsigned worldSize = world.size();
	unsigned tasksPerRank = job.tasksPerRank;
	unsigned d = worldSize * tasksPerRank;

	// compute schedule matrix (row = subepoch, columns = blocks of H, entry = group id who runs it)
	boost::numeric::ublas::matrix<mf_size_type> schedule = detail::computeDsgdSchedule(worldSize, tasksPerRank, job.stratumOrder, random_);
	LOG4CXX_DEBUG(detail::logger, "Schedule: " << schedule);

	// fire up the tasks
	std::vector<mpi2::Channel> channels(worldSize*tasksPerRank, mpi2::UNINITIALIZED);
	tm.spawnAll<detail::DsgdTask<Update, Regularize> >(tasksPerRank, channels, true);

	// send necessary data to each task
	mpi2::seed(channels, random_);
	mpi2::sendAll(channels, mpi2::marshal(job, eps, schedule));

	// wait for completion (use same polling delay as task manager)
	mpi2::economicRecvAll(channels, tm.pollDelay());
}


}


