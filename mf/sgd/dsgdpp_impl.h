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

#include <mf/sgd/dsgdpp.h> // help for compilers

#include <mf/matrix/op/shuffle.h>

namespace mf {

template<typename Update, typename Regularize, typename DistributedLoss,
	typename DistributedAdaptiveDecay,typename TestData,typename TestLoss>
void DsgdPpRunner::run(DsgdPpJob<Update, Regularize>& job, DistributedLoss& loss,
		mf_size_type epochs, DistributedAdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod,
		TestData* testData, TestLoss *testLoss) {
	LOG4CXX_INFO(detail::logger, "Starting DSGD++ (polling delay: " << mpi2::TaskManager::getInstance().pollDelay() << " microseconds)");

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

	detail::defaultRunner(job, loss, epochs, decay,
			boost::bind(&DsgdPpRunner::epoch<Update,Regularize>, this, _1, _2),
			trace, random_, balanceType, balanceMethod, testData, testLoss);

	LOG4CXX_INFO(detail::logger, "Finished DSGD++");
}

template<typename Update, typename Regularize, typename DistributedLoss,
	typename DistributedAdaptiveDecay>
void DsgdPpRunner::run(DsgdPpJob<Update, Regularize>& job, DistributedLoss& loss,
		mf_size_type epochs, DistributedAdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod) {
	run(job, loss, epochs, decay, trace, balanceType, balanceMethod, (DsgdPpFactorizationData<>*)NULL, (NoLoss*)NULL);
}

namespace detail {

template<typename Update, typename Regularize>
struct DsgdPpTask {
	static const std::string id() { return std::string("__mf/sgd/DsgdPpTask_")
			+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		rg::Random32 random = mpi2::getSeed(ch);

		// receive task tags and figure out my id
		std::vector<mpi2::Channel>& channels = info.pairwiseChannels();
		int id = info.groupId();
		const mf_size_type d = info.groupSize();

		// receive data descriptor
		DsgdPpJob<Update,Regularize> job(mpi2::UNINITIALIZED);
		double eps;
		boost::numeric::ublas::matrix<mf_size_type> schedule(2*d, d);
		ch.recv(*mpi2::unmarshal(job, eps, schedule));
//		LOG4CXX_DEBUG(detail::logger, id << ": schedule=" << schedule);

		// initialize
		DenseMatrixCM *Hnext  = NULL;    // next block to work on
		DenseMatrixCM *H = NULL;         // block being worked on
		DenseMatrixCM *Hprev  = NULL;    // block worked on previously
		boost::mpi::request HnextReq;    // MPI request for receiving next block to process
		boost::mpi::request HnextPointerReq; // Additional MPI request for receiving next block to process (only when local)
		boost::mpi::request HprevReq;    // MPI request for sending previous block being processed
		boost::mpi::request HprevPointerReq; // MPI request for sending previous block being processed (only when local)
		mpi2::PointerIntType HnextPointer = 0; // pointer to next block (when local, else 0)
		mpi2::PointerIntType HnextPointerOld = 0; // temporary variable used for sending pointer to Hnext
		mpi2::PointerIntType HprevPointer = 0; // pointer to previous block (when local, else 0)
		mpi2::PointerIntType HprevPointerOld = 0; // temporary variable used for sending pointer to Hprev
		if (ch.world().size() > 1) {
			Hnext = new DenseMatrixCM(0,0);
			H =  new DenseMatrixCM(0,0);
			Hprev = new DenseMatrixCM(0,0);
		}

		// run
		SgdRunner runner(random);
		const int FIRST = 0;
		const int SECOND = 1;
		const int LAST = 2*d-1;
		const int LAST_BUT_ONE = 2*d-2;
		for (mf_size_type subepoch = FIRST; subepoch <= LAST; subepoch++) {
			LOG4CXX_DEBUG(detail::logger, id << ": " << "Starting subepoch " << subepoch);
			mpi2::logBeginEvent("subepoch");

			// figure out which blocks to process
			mf_size_type b1 = id;                                    // current row block of W
			mf_size_type b2 = schedule(subepoch, id);                // current col block of H
			mf_size_type b2Prev = -1;                                // previous col block of H
			if (subepoch > FIRST) b2Prev = schedule(subepoch-1, id);
			mf_size_type b2Next = -1;                                // next col block of H
			if (subepoch < LAST) b2Next = schedule(subepoch+1, id);

//			LOG4CXX_DEBUG(detail::logger, id << ": b1=" << b1 << ", b2=" << b2 << ", b2prev=" << b2Prev << ", b2Next=" << b2Next);

			// get W and V
			mpi2::RemoteVar rv = job.dw.block(b1,0); // compiler yells if I don't use a temp...!
			DenseMatrix *bW = rv.getLocal<DenseMatrix>();
			rv = job.dv.block(b1, b2);
			SparseMatrix *bV = rv.getLocal<SparseMatrix>();

			// get H
			if (ch.world().size() == 1) { // single node
				mpi2::RemoteVar vH = job.dh.block(0,b2);
				H = vH.getLocal<DenseMatrixCM>();
			} else if (subepoch == FIRST) { // first epoch: read from env
				// get the current block
				mpi2::logBeginEvent("communication");
				mpi2::RemoteVar vH = job.dh.block(0,b2);
				vH.getCopy(*H); // synchronous
				mpi2::logEndEvent("communication");

				// prefetch next block
//				LOG4CXX_DEBUG(detail::logger, id << ": " << "isend HnextReq: (env)");
				vH = job.dh.block(0,b2Next);
				HnextReq = vH.igetCopy(*Hnext);
				HnextPointer = 0;
			} else { // subsequent epoch: communicate directly
				mpi2::logBeginEvent("communication");

				// finish up sending and receiving Hprev / Hnext (data or pointers)
//				LOG4CXX_DEBUG(detail::logger, id << ": " << "Finish communication Hprev/Hnext");
				std::vector<boost::mpi::request> reqs;
				if (subepoch > SECOND) reqs.push_back(HprevReq);
				reqs.push_back(HnextReq);
				boost::mpi::wait_all(reqs.begin(), reqs.end());

				// if pointers were exchanged (indicated by HprevPointer!=0 and HnextPointer!=0),
				// finish sending our old pointers
//				LOG4CXX_DEBUG(detail::logger, id << ": " << "Finish communication Hprev/Hnext pointers");
				reqs.clear();
				if (subepoch > SECOND && HprevPointer != 0) reqs.push_back(HprevPointerReq);
				if (HnextPointer != 0) reqs.push_back(HnextPointerReq);
				boost::mpi::wait_all(reqs.begin(), reqs.end());

				// if pointers were exchanged, update my pointers
//				LOG4CXX_DEBUG(detail::logger, id << ": " << "All communication finished");
				if (subepoch > SECOND && HprevPointer != 0) Hprev = mpi2::intToPointer<DenseMatrixCM>(HprevPointer);
				if (HnextPointer != 0) Hnext = mpi2::intToPointer<DenseMatrixCM>(HnextPointer);

				mpi2::logEndEvent("communication");

				// update Hprev, H, Hnext
				std::swap(Hprev, H);
				std::swap(H, Hnext);

				// send/receive previous / next blocksend block that has been processed in previous subepoch to next node (Hprev)
				if (subepoch < LAST) {
					// there are more subepochs
					mf_size_type idPrev; // id of task who gets Hprev
					for (idPrev=0; idPrev<d; idPrev++) {
						if (b2Prev == schedule(subepoch+1,idPrev)) break;
					}
					mf_size_type idNext; // id of task who has Hnext
					for (idNext=0; idNext<d; idNext++) {
						if (b2Next == schedule(subepoch-1,idNext)) break;
					}
//					LOG4CXX_DEBUG(detail::logger, id << ": idPrev=" << idPrev << ", idNext=" << idNext);


					// (1) receive HprevPointer
					if (channels[idPrev].remote().rank == channels[idPrev].local().rank) {
//						LOG4CXX_DEBUG(detail::logger, id << ": " << "irecv HprevReq (pointer): " << channels[idPrev]);
						HprevReq = channels[idPrev].irecv(HprevPointer); // receive pointer
					}

					// (2) receive Hnext / HnextPointer
					if (channels[idNext].remote().rank == channels[idNext].local().rank) {
//						LOG4CXX_DEBUG(detail::logger, id << ": " << "irecv HnextReq (pointer)" << channels[idNext]);
						HnextReq = channels[idNext].irecv(HnextPointer); // receive pointer
					} else {
//						LOG4CXX_DEBUG(detail::logger, id << ": " << "irecv HnextReq (data)" << channels[idNext]);
						HnextReq = channels[idNext].irecv(*Hnext);     // recv data
						HnextPointer = 0; // mark that we did not exchange pointers
					}

					// (1) send HnextPointerOld
					if (channels[idNext].remote().rank == channels[idNext].local().rank) {
//						LOG4CXX_DEBUG(detail::logger, id << ": " << "isend HnextPointerReq: " << channels[idNext]);
						HnextPointerOld = mpi2::pointerToInt(Hnext);
						HnextPointerReq = channels[idNext].isend(HnextPointerOld); // send pointer
					}

					// (2) send Hprev / HprevPointerOld
					if (channels[idPrev].remote().rank == channels[idPrev].local().rank) {
//						LOG4CXX_DEBUG(detail::logger, id << ": " << "isend HprevPointerReq: " << channels[idPrev]);
						HprevPointerOld = mpi2::pointerToInt(Hprev);
						HprevPointerReq = channels[idPrev].isend(HprevPointerOld); // send pointer
					} else {
//						LOG4CXX_DEBUG(detail::logger, id << ": " << "isend HprevReq: " << channels[idPrev]);
						HprevReq = channels[idPrev].isend(*Hprev);     // send data
						HprevPointer = 0; // mark that we did not exchange pointers
					}
				} else {
					// store H
					mpi2::RemoteVar vH = job.dh.block(0,b2Prev);
					HprevReq = vH.isetCopy(*Hprev);
				}
			}

			// run the SGD
			mpi2::logBeginEvent("computation");
			FactorizationData<> jobData(*bV, *bW, *H, job.nnz1(), job.dv.blockOffset1(b1),
					job.nnz2(), job.dv.blockOffset2(b2),job.nnz12max);
			SgdJob<Update,Regularize> sgdJob(jobData, job.update, job.regularize, job.order);
			double epsRegularize = job.regularize.rescaleStratumStepsize() ? eps/d : eps;
			// TODO: regularization may not work here; DON'T USE
			runner.epoch(sgdJob, eps, epsRegularize); // regularize called d times per row/column block!
			mpi2::logEndEvent("computation");

			// store H back in last two subepoch
			if (subepoch == LAST && ch.world().size()>1) {
				mpi2::logBeginEvent("communication");
				HprevReq.wait();
				mpi2::RemoteVar vH = job.dh.block(0,b2);
				vH.setCopy(*H); // synchronous
				mpi2::logEndEvent("communication");
			}

			// signal that subepoch is done & wait for go to next subepoch
			LOG4CXX_DEBUG(detail::logger, id << ": " << "Finished subepoch " << subepoch);

			// barrier needed when running on one node
			mpi2::logBeginEvent("barrier");
			if (ch.world().size() == 1) {
				barrier(channels);
			}
			mpi2::logEndEvent("barrier");

			// we are done with the subepoch
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

} // detail

template<typename Update, typename Regularize>
void DsgdPpRunner::epoch(DsgdPpJob<Update, Regularize>& job, double eps) {
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	unsigned worldSize = world.size();
	unsigned tasksPerRank = job.tasksPerRank;
	unsigned d = worldSize * tasksPerRank;

	// decide which block goes to which even / odd subepochs
	std::vector<mf_size_type> blocks0(d), blocks1(d);
	switch (job.stratumOrder) {
	case STRATUM_ORDER_SEQ:
		// even subepochs: even blocks
		// odd subepochs: odd blocks
		for (mf_size_type block=0; block<d; block++) {
			blocks0[block] = 2*block;
			blocks1[block] = 2*block+1;
		}
		break;
	case STRATUM_ORDER_RSEQ:
	case STRATUM_ORDER_WR:
	case STRATUM_ORDER_WOR:
	case STRATUM_ORDER_COWOR:
	{
		// even subepochs: random set of d blocks
		// odd subepochs: the remaining blocks
		blocks0 = rg::sample<mf_size_type>(random_, d, 2*d); // sorted order
		int i0 = 0;
		int i1 = 0;
		for (mf_size_type block=0; i1<d && block<2*d; block++) {
			if (i0<d && blocks0[i0]==block) {
				i0++;
				continue;
			}
			blocks1[i1] = block;
			i1++;
		}
		BOOST_ASSERT( i1 == d );
	}
		break;
	default:
		RG_THROW(rg::InvalidArgumentException, rg::paste("Invalid stratum order: ", job.stratumOrder));
	}

	// compute schedule matrix (row = subepoch, columns = blocks of H, entry = group id who runs it)
	// the schedule matrix has dimensions 2d x d; neighboring rows are guaranteed to not share entries
	boost::numeric::ublas::matrix<mf_size_type> schedule0 = detail::computeDsgdSchedule(worldSize, tasksPerRank, job.stratumOrder, random_);
	boost::numeric::ublas::matrix<mf_size_type> schedule1 = detail::computeDsgdSchedule(worldSize, tasksPerRank, job.stratumOrder, random_);

	boost::numeric::ublas::matrix<mf_size_type> schedule(2*d,d);
	for (int i=0; i<d; i++) {
		for (int id=0; id<d; id++) {
			schedule(2*i, id) = blocks0[ schedule0(i, id) ];
			schedule(2*i+1, id) = blocks1[ schedule1(i, id) ];
		}
	}
	LOG4CXX_DEBUG(detail::logger, "schedule=" << schedule);

	// fire up the tasks
	std::vector<mpi2::Channel> channels(worldSize*tasksPerRank, mpi2::UNINITIALIZED);
	tm.spawnAll<detail::DsgdPpTask<Update, Regularize> >(tasksPerRank, channels, true);

	// send necessary data to each task
	mpi2::seed(channels, random_);
	mpi2::sendAll(channels, mpi2::marshal(job, eps, schedule));

	// wait for completion (use same polling delay as task manager)
	mpi2::economicRecvAll(channels, tm.pollDelay());
}

} // mf



