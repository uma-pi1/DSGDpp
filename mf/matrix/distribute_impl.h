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
#include <queue>

#include <mf/matrix/distribute.h> // IDE hint

namespace mf {

namespace detail {

	/** Assigns blocks of a distributed matrix to a set of tasks. The method assumes that tasksPerRank
	 * tasks are spawned on each rank; it determines which of these tasks process which
	 * matrix blocks. The assignment satisfies: (1) each block is assigned to
	 * exactly one task, (2) tasks get assigned only local blocks, (3) the number of blocks assigned
	 * to each task is as balanced as possible.
	 *
	 * @param blocks a blocks1 x blocks2 matrix containing a remote variable storing each block
	 * @param ranks number of ranks (nodes) in the cluster
	 * @param tasksPerRank number of tasks per rank (>=1)
	 * @param[out] groupIds a blocks1 x blocks2 matrix to be filled with the group id of the
	 *                      task responsible for processing each block. (As in mpi2, group ids are
	 *                      ordered by rank, i.e., ids r*taskPerRank...(r+1)*tasksPerRank-1 belong to
	 *                      rank r.)
	 */
	void assignBlocksToTasks(
			const boost::numeric::ublas::matrix<mpi2::RemoteVar>& blocks,
			int ranks, int tasksPerRank,
			boost::numeric::ublas::matrix<int>& groupIds
	);

	/** Constructs the data sent to the tasks assigned by mf::scheduleLocalTaskPerBlock. Each task
	 * is sent a vector with one "argument" of type A per block to process. The argument usually
	 * contains information about the block (row/column, location), but may also contain other
	 * information. This method takes as input a function that constructs the argument
	 * for block (b1,b2) from b1, b2, and the block's location (a remote variable).
	 *
	 * @param blocks a blocks1 x blocks2 matrix containing a remote variable storing each block
	 * @param ranks number of ranks (nodes) in the cluster
	 * @param tasksPerRank number of tasks per rank (>=1)
	 * @param groupIds a blocks1 x blocks2 matrix to be filled with the group id of the
	 *                 task responsible for processing each block
	 * @param construct function for construction arguments; signature A f(mf_size_type b1,
	 *           mf_size_type b2, mpi2::RemoteVar block)
	 * @param[out] args a ranks*tasksPerRanks vector. The element at position i is the vector of
	 *                  arguments to be processed by the task with group id i.
	 */
	template<typename A, typename F>
	void constructTaskData(
			const boost::numeric::ublas::matrix<mpi2::RemoteVar>& blocks,
			int ranks, int tasksPerRank,
			const boost::numeric::ublas::matrix<int>& groupIds,
	//		const boost::function<A (mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block)>& construct,
			F& construct,
			std::vector<std::vector<A> >& args) {
		args.clear();
		args.resize(ranks * tasksPerRank);
		for (mf_size_type b2=0; b2<blocks.size2(); b2++) { // order is important!
			for (mf_size_type b1=0; b1<blocks.size1(); b1++) {
				args[groupIds(b1,b2)].push_back( construct(b1, b2, blocks(b1,b2)) );
			}
		}
	}
}

template<typename M, typename R, typename A, typename F>
void runTaskOnBlocks(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<R>& result,
		const F& construct,
		const std::string& taskId,
		int tasksPerRank, bool asyncRecv, int pollDelay)
{
	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
        if (pollDelay<0) pollDelay = tm.pollDelay();
	boost::mpi::communicator& world = tm.world();
	int worldSize = world.size();

	// spawn tasksPerRank tasks on each node
	std::vector<mpi2::Channel> channels(worldSize * tasksPerRank, mpi2::UNINITIALIZED);
	tm.spawnAll(taskId, tasksPerRank, channels);

	// assign blocks to tasks
	boost::numeric::ublas::matrix<int> groupIds(m.blocks1(), m.blocks2());
	detail::assignBlocksToTasks(m.blocks(), worldSize, tasksPerRank, groupIds);

	// compute task arguments and send
	std::vector<std::vector<A> > args(worldSize * tasksPerRank);
	detail::constructTaskData(m.blocks(), worldSize, tasksPerRank, groupIds, construct, args);
	mpi2::sendEach(channels, args);

	// collect results (sent one by one)
	result.resize(m.blocks1(), m.blocks2(), false);
	boost::numeric::ublas::matrix<boost::mpi::request> recv_reqs(m.blocks1(), m.blocks2());
	if (asyncRecv) {
		for (mf_size_type b2=0; b2<m.blocks2(); b2++) {
			for (mf_size_type b1=0; b1<m.blocks1(); b1++) {
				int index = groupIds(b1, b2);
				recv_reqs(b1,b2) = channels[index].irecv( result(b1,b2) );
			}
		}
		mpi2::economicWaitAll(recv_reqs.data().begin(), recv_reqs.data().end(), pollDelay);
	} else {
		for (mf_size_type b2=0; b2<m.blocks2(); b2++) {
			for (mf_size_type b1=0; b1<m.blocks1(); b1++) {
				int index = groupIds(b1, b2);
				boost::mpi::request req = channels[index].irecv( result(b1,b2) );
				mpi2::economicWait(req, pollDelay);
			}
		}
	}
}

template<typename M1, typename M2, typename M3, typename R>
void runFunctionPerAssignedBlock3(mpi2::Channel ch,
		boost::function<R (const M1&, const M2&, const M3&)> f) {
	// allocate temporary buffers
	M2 *w = new M2(0,0);
	mpi2::RemoteVar ww(mpi2::UNINITIALIZED);
	M2 *wNext = new M2(0,0);
	mpi2::RemoteVar wwNext(mpi2::UNINITIALIZED);
	M3 *h = new M3(0,0);
	mpi2::RemoteVar hh(mpi2::UNINITIALIZED);
	M3 *hNext = new M3(0,0);
	mpi2::RemoteVar hhNext(mpi2::UNINITIALIZED);

	// receive work
	std::vector<std::vector<mpi2::RemoteVar> > vars;
	ch.recv(vars);

	// split the work into (vars, index) pairs:
	// localVars: all required data is stored locally
	// remoteVars: some required data is not stored local
	std::vector<std::pair<std::vector<mpi2::RemoteVar>, unsigned> > localVars;
	std::vector<std::pair<std::vector<mpi2::RemoteVar>, unsigned> > remoteVars;
	for (unsigned i=0; i<vars.size(); i++) {
		if (vars[i][1].isLocal() && vars[i][2].isLocal()) {
			localVars.push_back( std::pair<std::vector<mpi2::RemoteVar>, unsigned>(vars[i], i) );
		} else {
			remoteVars.push_back(std::pair<std::vector<mpi2::RemoteVar>, unsigned>(vars[i], i));
		}
	}
	LOG4CXX_TRACE(detail::logger, ch.local() << ": " << localVars.size() << " local blocks, "
			<< remoteVars.size() << " remote blocks");

	// get the list of remote variables that we need to fetch (in the processing order)
	std::queue<mpi2::RemoteVar> fetchWs;
	std::queue<mpi2::RemoteVar> fetchHs;
	for (unsigned iRemote=0; iRemote<remoteVars.size(); iRemote++) {
		wwNext = remoteVars[iRemote].first[1];
		if (!wwNext.isLocal() && wwNext != ww) {
			fetchWs.push(wwNext);
			ww = wwNext;
		}

		hhNext = remoteVars[iRemote].first[2];
		if (!hhNext.isLocal() && hhNext != hh) {
			fetchHs.push(hhNext);
			hh = hhNext;
		}
	}

	// prefetch the next remote W and the next remote H (if any)
	boost::mpi::request wReq;
	if (!fetchWs.empty()) {
		ww = mpi2::RemoteVar(mpi2::UNINITIALIZED);
		wwNext = fetchWs.front(); fetchWs.pop();
		LOG4CXX_TRACE(detail::logger, ch.local() << ": Prefetching " << wwNext);
		wReq = wwNext.igetCopy<M2>(*wNext);
	}
	boost::mpi::request hReq;
	if (!fetchHs.empty()) {
		hh = mpi2::RemoteVar(mpi2::UNINITIALIZED);
		hhNext = fetchHs.front(); fetchHs.pop();
		LOG4CXX_TRACE(detail::logger, ch.local() << ": Prefetching " << hhNext);
		hReq = hhNext.igetCopy<M3>(*hNext);
	}

	// main loop: process each block
	std::vector<boost::mpi::request> reqs(vars.size());
	std::vector<R> results(vars.size());
	unsigned iLocal = 0, iRemote = 0;
	for (; iLocal < localVars.size() && iRemote < remoteVars.size(); ) {
		// first check if we can process a block with remote data
		mpi2::RemoteVar vBlock = remoteVars[iRemote].first[0];
		mpi2::RemoteVar wBlock= remoteVars[iRemote].first[1];
		mpi2::RemoteVar hBlock= remoteVars[iRemote].first[2];

		// check if prefetching of W has finished; if so, prefetch next required block
		if (wBlock == wwNext) {
			if (wReq.test()) {
				std::swap(w, wNext);
				ww = wwNext;
				if (!fetchWs.empty()) {
					wwNext = fetchWs.front(); fetchWs.pop();
					LOG4CXX_TRACE(detail::logger, ch.local() << ": Prefetching " << wwNext);
					wReq = wwNext.igetCopy<M2>(*wNext);
				} else {
					wwNext = mpi2::RemoteVar(mpi2::UNINITIALIZED);
				}
			}
		}

		// check if prefetching of H has finished; if so, prefetch next required block
		if (hBlock == hhNext) {
			if (hReq.test()) {
				std::swap(h, hNext);
				hh = hhNext;
				if (!fetchHs.empty()) {
					hhNext = fetchHs.front(); fetchHs.pop();
					LOG4CXX_TRACE(detail::logger, ch.local() << ": Prefetching " << hhNext);
					hReq = hhNext.igetCopy<M3>(*hNext);
				} else {
					hhNext = mpi2::RemoteVar(mpi2::UNINITIALIZED);
				}
			}
		}

		// check if we have all the data to process the remote block
		if ( (wBlock.isLocal() || wBlock == ww)
				&& (hBlock.isLocal() || hBlock== hh) ) {
                    //mpi2::logBeginEvent("remote");
			unsigned i = remoteVars[iRemote].second;
			results[i] = f(*vBlock.getLocal<M1>(),
						   wBlock.isLocal() ? *wBlock.getLocal<M2>() : *w,
						   hBlock.isLocal() ? *hBlock.getLocal<M3>() : *h);
			iRemote++
                            ;                    //mpi2::logEndEvent("remote");

		} else {
                    //mpi2::logBeginEvent("local");
			// process a local block
			vBlock = localVars[iLocal].first[0];
			wBlock = localVars[iLocal].first[1];
			hBlock = localVars[iLocal].first[2];

			// run the function
			unsigned i = localVars[iLocal].second;
			results[i] = f(*vBlock.getLocal<M1>(),
						   *wBlock.getLocal<M2>(),
						   *hBlock.getLocal<M3>());

			iLocal++;
                        //  mpi2::logEndEvent("local");
		}
	}

	// process remaining local blocks
	for (; iLocal<localVars.size(); iLocal++) {
		mpi2::RemoteVar vBlock = localVars[iLocal].first[0];
		mpi2::RemoteVar wBlock = localVars[iLocal].first[1];
		mpi2::RemoteVar hBlock = localVars[iLocal].first[2];

		// run the function
                //mpi2::logBeginEvent("local");

		unsigned i = localVars[iLocal].second;
		results[i] = f(*vBlock.getLocal<M1>(),
					   *wBlock.getLocal<M2>(),
					   *hBlock.getLocal<M3>());
                //  mpi2::logEndEvent("local");
	}

	// process remaining remote blocks
	for (; iRemote<remoteVars.size(); iRemote++) {
		mpi2::RemoteVar vBlock = remoteVars[iRemote].first[0];
		mpi2::RemoteVar wBlock= remoteVars[iRemote].first[1];
		mpi2::RemoteVar hBlock= remoteVars[iRemote].first[2];

		// finish prefetching of W (wait)
		if (wBlock == wwNext) {
                    mpi2::economicWait(wReq, mpi2::TaskManager::getInstance().pollDelay());
			std::swap(w, wNext);
			ww = wwNext;
			if (!fetchWs.empty()) {
				wwNext = fetchWs.front(); fetchWs.pop();
				LOG4CXX_TRACE(detail::logger, ch.local() << ": Prefetching " << wwNext);
				wReq = wwNext.igetCopy<M2>(*wNext);
			} else {
				wwNext = mpi2::RemoteVar(mpi2::UNINITIALIZED);
			}
		}

		// finish prefetching of H (wait)
		if (hBlock == hhNext) {
                    mpi2::economicWait(hReq, mpi2::TaskManager::getInstance().pollDelay());
			std::swap(h, hNext);
			hh = hhNext;
			if (!fetchHs.empty()) {
				hhNext = fetchHs.front(); fetchHs.pop();
				LOG4CXX_TRACE(detail::logger, ch.local() << ": Prefetching " << hhNext);
				hReq = hhNext.igetCopy<M3>(*hNext);
			} else {
				hhNext = mpi2::RemoteVar(mpi2::UNINITIALIZED);
			}
		}

		// process the remote block
                //mpi2::logBeginEvent("remote");

		unsigned i = remoteVars[iRemote].second;
		results[i] = f(*vBlock.getLocal<M1>(),
				wBlock.isLocal() ? *wBlock.getLocal<M2>() : *w,
				hBlock.isLocal() ? *hBlock.getLocal<M3>() : *h);
                //mpi2::logEndEvent("remote");

	}

	// clean up
        for (int i=0; i<results.size(); i++) {
            reqs[i] = ch.isend( results[i] );
        }
        mpi2::economicWaitAll(reqs, mpi2::TaskManager::getInstance().pollDelay());
	delete w;
	delete wNext;
	delete h;
	delete hNext;
}

extern template void runFunctionPerAssignedBlock3<SparseMatrix, DenseMatrix, DenseMatrixCM, double>(
		mpi2::Channel ch, boost::function<double (const SparseMatrix&, const DenseMatrix&, const DenseMatrixCM&)> f);

}
