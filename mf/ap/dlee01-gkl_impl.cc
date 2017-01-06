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
#include <mpi2/mpi2.h>

#include <util/evaluation.h>

#include <mf/id.h>
#include <mf/ap/apupdate.h>
#include <mf/ap/dlee01-gkl.h>
#include <mf/logger.h>
#include <mf/loss/loss.h>
#include <mf/matrix/op/unblock.h>

#include <mf/loss/gkl.h>

namespace mf {

namespace detail {
	// from lee01-gkl_impl.cc
	void lee01Gkl_h(const SparseMatrixCM &v, const DenseMatrix& w, DenseMatrixCM& h);
	void lee01Gkl_w(const SparseMatrix &v, DenseMatrix& w, const DenseMatrixCM& h);

	void lee01Gkl_h(const SparseMatrixCM &v, const DenseMatrix& w, DenseMatrixCM& h, const Empty& data,
			mf_size_type b1, mf_size_type b2) {
		lee01Gkl_h(v, w, h);
	}

	void lee01Gkl_w(const SparseMatrix &v, DenseMatrix& w, const DenseMatrixCM& h, const Empty& data,
			mf_size_type b1, mf_size_type b2) {
		lee01Gkl_w(v, w, h);
	}

	typedef ApUpdateW<Empty, lee01Gkl_w, ID_DLEE01GKL> Dlee01GklW;
	typedef ApUpdateH<Empty, lee01Gkl_h, ID_DLEE01GKL> Dlee01GklH;
}


void dlee01Gkl(DapFactorizationData<>& data, unsigned epochs, Trace& trace) {
	using namespace boost::numeric::ublas;
	LOG4CXX_INFO(detail::logger, "Starting distributed Lee (2001) algorithm for GKL");

	// initialize
	double currentLoss=0;
	double timeLoss=0;
	const std::string wUnblockedName = data.dw.name() + "_unblocked_gkl";
	mpi2::createCopyAll(wUnblockedName, DapFactorizationData<>::W(0,0));
	const std::string hUnblockedName = data.dh.name() + "_unblocked_gkl";
	mpi2::createCopyAll(hUnblockedName, DapFactorizationData<>::H(0,0));
	boost::numeric::ublas::matrix<double> result;

	rg::Timer t;
	t.start();
	unblockAll(data.dh, hUnblockedName);
	currentLoss = gkl(data.dv, data.dw, data.dh, hUnblockedName, data.tasksPerRank);
	t.stop();
	timeLoss=t.elapsedTime().nanos();
	LOG4CXX_INFO(mf::detail::logger, "Loss: " << currentLoss << " (" << t << ")");

	trace.clear();
	trace.add(new TraceEntry(currentLoss, timeLoss));

	// main loop
	for (mf_size_type epoch=0; epoch<epochs; epoch++) {
		// run epoch
		t.start();
		if (epoch % 2 == 0) {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating W)");
			unblockAll(data.dh, hUnblockedName);
			LOG4CXX_INFO(mf::detail::logger, "Unblocked H at all ranks");
			runTaskOnBlocks<SparseMatrix,double,detail::Dlee01GklW::Arg>(
					data.dv, result,
					boost::bind(detail::Dlee01GklW::constructArg, _1, _2, _3, boost::cref(data.dw), boost::cref(hUnblockedName), (detail::Empty())),
					detail::Dlee01GklW::id(), data.tasksPerRank);
		} else {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating H)");
			unblockAll(data.dw, wUnblockedName);
			LOG4CXX_INFO(mf::detail::logger, "Unblocked W at all ranks");
			runTaskOnBlocks<SparseMatrixCM,double,detail::Dlee01GklH::Arg>(
					*data.dvc, result,
					boost::bind(detail::Dlee01GklH::constructArg, _1, _2, _3, boost::cref(wUnblockedName), boost::cref(data.dh), (detail::Empty())),
					detail::Dlee01GklH::id(), data.tasksPerRank);
		}
		t.stop();
		double timeEpoch = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Finished epoch " << (epoch+1) << " (" << t << ")");

		// compute loss
		t.start();
		unblockAll(data.dh, hUnblockedName);
		currentLoss = gkl(data.dv, data.dw, data.dh, hUnblockedName, data.tasksPerRank);
		t.stop();
		timeLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Loss: " << currentLoss << " (" << t << ")");

		// update trace
		trace.add(new TraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeEpoch, timeLoss));
	}

	mpi2::eraseAll<DapFactorizationData<>::W>(wUnblockedName);
	mpi2::eraseAll<DapFactorizationData<>::H>(hUnblockedName);
	LOG4CXX_INFO(detail::logger, "Finished distributed Lee (2001) algorithm for GKL");
}

namespace detail {
	void dlee01GklRegisterTasks() {
		mpi2::registerTask<mf::detail::Dlee01GklW>();
		mpi2::registerTask<mf::detail::Dlee01GklH>();
	}
}


}
