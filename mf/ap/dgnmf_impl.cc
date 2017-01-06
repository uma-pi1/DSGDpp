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
#include <mf/ap/dgnmf.h>
#include <mf/logger.h>
#include <mf/loss/nzsl.h>
#include <mf/loss/sl.h>
#include <mf/matrix/op/unblock.h>

//#include <mf/matrix/op/sl.h>

namespace mf {

namespace detail {

	void gnmf_h(const SparseMatrixCM& v, const DenseMatrix& w, DenseMatrixCM& h);

	void gnmf_w(const SparseMatrix& v, DenseMatrix& w, const DenseMatrixCM& h);

	void gnmf_h(const SparseMatrixCM& v, const DenseMatrix& w, DenseMatrixCM& h, const Empty& data,
			mf_size_type b1, mf_size_type b2) {
		gnmf_h(v, w, h);
	}

	void gnmf_w(const SparseMatrix& v, DenseMatrix& w, const DenseMatrixCM& h, const Empty& data,
			mf_size_type b1, mf_size_type b2) {
		gnmf_w(v, w, h);
	}

	typedef ApUpdateW<Empty, gnmf_w, ID_DGNMF> DgnmfW;
	typedef ApUpdateH<Empty, gnmf_h, ID_DGNMF> DgnmfH;
}

void dgnmf(DapFactorizationData<>& data, unsigned epochs, Trace& trace, BalanceType type,
		BalanceMethod method, DsgdFactorizationData<>* testData) {
	using namespace boost::numeric::ublas;
	LOG4CXX_INFO(detail::logger, "Starting DGNMF ("
			<< "for squared loss and)");

	// initialize
	NzslLoss testLoss;
	double currentLoss=0;
	double timeLoss=0;
	const std::string wUnblockedName = data.dw.name() + "_unblocked_gnmf";
	mpi2::createCopyAll(wUnblockedName, DapFactorizationData<>::W(0,0));
	const std::string hUnblockedName = data.dh.name() + "_unblocked_gnmf";
	mpi2::createCopyAll(hUnblockedName, DapFactorizationData<>::H(0,0));
	boost::numeric::ublas::matrix<double> result;

	// compute initial loss
	rg::Timer t;
	t.start();
	unblockAll(data.dh, hUnblockedName);
	currentLoss = sl(data.dv, data.dw, data.dh, hUnblockedName, data.tasksPerRank);
	t.stop();
	timeLoss=t.elapsedTime().nanos();
	LOG4CXX_INFO(mf::detail::logger, "Loss: " << currentLoss << " (" << t << ")");

	// compute test loss and create a trace entry
	trace.clear();
	TraceEntry* entry;
	double currentTestLoss=0.0;
	double timeTestLoss=0.0;
	if (testData!=NULL){
		t.start();
		currentTestLoss = testLoss(*testData);
		t.stop();
		timeTestLoss=t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
		entry=new TraceEntry(currentLoss, timeLoss, currentTestLoss, timeTestLoss);
	}
	else{
		entry=new TraceEntry(currentLoss, timeLoss);

	}
	trace.add(entry);

	// main loop
	for (mf_size_type epoch=0; epoch<epochs; epoch++) {

		double timeBalancing = 0;

		// run epoch
		t.start();
		if (epoch % 2 == 0) {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating H)");
			unblockAll(data.dw, wUnblockedName);
			LOG4CXX_INFO(mf::detail::logger, "Unblocked W at all ranks");
			runTaskOnBlocks<SparseMatrixCM,double,detail::DgnmfH::Arg>(
					*data.dvc, result,
					boost::bind(detail::DgnmfH::constructArg, _1, _2, _3, boost::cref(wUnblockedName), boost::cref(data.dh), (detail::Empty())),
					detail::DgnmfH::id(), data.tasksPerRank);
		} else {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating W)");
			unblockAll(data.dh, hUnblockedName);
			LOG4CXX_INFO(mf::detail::logger, "Unblocked H at all ranks");
			runTaskOnBlocks<SparseMatrix,double,detail::DgnmfW::Arg>(
					data.dv, result,
					boost::bind(detail::DgnmfW::constructArg, _1, _2, _3, boost::cref(data.dw), boost::cref(hUnblockedName), (detail::Empty())),
					detail::DgnmfW::id(), data.tasksPerRank);
		}
		t.stop();
		double timeEpoch = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Finished epoch " << (epoch+1) << " (" << t << ")");

		t.start();
		balance(data, type, method);
		t.stop();
		timeBalancing=t.elapsedTime().nanos();

		// compute loss
		t.start();
		double currentLoss;
		unblockAll(data.dh, hUnblockedName);
		currentLoss = sl(data.dv, data.dw, data.dh, hUnblockedName, data.tasksPerRank);
		t.stop();
		timeLoss=t.elapsedTime().nanos();
		LOG4CXX_INFO(mf::detail::logger, "Loss: " << currentLoss << " (" << t << ")");

		// compute test loss and create a trace entry
		TraceEntry* entry;
		currentTestLoss=0.0;
		timeTestLoss=0.0;
		if (testData!=0){
			t.start();
			currentTestLoss = testLoss(*testData);
			t.stop();
			timeTestLoss=t.elapsedTime().nanos();
			LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
			// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
			// currrently the time for balancing is not in the trace
			entry=new TraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, timeEpoch, currentTestLoss, timeTestLoss);
		}
		else{
			// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
			// currrently the time for balancing is not in the trace
			entry=new TraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, timeEpoch);
		}
		trace.add(entry);

	}

	mpi2::eraseAll<DapFactorizationData<>::W>(wUnblockedName);
	mpi2::eraseAll<DapFactorizationData<>::H>(hUnblockedName);
	LOG4CXX_INFO(detail::logger, "Finished DGNMF ("
			<< "for nonzero squared loss)");
}

namespace detail {
	void dgnmfRegisterTasks() {
		mpi2::registerTask<mf::detail::DgnmfW>();
		mpi2::registerTask<mf::detail::DgnmfH>();
	}
}

}
