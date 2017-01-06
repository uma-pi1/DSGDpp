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
#include <mf/ap/dals.h>
#include <mf/lapack/lapack_wrapper.h>
#include <mf/logger.h>
#include <mf/loss/loss.h>
#include <mf/loss/l2.h>
#include <mf/loss/nzsl.h>
#include <mf/loss/nzl2.h>
#include <mf/matrix/op/scale.h>
#include <mf/matrix/op/sums.h>
#include <mf/matrix/op/unblock.h>

namespace mf {

namespace detail {
	void alsNzsl_w(const SparseMatrix& v, DenseMatrix& w, const DenseMatrixCM& h,
			const std::vector<mf_size_type>& nnz1, mf_size_type nnz1offset,
			double lambda, AlsRegularizer regularizer);

	void alsNzsl_h(const SparseMatrixCM& vc, const DenseMatrix& w, DenseMatrixCM& h,
			const std::vector<mf_size_type>& nnz2, mf_size_type nnz2offset,
			double lambda, AlsRegularizer regularizer);

	struct DalsData {
		DalsData() { }
		DalsData(double lambda, AlsRegularizer regularizer, const std::string& nnzName,
				const std::vector<mf_size_type>& nnzOffsets)
		: lambda(lambda), regularizer(regularizer), nnzName(nnzName), nnzOffsets(nnzOffsets) { }

		double lambda;
		AlsRegularizer regularizer;
		std::string nnzName;
		std::vector<mf_size_type> nnzOffsets;

		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & lambda;
			ar & regularizer;
			ar & nnzName;
			ar & nnzOffsets;
		}
	};

	void alsNzsl_w(const SparseMatrix& v, DenseMatrix& w, const DenseMatrixCM& h,
			const DalsData& data, mf_size_type b1, mf_size_type b2) {
		alsNzsl_w(v, w, h, *mpi2::env().get<std::vector<mf_size_type> >(data.nnzName),
				data.nnzOffsets[b1], data.lambda, data.regularizer);
	}

	void alsNzsl_h(const SparseMatrixCM& vc, const DenseMatrix& w, DenseMatrixCM& h,
			const DalsData& data, mf_size_type b1, mf_size_type b2) {
		alsNzsl_h(vc, w, h, *mpi2::env().get<std::vector<mf_size_type> >(data.nnzName),
				data.nnzOffsets[b2], data.lambda, data.regularizer);
	}


	typedef ApUpdateW<DalsData, alsNzsl_w, ID_DALS> DalsW;
	typedef ApUpdateH<DalsData, alsNzsl_h, ID_DALS> DalsH;
}

void dalsNzsl(DapFactorizationData<>& data, unsigned epochs, Trace& trace,
		double lambda, AlsRegularizer regularizer, BalanceType type, BalanceMethod method,
		DsgdFactorizationData<>* testData) {

	BOOST_ASSERT(data.dvc != NULL);
	NzslLoss testLoss; // used only if testData are provided

	using namespace boost::numeric::ublas;
	LOG4CXX_INFO(detail::logger, "Starting DALS ("
			<< "with balance type : "
			<< (type == BALANCE_NONE ? "None" : "")
			<< (type == BALANCE_L2 ? "L2" : "")
			<< (type == BALANCE_NZL2 ? "Nzl2" : "")
			<< " and balance method : "
			<< (method == BALANCE_SIMPLE ? "Simple" : "")
			<< (method == BALANCE_OPTIMAL ? "Optimal" : "")
			<< "for nonzero squared loss and "
			<< (regularizer == ALS_L2 ? "L2" : "NZL2") << "(" << lambda << ")");

	// initialize
	NzslLoss test;
	double currentLoss = 0;
	double timeLoss = 0;
	const std::string wUnblockedName = data.dw.name() + "_unblocked_dals";
	mpi2::createCopyAll(wUnblockedName, DapFactorizationData<>::W(0,0));
	const std::string hUnblockedName = data.dh.name() + "_unblocked_dals";
	mpi2::createCopyAll(hUnblockedName, DapFactorizationData<>::H(0,0));
	boost::numeric::ublas::matrix<double> result;
	detail::DalsData dataW(lambda, regularizer, data.nnz1name, data.dv.blockOffsets1());
	detail::DalsData dataH(lambda, regularizer, data.nnz2name, data.dvc->blockOffsets2());

	// compute initial loss
	rg::Timer t;
	t.start();
	unblockAll(data.dh, hUnblockedName);
	currentLoss = nzsl(data.dv, data.dw, hUnblockedName, data.tasksPerRank);
	if (lambda > 0) {
		if (regularizer == ALS_L2) {
			currentLoss += lambda*(l2(data.dw) + l2(data.dh));
		} else {
			currentLoss += lambda*(nzl2(data.dw,data.nnz1name,data.tasksPerRank)+nzl2(data.dh,data.nnz2name,data.tasksPerRank));
		}
	}
	t.stop();
	timeLoss=t.elapsedTime().nanos();
	LOG4CXX_INFO(mf::detail::logger, "Loss: " << currentLoss << " (" << t << ")");

	// compute test loss and create a trace entry
	trace.clear();
	AlsTraceEntry* entry;
	double currentTestLoss = 0;
	double timeTestLoss = 0;
	if (testData!=NULL) {
		t.start();
		currentTestLoss = testLoss(*testData);
		t.stop();
		timeTestLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
		entry = new AlsTraceEntry(currentLoss, timeLoss, currentTestLoss, timeTestLoss);
	}
	else{
		entry = new AlsTraceEntry(currentLoss, timeLoss);
	}
	trace.add(entry);

	// main loop
	for (mf_size_type epoch=0; epoch<epochs; epoch++) {
		boost::numeric::ublas::vector<double> factor;
		double timeRescaling = 0;

		// run epoch
		t.start();
		if (epoch % 2 == 0) {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating W)");
			unblockAll(data.dh, hUnblockedName);
			LOG4CXX_INFO(mf::detail::logger, "Unblocked H at all ranks");
			runTaskOnBlocks<SparseMatrix,double,detail::DalsW::Arg>(
					data.dv, result,
					boost::bind(detail::DalsW::constructArg, _1, _2, _3, boost::cref(data.dw), boost::cref(hUnblockedName), dataW),
					detail::DalsW::id(), data.tasksPerRank);
		} else {
			LOG4CXX_INFO(mf::detail::logger, "Starting epoch " << (epoch+1) << " (updating H)");
			unblockAll(data.dw, wUnblockedName);
			LOG4CXX_INFO(mf::detail::logger, "Unblocked W at all ranks");
			runTaskOnBlocks<SparseMatrixCM,double,detail::DalsH::Arg>(
					*data.dvc, result,
					boost::bind(detail::DalsH::constructArg, _1, _2, _3, boost::cref(wUnblockedName), boost::cref(data.dh), dataH),
					detail::DalsH::id(), data.tasksPerRank);
		}
		t.stop();
		double timeEpoch = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Finished epoch " << (epoch+1) << " (" << t << ")");

		// TODO: if balancing is used, loss computation below can be made more efficient
		// since we know the regularization loss already
		t.start();
		factor=balance(data, type, method);
		t.stop();
		timeRescaling=t.elapsedTime().nanos();

		// compute loss
		t.start();
		unblockAll(data.dh, hUnblockedName);
		currentLoss = nzsl(data.dv, data.dw, hUnblockedName, data.tasksPerRank);
		if (lambda > 0) {
			if (regularizer == ALS_L2) {
				currentLoss += lambda*(l2(data.dw) + l2(data.dh));
			} else {
				currentLoss += lambda*(nzl2(data.dw,data.nnz1name,data.tasksPerRank)+nzl2(data.dh,data.nnz2name,data.tasksPerRank));
			}
		}
		t.stop();
		timeLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Loss: " << currentLoss << " (" << t << ")");

		// compute test loss and create a trace entry
		if (testData!=0) {
			t.start();
			currentTestLoss=testLoss(*testData);
			t.stop();
			timeTestLoss=t.elapsedTime().nanos();
			LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
			entry = new AlsTraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, factor,timeRescaling, timeEpoch,currentTestLoss, timeTestLoss);
		}
		else{
			entry = new AlsTraceEntry(epoch+1, epoch/2 + 1, currentLoss, timeLoss, factor,timeRescaling, timeEpoch);
		}
		trace.add(entry);
	}

	mpi2::eraseAll<DapFactorizationData<>::W>(wUnblockedName);
	mpi2::eraseAll<DapFactorizationData<>::H>(hUnblockedName);
	LOG4CXX_INFO(detail::logger, "Starting DALS ("
			<< "with balance type : "
			<< (type == BALANCE_NONE ? "None" : "")
			<< (type == BALANCE_L2 ? "L2" : "")
			<< (type == BALANCE_NZL2 ? "Nzl2" : "")
			<< " and balance method : "
			<< (method == BALANCE_SIMPLE ? "Simple" : "")
			<< (method == BALANCE_OPTIMAL ? "Optimal" : "")
			<< "for nonzero squared loss and "
			<< (regularizer == ALS_L2 ? "L2" : "NZL2") << "(" << lambda << ")");;
}

namespace detail {
	void dalsRegisterTasks() {
		mpi2::registerTask<mf::detail::DalsW>();
		mpi2::registerTask<mf::detail::DalsH>();
	}
}

}
