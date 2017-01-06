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
#ifndef MFDSGD_RUN_H
#define MFDSGD_RUN_H

#include <log4cxx/logger.h>

#include <util/evaluation.h>

#include <mf/matrix/io/load.h>
#include <mf/matrix/io/loadDistributedMatrix.h>
#include <mf/matrix/io/ioProjected.h>
#include <mf/matrix/io/write.h>
#include <mf/matrix/op/project.h>
#include <mf/matrix/op/unblock.h>
#include <mf/loss/loss.h>
#include <mf/sgd/dsgd.h>
#include <mf/sgd/decay/decay_auto.h>
#include <mf/sgd/decay/decay_bolddriver.h>
#include <mf/sgd/decay/decay_sequential.h>
#include <mf/sgd/decay/decay_constant.h>
#include <mf/trace.h>

#include <tools/detail/mfdsgd-args.h>

#include <mf/loss/nzsl.h>
#include <mf/loss/biased-nzsl.h>
#include <mf/loss/nzrmse.h>
#include <mf/loss/sl.h>
#include <mf/loss/gkl.h>
#include <mf/loss/nzl2.h>
#include <mf/loss/biased-nzl2.h>
#include <mf/loss/l1.h>
#include <mf/loss/l2.h>
#include <mf/matrix/io/generateDistributedMatrix.h>

using namespace std;
using namespace mf;
using namespace rg;

extern log4cxx::LoggerPtr logger;

template<typename U,typename R,typename L, typename D>
//void runDsgd2(Args& args, U update, R regularize, L loss, D decay,
//		DsgdJob<U,R>& dsgdJob, DistributedDenseMatrix& dw, DistributedDenseMatrixCM& dh, Trace& trace) {
void runDsgd2(Args& args, U update, R regularize, L loss, D decay,
		DsgdJob<U,R>& dsgdJob, std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM>& factorsPair,
		std::vector<DistributedSparseMatrix>& dataVector, Trace& trace) {

	mf_size_type blocks1 = args.worldSize * args.tasksPerRank;
	mf_size_type blocks2 = args.worldSize * args.tasksPerRank;
	Timer t;
	DsgdRunner dsgdRunner(args.random);
	if (args.inputTestMatrixFile.length() == 0) {
		// run DSGD
		t.start();
		dsgdRunner.run(dsgdJob, loss, args.epochs, decay, trace, args.balanceType, args.balanceMethod);
		t.stop();
		LOG4CXX_INFO(logger, "Total time: " << t);
	} else {
//		DistributedSparseMatrix dvTest=loadMatrix<SparseMatrix>(args.inputTestMatrixFile,
//									"Vtest", true, args.tasksPerRank, args.worldSize, blocks1, blocks2);
//		LOG4CXX_INFO(logger, "Test matrix: "
//			<< dvTest.size1() << " x " << dvTest.size2() << ", " << nnz(dvTest) << " nonzeros, "
//			<< dvTest.blocks1() << " x " << dvTest.blocks2() << " blocks");
//
//		DsgdFactorizationData<> testData(dvTest,dw,dh,args.tasksPerRank);

		DsgdFactorizationData<> testData(dataVector[1], factorsPair.first, factorsPair.second, args.tasksPerRank);
		if (args.lossName.compare("Biased_Nzsl_Nzl2") == 0) {
			LOG4CXX_INFO(logger, "Using BiasedNzslLoss for test data");
			BiasedNzslLoss testLoss;
			// run DSGD
			t.start();
			dsgdRunner.run(dsgdJob, loss, args.epochs, decay, trace, args.balanceType, args.balanceMethod, &testData, &testLoss);
			t.stop();
			LOG4CXX_INFO(logger, "Total time: " << t);
		} else {
			LOG4CXX_INFO(logger, "Using NzslLoss for test data");
			NzslLoss testLoss;
			// run DSGD
			t.start();
			dsgdRunner.run(dsgdJob, loss, args.epochs, decay, trace, args.balanceType, args.balanceMethod, &testData, &testLoss);
			t.stop();
			LOG4CXX_INFO(logger, "Total time: " << t);
		}
	}
}

// run DSGD
template<typename U,typename R,typename L>
void runDsgd(Args& args, U update, R regularize, L loss) {

	mf_size_type blocks1 = args.worldSize * args.tasksPerRank;
	mf_size_type blocks2 = args.worldSize * args.tasksPerRank;

	std::vector<DistributedSparseMatrix> dataVector;
	if (args.inputTestMatrixFile.length() == 0) {
		dataVector=getDataMatrices<SparseMatrix>(args.inputMatrixFile, "V", true, args.tasksPerRank,
				args.worldSize, blocks1, blocks2, false, false);
	}else{
		dataVector=getDataMatrices<SparseMatrix>(args.inputMatrixFile, "V", true, args.tasksPerRank,
				args.worldSize, blocks1, blocks2, false, false, &args.inputTestMatrixFile);
	}

	std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> factorsPair= getFactors(args.inputRowFacFile,
			args.inputColFacFile,  args.tasksPerRank, args.worldSize, blocks1, blocks2, false);

//	// distribute the input matrices
//	DistributedSparseMatrix dv=loadMatrix<SparseMatrix>(args.inputMatrixFile,
//							"V", true, args.tasksPerRank, args.worldSize,blocks1,blocks2);
//	LOG4CXX_INFO(logger, "Data matrix: "
//		<< dv.size1() << " x " << dv.size2() << ", " << nnz(dv) << " nonzeros, "
//		<< dv.blocks1() << " x " << dv.blocks2() << " blocks");
//	DistributedDenseMatrix dw=loadMatrix<DenseMatrix>(args.inputRowFacFile,
//											"W", true, args.tasksPerRank, args.worldSize,blocks1,1);
//	LOG4CXX_INFO(logger, "Row factor matrix: "
//		<< dw.size1() << " x " << dw.size2() << ", " << dw.blocks1() << " x " << dw.blocks2() << " blocks");
//	DistributedDenseMatrixCM dh=loadMatrix<DenseMatrixCM>(args.inputColFacFile,
//											"H", false, args.tasksPerRank, args.worldSize,1,blocks2);
//	LOG4CXX_INFO(logger, "Column factor matrix: "
//		<< dh.size1() << " x " << dh.size2() << ", " << dh.blocks1() << " x " << dh.blocks2() << " blocks");

//	Timer t;
//	DsgdRunner dsgdRunner(args.random);
//	DsgdJob<U,R> dsgdJob(dv, dw, dh, update, regularize, args.sgdOrder, args.stratumOrder, args.mapReduce, args.tasksPerRank);

	DsgdJob<U,R> dsgdJob(dataVector[0], factorsPair.first, factorsPair.second, update, regularize, args.sgdOrder, args.stratumOrder, args.mapReduce, args.tasksPerRank);

	Trace trace;
	// add trace fields
	args.createTraceFields(trace);

	if (args.decayName.compare("Auto") == 0){

		// read sample matrix
		ProjectedSparseMatrix vsample;
		readProjectedMatrix(vsample, args.inputSampleMatrixFile);
		LOG4CXX_INFO(logger, "Sample matrix: "
			<< vsample.size1 << " x " << vsample.size2 << ", " << vsample.data.nnz() << " nonzeros");

		DistributedDecayAuto<U,R,L> decay(dsgdJob, loss, vsample, "decay", args.epsilon, args.tries);
//		runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, dw, dh, trace);
		runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, factorsPair, dataVector, trace);

	} else if (args.decayName.compare("BoldDriver") == 0){

		if (std::isnan(args.epsDecrease) && (std::isnan(args.epsIncrease))) {
			BoldDriver decay(args.epsilon);
//			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, dw, dh, trace);
			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, factorsPair, dataVector, trace);
		} else {
			BoldDriver decay(args.epsilon, args.epsDecrease, args.epsIncrease);
//			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, dw, dh, trace);
			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, factorsPair, dataVector, trace);
		}

	} else if (args.decayName.compare("Const") == 0) {

		DecayConstant decay(args.epsilon);
//		runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, dw, dh, trace);
		runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, factorsPair, dataVector, trace);

	
	}else if (args.decayName.compare("Sequential") == 0){

		if (std::isnan(args.alpha) && (std::isnan(args.A))) {
			SequentialDecay decay(args.epsilon);
//			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, dw, dh, trace);
			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, factorsPair, dataVector, trace);
		} else {
			SequentialDecay decay(args.epsilon, args.alpha, args.A);
//			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, dw, dh, trace);
			runDsgd2<>(args, update, regularize, loss, decay, dsgdJob, factorsPair, dataVector, trace);
		}

	}

	// write trace to an R file
	if (args.traceFile.length() > 0) {
		LOG4CXX_INFO(logger, "Writing trace to " << args.traceFile);
		trace.toRfile(args.traceFile, args.traceVar);
	}

	// write computed factors to file
	if (args.outputRowFacFile.length() > 0) {
		LOG4CXX_INFO(logger, "Writing row factors to " << args.outputRowFacFile);
		DenseMatrix w0;
//		unblock(dw, w0);
		unblock(factorsPair.first, w0);
		writeMatrix(args.outputRowFacFile, w0);
	}
	if (args.outputColFacFile.length() > 0) {
		LOG4CXX_INFO(logger, "Writing column factors to " << args.outputColFacFile);
		DenseMatrixCM h0;
//		unblock(dh, h0);
		unblock(factorsPair.second, h0);
		writeMatrix(args.outputColFacFile, h0);
	}
}


#endif
