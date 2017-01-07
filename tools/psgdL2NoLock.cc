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
 *
 * Illustrates matrix factorization with parallel SGD. We first create factors and then a data matrix
 * from these factors. This process ensures that we know the best factorization of the input.
 * These matrices are distributed across a cluster. We then try to reconstruct the factors
 * using PSGD.
 */
#include <iostream>
#include <sstream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/uniform_real.hpp>

#include <util/evaluation.h>

#include <mpi2/mpi2.h>
#include <mf/mf.h>

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

using namespace std;
using namespace mf;
using namespace mpi2;
using namespace rg;
using namespace boost::numeric::ublas;

// type of SGD
typedef UpdateTruncate<UpdateNzslL2> Update;
typedef RegularizeNone Regularize;
typedef SumLoss<NzslLoss, L2Loss> Loss;
typedef NzslLoss TestLoss;

int main(int argc, char* argv[]) {
	using namespace boost::program_options;
	// initialize mf library and mpi2
	boost::mpi::communicator& world = mfInit(argc, argv);

	mfStart();

	if (world.rank() == 0)
	{
#ifndef NDEBUG
		LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
#endif
		//data
		mf_size_type epochs;
		string inputSampleMatrixFile;
		string inputMatrixFile;
		string inputRowFacFile;
		string inputColFacFile;
		string outputRowFacFile;
		string outputColFacFile;
		string inputTestMatrixFile;
		string traceFile,traceVar;
		string shuffleStr;

		double lambda = 50;
		double eps0 = 0.01;
		// parameters for distribution
		int tasks;


		options_description desc("Options");
		desc.add_options()
				("help", "produce help message")
				("epochs", value<mf_size_type>(&epochs)->default_value(10), "number of epochs to run [10]")
				("lambda", value<double>(&lambda)->default_value(50), "lambda")
				("eps0", value<double>(&eps0)->default_value(0.01), "initial step size for BoldDriver")
				("tasks-per-rank", value<int>(&tasks)->default_value(1), "number of concurrent tasks [1]")
				("trace", value<string>(&traceFile)->default_value("trace.R"), "filename of trace [trace.R]")
				("traceVar", value<string>(&traceVar)->default_value("trace"), "variable name for trace [traceVar]")
				("input-file", value<string>(&inputMatrixFile), "input matrix")
				("input-test-file", value<string>(&inputTestMatrixFile), "input test matrix")
			    ("input-row-file", value<string>(&inputRowFacFile), "input initial row factor")
			    ("input-col-file", value<string>(&inputColFacFile), "input initial column factor")
			    ("output-row-file", value<string>(&outputRowFacFile), "output initial row factor")
			    ("output-col-file", value<string>(&outputColFacFile), "output initial column factor")
			    ("shuffle",value<string>(&shuffleStr)->default_value("seq"),"shuffle method eg seq, par, parAdd")
				;

		positional_options_description pdesc;
		pdesc.add("input-file", 1);
		pdesc.add("input-test-file", 2);
		pdesc.add("input-sample-matrix-file", 3);
		pdesc.add("input-row-file", 4);
		pdesc.add("input-col-file", 5);

		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
		notify(vm);

		if (vm.count("help") || vm.count("input-file")==0) {
			cout << "psgd with L2 NoLock (Hogwild-style) [options] <input-file> " << endl << endl;
			cout << desc << endl;
			return 1;
		}

		LOG4CXX_INFO(logger, "Using " << tasks << " parallel tasks");

		PsgdShuffle shuffle;
		if (shuffleStr.compare("seq") == 0)
			shuffle= PSGD_SHUFFLE_SEQ;
		else if (shuffleStr.compare("par") == 0)
			shuffle= PSGD_SHUFFLE_PARALLEL;
		else
			shuffle= PSGD_SHUFFLE_PARALLEL_ADDITIONAL_TASK;

		LOG4CXX_INFO(logger, "Using " << shuffle);



		// Read matrices
		Random32 random;
		SparseMatrix v,vTest;
		DenseMatrix w;
		DenseMatrixCM h;
		readMatrix(inputMatrixFile,v);
		readMatrix(inputTestMatrixFile,vTest);
		readMatrix(inputRowFacFile,w);
		readMatrix(inputColFacFile,h);

		// parameters for SGD
		SgdOrder order = SGD_ORDER_WOR;
		Update update = Update(UpdateNzslL2(lambda), -100, 100); // truncate for numerical stability
		Regularize regularize;
		Loss loss((NzslLoss()), L2Loss(lambda));
		TestLoss testLoss;
		BalanceType balanceType = BALANCE_NONE;
		BalanceMethod balanceMethod = BALANCE_OPTIMAL;

		// initialize the DSGD
		Timer t;
		PsgdRunner psgdRunner(random);
		PsgdJob<Update,Regularize> psgdJob(v, w, h, update, regularize, order, tasks, shuffle);
		BoldDriver decay(eps0);
		Trace trace;


		trace.addField("Loss", "L2");
		trace.addField("Shuffle_method", shuffle);
		trace.addField("input_file", inputMatrixFile);
		trace.addField("sample_matrix", inputSampleMatrixFile);
		trace.addField("tasks", tasks);

		// print the test loss
		FactorizationData<> testData(vTest, w, h);
		LOG4CXX_INFO(logger, "Initial test loss: " << testLoss(testData));

		// run HogwildSGD to try to reconstruct the original factors
		t.start();
		psgdRunner.run(psgdJob, loss, epochs, decay, trace, balanceType, balanceMethod, &testData, &testLoss);

		t.stop();
		LOG4CXX_INFO(logger, "Total time: " << t);

		// print the test loss
		LOG4CXX_INFO(logger, "Final test loss: " << testLoss(testData));

		// write trace to an R file
		LOG4CXX_INFO(logger, "Writing trace to " << traceFile);
		trace.toRfile(traceFile, traceVar);
		
					// write computed factors to file
				if (outputRowFacFile.length() > 0) {
					LOG4CXX_INFO(logger, "Writing row factors to " << outputRowFacFile);
					//DenseMatrix w0;
					//unblock(dw, w0);
					writeMatrix(outputRowFacFile, w);
				}
				if (outputColFacFile.length() > 0) {
					LOG4CXX_INFO(logger, "Writing column factors to " << outputColFacFile);
					//DenseMatrixCM h0;
					//unblock(dh, h0);
					writeMatrix(outputColFacFile, h);
				}
	}

	mfStop();
	mfFinalize();

	return 0;
}
