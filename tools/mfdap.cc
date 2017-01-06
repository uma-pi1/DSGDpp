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
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/random/uniform_real.hpp>

#include <util/evaluation.h>

#include <mpi2/mpi2.h>
#include <mf/mf.h>
#include <mf/matrix/io/generateDistributedMatrix.h>

// added by me ******************
#include "parse.h"
using namespace std;
using namespace mf;
using namespace mpi2;
using namespace rg;
//*******************************
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

struct Args {
	std::string inputMatrixFile, inputTestMatrixFile, inputRowFacFile, inputColFacFile, outputRowFacFile,
		   outputColFacFile, traceFile, traceVar, lossString, balanceString, balanceMethodString;

	std::string lossName;
	std::vector<double> lossArgs;

	mf::mf_size_type epochs, blocks;
	unsigned seed;
	rg::Random32 random;
	mf::AlsRescale alsRescale;
	double lambda;
	int tasksPerRank;
	int worldSize;
	int worldRank;
	boost::mpi::communicator world;
	mf::BalanceType balanceType;
	mf::BalanceMethod balanceMethod;

	void createTraceFields(mf::Trace& trace) {
		trace.addField("loss", lossString);
	}
};

bool run(Args& args) {

	std::vector<DistributedSparseMatrix> dataVector;
	if (args.inputTestMatrixFile.length() == 0) {
		dataVector=getDataMatrices<SparseMatrix>(args.inputMatrixFile,"V",true,args.tasksPerRank, args.worldSize, args.blocks, 1,false,true);
	}else{
		dataVector=getDataMatrices<SparseMatrix>(args.inputMatrixFile,"V",true,args.tasksPerRank, args.worldSize, args.blocks, 1,
				false,true, &args.inputTestMatrixFile);
	}
	
	std::vector<DistributedSparseMatrixCM> dataVectorVC=getDataMatrices<SparseMatrixCM>(args.inputMatrixFile,"Vcm",false,
			args.tasksPerRank, args.worldSize, 1, args.blocks,false,true);

	std::pair<DistributedDenseMatrix, DistributedDenseMatrixCM> factorsPair= getFactors(args.inputRowFacFile,
			args.inputColFacFile,  args.tasksPerRank, args.worldSize,args.blocks,args.blocks,false);



	
//	// load the input matrices
//	DistributedSparseMatrix dv = loadMatrix<SparseMatrix>(args.inputMatrixFile,
//							"V", true, args.tasksPerRank, args.worldSize, args.blocks, 1);
//	LOG4CXX_INFO(logger, "Data matrix: "
//		<< dv.size1() << " x " << dv.size2() << ", " << nnz(dv) << " nonzeros, "
//		<< dv.blocks1() << " x " << dv.blocks2() << " blocks");
//
//	DistributedSparseMatrixCM dvc = loadMatrix<SparseMatrixCM>(args.inputMatrixFile,
//							"VC", false, args.tasksPerRank, args.worldSize, 1, args.blocks);
//	LOG4CXX_INFO(logger, "Data matrix (CM): "
//		<< dvc.size1() << " x " << dvc.size2() << ", " << nnz(dvc) << " nonzeros, "
//		<< dvc.blocks1() << " x " << dvc.blocks2() << " blocks");
//
//	DistributedDenseMatrix dw = loadMatrix<DenseMatrix>(args.inputRowFacFile,
//											"W", true, args.tasksPerRank, args.worldSize, args.blocks, 1);
//	LOG4CXX_INFO(logger, "Row factor matrix: "
//		<< dw.size1() << " x " << dw.size2() << ", " << dw.blocks1() << " x " << dw.blocks2() << " blocks");
//
//	DistributedDenseMatrixCM dh = loadMatrix<DenseMatrixCM>(args.inputColFacFile,
//											"H", false, args.tasksPerRank, args.worldSize, 1, args.blocks);
//	LOG4CXX_INFO(logger, "Column factor matrix: "
//		<< dh.size1() << " x " << dh.size2() << ", " << dh.blocks1() << " x " << dh.blocks2() << " blocks");

	
//	if (!checkConformity(dv, dw, dh)) {
	if (!checkConformity(dataVector[0], factorsPair.first, factorsPair.second)) {
		std::cerr << "Input matrices are not conforming." << std::endl;
		return false;
	}
	
	// create factorization data
//	DapFactorizationData<> data(dv, dw, dh, args.tasksPerRank, &dvc);
	DapFactorizationData<> data(dataVector[0], factorsPair.first, factorsPair.second, args.tasksPerRank, &dataVectorVC[0]);
	
	DsgdFactorizationData<>* testJob = NULL;
	DistributedSparseMatrix *dvTest = NULL;
	if (args.inputTestMatrixFile.compare("") != 0) {
//		dvTest = new DistributedSparseMatrix(
//				loadMatrix<SparseMatrix>(args.inputTestMatrixFile,
//						"Vtest", true, args.tasksPerRank, args.worldSize, args.blocks, args.blocks)
//		);

		
		dvTest=&dataVector[1];
		
		LOG4CXX_INFO(logger, "Test matrix: "
						<< dvTest->size1() << " x " << dvTest->size2() << ", " << nnz(*dvTest) << " nonzeros, "
						<< dvTest->blocks1() << " x " << dvTest->blocks2() << " blocks");


//		testJob = new DsgdFactorizationData<>(*dvTest, dw, dh, args.tasksPerRank);
		testJob = new DsgdFactorizationData<>(*dvTest, factorsPair.first, factorsPair.second, args.tasksPerRank);
		
	}

	// tracing
	Trace trace;
	args.createTraceFields(trace);
	Timer t;
	t.start();

	// go
	if (args.lossName.compare("Nzsl") == 0) {
		args.lambda = 0;
		AlsRegularizer regularizer = ALS_L2;
		dalsNzsl(data, args.epochs, trace, args.lambda, regularizer, args.balanceType, args.balanceMethod, testJob);
	} else if (args.lossName.compare("Nzsl_L2") == 0) {
		if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
			std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
			return false;
		}
		args.lambda = args.lossArgs[0];
		AlsRegularizer regularizer = ALS_L2;
		dalsNzsl(data, args.epochs, trace, args.lambda, regularizer, args.balanceType, args.balanceMethod, testJob);
	} else if (args.lossName.compare("Nzsl_Nzl2") == 0) {
		// als with Nzsl_Nzl2
		if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
			std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
			return false;
		}
		args.lambda = args.lossArgs[0];
		AlsRegularizer regularizer = ALS_NZL2;
		dalsNzsl(data, args.epochs, trace, args.lambda, regularizer, args.balanceType, args.balanceMethod, testJob);
	} else if (args.lossName.compare("Sl") == 0) {
		// gnmf with Sl
		if (args.lossArgs.size() != 0) {
			std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
			return false;
		}
		dgnmf(data, args.epochs, trace, args.balanceType, args.balanceMethod, testJob);
	} else if (args.lossName.compare("Gkl") == 0) {
		// lee01 with Gkl
		if (args.lossArgs.size() != 0) {
			std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
			return false;
		}
		dlee01Gkl(data, args.epochs, trace);
	} else {
		std::cout << "Invalid loss " << args.lossString << std::endl;
		return false;
	}
	t.stop();
	LOG4CXX_INFO(logger, "Total time: " << t);

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

	return true;
}

int main(int argc, char *argv[]) {
	using namespace std;
	using namespace mf;
	using namespace mpi2;
	using namespace rg;
	using namespace boost::program_options;

	// initialize mf library and mpi2
	boost::mpi::communicator& world = mfInit(argc, argv);

	mfStart();
	bool result = true;

	if (world.rank() == 0) {
		boost::this_thread::sleep(boost::posix_time::milliseconds(100)); // so that arguments are logged nicely
		Args args;

		// parse command line
		options_description desc("Options");
		desc.add_options()
			("help", "produce help message")
			("input-file", value<string>(&args.inputMatrixFile), "filename of data matrix")
			("input-test-file", value<string>(&args.inputTestMatrixFile), "filename of test matrix")
			("input-row-file", value<string>(&args.inputRowFacFile), "filename of initial row factors")
			("input-col-file", value<string>(&args.inputColFacFile), "filename of initial column factors")
			("output-row-file", value<string>(&args.outputRowFacFile), "filename of final row factors")
			("output-col-file", value<string>(&args.outputColFacFile), "filename of final column factors")
			("trace", value<string>(&args.traceFile), "filename of trace [trace.R]")
			("trace-var", value<string>(&args.traceVar), "variable name for trace [traceVar]")
			("epochs", value<mf_size_type>(&args.epochs), "number of epochs to run [10]")
			("tasks-per-rank", value<int>(&args.tasksPerRank), "number of concurrent tasks per rank [1]")
			("seed", value<unsigned>(&args.seed), "seed for random number generator (system time if not set)")
			("loss", value<string>(&args.lossString), "loss function (e.g., \"Nzsl\", \"Nzsl_L2(0.5)\"))")
			("balance", value<string>(&args.balanceString), "Type of balancing (None, L2, Nzl2) [None]")
			("balance-method", value<string>(&args.balanceMethodString), "Balancing method (e.g., \"Simple\", \"Optimal\") [Simple]")
		;

		positional_options_description pdesc;
		pdesc.add("input-file", 1);
		pdesc.add("input-test-file", 2);
		pdesc.add("input-row-file", 3);
		pdesc.add("input-col-file", 4);
		pdesc.add("output-row-file", 5);
		pdesc.add("output-col-file", 6);

		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
		notify(vm);

		// check required arguments
		if (vm.count("help") || vm.count("input-file") == 0 || vm.count("input-row-file") == 0 ||
			vm.count("input-col-file") == 0 || vm.count("loss") == 0)
		{
			cerr << "Error: Options input-file, input-row-file, input-col-file, loss have to be present" << endl;
			cout << "alternating projection [options]" << endl << endl;
			cout << desc << endl;
			exit(1);
		}

		// set default arguments
		if (vm.count("seed") == 0) args.seed = time(NULL);
		if (vm.count("trace") == 0) { args.traceFile = ""; }
		if (vm.count("trace-var") == 0) args.traceVar = "trace";
		if (vm.count("input-test-file") == 0) args.inputTestMatrixFile = "";
		if (vm.count("seed") == 0) args.seed = time(NULL);
		if (vm.count("epochs") == 0) args.epochs = 10;
		if (vm.count("tasks-per-rank") == 0) args.tasksPerRank = 1;
		if (vm.count("output-row-file") == 0) { args.outputRowFacFile = ""; }
		if (vm.count("output-col-file") == 0) { args.outputColFacFile = ""; }
		if (vm.count("balance") == 0) { args.balanceString = "None"; }
		if (vm.count("balance-method") == 0) { args.balanceMethodString = "Simple"; }

		// print some information
		LOG4CXX_INFO(logger, "Input");
		LOG4CXX_INFO(logger, "    Input file: " << args.inputMatrixFile);
		LOG4CXX_INFO(logger, "    Input test file: " << (args.inputTestMatrixFile.length() == 0 ? "Disabled" :args.inputTestMatrixFile));
		LOG4CXX_INFO(logger, "    Input row factors: " << args.inputRowFacFile);
		LOG4CXX_INFO(logger, "    Input column factors: " << args.inputColFacFile);
		LOG4CXX_INFO(logger, "Output");
		LOG4CXX_INFO(logger, "    Output row factors: " << (args.outputRowFacFile.length() == 0 ? "Disabled" :args.outputRowFacFile));
		LOG4CXX_INFO(logger, "    Output column factors: " << (args.outputColFacFile.length() == 0 ? "Disabled" :args.outputColFacFile));
		if (args.traceFile.length() == 0) {
			LOG4CXX_INFO(logger, "    Trace: Disabled");
		} else {
			LOG4CXX_INFO(logger, "    Trace: " << args.traceFile << " (" << args.traceVar << ")");
		}
		LOG4CXX_INFO(logger, "Parallelization");
		LOG4CXX_INFO(logger, "    MPI ranks: " << world.size());
		LOG4CXX_INFO(logger, "    Tasks per rank: " << args.tasksPerRank);
		LOG4CXX_INFO(logger, "Alternating Projection options");
		LOG4CXX_INFO(logger, "    Seed: " << args.seed);
		LOG4CXX_INFO(logger, "    Epochs: " << args.epochs);
//		LOG4CXX_INFO(logger, "    Factorization rank: " << args.rank);
		LOG4CXX_INFO(logger, "    Loss function: " << args.lossString);

		// fill fields
		args.random = Random32(args.seed);
		args.world = world;
		args.worldSize = world.size();
		args.worldRank = world.rank();
		args.blocks = args.worldSize * args.tasksPerRank;

		// parsing
		parse::parseArg("loss", args.lossString, args.lossName, args.lossArgs);
		if (args.balanceString.compare("None") == 0) {
			args.balanceType = BALANCE_NONE;
		} else if (args.balanceString.compare("L2") == 0) {
			args.balanceType = BALANCE_L2;
		} else if (args.balanceString.compare("Nzl2") == 0) {
			args.balanceType = BALANCE_NZL2;
		} else {
			cerr << "Error: Invalid value for --balance: " << args.balanceString << endl;
			exit(1);
		}
		if (args.balanceMethodString.compare("Simple") == 0) {
			args.balanceMethod = BALANCE_SIMPLE;
		} else if (args.balanceMethodString.compare("Optimal") == 0) {
			args.balanceMethod = BALANCE_OPTIMAL;
		} else {
			cerr << "Error: Invalid value for --balance-method: " << args.balanceMethodString << endl;
			exit(1);
		}

		switch (args.balanceType){
		case BALANCE_NONE:
			LOG4CXX_INFO(logger, "    Balancing: Disabled");
			break;
		case BALANCE_L2:
			LOG4CXX_INFO(logger, "    Balancing: L2, " << (args.balanceMethod == BALANCE_SIMPLE ? "Simple" : "Optimal"));
			break;
		case BALANCE_NZL2:
			LOG4CXX_INFO(logger, "    Balancing: Nzl2, " << (args.balanceMethod == BALANCE_SIMPLE ? "Simple" : "Optimal"));
			break;
		}

		// let's go
		result = run(args);

	}

	mfStop();
	mfFinalize();

	// everything OK
	return result ? 0 : 1;

}





