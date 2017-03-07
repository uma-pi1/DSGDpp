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

#include <tools/detail/mfdsgd-run.h>
#include <tools/detail/mfdsgd-generated.h>

#include "parse.h"
log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

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
			("sgd-order", value<string>(&args.sgdOrderString), "order of SGD steps [WOR] (e.g., \"SEQ\", \"WR\", \"WOR\")")
			("stratum-order", value<string>(&args.stratumOrderString), "order of strata [COWOR] (e.g., \"SEQ\", \"RSEQ\", \"WR\", \"WOR\", \"COWOR\")")
			("map-reduce", value<bool>(&args.mapReduce), "whether to use the (slower) MapReduce implementation [false])")
			("seed", value<unsigned>(&args.seed), "seed for random number generator (system time if not set)")
			("rank", value<mf_size_type>(&args.rank), "rank of factorization")
			("update", value<string>(&args.updateString), "SGD update function (e.g., \"Sl\", \"Nzsl\", \"GklData\")")
			("regularize", value<string>(&args.regularizeString), "SGD regularization function (e.g., \"None\", \"L2(0.05)\", \"Nzl2(0.05)\")")
			("loss", value<string>(&args.lossString), "loss function (e.g., \"Nzsl\", \"Nzsl_L2(0.5)\"))")
			("abs", "if present, absolute values are taken after every SGD step")
			("truncate", value<string>(&args.truncateString), "if present, truncatation is enabled (e.g., --truncate \"(-1000, 1000)\"")
			("decay", value<string>(&args.decayString), "decay function (constant, bold driver, or auto)")
			("balance", value<string>(&args.balanceString), "Type of balancing (None, L2, Nzl2)")
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
			vm.count("input-col-file") == 0 || vm.count("update") == 0 || vm.count("regularize") == 0 ||
			vm.count("loss") == 0 || vm.count("rank") == 0 || vm.count("decay") == 0)
		{
			cerr << "Error: Options input-file, input-row-file, input-col-file, update, regularize, loss, rank, and decay have to be present" << endl;
			cout << "mfsgd [options]" << endl << endl;
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
		if (vm.count("sgd-order") == 0) { args.sgdOrderString = "WOR"; args.sgdOrder = SGD_ORDER_WOR; }
		if (vm.count("stratum-order") == 0) { args.stratumOrderString = "COWOR"; args.stratumOrder = STRATUM_ORDER_COWOR; }
		if (vm.count("map-reduce") == 0) { args.mapReduce = false; }
		if (vm.count("output-row-file") == 0) { args.outputRowFacFile = ""; }
		if (vm.count("output-col-file") == 0) { args.outputColFacFile = ""; }
		if (vm.count("balance") == 0) { args.balanceString = "None"; }

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
		LOG4CXX_INFO(logger, "DSGD options");
		LOG4CXX_INFO(logger, "    Seed: " << args.seed);
		LOG4CXX_INFO(logger, "    Epochs: " << args.epochs);
		LOG4CXX_INFO(logger, "    Factorization rank: " << args.rank);
		LOG4CXX_INFO(logger, "    Update function: " << args.updateString);
		LOG4CXX_INFO(logger, "    Regularize function: " << args.regularizeString);
		LOG4CXX_INFO(logger, "    Loss function: " << args.lossString);
		LOG4CXX_INFO(logger, "    Decay: " << args.decayString);

		// parse balancing
		args.balanceMethod = BALANCE_SIMPLE;
		if (args.balanceString.compare("None") == 0) {
			LOG4CXX_INFO(logger, "    Balancing: Disabled");
			args.balanceType = BALANCE_NONE;
		} else if (args.balanceString.compare("L2") == 0) {
			LOG4CXX_INFO(logger, "    Balancing: L2");
			args.balanceType = BALANCE_L2;
		} else if (args.balanceString.compare("NZL2") == 0) {
			LOG4CXX_INFO(logger, "    Balancing: Nzl2");
			args.balanceType = BALANCE_NZL2;
		} else {
			cerr << "Invalid arguments for balance; expected \"None\", \"L2\" or \"Nzl2\"" << endl;
			exit(1);
		}

		// parse abs
		if (vm.count("abs") == 0) {
			LOG4CXX_INFO(logger, "    Absolute function: Disabled");
			args.abs = false;
		} else {
			LOG4CXX_INFO(logger, "    Absolute function: Enabled");
			args.abs = true;
		}

		// parse truncate
		if (vm.count("truncate") == 0) {
			args.truncateArgs.resize(2);
			args.truncateArgs[0] = NAN;
			args.truncateArgs[1] = NAN;
			LOG4CXX_INFO(logger, "    Truncation: Disabled");
		} else {
			parse::parseTruncate("truncate", args.truncateString, args);
			LOG4CXX_INFO(logger, "    Truncation: (" << args.truncateArgs[0] << "," << args.truncateArgs[1] << ")");
		}

		// parse sgdOrder
		if (args.sgdOrderString.compare("SEQ") == 0) {
			LOG4CXX_INFO(logger, "    SGD step sequence: SEQ");
			args.sgdOrder = SGD_ORDER_SEQ;
		} else if (args.sgdOrderString.compare("WR") == 0) {
			LOG4CXX_INFO(logger, "    SGD step sequence: WR");
			args.sgdOrder = SGD_ORDER_WR;
		} else if (args.sgdOrderString.compare("WOR") == 0) {
			LOG4CXX_INFO(logger, "    SGD step sequence: WOR");
			args.sgdOrder = SGD_ORDER_WOR;
		} else {
			cerr << "Invalid arguments for sgdOrder; expected \"SEQ\", \"WR\" or \"WOR\"" << endl;
			exit(1);
		}

		// parse stratumOrder
		if (args.stratumOrderString.compare("SEQ") == 0) {
			LOG4CXX_INFO(logger, "    DSGD stratum sequence: SEQ");
			args.stratumOrder = STRATUM_ORDER_SEQ;
		} else if (args.stratumOrderString.compare("RSEQ") == 0) {
			LOG4CXX_INFO(logger, "    DSGD stratum sequence: RSEQ");
			args.stratumOrder = STRATUM_ORDER_RSEQ;
		} else if (args.stratumOrderString.compare("WR") == 0) {
			LOG4CXX_INFO(logger, "    DSGD stratum sequence: WR");
			args.stratumOrder = STRATUM_ORDER_WR;
		} else if (args.stratumOrderString.compare("WOR") == 0) {
			LOG4CXX_INFO(logger, "    DSGD stratum sequence: WOR");
			args.stratumOrder = STRATUM_ORDER_WOR;
		} else if (args.stratumOrderString.compare("COWOR") == 0) {
			LOG4CXX_INFO(logger, "    DSGD stratum sequence: COWOR");
			args.stratumOrder = STRATUM_ORDER_COWOR;
		} else {
			cerr << "Invalid arguments for stratumOrder; expected \"SEQ\", \"RSEQ\", \"WR\", \"WOR\" or \"COWOR\"" << endl;
			exit(1);
		}

		LOG4CXX_INFO(logger, "    Slow MapReduce implementation: " << args.mapReduce);

		// fill fields
		args.random = Random32(args.seed);
		args.world = world;
		args.worldSize = world.size();
		args.worldRank = world.rank();
		args.tries = args.worldSize * args.tasksPerRank;
		while (args.tries < 7) args.tries += args.worldSize;
		args.blocks1 = args.worldSize * args.tasksPerRank;
		args.blocks2 = args.worldSize * args.tasksPerRank;

		// parsing
		parse::parseArg("update", args.updateString, args.updateName, args.updateArgs);
		parse::parseArg("regularize", args.regularizeString, args.regularizeName, args.regularizeArgs);
		parse::parseArg("loss", args.lossString, args.lossName, args.lossArgs);
		parse::parseDecay("decay", args.decayString, args);

		// let's go
		result = runArgs(args);

	}

	mfStop();
	mfFinalize();

	// everything OK
	return result ? 0 : 1;

}














