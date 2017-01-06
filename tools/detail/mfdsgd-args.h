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
#ifndef MFDSGD_ARGS_H
#define MFDSGD_ARGS_H

#include <vector>
#include <boost/mpi/communicator.hpp>
#include <util/random.h>
#include <mf/types.h>
#include <mf/trace.h>
#include <mf/sgd/sgd.h>
#include <mf/sgd/dsgd.h>

using namespace std;

struct Args {
	std::string inputMatrixFile, inputTestMatrixFile, inputRowFacFile, inputColFacFile, outputRowFacFile,
		   outputColFacFile, traceFile, traceVar, sgdOrderString, stratumOrderString,
		   updateString, regularizeString, lossString, decayString, inputSampleMatrixFile, truncateString, absString, balanceString;

	std::string updateName, regularizeName, lossName, decayName;
	std::vector<double> updateArgs, regularizeArgs, lossArgs, truncateArgs;//, absArgs;
	std::vector<string> decayArgs;

	mf::mf_size_type epochs, rank, blocks1, blocks2;
	unsigned seed;
	rg::Random32 random;
	mf::SgdOrder sgdOrder;
	mf::StratumOrder stratumOrder;
	bool mapReduce;
	double epsilon, epsIncrease, epsDecrease, improvement,alpha, A;
	unsigned tries;
	int tasksPerRank;
	int worldSize;
	int worldRank;
	boost::mpi::communicator world;
	bool abs;
	mf::BalanceType balanceType;
	mf::BalanceMethod balanceMethod;
	bool averageDeltas;

	void createTraceFields(mf::Trace& trace) {
		trace.addField("update", updateString);
		trace.addField("regularize", regularizeString);
		// TODO trace.addField("lambda", lambda);
		trace.addField("rank", rank);
		trace.addField("input_file", inputMatrixFile);
		trace.addField("sample_matrix", inputSampleMatrixFile);
		trace.addField("nodes",worldSize);
		trace.addField("threads", tasksPerRank);
		trace.addField("barrier", mapReduce ? "true" : "false");
	}
};

#endif
