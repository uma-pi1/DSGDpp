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
 * Illustrates matrix factorization with distributed Gaussian non-negative matrix factorization.
 * We first create factors and then a data matrix
 * from these factors. THis process ensures that we know the best factorization of the input.
 * We then try to reconstruct the factors.
 */
#include <iostream>

#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/random/uniform_real.hpp>

#include <util/evaluation.h>
#include <util/io.h>

#include <mf/mf.h>

using namespace std;
using namespace mf;
using namespace rg;
using namespace boost::numeric::ublas;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

int main(int argc, char* argv[]) {
	boost::mpi::communicator& world = mfInit(argc, argv);

	// parameters for the factorization
	mf_size_type size1 = 10000;
	mf_size_type size2 = 10000;
	mf_size_type nnz = 1000000;
	double sigma = 1; // standard deviation
	mf_size_type r = 10;

	// parameters for ALS
	unsigned epochs = 20;
	SlLoss loss;
	mf_size_type testNnz = nnz/100;

	BalanceType type = BALANCE_L2;;
	BalanceMethod method = BALANCE_SIMPLE;

	// parameters for distribution
	int tasksPerRank = 2;
	mf_size_type blocks = world.size() * tasksPerRank;

	mfStart();

	if (world.rank() == 0) {
	#ifndef NDEBUG
		LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
	#endif
		// generate original factors by sampling from a uniform[0,1] distribution
		Random32 random; // note: this takes a default seed (not randomized!)
		DenseMatrix wIn(size1, r);
		DenseMatrixCM hIn(r, size2);
		generateRandom(wIn, random, boost::uniform_real<>(0,1));
		generateRandom(hIn, random, boost::uniform_real<>(0,1));

		// generate a sparse matrix by selecting random entries from the generated factors
		// and add small Gaussian noise
		SparseMatrix v;
		generateRandom(v, nnz, wIn, hIn, random);
		//addRandom(v, random, boost::normal_distribution<>(0, 0.1));
		LOG4CXX_INFO(logger, "Data matrix: "
			<< v.size1() << " x " << v.size2() << ", " << v.nnz() << " nonzeros");
		v.sort();
		LOG4CXX_INFO(logger, "Loss with original factors: " << loss((FactorizationData<>(v, wIn, hIn))));
		SparseMatrixCM vc;
		copyCm(v, vc);

		// create a test matrix (without noise)
		SparseMatrix vTest;
		generateRandom(vTest, testNnz, wIn, hIn, random);
		LOG4CXX_INFO(logger, "Test matrix: "
			<< v.size1() << " x " << v.size2() << ", " << vTest.nnz() << " nonzeros");

		// generate initial factors by sampling from a uniform[0,1] distribution
		DenseMatrix w(size1, r);
		DenseMatrixCM h(r, size2);
		generateRandom(w, random, boost::uniform_real<>(0.0, 1.0));
		generateRandom(h, random, boost::uniform_real<>(0.0, 1.0));

		// distribute the input matrices and test matrix
		DistributedSparseMatrix dv = distributeMatrix("V", blocks, 1, true, v);
		LOG4CXX_INFO(logger, "Distributed data matrix: "
				<< dv.blocks1() << " x " << dv.blocks2() << " blocks");
		DistributedSparseMatrixCM dvc = distributeMatrix("VC", 1, blocks, false, vc);
		LOG4CXX_INFO(logger, "Distributed data matrix (CM): "
				<< dvc.blocks1() << " x " << dvc.blocks2() << " blocks");
		DistributedSparseMatrix dvTest = distributeMatrix("Vtest", blocks, blocks, true, vTest);
		LOG4CXX_INFO(logger, "Distributed test matrix: "
				<< dvTest.blocks1() << " x " << dvTest.blocks2() << " blocks");
		DistributedDenseMatrix dw = distributeMatrix("W", blocks, 1, true, w);
		DistributedDenseMatrixCM dh = distributeMatrix("H", 1, blocks, false, h);
		LOG4CXX_INFO(logger, "Distributed factor matrices");


		// initialize
		DapFactorizationData<> data(dv, dw, dh, tasksPerRank, &dvc);
		DsgdFactorizationData<> testJob(dvTest, dw, dh, tasksPerRank);
		Trace trace;
		Timer t;

		// run GNMF to try to reconstruct the original factors
		t.start();
		dgnmf(data, epochs, trace, type, method, &testJob);
		t.stop();
		LOG4CXX_INFO(logger, "Total time: " << t);

		// write trace to an R file
		LOG4CXX_INFO(logger, "Writing trace to " << "/tmp/dgnmf-trace.R");
		trace.toRfile("/tmp/dgnmf-trace.R", "dgnmf");
	}

	mfStop();
	mfFinalize();

	return 0;
}

