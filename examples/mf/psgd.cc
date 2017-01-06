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
 *
 * Run with: psgd
 * (make sure to use a production build, otherwise it will be slow)
 */
#include <iostream>
#include <sstream>

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
typedef UpdateLock<Update> UpdateL;
typedef RegularizeNone Regularize;
typedef SumLoss<NzslLoss, L2Loss> Loss;
typedef NzslLoss TestLoss;

MPI2_TYPE_TRAITS(UpdateL);

int main(int argc, char* argv[]) {
	// initialize mf library and mpi2
	boost::mpi::communicator& world = mfInit(argc, argv);

	// parameters for the factorization
	mf_size_type size1 = 10000;
	mf_size_type size2 = 10000;
	mf_size_type nnz = 1000000;
	double sigma = sqrt(10); // standard deviation
	double lambda = 1/sigma/sigma;
	mf_size_type r = 10;

	// parameters for distribution
	int tasks = 4;

	// parameters for SGD
	double eps0 = 0.01;
	mf_size_type epochs = 20;
	SgdOrder order = SGD_ORDER_WOR;
	PsgdShuffle shuffle = PSGD_SHUFFLE_PARALLEL;
	Update update = Update(UpdateNzslL2(lambda), -10*sigma, 10*sigma); // truncate for numerical stability
	UpdateL updateLock = UpdateL(update, size1, size2); // with locking
	Regularize regularize;
	Loss loss((NzslLoss()), L2Loss(lambda));
	TestLoss testLoss;
	mf_size_type testNnz = nnz/100;
	BalanceType balanceType = BALANCE_NONE;
	BalanceMethod balanceMethod = BALANCE_OPTIMAL;

	// start mf library
	mfStart();

	if (world.rank() == 0)
	{
#ifndef NDEBUG
		LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
#endif
		LOG4CXX_INFO(logger, "Using " << tasks << " parallel tasks");

		// TODO: distribute matrix generation
		// generate original factors by sampling from a normal(0,sigma) distribution
		Random32 random; // note: this takes a default seed (not randomized!)
		DenseMatrix wIn(size1, r);
		DenseMatrixCM hIn(r, size2);
		generateRandom(wIn, random, boost::normal_distribution<>(0, sigma));
		generateRandom(hIn, random, boost::normal_distribution<>(0, sigma));

		// generate a sparse matrix by selecting random entries from the generated factors
		// and add small Gaussian noise
		SparseMatrix v;
		generateRandom(v, nnz, wIn, hIn, random);
		addRandom(v, random, boost::normal_distribution<>(0, 0.1));
		LOG4CXX_INFO(logger, "Data matrix: "
			<< v.size1() << " x " << v.size2() << ", " << v.nnz() << " nonzeros");
		LOG4CXX_INFO(logger, "Loss with original factors: " << loss((FactorizationData<>(v, wIn, hIn))));

		// create a test matrix (without noise)
		SparseMatrix vTest;
		generateRandom(vTest, testNnz, wIn, hIn, random);
		LOG4CXX_INFO(logger, "Test matrix: "
			<< v.size1() << " x " << v.size2() << ", " << vTest.nnz() << " nonzeros");

		// take a small sample and remove empty rows/columns
		ProjectedSparseMatrix Vsample;
		projectRandomSubmatrix(random, v, Vsample, v.size1()/5, v.size2()/5);
		projectFrequent(Vsample, 0);
		LOG4CXX_INFO(logger, "Sample matrix: "
			<< Vsample.data.size1() << " x " << Vsample.data.size2()
			<< ", " << Vsample.data.nnz() << " nonzeros");

		// generate initial factors by sampling from a uniform[-0.5,0.5] distribution
		DenseMatrix w(size1, r);
		DenseMatrixCM h(r, size2);
		generateRandom(w, random, boost::uniform_real<>(-0.5, 0.5));
		generateRandom(h, random, boost::uniform_real<>(-0.5, 0.5));

		// initialize
		Timer t;
		PsgdRunner psgdRunner(random);
		PsgdJob<Update,Regularize> psgdJob(v, w, h, update, regularize, order, tasks, shuffle);
		PsgdJob<UpdateL,Regularize> psgdJobLock(v, w, h, updateLock, regularize, order, tasks, shuffle);
		ParallelDecayAuto<Update,Regularize,Loss> decay(psgdJob, loss, Vsample, eps0, tasks);
		Trace trace;

		// print the test loss
		FactorizationData<> testData(vTest, w, h);
		LOG4CXX_INFO(logger, "Initial test loss: " << testLoss(testData));

		// run PSGD to try to reconstruct the original factors
		t.start();
//		psgdRunner.run(psgdJob, loss, epochs, decay, trace, balanceType, balanceMethod, &testData, &testLoss);
		psgdRunner.run(psgdJobLock, loss, epochs, decay, trace, balanceType, balanceMethod, &testData, &testLoss);

		t.stop();
		LOG4CXX_INFO(logger, "Total time: " << t);

		// print the test loss
		LOG4CXX_INFO(logger, "Final test loss: " << testLoss(testData));

		// write trace to an R file
		LOG4CXX_INFO(logger, "Writing trace to " << "/tmp/psgd-trace.R");
		trace.toRfile("/tmp/psgd-trace.R", "psgd");
	}

	mfStop();
	mfFinalize();

	return 0;
}
