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
 * Illustrates matrix factorization with ASGD. We first creates factors and then a data matrix
 * from these factors. This process ensures that we know the best factorization of the input.
 * These matrices are distributed across a cluster. We then try to reconstruct the factors
 * using ASGD.
 *
 * Run with: mpirun --hosts localhost,localhost asgd
 * (make sure to use a production build, otherwise it will be slow)
 */
#include <iostream>

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
//typedef UpdateTruncate<UpdateNzsl> Update;
typedef UpdateTruncate<UpdateNzslL2> Update;
//typedef UpdateTruncate<UpdateNzsl> Update;
typedef RegularizeNone Regularize;
//typedef RegularizeL2 Regularize;
//typedef RegularizeNzl2 Regularize;
//typedef NzslLoss Loss;
typedef SumLoss<NzslLoss, L2Loss> Loss;
//typedef SumLoss<NzslLoss, Nzl2Loss> Loss;
typedef NzslLoss TestLoss;

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
	int tasksPerRank = 2;
	mf_size_type blocks = world.size();

	// parameters for SGD
	double eps0 = 0.0025;
	mf_size_type epochs = 10;
	SgdOrder order = SGD_ORDER_WOR;
	StratumOrder stratumOrder = STRATUM_ORDER_WOR;
	Update update = Update(UpdateNzslL2(lambda), -10*sigma, 10*sigma); // truncate for numerical stability
//	Update update = Update(UpdateNzsl(), -10*sigma, 10*sigma); // truncate for numerical stability
	Regularize regularize;
//	Regularize regularize = Regularize(lambda);
	Loss loss((NzslLoss()), L2Loss(lambda));
//	Loss loss;
//	Loss loss((NzslLoss()), Nzl2Loss(lambda));
	TestLoss testLoss;
	mf_size_type testNnz = nnz/100;
	BalanceType balanceType = BALANCE_NONE;
	BalanceMethod balanceMethod = BALANCE_OPTIMAL;
	bool averageDeltas = true;

	// start mf library
	mpi2::registerTask<mf::detail::AsgdTask<Update, Regularize> >();
	mfStart();

	if (world.rank() == 0)
	{
#ifndef NDEBUG
		LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
#endif

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

		// distribute the input matrices and test matrix (cheaper than fetching W/H)
		DistributedSparseMatrix dv = distributeMatrix("V", blocks, 1, true, v);
		LOG4CXX_INFO(logger, "Distributed data matrix: "
					<< dv.blocks1() << " x " << dv.blocks2() << " blocks");
		DistributedSparseMatrix dvTest = distributeMatrix("VTest", blocks, 1, true, vTest);
		DistributedDenseMatrix dw = distributeMatrix("W", blocks, 1, true, w);

		DistributedDenseMatrixCM dh = distributeMatrix("H", 1, blocks, false, h);
		LOG4CXX_INFO(logger, "Distributed factor matrices");

		// initialize the DSGD
		Timer t;
		AsgdRunner asgdRunner(random);
		AsgdJob<Update,Regularize> asgdJob(dv, dw, dh, update, regularize, order,
				stratumOrder, tasksPerRank, averageDeltas);
		BoldDriver decay(eps0);
		//DistributedDecayAuto<Update,Regularize,Loss> decay(asgdJob, loss, Vsample, "decay", eps0,
				//world.size()*tasksPerRank);
		Trace trace;

		//std::string s1("Loss"),s2("nzsl"),s3("Regularize"),s4("L2"),s5("nodes"),s6("threads");

		trace.addField("Loss","nzsl");
		trace.addField("Regularize","L2");

		trace.addField("nodes",3);
		trace.addField("threads",4);


		// print the test loss
		AsgdFactorizationData<> testData(dvTest,dw,dh,tasksPerRank);

		// run DSGD to try to reconstruct the original factors
		t.start();
		asgdRunner.run(asgdJob, loss, epochs, decay, trace, balanceType, balanceMethod, &testData, &testLoss);

		t.stop();
		LOG4CXX_INFO(logger, "Total time: " << t);

		// write trace to an R file
		LOG4CXX_INFO(logger, "Writing trace to " << "/tmp/asgd-trace.R");
		trace.toRfile("/tmp/asgd-trace.R", "asgd");
	}

	mfStop();
	mfFinalize();

	return 0;
}
