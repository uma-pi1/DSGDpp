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
 * Illustrates matrix factorization with DSGD. We first creates factors and then a data matrix
 * from these factors. This process ensures that we know the best factorization of the input.
 * These matrices are distributed across a cluster. We then try to reconstruct the factors
 * using DSGD.
 *
 * Run with: mpirun --hosts localhost,localhost dsgd
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
typedef UpdateTruncate<UpdateSl> Update;
typedef RegularizeTruncate<RegularizeSl> Regularize;
typedef SlLoss Loss;

int main(int argc, char* argv[]) {
	// initialize mf library and mpi2
	boost::mpi::communicator& world = mfInit(argc, argv);

	// parameters for the factorization
	mf_size_type size1 = 10000;
	mf_size_type size2 = 10000;
	mf_size_type nnz = 5000000;
	mf_size_type r = 10;

	// parameters for distribution
	int tasksPerRank = 2;
	mf_size_type blocks1 = world.size() * tasksPerRank;
	mf_size_type blocks2 = world.size() * tasksPerRank;

	// parameters for SGD
	double epsMax = 0.001;
	mf_size_type epochs = 20;
	SgdOrder order = SGD_ORDER_WOR;
	StratumOrder stratumOrder = STRATUM_ORDER_WOR;
	Update update = Update((UpdateSl()), 0, 100);
	Regularize regularize((RegularizeSl()), 0, 100);
	Loss loss;

	// start mf library
	// TODO: need automatic registration
	mfStart();

	if (world.rank() == 0)
	{
#ifndef NDEBUG
		LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
#endif

		// TODO: distribute matrix generation
		// generate original factors by sampling from a uniform(0,1) distribution
		Random32 random; // note: this takes a default seed (not randomized!)
		DenseMatrix wIn(size1, r);
		DenseMatrixCM hIn(r, size2);
		generateRandom(wIn, random, boost::uniform_real<>(0,1));
		generateRandom(hIn, random, boost::uniform_real<>(0,1));

		// generate a sparse matrix by selecting random entries from the generated factors
		// and sample from a Poisson with mean equal to the entry
		// TODO: this generation process does not match the factorization model since we sample
		//       from the Poisson only at some entries of wh
		SparseMatrix v;
		generateRandom(v, nnz, wIn, hIn, random);
		LOG4CXX_INFO(logger, "Data matrix: "
			<< v.size1() << " x " << v.size2() << ", " << v.nnz() << " nonzeros");
		LOG4CXX_INFO(logger, "Loss with original factors: " << loss((FactorizationData<>(v, wIn, hIn))));

		// take a small sample and remove empty rows/columns
		ProjectedSparseMatrix Vsample;
		projectRandomSubmatrix(random, v, Vsample, v.size1()/5, v.size2()/5);
		projectFrequent(Vsample, 0);
		LOG4CXX_INFO(logger, "Sample matrix: "
			<< Vsample.data.size1() << " x " << Vsample.data.size2()
			<< ", " << Vsample.data.nnz() << " nonzeros");

		// generate initial factors by sampling from a uniform[0,1] distribution
		DenseMatrix w(size1, r);
		DenseMatrixCM h(r, size2);
		generateRandom(w, random, boost::uniform_real<>(0, 1));
		generateRandom(h, random, boost::uniform_real<>(0, 1));

		// distribute the input matrices
		DistributedSparseMatrix dv = distributeMatrix("V", blocks1, blocks2, true, v);
		LOG4CXX_INFO(logger, "Distributed data matrix: "
					<< dv.blocks1() << " x " << dv.blocks2() << " blocks");
		DistributedDenseMatrix dw = distributeMatrix("W", blocks1, 1, true, w);
		DistributedDenseMatrixCM dh = distributeMatrix("H", 1, blocks2, false, h);
		LOG4CXX_INFO(logger, "Distributed factor matrices");

		// initialize the DSGD
		Timer t;
		DsgdRunner dsgdRunner(random);
		DsgdJob<Update,Regularize> dsgdJob(dv, dw, dh, update, regularize, order,
				stratumOrder, false, tasksPerRank);
		DistributedDecayAuto<Update,Regularize,Loss> decay(dsgdJob, loss, Vsample, "decay", epsMax,
				world.size()*tasksPerRank);
		Trace trace;

		// run DSGD to try to reconstruct the original factors
		t.start();
		dsgdRunner.run(dsgdJob, loss, epochs, decay, trace);
		t.stop();
		LOG4CXX_INFO(logger, "Total time: " << t);

		// write trace to an R file
		LOG4CXX_INFO(logger, "Writing trace to " << "/tmp/dsgd-gkl-trace.R");
		trace.toRfile("/tmp/dsgd-gkl-trace.R", "dsgd.gkl");
	}

	mfStop();
	mfFinalize();

	return 0;
}
