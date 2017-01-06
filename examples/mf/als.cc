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
 * Illustrates matrix factorization with alternating least squares. We first creates factors and
 * then a data matrix
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

#include <mf/loss/nzsl.h>
#include <mf/loss/l2.h>

using namespace std;
using namespace mf;
using namespace rg;
using namespace boost::numeric::ublas;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

int main(int argc, char* argv[]) {
#ifndef NDEBUG
	LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
#endif

	// parameters for the factorization
	mf_size_type size1 = 10000;
	mf_size_type size2 = 10000;
	mf_size_type nnz = 1000000;
	double sigma = 1; // standard deviation
	double lambda = 1/sigma/sigma;
	mf_size_type r = 10;

	// parameters for ALS
	unsigned epochs = 20;
	AlsRegularizer regularizer = ALS_L2;
	typedef SumLoss<NzslLoss, L2Loss> Loss;
	typedef NzRmseLoss TestLoss;
	Loss loss((NzslLoss()), L2Loss(lambda));
	TestLoss testLoss;
	mf_size_type testNnz = nnz/100;

	BalanceType type = BALANCE_L2;;
	BalanceMethod method = BALANCE_SIMPLE;

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
	v.sort();
	LOG4CXX_INFO(logger, "Loss with original factors: " << loss((FactorizationData<>(v, wIn, hIn))));
	SparseMatrixCM vc;
	copyCm(v, vc);

	// create a test matrix (without noise)
	SparseMatrix vTest;
	generateRandom(vTest, testNnz, wIn, hIn, random);
	LOG4CXX_INFO(logger, "Test matrix: "
		<< v.size1() << " x " << v.size2() << ", " << vTest.nnz() << " nonzeros");

	// generate initial factors by sampling from a uniform[-0.5,0.5] distribution
	DenseMatrix w(size1, r);
	DenseMatrixCM h(r, size2);
	generateRandom(w, random, boost::uniform_real<>(-0.5, 0.5));
	generateRandom(h, random, boost::uniform_real<>(-0.5, 0.5));

	// print the test loss
	LOG4CXX_INFO(logger, "Initial test loss: "
			<< testLoss(FactorizationData<>(vTest,w,h)));

	FactorizationData<> testJob(vTest,w,h);

	// initialize
	FactorizationData<> data(v, w, h, 1, &vc);
	Trace trace;
	// here add fields to Trace
	//trace.addField("balancing-type", type);
	//trace.addField("balancing-method", method);
	Timer t;

	// run ALS to try to reconstruct the original factors
	t.start();
//	alsNzsl(data, epochs, trace, lambda, regularizer, rescale);
	alsNzsl(data, epochs, trace, lambda, regularizer, type, method ,&testJob);
	t.stop();
	LOG4CXX_INFO(logger, "Total time: " << t);

	// print the test loss
	LOG4CXX_INFO(logger, "Final test loss: "
			<< testLoss(FactorizationData<>(vTest,w,h)));

	// write trace to an R file
	string typeString, methodString;
	switch (type) {
	case BALANCE_NONE:
		typeString = "None";
		break;
	case BALANCE_L2:
		typeString = "L2";
		break;
	case BALANCE_NZL2:
		typeString = "Nzl2";
		break;
	}
	switch (method) {
	case BALANCE_SIMPLE:
		methodString = "Simple";
		break;
	case BALANCE_OPTIMAL:
		methodString = "Optimal";
		break;
	}
	string filename = "/tmp/als-trace.R";
	LOG4CXX_INFO(logger, "Writing trace to " << filename);
	trace.toRfile(filename, "als");

	return 0;
}

