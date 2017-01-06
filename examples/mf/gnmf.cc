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
 * Illustrates matrix factorization with GNMF. */
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
#ifndef NDEBUG
	LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
#endif

	// parameters for the factorization
	mf_size_type size1 = 100; //10000;
	mf_size_type size2 = 100; //10000;
	mf_size_type nnz = 50; //5000000;
	mf_size_type r = 2; // 10

	// parameters for GNMF
	unsigned epochs = 1600;
	mf_size_type testNnz = nnz/100;

	BalanceType type = BALANCE_L2;;
	BalanceMethod method = BALANCE_OPTIMAL;

	// generate original factors by sampling from a uniform[0,1] distribution
	Random32 random; // note: this takes a default seed (not randomized!)
	DenseMatrix wIn(size1, r);
	DenseMatrixCM hIn(r, size2);
	generateRandom(wIn, random, boost::uniform_real<>(0,1));
	generateRandom(hIn, random, boost::uniform_real<>(0,1));

	// clear one row
	for (mf_size_type k=0; k<r; k++) {
		wIn(0,k) = 0;
	}

	// generate a sparse matrix by selecting random entries from the generated factors
	// and add small Gaussian noise
	SparseMatrix v;
	generateRandom(v, nnz, wIn, hIn, random);
	//addRandom(v, random, boost::normal_distribution<>(0, 0.1));
	LOG4CXX_INFO(logger, "Data matrix: "
		<< v.size1() << " x " << v.size2() << ", " << v.nnz() << " nonzeros");
	v.sort();
	//LOG4CXX_INFO(logger, "Loss with original factors: " << loss((FactorizationData<>(v, wIn, hIn))));
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
	generateRandom(w, random, boost::uniform_real<>(0,1));
	LOG4CXX_INFO(logger, "Row factors: " << w.size1() << " x " << w.size2());
	generateRandom(h, random, boost::uniform_real<>(0,1));
	LOG4CXX_INFO(logger, "Column factors: " << h.size1() << " x " << h.size2());


	// initialize
	FactorizationData<> data(v, w, h, 1, &vc);
	FactorizationData<> testJob(vTest,w,h);
	Trace trace;
	Timer t;

	LOG4CXX_INFO(logger, "Start");
	// run GNMF to try to reconstruct the original factors
	t.start();
	gnmf(data, epochs, trace, type, method, &testJob);
	t.stop();
	LOG4CXX_INFO(logger, "Total time: " << t);

	// write trace to an R file
	LOG4CXX_INFO(logger, "Writing trace to " << "/tmp/gnmf-trace.R");
	trace.toRfile("/tmp/gnmf-trace.R", "gnmf");

	return 0;
}

