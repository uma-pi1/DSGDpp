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
 * Illustrates matrix factorization with Lee's GKL method. We first creates factors and then a data matrix
 * from these factors. THis process ensures that we know the best factorization of the input.
 * We then try to reconstruct the factors.
 */
#include <iostream>

#include <boost/math/distributions/normal.hpp>
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
	mf_size_type size1 = 10000;
	mf_size_type size2 = 10000;
	mf_size_type nnz = 5000000;
	mf_size_type rank = 10;

	// parameters for Lee01
	unsigned epochs = 20;

	// generate original factors by sampling from a uniform[0,1] distribution
	Random32 random; // note: this takes a default seed (not randomized!)
	DenseMatrix wIn(size1, rank);
	DenseMatrixCM hIn(rank, size2);
	generateRandom(wIn, random,  boost::uniform_real<>(0, 1));
	generateRandom(hIn, random, boost::uniform_real<>(0, 1));
	// div2(wIn, sums2(wIn));
	// div1(hIn, sums1(hIn));

	// generate a sparse matrix by selecting random entries from the generated factors
	// and sample from a Poisson with mean equal to the entry
	// TODO: this generation process does not match the factorization model since we sample
	//       from the Poisson only at some entries of wh
	SparseMatrix v;
	generateRandom(v, nnz, wIn, hIn, random);
	applyPoisson(v, random);
	LOG4CXX_INFO(logger, "Data matrix: "
		<< v.size1() << " x " << v.size2() << ", " << v.nnz() << " nonzeros");
	LOG4CXX_INFO(logger, "Loss with original factors: " << gkl(v, wIn, hIn));

	// generate initial factors by sampling from a uniform[0,1] distribution
	DenseMatrix w(size1, rank);
	DenseMatrixCM h(rank, size2);
	generateRandom(w, random, boost::uniform_real<>(0, 1));
	generateRandom(h, random, boost::uniform_real<>(0, 1));
	div2(w, sums2(w));
	div1(h, sums1(h));
	double scaleFactor = sqrt(sum(v));
	mult(w, scaleFactor);
	mult(h, scaleFactor);

	// perform the factorization
	FactorizationData<> data(v, w, h);
	Trace trace;
	lee01Gkl(data, epochs, trace);

	// write the trace
	LOG4CXX_INFO(logger, "Writing trace to " << "/tmp/lee01-gkl-trace.R");
	trace.toRfile("/tmp/lee01-gkl-trace.R", "lee01.gkl");

	return 0;
}
