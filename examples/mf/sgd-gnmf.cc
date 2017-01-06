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
 * Illustrates matrix factorization with SGD. We first creates factors and then a data matrix
 * from these factors. THis process ensures that we know the best factorization of the input.
 * We then try to reconstruct the factors using SGD.
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
	mf_size_type r = 10;

	// parameters for SGD
	double eps0 = 0.001;
	unsigned epochs = 10;
	mf_size_type testNnz = nnz/100;
	SgdOrder order = SGD_ORDER_WOR;
	typedef SlLoss Loss;
	Loss loss;
//	typedef UpdateAbs<UpdateNzsll> Update;
//	typedef RegularizeAbs<RegularizeSL> Regularize;
//	Update update = Update( (Update::Update()) );
//	Regularize regularize = Regularize( (Regularize::Regularize()) );
	typedef UpdateTruncate<UpdateSl> Update;
	typedef RegularizeTruncate<RegularizeSl> Regularize;
	Update update = Update( (Update::Update()), 0, 100 );
	Regularize regularize = Regularize( (Regularize::Regularize()), 0, 100 );

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

	// rescale initial factors to match frobenius norm
//	double l2v = l2(v);
//	double l2wh = sumOfSquares(w, h);
//	double f = std::sqrt(l2wh / l2v);


	// take a small sample and remove empty rows/columns
	ProjectedSparseMatrix Vsample;
	projectRandomSubmatrix(random, v, Vsample, v.size1()/5, v.size2()/5);
	projectFrequent(Vsample, 0);
	LOG4CXX_INFO(logger, "Sample matrix: "
		<< Vsample.data.size1() << " x " << Vsample.data.size2()
		<< ", " << Vsample.data.nnz() << " nonzeros");

	// initialize the SGD
	Timer t;
	SgdRunner sgdRunner(random);
	SgdJob<Update,Regularize> job(v, w, h, update, regularize, order);
	DecayAuto<Update,Regularize,Loss> decay(job, loss, Vsample, eps0, 8, 0.5, 1.05, false, true);
	Trace trace;

	// run SGD to try to reconstruct the original factors
	t.start();
	sgdRunner.run(job, loss, epochs, decay, trace);
	t.stop();
	LOG4CXX_INFO(logger, "Total time: " << t);

	// write trace to an R file
	LOG4CXX_INFO(logger, "Writing trace to " << "/tmp/sgd-gnmf-trace.R");
	trace.toRfile("/tmp/sgd-gnmf-trace.R", "sgd.gnmf");

	return 0;
}

