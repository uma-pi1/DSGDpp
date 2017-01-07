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
/**
 * creates synthetic data
 * (1) creates factors according to a distribution
 * (2) creates 2 train matrices
 * (3) creates 1 test file
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
	std::string outDir("/tmp/");



	// parameters for the factorization
	mf_size_type size1 = 100000; 
	mf_size_type size2 = 100000; 
	mf_size_type nnzSmall = 1000000; 
	mf_size_type nnzTest = 100000; 
	mf_size_type r = 10;



	double sigma = sqrt(10); // standard deviation for a Normal distribution N(0,10)



	// generate original factors by sampling from a N(0,10) distribution
	std::cout<<"Generating Solution factors..."<<std::endl;
	Random32 random; // note: this takes a default seed (not randomized!)
	DenseMatrix wIn(size1, r);
	DenseMatrixCM hIn(r, size2);
	generateRandom(wIn, random, boost::normal_distribution<>(0, sigma));
	generateRandom(hIn, random, boost::normal_distribution<>(0, sigma));
	//std::cout<<"Storing Solution factors..."<<std::endl;
	//writeMatrix(outDir+"solW.mma",wIn);
	//writeMatrix(outDir+"solH.mma",hIn);

//	std::cout<<"reading solution factors..."<<std::endl;
//	DenseMatrix wIn;
//	DenseMatrixCM hIn;
//	readMatrix(inDir+"solW.mma",wIn);
//	readMatrix(inDir+"solH.mma",hIn);

	// generate a sparse matrices by selecting random entries from the generated factors
	// and add small Gaussian noise
	SparseMatrix vSmall,vLarge;
	std::cout<<"Generating Small Matrix..."<<std::endl;
	generateRandom(vSmall, nnzSmall, wIn, hIn, random);
	std::cout<<"adding noise..."<<std::endl;
	addRandom(vSmall, random, boost::normal_distribution<>(0,1)); // the noise distribution is N(0,1)
	LOG4CXX_INFO(logger, "Small Data matrix: "
			<< vSmall.size1() << " x " << vSmall.size2() << ", " << vSmall.nnz() << " nonzeros");
	std::cout<<"Storing Small Matrix..."<<std::endl;
	writeMatrix(outDir+"train.mmc",vSmall);




	// create a test matrix (without noise)
	std::cout<<"Generating Test Matrix..."<<std::endl;
	SparseMatrix vTest;
	generateRandom(vTest, nnzTest, wIn, hIn, random);
	LOG4CXX_INFO(logger, "Test matrix: "
			<< vTest.size1() << " x " << vTest.size2() << ", " << vTest.nnz() << " nonzeros");
	std::cout<<"Storing Test Matrix..."<<std::endl;
	writeMatrix(outDir+"test.mmc",vTest);

	// generate initial factors by sampling from a uniform[-0.5,0.5] distribution
	std::cout<<"Generating Initial Factors..."<<std::endl;
	DenseMatrix w(size1, r);
	DenseMatrixCM h(r, size2);
	generateRandom(w, random, boost::uniform_real<>(-0.5, 0.5));
	generateRandom(h, random, boost::uniform_real<>(-0.5, 0.5));
	std::cout<<"Storing Initial Factors..."<<std::endl;
	writeMatrix(outDir+"W.mma",w);
	writeMatrix(outDir+"H.mma",h);

	return 0;
}
