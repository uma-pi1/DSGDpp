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
// projections of matrics
#include <iostream>

#include <boost/random/uniform_real.hpp>
#include <boost/math/distributions/normal.hpp>

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
	mfStart();

	if (world.rank() == 0)
	{
#ifndef NDEBUG
		LOG4CXX_WARN(logger, "Debug mode activated (runtimes may be slow).");
#endif
		Random32 random;
		unsigned tasksPerRank = 2;
		mf_size_type size1 = 200;
		mf_size_type size2 = 100;
		mf_size_type blocks1 = world.size();
		mf_size_type blocks2 = tasksPerRank;
		mf_size_type nnz = 500;

		// generate a random data matrix without empty rows/columns
		SparseMatrix v(size1, size2);
		generateRandom(v, nnz, random, boost::normal_distribution<>(0, 1));
		projectFrequent(v, 0);
		size1 = v.size1(); size2 = v.size2();
		LOG4CXX_INFO(logger, "Data matrix: "
			<< v.size1() << " x " << v.size2() << ", " << v.nnz() << " nonzeros");

		// take a small sample and remove empty rows/columns
		ProjectedSparseMatrix Vsample;
		projectRandomSubmatrix(random, v, Vsample, v.size1()/3, v.size2()/3);
		projectFrequent(Vsample, 0);
		LOG4CXX_INFO(logger, "Sample matrix: "
			<< Vsample.data.size1() << " x " << Vsample.data.size2()
			<< ", " << Vsample.data.nnz() << " nonzeros");

		// generate factor matrices
		mf_size_type r = 10;
		DenseMatrix w(size1, r);
		DenseMatrixCM h(r, size2);
		generateRandom(w, random, boost::uniform_real<>(-0.5, 0.5));
		generateRandom(h, random, boost::uniform_real<>(-0.5, 0.5));

		// extract the rows and columns corresponding to the sample
		DenseMatrix wSample;
		DenseMatrixCM hSample;
		project1(w, wSample, Vsample.map1);
		project2(h, hSample, Vsample.map2);
		LOG4CXX_INFO(logger, "Projected row factors:  " << wSample);
		LOG4CXX_INFO(logger, "Projected column factors:  " << hSample);

		// distribute the factor matrices
		DistributedDenseMatrix dw = distributeMatrix("w", blocks1, 1, true, w);
		DistributedDenseMatrixCM dh = distributeMatrix("h", 1, blocks2, false, h);

		// extract the same rows/columns from the distributed matrix
		DenseMatrix dwSample;
		DenseMatrixCM dhSample;
		project1(dw, dwSample, Vsample.map1);
		project2(dh, dhSample, Vsample.map2);
		LOG4CXX_INFO(logger, "Projected row factors:  " << dwSample);
		LOG4CXX_INFO(logger, "Projected column factors:  " << dhSample);
	}

	mfStop();
	mfFinalize();

	return 0;
}

