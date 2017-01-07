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

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/uniform_real.hpp>

#include <mpi2/mpi2.h>
#include <mf/mf.h>


log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

using namespace std;
using namespace mf;
using namespace mpi2;
using namespace rg;
using namespace boost::numeric::ublas;

// type of SGD
typedef UpdateTruncate<UpdateNzslNzl2> Update;
typedef UpdateLock<Update> UpdateL;
typedef RegularizeNone Regularize;
typedef SumLoss<NzslLoss, Nzl2Loss> Loss;
typedef NzslLoss TestLoss;


struct TrainingPoint {

	TrainingPoint(){}
	TrainingPoint (mf_size_type i , mf_size_type j, double x):i_(i), j_(j), x_(x){}

	mf_size_type i_;
	mf_size_type j_;
	double x_;

};


//global variables
mf_size_type size1;
mf_size_type size2;
mf_size_type b1;
mf_size_type b2;

mf_size_type minSize1;
mf_size_type minSize2;
mf_size_type remainder1;
mf_size_type remainder2;

int calculateBlocking(double size1, double size2, double nnz, double rank, double cache){

	int SizeOfInt = 8;
	double alpha = cache * 1024;
	double beta = (-1)*(size1+size2)*rank*SizeOfInt;
	double gamma = (-1)*nnz*3*SizeOfInt;

	double delta = beta*beta-4*alpha*gamma;

	int b = ((-1)*beta+sqrt(delta))/(2*alpha);

	return b;  
}


// compare function for sort
inline bool compFunction (TrainingPoint point1, TrainingPoint point2) {

	mf_size_type block1_i = min ((mf_size_type)floor(point1.i_/minSize1), b1 -1);
	mf_size_type block2_i = min ((mf_size_type)floor(point2.i_/minSize1), b1 -1);
	mf_size_type block1_j = min ((mf_size_type)floor(point1.j_/minSize2), b2 -1);
	mf_size_type block2_j = min ((mf_size_type)floor(point2.j_/minSize2), b2 -1);
	bool result = false;
	if (block1_i < block2_i) {
		result = true;
	} else if (block1_i == block2_i) {
		if (block1_j < block2_j) {
			result = true;
		} else {
			result =false;
		}
	}
	return result;
}


int main(int argc, char* argv[]) {
  
	using namespace boost::program_options;
	// initialize mf library and mpi2
	boost::mpi::communicator& world = mfInit(argc, argv);



	// parameters for the factorization
	mf_size_type rank = 10;
	int tasks = 4;
	mf_size_type epochs = 10;
	SgdOrder order = SGD_ORDER_WOR;
	BalanceType balanceType = BALANCE_NONE;
	BalanceMethod balanceMethod = BALANCE_OPTIMAL;
	double lambda = 0.05;
	double eps0 = 0.0125;
	mf_size_type cache = 256;
	string traceFile,traceVar;
	
	// start mf library
	mfStart();

	if (world.rank() == 0)
	{
#ifndef NDEBUG
		LOG4CXX_WARN(logger, "Warning: Debug mode activated (runtimes may be slow).");
#endif

		string inputMatrixFile;
		string inputRowFacFile;
		string inputColFacFile;
		string outputRowFacFile;
		string outputColFacFile;
		string inputTestMatrixFile;
		string traceFile,traceVar;



		options_description desc("Options");
		desc.add_options()
				("help", "produce help message")
				("epochs", value<mf_size_type>(&epochs)->default_value(10), "number of epochs to run [10]")
				("rank", value<mf_size_type>(&rank)->default_value(10), "rank of factorization [10]")
				("cache", value<mf_size_type>(&cache)->default_value(256), "last level cache (in KB) available per core [256]")
				("lambda", value<double>(&lambda)->default_value(0.05), "lambda")
				("eps0", value<double>(&eps0)->default_value(0.01), "initial step size for BoldDriver")
				("tasks-per-rank", value<int>(&tasks)->default_value(1), "number of concurrent tasks [1]")
				("trace", value<string>(&traceFile)->default_value("trace.R"), "filename of trace [trace]")
				("traceVar", value<string>(&traceVar)->default_value("trace"), "variable name for trace [traceVar]")
				("input-file", value<string>(&inputMatrixFile), "input matrix")
				("input-test-file", value<string>(&inputTestMatrixFile), "input test matrix")
			    ("input-row-file", value<string>(&inputRowFacFile), "input initial row factor")
			    ("input-col-file", value<string>(&inputColFacFile), "input initial column factor")
			    ("output-row-file", value<string>(&outputRowFacFile), "output initial row factor")
			    ("output-col-file", value<string>(&outputColFacFile), "output initial column factor")
				;

		positional_options_description pdesc;
		pdesc.add("input-file", 1);
		pdesc.add("input-test-file", 2);
		pdesc.add("input-row-file", 3);
		pdesc.add("input-col-file", 4);

		variables_map vm;
		store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
		notify(vm);

		if (vm.count("help") || vm.count("input-file")==0) {
			cout << "CSGD with NZL2  [options] <input-file> " << endl << endl;
			cout << desc << endl;
			return 1;
		}

		if (vm.count("output-row-file") == 0) { outputRowFacFile = ""; }
		if (vm.count("output-col-file") == 0) { outputColFacFile = ""; }

		LOG4CXX_INFO(logger, "Using " << tasks << " parallel tasks");
		
		
		///////////////////////////////////////

		Random32 random; // note: this takes a default seed (not randomized!)

		SparseMatrix v;
		readMatrix(inputMatrixFile, v, MM_COORD);
		mf_size_type nnz = v.nnz();
		SparseMatrix::index_array_type& vIndex1 = rowIndexData(v);
		SparseMatrix::index_array_type& vIndex2 = columnIndexData(v);
		SparseMatrix::value_array_type& vValues = v.value_data();
		
		SparseMatrix vTest;
		readMatrix(inputTestMatrixFile, vTest, MM_COORD);

		DenseMatrix w;
		readMatrix(inputRowFacFile, w, MM_ARRAY);
		DenseMatrixCM h;
		readMatrix(inputColFacFile, h, MM_ARRAY);

		// global variables
		size1 = v.size1();
		size2 = v.size2();
		int b = calculateBlocking(size1, size2, nnz, rank, cache);
		b1 = b;
		b2 = b;
		
		LOG4CXX_INFO(logger, "Using a " << b << " x "<< b<<" stratification");
		
		
		minSize1 = floor(size1/b1);
		minSize2 = floor(size2/b2);
		remainder1 = size1 % b1;
		remainder2 = size2 % b2;

		LOG4CXX_INFO(logger, "Data matrix: "
			<< size1 << " x " << size2 << ", " << nnz << " nonzeros");
		LOG4CXX_INFO(logger, "minSize1: " << minSize1 << " remainder1: " << remainder1);
		LOG4CXX_INFO(logger, "minSize2: " << minSize2 << " remainder2: " << remainder2);

		// sort v
		LOG4CXX_INFO(logger, "Sorting v ... ");
		std::vector<TrainingPoint> points;
		for (mf_size_type i = 0; i < v.nnz(); i++) {
			TrainingPoint point = TrainingPoint(vIndex1[i], vIndex2[i], vValues[i]);
			points.push_back(point);
		}
		sort(points.begin(), points.end(), compFunction);

		// compute nnz per block
		LOG4CXX_INFO(logger, "Computing nnz per block ... ");
		std::vector<mf_size_type> nnzPerBlock;
		for (mf_size_type i = 0; i < (b1*b2); i++) nnzPerBlock.push_back(0);
		for (mf_size_type i = 0; i < points.size(); i++) {
			vIndex1[i] = points.at(i).i_;
			vIndex2[i] = points.at(i).j_;
			vValues[i] = points.at(i).x_;

			mf_size_type block_i = min ((mf_size_type)floor(points.at(i).i_/minSize1), b1 -1);
			mf_size_type block_j = min ((mf_size_type)floor(points.at(i).j_/minSize2), b2 -1);
			mf_size_type blockId = (b2 * block_i) + block_j;

			if ((block_i > b1) || (block_j > b2)) {
				LOG4CXX_INFO(logger, "block_i " << block_i << " = floor( " << points.at(i).i_ << " / " << minSize1);
				LOG4CXX_INFO(logger, "block_j " << block_j << " = floor( " << points.at(i).j_ << " / " << minSize2);
			}
			nnzPerBlock[blockId]++;
		}
		LOG4CXX_INFO(logger, "nnzPerBlock of size: " << nnzPerBlock.size());

		// compute permutation
		LOG4CXX_INFO(logger, "Computing permutation ... ");
		std::vector<mf_size_type> permutation;
		for (mf_size_type i = 0; i < v.nnz(); i++) {
			permutation.push_back(i);
		}

		// compute offsets
		LOG4CXX_INFO(logger, "Computing offsets ... ");
		std::vector<mf_size_type> offsets((b1*b2)+1); offsets.clear();
		mf_size_type offset = 0;
		offsets.push_back(offset);
		for (mf_size_type i = 1; i < (b1*b2)+1; i++) {
			offset += nnzPerBlock[i-1];
			offsets.push_back(offset);
		}


		Update update = Update(UpdateNzslNzl2(lambda), -100, 100); // truncate for numerical stability
		Regularize regularize;
		Loss loss((NzslLoss()), Nzl2Loss(lambda));
		TestLoss testLoss;
		
		FactorizationData<> testData(vTest, w, h);
		LOG4CXX_INFO(logger, "Initial test loss: " << testLoss(testData));

		// initialize
		Timer t;
		StratifiedPsgdRunner stratifiedPsgdRunner (random, permutation, offsets);
		StratifiedPsgdJob<Update,Regularize> stratifiedPsgdJob(v, w, h, update, regularize, order, tasks);
		BoldDriver decay(eps0);
		Trace trace;

		t.start();
		stratifiedPsgdRunner.run(stratifiedPsgdJob, loss, epochs, decay, trace, balanceType, balanceMethod, &testData, &testLoss);
		t.stop();
		LOG4CXX_INFO(logger, "Total time: " << t);

		// write trace to an R file
		LOG4CXX_INFO(logger, "Writing trace to " << traceFile);
		trace.toRfile(traceFile, traceVar);
		
					// write computed factors to file
				if (outputRowFacFile.length() > 0) {
					LOG4CXX_INFO(logger, "Writing row factors to " << outputRowFacFile);
					//DenseMatrix w0;
					//unblock(dw, w0);
					writeMatrix(outputRowFacFile, w);
				}
				if (outputColFacFile.length() > 0) {
					LOG4CXX_INFO(logger, "Writing column factors to " << outputColFacFile);
					//DenseMatrixCM h0;
					//unblock(dh, h0);
					writeMatrix(outputColFacFile, h);
				}
		/**/
	}

	mfStop();
	mfFinalize();

	return 0;
}
