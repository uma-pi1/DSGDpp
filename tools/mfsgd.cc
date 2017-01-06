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
// TODO: add all kinds of arguments


#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/random/uniform_real.hpp>

#include <util/evaluation.h>
#include <mf/mf.h>

using namespace std;
using namespace mf;
using namespace rg;
using namespace boost::program_options;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

int main(int argc, char *argv[]) {
	string traceFile;
	string traceVar;
	string inputMatrixFile;
	mf_size_type epochs;

	// read command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
		("epochs", value<mf_size_type>(&epochs)->default_value(10), "number of epochs to run [10]")
		("trace", value<string>(&traceFile)->default_value("trace.R"), "filename of trace [trace.R]")
		("traceVar", value<string>(&traceVar)->default_value("trace"), "variable name for trace [traceVar]")
	    ("input-file", value<string>(&inputMatrixFile), "input matrix")
	    ;

	positional_options_description pdesc;
	pdesc.add("input-file", 1);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	if (vm.count("help") || vm.count("input-file")==0) {
		cout << "mfsgd [options] <input-file> <sample-file>" << endl << endl;
	    cout << desc << endl;
	    return 1;
	}

	// read input matrix
	SparseMatrix v;
	readMatrix(inputMatrixFile, v);
	LOG4CXX_INFO(logger, "Data matrix: "
		<< v.size1() << " x " << v.size2() << ", " << v.nnz() << " nonzeros");
	mf_size_type size1 = v.size1();
	mf_size_type size2 = v.size2();
	mf_size_type r = 50;

	// compute sample matrix
	Random32 random; // note: this takes a default seed (not randomized!)
	ProjectedSparseMatrix Vsample;
	projectRandomSubmatrix(random, v, Vsample, v.size1()/10, v.size2()/10);
	projectFrequent(Vsample, 0);
	LOG4CXX_INFO(logger, "Sample matrix: "
		<< Vsample.data.size1() << " x " << Vsample.data.size2()
		<< ", " << Vsample.data.nnz() << " nonzeros");

	// generate initial factors by sampling from a uniform[-0.5,0.5] distribution
	DenseMatrix w(size1, r);
	DenseMatrixCM h(r, size2);
	generateRandom(w, random, boost::uniform_real<>(-0.5, 0.5));
	generateRandom(h, random, boost::uniform_real<>(-0.5, 0.5));

	// parameters for SGD
	double epsMax = 0.1;
	double lambda = 50;
	typedef UpdateTruncate<UpdateNzsl> Update;
	typedef RegularizeL2 Regularize;
	typedef SumLoss<NzslLoss, L2Loss> Loss;
	Update update = Update(UpdateNzsl(), -1000, 10100); // truncate for numerical stability
	Regularize regularize = Regularize(lambda);
	Loss loss((NzslLoss()), L2Loss(lambda));

	// initialize the SGD
	Timer t;
	SgdRunner sgdRunner(random);
	SgdJob<Update,Regularize> job(v, w, h, update, regularize, SGD_ORDER_WOR);
	DecayAuto<Update,Regularize,Loss> decay(job, loss, Vsample, epsMax, 7);
	Trace trace;

	// run SGD
	t.start();
	sgdRunner.run(job, loss, epochs, decay, trace, BALANCE_NONE, BALANCE_SIMPLE);
	t.stop();
	LOG4CXX_INFO(logger, "Total time: " << t);


	// write trace to an R file
	LOG4CXX_INFO(logger, "Writing trace to " << traceFile);
	trace.toRfile(traceFile, traceVar);

	return 0;
}
