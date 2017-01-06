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
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <util/io.h>
#include <mpi2/mpi2.h>
#include <mf/mf.h>

#include "parse.h"

using namespace std;
using namespace mf;
using namespace mpi2;
using namespace boost;
using namespace boost::program_options;
using namespace rg;

// program arguments
struct Args {
	mf_size_type size1, size2, nnz;
	string values;
	unsigned seed;
	string outFilename;
	MatrixFileFormat format;
	Random32 random;
};

// generates a matrix of type M
template<typename M>
struct Generate {
	Generate(Args& args) : args(args) {	};

	static void run(Args& args) {
		Generate<M> f(args);
		parse::parseDistribution("values", args.values, f);
	}

	template<typename Dist>
	void operator()(Dist dist) {
		M m(args.size1, args.size2);
		cout << "Generating matrix..." << flush;
		generateRandom(m, args.nnz, args.random, dist);
		cout << " done." << endl << "Writing matrix..." << flush;
		writeMatrix(args.outFilename, m);
		cout << " done." << endl;
	};

	Args& args;
};

// main program
int main(int argc, char *argv[]) {
	Args args;

	// parse command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
		("size1", value<mf_size_type>(&args.size1), "number of rows to generate")
		("size2", value<mf_size_type>(&args.size2), "number of columns to generate")
		("nnz", value<mf_size_type>(&args.nnz), "number of non-zero entries to generate (full matrix if not set)")
		("values", value<string>(&args.values), "distribution of values (e.g., \"Normal(0,1)\", \"Uniform(-1,1)\")")
		("seed", value<unsigned>(&args.seed), "seed for random number generator (if not set, system time is used)")
		("output-file", value<string>(&args.outFilename), "output file");
	;
	positional_options_description pdesc;
	pdesc.add("output-file", 1);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	// check required arguments
	if (vm.count("help") || vm.count("values")==0 || vm.count("size1")==0 || vm.count("size2")==0) {
		cerr << "Error: Options size1,size2,values,output-file are required" << endl;
		cerr << "mfgenerate [options]" << endl;
	    cerr << desc << endl;
	    exit(1);
	}

	// set default arguments
	if (vm.count("nnz") == 0) args.nnz = args.size1*args.size2;
	if (vm.count("seed") == 0) args.seed = time(NULL);
	args.format = getMatrixFormat(args.outFilename);
	args.random = Random32(args.seed);

	// let's go (matrix currently generated in memory; if that turns out to be a problem
	// we may directly generate to a file)
	cout << "Output file : " << args.outFilename << endl;
	cout << "Matrix size : " << args.size1 << "x" << args.size2 << ", " << args.nnz << " nonzero entries"  << endl;
	cout << "Seed        : " << args.seed << endl;
	if (isSparse(args.format)) {
		cout << "Matrix type : sparse (format " << getName(args.format) << ")" << endl;
		Generate<SparseMatrix>::run(args);
	} else {
		cout << "Matrix type : dense (format " << getName(args.format) << ")" << endl;
		Generate<DenseMatrix>::run(args);
	}

	// everything OK
	return 0;
}
