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
/*
 * mfcreateInitialFactors.cc
 *
 *  Created on: Oct 4, 2011
 *      Author: chteflio
 */
#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <util/io.h>
#include <mpi2/mpi2.h>
#include <mf/mf.h>

#include "tools/parse.h"

using namespace std;
using namespace mf;
using namespace mpi2;
using namespace boost;
using namespace boost::program_options;
using namespace rg;

// program arguments
struct Args {
	mf_size_type size1, size2, rank;
	string values;
	unsigned seed;
	string outDir;
	Random32 random;
};

// generates random initial factors
struct Generate {
	Generate(Args& args) : args(args) {	};

	static void run(Args& args) {
		Generate f(args);
		parse::parseDistribution("values", args.values, f);
	}

	template<typename Dist>
	void operator()(Dist dist) {
		// generating initial row/col factors
		cout << "Generating Row and Col factors ..." << flush;
		DenseMatrix W0(args.size1, args.rank);
		DenseMatrixCM H0(args.rank, args.size2);
		generateRandom(W0, args.random, dist);
		generateRandom(H0, args.random, dist);
		cout << " done." << endl << "Writing factors..." << flush;
		cout << args.outDir+"W.mma    " << args.outDir+"H.mma" <<endl;
		writeMatrix(args.outDir+"W.mma", W0);
		writeMatrix(args.outDir+"H.mma", H0);
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
		("values", value<string>(&args.values), "distribution of values (e.g., \"Normal(0,1)\", \"Uniform(-1,1)\")")
		("rank", value<mf_size_type>(&args.rank), "rank of factorization")
		("seed", value<unsigned>(&args.seed), "seed for random number generator (if not set, system time is used)")
		("outDir", value<string>(&args.outDir), "output directory");
	;
	positional_options_description pdesc;
	pdesc.add("outDir", 1);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	// check required arguments
	if (vm.count("help") || vm.count("values")==0 || vm.count("size1")==0 || vm.count("size2")==0) {
		cerr << "Error: Options size1,size2,values,rank,outDir are required" << endl;
		cerr << "mfcreateInitialFactors [options]" << endl;
	    cerr << desc << endl;
	    exit(1);
	}

	// set default arguments
	if (vm.count("seed") == 0) args.seed = time(NULL);
	if (vm.count("rank") == 0) args.rank = 10;
	args.random = Random32(args.seed);

	// let's go (matrix currently generated in memory; if that turns out to be a problem
	// we may directly generate to a file)
	cout << "Output file : " << args.outDir << endl;
	cout << "Matrix size : " << args.size1 << "x" << args.size2 << endl;
	cout << "Rank        : " << args.rank << endl;
	cout << "Seed        : " << args.seed << endl;
	Generate::run(args);

	// everything OK
	return 0;
}




