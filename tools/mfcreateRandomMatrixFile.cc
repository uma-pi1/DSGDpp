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
 * mfcreateSeedFile.cc
 *
 *  Created on: May 9, 2012
 *      Author: chteflio
 *
 * This program creates a file descriptor for generating synthetic matrices on the fly
 * in a parallel or distributed manner
 *
 * For a given experiment you will need 2 such files:
 * (i) a file containing the original factors + the data matrix + the test matrix (optionally)
 * (ii) a file containing the starting points (initial factors)
 *
 * Before creating a file take into consideration the following:
 *
 * (1) parallelization happens in row-chunk manner. Please make sure that the blocks1 dimension
 * is enough fine-grained, so that the threads of each node will have enough row-blocks
 * to work on
 *
 * (2) make sure that the blocks1, blocks2 values are at least as much as the most
 * fine-blocked matrix that you want your file to be able to create
 *
 * (3) blocks_in_file mod blocks_of_the_matrix = 0        !!!
 *
 * (4) rows/columns_of_the_matrix mod blocks_in_file = 0  !!!
 *
 *
 */


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
// if nnz and nnzTest=0 then I am only interested in creating initial W and H
struct Args {
	mf_size_type size1, size2, nnz,nnzTest,rank;
	int blocks1,blocks2;
	string values,noise;
	string outFilename;
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
		("nnz", value<mf_size_type>(&args.nnz)->default_value(0), "number of non-zero entries to generate")
		("nnzTest", value<mf_size_type>(&args.nnzTest)->default_value(0), "number of test entries to generate ")
		("rank", value<mf_size_type>(&args.rank), "rank for the Worig and Horig ")
		("values", value<string>(&args.values), "distribution of values (e.g., \"Normal(0,1)\", \"Uniform(-1,1)\")")
		("noise", value<string>(&args.noise), "distribution of noise (e.g., \"Normal(0,1)\", \"Uniform(-1,1)\")")
		("blocks1", value<int>(&args.blocks1), "number of row-blocks to generate")
		("blocks2", value<int>(&args.blocks2), "number of column-blocks to generate")
		("output-file", value<string>(&args.outFilename), "output file (extension .rm)")
	;
	positional_options_description pdesc;
	pdesc.add("output-file", 1);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	// check required arguments
	if (vm.count("help") || vm.count("values")==0 || vm.count("size1")==0 || vm.count("size2")==0 ||
			vm.count("blocks1")==0||vm.count("blocks2")==0) {
		cerr << "Error: Options size1,size2,blocks1,blocks2,values,output-file are required" << endl;
		cerr << "mfcreateSeedFile [options]" << endl;
	    cerr << desc << endl;
	    exit(1);
	}

	if (!mf::detail::endsWith(args.outFilename,".rm")){
		cerr << "the output-file should have extension .rm" << endl;
		exit(1);
	}

	std::cout<<"create the seedDescriptor"<<std::endl;
	RandomMatrixDescriptor rmd;
	rmd=RandomMatrixDescriptor::create(args.size1,args.size2,args.blocks1,args.blocks2,args.nnz, args.nnzTest, args.rank, args.values, args.noise);
	std::cout<<"save the seedDescriptor file: "<<args.outFilename<<std::endl;
	rmd.save(args.outFilename);

	// everything OK
	return 0;
}


