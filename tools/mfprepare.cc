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
 * mfprepare.cc
 *
 *  Created on: Oct 20, 2011
 *      Author: chteflio
 */

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <mf/mf.h>

using namespace std;
using namespace mf;
using namespace boost;
using namespace boost::program_options;
using namespace rg;

/*
 * This tool prepares a SPARSE matrix stored in file 'input-file' for factorization.
 * (1) It samples the input matrix
 * (2) It projects out the zero row and columns
 * (3) It creates initial factors for a specific rank and according to some distribution
 *
 *  Example call ./mfprepare --input-file=/someDir/v.mmc --nnz=10000 --values="Uniform(0,1)" --rank=10
 *
 */

int main(int argc, char *argv[]){

	//potential arguments for sampling
	mf_size_type size1, size2, nnz;
	double fsize1,fsize2;
	unsigned seed;
	string matrixFile,sampleFile,extension;

	//potential arguments for projecting
	mf_size_type threshold;
	bool repeat;

	//potential arguments for generating factors
	string values,wFile,hFile;
	mf_size_type rank;

	// read command line
	options_description desc("Options");
	desc.add_options()
						("help", "produce help message")
						("input-file", value<string>(&matrixFile), "input file (SPARSE MATRIX)")
						("size1", value<mf_size_type>(&size1), "number of rows for sample matrix")
						("size2", value<mf_size_type>(&size2), "number of columns for sample matrix")
						("fsize1", value<double>(&fsize1), "fraction of the rows of the initial matrix to generate for sample matrix e.g. 0.6")
						("fsize2", value<double>(&fsize2), "fraction of the columns of the initial matrix to generate for sample matrix e.g. 0.6")
						("nnz", value<mf_size_type>(&nnz), "number of non-zero entries to generate for sample matrix")
						("seed", value<unsigned>(&seed), "seed for random number generator (if not set, system time is used)")
						//("format", value<string>(&extension), "file format of the data file (default: one of the matrix market formats)")
						("output-sample-file", value<string>(&sampleFile), "output file for sample matrix(if .mfp write also mappings)")
						("threshold", value<mf_size_type>(&threshold), "keep only rows and columns of the sample matrix whose nonzero entries are above threshold (default value is 0)")
						("repeat", value<bool>(&repeat), "repeatedly projects out rows and columns that do not pass the threshold until all of them do")
						("values", value<string>(&values), "distribution of values for the factors(e.g., \"Normal(0,1)\", \"Uniform(-1,1)\")")
						("rank", value<mf_size_type>(&rank), "rank for the factors")
						("output-row-file", value<string>(&wFile), "output file")
						("output-col-file", value<string>(&hFile), "output file")
						;



	positional_options_description pdesc;
	pdesc.add("input-file", 1);
	pdesc.add("output-sample-file", 2);
	pdesc.add("output-row-file", 3);
	pdesc.add("output-col-file", 4);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	// check required arguments
	if (vm.count("help") || vm.count("values")==0 ||vm.count("input-file")==0|| vm.count("rank")==0) {
		cerr << "Error: Options input-file, values are required" << endl;
		cerr << "mfprepare [options]" << endl;
		cerr << desc << endl;
		exit(1);
	}

	string commandPath=argv[0];
	commandPath=commandPath.substr(0,commandPath.rfind("/mfprepare"));

	boost::filesystem::path path(matrixFile);
	string outFile = path.filename().string();
	string base=outFile.substr(0,outFile.rfind("."));
	string outDir = path.parent_path().string();

	if (outDir.length()!=0)	outDir+="/";

	cout<<"reading matrix details..."<<endl;
	SparseMatrix m;
	// read the sizes of the matrix
	mf::detail::readMmCoord(
			matrixFile,
			boost::bind(mf::detail::mmInit<SparseMatrix>, boost::ref(m), false, _1, _2, _3),
			boost::bind(mf::detail::mmCheckProcess<SparseMatrix>, boost::ref(m), _1, _2),
			boost::bind(mf::detail::mmProcess<SparseMatrix>, boost::ref(m), _1, _2, _3),
			boost::bind(mf::detail::mmFreeze<SparseMatrix>, boost::ref(m))
			);

	size1=m.size1();
	size2=m.size2();

	// SAMPLING with mfsample
	std::stringstream mfsampleArgs;

	mfsampleArgs<<commandPath<<"/mfsample  --input-file="<<matrixFile;

	if (vm.count("size1")!=0) mfsampleArgs<<" --size1="<<size1;
	if (vm.count("size2")!=0) mfsampleArgs<<" --size2="<<size2;
	if (vm.count("fsize1")!=0) mfsampleArgs<<" --fsize1="<<fsize1;
	if (vm.count("fsize2")!=0) mfsampleArgs<<" --fsize2="<<fsize2;
	if (vm.count("nnz")!=0) mfsampleArgs<<" --nnz="<<nnz;
	if (vm.count("seed")!=0) mfsampleArgs<<" --seed="<<seed;
	//if (vm.count("format")!=0) mfsampleArgs<<" --format="<<extension;
	mfsampleArgs<<" --output-file=";
	if (vm.count("output-sample-file")==0){
		sampleFile=outDir+base+"-sample.mfp";
	}
	mfsampleArgs<<sampleFile;

	//cout<<mfsampleArgs.str()<<endl;

	//run mfsample
	std::system(mfsampleArgs.str().c_str());


	// PROJECTING with mfproject
	std::stringstream mfprojectArgs;

	mfprojectArgs<<commandPath<<"/mfproject --input-file="<<sampleFile<<" --output-file="<<sampleFile;
	if (vm.count("threshold")!=0) mfsampleArgs<<" --threshold="<<threshold;
	if (vm.count("repeat")!=0) mfsampleArgs<<" --repeat="<<repeat;
	//run mfproject
	cout<<"Projecting..."<<endl;
	//cout<<mfsampleArgs.str()<<endl;
	std::system(mfprojectArgs.str().c_str());


	// GENERATING FACTOR W with mfgenerate
	std::stringstream mfgenerateArgsW;

	mfgenerateArgsW<<commandPath<<"/mfgenerate --size1="<<size1<<" --size2="<<rank<<" --values=\""<<values<<"\"";

	if (vm.count("seed")!=0) mfgenerateArgsW<<" --seed="<<seed;
	mfgenerateArgsW<<" --output-file=";
	if (vm.count("output-row-file")!=0)
		mfgenerateArgsW<<wFile;
	else
		mfgenerateArgsW<<outDir<<base<<"-w0.mma";

	//run mfgenerate for W
	cout<<"Generating W..."<<endl;
	std::system(mfgenerateArgsW.str().c_str());

	// GENERATING FACTOR H with mfgenerate
	std::stringstream mfgenerateArgsH;

	mfgenerateArgsH<<commandPath<<"/mfgenerate --size2="<<size2<<" --size1="<<rank<<" --values=\""<<values<<"\"";

	if (vm.count("seed")!=0) mfgenerateArgsH<<" --seed="<<seed;
	mfgenerateArgsH<<" --output-file=";
	if (vm.count("output-col-file")!=0)
		mfgenerateArgsH<<hFile;
	else
		mfgenerateArgsH<<outDir<<base<<"-h0.mma";

	//run mfgenerate for H
	cout<<"Generating H..."<<endl;
	std::system(mfgenerateArgsH.str().c_str());

	return 0;
}





