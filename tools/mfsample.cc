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
 * mfsample.cc
 *
 *  Created on: Jul 13, 2011
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
 * This tool samples a SPARSE matrix stored in file 'input-file'
 * It decides the size of the projected matrix according to the values:
 * i) size1, size2
 * ii) fsize1, fsize2 (if i) is not present)
 * iii) nnz
 *
 * If the output file has extension .mfp, it also creates a file descriptor and stores
 * the mappings for rows and columns between the original and sampled matrix
 *
 * If the output has any other extension of matrix, it only stores the matrix' data
 *
 * This tool does NOT eliminate empty rows and columns from the resulting projected matrix
 */

int main(int argc, char *argv[]){
	// parse the command line
	mf_size_type size1, size2, nnz;
	double fsize1;
	double fsize2;
	unsigned seed;
	std::string inFilename,outFilename,extension, dataFile;



	// read command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
		("input-file", value<string>(&inFilename), "input file")
		("size1", value<mf_size_type>(&size1), "number of rows to generate")
		("size2", value<mf_size_type>(&size2), "number of columns to generate")
		("fsize1", value<double>(&fsize1), "fraction of the rows of the initial matrix to generate e.g. 0.6")
		("fsize2", value<double>(&fsize2), "fraction of the columns of the initial matrix to generate e.g. 0.6")
		("nnz", value<mf_size_type>(&nnz), "number of non-zero entries to generate")
		("seed", value<unsigned>(&seed), "seed for random number generator (if not set, system time is used)")
//		("format", value<string>(&extension)->default_value(""), "file format of blocks (default: one of the matrix market formats)")
		("output-file", value<string>(&outFilename), "output file (if .mfp write also mappings)");
	;

	positional_options_description pdesc;
	pdesc.add("input-file", 1);
	pdesc.add("output-file", 2);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	// check for exceptions
	if (vm.count("help")  || vm.count("input-file")==0 || vm.count("output-file")==0) {
		cerr << "Error: Options input-file, output-file have to be present" << endl;
		cout << "mfsample [options]" << endl;
	    cout << desc << endl;
	    return 1;
	}


	boost::filesystem::path path(outFilename);
	std::string outFile = path.filename().string();
	std::string outDir = path.parent_path().string();
	if (outDir.length()!=0)	outDir+="/";

	//std::cout<<"outFile: "<<outFile<<endl;
	//std::cout<<"outDir: "<<outDir<<endl;


	SparseMatrix v;
	std::cout<<"reading input matrix ... "<<endl;
	readMatrix(inFilename, v);
	std::cout<<"Data matrix: "<< v.size1() << " x " << v.size2()
			<< ", " << v.nnz() << " nonzeros"<<endl;

	// find the sizes for the sample matrix
	if (vm.count("size1") == 0)	{
		if (vm.count("fsize1") != 0) {
			size1=fsize1*v.size1();
		}
	}
	if (vm.count("size2") == 0){
		if (vm.count("fsize2") != 0){
			size2=fsize2*v.size2();
		}
	}

	if ((vm.count("size1") == 0 && vm.count("fsize1") == 0)||
			(vm.count("size2") == 0 && vm.count("fsize2") == 0)){

		// check for exceptions
		if (vm.count("nnz")==0){
			cerr << "Error: You should define ((size1 or fsize1) AND (size2 or fsize2)) OR nnz " << endl;
			return 1;
		}
		if (vm.count("nnz")!=0 && nnz>v.nnz()){
			cerr << "Error: The sample should be smaller than the initial matrix. NNZ<"<< v.nnz()<< endl;
			return 1;
		}
		int ratio=v.nnz()/nnz;
		size1=v.size1()/sqrt(ratio);
		size2=v.size2()/sqrt(ratio);
	}

	// find and print the seed
	if (vm.count("seed") == 0) seed = time(NULL);
	std::cout<<"seed: "<<seed<<endl;
	Random32 random(seed);

	ProjectedSparseMatrix Vsample;
	std::cout<<"Sampling..."<<endl;
	projectRandomSubmatrix(random, v, Vsample, size1, size2);
	std::cout<<"Sample matrix: "<< Vsample.data.size1()
			<< " x " << Vsample.data.size2()<< ", " << Vsample.data.nnz() << " nonzeros"<<endl;

	if (getMatrixFormat(outFile)==MF_PROJECTED_SPARSE_MATRIX){

		string base=outFile.substr(0,outFile.rfind("."));
		string descriptorFile=outFile;

		//extension=(extension==""?"mmc":getExtension(getMatrixFormat(extension)));
		descriptorFile=outDir+descriptorFile;

		IndexMapFileDescriptor fileDescriptor(descriptorFile,  inFilename);

		// write the files in disk
		writeProjectedMatrix(Vsample, fileDescriptor);
		fileDescriptor.save(descriptorFile);

	}
	else if (isSparse(getMatrixFormat(outFile))){

		dataFile=outDir+outFile;
		// write only the data in disk
		std::cout<<"Writing only the data..."<<endl;
		writeMatrix(dataFile, Vsample.data);

	}
	else{
		cerr << "Error: This not a matrix extension."<< endl;
		return 1;
	}

	return 0;
}
