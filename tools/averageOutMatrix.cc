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
 * averageOutMatrix.cc
 *
 *  Created on: Oct 4, 2011
 *      Author: chteflio
 */

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <mpi2/mpi2.h>
#include <mf/mf.h>

using namespace std;
using namespace mf;
using namespace mpi2;
using namespace boost::program_options;

int main(int argc, char *argv[]) {

	string inFilenameTrain;
	string outFilenameTrain;
	string inFilenameTest;
	string outFilenameTest;

	// read command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
	    ("inTrain", value<string>(&inFilenameTrain), "input file with training data")
	    ("inTest", value<string>(&inFilenameTest), "input file with test data")
		("outTrain", value<string>(&outFilenameTrain), "output file with training data")
		("outTest", value<string>(&outFilenameTest), "output file with test data");
	;

	positional_options_description pdesc;
	pdesc.add("inTrain", 1);
	pdesc.add("inTest", 2);
	pdesc.add("outTrain", 3);
	pdesc.add("outTest", 4);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	if (vm.count("help") || vm.count("inTrain")==0 || vm.count("outTrain")==0) {
		cout << "mfdblock [options] <inTrain> <outTrain>" << endl << endl;
	    cout << desc << endl;
	    return 1;
	}

	// read Train matrix
	std::cout<<"reading matrix: "<<std::endl;
	SparseMatrix v;
	readMatrix(inFilenameTrain,v);

	// calculate average according to Train matrix
	std::cout<<"calculating average: "<<std::endl;
	double sum=0;
	const SparseMatrix::value_array_type& values = v.value_data();
	for (mf_size_type p=0; p<v.nnz(); p++) {
		sum+=values[p];
	}

	double average=sum/v.nnz();

	std::cout<<"average value: "<<average<<std::endl;

//	// average out train matrix
//	for (mf_size_type i=0;i<v.size1();i++){
//		for (mf_size_type j=0;j<v.size2();j++){
//			v(i,j)-=average;
//		}
//	}
	std::cout<<"update matrix: "<<std::endl;
	for (mf_size_type p=0; p<v.nnz(); p++) {
		v.value_data()[p]-=average;
	}
	//v-=average;
	std::cout<<"writing train..."<<std::endl;
	writeMatrix(outFilenameTrain,v);

	// and now do average out the test matrix
	if (vm.count("inTest")!=0 && vm.count("outTest")!=0 ){
		std::cout<<"reading matrix: "<<std::endl;
		SparseMatrix vTest;
		readMatrix(inFilenameTest,vTest);
//		const SparseMatrix::index_array_type& indexTest1 = vTest.index1_data();
//		const SparseMatrix::index_array_type& indexTest2 = vTest.index2_data();


		std::cout<<"update matrix: "<<std::endl;
		for (mf_size_type p=0; p<vTest.nnz(); p++) {
			vTest.value_data()[p]-=average;
		}
		//vTest-=average;
		std::cout<<"writing test..."<<std::endl;
		writeMatrix(outFilenameTest,vTest);
	}

	return 0;
}


