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
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <mpi2/mpi2.h>
#include <mf/mf.h>

/*
 * 	Take as input a train matrix and 2 initial factors and returns two new factors scaled s.t.
 * 	E[ sum(WH) ] = E[ sum(V) ]
 * */



using namespace std;
using namespace mf;
using namespace mpi2;
using namespace boost::program_options;

int main(int argc, char *argv[]) {

	string inFilenameTrain;
	string wNewFile,hNewFile;
	string wFile,hFile;

	// read command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
	    ("inTrain", value<string>(&inFilenameTrain), "input file with training data")
	    ("W", value<string>(&wFile), "input file with W")
	    ("H", value<string>(&hFile), "input file with H")
		("Wnew", value<string>(&wNewFile), "new W factor")
		("Hnew", value<string>(&hNewFile), "new h factor")
	;

	positional_options_description pdesc;
	pdesc.add("inTrain", 1);
	pdesc.add("W", 2);
	pdesc.add("H", 3);
	pdesc.add("Wnew", 4);
	pdesc.add("Hnew", 4);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	if (vm.count("help") || vm.count("inTrain")==0|| vm.count("W")==0|| vm.count("H")==0) {
		cout << "scaleFactors [options] <inTrain> " << endl << endl;
	    cout << desc << endl;
	    return 1;
	}

	// read Train matrix
	std::cout<<"reading matrix... "<<std::endl;
	SparseMatrix v;
	readMatrix(inFilenameTrain,v);

	// calculate average according to Train matrix
	std::cout<<"calculating sum... "<<std::endl;
	double scaleFactor = sqrt(sum(v));

	std::cout<<"scaleFactor value: "<<scaleFactor<<std::endl;

	std::cout<<"reading factors... "<<std::endl;
	DenseMatrix w;
	readMatrix(wFile,w);
	DenseMatrixCM h;
	readMatrix(hFile,h);
	std::cout<<"updating factors... "<<std::endl;

	div2(w, sums2(w));
	div1(h, sums1(h));
	mult(w, scaleFactor);
	mult(h, scaleFactor);


	std::cout<<"writing factors back... "<<std::endl;
	writeMatrix(wNewFile,w);
	writeMatrix(hNewFile,h);

	return 0;
}




