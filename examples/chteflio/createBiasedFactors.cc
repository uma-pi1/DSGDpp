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
 * createBiasedFactors_v2.cc
 *
 *  Created on: Nov 10, 2011
 *      Author: chteflio
 */



#include <string>

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <util/io.h>
#include <mf/mf.h>



using namespace std;
using namespace mf;
using namespace mpi2;
using namespace boost;
using namespace boost::program_options;
using namespace rg;


// main program
int main(int argc, char *argv[]) {
	string inputFile,wInFile,hInFile,wOutFile,hOutFile;


	// parse command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
		("input-file", value<string>(&inputFile), "the data matrix")
		("input-row-file", value<string>(&wInFile), "the factor W")
		("input-col-file", value<string>(&hInFile), "the factor H")
		("output-row-file", value<string>(&wOutFile), "the new factor W")
		("output-col-file", value<string>(&hOutFile), "the new factor H")
	;
	positional_options_description pdesc;
	pdesc.add("input-file", 1);
	pdesc.add("input-row-file", 2);
	pdesc.add("input-col-file", 3);
	pdesc.add("output-row-file", 4);
	pdesc.add("output-col-file", 5);


	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	// check required arguments
	if (vm.count("help") || vm.count("input-file")==0 || vm.count("input-row-file")==0 || vm.count("input-col-file")==0) {
		cerr << "Error: Options input-file,input-row-file,input-col-file are required" << endl;
		cerr << "createBiasedFactors [options]" << endl;
	    cerr << desc << endl;
	    exit(1);
	}

	typedef SparseMatrix M;

	cout<<"Reading Matrices..."<<endl;
	SparseMatrix v;
	DenseMatrix w;
	DenseMatrixCM h;
	readMatrix(inputFile,v);
	readMatrix(wInFile,w);
	readMatrix(hInFile,h);

	SparseMatrixCM vc;
	copyCm(v,vc);

	std::vector<mf_size_type> nnz1,nnz2;
	mf_size_type nnz12max;
	cout<<"calculating non zero entries..."<<endl;
	nnz12(v,nnz1,nnz2,nnz12max);


	cout<<"calculating Sums..."<<endl;
	boost::numeric::ublas::vector<M::value_type> sum1(nnz1.size());
	boost::numeric::ublas::vector<M::value_type> sum2(nnz2.size());

	M::index_array_type& index1 = v.index1_data();
	M::index_array_type& index2 = v.index2_data();
	M::value_array_type& values = v.value_data();

	double sum=0;
	mf_size_type row=0;
	for (mf_size_type i=0; i<v.nnz(); i++) {
		M::size_type i1 = index1[i];
		M::size_type i2 = index2[i];
		M::value_type value = values[i];

		if (row==i1){//add
			sum+=value;
		}
		else{// store and move to next row
			sum1(row)=sum;
			sum=value;
			row++;
		}
	}
	sum1(row)=sum;// store last element

	M::index_array_type& indexC1 = vc.index1_data();
	M::index_array_type& indexC2 = vc.index2_data();
	M::value_array_type& valuesC = vc.value_data();
	sum=0;
	mf_size_type col=0;
	for (mf_size_type i=0; i<vc.nnz(); i++) {
		M::size_type i1 = indexC1[i];
		M::size_type i2 = indexC2[i];
		M::value_type value = valuesC[i];

		if (col==i1){//add
			sum+=value;
		}
		else{// store and move to next row
			sum2(col)=sum;
			sum=value;
			col++;
		}
	}
	sum2(col)=sum;// store last element

	cout<<"updating factors..."<<endl;
	for (int i=0;i<nnz1.size();i++){
		w(i,0)=sum1(i)/nnz1[i];
	}
	for (int i=0;i<nnz2.size();i++){
		h(0,i)=sum2(i)/nnz2[i];
	}

	cout<<"writing factors to disk..."<<endl;
	writeMatrix(wOutFile,w);
	writeMatrix(hOutFile,h);

	// everything OK
	return 0;
}
