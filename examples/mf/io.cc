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
#include <iostream>
#include <fstream>
#include <string>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <util/evaluation.h>

#include <mf/mf.h>

using namespace std;
using namespace boost::numeric::ublas;
using namespace rg;
using namespace mf;

int main(int argc, char *argv[]) {
	coordinate_matrix<double> m(0,0,0);
	string base("/home/rgemulla/data/netflix/probe.perm");
	string fileMmCoord = base + ".mmc";

	// read in matrix market format
	Timer t;
	cout << "Reading " << fileMmCoord << "... " << flush;
	t.start();
	readMatrix(fileMmCoord, m);
	t.stop();
	cout << "done (" << t << "), m=" << m.size1() << ", n=" << m.size2() << ", nz=" << m.nnz();
	cout << endl;
	//cout << M << endl;

	// write out to (portable) Boost text format
	string fileBoostText = base + ".bst";
	cout << "Writing " << fileBoostText << "... " << flush;
	t.start();
	writeMatrix(fileBoostText, m);
	t.stop();
	cout << "done (" << t << ")" << endl;

	// read (portable) Boost text format
	m = coordinate_matrix<double>(0,0,0);
	cout << "Reading " << fileBoostText << "... " << flush;
	t.start();
	readMatrix(fileBoostText, m);
	t.stop();
	cout << "done (" << t << "), m=" << m.size1() << ", n=" << m.size2() << ", nz=" << m.nnz();
	cout << endl;

	// write out to (non-portable) Boost binary format
	string fileBoostBin = base + ".bsb";
	cout << "Writing " << fileBoostBin << "... " << flush;
	t.start();
	writeMatrix(fileBoostBin, m);
	t.stop();
	cout << "done (" << t << ")" << endl;

	// read (non-portable) Boost binary format
	m = coordinate_matrix<double>(0,0,0);
	cout << "Reading " << fileBoostBin << "... " << flush;
	t.start();
	readMatrix(fileBoostBin, m);
	t.stop();
	cout << "done (" << t << "), m=" << m.size1() << ", n=" << m.size2() << ", nz=" << m.nnz();
	cout << endl;

	return 0;
}
