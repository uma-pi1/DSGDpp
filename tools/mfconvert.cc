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

#include <mf/mf.h>

using namespace std;
using namespace mf;

int main(int argc, char *argv[]) {
	if (argc != 3) {
		cout << "Usage: mfconvert <in-file> <out-file>" << endl;
		cout << "Supported extensions: .mma .mmc .bsb .bst .bdb .bdt" << endl;
		return 1;
	}

	string fIn = argv[1];
	string fOut = argv[2];
	MatrixFileFormat format = getMatrixFormat(fIn);
	DenseMatrix d;
	SparseMatrix s;
	switch (format) {
	case MM_ARRAY:
	case BOOST_DENSE_TEXT:
	case BOOST_DENSE_BIN:
		cout << "Reading " << fIn << "... ";
		readMatrix(fIn, d, format);
		cout << "done." << endl << "Writing " << fOut << "... ";
		writeMatrix(fOut, d);
		cout << "done." << endl;
		break;
	case MM_COORD:
	case BOOST_SPARSE_TEXT:
	case BOOST_SPARSE_BIN:
		cout << "Reading " << fIn << "... ";
		readMatrix(fIn, s, format);
		cout << "done." << endl << "Writing " << fOut << "... ";
		writeMatrix(fOut, s);
		cout << "done." << endl;
		break;
	default:
		cout << "Reading from format " << format << " not supported."<< endl;
		return 1;
	}

	return 0;
}
