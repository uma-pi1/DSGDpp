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
 * mfproject.cc
 *
 *  Created on: Jul 15, 2011
 *      Author: fmakari
 */

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <util/exception.h>
#include <util/io.h>

#include <mf/mf.h>

using namespace std;
using namespace mf;

using namespace boost;
using namespace boost::program_options;
using namespace rg;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

int main(int argc, char *argv[]) {
	string inFilename;
	string outFilename;
	mf_size_type threshold;
	bool repeat = false;

	// parse command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
		("input-file", value<string>(&inFilename), "input file (.mmc or .mfp)")
		("output-file", value<string>(&outFilename), "output file (.mmc or .mfp)")
		("threshold", value<mf_size_type>(&threshold), "keep only rows and columns of the input matrix whose number of nonzero entries is above this threshold [0])")
		("repeat", value<bool>(&repeat), "if present, repeatedly project out rows and columns that do not pass the threshold until all of them do")
	;

	positional_options_description pdesc;
	pdesc.add("input-file", 1);
	pdesc.add("output-file", 2);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	// check for exceptions
	if (vm.count("help") || vm.count("input-file") == 0 || vm.count("output-file") == 0) {
		cerr << "Error: Option input-file and output-file has to be present" << endl;
		cout << "mfproject [options]" << endl;
	    cout << desc << endl;
	    exit(1);
	}

	if (vm.count("threshold") == 0) threshold = 0;

	// check format - allowed formats: MF_PROJECTED_SPARSE_MATRIX and MM_COORD
	MatrixFileFormat inFormat = getMatrixFormat(inFilename);
	MatrixFileFormat outFormat = getMatrixFormat(outFilename);
	if ((inFormat != MF_PROJECTED_SPARSE_MATRIX) && (inFormat != MM_COORD)) {
		cerr << "Error: Wrong input-file format" << endl;
		exit(1);
	}
	if ((outFormat != MF_PROJECTED_SPARSE_MATRIX) && (outFormat != MM_COORD)) {
		cerr << "Error: Wrong output-file format" << endl;
		exit(1);
	}

	LOG4CXX_INFO(logger, "Input file: " << inFilename);
	LOG4CXX_INFO(logger, "Output file: " << outFilename);
	LOG4CXX_INFO(logger, "Treshold: " << threshold);
	LOG4CXX_INFO(logger, "Repeat: " << repeat);

	// read input
	LOG4CXX_INFO(logger, "Reading input matrix");
	ProjectedSparseMatrix matrix;
	std::string matrixFilename = inFilename;
	if (inFormat == MF_PROJECTED_SPARSE_MATRIX) {
		IndexMapFileDescriptor desc;
		desc.load(inFilename);
		matrixFilename = desc.matrixFilename;
		readProjectedMatrix(matrix, inFilename);
		LOG4CXX_INFO(logger, "Input matrix projected from a "
			<< matrix.size1 << " x " << matrix.size2 << " matrix");
	} else {
		readMatrix(inFilename, matrix.data);
		matrix.size1 = matrix.data.size1();
		matrix.size2 = matrix.data.size2();
		matrix.map1.resize(matrix.size1, false);
		matrix.map2.resize(matrix.size2, false);
		for (mf_size_type i=0; i<matrix.size1; i++) {
			matrix.map1[i] = i;
		}
		for (mf_size_type j=0; j<matrix.size2; j++) {
			matrix.map2[j] = j;
		}
	}
	LOG4CXX_INFO(logger, "Input matrix: "
		<< matrix.data.size1() << " x " << matrix.data.size2() << ", " << nnz(matrix.data) << " nonzeros");

	// do the projection
	mf_size_type nnzNew = nnz(matrix.data);
	mf_size_type nnzOld = -1;
	do {
		LOG4CXX_INFO(logger, "Removing infrequent rows/columns");
		projectFrequent(matrix, threshold);
		nnzOld = nnzNew;
		nnzNew = nnz(matrix.data);
		LOG4CXX_INFO(logger, "Projected matrix: "
				<< matrix.data.size1() << " x " << matrix.data.size2() << ", " << nnz(matrix.data) << " nonzeros");
	} while (repeat && nnzNew != nnzOld);

	// write output
	LOG4CXX_INFO(logger, "Writing output matrix");
	if (outFormat == MM_COORD) {
		writeMatrix(outFilename, matrix.data);
	} else {
		IndexMapFileDescriptor desc(outFilename, matrixFilename);
		desc.save(outFilename);
		writeProjectedMatrix(matrix, desc);
	}

	// everything OK
	return 0;
}
