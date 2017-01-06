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

using namespace std;
using namespace mf;
using namespace mpi2;
using namespace boost::program_options;

int main(int argc, char *argv[]) {
	mf_size_type blocks1;
	mf_size_type blocks2;
	string extension;
	string inFilename;
	string outBaseFilename;
	int tasksPerRank;

	// read command line
	options_description desc("Options");
	desc.add_options()
		("help", "produce help message")
	    ("blocks1", value<mf_size_type>(&blocks1)->default_value(1), "number of row blocks")
	    ("blocks2", value<mf_size_type>(&blocks2)->default_value(1), "number of column blocks")
	    ("threads", value<int>(&tasksPerRank)->default_value(1), "number of threads per node")
	    ("format", value<string>(&extension)->default_value(""), "file format of blocks (default: one of the matrix market formats)")
	    ("input-file", value<string>(&inFilename), "input file")
		("output-base-file", value<string>(&outBaseFilename), "output file (no ending)");
	;

	positional_options_description pdesc;
	pdesc.add("input-file", 1);
	pdesc.add("output-base-file", 2);

	variables_map vm;
	store(command_line_parser(argc, argv).options(desc).positional(pdesc).run(), vm);
	notify(vm);

	if (vm.count("help") || vm.count("input-file")==0 || vm.count("output-base-file")==0) {
		cout << "mfdblock [options] <input-file> <output-base-file>" << endl << endl;
	    cout << desc << endl;
	    return 1;
	}

	// fire up
	boost::mpi::communicator& world = mfInit(argc, argv);
	mfStart();

	if (world.rank() == 0) {

		// print options
		boost::filesystem::path path(outBaseFilename);
		string outFile = path.filename().string();
		string outDir = path.parent_path().string();
		if (!outDir.empty()) outDir += "/";
		cout << "Input file        : " << inFilename << endl;
		cout << "Output directory  : " << outDir << endl;
		cout << "Output file       : " << outFile << ".xml (+ blocks)" << endl;
		cout << "Output format     : " << (extension=="" ? "AUTOMATIC" : extension) << endl;
		cout << "Blocks            : " << blocks1 << "x" << blocks2 << endl;
		cout << "Threads@nodes     : " << tasksPerRank << "@" << world.size() << endl;
		cout << endl;

		// go
		MatrixFileFormat format = getMatrixFormat(inFilename);
		if (isSparse(format)) {
			cout << "Reading " << inFilename << "... " << endl;
			DistributedSparseMatrix mV = loadMatrix<SparseMatrix>("V", blocks1, blocks2, true, inFilename);
			cout << endl;

			cout << "Writing blocks ... " << endl;
			BlockedMatrixFileDescriptor f;
			if (extension == "") {
				f = BlockedMatrixFileDescriptor::create(mV, outDir, outFile);
			} else {
				f = BlockedMatrixFileDescriptor::create(mV, outDir, outFile, getMatrixFormat(extension));
			}
			storeMatrix(mV, f, tasksPerRank);
			f.path = "";
			f.save(outDir + outFile + ".xml");
		} else {
			cout << "Reading " << inFilename << "... " << endl;
			DistributedDenseMatrix mV = loadMatrix<DenseMatrix>("V", blocks1, blocks2, true, inFilename);
			cout << endl;

			cout << "Writing blocks ... " << endl;
			BlockedMatrixFileDescriptor f;
			if (extension == "") {
				f = BlockedMatrixFileDescriptor::create(mV, outDir, outFile);
			} else {
				f = BlockedMatrixFileDescriptor::create(mV, outDir, outFile, getMatrixFormat(extension));
			}
			storeMatrix(mV, f, tasksPerRank);
			f.path = "";
			f.save(outDir + outFile + ".xml");
		}
	}
	mfStop();
	mfFinalize();

	return 0;
}
