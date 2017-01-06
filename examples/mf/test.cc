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
using namespace mpi2;




int main(int argc, char* argv[]) {
	boost::mpi::communicator& world = mfInit(argc, argv);
	mfStart();

	// load and block a matrix
	string fV("/home/rgemulla/data/matrix/tiny.mmc");
	DistributedSparseMatrix mV = loadMatrix<SparseMatrix>(
			"V", 1, 2, true, fV);

	// create a file descriptor
	std::string path = "/home/rgemulla/0/";
	std::string baseFilename = "V";
	BlockedMatrixFileDescriptor f = BlockedMatrixFileDescriptor::create(mV, path, baseFilename);
	f.save(path + baseFilename + ".xml");
	BlockedMatrixFileDescriptor f2;
	f2.load(path + baseFilename + ".xml");
	f2.save(path + baseFilename + "2.xml");
	mfStop();
	mfFinalize();

	return 0;
}
