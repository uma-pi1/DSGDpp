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

#include <mpi2/mpi2.h>


log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

int main(int argc, char* argv[]) {
	using namespace std;
	using namespace mpi2;

	boost::mpi::communicator& world = mpi2init(argc, argv);

	// fire up task managers
	// this blocks on all but the root node
	mpi2start();

	// play around with the tasks
	// the root node initiates communication
	if (world.rank() == 0) {
		// just wait for input
		string x;
		cin >> x;
	}

	// shut down
	LOG4CXX_INFO(logger, "");
	mpi2stop();
	mpi2finalize();

	return 0;
}
