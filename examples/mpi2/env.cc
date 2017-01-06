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
/** \file
 * Examples of using mpi2's Env environment.
 */
#include <iostream>
#include <string>

#include <util/io.h>

#include <mpi2/mpi2.h>

using namespace std;
using namespace mpi2;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

int main(int argc, char *argv[]) {
	boost::mpi::communicator& world = mpi2init(argc, argv);
	if (world.size() < 2) {
		cerr << "ERROR: You need to run 'env' on at least 2 ranks!" << endl;
		cerr << "Try 'mpirun -np 2 " << argv[0] << "'" << endl;
		return(-1);
	}
	mpi2start();
	boost::this_thread::sleep(boost::posix_time::milliseconds(100)); // just to ensure output in right order

	if (world.rank() == 0) {
		// get the environment of this rank
		Env& env = mpi2::env();

		// create some variables
		LOG4CXX_INFO(logger, "Creating two local variables...");
		RemoteVar var1 = env.create(string("var1"), new int(123));
		RemoteVar var2 = env.create(string("var2"), new int(456));
		LOG4CXX_INFO(logger, var1 << ": " << *env.get<int>("var1"));
		LOG4CXX_INFO(logger, var2 << ": " << *env.get<int>("var2"));

		// update a local variable directly
		LOG4CXX_INFO(logger, "Updating var1...");
		env.setCopy("var1", 789);
		LOG4CXX_INFO(logger, var1 << ": " << *env.get<int>("var1"));

		// creating a remote variable
		LOG4CXX_INFO(logger, "Create var3 on rank 1...");
		RemoteVar var3 = createCopy(1, "var3", 135);

		// reading a remote variable
		int var3copy;
		var3.getCopy(var3copy);
		LOG4CXX_INFO(logger, var3 << ": " << var3copy);

		// update a remote variable
		LOG4CXX_INFO(logger, "Update var3 on rank 1...");
		var3.setCopy(246);
		var3.getCopy(var3copy);
		LOG4CXX_INFO(logger, var3 << ": " << var3copy);

		// local variables can be used just as remote variables
		LOG4CXX_INFO(logger, "Accessing local variable var1 through RemoteVar API...");
		int var1copy;
		var1.getCopy(var1copy);
		LOG4CXX_INFO(logger, var1 << ": " << var1copy);
		LOG4CXX_INFO(logger, var1 << ": " << *var1.getLocal<int>()); // local access does not copy!

		// list the environment on all ranks
		LOG4CXX_INFO(logger, "Printing all environments...");
		lsAll();
	}

	// shut down
	logger->info("");
	mpi2stop();
	mpi2finalize();

	return 0;
}
