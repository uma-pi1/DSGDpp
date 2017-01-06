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
 *
 * Shows how to spawn a set of tasks.
 */

#include <iostream>
#include <map>
#include <queue>
#include <set>

#include <boost/mpi/communicator.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include <util/io.h>

#include <mpi2/mpi2.h>

using namespace std;
using namespace mpi2;
using namespace rg;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

// simple task
struct Task1 {
	static inline string id() { return "task1"; }
	static inline void run(Channel ch, TaskInfo info) {
		LOG4CXX_INFO(logger, "Task " << (info.groupId()+1)<< "/"<< info.groupSize() << ": " << ch);
		ch.send(); // ack
	}
};

int main(int argc, char* argv[]) {
	boost::mpi::communicator& world = mpi2init(argc, argv);

	// register a simple task
	TaskManager& tm = TaskManager::getInstance();
	registerTask<Task1>();

	// fire up task managers
	// this blocks on all but the root node
	mpi2start();
	boost::this_thread::sleep(boost::posix_time::milliseconds(100)); // just to ensure output in right order

	// play around with the tasks
	// the root node initiates communication
	if (world.rank() == 0) {
		Channel ch(UNINITIALIZED);
		std::vector<Channel> channels;

		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Spawning one task on rank 0");
		ch = tm.spawn<Task1>(0);
		ch.recv();
		LOG4CXX_INFO(logger, "Source channel: " << ch);

		if (world.size() > 1) {
			LOG4CXX_INFO(logger, "");
			LOG4CXX_INFO(logger, "Spawning three tasks on rank 1");
			tm.spawn<Task1>(1, 3, channels);
			recvAll(channels);
			LOG4CXX_INFO(logger, rg::paste("Source channels: ", channels));
		}

		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Spawning one task on each rank");
		tm.spawnAll<Task1>(channels);
		recvAll(channels);
		LOG4CXX_INFO(logger, rg::paste("Source channels: ", channels));

		LOG4CXX_INFO(logger, "");
		LOG4CXX_INFO(logger, "Spawning three tasks on each rank");
		tm.spawnAll<Task1>(3, channels);
		recvAll(channels);
		LOG4CXX_INFO(logger, rg::paste("Source channels: ", channels));
	}

	// shut down
	LOG4CXX_INFO(logger, "");
	mpi2stop();
	mpi2finalize();

	return 0;
}
