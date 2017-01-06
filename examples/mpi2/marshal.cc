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
 * Demonstrates mpi2 marshalling capabilities. Marshalling allows to send many values in 1 message.
 */

#include <mpi2/mpi2.h>

using namespace std;
using namespace mpi2;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

// simple task
struct Task1 {
	static inline string id() { return "task1"; }
	static inline void run(Channel ch, TaskInfo info) {
		LOG4CXX_INFO(logger, "Task " << (info.groupId()+1)<< "/"<< info.groupSize() << ": " << ch);
		unsigned value1;
		string value2;
		bool value3;
		ch.recv(*mpi2::unmarshal(value1, value2, value3));
		LOG4CXX_INFO(logger, "Task " << (info.groupId()+1)<< "/"<< info.groupSize()
				<< " received " << "(" << value1 << ", " << value2 << ", " << value3 << ")");
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

	if (world.rank() == 0) {
		Channel ch(UNINITIALIZED);
		std::vector<Channel> channels;

		// some values
		unsigned value1 = 10;
		string value2 = "a string";
		bool value3 = true;

		LOG4CXX_INFO(logger, "Spawning two tasks on each rank");
		tm.spawnAll<Task1>(2, channels);
		LOG4CXX_INFO(logger, "Sending " << "(" << value1 << ", " << value2 << ", "
				<< value3 << ") to all ranks");
		sendAll(channels, mpi2::marshal(value1, value2, value3));
	}

	// shut down
	mpi2stop();
	mpi2finalize();

	return 0;
}
