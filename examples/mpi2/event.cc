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
 * Examples of using mpi2's event logging facilities.
 *
 * For events to be logged, please set then log level of the "evnt" logger to "INFO" in log4j.properties.
 */
#include <iostream>
#include <string>

#include <util/random.h>

#include <mpi2/mpi2.h>

using namespace std;
using namespace mpi2;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

// simple task
struct Task1 {
	static inline string id() { return "task1"; }
	static inline void run(Channel ch, TaskInfo info) {
		rg::Random32 random = mpi2::getSeed(ch);
		logBeginEvent("waittask");


		logBeginEvent("wait1");
		boost::this_thread::sleep(boost::posix_time::milliseconds( random.nextInt(1000) ));
		logEndEvent("wait1");

		logBeginEvent("wait2");
		boost::this_thread::sleep(boost::posix_time::milliseconds( random.nextInt(1000) ));
		logEndEvent("wait2");

		logBeginEvent("barrier");
		mpi2::barrier(info.pairwiseChannels());
		logEndEvent("barrier");

		logBeginEvent("wait3");
		boost::this_thread::sleep(boost::posix_time::milliseconds( random.nextInt(1000) ));
		logEndEvent("wait3");

		logEndEvent("waittask");

		ch.send();
	}
};

int main(int argc, char *argv[]) {
	boost::mpi::communicator& world = mpi2init(argc, argv);
	TaskManager& tm = TaskManager::getInstance();
	registerTask<Task1>();
	mpi2start();

	if (world.rank() == 0) {
		rg::Random32 random;
		logBeginEvent("main");

		std::vector<Channel> channels;
		tm.spawnAll<Task1>(3, channels, true);
		mpi2::seed(channels, random);
		recvAll(channels);

		tm.spawnAll<Task1>(3, channels, true);
		mpi2::seed(channels, random);
		recvAll(channels);

		logEndEvent("main");
	}

	// shut down
	logger->info("");
	mpi2stop();
	mpi2finalize();

	return 0;
}
