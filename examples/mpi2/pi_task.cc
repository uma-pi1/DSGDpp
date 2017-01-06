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
 * A simple example that demonstrates the usage of mpi2. Spawns a couple of tasks that each run
 * Monte Carlo integration to estimate the value of pi; then averages and outputs the result.
 */
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <set>

#include <boost/foreach.hpp>

#include <util/io.h>
#include <util/random.h>

#include <mpi2/mpi2.h>

using namespace std;
using namespace mpi2;
using namespace rg;

log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("main"));

/** Task that computes pi via Monte Carlo integration */
struct PiTask {
	static inline string id() { return "PiTask"; }
	static inline void run(Channel ch, TaskInfo info) {
		// receive the seed for the random number generator
		Random32 random = getSeed(ch);

		// receive the number of iterations to perform
		unsigned n;
		ch.recv(n);

		// run Monte Carlo integration
		LOG4CXX_INFO(logger, "Task " << info.groupId() << ": Generating " << n
				<< " MC samples");
		int count = 0;
		for (unsigned i=0; i<n; i++) {
			double x = random.nextDouble(); // actually need [-1,1]; but [0,1] is OK since we square
			double y = random.nextDouble();
			if (x*x+ y*y < 1) count++;
		}
		double est = (double)count/n * 4.; // area unit circle = pi; area square = 4

		// send back the results
		LOG4CXX_INFO(logger, "Task " << info.groupId() << ": Sending " << est);
		ch.send(est);
	}
};

int main(int argc, char* argv[]) {
	// initialize mpi2
	boost::mpi::communicator& world = mpi2init(argc, argv);

	// register PiTask (this is required to be able to actually run it!)
	TaskManager& tm = TaskManager::getInstance();
	registerTask<PiTask>();

	// fire up task managers (this blocks on all but the root node)
	mpi2start();

	// main driver; only executed at root rank
	if (world.rank() == 0) {
		// spawn 3 copies of PiTask at each rank
		std::vector<Channel> channels;
		int threads = 3;
		tm.spawnAll<PiTask>(threads, channels);
		
		std::cout<<"channels.size: "<<channels.size()<<std::endl;

		// send them each a different random number seed
		Random32 random; // note: this takes a default seed (not randomized!)
		seed(channels, random);

		// send them the number of samples to take
		sendAll(channels, 100000); // each thread computes 100000 samples

		// receive the results
		std::vector<double> results(channels.size());
		recvAll(channels, results);

		// aggregate and output the result
		double pi = std::accumulate(results.begin(), results.end(), 0.) / channels.size();
		LOG4CXX_INFO(logger, "Estimate of pi: " << pi);
	}

	// shut down task managers
	mpi2stop();

	// shut down mpi2 and mpi
	mpi2finalize();

	return 0;
}
