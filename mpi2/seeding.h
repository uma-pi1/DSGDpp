/** \file
 *
 * Methods to seed PRNGs across a task group.
 */

#ifndef MPI2_SEEDING_H
#define MPI2_SEEDING_H

#include <util/random.h>
#include <mpi2/channel.h>

namespace mpi2 {

enum SeedingMethod { NAIVE };

/** Seed each of the channels with a random number seed. The channels
 * have to call mpi2::getSeed(Channel) to get an instance of the PRNG.
 */
inline void seed(std::vector<Channel>& channels, rg::Random32& random,
		SeedingMethod method = NAIVE, bool ignoreInactive=true) {
	std::vector<unsigned> seeds(channels.size());

	switch (method) {
	case NAIVE:
		for (unsigned i=0; i<channels.size(); i++) {
			seeds[i] = random();
		}
		break;
	}
	mpi2::sendEach(channels, seeds, ignoreInactive);
}

inline void seed(Channel channel, rg::Random32& random,
		SeedingMethod method = NAIVE, bool ignoreInactive=true) {
	if (!channel.active()) return;
	unsigned seed;
	switch (method) {
	case NAIVE:
		seed = random();
		break;
	}
	channel.send(seed);
}


/** Receives a seed sent by mpi2::seed() and initializes a PRNG. */
inline rg::Random32 getSeed(Channel ch) {
	unsigned seed;
	ch.recv(seed);
	return rg::Random32(seed);
}

}

#endif
