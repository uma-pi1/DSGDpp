/** \file
 *
 * Implementation for channel.h (DO NOT INCLUDE DIRECTLY) */

#include <mpi2/channel.h> // just to help ides
#include <boost/thread/thread.hpp>

namespace mpi2 {

inline boost::mpi::status economicWait(boost::mpi::request req, unsigned pollDelay) {
	boost::optional<boost::mpi::status> msg;
	while ( !(msg = req.test()) ) {
		boost::this_thread::sleep(boost::posix_time::microsec(pollDelay));
	}
	return *msg;
}

inline void economicWaitAll(std::vector<boost::mpi::request> reqs, unsigned pollDelay) {
	while (reqs.size() > 0) {
		for (unsigned i = 0; i<reqs.size(); i++) {
			boost::optional<boost::mpi::status> msg = reqs[i].test();
			if (msg) {
				if (i < reqs.size()-1) {
					reqs[i] = reqs[reqs.size()-1];
				}
				reqs.pop_back();
				i--;
			}
		}
		if (reqs.size() > 0) {
			boost::this_thread::sleep(boost::posix_time::microsec(pollDelay));
		}
	}
}

template<typename ForwardIterator>
inline void economicWaitAll(ForwardIterator begin, ForwardIterator end, unsigned pollDelay) {
	// TODO: this should be done more efficiently
	std::vector<boost::mpi::request> reqs;
	for (ForwardIterator it = begin; it != end; ++it) {
		reqs.push_back(*it);
	}
	economicWaitAll(reqs, pollDelay);
}

inline std::vector<boost::mpi::request> isendAll(const std::vector<Channel>& channels, bool ignoreInactive) {
	const int n = channels.size();
	std::vector<boost::mpi::request> reqs;
	reqs.reserve(n);
	for (int i=0; i<n; i++) {
		if (ignoreInactive && !channels[i].active()) continue;
		reqs.push_back(channels[i].isend());
	}
	return reqs;
}

inline void sendAll(const std::vector<Channel>& channels, bool ignoreInactive) {
	std::vector<boost::mpi::request> reqs = isendAll(channels, ignoreInactive);
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}

template<typename T>
std::vector<boost::mpi::request> isendAll(const std::vector<Channel>& channels, const T& value, bool ignoreInactive) {
	const int n = channels.size();
	std::vector<boost::mpi::request> reqs;
	reqs.reserve(n);
	for (int i=0; i<n; i++) {
		if (ignoreInactive && !channels[i].active()) continue;
		reqs.push_back(channels[i].isend(value));
	}
	return reqs;
}

template<typename T>
void sendAll(const std::vector<Channel>& channels, const T& value, bool ignoreInactive) {
	std::vector<boost::mpi::request> reqs = isendAll(channels, value, ignoreInactive);
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}


inline std::vector<boost::mpi::request> irecvAll(const std::vector<Channel>& channels, bool ignoreInactive) {
	const int n = channels.size();
	std::vector<boost::mpi::request> reqs;
	reqs.reserve(n);
	for (int i=0; i<n; i++) {
		if (ignoreInactive && !channels[i].active()) continue;
		reqs.push_back(channels[i].irecv());
	}
	return reqs;
}

inline void recvAll(const std::vector<Channel>& channels, bool ignoreInactive) {
	std::vector<boost::mpi::request> reqs = irecvAll(channels, ignoreInactive);
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}

inline void economicRecvAll(const std::vector<Channel>& channels, unsigned pollDelay, bool ignoreInactive) {
	std::vector<boost::mpi::request> reqs = irecvAll(channels, ignoreInactive);
	economicWaitAll(reqs, pollDelay);
}

template<typename T>
std::vector<boost::mpi::request> irecvAll(const std::vector<Channel>& channels, std::vector<T>& values, bool ignoreInactive) {
	// TODO: is this OK? writing elements to a vector must be thread-safe
	//       my guess is that resizing is sufficient
	const int n = channels.size();
	values.resize(n);
	std::vector<boost::mpi::request> reqs;
	reqs.reserve(n);
	for (int i=0; i<n; i++) {
		if (ignoreInactive && !channels[i].active()) continue;
		reqs.push_back(channels[i].irecv(values[i]));
	}
	return reqs;
}

template<typename T>
void recvAll(const std::vector<Channel>& channels, std::vector<T>& values, bool ignoreInactive) {
	std::vector<boost::mpi::request> reqs = irecvAll(channels, values, ignoreInactive);
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}

template<typename T>
void economicRecvAll(const std::vector<Channel>& channels, std::vector<T>& values, unsigned pollDelay, bool ignoreInactive) {
	std::vector<boost::mpi::request> reqs = irecvAll(channels, values, ignoreInactive);
	economicWaitAll(reqs, pollDelay);
}

inline void barrier(const std::vector<Channel>& channels) {
	// TODO: can be done more efficiently (just one task endpoint)
	std::vector<boost::mpi::request> reqs = isendAll(channels);
	recvAll(channels);
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}

inline void economicBarrier(const std::vector<Channel>& channels, unsigned pollDelay) {
	// TODO: can be done more efficiently (just one task endpoint)
	std::vector<boost::mpi::request> reqs = isendAll(channels);
	economicRecvAll(channels, pollDelay);
	boost::mpi::wait_all(reqs.begin(), reqs.end()); // quits immediately
}

template<typename T>
std::vector<boost::mpi::request> isendEach(const std::vector<Channel>& channels, const std::vector<T>& values, bool ignoreInactive) {
	const int n = channels.size();
	std::vector<boost::mpi::request> reqs;
	reqs.reserve(n);
	for (int i=0; i<n; i++) {
		if (ignoreInactive && !channels[i].active()) continue;
		reqs.push_back(channels[i].isend(values[i]));
	}
	return reqs;
}

template<typename T>
void sendEach(const std::vector<Channel>& channels, const std::vector<T>& values, bool ignoreInactive) {
	std::vector<boost::mpi::request> reqs = isendEach(channels, values, ignoreInactive);
	boost::mpi::wait_all(reqs.begin(), reqs.end());
}

}
