/** \file
 *
 * mpi2 communication channels.
 */

#ifndef MPI2_CHANNEL_H
#define MPI2_CHANNEL_H

#include <iostream>

#include <boost/mpi.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/thread.hpp>

#include <mpi2/uninitialized.h>
#include <mpi2/types.h>

namespace mpi2 {

/** An endpoint of an mpi2 communication channel. Consists of a rank and a tag. */
struct Endpoint {
	Endpoint(SerializationConstructor _) : rank(-1), tag(-1) { };
	Endpoint(int r, int t) : rank(r), tag(t) { };

	int rank;
	int tag;

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & rank;
		ar & tag;
	}
};
}

MPI2_SERIALIZATION_CONSTRUCTOR(mpi2::Endpoint);

namespace mpi2 {

inline boost::mpi::status economicWait(boost::mpi::request req, unsigned pollDelay);
inline void economicWaitAll(std::vector<boost::mpi::request> reqs, unsigned pollDelay);
template<typename ForwardIterator>
inline void economicWaitAll(ForwardIterator begin, ForwardIterator end, unsigned pollDelay);

/** Formatted output of an Endpoint (as rank#tag) */
template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
		std::basic_ostream<CharT, Traits>& out, const Endpoint& e)
{
	return out << e.rank << "#" << e.tag;
}

/** An mpi2 communication channel. A channel connects two endpoints; it provides facilities
 * to send or receive messages.
 */
class Channel {
public:
	/** Returns the local endpoint */
	Endpoint local() const { return local_; }

	/** Returns the remote endpoint */
	Endpoint remote() const { return remote_; }

	/** Checks whether the channel is active */
	bool active() const { return active_; };

	void deactivate() {
		active_ = false;
	}

	void activate() {
		active_ = true;
	}

	/** Constructs an uninitialized channel */
	Channel(SerializationConstructor _) : world_(NULL), local_(UNINITIALIZED), remote_(UNINITIALIZED) {
		active_ = false;
	}

	/** Constructs a new communication channel for the specified endpoints */
	Channel(boost::mpi::communicator& world, Endpoint me, Endpoint other)
	: world_(&world), local_(me), remote_(other) {
		active_ = true;
	}

	inline boost::mpi::communicator& world() {
		return *world_;
	}

	/** Receives a message without any data from the other endpoint.
	 *
	 * @see boost::mpi::recv(int, int)
	 */
	inline boost::mpi::status recv() const {
		return world_->recv(remote_.rank, local_.tag);
	}


	inline boost::mpi::status economicRecv(unsigned pollDelay) const {
		boost::mpi::request req = world_->irecv(remote_.rank, local_.tag);
		return economicWait(req, pollDelay);
	}

	/** Receives a value from the other endpoint.
	 *
	 * @see boost::mpi::recv(int, int, T&)
	 */
	template<typename T>
	inline boost::mpi::status recv(T& value) const {
		return world_->recv(remote_.rank, local_.tag, value);
	}

	template<typename T>
	inline boost::mpi::status economicRecv(T& value, unsigned pollDelay) const {
		boost::mpi::request req = world_->irecv(remote_.rank, local_.tag, value);
		return economicWait(req, pollDelay);
	}


	/** Asynchronously receives a message without any data from the other endpoint.
	 *
	 * @returns an boost::mpi::request to check for completion
	 * @see boost::mpi::irecv(int, int)
	 */
	inline boost::mpi::request irecv() const {
		return world_->irecv(remote_.rank, local_.tag);
	}

	/** Asynchronously receives a value from the other endpoint.
	 *
	 * @returns an boost::mpi::request to check for completion
	 * @see boost::mpi::irecv(int, int, T&)
	 */
	template<typename T>
	inline boost::mpi::request irecv(T& value) const {
		return world_->irecv(remote_.rank, local_.tag, value);
	}

	/** Asynchronously receives an array of values to the other endpoint.
	 *
	 * @returns an boost::mpi::request to check for completion
	 * @see boost::mpi::isend(int, int, const T&)
	 */
	template<typename T>
	inline boost::mpi::request irecv(T* value, int n) const {
		return world_->irecv(remote_.rank, local_.tag, value, n);
	}

	/** Sends a message without any data to the other endpoint.
	 *
	 * @see boost::mpi::send(int, int)
	 */
	inline void send() const {
		world_->send(remote_.rank, remote_.tag);
	}

	inline void economicSend(unsigned pollDelay) const {
		boost::mpi::request req = world_->isend(remote_.rank, remote_.tag);
		economicWait(req, pollDelay);
	}

	/** Sends a value to the other endpoint.
	 *
	 * @see boost::mpi::send(int, int, const T&)
	 */
	template<typename T>
	inline void send(const T& value) const {
		world_->send(remote_.rank, remote_.tag, value);
	}

	template<typename T>
	inline void economicSend(const T& value, unsigned pollDelay) const {
		boost::mpi::request req = world_->isend(remote_.rank, remote_.tag, value);
		economicWait(req, pollDelay);
	}

	/** Asynchronously sends a message without any data to the other endpoint.
	 *
	 * @returns an boost::mpi::request to check for completion
	 * @see boost::mpi::isend(int, int)
	 */
	inline boost::mpi::request isend() const {
		return world_->isend(remote_.rank, remote_.tag);
	}

	/** Asynchronously sends a value to the other endpoint.
	 *
	 * @returns an boost::mpi::request to check for completion
	 * @see boost::mpi::isend(int, int, const T&)
	 */
	template<typename T>
	inline boost::mpi::request isend(const T& value) const {
		return world_->isend(remote_.rank, remote_.tag, value);
	}

	/** Asynchronously sends an array of values to the other endpoint.
	 *
	 * @returns an boost::mpi::request to check for completion
	 * @see boost::mpi::isend(int, int, const T&)
	 */
	template<typename T>
	inline boost::mpi::request isend(const T* value, int n) const {
		return world_->isend(remote_.rank, remote_.tag, value, n);
	}

private:
	/** The MPI communicator used by this channel */
	boost::mpi::communicator* world_;

	/** The endpoint of the local task */
	Endpoint local_;

	/** The endpoint of the remote task */
	Endpoint remote_;

	/** Whether the channel is active (inactive channels are ignored) */
	bool active_;
};

/** Formatted output of a Channel (as rank#tag->rank#tag) */
template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
		std::basic_ostream<CharT, Traits>& out, const Channel& ch)
{
	if (ch.active()) {
		return out << ch.local() << "->" << ch.remote();
	} else {
		return out << "INACTIVE";
	}
}


/** Send a message without value to all channels. Method uses asynchronous communication. */
std::vector<boost::mpi::request> isendAll(const std::vector<Channel>& channels, bool ignoreInactive=true);

/** Send a message without value to all channels. Method uses asynchronous communication. */
void sendAll(const std::vector<Channel>& channels, bool ignoreInactive=true);

/** Send a value to all channels. Method uses asynchronous communication. */
template<typename T>
std::vector<boost::mpi::request> isendAll(const std::vector<Channel>& channels, const T& value, bool ignoreInactive=true);

/** Send a value to all channels. Method uses asynchronous communication. */
template<typename T>
void sendAll(const std::vector<Channel>& channels, const T& value, bool ignoreInactive=true);

/** Receive a message without value from all channels. Method uses asynchronous communication. */
std::vector<boost::mpi::request> irecvAll(const std::vector<Channel>& channels, bool ignoreInactive=true);

/** Receive a message without value from all channels. Method uses asynchronous communication. */
void recvAll(const std::vector<Channel>& channels, bool ignoreInactive=true);

void economicRecvAll(const std::vector<Channel>& channels, unsigned pollDelay, bool ignoreInactive=true);

/** Receive a value from all channels. Method uses asynchronous communication. */
template<typename T>
std::vector<boost::mpi::request> irecvAll(const std::vector<Channel>& channels, std::vector<T>& values, bool ignoreInactive=true);

/** Receive a value from all channels. Method uses asynchronous communication. */
template<typename T>
void recvAll(const std::vector<Channel>& channels, std::vector<T>& values, bool ignoreInactive=true);

template<typename T>
void economicRecvAll(const std::vector<Channel>& channels, std::vector<T>& values, unsigned pollDelay, bool ignoreInactive=true);


/** Barrier between the given channels. This method has to be called at all tasks using the
 * appropriate channels.
 */
void barrier(const std::vector<Channel>& channels);

void economicBarrier(const std::vector<Channel>& channels, unsigned pollDelay);

/** Sends the i-th value to the i-th channel. Method uses asynchronous communication. */
template<typename T>
std::vector<boost::mpi::request> isendEach(const std::vector<Channel>& channels, const std::vector<T>& values, bool ignoreInactive=true);

/** Sends the i-th value to the i-th channel. Method uses asynchronous communication. */
template<typename T>
void sendEach(const std::vector<Channel>& channels, const std::vector<T>& values, bool ignoreInactive=true);

} // namespace mpi2

#include <mpi2/channel_impl.h>

#endif
