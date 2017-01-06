/** \file
 *
 * Methods involving task management, e.g., Task and TaskManager.
 */

#ifndef MPI2_TASK_MANAGER_H
#define MPI2_TASK_MANAGER_H

#include <map>
#include <queue>
#include <string>

#include <boost/function.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/mpi.hpp>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition.hpp>
#include <boost/threadpool.hpp>

#include <log4cxx/logger.h>

#include <util/exception.h>

#include <mpi2/logger.h>
#include <mpi2/task.h>
#include <mpi2/channel.h>
#include <mpi2/registry.h>
#include <mpi2/types.h>

namespace mpi2 {

// -- TaskManager (declaration) ------------------------------------------------------------------

/** Main driver for task management. Handles spawning of tasks and setup of communication
 * channels. Each rank has precisely one task manager; see the documentation of mpi2init()
 * and mpi2start() for information about firing up task managers.
 */
class TaskManager {
public:
	// -- Construction (singleton) ----------------------------------------------------------------

	/** Returns the instance of the task manager that runs at the current rank. Requires
	 * proper initialization via mpi2start(). */
	static inline TaskManager& getInstance() {
		return *theTaskManager_;
	}


	// -- Getters ---------------------------------------------------------------------------------

	/** Returns the tag of the task manager. The task manager listens on this tag for
	 * new commands. */
	int inline tag() const {
		return tag_;
	}

	/** Returns a fresh tag, guaranteeing that the tag is not in use. */
	int unusedTag();

	/** Returns n fresh tags, guaranteeing that the tag is not in use. */
	void unusedTags(int n, std::vector<int>& tags);

	/** Returns the MPI communicator used by this task manager. */
	inline boost::mpi::communicator& world() {
		return world_;
	}

	// -- Main logic ------------------------------------------------------------------------------

	/** Spawns the specified task at the specified rank.
	 *
	 * @return A communication channel to the newly spawned task
	 */
	Channel spawn(int rank, TaskId task);

	/** \copydoc spawn(int, TaskId) */
	template<typename IdentifiableTask>
	Channel spawn(int rank) {
		automaticRegister<IdentifiableTask>(rank);
		return spawn(rank, IdentifiableTask::id());
	}

	/** Spawns the specified task n times at the specified rank (in a task group).
	 *
	 * @return A vector of communication channel to the newly spawned tasks */
	void spawn(int rank, TaskId taskId, int n, std::vector<Channel>& channels, bool pairwiseChannels=false);

	/** \copydoc spawn(int, TaskId, int, std::vector<Channel>&) */
	template<typename IdentifiableTask>
	void spawn(int rank, int n, std::vector<Channel>& channels, bool pairwiseChannels=false) {
		automaticRegister<IdentifiableTask>(rank);
		spawn(rank, IdentifiableTask::id(), n, channels, pairwiseChannels);
	}

	/** Spawns the specified task on all ranks (in a task group).
	 *
	 * @return A vector of communication channel to the newly spawned tasks */
	void spawnAll(TaskId taskId, std::vector<Channel>& channels, bool pairwiseChannels=false);

	/** \copydoc spawnAll(TaskId, std::vector<Channel>&) */
	template<typename IdentifiableTask>
	void spawnAll(std::vector<Channel>& channels, bool pairwiseChannels=false) {
		if (world_.size() == 1)
			automaticRegister<IdentifiableTask>(world_.rank());
		spawnAll(IdentifiableTask::id(), channels, pairwiseChannels);
	}

	/** Spawns the specified task n times on each rank (in a task group).
	 *
	 * @return A vector of communication channel to the newly spawned tasks. Sorted by rank. */
	void spawnAll(TaskId taskId, int n, std::vector<Channel>& channels, bool pairwiseChannels=false);

	/** \copydoc spawnAll(TaskId, int, std::vector<Channel>&) */
	template<typename IdentifiableTask>
	void spawnAll(int n, std::vector<Channel>& channels, bool pairwiseChannels=false) {
		if (world_.size() == 1)
			automaticRegister<IdentifiableTask>(world_.rank());
		spawnAll(IdentifiableTask::id(), n, channels, pairwiseChannels);
	}

	/** Returns the current polling delay */
	unsigned pollDelay() const {
		return pollDelay_;
	}

	/** Set the polling delay. If the task manager does not receive a message, wait this many
	 * microseconds before polling again */
	void setPollDelay(unsigned pollDelay) {
		pollDelay_ = pollDelay;
	}

	/** Checks whether the task manager runs in parallel mode. */
	bool parallelMode() {
		return world_.size() == 1;
	}

	/** Hands the given request over to the task manager, which waits for its completion and
	 * then frees up resources. Once this method has been called, the request must not be used
	 * anymore.
	 */
	void finalizeRequest(boost::mpi::request req) {
		boost::mutex::scoped_lock lock(pendingRequestsMutex_);
		if (!req.test()) {
			pendingRequests_.push_back(req);
		}
	}

	// -- Private section -------------------------------------------------------------------------
private:
	/** Constructor. Called by mpi2init(). */
	TaskManager(int tag = 0) : tag_(tag), nextTag_(tag + 1), pollDelay_(0) {
	}

	// singleton
	static TaskManager* theTaskManager_; // initialized in mpi2init()
	TaskManager(); // no implementation
	TaskManager(const TaskManager&); // no implementation
	TaskManager& operator=(const TaskManager&); // no implementation

	// friend access to initialization methods
	friend boost::mpi::communicator& mpi2init(int& argc, char**& argv);
	friend void mpi2start(unsigned);
	friend void mpi2stop();

	void shutdown();

	void checkPendingRequests();

	template<typename IdentifiableTask>
	void automaticRegister(int rank) {
		std::string id = IdentifiableTask::id();
		if (TaskRegistry::getInstance().get(id) == NULL) {
			if (rank == world_.rank()) {
				registerTask<IdentifiableTask>();
				LOG4CXX_DEBUG(detail::logger, "Task " << id << " registered automatically at rank " << world_.rank());
			} else {
				LOG4CXX_ERROR(detail::logger, "Task " << id << " used without registration at rank " << world_.rank());
			}
		}
	}

	/** Task manager terminates upon receiving this task id */
	static const std::string TASK_ID_QUIT;

	/** Message type received by the task manager in order to spawn task */
	class TaskRequest {
	public:
		/** Tag to send response to */
		int responseTag;

		/** The identifier of the task */
		TaskId taskId;

		/** Source tags, one for each task to spawn */
		std::vector<int> sourceTags;

		/** The group id of the first task of the task group */
		int groupOffset;

		/** Total number of tasks in the task group */
		int groupSize;

		/** Whether to establish pairwise channels between the tasks of a task group */
		bool pairwiseChannels;

		TaskRequest() { };
		TaskRequest(int responseTag, TaskId taskId, std::vector<int> sourceTags,
				bool pairwiseChannels=false, int groupSize = 1, int groupOffset = 0)
		: responseTag(responseTag), taskId(taskId), sourceTags(sourceTags),
		  groupOffset(groupOffset), groupSize(groupSize), pairwiseChannels(pairwiseChannels) {

		};

		TaskRequest(int responseTag, TaskId taskId, int sourceTag,
				bool pairwiseChannels=false, int groupSize = 1, int groupOffset = 0)
		: responseTag(responseTag), taskId(taskId),
		  groupOffset(groupOffset), groupSize(groupSize), pairwiseChannels(pairwiseChannels) {
			sourceTags.resize(1, sourceTag);
		};

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & responseTag;
			ar & taskId;
			ar & sourceTags;
			ar & pairwiseChannels;
			ar & groupOffset;
			ar & groupSize;
		}
	};

	/** Message type sent by the task manager after tasks have been spawned */
	class TaskResponse {
	public:
		/** A tag to communicate with each of the spawned tasks */
		std::vector<int> taskTags;

		TaskResponse() { };

		TaskResponse(std::vector<int> taskTags) : taskTags(taskTags) {
		}

		TaskResponse(int taskTag) {
			taskTags.resize(1,taskTag);
		};

		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & taskTags;
		}
	};

	/** Main loop of the task manager. Receives and handles requests. */
	void run();

	/** Spawns a group of tasks (distributed mode) */
	void schedule(int source, TaskRequest taskRequest);

	/** Spawns a group of tasks (parallel mode) */
	std::vector<int> schedule(TaskRequest taskRequest);

	/** Run a new task, cleanup when finished. */
	void startTask(boost::mpi::communicator& world, int source, int sourceTag, Task task, int taskTag, TaskInfo info);

	/** Pool of threads */
	boost::threadpool::pool threadPool_;

		/** Mutex for the thread pool */
	boost::mutex mutex_;

	/** MPI communicator used by this task manager */
	boost::mpi::communicator world_;

	/** Tag on which this task manager listens */
	const int tag_;

	/** Next unused tag */
	int nextTag_;

	/** Mutex for nextTag_ */
	boost::mutex nextTagMutex_;
	boost::mutex spawnMutex_;

	/** Delay for poll thread (in microseconds) */
	unsigned pollDelay_;

	/** Requests to be finalized by the task manager. See #finalizeRequest().*/
	std::vector<boost::mpi::request> pendingRequests_;
	boost::mutex pendingRequestsMutex_;
};

} // namespace mpi2


#endif
