/** \file
 *
 * Methods to register tasks and types at workers.
 */

#ifndef MPI2_REGISTRY_H
#define MPI2_REGISTRY_H

#include <map>

#include <boost/thread/mutex.hpp>

#include <mpi2/channel.h>
#include <mpi2/env_fwd.h>
#include <mpi2/types.h>
#include <mpi2/task.h>

namespace mpi2 {

// -- TaskRegistry --------------------------------------------------------------------------------

/** Maps task identifiers to tasks, i.e., to their implementations. Every task that will be run by
 * mpi2 need to be registered at all nodes. The task registry is thread-safe.
 *
 * This class is best
 * accessed using the utility methods <code>mpi2::registerTask(TaskId taskId, Task task)</code>
 * and <code>mpi2::registerTask()</code>.
 *
 * @see Task
 */
class TaskRegistry {
public:
	/** Returns the task registry for this rank */
	static inline TaskRegistry& getInstance() {
		return theRegistry_;
	}

	/** Registers a task. */
	void registerTask(TaskId taskId, Task task, bool reregister = false) {
		boost::unique_lock<boost::shared_mutex> lock(mutex_);
		if (!reregister && registeredTasks_.find(taskId) != registeredTasks_.end()) {
			RG_THROW(rg::InvalidArgumentException, "TaskId " + taskId + " already exists");
		}
		registeredTasks_[taskId] = task;
	}

	/** Returns the task corresponding to the specified taskId or NULL if there is no
	 * such task. */
	Task get(TaskId taskId) const {
		boost::shared_lock<boost::shared_mutex> lock(mutex_);
		std::map<TaskId, Task>::const_iterator it = registeredTasks_.find(taskId);
		if (it == registeredTasks_.end()) {
			return NULL;
		}
		return it->second;
	};

private:
	TaskRegistry() { };
	TaskRegistry(const TaskRegistry& _); 	// no implementation
	TaskRegistry& operator=(const TaskRegistry&); // no implementation

	/** task registry for this rank */
	static TaskRegistry theRegistry_;

	/** Maps task identifiers to their implementation */
	std::map<TaskId, Task> registeredTasks_;

	/** Mutex for thread-safe access to registeredTasks_ */
	mutable boost::shared_mutex mutex_;
};


// -- Utility methods -----------------------------------------------------------------------------

/** Registers a task at the current rank.
 *
 * @see TaskRegistry
 */
inline void registerTask(TaskId taskId, Task task, bool reregister = false) {
	TaskRegistry::getInstance().registerTask(taskId, task, reregister);
}

/** Registers a task at the current rank.
 *
 * @tparam IdentifiableTask The task to register. Supplied type must have static methods
 * <code>TaskId id()</code> and
 * <code>run(boost::mpi::communicator& world, int source, int sourceTag, int taskTag)</code>.
 *
 * @see TaskRegistry
 */
template<typename IdentifiableTask>
inline void registerTask(bool reregister = false) {
	registerTask(IdentifiableTask::id(), IdentifiableTask::run, reregister);
}

// forward declaration
namespace detail {
	template<typename Cons>
	void registerEnvTasksForTypes();
}

/** Registers a type for use with mpi2.
 *
 * @tparam T a type (with mpi2::TypeTraits in scope)
 */
template<typename T>
inline void registerType() {
	detail::registerEnvTasksForTypes<Cons<T> >();
}

/** Registers a list of types for use with mpi2.
 *
 * @tparam Cons a list of types (each with mpi2::TypeTraits in scope)
 */
template<typename Cons>
inline void registerTypes() {
	registerType<typename Cons::Head>();
	registerTypes<typename Cons::Tail>();
}

template<>
inline void registerTypes<Nil>() {
};

}

#endif
