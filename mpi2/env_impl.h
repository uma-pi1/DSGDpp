/** \file
 *
 * Implementation for env.h (DO NOT INCLUDE DIRECTLY) */
#include <mpi2/env.h> // just to help IDEs
#include <mpi2/logger.h>
#include <mpi2/registry.h>
#include <mpi2/task-manager.h>

namespace mpi2 {

// -- Env implementation (inlines and templates) --------------------------------------------------

inline bool Env::defined(Var var) const {
	boost::shared_lock<boost::shared_mutex> lock(varsMutex_);
	bool b = vars_.find(var) != vars_.end();
	return b;
}

// synchronize changes with const method below
template<typename T>
T* Env::get(const Var& var) {
	LOG4CXX_TRACE(detail::logger, "@" << rank_ << ": "
			<< "Entering get(\"" << var << "\")");

	boost::shared_lock<boost::shared_mutex> lock(varsMutex_);
	std::map<Var, VarEntry*>::iterator it = vars_.find(var);
	T* value;
	if (it == vars_.end()) {
		value = NULL;
	} else {
		VarEntry *v = it->second;
		boost::mutex::scoped_lock lock2(v->mutex);
		value = boost::any_cast<T*>(v->value);
	}

	LOG4CXX_TRACE(detail::logger, "@" << rank_ << ": "
			<< "Leaving get(\"" << var << "\")");
	return value;
}

// same code as above
template<typename T>
T* const Env::get(const Var& var) const {
	LOG4CXX_TRACE(detail::logger, "@" << rank_ << ": "
			<< "Entering get(\"" << var << "\")");

	boost::shared_lock<boost::shared_mutex> lock(varsMutex_);
	std::map<Var, VarEntry*>::iterator it = vars_.find(var);
	T* const value;
	if (it == vars_.end()) {
		value = NULL;
	} else {
		VarEntry *v = it->second;
		boost::mutex::scoped_lock lock2(v->mutex);
		value = boost::any_cast<T* const>(v->value);
	}

	LOG4CXX_TRACE(detail::logger, "@" << rank_ << ": "
			<< "Leaving get(\"" << var << "\")");
	return value;
}

template<typename T>
void Env::setCopy(const Var& var, const T& value) {
	if (!defined(var)) {
		RG_THROW(rg::InvalidArgumentException, var + "not defined");
	}
	*get<T>(var) = value;
}

inline const std::type_info & Env::type(const Var& var) const {
	boost::shared_lock<boost::shared_mutex> lock(varsMutex_);
	std::map<Var, VarEntry*>::const_iterator it = vars_.find(var);
	if (it == vars_.end()) {
		return typeid(NULL);
	} else {
		VarEntry *v = it->second;
		boost::mutex::scoped_lock lock2(v->mutex);
		const std::type_info &type = v->value.type();
		return type;
	}
}

template<typename T>
RemoteVar Env::create(const Var& var, T* value,
		DeleteOptions delete_on_erase) {
	boost::shared_lock<boost::shared_mutex> lock(varsMutex_);
	bool found = vars_.find(var) != vars_.end();
	if (found) {
		RG_THROW(rg::IllegalStateException, std::string("name '") + var + "' exits already");
	}
	VarEntry* v = new VarEntry();
	v->value = value;
	v->delete_on_erase = delete_on_erase;
	vars_[var] = v;
	return RemoteVar(rank_, var);
}

template<typename T>
void Env::erase(const Var& var) {
	boost::shared_lock<boost::shared_mutex> lock(varsMutex_);
	std::map<Var, VarEntry*>::iterator it = vars_.find(var);
	bool found = it != vars_.end();
	if (!found) {
		RG_THROW(rg::IllegalStateException, std::string("name '") + var + "' does not exist");
	}
	VarEntry *v = it->second;
	{
		boost::mutex::scoped_lock lock2(v->mutex);
		vars_.erase(it);
	}
	switch(v->delete_on_erase) {
	case delete_value:
		delete boost::any_cast<T*>(v->value);
		break;
	case delete_array:
		delete[] boost::any_cast<T*>(v->value);
		break;
	case no_delete:
		break;
	}
	delete v;
}

inline std::vector<Env::Var> Env::vars() const {
	boost::shared_lock<boost::shared_mutex> lock(varsMutex_);
	std::vector<Var> result;
	for(std::map<Var, VarEntry*>::const_iterator it = vars_.begin(); it!=vars_.end(); ++it) {
		result.push_back(it->first);
	}
	return result;
}

// -- Tasks for remote access (implementation) ----------------------------------------------------

namespace detail {
	template<typename T>
	struct TaskEnvFetch {
		static inline std::string id() { return std::string("mpi2.env.TaskEnvFetch/") + TypeTraits<T>::name(); }
		static inline void run(Channel ch, TaskInfo info) {
			std::string name;
			ch.recv(name);
			boost::mpi::request req = ch.isend(*(mpi2::env().get<T>(name)));
			TaskManager::getInstance().finalizeRequest(req);
		}
	};

	template<typename T>
	struct TaskEnvCreate {
		static inline std::string id() { return std::string("mpi2.env.TaskEnvCreate/") + TypeTraits<T>::name(); }
		static inline void run(Channel ch, TaskInfo info) {
			std::string name;
			ch.recv(name);
			T* value = Mpi2Constructor<T>::construct();
			ch.economicRecv(*value, TaskManager::getInstance().pollDelay());
			mpi2::env().create(name, value);
			boost::mpi::request req = ch.isend(); // ack
			TaskManager::getInstance().finalizeRequest(req);
		}
	};

	template<typename T>
	struct TaskEnvErase {
		static inline std::string id() { return std::string("mpi2.env.TaskEnvErase/") + TypeTraits<T>::name(); }
		static inline void run(Channel ch, TaskInfo info) {
			std::string name;
			ch.recv(name);
			mpi2::env().erase<T>(name);
			boost::mpi::request req = ch.isend(); // ack
			TaskManager::getInstance().finalizeRequest(req);
		}
	};

	template<typename T>
	struct TaskEnvStore {
		static inline std::string id() { return std::string("mpi2.env.TaskEnvStore/") + TypeTraits<T>::name(); }
		static inline void run(Channel ch, TaskInfo info) {
			std::string name;
			ch.recv(name);
			//T* value = new T();
			T* value = mpi2::env().get<T>(name); // TODO: locking...
			boost::mpi::request req = ch.irecv(*value);
			TaskManager::getInstance().finalizeRequest(req);
		}
	};
} // namespace detail


// -- Methods for remote access (implementation) --------------------------------------------------

template<typename T>
void getCopy(int rank, const std::string& name, T& result) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	if (rank == world.rank()) {
		result = *(env().get<T>(name));
	} else {
		Channel ch = tm.spawn<detail::TaskEnvFetch<T> >(rank);
		ch.send(name);
		ch.economicRecv(result, tm.pollDelay());
	}
}

template<typename T>
boost::mpi::request igetCopy(int rank, const std::string& name, T& result) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	Channel ch = tm.spawn<detail::TaskEnvFetch<T> >(rank);
	ch.send(name);
	return ch.irecv(result);
}

template<typename T>
void setCopy(int rank, const std::string& name, const T& value) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	if (rank == world.rank()) {
		*env().get<T>(name) = value;
	} else {
		Channel ch = tm.spawn<detail::TaskEnvStore<T> >(rank);
		ch.send(name);
		ch.economicSend(value, tm.pollDelay());
	}
}

template<typename T>
boost::mpi::request isetCopy(int rank, const std::string& name, const T& value) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	Channel ch = tm.spawn<detail::TaskEnvStore<T> >(rank);
	ch.send(name);
	return ch.isend(value);
}

template<typename T>
void setCopyAll(const std::string& name, const T& value) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	std::vector<boost::mpi::request> reqs;
	for (int rank=0; rank<world.size(); rank++) {
		if (rank == world.rank()) {
			*env().get<T>(name) = value;
		} else {
			Channel ch = tm.spawn<detail::TaskEnvStore<T> >(rank);
			reqs.push_back(ch.isend(name)); // TODO: marshall
			reqs.push_back(ch.isend(value));
		}
	}
	economicWaitAll(reqs, tm.pollDelay());
}

template<typename T>
RemoteVar createCopy(int rank, const std::string& name, const T& value) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	if (rank == world.rank()) {
		T* v = Mpi2Constructor<T>::construct();
		*v = value;
		env().create<T>(name, v);
	} else {
		Channel ch = tm.spawn<detail::TaskEnvCreate<T> >(rank);
		std::vector<boost::mpi::request> reqs(3);
		reqs[0] = ch.isend(name); // TODO: marshall
		reqs[1] = ch.isend(value);
		reqs[2] = ch.irecv();
		economicWaitAll(reqs, tm.pollDelay());
	}
	return RemoteVar(rank, name);
}

template<typename T>
void createCopyAll(const std::string& name, const T& value) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	std::vector<boost::mpi::request> reqs;
	for (int rank=0; rank<world.size(); rank++) {
		if (rank == world.rank()) {
			T* v = Mpi2Constructor<T>::construct();
			*v = value;
			env().create<T>(name, v);
		} else {
			Channel ch = tm.spawn<detail::TaskEnvCreate<T> >(rank);
			reqs.push_back(ch.isend(name)); // TODO: marshall
			reqs.push_back(ch.isend(value));
			reqs.push_back(ch.irecv());
		}
	}
	economicWaitAll(reqs, tm.pollDelay());
}

template<typename T>
void erase(int rank, const std::string& name) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	if (rank == world.rank()) {
		env().erase<T>(name);
	} else {
		Channel ch = tm.spawn<detail::TaskEnvErase<T> >(rank);
		ch.isend(name);
		ch.economicRecv(tm.pollDelay());
	}
}

template<typename T>
void eraseAll(const std::string& name) {
	TaskManager& tm = TaskManager::getInstance();
	boost::mpi::communicator& world = tm.world();
	std::vector<boost::mpi::request> reqs;
	for (int rank=0; rank<world.size(); rank++) {
		if (rank == world.rank()) {
			env().erase<T>(name);
		} else {
			Channel ch = tm.spawn<detail::TaskEnvErase<T> >(rank);
			reqs.push_back(ch.isend(name));
			reqs.push_back(ch.irecv());
		}
	}
	economicWaitAll(reqs, tm.pollDelay());
}


// -- Tasks registration --------------------------------------------------------------------------

// forward declaration (from register.h)
template<typename IdentifiableTask>
inline void registerTask(bool reregister = false);

namespace detail {

/** Register all env tasks for each type in the given list of types
 * @param tm the task manager at which to register (see mpi2::Cons)
 * @tparam Cons a list of types
 */
template<typename Cons>
void registerEnvTasksForTypes() {
	registerTask<TaskEnvFetch<typename Cons::Head> >(true);
	registerTask<TaskEnvStore<typename Cons::Head> >(true);
	registerTask<TaskEnvCreate<typename Cons::Head> >(true);
	registerTask<TaskEnvErase<typename Cons::Head> >(true);
	registerEnvTasksForTypes<typename Cons::Tail>();
};

template<>
inline void registerEnvTasksForTypes<Nil>() {
};

}

} // namespace mpi2

