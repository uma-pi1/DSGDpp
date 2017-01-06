/** \file
 *
 * Methods for handling environments. */

#ifndef MF_ENV_FWD_H
#define MF_ENV_FWD_H

#include <vector>
#include <set>
#include <map>
#include <string>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/thread.hpp>
#include <boost/any.hpp>

#include <util/exception.h>
#include <util/io.h>

#include <mpi2/task.h>
#include <mpi2/types.h>
#include <mpi2/uninitialized.h>

namespace mpi2 {

class RemoteVar;

// -- Env declaration -----------------------------------------------------------------------------

/** A set of named variables and their values. Access to the
 * environment itself is thread-safe, but access to the values of the variables
 * is currently not. Optionally, environments may know on which MPI node they
 * are stored (i.e., the node's rank).
 *
 * The global environment of the current tank can be retrieved via mpi2::env().
 *
 * Note that all the types used in an environment have to be registered
 * via mpi2::registerType() or mpi2::registerTypes().
 */
class Env {
	friend class RemoteVar;

public:
	/** Type of variable names */
	typedef std::string Var; // variable name // TODO: make hierarchical

	/** Options of what to do when a value gets removed from the environment */
	enum DeleteOptions { no_delete, delete_value, delete_array };

private:
	/** An entry for a variable */
	struct VarEntry {
		/** Mutex for exclusive access */
		boost::mutex mutex;
		// TODO: keep track of readers and writers to make use of entries safer

		/** Value of the variable */
		boost::any value;

		/** What to do when a variable gets removed */
		DeleteOptions delete_on_erase;
	};

	/** Maps variable names to their values */
	std::map<Var, VarEntry*> vars_;

	/** Mutex for ensuring exclusive access to vars_ */
	mutable boost::shared_mutex varsMutex_;

	/** Rank of node at which this environment is stored */
	const int rank_;

	Env(const Env&); // forbidden; avoid accidental copy construction

public:
	/** Construct a new environment */
	Env(int rank=-1) : rank_(rank) {
	}

	/** Delete the environment */
	~Env() {
		// TODO: free memory
	}

	/** Checks whether the variable of the given name is stored in this environment */
	bool defined(Var var) const;

	/** Returns a pointer to the value of the specified variable without copying.
	 * If the value is not defined, returns NULL.
	 *
	 * @param var name of variable
	 * @tparam T type of variable
	 */
	template<typename T>
	T* get(const Var& var);

	/** \copydoc get(const Var& var) */
	template<typename T>
	T* const get(const Var& var) const;

	/** Update the value of the variable (copies argument).
	 * @tparam type of variable (must match current type!)
	 * @throws rg::InvalidArgumentException if var is not defined
	 */
	template<typename T>
	void setCopy(const Var& var, const T& value);

	/** Returns information about the type of the specified variable. If the variable
	 * is undefined, returns NULL.
	 *
	 * @param var name of variable
	 */
	const std::type_info & type(const Var& var) const;

	/** Creates a new variable and assigns it the specified value (without copying).
	 *
	 * @param var name of the new variable
	 * @param value pointer to value of new variable (this pointer is stored)
	 * @param delete_on_erase what to do when the variable is removed from the environment
	 * @tparam T type of variable
	 * @throws rg::IllegalStateException if a variable of the specified name has
	 *         been defined already
	 */
	template<typename T>
	RemoteVar create(const Var& var, T* value, DeleteOptions delete_on_erase = delete_value);

	/** Removes the specified variable from the environment. This
	 * method may free the memory of the variable's value (depending
	 * on the variables DeleteOptions.
	 *
	 * @tparam T the variables type
	 */
	template<typename T>
	void erase(const Var& var);

	/** Returns a list of variable names stored in this environment. The list is not
	 * backed by this environment, i.e., modifications to this list do not affect
	 * this environment.
	 */
	std::vector<Var> vars() const;
};

// -- Global environment (declaration) ------------------------------------------------------------

/** Initializes the global environment of the current rank.
 * @param rank rank of the current node
 */
void initEnv(int rank=-1);

/** Returns a pointer to the global environment stored on this rank */
Env& env();

/** Destroys the global environment on the corrent node */
void destroyEnv();


// -- Methods for remote access (declaration) -----------------------------------------------------

/** Fetches the variable of the specified name from node 'rank'. The variable is copied into
 * the specified 'result' value (even if the variable turns out to be local). */
template<typename T>
void getCopy(int rank, const std::string& name, T& result);

template<typename T>
boost::mpi::request igetCopy(int rank, const std::string& name, T& result);

/** Overwrites the value of the specified variable in the environment of node 'rank'.
 * The variable is copied (even if the enviroment is local). */
template<typename T>
void setCopy(int rank, const std::string& name, const T& value);

template<typename T>
boost::mpi::request isetCopy(int rank, const std::string& name, const T& value);

/** Overwrites the value of the specified variable in the environment on all ranks.
 * The variable is copied (even if the enviroment is local).
 *
 * TODO: add real support for replicated variables
 */
template<typename T>
void setCopyAll(const std::string& name, const T& value);

/** Creates the specified variable in the environment of node 'rank'.
 * The variable is copied (even if the enviroment is local). */
template<typename T>
RemoteVar createCopy(int rank, const std::string& name, const T& value);

/** Creates the specified variable in the environment on all ranks.
 * The variable is copied (even if the enviroment is local).
 *
 * TODO: add real support for replicated variables
 */
template<typename T>
void createCopyAll(const std::string& name, const T& value);

/** Deletes the specified variable at the specified rank. The variable must exist already. */
template<typename T>
void erase(int rank, const std::string& name);

/** Erases the specified variable in the environment on all ranks.
 * The variable must exist already.
 */
template<typename T>
void eraseAll(const std::string& name);

namespace detail {
/** Registers tasks for remote access to environments */
void registerEnvTasks();
}

// -- RemoteVar declaration -----------------------------------------------------------------------

/** A remote variable is a pointer to a variable stored on some node in the cluster.
 * It can be used to access this variable in multiple ways. */
class RemoteVar {
public:
	/** SerializationConstructor for a RemoteVar. */
	RemoteVar(SerializationConstructor _) : rank_(-1) { }; // for serialization

	/** Default constructor */
	RemoteVar(int rank, const Env::Var& var) : rank_(rank), var_(var) { };

	/** Returns the rank of the node at which the variable is stored */
	inline int rank() const { return rank_; }

	/** Returns the variable name */
	inline const Env::Var& var() const { return var_; }

	/** Checks whether the variable is stored at the current node */
	inline bool isLocal() const {
		return rank_ == env().rank_;
	}

	/** Gets a pointer to the variable if stored on local node (see Env::get).
	 * @throws rg::InvalidArgumentException if variable is not stored locally
	 */
	template<typename T>
	inline T* getLocal() {
		if (!isLocal()) {
			RG_THROW(rg::InvalidArgumentException, rg::paste("remote var ", *this,  " not local at rank ", env().rank_));
		}
		return env().get<T>(var_);
	}

	/** \copydoc getLocal() */
	template<typename T>
	inline T* const getLocal() const {
		if (!isLocal()) {
			RG_THROW(rg::InvalidArgumentException, rg::paste("remote var ", *this,  " not local at rank ", env().rank_));
		}
		return env().get<T>(var_);
	}

	/** Fetches the variable. Data is copied into the specified 'result'
	 * value (even if the variable turns out to be local). */
	template<typename T>
	inline void getCopy(T& result) const {
		mpi2::getCopy(rank_, var_, result);
	}

	/** Fetches the variable. Data is copied into the specified 'result'
	 * value (even if the variable turns out to be local). Communication is asynchronous;
	 * returns immediately. */
	template<typename T>
	boost::mpi::request igetCopy(T& result) const {
		return mpi2::igetCopy(rank_, var_, result);
	}

	/** Overwrites the value of the variable.
	 * Data is copied (even if the enviroment is local). */
	template<typename T>
	inline void setCopy(const T& value) {
		mpi2::setCopy(rank_, var_, value);
	}

	template<typename T>
	inline boost::mpi::request isetCopy(const T& value) {
		return mpi2::isetCopy(rank_, var_, value);
	}

	/** Create the variable. Data is copied (even if the enviroment is local). */
	template<typename T>
	inline void createCopy(const T& value) {
		mpi2::createCopy(rank_, var_, value);
	}

	/** Erases the remote variable. The variable must exist already. */
	template<typename T>
	void erase() {
		mpi2::erase<T>(rank_, var_);
	}

	bool operator==(const RemoteVar& b) const {
		return rank_ == b.rank_ && var_ == b.var_;
	}

	bool operator !=(const RemoteVar& b) const {
		return !(operator==(b));
	}

private:
	/** Rank of node that stores the variable */
	int rank_;

	/** Variable name */
	Env::Var var_;

	// access to default constructor for collections
	// this allows collections to be constructed or resized; in this case,
	// uninitialized RemoteVars will used as default elements
	friend class boost::numeric::ublas::matrix<RemoteVar>;
	friend class boost::numeric::ublas::unbounded_array<RemoteVar>;
	friend class std::vector<RemoteVar>;

	RemoteVar() { };


	// serialization
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & rank_;
		ar & var_;
	}
};

} // namespace mpi2

MPI2_SERIALIZATION_CONSTRUCTOR(mpi2::RemoteVar)

namespace mpi2 {


// -- Utility methods (declaration) ---------------------------------------------------------------

/** Prints the names of all variables stored on each node */
void lsAll();

/** Formatted output of a remote variable*/
template<typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>& operator<<(
		std::basic_ostream<CharT, Traits>& out, const RemoteVar& remoteVar)
{
	out << remoteVar.var() << "@" << remoteVar.rank();
	return out;
}


// -- Tasks registration (declaration)-------------------------------------------------------------

namespace detail {
/** Register all env tasks for each type in the given list of types
 * @param tm the task manager at which to register (see mpi2::Cons)
 * @tparam Cons a list of types
 */
template<typename Cons>
void registerEnvTasksForTypes();

} // namespace detail

} // namespace mpi2

#endif
