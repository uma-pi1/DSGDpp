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
 *  Methods to facilitate parallel processing on distributed matrices.
 */

#ifndef MF_MATRIX_DISTRIBUTE_H
#define MF_MATRIX_DISTRIBUTE_H

#include <vector>
#include <util/exception.h>
#include <mpi2/env.h>

#include <mf/logger.h>
#include <mf/matrix/distributed_matrix.h>

namespace mf {

/** Argument constructor for mf::runTaskOnBlocks for factorization tasks. Produces a vector of
 * three remote variables: one for block (b1,b2) of a distributed matrix, one for block (b1,0) of
 * a conforming row factor matrix, and one for block (0,b2) of a conforming column factor matrix.
 *
 * @param b1 the row block number
 * @param b2 the column block number
 * @param block remote variable referencing the blocks data
 * @param w conforming row factor matrix (blocks1 x 1 blocks)
 * @param h conforming column factor matrix (1 x blocks2 blocks)
 * @tparam M2 type of row factor matrix
 * @tparam M3 type of column factor matrix
 */
template<typename M2, typename M3>
inline std::vector<mpi2::RemoteVar> argBlockRV3(
		mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
		const DistributedMatrix<M2>& w, const DistributedMatrix<M3>& h) {
	std::vector<mpi2::RemoteVar> result;
	result.push_back(block);
	result.push_back(w.block(b1,0));
	result.push_back(h.block(0,b2));
	return result;
}

/** Spawns a set of tasks that run some code on each block of a distributed matrix.
 *
 * This method spawns 'tasksPerRank' instances of the given task ('taskId')
 * on each node. Each task is responsible for processing a subset of the blocks of the input
 * matrix. In particular, each task is sent a vector of "arguments" (one for each block
 * assigned to it) constructed by the specified 'construct' function (see mf::argBlockRV and
 * mf::argBlockRV3 for examples). The argument usually
 * contains information about the block (row/column, location), but may also contain other
 * information. Each task then runs some function on each of the blocks/arguments and sends back the
 * results one-by-one (i.e., one result message per argument).
 *
 * @param m a distributed matrix (blocks1 x blocks2)
 * @param result a matrix used to store the result (blocks1 x blocks2)
 * @param constructTaskArg a function that takes block coordinates as inputs and
 *        returns the argument to the task run on that block
 * @param taskId the task to run
 * @param tasksPerRank number of parallel tasks to run on each node
 * @param asyncRecv whether to receive results asynchronously (requires 'result' to be threadsafe)
 *
 * @tparam M type of matrix blocks
 * @tparam R type of result
 * @tparam A type of arguments for block-local functions
 * @tparam F type of function for constructing arguments; signature A f(mf_size_type b1,
 *           mf_size_type b2, mpi2::RemoteVar block)
 *
 */
template<typename M, typename R, typename A, typename F>
void runTaskOnBlocks(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<R>& result,
		const F& construct,
		const std::string& taskId,
		int tasksPerRank=1, bool asyncRecv = true, int pollDelay = -1);

/** Specialization of runTasksOnBlocks for the common case in which task arguments are locations
 * of the input block as well as the block of corresponding row and column factors (see
 * mf::argBlockRV3). Use mf::runFunctionPerBlock for quickly implementing the corresponding
 * task. */
template<typename M1, typename M2, typename M3, typename R>
void runTaskOnBlocks3(const DistributedMatrix<M1>& v,
		const DistributedMatrix<M2>& w, const DistributedMatrix<M3>& h,
		boost::numeric::ublas::matrix<R>& result,
		const std::string& taskId,
		int tasksPerRank=1, bool asyncRecv = true, int pollDelay = -1) {
	runTaskOnBlocks<M1,R,std::vector<mpi2::RemoteVar> >(v, result,
			boost::bind(argBlockRV3<M2,M3>, _1, _2, _3, boost::cref(w), boost::cref(h)),
			taskId, tasksPerRank, asyncRecv, pollDelay);
}

// precompile
extern template void runTaskOnBlocks3<SparseMatrix,DenseMatrix,DenseMatrixCM>(
		const DistributedMatrix<SparseMatrix>& v,
		const DistributedMatrix<DenseMatrix>& w, const DistributedMatrix<DenseMatrixCM>& h,
		boost::numeric::ublas::matrix<double>& result, const std::string& taskId,
		int tasksPerRank=1, bool asyncRecv = true, int pollDelay = -1);


/** Helper method to implement tasks for mf::runTaskOnBlocks for arguments that are block
 * locations and corresponding block locations from row and column factors (see mf::argBlockRV3).
 * Takes care of all communication; this method invokes the specified function on all
 * blocks assigned to the current task.
 *
 * Implement the task's run method as follows:
 * <code>
 * static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
 *		runFunctionPerAssignedBlock3<M1,M2,M3,R>(ch, &f);
 * }
 * </code>
 *
 * @param ch
 * @param f
 */
template<typename M1, typename M2, typename M3, typename R>
void runFunctionPerAssignedBlock3(mpi2::Channel ch,
		boost::function<R (const M1&, const M2&, const M3&)> f);


namespace detail {
	/** Internal type of argument used for per-block task */
	template<typename ArgType>
	struct PerBlockTaskArg {
		typedef ArgType Type;

		PerBlockTaskArg() : b1(-1), b2(-1), block(mpi2::UNINITIALIZED), arg() {
		}

		PerBlockTaskArg(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar rv, ArgType arg)
		: b1(b1), b2(b2), block(rv), arg(arg) {
		}

		static PerBlockTaskArg construct(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block, ArgType arg) {
			return PerBlockTaskArg(b1, b2, block, arg);
		}

		mf_size_type b1;
		mf_size_type b2;
		mpi2::RemoteVar block;
		ArgType arg;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & b1;
			ar & b2;
			ar & block;
			ar & arg;
		}
	};
}

// -- Tasks with arguments ------------------------------------------------------------------------

/** Tasks that runs a function on each block of a matrix and collects return values.
 *
 * The function gets passed
 * (1) the row block number,
 * (2) the column block number,
 * (3) the matrix block, and
 * (4) a user-defined argument (same for all blocks).
 *
 * @tparam M matrix type (of blocks)
 * @tparam ReturnType type of result of function
 * @tparam ArgType type of argument given to the function
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, typename ReturnType, typename ArgType, ReturnType (*f)(mf_size_type, mf_size_type, M&, ArgType), unsigned UNIQUE_ID>
struct PerBlockTaskReturnArgIndex {
	typedef PerBlockTaskReturnArgIndex<M, ReturnType, ArgType, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<ArgType> Arg;
	typedef ReturnType Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTaskWithArg", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<ReturnType> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			results[i] = f(vars[i].b1, vars[i].b2, m, vars[i].arg);
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

/** Tasks that runs a function on each block of a matrix and collects return values.
 *
 * The function gets passed
 * (1) the matrix block,
 * (2) a user-defined argument (same for all blocks).
 *
 * @tparam M matrix type (of blocks)
 * @tparam ReturnType type of result of function
 * @tparam ArgType type of argument given to the function
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, typename ReturnType, typename ArgType, ReturnType (*f)(M&, ArgType), unsigned UNIQUE_ID>
struct PerBlockTaskReturnArg {
	typedef PerBlockTaskReturnArg<M, ReturnType, ArgType, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<ArgType> Arg;
	typedef ReturnType Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTaskWithArg", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<ReturnType> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			results[i] = f(m, vars[i].arg);
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

/** Tasks that runs a void function on each block of a matrix.
 *
 * The function gets passed
 * (1) the row block number,
 * (2) the column block number,
 * (3) the matrix block, and
 * (4) a user-defined argument (same for all blocks).
 *
 * @tparam M matrix type (of blocks)
 * @tparam ArgType type of argument given to the function
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, typename ArgType, void (*f)(mf_size_type, mf_size_type, M&, ArgType), unsigned UNIQUE_ID>
struct PerBlockTaskVoidArgIndex {
	typedef PerBlockTaskVoidArgIndex<M, ArgType, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<ArgType> Arg;
	typedef int Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTaskWithArg", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<Return> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			f(vars[i].b1, vars[i].b2, m, vars[i].arg);
			reqs[i] = ch.isend((int)0);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

/** Tasks that runs a void function on each block of a matrix.
 *
 * The function gets passed
 * (1) the matrix block, and
 * (2) a user-defined argument (same for all blocks).
 *
 * @tparam M matrix type (of blocks)
 * @tparam ArgType type of argument given to the function
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, typename ArgType, void (*f)(M&, ArgType), unsigned UNIQUE_ID>
struct PerBlockTaskVoidArg {
	typedef PerBlockTaskVoidArg<M, ArgType, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<ArgType> Arg;
	typedef int Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTaskWithArg", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<Return> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			f(m, vars[i].arg);
			reqs[i] = ch.isend((int)0);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};


// -- Tasks without arguments --------------------------------------------------------------------

namespace detail {
	struct NoArg {
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
		}
	};
}

/** Tasks that runs a function on each block of a matrix and collects return values.
 *
 * The function gets passed
 * (1) the row block number,
 * (2) the column block number,
 * (3) the matrix block.
 *
 * @tparam M matrix type (of blocks)
 * @tparam ReturnType type of result of function
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, typename ReturnType, ReturnType (*f)(mf_size_type, mf_size_type, M&), unsigned UNIQUE_ID>
struct PerBlockTaskReturnIndex {
	typedef PerBlockTaskReturnIndex <M, ReturnType, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<detail::NoArg> Arg;
	typedef ReturnType Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTask", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<ReturnType> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			results[i] = f(vars[i].b1, vars[i].b2, m);
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

/** Tasks that runs a function on each block of a matrix and collects return values.
 *
 * The function gets passed
 * (1) the matrix block
 *
 * @tparam M matrix type (of blocks)
 * @tparam ReturnType type of result of function
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, typename ReturnType, ReturnType (*f)(M&), unsigned UNIQUE_ID>
struct PerBlockTaskReturn {
	typedef PerBlockTaskReturn<M, ReturnType, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<detail::NoArg> Arg;
	typedef ReturnType Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTask", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<ReturnType> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			results[i] = f(m);
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

/** Tasks that runs a void function on each block of a matrix.
 *
 * The function gets passed
 * (1) the row block number,
 * (2) the column block number,
 * (3) the matrix block
 *
 * @tparam M matrix type (of blocks)
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, void (*f)(mf_size_type, mf_size_type, M&), unsigned UNIQUE_ID>
struct PerBlockTaskVoidIndex {
	typedef PerBlockTaskVoidIndex <M, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<detail::NoArg> Arg;
	typedef int Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTask", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<Return> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			f(vars[i].b1, vars[i].b2, m);
			reqs[i] = ch.isend((int)0);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};


/** Tasks that runs a void function on each block of a matrix.
 *
 * The function gets passed
 * (1) the matrix block
 *
 * @tparam M matrix type (of blocks)
 * @tparam f the function to call
 * @tparam UNIQUE_ID a unique identifier (see mf/id.h)
 */
template<typename M, void (*f)(M&), unsigned UNIQUE_ID>
struct PerBlockTaskVoid {
	typedef PerBlockTaskVoid<M, f, UNIQUE_ID> Task;
	typedef M Matrix;
	typedef detail::PerBlockTaskArg<detail::NoArg> Arg;
	typedef int Return;

	static const std::string id() { return rg::paste(
			"__mf/matrix/op/PerBlockTask", mpi2::TypeTraits<M>::name(), "_", UNIQUE_ID) ; }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> vars;
		ch.recv(vars);
		std::vector<boost::mpi::request> reqs(vars.size());
		std::vector<Return> results(vars.size());
		for (unsigned i=0; i<vars.size(); i++) {
			M& m = *vars[i].block.template getLocal<M>();
			f(m);
			reqs[i] = ch.isend((int)0);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};


/** Runs a task of type PerBlockTaskReturnArgIndex or PerBlockTaskReturnArg and collects
 * results. */
template<typename PerBlockTask>
void runTaskOnBlocks(
		const DistributedMatrix<typename PerBlockTask::Task::Matrix>& m,
		typename PerBlockTask::Task::Arg::Type arg,
		boost::numeric::ublas::matrix<typename PerBlockTask::Task::Return>& result,
		int tasksPerRank = 1, bool asyncRecv = true, int pollDelay = -1) {
	result.resize(m.blocks1(), m.blocks2(), false);
	runTaskOnBlocks<typename PerBlockTask::Task::Matrix,
	typename PerBlockTask::Task::Return, typename PerBlockTask::Task::Arg>(
			m,
			result,
			boost::bind(PerBlockTask::Task::Arg::construct, _1, _2, _3, arg),
			PerBlockTask::id(), tasksPerRank, asyncRecv, pollDelay);
};

/** Runs a task of type PerBlockTaskReturnIndex or PerBlockTaskReturn and collects
 * results. */
template<typename PerBlockTask>
void runTaskOnBlocks(
		const DistributedMatrix<typename PerBlockTask::Task::Matrix>& m,
		boost::numeric::ublas::matrix<typename PerBlockTask::Task::Return>& result,
		int tasksPerRank = 1, bool asyncRecv = true, int pollDelay = -1) {
	runTaskOnBlocks<PerBlockTask>(m, detail::NoArg(), result, tasksPerRank, asyncRecv, pollDelay);
};

/** Runs a task of type PerBlockTaskVoidArgIndex, PerBlockTaskVoidArg, PerBlockTaskReturnArgIndex,
 * PerBlockTaskReturnArg and drops results. */
template<typename PerBlockTask>
void runTaskOnBlocks(
		const DistributedMatrix<typename PerBlockTask::Task::Matrix>& m,
		typename PerBlockTask::Task::Arg::Type arg,
		int tasksPerRank = 1, bool asyncRecv = true, int pollDelay = -1) {
	boost::numeric::ublas::matrix<typename PerBlockTask::Task::Return> result(m.blocks1(), m.blocks2());
	runTaskOnBlocks<PerBlockTask>(m, arg, result, tasksPerRank, asyncRecv, pollDelay);
};

/** Runs a task of type PerBlockTaskVoidIndex, PerBlockTaskVoid, PerBlockTaskReturnIndex,
 * PerBlockTaskReturn and drops results. */
template<typename PerBlockTask>
void runTaskOnBlocks(
		const DistributedMatrix<typename PerBlockTask::Task::Matrix>& m,
		int tasksPerRank = 1, bool asyncRecv = true, int pollDelay = -1) {
	runTaskOnBlocks<PerBlockTask>(m, detail::NoArg(), tasksPerRank, asyncRecv, pollDelay);
};

} // namespace mf

#include <mf/matrix/distribute_impl.h>

#endif
