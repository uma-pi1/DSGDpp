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
#ifndef MF_AP_APTASK_H
#define MF_AP_APTASK_H

#include <boost/function.hpp>
#include <util/io.h>

namespace mf {

namespace detail {
	/** Task that runs a function on each block a matrix that (1) is partioned by row, (2)
	 * has row factors that are also partitioned by row, (3) has column factors that are
	 * stored on every node.
	 *
	 * @tparam f the function to run on each block
	 * @tparam UNIQUE_ID a unique identifier used for constructing task name (TODO: replace by string)
	 */
	template<double (*f)(const SparseMatrix&, const DenseMatrix&, const DenseMatrixCM&),
			unsigned UNIQUE_ID>
	struct ApTaskW {
		struct Arg {
			Arg() :vBlock(mpi2::UNINITIALIZED), wBlock(mpi2::UNINITIALIZED) { };
			Arg(mpi2::RemoteVar vBlock, mpi2::RemoteVar wBlock, const std::string& hUnblockedName,
					mf_size_type b1, mf_size_type b2)
			: vBlock(vBlock), wBlock(wBlock), hUnblockedName(hUnblockedName), b1(b1), b2(b2) {
			}
			mpi2::RemoteVar vBlock;
			mpi2::RemoteVar wBlock;
			std::string hUnblockedName;
			mf_size_type b1;
			mf_size_type b2;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				ar & vBlock;
				ar & wBlock;
				ar & hUnblockedName;
				ar & b1;
				ar & b2;
			}
		};

		static inline Arg
		constructArg(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const DistributedMatrix<DenseMatrix>& w, const std::string& hUnblockedName) {
			return Arg(block, w.block(b1, b2), hUnblockedName, b1, b2);
		}

		static const std::string id() { return rg::paste("__mf/matrix/op/ApTaskW_", UNIQUE_ID); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			std::vector<Arg> args;
			ch.recv(args);
			std::vector<boost::mpi::request> reqs(args.size());
			std::vector<double> result(args.size(), 0);
			DenseMatrix num;
			boost::numeric::ublas::vector<double> denom;
			for (unsigned i=0; i<args.size(); i++) {
				Arg& arg = args[i];
				const SparseMatrix& v = *arg.vBlock.template getLocal<SparseMatrix>();
				const DenseMatrix& w = *arg.wBlock.template getLocal<DenseMatrix>();
				const DenseMatrixCM& h = *mpi2::env().get<DenseMatrixCM>(arg.hUnblockedName);
				result[i] = f(v,w,h);
				reqs[i] = ch.isend(result[i]); // result
			}
			boost::mpi::wait_all(reqs.begin(), reqs.end());
		}
	};

	/** Task that runs a function on each block a matrix that (1) is partioned by row, (2)
	 * has row factors that are also partitioned by row, (3) has column factors that are
	 * stored on every node.
	 *
	 * @tparam f the function to run on each block
	 * @tparam UNIQUE_ID a unique identifier used for constructing task name (TODO: replace by string)
	 */
	template<double (*f)(const SparseMatrix&, const DenseMatrix&, const DenseMatrixCM&, int threads),
			unsigned UNIQUE_ID>
	struct ApTaskWThreads {
		struct Arg {
			Arg() :vBlock(mpi2::UNINITIALIZED), wBlock(mpi2::UNINITIALIZED) { };
			Arg(mpi2::RemoteVar vBlock, mpi2::RemoteVar wBlock, const std::string& hUnblockedName,
					mf_size_type b1, mf_size_type b2, int threads)
			: vBlock(vBlock), wBlock(wBlock), hUnblockedName(hUnblockedName), b1(b1), b2(b2),
			  threads(threads) {
			}
			mpi2::RemoteVar vBlock;
			mpi2::RemoteVar wBlock;
			std::string hUnblockedName;
			mf_size_type b1;
			mf_size_type b2;
			int threads;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				ar & vBlock;
				ar & wBlock;
				ar & hUnblockedName;
				ar & b1;
				ar & b2;
				ar & threads;
			}
		};

		static inline Arg
		constructArg(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const DistributedMatrix<DenseMatrix>& w, const std::string& hUnblockedName,
				int threads) {
			return Arg(block, w.block(b1, b2), hUnblockedName, b1, b2, threads);
		}

		static const std::string id() { return rg::paste("__mf/matrix/op/ApTaskWThreads_", UNIQUE_ID); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			std::vector<Arg> args;
			ch.recv(args);
			std::vector<boost::mpi::request> reqs(args.size());
			std::vector<double> result(args.size(), 0);
			DenseMatrix num;
			boost::numeric::ublas::vector<double> denom;
			for (unsigned i=0; i<args.size(); i++) {
				Arg& arg = args[i];
				const SparseMatrix& v = *arg.vBlock.template getLocal<SparseMatrix>();
				const DenseMatrix& w = *arg.wBlock.template getLocal<DenseMatrix>();
				const DenseMatrixCM& h = *mpi2::env().get<DenseMatrixCM>(arg.hUnblockedName);
				result[i] = f(v,w,h,arg.threads);
				reqs[i] = ch.isend(result[i]); // result
			}
			boost::mpi::wait_all(reqs.begin(), reqs.end());
		}
	};
}

}

#endif
