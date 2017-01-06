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
#ifndef MF_AP_APUPDATE_H
#define MF_AP_APUPDATE_H

#include <boost/function.hpp>
#include <util/io.h>

#include <mpi2/mpi2.h>

#include <mf/matrix/distributed_matrix.h>

namespace mf {

namespace detail {
	struct Empty {
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
		}
	};

	template<typename T, void (*f)(const SparseMatrixCM&, const DenseMatrix&, DenseMatrixCM&, const T&,
			mf_size_type, mf_size_type),
			unsigned UNIQUE_ID>
	struct ApUpdateH {
		struct Arg {
			Arg() :vBlock(mpi2::UNINITIALIZED), hBlock(mpi2::UNINITIALIZED) { };
			Arg(mpi2::RemoteVar vBlock, const std::string& wUnblockedName,
					mpi2::RemoteVar hBlock, T data, mf_size_type b1, mf_size_type b2)
			: vBlock(vBlock), wUnblockedName(wUnblockedName), hBlock(hBlock), data(data), b1(b1), b2(b2) {
			}
			mpi2::RemoteVar vBlock;
			std::string wUnblockedName;
			mpi2::RemoteVar hBlock;
			T data;
			mf_size_type b1;
			mf_size_type b2;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				ar & vBlock;
				ar & wUnblockedName;
				ar & hBlock;
				ar & data;
				ar & b1;
				ar & b2;
			}
		};

		static inline Arg
		constructArg(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const std::string& wUnblockedName, const DistributedMatrix<DenseMatrixCM>& h,
				T data) {
			return Arg(block, wUnblockedName, h.block(b1, b2), data, b1, b2);
		}

		static const std::string id() { return rg::paste("__mf/matrix/op/ApUpdateH", UNIQUE_ID); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			std::vector<Arg> args;
			ch.recv(args);
			std::vector<boost::mpi::request> reqs(args.size());
			std::vector<double> result(args.size(), 0);
			for (unsigned i=0; i<args.size(); i++) {
				Arg& arg = args[i];
				const SparseMatrixCM& v = *arg.vBlock.template getLocal<SparseMatrixCM>();
				const DenseMatrix& w = *mpi2::env().get<DenseMatrix>(arg.wUnblockedName);
				DenseMatrixCM& h = *arg.hBlock.template getLocal<DenseMatrixCM>();
				f(v, w, h, arg.data, arg.b1, arg.b2);
				reqs[i] = ch.isend(result[i]); // result
			}
			boost::mpi::wait_all(reqs.begin(), reqs.end());
		}
	};

	template<typename T, void (*f)(const SparseMatrix&, DenseMatrix&, const DenseMatrixCM&, const T&,
			mf_size_type, mf_size_type),
			unsigned UNIQUE_ID>
	struct ApUpdateW {
		struct Arg {
			Arg() :vBlock(mpi2::UNINITIALIZED), wBlock(mpi2::UNINITIALIZED) { };

			Arg(mpi2::RemoteVar vBlock, mpi2::RemoteVar wBlock, const std::string& hUnblockedName,
					T data, mf_size_type b1, mf_size_type b2)
			: vBlock(vBlock), wBlock(wBlock), hUnblockedName(hUnblockedName), data(data), b1(b1), b2(b2) { }

			mpi2::RemoteVar vBlock;
			mpi2::RemoteVar wBlock;
			std::string hUnblockedName;
			T data;
			mf_size_type b1;
			mf_size_type b2;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				ar & vBlock;
				ar & wBlock;
				ar & hUnblockedName;
				ar & data;
				ar & b1;
				ar & b2;
			}
		};

		static inline Arg
		constructArg(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const DistributedMatrix<DenseMatrix>& w, const std::string& hUnblockedName,
				T data) {
			return Arg(block, w.block(b1, b2), hUnblockedName, data, b1, b2);
		}

		static const std::string id() { return rg::paste("__mf/matrix/op/ApUpdateW", UNIQUE_ID); }

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			std::vector<Arg> args;
			ch.recv(args);
			std::vector<boost::mpi::request> reqs(args.size());
			std::vector<double> result(args.size(), 0);
			for (unsigned i=0; i<args.size(); i++) {
				Arg& arg = args[i];
				const SparseMatrix& v = *arg.vBlock.template getLocal<SparseMatrix>();
				DenseMatrix& w = *arg.wBlock.template getLocal<DenseMatrix>();
				const DenseMatrixCM& h = *mpi2::env().get<DenseMatrixCM>(arg.hUnblockedName);
				f(v, w, h, arg.data, arg.b1, arg.b2);
				reqs[i] = ch.isend(result[i]); // result
			}
			boost::mpi::wait_all(reqs.begin(), reqs.end());
		}
	};
}

}

#endif
