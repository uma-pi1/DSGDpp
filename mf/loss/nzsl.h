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
#ifndef MF_LOSS_NZSL_H
#define MF_LOSS_NZSL_H

#include <mf/matrix/coordinate.h>
#include <mf/ap/aptask.h>
#include <mf/loss/loss.h>
#include <mf/id.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/op/sum.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

/** Only process entries [begin, end). */
template<typename W, typename H>
inline double nzsl(const SparseMatrix& v,
		const W& w, const H& h, mf_size_type begin, mf_size_type end) {
	typedef SparseMatrix M;
	typename M::value_type result = 0;

	const typename M::index_array_type& index1 = rowIndexData(v);
	const typename M::index_array_type& index2 = columnIndexData(v);
	const typename M::value_array_type& values = v.value_data();
	mf_size_type rank = w.size2();
	for (mf_size_type i=begin; i<end; i++) {
		typename M::value_type ip = 0;
		typename M::size_type i1 = index1[i];
		typename M::size_type i2 = index2[i];
		typename M::value_type value = values[i];
		for (mf_size_type r=0; r<rank; r++) {
			ip += w(i1,r) * h(r,i2);
		}
		typename M::value_type diff = value - ip;
		result += diff*diff;
	}

	return result;
}

template<typename W, typename H>
inline double nzsl(const SparseMatrix& v,
		const W& w, const H& h) {
	return nzsl(v, w, h, 0, v.nnz());
}

// -- parallel ------------------------------------------------------------------------------------

namespace detail {
	template<typename M1, typename M2, typename M3>
	struct ParallelNzslTask {
		static const std::string id() { return std::string("__mf/loss/ParallelNzslTask_")
				+ mpi2::TypeTraits<M1>::name() + "_" + mpi2::TypeTraits<M2>::name() + "_" + mpi2::TypeTraits<M3>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			// receive data
			mpi2::PointerIntType pV, pW, pH, pSplit;
			ch.recv(*mpi2::unmarshal(pV, pW, pH, pSplit));
			M1& v = *mpi2::intToPointer<M1>(pV);
			M2& w = *mpi2::intToPointer<M2>(pW);
			M3& h = *mpi2::intToPointer<M3>(pH);
			std::vector<mf_size_type>& split = *mpi2::intToPointer<std::vector<mf_size_type> >(pSplit);
			
			// compute loss and send back
			int p = info.groupId();
			double loss = nzsl(v, w, h, split[p], split[p+1]);
			ch.send(loss);
		}
	};
}

template<typename W, typename H>
inline double nzsl(const SparseMatrix& v,
		const W& w, const H& h, int tasks) {
	BOOST_ASSERT( tasks > 0 );
	if (tasks == 1) {
		return nzsl(v, w, h);
	} else {
		typedef detail::ParallelNzslTask<SparseMatrix, W, H> Task;
		std::vector<mf_size_type> split = mpi2::split((mf_size_type)v.nnz(), tasks);
		
// 		for(int i=0; i<split.size(); i++){
// 		  std::cout<<i<<" --> "<<split[i+1]-split[i]<<std::endl;
// 		}
		
		
		
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		std::vector<mpi2::Channel> channels;
		tm.spawn<Task>(tm.world().rank(), tasks, channels);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&v),
				mpi2::pointerToInt(&w), mpi2::pointerToInt(&h), mpi2::pointerToInt(&split)));
		std::vector<double> losses;
		mpi2::economicRecvAll(channels, losses, tm.pollDelay());
		return std::accumulate(losses.begin(), losses.end(), 0.);
	}
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	template<typename M1, typename M2, typename M3>
	struct NzslTask {
		static const std::string id() { return std::string("__mf/loss/NzslTask_")
				+ mpi2::TypeTraits<M1>::name() + "_" + mpi2::TypeTraits<M2>::name() + "_" + mpi2::TypeTraits<M3>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			runFunctionPerAssignedBlock3<M1,M2,M3, typename M1::value_type>(ch, &f);
		}
		static inline typename M1::value_type f(const M1& v, const M2& w, const M3& h) {
			return nzsl(v, w, h);
		}
	};
}

inline void nzsl(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		boost::numeric::ublas::matrix<SparseMatrix::value_type>& result, int tasksPerRank=1) {
	runTaskOnBlocks3(v, w, h, result,
			detail::NzslTask<SparseMatrix,DenseMatrix,DenseMatrixCM>::id(),
			tasksPerRank);
}

inline SparseMatrix::value_type nzsl(const DistributedSparseMatrix& v,
		const DistributedDenseMatrix& w, const DistributedDenseMatrixCM& h,
		int tasksPerRank=1) {
	boost::numeric::ublas::matrix<SparseMatrix::value_type> result(v.blocks1(), v.blocks2());
	nzsl(v, w, h, result, tasksPerRank);
	return sum(result);
}

namespace detail {
	typedef ApTaskWThreads<mf::nzsl<DenseMatrix,DenseMatrixCM>, ID_NZSL_AP> NzslApTaskWThreads;
}

inline double nzsl(const DistributedSparseMatrix& v, const DistributedDenseMatrix& w,
		const std::string& hUnblockedName, int tasksPerRank=1, int threadsPerTask=1) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<SparseMatrix,double,detail::NzslApTaskWThreads::Arg>(
						v, result,
						boost::bind(detail::NzslApTaskWThreads::constructArg, _1, _2, _3, boost::cref(w), boost::cref(hUnblockedName), threadsPerTask),
						detail::NzslApTaskWThreads::id(), tasksPerRank);
	double d = std::accumulate(result.data().begin(), result.data().end(), 0.);
	return d;
}

// -- Loss ----------------------------------------------------------------------------------------

struct NzslLoss : public LossConcept, DistributedLossConcept {
	NzslLoss() {};
	NzslLoss(mpi2::SerializationConstructor _) { };

	double operator()(const FactorizationData<>& data) {
		return nzsl(data.v, data.w, data.h, data.tasks);
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return nzsl(data.dv, data.dw, data.dh, data.tasksPerRank);
	}

	double operator()(const DsgdPpFactorizationData<>& data) {
		return nzsl(data.dv, data.dw, data.dh, data.tasksPerRank);
	}

	double operator()(const AsgdFactorizationData<>& data) {
		return nzsl(data.dv, data.dw, data.hWorkName, 1, data.tasksPerRank);
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
	}
};

}

MPI2_TYPE_TRAITS(mf::NzslLoss);

#endif
