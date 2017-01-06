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
#ifndef MF_LOSS_L2_H
#define MF_LOSS_L2_H

#include <numeric>

#include <mf/loss/loss.h>
#include <mf/id.h>
#include <mf/matrix/distribute.h>

namespace mf {

// TODO: This does not actually compute the l2-norm but the sqaure of the l2-norm. This should
// somehow be indicated in the function names

// -- sequential ----------------------------------------------------------------------------------

namespace detail {
	template <typename T>
	struct l2Op : std::binary_function<T, T, T> {
		inline T operator()(const T& state, const T& x) const {
			return state + x*x;
		}
	};
} // detail

template<typename T, typename L, typename A>
inline T l2(const boost::numeric::ublas::matrix<T,L,A>& m,  mf_size_type begin, mf_size_type end) {
	BOOST_ASSERT( begin >= 0 && begin < end && end <= m.data().size() );
	return std::accumulate(m.data().begin() + begin, m.data().begin() + end, (T)0, detail::l2Op<T>());
}

template<typename T, typename L, typename A>
inline T l2(const boost::numeric::ublas::matrix<T,L,A>& m) {
	return l2(m, 0, m.data().size());
}

template<class T, class L, std::size_t IB, class IA, class TA>
inline T l2(const boost::numeric::ublas::coordinate_matrix<T,L,IB,IA,TA>& m,
		mf_size_type begin, mf_size_type end) {
	BOOST_ASSERT( begin >= 0 && begin < end && end <= m.nnz() );
	return std::accumulate(m.value_data().begin() + begin, m.value_data().begin() + end,
			(T)0, detail::l2Op<T>());
}

template<class T, class L, std::size_t IB, class IA, class TA>
inline T l2(const boost::numeric::ublas::coordinate_matrix<T,L,IB,IA,TA>& m) {
	return l2(m, 0, m.nnz());
}


// -- parallel ------------------------------------------------------------------------------------

namespace detail {
	template<typename M>
	struct ParallelL2Task {
		static const std::string id() { return std::string("__mf/loss/ParallelL2Task_")
				+ mpi2::TypeTraits<M>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			// receive data
			mpi2::PointerIntType pM, pSplit;
			ch.recv(*mpi2::unmarshal(pM, pSplit));
			M& m = *mpi2::intToPointer<M>(pM);
			std::vector<mf_size_type>& split = *mpi2::intToPointer<std::vector<mf_size_type> >(pSplit);

			// compute loss and send back
			int p = info.groupId();
			typename M::value_type loss = l2(m, split[p], split[p+1]);
			ch.send(loss);
		}
	};
}

template<typename T, typename L, typename A>
inline T l2(const boost::numeric::ublas::matrix<T,L,A>& m, int tasks) {
	BOOST_ASSERT( tasks > 0 );
	if (tasks == 1) {
		return l2(m);
	} else {
		typedef detail::ParallelL2Task< boost::numeric::ublas::matrix<T,L,A> > Task;
		std::vector<mf_size_type> split = mpi2::split((mf_size_type)m.data().size(), tasks);
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		std::vector<mpi2::Channel> channels;
		tm.spawn<Task>(tm.world().rank(), tasks, channels);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&m), mpi2::pointerToInt(&split)));
		std::vector<T> losses;
		mpi2::economicRecvAll(channels, losses, tm.pollDelay());
		return std::accumulate(losses.begin(), losses.end(), (T)0.);
	}
}

template<class T, class L, std::size_t IB, class IA, class TA>
inline T l2(const boost::numeric::ublas::coordinate_matrix<T,L,IB,IA,TA>& m, int tasks) {
	BOOST_ASSERT( tasks > 0 );
	if (tasks == 1) {
		return l2(m);
	} else {
		typedef detail::ParallelL2Task< boost::numeric::ublas::coordinate_matrix<T,L,IB,IA,TA> > Task;
		std::vector<mf_size_type> split = mpi2::split((mf_size_type)m.nnz(), tasks);
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		std::vector<mpi2::Channel> channels;
		tm.spawn<Task>(tm.world().rank(), tasks, channels);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&m), mpi2::pointerToInt(&split)));
		std::vector<T> losses;
		mpi2::economicRecvAll(channels, losses, tm.pollDelay());
		return std::accumulate(losses.begin(), losses.end(), (T)0.);
	}
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	template<typename M>
	typename M::value_type L2TaskF(M& m, int threadsPerTask) { return l2(m, threadsPerTask); }

	template<typename M>
	struct L2Task : public PerBlockTaskReturnArg<M, typename M::value_type, int, L2TaskF<M>, ID_L2> {
		typedef PerBlockTaskReturnArg<M, typename M::value_type, int, L2TaskF<M>, ID_L2> Task;
	};
}

template<typename M>
inline void l2(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<typename M::value_type>& result, int tasksPerRank=1, int threadsPerTask=1) {
	runTaskOnBlocks< detail::L2Task<M> >(m, threadsPerTask, result, tasksPerRank);
}

template<typename M>
inline typename M::value_type l2(const DistributedMatrix<M>& m, int tasksPerRank=1, int threadsPerTask=1) {
	boost::numeric::ublas::matrix<typename M::value_type> ss(m.blocks1(), m.blocks2());
	l2(m, ss, tasksPerRank, threadsPerTask);
	return sum(ss);
}


// -- Loss ----------------------------------------------------------------------------------------

struct L2Loss : public LossConcept, DistributedLossConcept {
	L2Loss(mpi2::SerializationConstructor _) : lambda(FP_NAN) {
	}

	L2Loss(double lambda) : lambda(lambda) { };

	double operator()(const FactorizationData<>& data) {
		if(lambda==0) return 0;
		return lambda*(l2(data.w, data.tasks) + l2(data.h, data.tasks));
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return lambda*(l2(data.dw, data.tasksPerRank) + l2(data.dh, data.tasksPerRank));
	}

	double operator()(const DsgdPpFactorizationData<>& data) {
		return lambda*(l2(data.dw, data.tasksPerRank) + l2(data.dh, data.tasksPerRank));
	}

	double operator()(const AsgdFactorizationData<>& data) {
		return lambda*(l2(data.dw, 1, data.tasksPerRank) + l2(data.dh, 1, data.tasksPerRank));
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & lambda;
	}

	double lambda;
};

}

MPI2_SERIALIZATION_CONSTRUCTOR(mf::L2Loss);


#endif
