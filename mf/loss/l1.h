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
#ifndef MF_LOSS_L1_H
#define MF_LOSS_L1_H

#include <cmath>
#include <numeric>

#include <mf/loss/loss.h>
#include <mf/id.h>
#include <mf/matrix/distribute.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

namespace detail {
	template <typename T>
	struct l1Op : std::binary_function<T, T, T> {
		inline T operator()(const T& state, const T& x) const {
			return state + std::fabs(x);
		}
	};
}

template<class T, class L, class A>
inline T l1(const boost::numeric::ublas::matrix<T,L,A>& m) {
	return std::accumulate(m.data().begin(), m.data().end(), (T)0, detail::l1Op<T>());
}

template<class T, class L, std::size_t IB, class IA, class TA>
inline T l1(const boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>& m) {
	const typename boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>::value_array_type&
		values = m.value_data();
	return std::accumulate(values.begin(), values.begin()+m.nnz(), (T)0, detail::l1Op<T>());
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	template<typename M>
	typename M::value_type L1TaskF(M& m) { return l1(m); }

	template<typename M>
	struct L1Task : public PerBlockTaskReturn<M, typename M::value_type, L1TaskF<M>, ID_L1> {
		typedef PerBlockTaskReturn<M, typename M::value_type, L1TaskF<M>, ID_L1> Task;
	};
}

template<typename M>
inline void l1(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<typename M::value_type>& result, int tasksPerRank=1) {
	runTaskOnBlocks< detail::L1Task<M> >(m, result, tasksPerRank);
}

template<typename M>
inline typename M::value_type l1(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	boost::numeric::ublas::matrix<typename M::value_type> ss(m.blocks1(), m.blocks2());
	l1(m, ss, tasksPerRank);
	return sum(ss);
}

// -- Loss ----------------------------------------------------------------------------------------

struct L1Loss : public LossConcept, DistributedLossConcept {
	L1Loss(mpi2::SerializationConstructor _) : lambdaW(FP_NAN), lambdaH(FP_NAN) {
	}

	L1Loss(double lambda) : lambdaW(lambda), lambdaH(lambda) { };

	L1Loss(double lambdaW, double lambdaH) : lambdaW(lambdaW), lambdaH(lambdaH) { };

	double operator()(const FactorizationData<>& data) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of L1Loss not yet implemented, using sequential computation.");
		}

		double result = 0.;
		if (lambdaW != 0.) result += lambdaW*l1(data.w);
		if (lambdaH != 0.) result += lambdaH*l1(data.h);
		return result;
	}

	double operator()(const DsgdFactorizationData<>& data) {
		double result = 0.;
		if (lambdaW != 0.) result += l1(data.dw, data.tasksPerRank);
		if (lambdaH != 0.) result += l1(data.dh, data.tasksPerRank);
		return result;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & lambdaW;
		ar & lambdaH;
	}

	double lambdaW;
	double lambdaH;
};

}

MPI2_SERIALIZATION_CONSTRUCTOR(mf::L1Loss);


#endif
