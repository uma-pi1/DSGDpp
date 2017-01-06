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
#ifndef MF_MATRIX_OP_SUM_H
#define MF_MATRIX_OP_SUM_H

#include <numeric>

#include <mf/id.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

template<class T, class L, class A>
inline T sum(const boost::numeric::ublas::matrix<T,L,A>& m) {
	return std::accumulate(m.data().begin(), m.data().end(), (T)0);
}

template<class T, class L, std::size_t IB, class IA, class TA>
inline T sum(const boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>& m) {
	const typename boost::numeric::ublas::coordinate_matrix<T, L, IB, IA, TA>::value_array_type&
		values = m.value_data();
	return std::accumulate(values.begin(), values.begin()+m.nnz(), (T)0);
}


// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	template<typename M>
	typename M::value_type SumTaskF(M& m) {
		return sum(m);
	}

	template<typename M>
	struct SumTask : public PerBlockTaskReturn<M, typename M::value_type, SumTaskF<M>, ID_SUM> {
		typedef PerBlockTaskReturn<M, typename M::value_type, SumTaskF<M>, ID_SUM> Task;
	};
}

template<typename M>
inline void sum(const DistributedMatrix<M>& m,
		boost::numeric::ublas::matrix<typename M::value_type>& sums, int tasksPerRank=1) {
	runTaskOnBlocks< detail::SumTask<M> >(m, sums, tasksPerRank);
}

template<typename M>
inline typename M::value_type sum(const DistributedMatrix<M>& m, int tasksPerRank=1) {
	boost::numeric::ublas::matrix<typename M::value_type> sums(m.blocks1(), m.blocks2());
	sum(m, sums, tasksPerRank);
	return sum(sums);
}

} // namespace mf



#endif
