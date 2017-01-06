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
#ifndef MF_MATRIX_OP_SCALE_H
#define MF_MATRIX_OP_SCALE_H

#include <mf/id.h>
#include <mf/matrix/distributed_matrix.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/op/sum.h>

namespace mf {

template<typename M>
inline void mult(M& m, typename M::value_type value) {
	m *= value;
}

template<typename M>
inline void div(M& m, typename M::value_type value) {
	m /= value;
}

namespace detail {
	template<typename M>
	struct MultTask : public PerBlockTaskVoidArg<M, typename M::value_type, mf::mult<M>, ID_MULT> {
		typedef PerBlockTaskVoidArg<M, typename M::value_type, mf::mult<M>, ID_MULT> Task;
	};

	template<typename M>
	struct DivTask : public PerBlockTaskVoidArg<M, typename M::value_type, mf::div<M>, ID_DIV> {
		typedef PerBlockTaskVoidArg<M, typename M::value_type, mf::div<M>, ID_DIV> Task;
	};
}

template<typename M>
inline void mult(DistributedMatrix<M>& m, typename M::value_type value, int tasksPerRank = 1) {
	runTaskOnBlocks<detail::MultTask<M> >(m, value, tasksPerRank);
}

template<typename M>
inline void div(DistributedMatrix<M>& m, typename M::value_type value, int tasksPerRank = 1) {
	runTaskOnBlocks<detail::DivTask<M> >(m, value, tasksPerRank);
}


template<typename T, typename L, typename A, typename V>
inline void mult1(boost::numeric::ublas::matrix<T,L,A>& m, const V& values1) {
	for (mf_size_type j=0; j<m.size2(); j++) { // optimized for CM
		for (mf_size_type i=0; i<m.size1(); i++) {
			m(i,j) *= values1[i];
		}
	}
}

template<typename M, typename V>
inline void div1(M& m, const V& values1) {
	for (mf_size_type j=0; j<m.size2(); j++) { // optimized for CM
		for (mf_size_type i=0; i<m.size1(); i++) {
			m(i,j) /= values1[i];
		}
	}
}

template<typename M, typename V>
inline void mult2(M& m, const V& values2) {
	for (mf_size_type i=0; i<m.size1(); i++) { // optimized for RM
		for (mf_size_type j=0; j<m.size2(); j++) {
			m(i,j) *= values2[j];
		}
	}
}

template<typename T, typename L, typename A, typename V>
inline void div2(boost::numeric::ublas::matrix<T,L,A>& m, const V& values2) {
	for (mf_size_type i=0; i<m.size1(); i++) { // optimized for RM
		for (mf_size_type j=0; j<m.size2(); j++) {
			m(i,j) /= values2[j];
		}
	}
}

namespace detail {
	template<typename M, typename V>
	void Mult1TaskF(M& m, V values) {
		mult1(m, values);
	}

	template<typename M, typename V>
	struct Mult1Task : public PerBlockTaskVoidArg<M, V, Mult1TaskF<M, V>, ID_MULT1> {
		typedef PerBlockTaskVoidArg<M, V, Mult1TaskF<M, V>, ID_MULT1> Task;
	};

	template<typename M, typename V>
	void Mult2TaskF(M& m, V values) {
		mult2(m, values);
	}

	template<typename M, typename V>
	struct Mult2Task : public PerBlockTaskVoidArg<M, V, Mult2TaskF<M, V>, ID_MULT2> {
		typedef PerBlockTaskVoidArg<M, V, Mult2TaskF<M, V>, ID_MULT2> Task;
	};

	template<typename M, typename V>
	void Div1TaskF(M& m, V values) {
		div1(m, values);
	}

	template<typename M, typename V>
	struct Div1Task : public PerBlockTaskVoidArg<M, V, Div1TaskF<M, V>, ID_DIV1> {
		typedef PerBlockTaskVoidArg<M, V, Div1TaskF<M, V>, ID_DIV1> Task;
	};
}

template<typename M, typename V>
inline void mult1(DistributedMatrix<M>& m, const V& values, int tasksPerRank = 1) {
	BOOST_ASSERT( m.blocks1() == 1);
	runTaskOnBlocks<detail::Mult1Task<M,V> >(m, values, tasksPerRank);
}

template<typename M, typename V>
inline void mult2(DistributedMatrix<M>& m, const V& values, int tasksPerRank = 1) {
	BOOST_ASSERT( m.blocks2() == 1);
	runTaskOnBlocks<detail::Mult2Task<M,V> >(m, values, tasksPerRank);
}

template<typename M, typename V>
inline void div1(DistributedMatrix<M>& m, const V& values, int tasksPerRank = 1) {
	BOOST_ASSERT( m.blocks2() == 1);
	runTaskOnBlocks<detail::Div1Task<M,V> >(m, values, tasksPerRank);
}


} // namespace mf



#endif
