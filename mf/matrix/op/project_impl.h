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
 * Implementation for matrix/project.h
 * DO NOT INCLUDE DIRECTLY
 */

#include <mf/matrix/op/project.h> // help for compilers

namespace mf {

template<typename M>
void project1(const M& m, M& result, std::vector<mf_size_type> indexes1) {
	result.resize(indexes1.size(), m.size2(), false);
	for (mf_size_type i=0; i<indexes1.size(); i++) {
		for (mf_size_type j=0; j<m.size2(); j++) {
			result(i,j) = m(indexes1[i], j);
		}
	}
}

template<typename M>
void project2(const M& m, M& result, std::vector<mf_size_type> indexes2) {
	result.resize(m.size1(), indexes2.size(), false);
	for (mf_size_type j=0; j<indexes2.size(); j++) {
		for (mf_size_type i=0; i<m.size1(); i++) {
			result(i,j) = m(i, indexes2[j]);
		}
	}
}

namespace detail {
	struct ProjectTaskArg {
		ProjectTaskArg(mpi2::SerializationConstructor _) : block(mpi2::UNINITIALIZED) { };
		ProjectTaskArg(mpi2::RemoteVar block, std::vector<mf_size_type> blockIndexes, bool projectRows)
		: block(block), blockIndexes(blockIndexes), projectRows(projectRows)
		{ }
		mpi2::RemoteVar block;
		std::vector<mf_size_type> blockIndexes;
		bool projectRows;

	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & block;
			ar & blockIndexes;
			ar & projectRows;
		}
	};

	inline ProjectTaskArg argProjectTask(
			mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
			const std::vector<std::vector<mf_size_type> >& blockIndexes,
			bool projectRows) {
		mf_size_type index = projectRows ? b1 : b2;
		return ProjectTaskArg(block, blockIndexes[index], projectRows);
	}
}
}

MPI2_SERIALIZATION_CONSTRUCTOR(mf::detail::ProjectTaskArg);

namespace mf { namespace detail {

	template<typename M>
	struct ProjectTask {
		static const std::string id() { return std::string("__mf/matrix/ProjectTask_") + mpi2::TypeTraits<M>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			std::vector<ProjectTaskArg> args;
			ch.recv(args);
			M results[args.size()];
			std::vector<boost::mpi::request> reqs(args.size());
			for (unsigned k=0; k<args.size(); k++) {
				if (args[k].projectRows) {
					project1(*args[k].block.getLocal<M>(), results[k], args[k].blockIndexes);
				} else {
					project2(*args[k].block.getLocal<M>(), results[k], args[k].blockIndexes);
				}
				reqs[k] = ch.isend(results[k]);
			}
			boost::mpi::wait_all(reqs.begin(), reqs.end());
		}
	};
}

template<typename M>
void project1(const DistributedMatrix<M>& m, M& result,
		std::vector<mf_size_type> indexes1, int tasksPerRank) {
	if (m.blocks2() != 1){
		RG_THROW(rg::InvalidArgumentException, "blocks2() must be equal to 1");
	}

	// split the indexes
	std::vector<std::vector<mf_size_type> > blockRows;
	splitIndexes(indexes1, m.blockOffsets1(), blockRows);
	using rg::operator<<;

	// run the task
	boost::numeric::ublas::matrix<M> blockResult;
	runTaskOnBlocks<M, M, detail::ProjectTaskArg>(
			m, blockResult,
			boost::bind(detail::argProjectTask, _1, _2, _3, boost::cref(blockRows), true),
			detail::ProjectTask<M>::id(),
			tasksPerRank);

	// construct result
	result.resize(indexes1.size(), m.size2(), false);
	mf_size_type start1 = 0;
	for (mf_size_type b1=0; b1<m.blocks1(); b1++) {
		M& mSample = blockResult(b1,0);
		mf_size_type stop1 = start1 + mSample.size1();
		boost::numeric::ublas::subrange(result, start1,stop1, 0,m.size2()) = mSample;
		start1 = stop1;
	}
}

template<typename M>
void project2(const DistributedMatrix<M>& m, M& result,
		std::vector<mf_size_type> indexes2, int tasksPerRank) {
	using namespace boost::numeric::ublas;

	if (m.blocks1() != 1){
		RG_THROW(rg::InvalidArgumentException, "blocks1() must be equal to 1");
	}

	// split the indexes
	std::vector<std::vector<mf_size_type> > blockCols;
	splitIndexes(indexes2, m.blockOffsets2(), blockCols);

	// run the task
	boost::numeric::ublas::matrix<M> blockResult;
	runTaskOnBlocks<M, M, detail::ProjectTaskArg>(
			m, blockResult,
			boost::bind(detail::argProjectTask, _1, _2, _3, boost::cref(blockCols), false),
			detail::ProjectTask<M>::id(),
			tasksPerRank);

	// construct result
	result.resize(m.size1(), indexes2.size(), false);
	mf_size_type start2 = 0;
	for (mf_size_type b2=0; b2<m.blocks2(); b2++) {
		M& mSample = blockResult(0,b2);
		mf_size_type stop2 = start2 + mSample.size2();
		noalias(subrange(result, 0,m.size1(), start2,stop2)) = mSample;
		start2 = stop2;
	}
}

template<typename Min, typename Mout>
void projectSubrange(const Min& source, Mout& target,
		mf_size_type start1, mf_size_type stop1, mf_size_type start2, mf_size_type stop2) {
	target = boost::numeric::ublas::subrange(source, start1, stop1, start2, stop2);
}

/** More efficient specialization of mf::projectSubrange for sparse matrices. */
template<>
inline void projectSubrange<SparseMatrix,SparseMatrix>(const SparseMatrix& source, SparseMatrix& target,
		mf_size_type start1, mf_size_type stop1, mf_size_type start2, mf_size_type stop2) {
    typedef SparseMatrix::const_iterator1 i1_t;
    typedef SparseMatrix::const_iterator2 i2_t;

	target.resize(stop1-start1, stop2-start2, false);
	target.clear();
	i1_t i1end = source.find1(0, stop1, 0);
    for (i1_t i1 = source.find1(0, start1, 0); i1 != i1end; ++i1) { // rows
    	for (i2_t i2 = i1.begin(); i2 != i1.end(); ++i2) { // columns
    		if (i2.index2() < start2) continue;
    		if (i2.index2() >= stop2) break;
    		target.append_element(i2.index1()-start1, i2.index2()-start2, *i2);
    	}
    }
    target.sort(); // actually unnecessary but don't know how to avoid it
}

/** More efficient specialization of mf::projectSubrange for sparse matrices (column-major). */
template<>
inline void projectSubrange<SparseMatrixCM,SparseMatrixCM>(const SparseMatrixCM& source, SparseMatrixCM& target,
		mf_size_type start1, mf_size_type stop1, mf_size_type start2, mf_size_type stop2) {
	typedef SparseMatrixCM::const_iterator2 i2_t;
	typedef SparseMatrixCM::const_iterator1 i1_t;

	target.resize(stop1-start1, stop2-start2, false);
	target.clear();
	i2_t i2end = source.find2(0, 0, stop2);
    for (i2_t i2 = source.find2(0, 0, start2); i2 != i2end; ++i2) { // columns
    	for (i1_t i1 = i2.begin(); i1 != i2.end(); ++i1) { // rows
    		if (i1.index1() < start1) continue;
    		if (i1.index1() >= stop1) break;
    		target.append_element(i1.index1()-start1, i2.index2()-start2, *i1);
    	}
    }
    target.sort(); // actually unnecessary but don't know how to avoid it
}

}
