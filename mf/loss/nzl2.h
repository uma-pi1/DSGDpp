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
#ifndef MF_LOSS_NZL2_H
#define MF_LOSS_NZL2_H

#include <mf/loss/loss.h>
#include <mf/matrix/distribute.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

/** Calculates nnz-weighted L2 loss for a row-factor matrix.
 * 	@param m input matrix
 * 	@param nnz nnz values (weights for each row)
 * 	@param offset starting offset in nnz vector
 */
inline double nzl2(const DenseMatrix& m, const std::vector<mf_size_type>& nnz, mf_size_type begin, mf_size_type end, mf_size_type nnzOffset = 0) {
	const DenseMatrix::array_type& values = m.data();
	mf_size_type p = begin*m.size2();
	double result;

	for (mf_size_type i=begin; i<end; i++) {
	//for (mf_size_type i=0; i<m.size1(); i++) {
		double v = 0;
		for (mf_size_type j=0; j<m.size2(); j++) {
			double vv = values[p];
			v += vv*vv;
			++p;
		}
		result += nnz[i + nnzOffset] * v;
	}
	return result;
}

/** Calculates nnz-weighted L2 loss for a row-factor matrix.
 * 	@param m input matrix
 * 	@param nnz nnz values (weights for each row)
 * 	@param offset starting offset in nnz vector
 */
inline double nzl2(const DenseMatrix& m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	return nzl2(m, nnz,0, m.size1(), nnzOffset);
}



/** Calculates nnz-weighted L2 loss for a column-factor matrix.
 * 	@param m input matrix
 * 	@param nnz nnz values (weights for each column)
 * 	@param offset starting offset in nnz vector
 */
inline double nzl2(const DenseMatrixCM& m, const std::vector<mf_size_type>& nnz, mf_size_type begin, mf_size_type end, mf_size_type nnzOffset = 0) {
	const DenseMatrixCM::array_type& values = m.data();

	mf_size_type p = begin*m.size1();
	double result;

	for (mf_size_type j=begin; j<end; j++) {
		double v = 0;
		for (mf_size_type i=0; i<m.size1(); i++) {
			double vv = values[p];
			v += vv*vv;
			++p;
		}
		result += nnz[j + nnzOffset] * v;
	}
	return result;
}
/** Calculates nnz-weighted L2 loss for a column-factor matrix.
 * 	@param m input matrix
 * 	@param nnz nnz values (weights for each column)
 * 	@param offset starting offset in nnz vector
 */
inline double nzl2(const DenseMatrixCM& m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	return nzl2(m, nnz, 0, m.size2(), nnzOffset);
}

// -- parallel ------------------------------------------------------------------------------------

namespace detail {
	template<typename M>
	struct ParallelNzl2Task {
		static const std::string id() { return std::string("__mf/loss/ParallelNzl2Task_")
				+ mpi2::TypeTraits<M>::name(); }
		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			// receive data
			mpi2::PointerIntType pM, pNnz, pSplit;
			mf_size_type nnzOffset;
			ch.recv(*mpi2::unmarshal(pM, pNnz, pSplit, nnzOffset));

			M& m = *mpi2::intToPointer<M>(pM);
			std::vector<mf_size_type>& split = *mpi2::intToPointer<std::vector<mf_size_type> >(pSplit);
			std::vector<mf_size_type>& nnz = *mpi2::intToPointer<std::vector<mf_size_type> >(pNnz);

			// compute loss and send back
			int p = info.groupId();

			typename M::value_type loss = nzl2(m, nnz, split[p], split[p+1],nnzOffset);
			ch.send(loss);
		}
	};
}

template<typename T, typename L, typename A>
inline T nzl2(const boost::numeric::ublas::matrix<T,L,A>& m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset, int tasks,bool isRowFactor=true) {
	BOOST_ASSERT( tasks > 0 );
	if (tasks == 1) {
		return nzl2(m,nnz,nnzOffset);
	} else {
		typedef detail::ParallelNzl2Task< boost::numeric::ublas::matrix<T,L,A> > Task;

		std::vector<mf_size_type> split;
		if(isRowFactor){
			split=mpi2::split(m.size1(), tasks);
		}else{
			split=mpi2::split(m.size2(), tasks);
		}

		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		std::vector<mpi2::Channel> channels;
		tm.spawn<Task>(tm.world().rank(), tasks, channels);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&m),mpi2::pointerToInt(&nnz), mpi2::pointerToInt(&split),nnzOffset));


		std::vector<T> losses;
		mpi2::economicRecvAll(channels, losses, tm.pollDelay());
		return std::accumulate(losses.begin(), losses.end(), (T)0.);

	}
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {

/**	A task for calculating the NZL2 loss (regularization part) */
struct Nzl2LossTask {
	/**
	 * 	The argument that is necessary for a task NzL2LossTask. Described in terms of:
	 * 	(1) the block of W or H on which the task will operate
	 * 	(2) the vector of sums of nnz entries the row or column for W or H respectively
	 * 	(3) the start and end indices defining the range of the nnz vector which correspond to the specific block
	 * 	(4) the information about if the input block belongs to W (row=true) or H (row=false)
	 */
	struct Arg {
	public:
		Arg() : data(mpi2::UNINITIALIZED) {};

		Arg(mpi2::RemoteVar block, const std::string& nnzName, mf_size_type nnzOffset, bool isRowFactor, int threads=1)
		: data(block), nnzName(nnzName), nnzOffset(nnzOffset), isRowFactor(isRowFactor), threads(threads) {}

		static Arg constructArgW(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName, int threads=1) {
			return Arg(block, nnzName, m.blockOffset1(b1), true, threads);
		}

		static Arg constructArgH(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
				const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName, int threads=1) {
			return Arg(block, nnzName, m.blockOffset2(b2), false, threads);
		}

		mpi2::RemoteVar data;
		std::string nnzName;
		mf_size_type nnzOffset;
		bool isRowFactor;
		int threads;

	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version) {
			ar & data;
			ar & nnzName;
			ar & nnzOffset;
			ar & isRowFactor;
			ar & threads;
		}
	};

	static const std::string id() {	return std::string("__mf/matrix/op/Nzl2LossTask") ;}

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		std::vector<Arg> args;
		ch.recv(args);
		std::vector<boost::mpi::request> reqs(args.size());
		std::vector<double> results(args.size());

		for (unsigned i=0; i<args.size(); i++) {
			Arg& arg = args[i];
			const std::vector<mf_size_type>& nnz = *mpi2::env().get<std::vector<mf_size_type> >(arg.nnzName);
			if (arg.isRowFactor){
				const DenseMatrix& m = *arg.data.getLocal<DenseMatrix>();
				results[i] = nzl2(m, nnz, arg.nnzOffset, arg.threads,arg.isRowFactor);
			}
			else{
				const DenseMatrixCM& m = *arg.data.getLocal<DenseMatrixCM>();
				results[i] = nzl2(m, nnz, arg.nnzOffset, arg.threads,arg.isRowFactor);
			}
			reqs[i] = ch.isend(results[i]);
		}
		boost::mpi::wait_all(reqs.begin(), reqs.end());
	}
};

} // namespace detail



inline double nzl2(const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName, int tasksPerRank, int threadsPerTask=1) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<DenseMatrix, double, detail::Nzl2LossTask::Arg>(
			m,
			result,
			boost::bind(detail::Nzl2LossTask::Arg::constructArgW, _1, _2, _3, boost::cref(m), boost::cref(nnzName), threadsPerTask),
			detail::Nzl2LossTask::id(),
			tasksPerRank,
			false);
	return sum(result);
}

inline double nzl2(const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName, int tasksPerRank, int threadsPerTask=1) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<DenseMatrixCM, double, detail::Nzl2LossTask::Arg>(
			m,
			result,
			boost::bind(detail::Nzl2LossTask::Arg::constructArgH, _1, _2, _3, boost::cref(m), boost::cref(nnzName), threadsPerTask),
			detail::Nzl2LossTask::id(),
			tasksPerRank,
			false);
	return sum(result);
}

// -- Loss ----------------------------------------------------------------------------------------

struct Nzl2Loss : public LossConcept, DistributedLossConcept {
	Nzl2Loss(mpi2::SerializationConstructor _) : lambda(FP_NAN) {
	}

	Nzl2Loss(double lambda) : lambda(lambda) { };

	double operator()(const FactorizationData<>& data) {
		if(lambda==0) return 0;
		return lambda*(
				nzl2(data.w, *data.nnz1, data.nnz1offset,data.tasks,true)
				+ nzl2(data.h, *data.nnz2, data.nnz2offset,data.tasks,false)
				);
	}

	double operator()(const DsgdFactorizationData<>& data) {
		return lambda*(
				nzl2(data.dw, data.nnz1name, data.tasksPerRank)
				+ nzl2(data.dh, data.nnz2name, data.tasksPerRank)
		);
	}

	double operator()(const DsgdPpFactorizationData<>& data) {
		return lambda*(
				nzl2(data.dw, data.nnz1name, data.tasksPerRank)
				+ nzl2(data.dh, data.nnz2name, data.tasksPerRank)
		);
	}

	double operator()(const AsgdFactorizationData<>& data) {
		return lambda*(
				nzl2(data.dw, data.nnz1name, 1, data.tasksPerRank)
				+ nzl2(data.dh, data.nnz2name, 1, data.tasksPerRank)
		);
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

MPI2_SERIALIZATION_CONSTRUCTOR(mf::Nzl2Loss);



#endif
