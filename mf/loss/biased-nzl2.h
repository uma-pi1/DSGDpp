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
#ifndef MF_LOSS_BIASED_NZL2_H
#define MF_LOSS_BIASED_NZL2_H

#include <mf/loss/loss.h>
#include <mf/matrix/distribute.h>
#include <mf/matrix/op/sum.h>

namespace mf {

// -- sequential ----------------------------------------------------------------------------------

inline double biasedNzl2Factors(const DenseMatrix& m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	const DenseMatrix::array_type& values = m.data();

	mf_size_type p = 0;
	double result;
	for (mf_size_type i=0; i<m.size1(); i++) {
		double v = 0;
		p++; // skip bias
		for (mf_size_type j=1; j<m.size2(); j++) {
			double vv = values[p];
			v += vv*vv;
			++p;
		}
		result += nnz[i + nnzOffset] * v;
	}
	return result;
}

inline double biasedNzl2Factors(const DenseMatrixCM& m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	const DenseMatrixCM::array_type& values = m.data();

	mf_size_type p = 0;
	double result;
	for (mf_size_type j=0; j<m.size2(); j++) {
		double v = 0;
		p++; // skip bias
		for (mf_size_type i=1; i<m.size1(); i++) {
			double vv = values[p];
			v += vv*vv;
			++p;
		}
		result += nnz[j + nnzOffset] * v;
	}
	return result;
}

inline double biasedNzl2Bias(const DenseMatrix& m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	boost::numeric::ublas::matrix_column<const DenseMatrix> col(m, 0);
	double result;
	for (mf_size_type i=0; i<m.size1(); i++) {
		double v = 0;
		v = col[i] * col[i];
		result += nnz[i + nnzOffset] * v;
	}
	return result;
}

inline double biasedNzl2Bias(const DenseMatrixCM& m, const std::vector<mf_size_type>& nnz, mf_size_type nnzOffset = 0) {
	boost::numeric::ublas::matrix_row<const DenseMatrixCM> row(m, 0);
	double result;
	for (mf_size_type j=0; j<m.size2(); j++) {
		double v = 0;
		v = row[j] * row[j];
		result += nnz[j + nnzOffset] * v;
	}
	return result;
}

// -- distributed ---------------------------------------------------------------------------------

namespace detail {
	struct BiasedNzl2FactorsTask {
		struct Arg {
		public:
			Arg() : data(mpi2::UNINITIALIZED) {};

			Arg(mpi2::RemoteVar block, const std::string& nnzName, mf_size_type nnzOffset, bool isRowFactor)
			: data(block), nnzName(nnzName), nnzOffset(nnzOffset), isRowFactor(isRowFactor){}

			static Arg constructArgW(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
					const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName) {
				return Arg(block, nnzName, m.blockOffset1(b1), true);
			}

			static Arg constructArgH(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
					const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName) {
				return Arg(block, nnzName, m.blockOffset2(b2), false);
			}

			mpi2::RemoteVar data;
			std::string nnzName;
			mf_size_type nnzOffset;
			bool isRowFactor;

		private:
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				ar & data;
				ar & nnzName;
				ar & nnzOffset;
				ar & isRowFactor;
			}
		};

		static const std::string id() {	return std::string("__mf/loss/BiasedNzl2FactorsTask") ;}
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
					results[i] = biasedNzl2Factors(m, nnz, arg.nnzOffset);
				}
				else{
					const DenseMatrixCM& m = *arg.data.getLocal<DenseMatrixCM>();
					results[i] = biasedNzl2Factors(m, nnz, arg.nnzOffset);
				}
				reqs[i] = ch.isend(results[i]);
			}
			boost::mpi::wait_all(reqs.begin(), reqs.end());
		}
	};

	struct BiasedNzl2BiasTask {

		struct Arg {
		public:
			Arg() : data(mpi2::UNINITIALIZED) {};

			Arg(mpi2::RemoteVar block, const std::string& nnzName, mf_size_type nnzOffset, bool isRowFactor)
			: data(block), nnzName(nnzName), nnzOffset(nnzOffset), isRowFactor(isRowFactor){}

			static Arg constructArgW(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
					const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName) {
				return Arg(block, nnzName, m.blockOffset1(b1), true);
			}

			static Arg constructArgH(mf_size_type b1, mf_size_type b2, mpi2::RemoteVar block,
					const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName) {
				return Arg(block, nnzName, m.blockOffset2(b2), false);
			}

			mpi2::RemoteVar data;
			std::string nnzName;
			mf_size_type nnzOffset;
			bool isRowFactor;

		private:
			friend class boost::serialization::access;
			template<class Archive>
			void serialize(Archive & ar, const unsigned int version) {
				ar & data;
				ar & nnzName;
				ar & nnzOffset;
				ar & isRowFactor;
			}
		};

		static const std::string id() {	return std::string("__mf/loss/BiasedNzl2BiasTask") ;}
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
					results[i] = biasedNzl2Bias(m, nnz, arg.nnzOffset);
				}
				else{
					const DenseMatrixCM& m = *arg.data.getLocal<DenseMatrixCM>();
					results[i] = biasedNzl2Bias(m, nnz, arg.nnzOffset);
				}
				reqs[i] = ch.isend(results[i]);
			}
			boost::mpi::wait_all(reqs.begin(), reqs.end());
		}
	};
} // namespace detail

inline double biasedNzl2Factors(const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName, int tasksPerRank) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<DenseMatrix, double, detail::BiasedNzl2FactorsTask::Arg>(
			m,
			result,
			boost::bind(detail::BiasedNzl2FactorsTask::Arg::constructArgW, _1, _2, _3, boost::cref(m), boost::cref(nnzName)),
			detail::BiasedNzl2FactorsTask::id(),
			tasksPerRank,
			false);
	return sum(result);
}

inline double biasedNzl2Factors(const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName, int tasksPerRank) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<DenseMatrixCM, double, detail::BiasedNzl2FactorsTask::Arg>(
			m,
			result,
			boost::bind(detail::BiasedNzl2FactorsTask::Arg::constructArgH, _1, _2, _3, boost::cref(m), boost::cref(nnzName)),
			detail::BiasedNzl2FactorsTask::id(),
			tasksPerRank,
			false);
	return sum(result);
}

inline double biasedNzl2Bias(const DistributedMatrix<DenseMatrix>& m, const std::string& nnzName, int tasksPerRank) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<DenseMatrix, double, detail::BiasedNzl2BiasTask::Arg>(
			m,
			result,
			boost::bind(detail::BiasedNzl2BiasTask::Arg::constructArgW, _1, _2, _3, boost::cref(m), boost::cref(nnzName)),
			detail::BiasedNzl2BiasTask::id(),
			tasksPerRank,
			false);
	return sum(result);
}

inline double biasedNzl2Bias(const DistributedMatrix<DenseMatrixCM>& m, const std::string& nnzName, int tasksPerRank) {
	boost::numeric::ublas::matrix<double> result;
	runTaskOnBlocks<DenseMatrixCM, double, detail::BiasedNzl2BiasTask::Arg>(
			m,
			result,
			boost::bind(detail::BiasedNzl2BiasTask::Arg::constructArgH, _1, _2, _3, boost::cref(m), boost::cref(nnzName)),
			detail::BiasedNzl2BiasTask::id(),
			tasksPerRank,
			false);
	return sum(result);
}

// -- Loss ----------------------------------------------------------------------------------------

struct BiasedNzl2Loss : public LossConcept, DistributedLossConcept {
	BiasedNzl2Loss(mpi2::SerializationConstructor _) : lambdaW(FP_NAN), lambdaH(FP_NAN), lambdaRow(FP_NAN), lambdaCol(FP_NAN) {
	}

	BiasedNzl2Loss(double lambda) :
			lambdaW(lambda), lambdaH(lambda), lambdaRow(lambda), lambdaCol(lambda) { };

	BiasedNzl2Loss(double lambdaFactors, double lambdaBias) :
			lambdaW(lambdaFactors), lambdaH(lambdaFactors), lambdaRow(lambdaBias), lambdaCol(lambdaBias) { };

	BiasedNzl2Loss(double lambdaW, double lambdaH, double lambdaRow, double lambdaCol) :
		lambdaW(lambdaW), lambdaH(lambdaH), lambdaRow(lambdaRow), lambdaCol(lambdaCol) { };

	double operator()(const FactorizationData<>& data) {
		if (data.tasks > 1) {
			LOG4CXX_WARN(detail::logger, "Parallel computation of BiasedNzl2Loss not yet implemented, using sequential computation.");
		}
		double result = 0.;
		if (lambdaW != 0) result += lambdaW * biasedNzl2Factors(data.w, *data.nnz1, data.nnz1offset);
		if (lambdaH != 0) result += lambdaH * biasedNzl2Factors(data.h, *data.nnz2, data.nnz2offset);
		if (lambdaRow != 0) result += lambdaRow * biasedNzl2Bias(data.w, *data.nnz1, data.nnz1offset);
		if (lambdaCol != 0) result += lambdaCol * biasedNzl2Bias(data.h, *data.nnz2, data.nnz2offset);
		return result;
	}

	double operator()(const DsgdFactorizationData<>& data) {
		double result = 0.;
		if (lambdaW != 0) result += lambdaW * biasedNzl2Factors(data.dw, data.nnz1name, data.tasksPerRank);
		if (lambdaH != 0) result += lambdaH * biasedNzl2Factors(data.dh, data.nnz2name, data.tasksPerRank);
		if (lambdaRow != 0) result += lambdaRow * biasedNzl2Bias(data.dw, data.nnz1name, data.tasksPerRank);
		if (lambdaCol != 0) result += lambdaCol * biasedNzl2Bias(data.dh, data.nnz2name, data.tasksPerRank);
		return result;
	}

private:
	friend class ::boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version) {
		ar & lambdaW;
		ar & lambdaH;
		ar & lambdaRow;
		ar & lambdaCol;
	}

	double lambdaW;
	double lambdaH;
	double lambdaRow;
	double lambdaCol;
};

}

MPI2_TYPE_TRAITS(mf::BiasedNzl2Loss);
MPI2_SERIALIZATION_CONSTRUCTOR(mf::BiasedNzl2Loss);


#endif
