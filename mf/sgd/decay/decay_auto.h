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
#ifndef MF_SGD_DECAY_DECAY_AUTO_H
#define MF_SGD_DECAY_DECAY_AUTO_H

#include <algorithm>

#include <boost/foreach.hpp>

#include <mf/matrix/op/project.h>
#include <mf/sgd/decay/decay.h>
#include <mf/sgd/sgd.h>

#include <util/exception.h>
#include <util/io.h>
#include <util/random.h>

namespace mf {

namespace detail {

/** A decay function that tries to automatically select the optimum step size based on
 * a sample. Initially, the step size that works best on the sample is selected. This
 * choice works well initially, but will lead to overly large choices once the solution
 * is close to the optimum (which maybe away from the sample-local optimum). As soon
 * as this is detected, the BoldDriver is used for the remaining steps.
 *
 * @tparam Update type of update function (model of UpdateConcept)
 * @tparam Regularize type of regularization function (model of RegularizeConcept)
 * @tparam Loss type of loss function (model of LossConcept)
 */
template<typename Update, typename Regularize, typename Loss>
class AbstractDecayAuto {
public:
	AbstractDecayAuto(Sgd<Update,Regularize> sgd, Loss loss,
			double eps, unsigned tries, double decrease = 0.5, double increase=1.05, bool
			allowIncrease = false)
	: sgd(sgd), loss(loss),
	  eps_(eps), initialEps_(eps_), tries(tries), decrease_(decrease), increase_(increase),
	  fallback(false), allowIncrease(allowIncrease) {
		if (tries <= 1) {
			RG_THROW(rg::InvalidArgumentException, "tries has to be larger than 1");
		}
	};

protected:
	template<typename FD, typename F>
	inline double nextEps(FD& data, F findBestEps, double* previousLoss, double* currentLoss,
			rg::Random32& random) {
		if (previousLoss != NULL && *previousLoss < *currentLoss && !fallback) {
			LOG4CXX_INFO(detail::logger, "Falling back to bold driver decay");
			fallback = true;
		}
		if (!fallback) {
retry:
			// construct a set of step sizes to try
			double maxEps = 2*eps_;
			if (!allowIncrease && maxEps>initialEps_) maxEps = initialEps_;
			double delta = (maxEps - (eps_/2)) / (tries-1);
			//std::cout<<" delta: "<<delta<<std::endl;
			eps_ = maxEps;
			std::vector<double> epsToTry;
			for (unsigned i=0; i<tries; i++) {
				epsToTry.push_back(eps_);
				//std::cout<<" eps: "<<eps_<<std::endl;
				if (previousLoss == NULL) {
					//std::cout<<" Null previous loss "<<std::endl;
					eps_/=2;
				} else if (i != tries-1) {
					eps_ -= delta;
				}
			}

			//std::cout<<"After loop eps: "<<eps_<<std::endl;

			// find the best loss on these step sizes
			std::vector<double> losses = findBestEps(data, random, epsToTry);
			int bestIndex = -1;
			double bestLoss = INFINITY;
			for(int i=0; i<tries; i++) {
				LOG4CXX_DEBUG(detail::logger, "Tried eps=" << epsToTry[i] << ", loss=" << losses[i]);
				//std::cout<<"Tried eps=" << epsToTry[i] << ", loss=" << losses[i]<<std::endl;

				if (!isnan(losses[i]) && losses[i] < bestLoss) {
					bestLoss = losses[i];
					bestIndex = i;
				}
			}

			// decide whether to accept the step size that gave the best loss
			bool accept = false;
			if (bestIndex == 0) {
				// accept when it is the biggest one tried
				accept = true;
			} else if (!isnan(losses[bestIndex-1]) && losses[bestIndex-1]<losses[bestIndex]*100) {
				// accept when the next-largest step size tried was not significantly worse (factor 100)
				accept = true;
			} else {
				// the next-largest step size performed much worse; pick the next-smaller one to be safe
				bestIndex++;
				if (bestIndex < tries) accept = true; // otherwise retry
			}

			if (accept) {
				eps_ = epsToTry[bestIndex];
				//std::cout<<"after acceptance eps: "<<eps_<<std::endl;
			} else {
				LOG4CXX_WARN(detail::logger, "Could not find an initial step size after "
						<< tries << " tries. Trying again.");
				eps_ /= 2;
				//std::cout<<" Go to retry with eps: "<<eps_<<std::endl;
				goto retry;
			}
		} else {
			if (*previousLoss <= *currentLoss) {
				eps_ *= decrease_;
			} else {
				eps_ *= increase_;
			}
		}
		return eps_;
	}

protected:
	Sgd<Update,Regularize> sgd;
	Loss loss;
	double eps_;
	double initialEps_;
	unsigned tries;
	const double decrease_;
	const double increase_;
	bool fallback;
	bool allowIncrease;
	double scaleFactor;
};

} // namespace detail

/** @copydoc detail::AbstractDecayAuto */
template<typename Update, typename Regularize, typename Loss>
class DecayAuto : public detail::AbstractDecayAuto<Update,Regularize,Loss>,
				  public AdaptiveDecayConcept {
	using detail::AbstractDecayAuto<Update,Regularize,Loss>::sgd;

public:
	DecayAuto(Sgd<Update,Regularize> sgd, Loss loss, const ProjectedSparseMatrix& sample,
			double eps, unsigned tries, double decrease = 0.5, double increase=1.05, bool
			allowIncrease = false, bool scale=false)
	: detail::AbstractDecayAuto<Update,Regularize,Loss>(sgd, loss, eps, tries, decrease, increase, allowIncrease),
	  sample(sample)
	{
		typedef detail::AbstractDecayAuto<Update,Regularize,Loss> ADA;
		if (scale) {
			ADA::scaleFactor = std::min(
					(double)sample.data.size1() / sample.size1,
					(double)sample.data.size2() / sample.size2
					);
		} else {
			ADA::scaleFactor = 1;
		}
		nnz12(sample.data, nnz1, nnz2, nnz12max);
		LOG4CXX_INFO(detail::logger, "Initialized automatic decay with scale factor of " << ADA::scaleFactor);
	};

	inline std::vector<double> findBestEps(FactorizationData<>& data, rg::Random32& random,
			const std::vector<double> epsToTry) {
		std::vector<double> losses;
		SgdRunner sgdRunner(random);
		project1(data.w, wSample, sample.map1);
		project2(data.h, hSample, sample.map2);

		for (unsigned index=0; index<epsToTry.size(); index++) {
			double eps = epsToTry[index];
			wSampleCopy = wSample;
			hSampleCopy = hSample;
			FactorizationData<> jobData(sample.data, wSampleCopy, hSampleCopy, nnz1, 0, nnz2, 0, nnz12max);
			SgdJob<Update,Regularize> job(jobData, sgd.update, sgd.regularize, sgd.order);
			sgdRunner.epoch(job, eps);
			double loss = this->loss(job);
			losses.push_back(loss);
		};

		return losses;
	}

	inline double operator()(FactorizationData<>& data, double* previousLoss, double* currentLoss,
			rg::Random32& random) {
		return this->nextEps(data,
				boost::bind(&DecayAuto<Update,Regularize,Loss>::findBestEps, this, _1, _2, _3),
				previousLoss, currentLoss, random) * detail::AbstractDecayAuto<Update,Regularize,Loss>::scaleFactor;
	}

protected:
	const ProjectedSparseMatrix& sample;
	std::vector<mf_size_type> nnz1;
	std::vector<mf_size_type> nnz2;
	//mf_size_type nnz12max;
	mf_size_type nnz12max;
	DenseMatrix wSample;
	DenseMatrix wSampleCopy;
	DenseMatrixCM hSample;
	DenseMatrixCM hSampleCopy;
};

/** @copydoc detail::AbstractDecayAuto */
template<typename Update, typename Regularize, typename Loss>
class ParallelDecayAuto : public DecayAuto<Update, Regularize, Loss> {
	using DecayAuto<Update,Regularize,Loss>::sample;
	using DecayAuto<Update,Regularize,Loss>::wSample;
	using DecayAuto<Update,Regularize,Loss>::hSample;

public:
	ParallelDecayAuto(Sgd<Update,Regularize> sgd, Loss loss, const ProjectedSparseMatrix& sample,
			double eps, unsigned tries, double decrease = 0.5, double increase=1.05, bool
			allowIncrease = false, bool scale=false)
	: DecayAuto<Update,Regularize,Loss>(sgd, loss, sample, eps, tries, decrease, increase, allowIncrease, scale),
	  mySgd(sgd) {
	}

	inline std::vector<double> findBestEps(FactorizationData<>& data, rg::Random32& random,
			const std::vector<double> epsToTry) {
		project1(data.w, wSample, sample.map1);
		project2(data.h, hSample, sample.map2);

		// run tasks that compute the losses
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		boost::mpi::communicator& world = tm.world();
		std::vector<mpi2::Channel> channels;
		tm.spawn<ParallelDecayAutoTask>(world.rank(), epsToTry.size(), channels);
		mpi2::seed(channels, random);
		mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&data), mpi2::pointerToInt(this)));
		mpi2::sendEach(channels, epsToTry);

		// receive results
		std::vector<double> losses;
		mpi2::economicRecvAll(channels, losses, tm.pollDelay());

		// return result
		return losses;
	}

	inline double operator()(FactorizationData<>& data, double* previousLoss, double* currentLoss,
			rg::Random32& random) {
		return this->nextEps(data,
				boost::bind(&ParallelDecayAuto<Update,Regularize,Loss>::findBestEps, this, _1, _2, _3),
				previousLoss, currentLoss, random) * detail::AbstractDecayAuto<Update,Regularize,Loss>::scaleFactor;
	}

private:
	Sgd<Update,Regularize>& mySgd;

	struct ParallelDecayAutoTask {
		static const std::string id() {
			return std::string("__mf/sgd/decay_ParallelDecayAutoTask")
				+ "_" +  mpi2::TypeTraits<Update>::name()
				+ "_" +  mpi2::TypeTraits<Regularize>::name()
				+ "_" +  mpi2::TypeTraits<Loss>::name();
		}

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			rg::Random32 random = mpi2::getSeed(ch);
			mpi2::PointerIntType pData;
			mpi2::PointerIntType pDecay;
			ch.recv(*mpi2::unmarshal(pData, pDecay));
			FactorizationData<>& data = *mpi2::intToPointer<FactorizationData<> >(pData);
			ParallelDecayAuto<Update, Regularize, Loss>& decay = *mpi2::intToPointer<ParallelDecayAuto<Update, Regularize, Loss> >(pDecay);
			double eps;
			ch.recv(eps);

			DenseMatrix wSampleCopy = decay.wSample;
			DenseMatrixCM hSampleCopy = decay.hSample;
			FactorizationData<> jobData(decay.sample.data, wSampleCopy, hSampleCopy, decay.nnz1, 0, decay.nnz2, 0, decay.nnz12max);
			SgdJob<Update,Regularize> job(jobData, decay.mySgd.update, decay.mySgd.regularize, decay.mySgd.order);
			SgdRunner sgdRunner(random);
			sgdRunner.epoch(job, eps);
			double loss = decay.loss(job);
			ch.send(loss);
		}
	};
};


namespace detail {
	template<typename Update, typename Regularize, typename Loss>
	struct DistributedDecayAutoTask {
		static const std::string id() {
			return std::string("__mf/sgd/decay_DistributedDecayAutoTask")
				+ "_" +  mpi2::TypeTraits<Update>::name()
				+ "_" +  mpi2::TypeTraits<Regularize>::name()
				+ "_" +  mpi2::TypeTraits<Loss>::name();
		}

		static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
			rg::Random32 random = mpi2::getSeed(ch);
			Sgd<Update,Regularize> sgd(mpi2::UNINITIALIZED);
			Loss loss(mpi2::UNINITIALIZED);
			std::string varNameBase;
			mf_size_type nnz12max;
			double eps;
			ch.recv(*mpi2::unmarshal(sgd, loss, varNameBase));
			ch.recv(nnz12max);
			ch.recv(eps);

			ProjectedSparseMatrix* sample =
					mpi2::env().get<ProjectedSparseMatrix>(varNameBase + "_sample");
			DenseMatrix wSampleCopy =
					*mpi2::env().get<DenseMatrix>(varNameBase + "_wSample");
			DenseMatrixCM hSampleCopy =
					*mpi2::env().get<DenseMatrixCM>(varNameBase + "_hSample");
			std::vector<mf_size_type>* nnz1 =
					mpi2::env().get<std::vector<mf_size_type> >(varNameBase + "_sample_nnz1");
			std::vector<mf_size_type>* nnz2 =
					mpi2::env().get<std::vector<mf_size_type> >(varNameBase + "_sample_nnz2");


			FactorizationData<> jobData(sample->data, wSampleCopy, hSampleCopy, *nnz1, 0, *nnz2, 0, nnz12max);
			SgdJob<Update,Regularize> job(jobData, sgd.update, sgd.regularize, sgd.order);
			SgdRunner sgdRunner(random);
			sgdRunner.epoch(job, eps);
			double result = loss(job);
			ch.send(result);
		}
	};
}

/** @copydoc detail::AbstractDecayAuto */
template<typename Update, typename Regularize, typename Loss>
class DistributedDecayAuto : public detail::AbstractDecayAuto<Update,Regularize,Loss>,
				  public DistributedAdaptiveDecayConcept {
	using detail::AbstractDecayAuto<Update,Regularize,Loss>::sgd;
	using detail::AbstractDecayAuto<Update,Regularize,Loss>::loss;

public:
	/** tries currently MUST BE a multiple of the number of ranks */
	DistributedDecayAuto(Sgd<Update,Regularize> sgd, Loss loss,
			const ProjectedSparseMatrix& sample, const std::string& varNameBase,
			double eps, unsigned tries, double decrease = 0.5, double increase=1.05, bool
			allowIncrease = false, bool scale=false)
	: detail::AbstractDecayAuto<Update,Regularize,Loss>(sgd, loss, eps, tries, decrease, increase, allowIncrease),
	  varNameBase(varNameBase)
	{
		// store the sample on all nodes
		// TODO: add support for replicated matrices!
		mpi2::createCopyAll(varNameBase + "_sample", sample);
		mpi2::createCopyAll(varNameBase + "_wSample", DenseMatrix(0,0));
		mpi2::createCopyAll(varNameBase + "_hSample", DenseMatrixCM(0,0));

		std::vector<mf_size_type> nnz1, nnz2;
		mf_size_type nnz12max;
		nnz12(sample.data, nnz1, nnz2,nnz12max);
		mpi2::createCopyAll(varNameBase + "_sample_nnz1", nnz1);
		mpi2::createCopyAll(varNameBase + "_sample_nnz2", nnz2);

		typedef detail::AbstractDecayAuto<Update,Regularize,Loss> ADA;
		if (scale) {
			ADA::scaleFactor = std::min(
					(double)sample.data.size1() / sample.size1,
					(double)sample.data.size2() / sample.size2
					);
		} else {
			ADA::scaleFactor = 1;
		}
		LOG4CXX_INFO(detail::logger, "Initialized automatic decay with scale factor of " << ADA::scaleFactor);
	};

	inline std::vector<double> findBestEps(DsgdFactorizationData<>& data, rg::Random32& random,
			const std::vector<double>& epsToTry) {
		// get sample rows/columns of current factors
		ProjectedSparseMatrix* sample =
				mpi2::env().get<ProjectedSparseMatrix>(varNameBase + "_sample");
		project1(data.dw, wSample, sample->map1, data.tasksPerRank);
		project2(data.dh, hSample, sample->map2, data.tasksPerRank);

		// store them on all ranks
		mpi2::setCopyAll(varNameBase + "_wSample", wSample);
		mpi2::setCopyAll(varNameBase + "_hSample", hSample);

		// run a tasks on all ranks that compute the losses
		// TODO: add support for automatic selection of the number of tasks
		mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();
		boost::mpi::communicator& world = tm.world();
		std::vector<mpi2::Channel> channels;
		tm.spawnAll<detail::DistributedDecayAutoTask<Update,Regularize,Loss> >(
				epsToTry.size()/world.size(), channels);
		mpi2::seed(channels, random);
		mpi2::sendAll(channels, mpi2::marshal(sgd, loss, varNameBase));
		mpi2::sendAll(channels, data.nnz12max);
		mpi2::sendEach(channels, epsToTry);

		// receive results
		std::vector<double> losses;
		mpi2::recvAll(channels, losses);

		// return result
		return losses;
	}

	inline double operator()(DsgdFactorizationData<>& data,
			double* previousLoss, double* currentLoss, rg::Random32& random) {
		return this->nextEps(data,
				boost::bind(&DistributedDecayAuto<Update,Regularize,Loss>::findBestEps, this, _1, _2, _3),
				previousLoss, currentLoss, random) * detail::AbstractDecayAuto<Update,Regularize,Loss>::scaleFactor;
	}


private:
	const std::string varNameBase;
	DenseMatrix wSample;
	DenseMatrixCM hSample;
};

}

#endif
