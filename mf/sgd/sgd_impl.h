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
 * Implementation for sgd/sgd.h
 * DO NOT INCLUDE DIRECTLY
 */

#include <algorithm>

#include <mf/sgd/sgd.h> // help for compilers

#include <mf/sgd/decay/decay_constant.h>
#include <mf/loss/loss.h>


// if the following line is commented out, we will not use prefetching (slower)
#define USE_PREFETCHING

namespace mf {

namespace detail {
template<typename Job, typename Loss, typename AdaptiveDecay, typename Epoch,typename TestData,typename TestLoss>
void defaultRunner(Job& job, Loss& loss, mf_size_type epochs, AdaptiveDecay& decay,
		Epoch runEpoch, Trace& trace, rg::Random32& random,
		BalanceType balanceType, BalanceMethod balanceMethod,
		TestData* testData, TestLoss *testLoss) {
	// print information about training point order
	switch (job.order) {
	case SGD_ORDER_SEQ:
		LOG4CXX_INFO(detail::logger, "Using SEQ order for selecting training points");
		break;
	case SGD_ORDER_WR:
		LOG4CXX_INFO(detail::logger, "Using WR order for selecting training points");
		break;
	case SGD_ORDER_WOR:
		LOG4CXX_INFO(detail::logger, "Using WOR order for selecting training points");
		break;
	}

	// initialize
	double previousLoss = 0;
	double timeLoss=0.0;
	rg::Timer t;
	t.start();
	mpi2::logBeginEvent("loss");
	double currentLoss = loss(job);
	mpi2::logEndEvent("loss");
	t.stop();
	timeLoss=t.elapsedTime().nanos();
	trace.clear();
	LOG4CXX_INFO(detail::logger, "Loss: " << currentLoss << " (" << t << ")");

	SgdTraceEntry* entry;
	double currentTestLoss=0.0;
	double timeTestLoss=0.0;
	if (testData!=NULL){
		t.start();
		mpi2::logBeginEvent("testloss");
		currentTestLoss=(*testLoss)(*testData);
		mpi2::logEndEvent("testloss");
		t.stop();
		timeTestLoss=t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
		// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
		entry=new SgdTraceEntry(currentLoss, timeLoss, currentTestLoss,timeTestLoss);
	}
	else{
		// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
		entry=new SgdTraceEntry(currentLoss, timeLoss);
	}
	trace.add(entry);

	// main loop
	for (mf_size_type epoch=0; epoch<epochs; epoch++) {
		// update step size
		mpi2::logBeginEvent("stepsize");
		t.start();
		double eps;
		if (epoch == 0) {
			eps = decay(job, NULL, &currentLoss, random);
		} else {
			eps = decay(job, &previousLoss, &currentLoss, random);
		}
		t.stop();
		double timeEps = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Step size: " << eps << " (" << t << ")");
		mpi2::logEndEvent("stepsize");

		// run epoch
		mpi2::logBeginEvent("epoch");
		LOG4CXX_INFO(detail::logger, "Starting epoch " << (epoch+1));
		t.start();
		runEpoch(job, eps);
		t.stop();
		double timeEpoch = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Finished epoch " << (epoch+1) << " (" << t << ")");
		mpi2::logEndEvent("epoch");

		// balance
		mpi2::logBeginEvent("balance");
		balance(job, balanceType, balanceMethod);
		mpi2::logEndEvent("balance");

		// compute loss
		previousLoss = currentLoss;
		t.start();
		mpi2::logBeginEvent("loss");
		currentLoss = loss(job);
		mpi2::logEndEvent("loss");
		t.stop();
		double timeLoss = t.elapsedTime().nanos();
		LOG4CXX_INFO(detail::logger, "Loss: " << currentLoss << " (" << t << ")");

		SgdTraceEntry* entry;
		currentTestLoss=0.0;
		timeTestLoss=0.0;
		if (testData!=0){
			t.start();
			mpi2::logBeginEvent("testloss");
			currentTestLoss=(*testLoss)(*testData);
			mpi2::logEndEvent("testloss");
			t.stop();
			timeTestLoss=t.elapsedTime().nanos();
			LOG4CXX_INFO(detail::logger, "Test loss: " << currentTestLoss << " (" << t << ")");
			// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
			entry=new SgdTraceEntry(epoch+1, epoch+1, currentLoss, eps, timeEps, timeEpoch, timeLoss, currentTestLoss, timeTestLoss);
		}
		else{
			// preserve the memory. Otherwise the trace will lose its information. memory release with the program's exit
			entry=new SgdTraceEntry(epoch+1, epoch+1, currentLoss, eps, timeEps, timeEpoch, timeLoss);
		}
		trace.add(entry);
	}
}
}

template<typename Update, typename Regularize, typename Loss, typename AdaptiveDecay,typename TestData,typename TestLoss>
void SgdRunner::run(SgdJob<Update, Regularize>& job, Loss& loss, mf_size_type epochs, AdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod, TestData* testData, TestLoss *testLoss) {
	LOG4CXX_INFO(detail::logger, "Starting SGD");
	detail::defaultRunner(job, loss, epochs, decay,
			boost::bind(&SgdRunner::epoch<Update,Regularize>, this, _1, _2),
			trace, random_, balanceType, balanceMethod, testData, testLoss);
	LOG4CXX_INFO(detail::logger, "Finished SGD");
}

template<typename Update, typename Regularize, typename Loss, typename AdaptiveDecay>
void SgdRunner::run(SgdJob<Update, Regularize>& job, Loss& loss, mf_size_type epochs, AdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod) {
	run(job, loss, epochs, decay, trace, balanceType, balanceMethod, (FactorizationData<>*)NULL, (NoLoss*)NULL);
}

template<typename Update, typename Regularize, typename Decay>
void SgdRunner::update(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay) {
	switch (job.order) {
	case SGD_ORDER_SEQ:
		updateSequential(job, steps, decay);
		break;
	case SGD_ORDER_WR:
		updateWr(job, steps, decay);
		break;
	case SGD_ORDER_WOR:
		updateWor(job, steps, decay);
		break;
	default:
		RG_THROW(rg::InvalidArgumentException, "Unknown SGD order");
		break;
	}
}
template<typename Update, typename Regularize>
void SgdRunner::update(SgdJob<Update, Regularize>& job, mf_size_type steps, double eps) {
	DecayConstant decay(eps);
	update(job, steps, decay);
}

template<typename Update, typename Regularize>
void SgdRunner::epoch(SgdJob<Update, Regularize>& job, double eps) {
	epoch(job, eps, eps);
}
template<typename Update, typename Regularize>
void SgdRunner::epoch(SgdJob<Update, Regularize>& job, double epsUpdate, double epsRegularize) {
	update(job, job.nnz, epsUpdate);
	regularize(job, epsRegularize);
}

template<typename Update, typename Regularize, typename Decay>
void SgdRunner::updateSequential(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay) {
	mf_size_type remainingSteps = steps;
	while (remainingSteps > 0) {
		mf_size_type end = std::min(remainingSteps, job.nnz);
		updateSequential(job, decay, 0, end, steps - remainingSteps);
		remainingSteps -= end;
	}
}

template<typename Update, typename Regularize, typename Decay>
void SgdRunner::updateSequential(SgdJob<Update, Regularize>& job, Decay& decay,
		mf_size_type begin, mf_size_type end, mf_size_type decayOffset) {
	mf_size_type n = end - begin;
	for (mf_size_type step=0; step<n; step++) {
		// no prefetching needed (sequential read)
		const mf_size_type pos = begin + step;
		const double eps = decay(decayOffset + step);
		job.update(job, job.vIndex1[pos], job.vIndex2[pos], job.vValues[pos], eps);
	}
}

template<typename Update, typename Regularize, typename Decay>
void SgdRunner::updateWr(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay) {
	updateWr(job, steps, decay, random_, 0, job.nnz, 0);
}

template<typename Update, typename Regularize, typename Decay>
void SgdRunner::updateWr(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay,
		rg::Random32& random, mf_size_type begin, mf_size_type end, mf_size_type decayOffset) {
	mf_size_type n = end - begin;

	mf_size_type currentPos = -1;
	mf_size_type nextPos = begin + random.nextInt(n);
	mf_size_type next2Pos = begin + random.nextInt(n);
	for (mf_size_type step=0; step<steps; step++) {
		// compute positions
		currentPos = nextPos;
		nextPos = next2Pos;
#ifdef USE_PREFETCHING // with prefetching
		// prefetch rows/columns needed in next step
		__builtin_prefetch(&job.wValues[job.vIndex1[nextPos]], 1, 2); // write and locality=2
		__builtin_prefetch(&job.hValues[job.vIndex2[nextPos]], 1, 2); // write and locality=2
#endif
		// prefetch training point needed in 2 steps
		next2Pos = begin + random.nextInt(n);
#ifdef USE_PREFETCHING // with prefetching
		__builtin_prefetch(&job.vIndex1[next2Pos], 0, 1);
		__builtin_prefetch(&job.vIndex2[next2Pos], 0, 1);
		__builtin_prefetch(&job.vValues[next2Pos], 0, 1);
#endif
		// execute the current step
		const double eps = decay(decayOffset + step);
		job.update(job, job.vIndex1[currentPos], job.vIndex2[currentPos], job.vValues[currentPos], eps);
	}
}

/** n = total number of training points, k=number of points to be shuffled */
inline void SgdRunner::permute(rg::Random32& random, std::vector<mf_size_type>& permutation, int n, int k) {
	// initialize permutation array (stores order of training points)
	if (permutation.size() != n) {
		permutation.resize(n);
		for (mf_size_type i=0; i<n; i++) {
			permutation[i] = i;
		}
	}


	// compute permutation (this is the standard Knuth shuffle with prefetching)
	mf_size_type currentIndex = 0;
	mf_size_type nextIndex = random.nextInt(n);
	for (mf_size_type nextStep=1; nextStep<k; nextStep++) {
		currentIndex = nextIndex;
		nextIndex = nextStep + random.nextInt(n-nextStep);
#ifdef USE_PREFETCHING // with prefetching
		__builtin_prefetch(&permutation[nextIndex], 1, 0); // write
#endif
		std::swap(permutation[nextStep-1], permutation[currentIndex]);
	}
	// last element does not need to be swapped

}


template<typename Update, typename Regularize, typename Decay>
void SgdRunner::updateWor(SgdJob<Update, Regularize>& job, mf_size_type steps, Decay& decay) {
	mf_size_type remainingSteps = steps;
	while (remainingSteps > 0) {
		mf_size_type end = std::min(remainingSteps, job.nnz);
		permute(random_, permutation_, job.nnz, end);
		updateWor(job, decay, random_, 0, end, steps - remainingSteps, permutation_);
		remainingSteps -= end;
	}
}

template<typename Update, typename Regularize, typename Decay>
void SgdRunner::updateWor(SgdJob<Update, Regularize>& job, Decay& decay,
		rg::Random32& random, mf_size_type begin, mf_size_type end, mf_size_type decayOffset,
		const std::vector<mf_size_type>& permutation) {
	// handle border cases
	mf_size_type n = end - begin;

	if (n == 0) {
		return;
	} else if (n == 1) {
		mf_size_type currentPos =  permutation[begin];
		const double eps = decay(0);
		job.update(job, job.vIndex1[currentPos], job.vIndex2[currentPos], job.vValues[currentPos], eps);
		return;
	}

	// run SGD
#ifdef USE_PREFETCHING // with prefetching
	mf_size_type currentPos = -1;
	mf_size_type nextPos =  permutation[begin];
	mf_size_type next2Pos =  permutation[begin + 1];

	for (mf_size_type step=0; step<n-2; step++) {
			// compute positions
			currentPos = nextPos;
			nextPos = next2Pos;

			// prefetch rows/column factors needed in next step
			__builtin_prefetch(&job.wValues[job.vIndex1[nextPos]], 1, 2); // write and locality=2
			__builtin_prefetch(&job.hValues[job.vIndex2[nextPos]], 1, 2); // write and locality=2

			// prefetch training point needed in 2 steps
			next2Pos = permutation[begin + step + 2];
			__builtin_prefetch(&job.vIndex1[next2Pos], 0, 1);
			__builtin_prefetch(&job.vIndex2[next2Pos], 0, 1);
			__builtin_prefetch(&job.vValues[next2Pos], 0, 1);

		// execute the current step
		const double eps = decay(decayOffset + step);
		job.update(job, job.vIndex1[currentPos], job.vIndex2[currentPos], job.vValues[currentPos], eps);
	}

	currentPos = permutation[begin + n-2];
	double eps = decay(decayOffset + n-2);
	job.update(job, job.vIndex1[currentPos], job.vIndex2[currentPos], job.vValues[currentPos], eps);

	currentPos = permutation[begin + n-1];
	eps = decay(decayOffset + n-1);
	job.update(job, job.vIndex1[currentPos], job.vIndex2[currentPos], job.vValues[currentPos], eps);
#else // without prefetching
	mf_size_type currentPos;
	for (mf_size_type step=0; step<n; step++) {
		currentPos=permutation[begin+step];
		// execute the current step
		const double eps = decay(decayOffset + step);
		job.update(job, job.vIndex1[currentPos], job.vIndex2[currentPos], job.vValues[currentPos], eps);
	}

#endif
}

template<typename Update, typename Regularize>
void SgdRunner::regularize(SgdJob<Update, Regularize>& job, double eps) {
	job.regularize(job, eps);
}


}
