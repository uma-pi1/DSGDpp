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
 * Implementation for sgd/dsgd.h
 * DO NOT INCLUDE DIRECTLY
 */

#include <set>

#include <algorithm>

#include "mystratifiedpsgd.h"

#include <mf/matrix/op/shuffle.h>
#include <stdint.h>

namespace mf {
  
  uint64_t get_cpu_cycle_counter(){
    unsigned int lo,hi;
    __asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

template<typename Update, typename Regularize, typename Loss,
	typename AdaptiveDecay,typename TestData,typename TestLoss>
void StratifiedPsgdRunner::run(StratifiedPsgdJob<Update, Regularize>& job, Loss& loss,
		mf_size_type epochs, AdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod,
		TestData* testData, TestLoss *testLoss) {
	LOG4CXX_INFO(detail::logger, "Starting Stratified PSGD (polling delay: " << mpi2::TaskManager::getInstance().pollDelay() << " microseconds)");

	detail::defaultRunner(job, loss, epochs, decay,
			boost::bind(&StratifiedPsgdRunner::epoch<Update,Regularize>, this, _1, _2),
			trace, random_, balanceType, balanceMethod, testData, testLoss);

	LOG4CXX_INFO(detail::logger, "Finished Stratified PSGD");
}


template<typename Update, typename Regularize, typename Loss,
	typename AdaptiveDecay>
void StratifiedPsgdRunner::run(StratifiedPsgdJob<Update, Regularize>& job, Loss& loss,
		mf_size_type epochs, AdaptiveDecay& decay,
		Trace& trace, BalanceType balanceType, BalanceMethod balanceMethod) {
	run(job, loss, epochs, decay, trace, balanceType, balanceMethod, (FactorizationData<>*)NULL, (NoLoss*)NULL);
}

namespace detail {

/** n = total number of training points, k=number of points to be shuffled */
inline void permute(rg::Random32& random, std::vector<mf_size_type>& permutation, mf_size_type n, mf_size_type k) {
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

}/**/


inline void myPermute(rg::Random32& random, std::vector<mf_size_type>& permutation, mf_size_type begin, mf_size_type end) {

	// compute permutation (this is the standard Knuth shuffle with prefetching)
	mf_size_type currentIndex = begin;
	mf_size_type range = end - begin;
	mf_size_type nextIndex = begin + random.nextInt(range);
		
	for (mf_size_type nextStep = 1; nextStep < range; nextStep++) {
		currentIndex = nextIndex;
		nextIndex = nextStep + begin + random.nextInt(range - nextStep);
#ifdef USE_PREFETCHING // with prefetching
		__builtin_prefetch(&permutation[begin + nextIndex], 1, 0); // write
#endif
		std::swap(permutation[begin + nextStep - 1], permutation[currentIndex]);
	}
	// last element does not need to be swapped

}


template<typename Update, typename Regularize>
struct StratifiedPsgdUpdateWorTask {
	static const std::string id() { return std::string("__mf/sgd/StratifiedPsgdUpdateWorTask_")
			+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		rg::Random32 random = mpi2::getSeed(ch);

		// receive task tags and figure out my id
		std::vector<mpi2::Channel>& channels = info.pairwiseChannels();
		mf_size_type id = info.groupId();
		mf_size_type tasks = info.groupSize();

		// receive data descriptor
		double eps;
		mpi2::PointerIntType pJob, pPermutation, pOffsets;
		
		

		

		ch.recv(*mpi2::unmarshal(pJob, pPermutation));
		ch.recv(*mpi2::unmarshal(pOffsets, eps));
		

		StratifiedPsgdJob<Update, Regularize>& job = *mpi2::intToPointer<StratifiedPsgdJob<Update, Regularize> >(pJob);
		std::vector<mf_size_type>& permutation = *mpi2::intToPointer<std::vector<mf_size_type> >(pPermutation);
		std::vector<mf_size_type>& offsets = *mpi2::intToPointer<std::vector<mf_size_type> >(pOffsets);
// 		std::cout<<"got data. offsets: "<<offsets.size() <<std::endl;

		mf_size_type d = sqrt(offsets.size());
		mf_size_type blocksPerTask = offsets.size() / tasks;

		rg::Timer t;
		double timeSchedule = 0.;
		double timeSgd = 0.;
		double timePermute = 0.;


		// create schedule
		std::vector<mf_size_type> schedule(blocksPerTask);
		for (mf_size_type i = 0; i < blocksPerTask; i++) {
			schedule[i] = i;
		}
		SgdRunner::permute(random, schedule, blocksPerTask, blocksPerTask);// wor stratum schedule
// 		std::cout<<"permute schedule ok. blocksPerTask: "<<blocksPerTask<<std::endl;

		// permute training points inside each block
		mf_size_type begin = offsets[id*blocksPerTask];
		mf_size_type end = offsets[(id+1)*blocksPerTask] -1;

// 		std::cout<<"begin end ok "<<std::endl;

		DecayConstant decay(eps);
		for (mf_size_type s = 0; s < blocksPerTask; s++) {
			mf_size_type blockBegin = offsets[(id*blocksPerTask) + schedule[s]];
			mf_size_type blockEnd = offsets[(id*blocksPerTask) + schedule[s] + 1] -1;
			
// 			std::cout<<"s: "<<s<<" schedule[s]: "<<schedule[s]<<" id*blocksPerTask: "<<id*blocksPerTask <<" blockBegin: "<<blockBegin<<" blockEnd: "<<blockEnd<<std::endl;
			if(blockBegin <  blockEnd){
			  	myPermute(random, permutation, blockBegin, blockEnd);// WOR for training point selection
				SgdRunner::updateWor(job, decay, random, blockBegin, blockEnd, 0, permutation);			  
			}


// 			barrier(channels);

		}

		
		ch.send();
	}
};


template<typename Update, typename Regularize>
struct StratifiedPsgdUpdateSEQTask {
	static const std::string id() { return std::string("__mf/sgd/StratifiedPsgdUpdateSEQTask_")
			+ mpi2::TypeTraits<Update>::name() + "_" + mpi2::TypeTraits<Regularize>::name(); }

	static inline void run(mpi2::Channel ch, mpi2::TaskInfo info) {
		rg::Random32 random = mpi2::getSeed(ch);

		// receive task tags and figure out my id
		std::vector<mpi2::Channel>& channels = info.pairwiseChannels();
		mf_size_type id = info.groupId();
		mf_size_type tasks = info.groupSize();

		// receive data descriptor
		double eps;
		mpi2::PointerIntType pJob, pPermutation, pOffsets;

		ch.recv(*mpi2::unmarshal(pJob, pPermutation));
		ch.recv(*mpi2::unmarshal(pOffsets, eps));
		

		StratifiedPsgdJob<Update, Regularize>& job = *mpi2::intToPointer<StratifiedPsgdJob<Update, Regularize> >(pJob);
		std::vector<mf_size_type>& permutation = *mpi2::intToPointer<std::vector<mf_size_type> >(pPermutation);
		std::vector<mf_size_type>& offsets = *mpi2::intToPointer<std::vector<mf_size_type> >(pOffsets);

		mf_size_type d = sqrt(offsets.size());
		mf_size_type blocksPerTask = offsets.size() / tasks;

		rg::Timer t;
		double timeSchedule = 0.;
		double timeSgd = 0.;
		double timePermute = 0.;


		// create schedule
		std::vector<mf_size_type> schedule(blocksPerTask);
		for (mf_size_type i = 0; i < blocksPerTask; i++) {
			schedule[i] = i;
		}
// 		SgdRunner::permute(random, schedule, blocksPerTask, blocksPerTask);
		
		
		

		// permute training points inside each block
		mf_size_type begin = offsets[id*blocksPerTask];
		mf_size_type end = offsets[(id+1)*blocksPerTask] -1;


		DecayConstant decay(eps);
		for (mf_size_type s = 0; s < blocksPerTask; s++) {

			mf_size_type blockBegin = offsets[(id*blocksPerTask) + schedule[s]];
			mf_size_type blockEnd = offsets[(id*blocksPerTask) + schedule[s] + 1] -1;
			SgdRunner::updateWor(job, decay, random, blockBegin, blockEnd, 0, permutation);// it is still  ok with the updateWor. This version works on the specific permutation. If permutation is not really permuted, we have sequential
			
// 			barrier(channels);
		}

		ch.send();
	}
};

}


template<typename Update, typename Regularize>
void StratifiedPsgdRunner::epoch(StratifiedPsgdJob<Update, Regularize>& job, double eps) {

	mpi2::TaskManager& tm = mpi2::TaskManager::getInstance();

	unsigned tasks = job.tasks;
	rg::Timer t;

	std::vector<mf_size_type>& permutation = permutation_;
	std::vector<mf_size_type>& offsets = offsets_;
	

	// fire up the tasks
// 	std::vector<mpi2::Channel> channels;
	std::vector<mpi2::Channel> channels(tasks, mpi2::UNINITIALIZED);
// 	if(job.order == SGD_ORDER_SEQ){
// 	  tm.spawn<detail::StratifiedPsgdUpdateSEQTask<Update, Regularize> >(tm.world().rank(), tasks, channels, true);	  
// 	}else{
	  long long start_cycles, end_cycles, total_cycles;
	  start_cycles = get_cpu_cycle_counter();
	  tm.spawn<detail::StratifiedPsgdUpdateWorTask<Update, Regularize> >(tm.world().rank(), tasks, channels, true);	  
	  
	  
// 	}
	
	
	
	

	// send necessary data to each task
	mpi2::seed(channels, random_);
	mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&job), mpi2::pointerToInt(&permutation)));
	mpi2::sendAll(channels, mpi2::marshal(mpi2::pointerToInt(&offsets), eps));
	

	mpi2::economicRecvAll(channels, tm.pollDelay());
	end_cycles = get_cpu_cycle_counter();
	total_cycles = end_cycles-start_cycles;
// 	std::cout<<"Cycles for this epoch: "<<total_cycles<<std::endl;

}/**/



}


